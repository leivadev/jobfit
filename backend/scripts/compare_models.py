"""Offline comparison of bi-encoders and rerankers: quality (GPU), then RAM + latency (1-thread CPU).

Not part of the production pipeline. Stages share a work dir (`~/.cache/jobfit-exp`) so each is
resumable. Quality stages need a CUDA torch venv (the project venv is CPU-only on purpose);
the `cpu` stage is meant for the project venv, which matches production's torch build.

    PYTHONPATH=src:scripts python scripts/compare_models.py prep
    PYTHONPATH=src:scripts python scripts/compare_models.py embed minilm bge-small ...
    PYTHONPATH=src:scripts python scripts/compare_models.py eval
    PYTHONPATH=src:scripts python scripts/compare_models.py rerank minilm
    uv run --with einops --with psutil python scripts/compare_models.py cpu minilm ...

Weak label as in `evaluate_scale.py`: a Job is relevant to a CV if they share a Primary Keyword.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

WORK = Path.home() / ".cache" / "jobfit-exp"  # not /tmp: WSL clears it on reboot
META_PATH = WORK / "meta45.pkl"  # build_index.build_metadata() output, all 45 keywords
CANDIDATE_DATASET = "lang-uk/recruitment-dataset-candidate-profiles-english"
SEED = 42
BGE_QUERY = "Represent this sentence for searching relevant passages: "

# chunk = tokens per chunk before the model adds prefix/special tokens. Models with a longer
# native context are capped at 512 to fit the 4 GB GPU, so their long-context advantage is
# NOT exercised here.
BI_MODELS = {
    "minilm": dict(name="sentence-transformers/all-MiniLM-L6-v2", chunk=256, q="", d="", remote=False),
    "bge-small": dict(name="BAAI/bge-small-en-v1.5", chunk=480, q=BGE_QUERY, d="", remote=False),
    "bge-base": dict(name="BAAI/bge-base-en-v1.5", chunk=480, q=BGE_QUERY, d="", remote=False),
    "gte-base": dict(name="Alibaba-NLP/gte-base-en-v1.5", chunk=480, q="", d="", remote=True),
    "nomic": dict(name="nomic-ai/nomic-embed-text-v1.5", chunk=480, q="search_query: ", d="search_document: ", remote=True),
}
RERANKERS = {
    "ms-l6": "cross-encoder/ms-marco-MiniLM-L6-v2",
    "ms-l12": "cross-encoder/ms-marco-MiniLM-L12-v2",
    "bge-rr-base": "BAAI/bge-reranker-base",
    "mxbai-base": "mixedbread-ai/mxbai-rerank-base-v1",
}


def prep(n_jobs: int, n_cvs: int) -> None:
    from datasets import load_dataset

    WORK.mkdir(exist_ok=True)
    meta = pd.read_pickle(META_PATH)
    jobs = meta.sample(n=n_jobs, random_state=SEED).reset_index(drop=True)
    jobs.to_parquet(WORK / "jobs.parquet")

    counts = jobs["keyword"].value_counts()
    usable = set(counts[counts >= 5].index) - {"Other"}
    cands = load_dataset(CANDIDATE_DATASET, split="train").to_pandas()
    cands = cands[cands["Primary Keyword"].isin(usable)]
    cvs = cands.sample(n=n_cvs, random_state=SEED).reset_index(drop=True)[["CV", "Primary Keyword"]]
    cvs.to_parquet(WORK / "cvs.parquet")
    print(f"jobs={len(jobs)} cvs={len(cvs)} usable_keywords={len(usable)} (Other + <5 jobs excluded)")
    print(cvs["Primary Keyword"].value_counts().to_string())


def _chunk_embed(model, texts: list[str], chunk_tokens: int, prefix: str) -> np.ndarray:
    """Chunk + mean-pool like production (ADR-0007), skipping Normalize modules; L2-normalize at the end."""
    import torch
    from sentence_transformers.models import Normalize
    from tqdm import tqdm

    from backend.domain.embeddings import _chunk_text

    model.tokenizer.model_max_length = int(1e9)
    chunks_per_text = [_chunk_text(t, model.tokenizer, chunk_tokens) for t in tqdm(texts, desc="chunking")]
    counts = [len(c) for c in chunks_per_text]
    flat = [prefix + c for chunks in chunks_per_text for c in chunks]
    order = np.argsort([-len(c) for c in flat])
    modules = [m for m in model if not isinstance(m, Normalize)]
    device = next(model.parameters()).device

    out = np.zeros((len(flat), model.get_embedding_dimension()), dtype=np.float32)
    with torch.no_grad():
        for start in tqdm(range(0, len(flat), 32), desc="embedding"):
            idx = order[start : start + 32]
            features = model.preprocess([flat[i] for i in idx])
            features = {k: v.to(device) if torch.is_tensor(v) else v for k, v in features.items()}
            for module in modules:
                features = module(features)
            out[idx] = features["sentence_embedding"].float().cpu().numpy()

    vectors = np.zeros((len(texts), out.shape[1]), dtype=np.float32)
    pos = 0
    for i, n in enumerate(counts):
        if n:
            vectors[i] = out[pos : pos + n].mean(axis=0)
        pos += n
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    return vectors


def embed(keys: list[str]) -> None:
    import torch
    from sentence_transformers import SentenceTransformer

    jobs = pd.read_parquet(WORK / "jobs.parquet")
    cvs = pd.read_parquet(WORK / "cvs.parquet")
    for key in keys:
        spec = BI_MODELS[key]
        if (WORK / f"{key}.jobs.npy").exists():
            print(f"[{key}] cached")
            continue
        t0 = time.monotonic()
        model = SentenceTransformer(spec["name"], trust_remote_code=spec["remote"], device="cuda")
        model.half()
        model.max_seq_length = min(512, model.max_seq_length or 512)
        cv_vecs = _chunk_embed(model, cvs["CV"].tolist(), spec["chunk"], spec["q"])
        t1 = time.monotonic()
        job_vecs = _chunk_embed(model, jobs["job_text"].tolist(), spec["chunk"], spec["d"])
        np.save(WORK / f"{key}.cvs.npy", cv_vecs)
        np.save(WORK / f"{key}.jobs.npy", job_vecs)
        print(f"[{key}] dims={job_vecs.shape[1]} cvs={t1 - t0:.0f}s jobs={time.monotonic() - t1:.0f}s")
        del model
        torch.cuda.empty_cache()


def _rankings(sims: np.ndarray, job_ids: np.ndarray, rel_by_kw: dict, kws: list[str], k: int = 100):
    top = np.argpartition(-sims, k, axis=1)[:, :k]
    rows = np.arange(sims.shape[0])[:, None]
    top = top[rows, np.argsort(-sims[rows, top], axis=1)]
    return [(list(job_ids[top[i]]), rel_by_kw[kws[i]]) for i in range(len(kws))], top


def _load_ground_truth():
    jobs = pd.read_parquet(WORK / "jobs.parquet")
    cvs = pd.read_parquet(WORK / "cvs.parquet")
    job_ids = jobs["id"].to_numpy()
    rel_by_kw = jobs.groupby("keyword")["id"].apply(set).to_dict()
    return jobs, cvs, job_ids, rel_by_kw


def evaluate() -> None:
    from evaluate_scale import mean_metrics

    jobs, cvs, job_ids, rel_by_kw = _load_ground_truth()
    kws = cvs["Primary Keyword"].tolist()
    rows = {}
    for key in BI_MODELS:
        if not (WORK / f"{key}.jobs.npy").exists():
            continue
        sims = np.load(WORK / f"{key}.cvs.npy") @ np.load(WORK / f"{key}.jobs.npy").T
        rankings, _ = _rankings(sims, job_ids, rel_by_kw, kws)
        rows[key] = mean_metrics(rankings)
        by_kw = {}
        for kw in sorted(set(kws)):
            sel = [r for r, k in zip(rankings, kws) if k == kw]
            by_kw[kw] = (mean_metrics(sel)["Precision@10"], len(sel))
        rows[key]["_by_kw"] = by_kw
    table = pd.DataFrame({k: {m: v for m, v in r.items() if m != "_by_kw"} for k, r in rows.items()})
    print(f"N cvs={len(cvs)}, jobs={len(jobs)}, weak label = same Primary Keyword\n")
    print(table.round(3).to_string())
    kw_table = pd.DataFrame({k: {kw: v[0] for kw, v in r["_by_kw"].items()} for k, r in rows.items()})
    kw_table["n_cvs"] = pd.Series({kw: n for kw, (_, n) in next(iter(rows.values()))["_by_kw"].items()})
    print("\nPrecision@10 by CV keyword:\n")
    print(kw_table.round(3).to_string())


def rerank_eval(bi_key: str, n_cvs: int) -> None:
    import torch
    from sentence_transformers import CrossEncoder

    from evaluate_scale import mean_metrics

    jobs, cvs, job_ids, rel_by_kw = _load_ground_truth()
    sample = cvs.sample(n=min(n_cvs, len(cvs)), random_state=SEED)
    kws = sample["Primary Keyword"].tolist()
    sims = np.load(WORK / f"{bi_key}.cvs.npy")[sample.index] @ np.load(WORK / f"{bi_key}.jobs.npy").T
    rankings, top = _rankings(sims, job_ids, rel_by_kw, kws)
    top20 = top[:, :20]
    results = {"no rerank": mean_metrics([(r[:20], rel) for r, rel in rankings])}
    job_texts = jobs["job_text"].tolist()
    cv_texts = sample["CV"].tolist()

    for name, model_id in RERANKERS.items():
        try:
            model = CrossEncoder(model_id, device="cuda", model_kwargs={"torch_dtype": torch.float16})
            t0 = time.monotonic()
            reranked = []
            for i in range(len(kws)):
                pairs = [(cv_texts[i], job_texts[j]) for j in top20[i]]
                scores = model.predict(pairs, batch_size=20, show_progress_bar=False)
                order = np.argsort(-scores)
                reranked.append(([job_ids[top20[i][o]] for o in order], rel_by_kw[kws[i]]))
                if (i + 1) % 20 == 0:
                    print(f"[{name}] {i + 1}/{len(kws)} cvs, {time.monotonic() - t0:.0f}s", flush=True)
            results[name] = mean_metrics(reranked)
            print(f"[{name}] done in {time.monotonic() - t0:.0f}s (GPU)")
            del model
            torch.cuda.empty_cache()
        except Exception as error:  # report, don't hide: a model that fails to load is a finding
            print(f"[{name}] FAILED: {type(error).__name__}: {error}")
    print(f"\nbi-encoder={bi_key}, top-20 reranked, N cvs={len(kws)}\n")
    print(pd.DataFrame(results).round(3).to_string())


def cpu_bench(keys: list[str]) -> None:
    """RAM + latency on 1 torch thread (production has 1 vCPU). Run each key in a fresh process."""
    import psutil
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    torch.set_num_threads(1)
    rss = lambda: round(psutil.Process().memory_info().rss / 1e6)
    cvs = pd.read_parquet(WORK / "cvs.parquet")
    jobs = pd.read_parquet(WORK / "jobs.parquet")
    cv = cvs["CV"].iloc[0]
    for key in keys:
        base = rss()
        if key in BI_MODELS:
            spec = BI_MODELS[key]
            model = SentenceTransformer(spec["name"], trust_remote_code=spec["remote"], device="cpu")
            model.max_seq_length = min(512, model.max_seq_length or 512)
            loaded = rss()
            _chunk_embed(model, [cv], spec["chunk"], spec["q"])
            times = []
            for _ in range(5):
                t0 = time.monotonic()
                _chunk_embed(model, [cv], spec["chunk"], spec["q"])
                times.append(time.monotonic() - t0)
            extra = f"cv_encode_ms={1000 * np.median(times):.0f} dims={model.get_embedding_dimension()}"
        else:
            model = CrossEncoder(RERANKERS[key], device="cpu")
            loaded = rss()
            pairs = [(cv, t) for t in jobs["job_text"].iloc[:20]]
            model.predict(pairs[:2], show_progress_bar=False)
            times = []
            for _ in range(3):
                t0 = time.monotonic()
                model.predict(pairs, batch_size=20, show_progress_bar=False)
                times.append(time.monotonic() - t0)
            extra = f"rerank20_ms={1000 * np.median(times):.0f}"
        params = sum(p.numel() for p in (model.model if key in RERANKERS else model).parameters()) / 1e6
        print(json.dumps(dict(model=key, params_M=round(params), rss_before=base, rss_loaded=loaded, rss_after=rss(), info=extra)))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--jobs", type=int, default=30000)
    p.add_argument("--cvs", type=int, default=1000)
    sub.add_parser("embed").add_argument("keys", nargs="+")
    sub.add_parser("eval")
    p = sub.add_parser("rerank")
    p.add_argument("bi_key")
    p.add_argument("--cvs", type=int, default=100)
    sub.add_parser("cpu").add_argument("keys", nargs="+")
    args = parser.parse_args()
    {
        "prep": lambda: prep(args.jobs, args.cvs),
        "embed": lambda: embed(args.keys),
        "eval": evaluate,
        "rerank": lambda: rerank_eval(args.bi_key, args.cvs),
        "cpu": lambda: cpu_bench(args.keys),
    }[args.cmd]()


if __name__ == "__main__":
    main()
