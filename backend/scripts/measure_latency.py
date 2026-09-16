"""Measures per-request latency of bi-encoder-only vs. bi-encoder + cross-encoder rerank.

Complements evaluate.py/evaluate_scale.py's quality metrics with the other half of the
question: is the rerank's quality gain worth its cost? Forces CPU (`CUDA_VISIBLE_DEVICES`
unset to ""), since production (Azure Container Apps) is a CPU-only deployment — GPU
timings here would misrepresent what a real /recommend request costs.

Manual, one-time invocation (`uv run python scripts/measure_latency.py [-n 50]`), same
offline-script pattern as build_index.py / evaluate.py.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # must precede any torch import (see module docstring)

import argparse
import statistics
import sys
import time
from pathlib import Path

import faiss
import pandas as pd
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backend.domain.embeddings import BiEncoder
from backend.domain.matching import KEYWORD_VOCABULARY
from backend.domain.rerank import TOP_K as CUTOFF
from backend.domain.rerank import CrossEncoder, ShortlistItem, rerank
from backend.domain.search import TOP_K as SHORTLIST_SIZE
from backend.domain.search import JobSearchIndex

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
CANDIDATE_DATASET_NAME = "lang-uk/recruitment-dataset-candidate-profiles-english"
SEED = 42
WARMUP_REQUESTS = 3


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    index = min(int(len(values) * p), len(values) - 1)
    return values[index]


def summarize(label: str, samples_ms: list[float]) -> str:
    return (
        f"{label:<28}{statistics.mean(samples_ms):>10.1f}"
        f"{statistics.median(samples_ms):>10.1f}{percentile(samples_ms, 0.95):>10.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-requests", type=int, default=50)
    args = parser.parse_args()

    import torch

    print(f"[measure_latency] torch device forced to cpu (torch.cuda.is_available()={torch.cuda.is_available()})")

    print("[measure_latency] loading local Job Index artifacts...")
    search_index = JobSearchIndex(faiss.read_index(str(ARTIFACTS_DIR / "jobs.index")))
    metadata = pd.read_parquet(ARTIFACTS_DIR / "jobs_metadata.parquet")

    print("[measure_latency] loading candidate profiles dataset...")
    candidates = load_dataset(CANDIDATE_DATASET_NAME, split="train").to_pandas()
    qualifying = candidates[candidates["Primary Keyword"].isin(KEYWORD_VOCABULARY)]
    sample = qualifying.sample(n=args.num_requests + WARMUP_REQUESTS, random_state=SEED)

    print("[measure_latency] loading Bi-encoder + Cross-encoder models...")
    bi_encoder = BiEncoder()
    cross_encoder = CrossEncoder()

    encode_ms, baseline_search_ms, shortlist_search_ms, rerank_ms = [], [], [], []
    baseline_total_ms, rerank_total_ms = [], []

    for i, (_, row) in enumerate(sample.iterrows(), start=1):
        candidate_profile = row["CV"]

        t0 = time.perf_counter()
        query_vector = bi_encoder.encode([candidate_profile])[0]
        t1 = time.perf_counter()

        search_index.search(query_vector, top_k=CUTOFF)
        t2 = time.perf_counter()

        shortlist_results = search_index.search(query_vector, top_k=SHORTLIST_SIZE)
        t3 = time.perf_counter()

        shortlist = [
            ShortlistItem(job_row=result.job_row, job_text=metadata.iloc[result.job_row]["job_text"])
            for result in shortlist_results
        ]
        rerank(candidate_profile, shortlist, cross_encoder, top_k=CUTOFF)
        t4 = time.perf_counter()

        if i <= WARMUP_REQUESTS:
            continue  # discard cold-start iterations (lazy model/tokenizer init)

        encode_ms.append((t1 - t0) * 1000)
        baseline_search_ms.append((t2 - t1) * 1000)
        shortlist_search_ms.append((t3 - t2) * 1000)
        rerank_ms.append((t4 - t3) * 1000)
        baseline_total_ms.append((t2 - t0) * 1000)
        rerank_total_ms.append((t4 - t0) * 1000)

    print(f"\n[measure_latency] N={args.num_requests} requests (+{WARMUP_REQUESTS} warmup, discarded), device=cpu")
    print(f"{'stage':<28}{'mean(ms)':>10}{'p50(ms)':>10}{'p95(ms)':>10}")
    print(summarize("encode (bi-encoder)", encode_ms))
    print(summarize(f"baseline search (top_k={CUTOFF})", baseline_search_ms))
    print(summarize("baseline total", baseline_total_ms))
    print(summarize(f"shortlist search (top_k={SHORTLIST_SIZE})", shortlist_search_ms))
    print(summarize("cross-encoder rerank", rerank_ms))
    print(summarize("rerank total", rerank_total_ms))

    overhead_ms = statistics.mean(rerank_total_ms) - statistics.mean(baseline_total_ms)
    overhead_pct = overhead_ms / statistics.mean(baseline_total_ms) * 100
    print(f"\nrerank overhead: +{overhead_ms:.1f}ms (+{overhead_pct:.0f}%) per request vs. baseline")


if __name__ == "__main__":
    main()
