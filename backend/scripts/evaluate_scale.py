"""Large-N evaluation using an automated weak relevance label instead of hand-labeled pairs.

Complements `evaluate.py`'s 25 hand-labeled gold pairs: that eval answers "does the model
find *the* right job for this candidate" (precise but N=25, not representative of 210k
candidates). This script answers a coarser question at real scale, with a label that costs
nothing to compute: "relevant jobs for a candidate" = all jobs sharing the candidate's
`Primary Keyword` (same 18-value vocabulary already used for gold-set sampling, see
`docs/design/phase-8-evaluation.md`). No candidate/job ids are hardcoded — the sample is
drawn fresh each run via a fixed seed, from whatever the datasets currently contain.

Because a keyword category can hold thousands of jobs, Precision@10 is the metric that
matters here: does reranking increase the fraction of the top 10 that's topically
on-target.

Manual, one-time invocation (`uv run python scripts/evaluate_scale.py [--sample-frac 0.01]
[--limit N]`), same offline-script pattern as `build_index.py` / `evaluate.py`.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import faiss
import pandas as pd
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backend.domain.embeddings import BiEncoder
from backend.domain.matching import KEYWORD_VOCABULARY
from backend.domain.rerank import TOP_K as DISPLAY_CUTOFF
from backend.domain.rerank import CrossEncoder, ShortlistItem, rerank
from backend.domain.search import TOP_K as SHORTLIST_SIZE
from backend.domain.search import JobSearchIndex

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
CANDIDATE_DATASET_NAME = "lang-uk/recruitment-dataset-candidate-profiles-english"
SEED = 42


def precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    hits = sum(1 for job_id in ranked_ids[:k] if job_id in relevant_ids)
    return hits / k


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    for rank, job_id in enumerate(ranked_ids[:k], start=1):
        if job_id in relevant_ids:
            return 1 / rank
    return 0.0


def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Binary-relevance NDCG@k: rewards ranking multiple relevant items higher, not just
    their presence in the top k. Matters more here than in evaluate.py's gold set, since a
    candidate's weak-labeled relevant set can hold many jobs, not just one."""
    if not relevant_ids:
        return 0.0
    dcg = sum(
        1 / math.log2(rank + 1)
        for rank, job_id in enumerate(ranked_ids[:k], start=1)
        if job_id in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """AP@k, normalized by min(k, |relevant|) so a perfect ranking scores 1.0 even when
    fewer than k relevant items exist. Averaging this across queries gives MAP@k."""
    if not relevant_ids:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, job_id in enumerate(ranked_ids[:k], start=1):
        if job_id in relevant_ids:
            hits += 1
            precision_sum += hits / rank
    denominator = min(len(relevant_ids), k)
    return precision_sum / denominator if denominator > 0 else 0.0


def hit_rate_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """1.0 if any relevant item is in the top k, else 0.0."""
    return 1.0 if any(job_id in relevant_ids for job_id in ranked_ids[:k]) else 0.0


# (name, fn, k) — Precision/NDCG/HitRate stay at DISPLAY_CUTOFF (what /recommend actually
# shows); MRR/MAP also get a SHORTLIST_SIZE reading, to separate a retrieval miss
# (bi-encoder never shortlisted the right job) from a reranking miss (it was in the
# shortlist, cross-encoder just didn't surface it in the top DISPLAY_CUTOFF). Both
# constants are imported from production (search.py/rerank.py), not hardcoded, so this
# eval can't silently drift out of sync with what /recommend actually does.
METRICS = [
    (f"Precision@{DISPLAY_CUTOFF}", precision_at_k, DISPLAY_CUTOFF),
    (f"MRR@{DISPLAY_CUTOFF}", reciprocal_rank, DISPLAY_CUTOFF),
    (f"MRR@{SHORTLIST_SIZE}", reciprocal_rank, SHORTLIST_SIZE),
    (f"NDCG@{DISPLAY_CUTOFF}", ndcg_at_k, DISPLAY_CUTOFF),
    (f"MAP@{DISPLAY_CUTOFF}", average_precision_at_k, DISPLAY_CUTOFF),
    (f"MAP@{SHORTLIST_SIZE}", average_precision_at_k, SHORTLIST_SIZE),
    (f"HitRate@{DISPLAY_CUTOFF}", hit_rate_at_k, DISPLAY_CUTOFF),
]


def mean_metrics(rankings: list[tuple[list[str], set[str]]]) -> dict[str, float]:
    return {
        name: sum(metric_fn(ranked, relevant, k) for ranked, relevant in rankings) / len(rankings)
        for name, metric_fn, k in METRICS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-frac", type=float, default=0.01)
    parser.add_argument("--limit", type=int, default=None, help="cap sample size, for smoke-testing timing")
    args = parser.parse_args()

    print("[evaluate_scale] loading local Job Index + Metadata artifacts...")
    search_index = JobSearchIndex(faiss.read_index(str(ARTIFACTS_DIR / "jobs.index")))
    metadata = pd.read_parquet(ARTIFACTS_DIR / "jobs_metadata.parquet")
    job_ids = metadata["id"].to_numpy()
    jobs_by_keyword = metadata.groupby("keyword")["id"].apply(set).to_dict()

    print("[evaluate_scale] loading candidate profiles dataset...")
    candidates = load_dataset(CANDIDATE_DATASET_NAME, split="train").to_pandas()
    qualifying = candidates[candidates["Primary Keyword"].isin(KEYWORD_VOCABULARY)]
    sample = qualifying.sample(frac=args.sample_frac, random_state=SEED)
    if args.limit is not None:
        sample = sample.head(args.limit)
    print(f"[evaluate_scale] sampled {len(sample)} of {len(qualifying)} qualifying candidates "
          f"({len(candidates)} total in dataset)")

    print("[evaluate_scale] loading Bi-encoder + Cross-encoder models...")
    bi_encoder = BiEncoder()
    cross_encoder = CrossEncoder()

    baseline_rankings: list[tuple[list[str], set[str]]] = []
    rerank_rankings: list[tuple[list[str], set[str]]] = []

    start = time.monotonic()
    for i, (_, row) in enumerate(sample.iterrows(), start=1):
        candidate_profile = row["CV"]
        relevant_ids = jobs_by_keyword.get(row["Primary Keyword"], set())

        query_vector = bi_encoder.encode([candidate_profile])[0]

        # Store full SHORTLIST_SIZE-length rankings — metric fns slice to k themselves,
        # so one pass here covers both the @10 and @100 readings.
        baseline_results = search_index.search(query_vector, top_k=SHORTLIST_SIZE)
        baseline_ranked_ids = [job_ids[result.job_row] for result in baseline_results]
        baseline_rankings.append((baseline_ranked_ids, relevant_ids))

        shortlist = [
            ShortlistItem(job_row=result.job_row, job_text=metadata.iloc[result.job_row]["job_text"])
            for result in baseline_results
        ]
        rerank_results = rerank(candidate_profile, shortlist, cross_encoder, top_k=SHORTLIST_SIZE)
        rerank_ranked_ids = [job_ids[result.job_row] for result in rerank_results]
        rerank_rankings.append((rerank_ranked_ids, relevant_ids))

        if i % 25 == 0 or i == len(sample):
            elapsed = time.monotonic() - start
            rate = elapsed / i
            print(f"[evaluate_scale] {i}/{len(sample)} candidates "
                  f"({rate:.2f}s/candidate, ~{rate * len(sample) / 60:.1f}min total est.)")

    baseline_metrics = mean_metrics(baseline_rankings)
    rerank_metrics = mean_metrics(rerank_rankings)

    print(f"\n[evaluate_scale] N={len(sample)} candidates, weak label = same Primary Keyword")
    print(f"{'metric':<15}{'baseline':>12}{'rerank':>12}{'delta':>12}")
    for label, _, _ in METRICS:
        baseline_value = baseline_metrics[label]
        rerank_value = rerank_metrics[label]
        delta = rerank_value - baseline_value
        print(f"{label:<15}{baseline_value:>12.3f}{rerank_value:>12.3f}{delta:>+12.3f}")


if __name__ == "__main__":
    main()
