"""Ranking metrics shared by evaluate.py and evaluate_scale.py."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backend.domain.rerank import RESPONSE_SIZE as DISPLAY_CUTOFF
from backend.domain.search import SHORTLIST_SIZE


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
    their presence in the top k (unlike Precision@k)."""
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


def average_precision_at_k(
    ranked_ids: list[str], relevant_ids: set[str], k: int
) -> float:
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
        name: sum(metric_fn(ranked, relevant, k) for ranked, relevant in rankings)
        / len(rankings)
        for name, metric_fn, k in METRICS
    }
