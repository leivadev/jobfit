from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sentence_transformers import CrossEncoder as _SentenceTransformersCrossEncoder

TOP_K = 10


@dataclass(frozen=True)
class ShortlistItem:
    job_row: int
    job_text: str


@dataclass(frozen=True)
class RerankResult:
    job_row: int
    rerank_score: float


class CrossEncoderModel(Protocol):
    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray: ...


class CrossEncoder:
    """`ms-marco-MiniLM-L6-v2` Cross-encoder that scores (Candidate Profile, Job) pairs.

    Scores the top-100 shortlist from `search.py`; too slow to run over the
    whole Job Index (see CONTEXT.md's Cross-encoder definition).
    """

    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
    # Pinned revision, not the moving default branch — same rationale as
    # BiEncoder.MODEL_REVISION.
    MODEL_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"

    def __init__(self) -> None:
        self.model = _SentenceTransformersCrossEncoder(
            self.MODEL_NAME, revision=self.MODEL_REVISION
        )

    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        return np.asarray(self.model.predict(pairs), dtype=np.float32)


def rerank(
    candidate_profile: str,
    shortlist: list[ShortlistItem],
    model: CrossEncoderModel,
    top_k: int = TOP_K,
) -> list[RerankResult]:
    """Score `shortlist` with the Cross-encoder and return the top-k by Rerank Score.

    Pure given `model`: `model.predict` may not be (e.g. a real CrossEncoder
    moves itself onto a device), but this function itself has no side effects.
    """
    if not shortlist:
        return []

    pairs = [(candidate_profile, item.job_text) for item in shortlist]
    scores = model.predict(pairs)

    results = [
        RerankResult(job_row=item.job_row, rerank_score=float(score))
        for item, score in zip(shortlist, scores)
    ]
    results.sort(key=lambda result: result.rerank_score, reverse=True)
    return results[:top_k]


@dataclass(frozen=True)
class TieredJob:
    """A shortlisted Job carrying its Rerank Score plus precomputed `keyword_match`/`exp_distance`.

    Decoupled from `matching.CandidateSignals` and Job Metadata so
    `tier_by_fit` stays pure: callers compute `keyword_match`/`exp_distance`
    from Job Metadata first, then hand in the plain result.
    """

    job_row: int
    rerank_score: float
    keyword_match: bool
    exp_distance: int | None


def tier_by_fit(jobs: list[TieredJob]) -> list[TieredJob]:
    """Tiered sort: `keyword_match` first, then `exp_distance` ascending, then `rerank_score` descending as tiebreak.

    Applied as three stable sorts from least to most significant criterion, so
    a criterion that's constant across `jobs` (e.g. every `exp_distance` is
    `None` because the candidate didn't declare `exp_years`) leaves the prior
    order untouched — ranking degrades to pure `rerank_score` ordering exactly
    when a signal wasn't provided, with no special-casing needed. Within the
    `exp_distance` pass, a job whose distance is `None` sorts after every job
    with a known distance in the same `keyword_match` tier; this is a genuine
    no-op only when every job's distance is `None` (signal not declared) — for
    the rarer case of one job's Job Metadata carrying an off-scale
    `exp_years` bucket alongside others with known distances, it sorts that
    job last within its tier rather than truly neutrally. Pure.
    """
    ranked = sorted(jobs, key=lambda job: job.rerank_score, reverse=True)
    ranked.sort(key=lambda job: (job.exp_distance is None, job.exp_distance))
    ranked.sort(key=lambda job: not job.keyword_match)
    return ranked
