from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sentence_transformers import CrossEncoder as _SentenceTransformersCrossEncoder

RESPONSE_SIZE = 10


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

    Too slow to run over the whole Job Index, so it only scores the `search.py`
    shortlist (see CONTEXT.md's Cross-encoder definition).
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
    top_k: int = RESPONSE_SIZE,
) -> list[RerankResult]:
    """Score `shortlist` with the Cross-encoder and return the top-k by Rerank Score."""
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

    Precomputed by the caller so `tier_by_fit` needs no Job Metadata.
    """

    job_row: int
    rerank_score: float
    keyword_match: bool
    exp_distance: int | None


def tier_by_fit(jobs: list[TieredJob]) -> list[TieredJob]:
    """Tiered sort: `keyword_match`, then `exp_distance` ascending, then `rerank_score` descending.

    Three stable sorts, least to most significant, so a criterion that's constant
    across `jobs` (e.g. no declared `exp_years`) leaves the prior order untouched.
    A `None` distance sorts after known distances within its tier.
    """
    ranked = sorted(jobs, key=lambda job: job.rerank_score, reverse=True)
    ranked.sort(key=lambda job: (job.exp_distance is None, job.exp_distance))
    ranked.sort(key=lambda job: not job.keyword_match)
    return ranked
