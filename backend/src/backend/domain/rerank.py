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
