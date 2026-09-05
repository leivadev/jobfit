import numpy as np
import pytest

from backend.domain.rerank import TOP_K, RerankResult, ShortlistItem, rerank


class FakeCrossEncoderModel:
    """Synthetic Cross-encoder: returns a fixed score per pair, keyed by job text."""

    def __init__(self, scores_by_job_text: dict[str, float]) -> None:
        self._scores_by_job_text = scores_by_job_text
        self.received_pairs: list[tuple[str, str]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        self.received_pairs = pairs
        return np.array(
            [self._scores_by_job_text[job_text] for _, job_text in pairs],
            dtype=np.float32,
        )


def make_shortlist(n: int) -> list[ShortlistItem]:
    return [ShortlistItem(job_row=i, job_text=f"job-{i}") for i in range(n)]


def test_rerank_returns_top_10_by_default():
    shortlist = make_shortlist(100)
    model = FakeCrossEncoderModel(
        {item.job_text: float(item.job_row) for item in shortlist}
    )

    results = rerank("candidate profile", shortlist, model)

    assert len(results) == TOP_K


def test_rerank_orders_by_score_descending():
    shortlist = make_shortlist(5)
    model = FakeCrossEncoderModel(
        {"job-0": 0.1, "job-1": 0.9, "job-2": 0.5, "job-3": 0.7, "job-4": 0.3}
    )

    results = rerank("candidate profile", shortlist, model)

    assert [r.job_row for r in results] == [1, 3, 2, 4, 0]


def test_rerank_returns_rerank_result_with_job_index_and_score():
    shortlist = make_shortlist(1)
    model = FakeCrossEncoderModel({"job-0": 0.42})

    results = rerank("candidate profile", shortlist, model)

    assert results == [RerankResult(job_row=0, rerank_score=pytest.approx(0.42))]


def test_rerank_respects_custom_top_k():
    shortlist = make_shortlist(20)
    model = FakeCrossEncoderModel(
        {item.job_text: float(item.job_row) for item in shortlist}
    )

    results = rerank("candidate profile", shortlist, model, top_k=3)

    assert len(results) == 3
    assert [r.job_row for r in results] == [19, 18, 17]


def test_rerank_pairs_candidate_profile_with_each_job_text():
    shortlist = make_shortlist(3)
    model = FakeCrossEncoderModel({item.job_text: 0.0 for item in shortlist})

    rerank("candidate profile", shortlist, model)

    assert model.received_pairs == [
        ("candidate profile", "job-0"),
        ("candidate profile", "job-1"),
        ("candidate profile", "job-2"),
    ]


def test_rerank_returns_empty_list_for_empty_shortlist():
    model = FakeCrossEncoderModel({})

    results = rerank("candidate profile", [], model)

    assert results == []


def test_rerank_returns_fewer_than_top_k_when_shortlist_is_smaller():
    shortlist = make_shortlist(4)
    model = FakeCrossEncoderModel(
        {item.job_text: float(item.job_row) for item in shortlist}
    )

    results = rerank("candidate profile", shortlist, model)

    assert len(results) == 4
