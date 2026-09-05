import faiss
import numpy as np
import pytest

from backend.domain.search import TOP_K, JobSearchIndex, SearchResult


def make_index(n=5, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    embeddings = rng.random((n, dim), dtype=np.float32)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, embeddings


def test_search_returns_top_100_by_default():
    index, embeddings = make_index(n=150)
    search_index = JobSearchIndex(index)

    results = search_index.search(embeddings[0])

    assert len(results) == TOP_K


def test_search_respects_smaller_index_than_top_k():
    index, embeddings = make_index(n=5)
    search_index = JobSearchIndex(index)

    results = search_index.search(embeddings[0])

    assert len(results) == 5


def test_search_finds_exact_match_as_top_result():
    index, embeddings = make_index(n=10)
    search_index = JobSearchIndex(index)

    results = search_index.search(embeddings[3])

    assert results[0] == SearchResult(
        job_row=3, similarity_score=pytest.approx(1.0, abs=1e-5)
    )


def test_search_results_are_sorted_descending_by_similarity_score():
    index, embeddings = make_index(n=20)
    search_index = JobSearchIndex(index)

    results = search_index.search(embeddings[0])

    scores = [result.similarity_score for result in results]
    assert scores == sorted(scores, reverse=True)


def test_search_normalizes_unnormalized_query_vector():
    index, embeddings = make_index(n=10)
    search_index = JobSearchIndex(index)
    unnormalized_query = embeddings[4] * 37.0

    results = search_index.search(unnormalized_query)

    assert results[0].job_row == 4
    assert results[0].similarity_score == pytest.approx(1.0, abs=1e-5)


def test_search_does_not_mutate_query_vector():
    index, embeddings = make_index(n=5)
    search_index = JobSearchIndex(index)
    query = embeddings[0].copy()
    original = query.copy()

    search_index.search(query)

    assert np.array_equal(query, original)


def test_search_respects_custom_top_k():
    index, embeddings = make_index(n=50)
    search_index = JobSearchIndex(index)

    results = search_index.search(embeddings[0], top_k=10)

    assert len(results) == 10
