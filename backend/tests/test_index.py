import faiss
import numpy as np
import pandas as pd
import pytest
from index import build_faiss_index, save_artifacts


def make_embeddings(n=5, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, dim), dtype=np.float32)


def search_top1(index, embeddings, row):
    """Normalize the given row of `embeddings` and search `index` for its nearest neighbor."""
    query = np.ascontiguousarray(embeddings[row : row + 1])
    faiss.normalize_L2(query)
    distances, indices = index.search(query, 1)
    return indices[0][0], distances[0][0]


def test_build_faiss_index_ntotal_matches_input_count():
    embeddings = make_embeddings(n=5)

    index = build_faiss_index(embeddings)

    assert index.ntotal == 5


def test_build_faiss_index_self_similarity_is_one():
    embeddings = make_embeddings(n=5)
    index = build_faiss_index(embeddings)

    matched_row, similarity = search_top1(index, embeddings, 2)

    assert matched_row == 2
    assert similarity == pytest.approx(1.0, abs=1e-5)


def test_build_faiss_index_does_not_mutate_input():
    embeddings = make_embeddings(n=3)
    original = embeddings.copy()

    build_faiss_index(embeddings)

    assert np.array_equal(embeddings, original)


def test_save_artifacts_writes_index_and_metadata(tmp_path):
    embeddings = make_embeddings(n=3)
    index = build_faiss_index(embeddings)
    metadata = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "position": ["Python Developer", "QA Engineer", "DevOps Engineer"],
            "company": ["Acme", "Globex", "Initech"],
            "exp_years": ["1-3", "3-5", "5+"],
            "keyword": ["Python", "QA", "DevOps"],
            "job_text": ["Python Developer text", "QA Engineer text", "DevOps Engineer text"],
        }
    )
    index_path = tmp_path / "jobs.index"
    metadata_path = tmp_path / "jobs_metadata.parquet"

    save_artifacts(index, metadata, index_path, metadata_path)

    assert index_path.exists()
    assert metadata_path.exists()


def test_save_artifacts_preserves_row_alignment(tmp_path):
    """Row i of the FAISS index must correspond to row i of the metadata."""
    embeddings = make_embeddings(n=4)
    index = build_faiss_index(embeddings)
    metadata = pd.DataFrame(
        {
            "id": ["job-0", "job-1", "job-2", "job-3"],
            "position": ["P0", "P1", "P2", "P3"],
            "company": ["C0", "C1", "C2", "C3"],
            "exp_years": ["1-3"] * 4,
            "keyword": ["Python"] * 4,
            "job_text": ["T0", "T1", "T2", "T3"],
        }
    )
    index_path = tmp_path / "jobs.index"
    metadata_path = tmp_path / "jobs_metadata.parquet"

    save_artifacts(index, metadata, index_path, metadata_path)

    loaded_index = faiss.read_index(str(index_path))
    loaded_metadata = pd.read_parquet(metadata_path)

    normalized_embeddings = embeddings.copy()
    faiss.normalize_L2(normalized_embeddings)

    for i in range(4):
        np.testing.assert_allclose(loaded_index.reconstruct(i), normalized_embeddings[i], atol=1e-6)
        assert loaded_metadata.iloc[i]["id"] == f"job-{i}"
