from dataclasses import dataclass

import faiss
import numpy as np

TOP_K = 100


@dataclass(frozen=True)
class SearchResult:
    job_row: int
    similarity_score: float


class JobSearchIndex:
    """Adapter over the FAISS Job Index (ADR-0002).

    FAISS itself is never referenced outside this module — callers pass a
    query vector and get back SearchResults.
    """

    def __init__(self, index: faiss.Index) -> None:
        self._index = index

    def search(
        self, query_vector: np.ndarray, top_k: int = TOP_K
    ) -> list[SearchResult]:
        """Retrieve the top-k nearest Jobs to `query_vector` by Similarity Score.

        L2-normalizes a copy of the query vector so inner-product search over
        the (already-normalized) Job Index is equivalent to cosine similarity,
        matching how the index was built (see `app/index.py::build_faiss_index`).
        Pure: `query_vector` is copied before normalization, never mutated.
        """
        query = np.array(query_vector, dtype=np.float32, copy=True, order="C").reshape(
            1, -1
        )
        faiss.normalize_L2(query)

        similarities, rows = self._index.search(query, top_k)

        return [
            SearchResult(job_row=int(row), similarity_score=float(score))
            for row, score in zip(rows[0], similarities[0])
            if row != -1
        ]
