from pathlib import Path

import faiss
import numpy as np
import pandas as pd


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a cosine-similarity FAISS index over Job embeddings (see ADR-0002).

    L2-normalizes a copy of `embeddings`, then indexes with IndexFlatIP so
    inner product search is equivalent to cosine similarity. Pure.
    """
    normalized = np.array(embeddings, dtype=np.float32, copy=True, order="C")
    faiss.normalize_L2(normalized)

    index = faiss.IndexFlatIP(normalized.shape[1])
    index.add(normalized)
    return index


def save_artifacts(
    index: faiss.Index,
    metadata: pd.DataFrame,
    index_path: Path,
    metadata_path: Path,
) -> None:
    """Write the Job Index and Job Metadata Artifacts to local paths."""
    faiss.write_index(index, str(index_path))
    metadata.to_parquet(metadata_path, index=False)
