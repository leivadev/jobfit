import io

import faiss
import numpy as np
import pandas as pd

from backend.config import Settings
from backend.domain.search import JobSearchIndex
from backend.r2 import JOBS_INDEX_KEY, JOBS_METADATA_KEY, build_r2_client

__all__ = ["JOBS_INDEX_KEY", "JOBS_METADATA_KEY", "build_r2_client", "load_artifacts"]


def load_artifacts(settings: Settings, client=None) -> tuple[JobSearchIndex, pd.DataFrame]:
    """Download the Job Index and Job Metadata Artifacts from R2 into memory.

    Called once at startup (see ADR-0006). Any failure — unreachable R2,
    missing key, corrupt Artifact — raises rather than falling back, so the
    caller can fail the process fast instead of serving a partial/stale state.
    """
    client = client or build_r2_client(settings)

    index_bytes = _get_object_bytes(client, settings.r2_bucket_name, JOBS_INDEX_KEY)
    metadata_bytes = _get_object_bytes(client, settings.r2_bucket_name, JOBS_METADATA_KEY)

    index = faiss.deserialize_index(np.frombuffer(index_bytes, dtype=np.uint8))
    metadata = pd.read_parquet(io.BytesIO(metadata_bytes))

    return JobSearchIndex(index), metadata


def _get_object_bytes(client, bucket: str, key: str) -> bytes:
    return client.get_object(Bucket=bucket, Key=key)["Body"].read()
