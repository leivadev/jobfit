from pathlib import Path

from backend.config import Settings
from backend.r2 import JOBS_INDEX_KEY, JOBS_METADATA_KEY


def upload_artifacts(
    client,
    settings: Settings,
    index_path: Path,
    metadata_path: Path,
) -> None:
    """Upload the Job Index and Job Metadata Artifacts to fixed R2 keys, overwriting the previous run in place (see ADR-0006)."""
    client.upload_file(str(index_path), settings.r2_bucket_name, JOBS_INDEX_KEY)
    client.upload_file(str(metadata_path), settings.r2_bucket_name, JOBS_METADATA_KEY)
