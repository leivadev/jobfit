from pathlib import Path

import boto3

from backend.config import Settings

JOBS_INDEX_KEY = "jobs.index"
JOBS_METADATA_KEY = "jobs_metadata.parquet"


def build_r2_client(settings: Settings):
    """Build a boto3 S3 client against R2's S3-compatible endpoint (see ADR-0003)."""
    return boto3.client(
        "s3",
        endpoint_url=settings.r2_endpoint_url,
        aws_access_key_id=settings.r2_access_key_id,
        aws_secret_access_key=settings.r2_secret_access_key,
    )


def upload_artifacts(
    client,
    settings: Settings,
    index_path: Path,
    metadata_path: Path,
) -> None:
    """Upload the Job Index and Job Metadata Artifacts to fixed R2 keys, overwriting the previous run in place (see ADR-0006)."""
    client.upload_file(str(index_path), settings.r2_bucket_name, JOBS_INDEX_KEY)
    client.upload_file(str(metadata_path), settings.r2_bucket_name, JOBS_METADATA_KEY)
