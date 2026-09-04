from pathlib import Path
from unittest.mock import MagicMock

from storage import JOBS_INDEX_KEY, JOBS_METADATA_KEY, build_r2_client, upload_artifacts

from backend.config import Settings


def make_settings():
    return Settings(
        _env_file=None,
        r2_bucket_name="jobfit-artifacts",
        r2_account_id="account-id",
        r2_access_key_id="access-key",
        r2_secret_access_key="secret-key",
        r2_endpoint_url="https://account-id.r2.cloudflarestorage.com",
    )


def test_build_r2_client_targets_r2_endpoint():
    settings = make_settings()

    client = build_r2_client(settings)

    assert client.meta.endpoint_url == settings.r2_endpoint_url


def test_upload_artifacts_uploads_index_and_metadata_to_fixed_keys():
    settings = make_settings()
    client = MagicMock()
    index_path = Path("jobs.index")
    metadata_path = Path("jobs_metadata.parquet")

    upload_artifacts(client, settings, index_path, metadata_path)

    client.upload_file.assert_any_call(str(index_path), settings.r2_bucket_name, JOBS_INDEX_KEY)
    client.upload_file.assert_any_call(str(metadata_path), settings.r2_bucket_name, JOBS_METADATA_KEY)
    assert client.upload_file.call_count == 2
