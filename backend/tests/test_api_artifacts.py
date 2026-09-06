import io
from unittest.mock import MagicMock

import faiss
import numpy as np
import pandas as pd
import pytest

from backend.api.artifacts import (
    JOBS_INDEX_KEY,
    JOBS_METADATA_KEY,
    build_r2_client,
    load_artifacts,
)
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


def make_fixture_artifacts() -> tuple[bytes, bytes]:
    embeddings = np.random.default_rng(0).random((3, 8), dtype=np.float32)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(8)
    index.add(embeddings)
    index_bytes = faiss.serialize_index(index).tobytes()

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
    buf = io.BytesIO()
    metadata.to_parquet(buf, index=False)

    return index_bytes, buf.getvalue()


def make_fake_client(index_bytes: bytes, metadata_bytes: bytes):
    bodies = {
        JOBS_INDEX_KEY: index_bytes,
        JOBS_METADATA_KEY: metadata_bytes,
    }

    def get_object(Bucket, Key):
        return {"Body": io.BytesIO(bodies[Key])}

    client = MagicMock()
    client.get_object.side_effect = get_object
    return client


def test_build_r2_client_targets_r2_endpoint():
    settings = make_settings()

    client = build_r2_client(settings)

    assert client.meta.endpoint_url == settings.r2_endpoint_url


def test_load_artifacts_returns_search_index_and_metadata():
    settings = make_settings()
    index_bytes, metadata_bytes = make_fixture_artifacts()
    client = make_fake_client(index_bytes, metadata_bytes)

    search_index, metadata = load_artifacts(settings, client=client)

    assert len(metadata) == 3
    assert list(metadata["position"]) == ["Python Developer", "QA Engineer", "DevOps Engineer"]
    results = search_index.search(np.ones(8, dtype=np.float32))
    assert len(results) == 3


def test_load_artifacts_reads_from_configured_bucket():
    settings = make_settings()
    index_bytes, metadata_bytes = make_fixture_artifacts()
    client = make_fake_client(index_bytes, metadata_bytes)

    load_artifacts(settings, client=client)

    client.get_object.assert_any_call(Bucket=settings.r2_bucket_name, Key=JOBS_INDEX_KEY)
    client.get_object.assert_any_call(Bucket=settings.r2_bucket_name, Key=JOBS_METADATA_KEY)


def test_load_artifacts_propagates_errors():
    settings = make_settings()
    client = MagicMock()
    client.get_object.side_effect = RuntimeError("R2 unreachable")

    with pytest.raises(RuntimeError):
        load_artifacts(settings, client=client)
