import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import build_index
import pytest
from build_index import run
from dataset import load_raw_jobs

from backend.config import Settings

pytestmark = pytest.mark.integration


def test_build_index_runs_end_to_end_on_small_subset(tmp_path, monkeypatch):
    """The only place load_raw_jobs, embed_texts, and the R2 upload run together.

    Uses the real dataset and embedding model but a mocked R2 client: the real
    client would overwrite the production jobs.index/jobs_metadata.parquet at
    ADR-0006's fixed keys, which is exactly what corrupted them before.
    """
    df = load_raw_jobs()
    subset = df[df["Primary Keyword"] == "Python"].head(30)
    settings = Settings()
    index_path = tmp_path / "jobs.index"
    metadata_path = tmp_path / "jobs_metadata.parquet"
    mock_client = MagicMock()
    monkeypatch.setattr(build_index, "build_r2_client", lambda settings: mock_client)

    index, metadata = run(subset, index_path, metadata_path, settings)

    assert index.ntotal == len(metadata)
    assert index.ntotal > 0
    assert index_path.exists()
    assert metadata_path.exists()
    assert mock_client.upload_file.call_count == 2
