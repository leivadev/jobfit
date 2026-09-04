import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import pytest
from build_index import run
from dataset import load_raw_jobs

from backend.config import Settings

pytestmark = pytest.mark.integration


def test_build_index_runs_end_to_end_on_small_subset(tmp_path):
    """The only place load_raw_jobs, embed_texts, and the R2 upload run together."""
    df = load_raw_jobs()
    subset = df[df["Primary Keyword"] == "Python"].head(30)
    settings = Settings()
    index_path = tmp_path / "jobs.index"
    metadata_path = tmp_path / "jobs_metadata.parquet"

    index, metadata = run(subset, index_path, metadata_path, settings)

    assert index.ntotal == len(metadata)
    assert index.ntotal > 0
    assert index_path.exists()
    assert metadata_path.exists()
