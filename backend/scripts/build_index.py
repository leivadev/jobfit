"""Orchestrates the offline pipeline end-to-end: dataset -> embeddings -> FAISS index -> R2.

Manual, one-time invocation (`uv run python scripts/build_index.py`); rerun by hand if the
source dataset changes. No cron/CI trigger — see docs/design/phase-1-offline-pipeline.md.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

import faiss
import pandas as pd
from dataset import deduplicate_jobs, filter_jobs, load_raw_jobs
from dotenv import load_dotenv
from embeddings import embed_texts, load_model
from index import build_faiss_index, save_artifacts
from storage import build_r2_client, upload_artifacts
from text import build_job_text, clean_text

load_dotenv(Path(__file__).parent.parent / ".env")

from backend.config import Settings

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"

# Verified against the real dataset's 45 Primary Keyword values; excludes
# long-tail values and single-row mislabeled outliers (see phase-1 design doc).
KEYWORDS = [
    "QA",
    "QA Automation",
    "DevOps",
    "iOS",
    "Android",
    "Data Analyst",
    "Data Engineer",
    "Data Science",
    "JavaScript",
    ".NET",
    "Java",
    "Node.js",
    "PHP",
    "Python",
    "C++",
    "Ruby",
    "Golang",
    "SQL",
]


def build_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Filter, dedup, and shape raw dataset rows into the Job Metadata Artifact schema."""
    filtered = filter_jobs(df, KEYWORDS)
    deduped = deduplicate_jobs(filtered)

    job_text = [
        clean_text(build_job_text(position, description))
        for position, description in zip(deduped["Position"], deduped["Long Description"])
    ]

    return pd.DataFrame(
        {
            "id": deduped["id"].to_numpy(),
            "position": deduped["Position"].to_numpy(),
            "company": deduped["Company Name"].to_numpy(),
            "exp_years": deduped["Exp Years"].to_numpy(),
            "keyword": deduped["Primary Keyword"].to_numpy(),
            "job_text": job_text,
        }
    )


def run(
    df: pd.DataFrame,
    index_path: Path,
    metadata_path: Path,
    settings: Settings,
) -> tuple[faiss.Index, pd.DataFrame]:
    """Run filter -> dedup -> embed -> index -> save -> upload on `df`. Writes to R2."""
    print(f"[build_index] filtering + deduping {len(df)} raw rows...")
    metadata = build_metadata(df)
    print(f"[build_index] {len(metadata)} rows after filter/dedup")

    print("[build_index] loading embedding model...")
    model = load_model()

    print(f"[build_index] embedding {len(metadata)} job texts...")
    embeddings = embed_texts(metadata["job_text"].tolist(), model)
    print(f"[build_index] embedded, shape={embeddings.shape}")

    print("[build_index] building FAISS index...")
    index = build_faiss_index(embeddings)

    print(f"[build_index] saving artifacts to {index_path} and {metadata_path}...")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    save_artifacts(index, metadata, index_path, metadata_path)

    print(f"[build_index] uploading artifacts to R2 bucket {settings.r2_bucket_name}...")
    client = build_r2_client(settings)
    upload_artifacts(client, settings, index_path, metadata_path)
    print("[build_index] upload complete")

    return index, metadata


def main() -> None:
    settings = Settings()  # type: ignore[call-arg]
    print("[build_index] loading raw dataset...")
    df = load_raw_jobs()
    print(f"[build_index] loaded {len(df)} raw rows")
    run(df, ARTIFACTS_DIR / "jobs.index", ARTIFACTS_DIR / "jobs_metadata.parquet", settings)
    print("[build_index] done")


if __name__ == "__main__":
    main()
