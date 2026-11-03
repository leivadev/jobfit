"""Compares Bi-encoder-only vs. Bi-encoder + Cross-encoder rerank against a hand-labeled
ground truth (see docs/design/phase-8-evaluation.md).

Manual, one-time invocation (`uv run python scripts/evaluate.py`); rerun by hand whenever
the embedding/rerank models or the Job Index change. No cron/CI trigger, same pattern as
`build_index.py`.
"""

import datetime
import json
import sys
from pathlib import Path

import faiss
import pandas as pd
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval_metrics import METRICS, mean_metrics

from backend.domain.embeddings import BiEncoder
from backend.domain.rerank import CrossEncoder, ShortlistItem, rerank
from backend.domain.search import SHORTLIST_SIZE, JobSearchIndex

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
GROUND_TRUTH_PATH = Path(__file__).parent / "eval_ground_truth.json"
CANDIDATE_DATASET_NAME = "lang-uk/recruitment-dataset-candidate-profiles-english"


def load_ground_truth(path: Path) -> dict[str, list[str]]:
    with open(path) as f:
        return json.load(f)


def load_candidate_profiles(candidate_ids: set[str]) -> dict[str, str]:
    """Fetch raw CV text for `candidate_ids` from the HF dataset, by its own `id` column.

    Never persisted locally beyond this run (see phase-8 design: no CV text committed to
    the repo, only ids in `eval_ground_truth.json`).
    """
    df = load_dataset(CANDIDATE_DATASET_NAME, split="train").to_pandas()
    subset = df[df["id"].isin(candidate_ids)]
    return dict(zip(subset["id"], subset["CV"]))


def main() -> None:
    ground_truth = load_ground_truth(GROUND_TRUTH_PATH)
    print(f"[evaluate] {len(ground_truth)} ground truth pairs loaded")

    print("[evaluate] loading local Job Index + Metadata artifacts...")
    index_path = ARTIFACTS_DIR / "jobs.index"
    metadata_path = ARTIFACTS_DIR / "jobs_metadata.parquet"
    build_date = datetime.datetime.fromtimestamp(
        index_path.stat().st_mtime, tz=datetime.UTC
    ).date()
    search_index = JobSearchIndex(faiss.read_index(str(index_path)))
    metadata = pd.read_parquet(metadata_path)
    job_ids = metadata["id"].to_numpy()

    print("[evaluate] fetching candidate CV text from HF dataset...")
    candidate_profiles = load_candidate_profiles(set(ground_truth))

    print("[evaluate] loading Bi-encoder + Cross-encoder models...")
    bi_encoder = BiEncoder()
    cross_encoder = CrossEncoder()

    baseline_rankings: list[tuple[list[str], set[str]]] = []
    rerank_rankings: list[tuple[list[str], set[str]]] = []

    for candidate_id, relevant_job_ids in ground_truth.items():
        candidate_profile = candidate_profiles[candidate_id]
        relevant_ids = set(relevant_job_ids)

        query_vector = bi_encoder.encode_query([candidate_profile])[0]

        # Store full SHORTLIST_SIZE-length rankings — metric fns slice to k themselves,
        # so one pass here covers both the @10 and @100 readings.
        baseline_results = search_index.search(query_vector, top_k=SHORTLIST_SIZE)
        baseline_ranked_ids = [job_ids[result.job_row] for result in baseline_results]
        baseline_rankings.append((baseline_ranked_ids, relevant_ids))

        shortlist = [
            ShortlistItem(
                job_row=result.job_row,
                job_text=metadata.iloc[result.job_row]["job_text"],
            )
            for result in baseline_results
        ]
        rerank_results = rerank(
            candidate_profile, shortlist, cross_encoder, top_k=SHORTLIST_SIZE
        )
        rerank_ranked_ids = [job_ids[result.job_row] for result in rerank_results]
        rerank_rankings.append((rerank_ranked_ids, relevant_ids))

    baseline_metrics = mean_metrics(baseline_rankings)
    rerank_metrics = mean_metrics(rerank_rankings)

    print(
        f"\n[evaluate] Job Index artifacts built {build_date} (local, not re-fetched from R2)"
    )
    print(f"{'metric':<15}{'baseline':>12}{'rerank':>12}{'delta':>12}")
    for label, _, _ in METRICS:
        baseline_value = baseline_metrics[label]
        rerank_value = rerank_metrics[label]
        delta = rerank_value - baseline_value
        print(f"{label:<15}{baseline_value:>12.3f}{rerank_value:>12.3f}{delta:>+12.3f}")


if __name__ == "__main__":
    main()
