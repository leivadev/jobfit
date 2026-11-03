"""Large-N evaluation using an automated weak relevance label instead of hand-labeled pairs.

Complements `evaluate.py`'s 25 hand-labeled gold pairs: that eval answers "does the model
find *the* right job for this candidate" (precise but N=25, not representative of 210k
candidates). This script answers a coarser question at real scale, with a label that costs
nothing to compute: "relevant jobs for a candidate" = all jobs sharing the candidate's
`Primary Keyword` (same 18-value vocabulary already used for gold-set sampling, see
`docs/design/phase-8-evaluation.md`). No candidate/job ids are hardcoded — the sample is
drawn fresh each run via a fixed seed, from whatever the datasets currently contain.

Because a keyword category can hold thousands of jobs, Precision@10 is the metric that
matters here: does reranking increase the fraction of the top 10 that's topically
on-target.

Manual, one-time invocation (`uv run python scripts/evaluate_scale.py [--sample-frac 0.01]
[--limit N]`), same offline-script pattern as `build_index.py` / `evaluate.py`.
"""

import argparse
import sys
import time
from pathlib import Path

import faiss
import pandas as pd
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval_metrics import METRICS, mean_metrics

from backend.domain.embeddings import BiEncoder
from backend.domain.matching import KEYWORD_VOCABULARY
from backend.domain.rerank import CrossEncoder, ShortlistItem, rerank
from backend.domain.search import SHORTLIST_SIZE, JobSearchIndex

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
CANDIDATE_DATASET_NAME = "lang-uk/recruitment-dataset-candidate-profiles-english"
SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-frac", type=float, default=0.01)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap sample size, for smoke-testing timing",
    )
    args = parser.parse_args()

    print("[evaluate_scale] loading local Job Index + Metadata artifacts...")
    search_index = JobSearchIndex(faiss.read_index(str(ARTIFACTS_DIR / "jobs.index")))
    metadata = pd.read_parquet(ARTIFACTS_DIR / "jobs_metadata.parquet")
    job_ids = metadata["id"].to_numpy()
    jobs_by_keyword = metadata.groupby("keyword")["id"].apply(set).to_dict()

    print("[evaluate_scale] loading candidate profiles dataset...")
    candidates = load_dataset(CANDIDATE_DATASET_NAME, split="train").to_pandas()
    qualifying = candidates[candidates["Primary Keyword"].isin(KEYWORD_VOCABULARY)]
    sample = qualifying.sample(frac=args.sample_frac, random_state=SEED)
    if args.limit is not None:
        sample = sample.head(args.limit)
    print(
        f"[evaluate_scale] sampled {len(sample)} of {len(qualifying)} qualifying candidates "
        f"({len(candidates)} total in dataset)"
    )

    print("[evaluate_scale] loading Bi-encoder + Cross-encoder models...")
    bi_encoder = BiEncoder()
    cross_encoder = CrossEncoder()

    baseline_rankings: list[tuple[list[str], set[str]]] = []
    rerank_rankings: list[tuple[list[str], set[str]]] = []

    start = time.monotonic()
    for i, (_, row) in enumerate(sample.iterrows(), start=1):
        candidate_profile = row["CV"]
        relevant_ids = jobs_by_keyword.get(row["Primary Keyword"], set())

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

        if i % 25 == 0 or i == len(sample):
            elapsed = time.monotonic() - start
            rate = elapsed / i
            print(
                f"[evaluate_scale] {i}/{len(sample)} candidates "
                f"({rate:.2f}s/candidate, ~{rate * len(sample) / 60:.1f}min total est.)"
            )

    baseline_metrics = mean_metrics(baseline_rankings)
    rerank_metrics = mean_metrics(rerank_rankings)

    print(
        f"\n[evaluate_scale] N={len(sample)} candidates, weak label = same Primary Keyword"
    )
    print(f"{'metric':<15}{'baseline':>12}{'rerank':>12}{'delta':>12}")
    for label, _, _ in METRICS:
        baseline_value = baseline_metrics[label]
        rerank_value = rerank_metrics[label]
        delta = rerank_value - baseline_value
        print(f"{label:<15}{baseline_value:>12.3f}{rerank_value:>12.3f}{delta:>+12.3f}")


if __name__ == "__main__":
    main()
