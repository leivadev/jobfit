# Evaluation Methodology

See `CONTEXT.md` for term definitions (Job, Candidate Profile, Bi-encoder, Cross-encoder, Recommendation).

Offline only, using `lang-uk/recruitment-dataset-candidate-profiles-english` (~230k anonymized CVs from the same recruitment domain as the Job corpus). This dataset is never exposed in production, only used to score the system before/after changes.

## Method

1. Take a sample of real Candidate Profiles from the dataset.
2. Establish ground truth: if the dataset lets us infer which Job(s) each candidate actually applied to, use that as approximate ground truth. Otherwise, build a manual ground truth of 20-30 hand-labeled Candidate Profile ↔ Job pairs.
3. Compute metrics: Precision@10, Recall@10, MRR.
4. Compare baseline (Bi-encoder only) vs. Bi-encoder + Cross-encoder rerank, to produce a measurable improvement number, not just a qualitative "it looks better."

## Reporting

Results get documented in the repo README once Phase 8 (quantitative evaluation) is implemented — this is the evidence that the Cross-encoder rerank step is worth its added latency, not just an assumption.
