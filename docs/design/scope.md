# MVP Scope

## In scope

- Anonymous CV upload (PDF, DOCX, or plain text).
- Text extraction, embedding, Job Index search for the top-N most similar Jobs.
- Cross-encoder rerank of the Bi-encoder's top-100 to improve precision.
- Simple frontend: upload CV, view ranked Recommendations with score and snippet.
- Public deployment, no login required.
- Offline quantitative evaluation against the real-CV dataset (Precision@k, Recall@k, MRR) — see `docs/design/evaluation.md`.
- Job corpus filtered to the most-represented `Primary Keyword` values in the source dataset (QA, QA Automation, DevOps, iOS, Android, Data Analyst, Data Engineer, Data Science, and major language keywords like JavaScript, Python, Java, .NET) — see `docs/design/phase-1-offline-pipeline.md`. The dataset has no unified "backend"/"frontend"/"mobile" category; an earlier draft assumed one.

## Out of scope (future phase, not MVP)

- User authentication / saved search history.
- Bidirectional matching (a company searching for candidates).
- A custom fine-tuned model via contrastive learning (ConFit-style) — documented as a possible v2, not built now. See `docs/research/embedding-models.md`.
- Hybrid score combining semantic similarity with NER-extracted skills — phase 2 improvement, doesn't block the MVP.
