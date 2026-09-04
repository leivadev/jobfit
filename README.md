# JobFit

> Dense Vector Representation-based IT Job Recommendation Engine

A visitor uploads their resume and receives the most relevant IT job postings, using **semantic embeddings** over a real job listings dataset. Portfolio project focused on embeddings, vector search, and reranking, combined with software engineering (backend, frontend, deployment, user data privacy).

**Status**: in design and scaffolding phase. There is no working pipeline or API yet — this README documents the agreed-upon architecture, not already-implemented functionality.

## Architecture

```mermaid
flowchart TD
    subgraph Offline[" "]
        direction LR
        HF[("HF Dataset")] --> PIPE["scripts/build_index.py\n(offline batch pipeline)"]
    end

    subgraph R2["Cloudflare R2"]
        direction LR
        IDX[("jobs.index\nFAISS")]
        META[("jobs_metadata\n.parquet")]
    end

    subgraph Backend["Backend · Railway / Fly.io · jobfit-api.leivadev.com"]
        direction LR
        EX["Text\nextraction"] --> EMB["Resume\nembedding"] --> SEARCH["FAISS\nsearch"] --> RERANK["Cross-encoder\nrerank"]
    end

    FE["Frontend SPA\nReact + Vite\nCloudflare Workers\njobfit-app.leivadev.com"]

    PIPE -- "upload (wrangler r2 / boto3)" --> R2
    Backend -- "boto3 S3 API, on startup" --> R2
    FE -- "POST /recommend (multipart resume)" --> Backend
    Backend -- "JSON response" --> FE
```

The offline pipeline and the online service are decoupled: the pipeline runs once (or whenever the dataset is updated) and uploads its artifacts to a fixed key in R2, overwriting the previous run (see ADR-0006); the backend only downloads them at startup and serves them from memory.

**There is no database.** The job corpus is queried via vector similarity, not relational queries, and fits entirely in memory. The user's resume is processed in memory and never persisted (see [Privacy](#privacy)).

## Stack

| Component | Choice | Reason |
| --- | --- | --- |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Fast, lightweight, good baseline |
| Reranking | Cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2` or similar) | Improves precision over the top-100 |
| Vector search | FAISS (`IndexFlatIP`, in-memory) | Sufficient for 10-20k vectors |
| Backend | FastAPI | Async, typed, automatic OpenAPI |
| Resume extraction | `pypdf`/`pdfplumber`, `python-docx` | PDF and DOCX coverage |
| Dependency manager (backend) | `uv` | Fast, lockfile, single binary |
| Frontend | React + Vite + Tailwind | Standard, quick to build for a single screen |
| Dependency manager (frontend) | `pnpm` | Efficient, integrates well with the Wrangler ecosystem |
| Backend deployment | Railway or Fly.io | Supports heavy ML dependencies (torch, faiss) |
| Frontend deployment | Cloudflare Workers (Static Assets) | Free tier, same ecosystem as R2 |
| Artifact storage | Cloudflare R2 | Zero egress fees, S3-compatible |

## Data

- **Job postings**: [`lang-uk/recruitment-dataset-job-descriptions-english`](https://huggingface.co/datasets/lang-uk/recruitment-dataset-job-descriptions-english) (~142k IT postings, Djinni platform, 2020-2023, MIT). Filtered to the most represented `Primary Keyword` categories (QA, DevOps, iOS/Android, Data, main languages), deduplicated — see `docs/design/phase-1-offline-pipeline.md`.
- **Offline evaluation**: [`lang-uk/recruitment-dataset-candidate-profiles-english`](https://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-english) (~230k anonymized resumes), never exposed in production.

## Repo structure

```
jobfit/
├── backend/            # FastAPI + offline pipeline (uv)
│   ├── app/             # API, extraction, embeddings, search, rerank
│   ├── scripts/          # build_index.py (offline batch pipeline)
│   └── tests/
├── frontend/            # React + Vite, deployed on Cloudflare Workers
├── docs/
│   ├── adr/              # Architecture Decision Records
│   ├── design/           # API contract, scope, evaluation, frontend
│   └── research/         # Open research (not yet decisions)
└── CONTEXT.md           # Shared domain vocabulary
```

## Local development

### Backend

```bash
cd backend
uv sync
uv run pytest
uv run ruff check .
```

### Frontend

```bash
cd frontend
pnpm install
pnpm dev
```

## Privacy

Anyone can upload their real resume to a public demo. To ensure privacy:

- The resume is processed **in memory**, never written to disk or persisted.
- Resume content is not logged, only aggregate metrics (size, processing time, errors).
- Rate limiting on the public endpoint, file size limit, and MIME type validation.

## API

See [`docs/design/api-contract.md`](docs/design/api-contract.md) for the full `/recommend` and `/health` contract.

## Evaluation

Offline metrics (Precision@10, Recall@10, MRR) on the real resume dataset, comparing bi-encoder only vs. bi-encoder + cross-encoder rerank. Full methodology in [`docs/design/evaluation.md`](docs/design/evaluation.md); results documented here once Phase 8 of the plan is implemented.

## Project status

See `docs/adr/` for architecture decisions already made and their rationale. The full plan (phases, scope, open decisions) is managed outside this repo as a design document; this README is updated as each phase is implemented.

## License

[MIT](LICENSE).
