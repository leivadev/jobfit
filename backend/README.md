# JobFit backend

FastAPI service (`src/backend/`) plus the offline pipeline (`app/`, `scripts/`). See the root `README.md` for architecture and `docs/adr/` for decisions.

## Setup

```bash
uv sync
cp .env.example .env   # fill in R2 credentials and HF_TOKEN
uv run pytest
uv run ruff check .
uv run uvicorn backend.api.app:app --reload
```

## Environment variables

| Variable | Purpose |
| --- | --- |
| `R2_BUCKET_NAME`, `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_ENDPOINT_URL` | Cloudflare R2 access for Artifacts (required) |
| `HF_TOKEN` | Hugging Face token for dataset and model downloads |
| `CORS_ALLOWED_ORIGINS` | Comma-separated allowed origins; defaults to the local Vite dev server |
| `RATE_LIMIT_KEY_STRATEGY` | `remote_address` (default) or `forwarded_for`. Use `forwarded_for` behind a proxy (e.g. Azure Container Apps), otherwise all visitors share one rate-limit bucket |

## Offline pipeline

```bash
uv run python scripts/build_index.py          # dataset -> bge-small embeddings -> FAISS -> R2
uv run python scripts/evaluate_scale.py       # weak-label evaluation, baseline vs rerank
uv run python scripts/measure_latency.py      # CPU latency, baseline vs rerank
```

`build_index.py` overwrites the Artifacts in R2 at fixed keys (ADR-0006). Run it before deploying any change to the Bi-encoder or keyword vocabulary, since query and index vectors must come from the same model.
