# Phase 1: Offline Pipeline

See `CONTEXT.md` for term definitions (Job, Job Index, Job Metadata, Artifact, Offline Pipeline, Bi-encoder). See `docs/adr/0002-faiss-in-memory-index.md`, `docs/adr/0003-r2-via-s3-api-not-binding.md`, `docs/adr/0006-fixed-key-r2-artifacts.md`, and `docs/adr/0007-chunk-mean-pool-embeddings.md` for the decisions this design builds on.

Reviewed with the `grilling` skill before ticketing: split into 5 independently-testable sub-units (1a-1e) under this parent, because this is the single most blocking phase in the whole project and each unit deserves its own red-green-refactor cycle.

## Design principle

Separate **pure, testable** functions (filtering, cleaning, text building, chunking) from functions with **I/O or external dependencies** (dataset download, model encoding, artifact upload). Pure functions get real unit tests; I/O functions are covered by one integration smoke test, not strict unit tests.

## Code layout

Pipeline code lives under `backend/`, separate from the installable `backend/src/backend/` package that will hold the FastAPI service:

```
backend/
├── app/            # dataset.py, text.py, embeddings.py, index.py, storage.py
├── scripts/        # build_index.py
├── src/backend/    # installable package: config.py (Settings), future FastAPI app
└── tests/
```

`backend/app/` and `backend/scripts/` import shared config from the installable package (`from backend.config import Settings`), since pipeline code already runs via `uv run` inside `backend/` with the package importable. This is the only cross-import: the pipeline depends on the backend package's config, not the other way around.

### `Settings` (`backend/src/backend/config.py`)

A `pydantic-settings BaseSettings` shared between the pipeline's R2 upload (1e) and the backend's future R2 read-at-startup path. Scoped to R2 fields only for now — add fields when the code that needs them exists, not speculatively:

- `r2_bucket_name`
- `r2_account_id`
- `r2_access_key_id`
- `r2_secret_access_key`
- `r2_endpoint_url`

## Source dataset

`lang-uk/recruitment-dataset-job-descriptions-english` on Hugging Face, 141,897 rows. Verified via HF datasets-server API (not assumed):

| Column | Type | Notes |
| --- | --- | --- |
| `id` | string (UUID) | Stable identifier — use this, not a positional index |
| `Position` | string | |
| `Long Description` | string | Contains literal `\r\n` and markdown-like formatting (e.g. `*Requirements*`). No nulls. Length: min 51, median 1,629, mean 1,801, max 12,578 chars (verified via HF `/statistics`) |
| `Company Name` | string | |
| `Exp Years` | string | |
| `Primary Keyword` | string | 45 distinct values, no nulls. **No unified "Backend"/"Frontend"/"Mobile" category exists** — see 1a below |
| `English Level` | string | Unused |
| `Published` | string | Unused |
| `Long Description_lang` | string | Unused |

No `url` column exists — an earlier design draft assumed one; corrected after checking the real schema.

## 1a — Dataset filtering (`app/dataset.py`)

```python
def load_raw_jobs() -> pd.DataFrame:
    """Download the dataset via `datasets`."""

def filter_jobs(df: pd.DataFrame, keywords: list[str], min_desc_chars: int = 200) -> pd.DataFrame:
    """Filter by Primary Keyword in keywords, drop empty/too-short descriptions. Pure."""

def deduplicate_jobs(df: pd.DataFrame) -> pd.DataFrame:
    """Dedup by (Company Name, Position), case-insensitive/whitespace-normalized comparison,
    keeping the first-seen row with its original casing. Pure."""
```

### Keyword taxonomy

An earlier draft assumed `Primary Keyword` values of "backend, frontend, data, QA, devops, mobile." Verified against the real dataset (HF `/statistics`, all 141,897 rows): those two-category names don't exist. The real values are per-language and per-discipline labels. `keywords` for 1a:

```python
["QA", "QA Automation", "DevOps", "iOS", "Android",
 "Data Analyst", "Data Engineer", "Data Science",
 "JavaScript", ".NET", "Java", "Node.js", "PHP", "Python", "C++", "Ruby", "Golang", "SQL"]
```

Excludes long-tail values (Scala=492, Rust=151) and single-row mislabeled outliers (e.g. "React", "SAP").

**Row-count target**: no fixed number. Pre-dedup, QA+DevOps+Mobile+Data alone already sums to 31,329 rows, and dedup rate on `(Company Name, Position)` can't be computed via the API without a full download. Measure the actual post-filter, post-dedup count once 1a is implemented and record it here — don't engineer the filter to hit a pre-picked range.

**Measured** (full download via `load_raw_jobs`, `min_desc_chars=200`): 141,897 raw rows → 92,416 post-filter → 80,224 post-filter-and-dedup.

**Tests**: `filter_jobs` and `deduplicate_jobs` against a small fixture DataFrame (10-15 hand-built rows) using **real** `Primary Keyword` values from the list above (not placeholder strings) — a typo'd keyword string should be catchable by the fixture, not just by production silently filtering out everything.

## 1b — Text cleaning (`app/text.py`)

```python
def clean_text(text: str) -> str:
    """Normalize CRLF, collapse multiple spaces, strip asterisk emphasis (*word*, **word**)
    and line-start bullet markers (-, *). Pure."""

def build_job_text(position: str, description: str) -> str:
    """job_text = f"{position}\n{description}". Pure."""

def chunk_text(text: str, tokenizer, max_tokens: int = 256) -> list[str]:
    """Split text into non-overlapping windows of at most max_tokens tokens, using the
    real model tokenizer (not a char/word-count heuristic). Pure given the tokenizer —
    a deliberate exception to the pure/I-O split, since an approximate heuristic risks
    mis-chunking near the boundary. See ADR-0007."""
```

**Tests**:
- `clean_text` must include a case with literal `\r\n` and asterisk-markdown (`*Requirements*`) taken from a real dataset row — not a synthetic clean string, since that's what actually breaks naive implementations here. No link/header stripping: not observed in this dataset, don't build for markdown syntax that doesn't appear here.
- `chunk_text` tested with the real `all-MiniLM-L6-v2` tokenizer (fast, tokenization only, no encoding) — verify chunk count and per-chunk token count against known long/short inputs, including a description under 256 tokens (single-chunk case) and one well over it (multi-chunk case, using the real max-length row noted above).

## 1c — Embedding generation (`app/embeddings.py`)

```python
def embed_texts(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """For each text: chunk via chunk_text(text, model.tokenizer), encode each chunk
    (batch_size=128, device auto-detect), mean-pool the raw (unnormalized) chunk vectors
    into one (384,) vector per text. Returns (N, 384) float32. Normalization happens
    later, in build_faiss_index — do not normalize per-chunk or per-text here.
    See ADR-0007."""
```

Model loaded with a pinned HF revision hash (not just the `all-MiniLM-L6-v2` tag) — an unpinned tag could be silently updated upstream between pipeline runs, desyncing old FAISS vectors from new query vectors.

**Constraint on future work**: query-time Candidate Profile embedding (API-contract phase, not built here) must reuse this same chunk+mean-pool path, not truncate — see ADR-0007.

**Tests**: none strict (network/model dependency) — covered only by the integration smoke test below.

## 1d — FAISS index build (`app/index.py`)

```python
def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """faiss.normalize_L2 + IndexFlatIP. Deterministic given the input."""

def save_artifacts(index: faiss.Index, metadata: pd.DataFrame, index_path: Path, metadata_path: Path) -> None:
```

`scripts/build_index.py` writes to `backend/artifacts/` (gitignored) before upload — keeps build output inspectable locally during development.

### `jobs_metadata.parquet` schema

| Column | Type | Source |
| --- | --- | --- |
| `id` | str (UUID) | Dataset's `id` — stable key, not positional |
| `position` | str | `Position` |
| `company` | str | `Company Name` |
| `exp_years` | str | `Exp Years` |
| `keyword` | str | `Primary Keyword` |
| `job_text` | str | `clean_text(build_job_text(Position, Long Description))` |

**Invariant**: row `i` of the FAISS index must correspond to row `i` of the parquet. This needs an explicit test — it's the easiest thing to silently break with a reindex bug.

**Tests**: `build_faiss_index` with synthetic vectors — verify `index.ntotal` matches input count, and that searching a vector identical to an indexed one returns similarity ~1.0 (cosine, post-normalization).

## 1e — R2 upload (`app/storage.py`, called from `scripts/build_index.py`)

Upload `jobs.index` and `jobs_metadata.parquet` to fixed keys in the R2 bucket via `boto3` against R2's S3-compatible endpoint (per ADR-0003), using `Settings` for bucket name and credentials. Each run **overwrites the previous artifacts in place** — not versioned; see ADR-0006 for why, and for the accepted risk of a mismatched pair if one upload succeeds and the other fails mid-run.

**Tests**: none strict (network dependency) — covered only by the integration smoke test.

## Integration smoke test

Run `scripts/build_index.py` end-to-end on a tiny subset (~20 rows) and verify both artifacts get produced with matching row counts (`index.ntotal == len(metadata_df)`). This is the only place `load_raw_jobs`, `embed_texts`, and the R2 upload get exercised at all.

Marked `@pytest.mark.integration`, excluded from the default `backend-ci.yml` run (`pytest -m "not integration"`) — it needs network access and R2 credentials that CI doesn't have configured. Run manually/locally.

## Orchestration

`scripts/build_index.py` is a one-time manual invocation (`uv run python scripts/build_index.py`), rerun by hand if the source dataset changes. No CI trigger or cron — nothing in scope requires automatic dataset refresh.

## Definition of Done

- [x] 1a: `filter_jobs`, `deduplicate_jobs` unit-tested against a fixture DataFrame using real `Primary Keyword` values; actual post-filter/post-dedup row count measured and recorded above
- [ ] 1b: `clean_text`, `build_job_text`, `chunk_text` unit-tested, including the real CRLF/markdown case and real tokenizer-based chunking
- [ ] 1c: `embed_texts` implemented with chunk+mean-pool and a pinned model revision (no strict unit test)
- [ ] 1d: `build_faiss_index`, `save_artifacts` unit-tested with synthetic vectors, including the index↔parquet row alignment invariant
- [ ] 1e: R2 upload implemented via shared `Settings` (no strict unit test)
- [ ] Integration smoke test passes locally on a ~20-row subset, excluded from default CI
