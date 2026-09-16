# API Contract

See `CONTEXT.md` for term definitions (Job, CV, Candidate Profile, Similarity Score, Rerank Score, Recommendation).

## `POST /recommend`

- **Input**: a CV file (multipart, PDF/DOCX) or plain text, plus two **optional** multipart form fields:
  - `exp_years`: single value, one of the 5 buckets on the ordered scale `no_exp`, `1y`, `2y`, `3y`, `5y` (same vocabulary as Job Metadata's `exp_years`).
  - `keywords`: zero or more values, each one of the 18-value controlled vocabulary used by Job Metadata's `keyword` (`.NET`, `Android`, `C++`, `Data Analyst`, `Data Engineer`, `Data Science`, `DevOps`, `Golang`, `Java`, `JavaScript`, `Node.js`, `PHP`, `Python`, `QA`, `QA Automation`, `Ruby`, `SQL`, `iOS`).
  - Both fields are independently optional; a request with neither behaves exactly as if they didn't exist. A value outside either controlled vocabulary is rejected with 400.
- **Process**: extract text from the CV → build a Candidate Profile → embed it with the Bi-encoder → retrieve the top-20 Jobs from the Job Index by Similarity Score → rerank with the Cross-encoder → return the top-10 as Recommendations. `exp_years`/`keywords` do not affect ranking (see the "Recommendation" ticket for that) — they only annotate the response.
- **Output**: a ranked list of Recommendations.

### Response contract

```json
{
  "results": [
    {
      "position": "Backend Developer (Python)",
      "company": "Acme Corp",
      "rerank_score": 0.83,
      "exp_years": "2y",
      "keyword": "Python",
      "snippet": "...",
      "keyword_match": true,
      "exp_distance": 1
    }
  ]
}
```

The field is `rerank_score`, not `score`: only the Cross-encoder's score is exposed to the client. The Bi-encoder's Similarity Score is an internal detail of candidate selection and never appears in the response — see `CONTEXT.md` for why these are modeled as two distinct concepts.

- `keyword_match: bool` — whether any of the candidate's declared `keywords` overlaps the job's `keyword`. `false` when the candidate didn't declare any `keywords` (absent signal, not a negative match).
- `exp_distance: int | None` — symmetric ordinal distance between the candidate's declared `exp_years` bucket and the job's, on the `no_exp`/`1y`/`2y`/`3y`/`5y` scale (0 = exact match, up to 4 = maximally apart). `None` when the candidate didn't declare `exp_years`, or when the job's `exp_years` isn't one of the five buckets (Job Metadata's `exp_years` isn't validated against the scale at ingestion, unlike `keyword`).

## `GET /health`

Liveness check, no request body. Used by the deployment platform (Railway/Fly.io), not by the frontend.

## Uploaded CV handling (privacy)

Design requirement, not a detail — any stranger can upload a real CV to a public demo:

- The CV is processed **in memory only**, never written to disk or persisted to a database.
- CV content is never logged, only aggregate metrics (file size, processing time, whether an error occurred).
- The frontend shows a visible notice: "your CV is processed in memory and not stored."
- Basic rate limiting (e.g. `slowapi`) on the public endpoint.
- File size limit (e.g. 5 MB) and MIME type validation before parsing.
