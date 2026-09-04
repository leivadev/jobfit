# API Contract

See `CONTEXT.md` for term definitions (Job, CV, Candidate Profile, Similarity Score, Rerank Score, Recommendation).

## `POST /recommend`

- **Input**: a CV file (multipart, PDF/DOCX) or plain text.
- **Process**: extract text from the CV → build a Candidate Profile → embed it with the Bi-encoder → retrieve the top-100 Jobs from the Job Index by Similarity Score → rerank with the Cross-encoder → return the top-10 as Recommendations.
- **Output**: a ranked list of Recommendations.

### Response contract

```json
{
  "results": [
    {
      "position": "Backend Developer (Python)",
      "company": "Acme Corp",
      "rerank_score": 0.83,
      "exp_years": "2-3",
      "keyword": "Python",
      "snippet": "..."
    }
  ]
}
```

The field is `rerank_score`, not `score`: only the Cross-encoder's score is exposed to the client. The Bi-encoder's Similarity Score is an internal detail of candidate selection and never appears in the response — see `CONTEXT.md` for why these are modeled as two distinct concepts.

## `GET /health`

Liveness check, no request body. Used by the deployment platform (Railway/Fly.io), not by the frontend.

## Uploaded CV handling (privacy)

Design requirement, not a detail — any stranger can upload a real CV to a public demo:

- The CV is processed **in memory only**, never written to disk or persisted to a database.
- CV content is never logged, only aggregate metrics (file size, processing time, whether an error occurred).
- The frontend shows a visible notice: "your CV is processed in memory and not stored."
- Basic rate limiting (e.g. `slowapi`) on the public endpoint.
- File size limit (e.g. 5 MB) and MIME type validation before parsing.
