# JobFit frontend

React + Vite + Tailwind single-page app: upload a CV, get ranked job Recommendations. Deployed on Cloudflare Workers (Static Assets). See the root `README.md` and `docs/design/frontend.md`.

## Setup

```bash
pnpm install
cp .env.example .env   # VITE_API_BASE_URL, defaults to http://localhost:8000
pnpm dev
```

## Scripts

| Command | Purpose |
| --- | --- |
| `pnpm dev` | Vite dev server |
| `pnpm build` | Type-check and build |
| `pnpm lint` | ESLint |
| `pnpm test` | Unit tests (vitest) |
| `pnpm test:e2e` | End-to-end tests (playwright) |
| `pnpm deploy` | Build and `wrangler deploy` (requires Cloudflare auth) |

The file-size and MIME checks in `src/lib/validateFile.ts` mirror the backend limits in `backend/src/backend/api/upload_limits.py`; keep them in sync.
