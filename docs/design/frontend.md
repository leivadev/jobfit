# Frontend Design

Deliberately simple: the learning focus of this project is the ML engine, not the frontend.

- React + Vite + Tailwind.
- Single screen: CV dropzone, "Recommend jobs" button, list of Recommendations with a visual score (bar or percentage) and snippet.
- Staged loading state: the backend scales to zero, so the frontend first polls `GET /health` (up to 90 s) to wake it, then sends `/recommend`; stages are `checking`, `waking`, `processing`, `slow`. A rate limit (429) and a backend that never becomes healthy get their own messages.
- Privacy notice shown before upload — see `docs/design/api-contract.md#uploaded-cv-handling-privacy`.
