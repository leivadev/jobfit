# Frontend Design

Deliberately simple: the learning focus of this project is the ML engine, not the frontend.

- React + Vite + Tailwind.
- Single screen: CV dropzone, "Recommend jobs" button, list of Recommendations with a visual score (bar or percentage) and snippet.
- Loading state while the backend processes the request (can take a few seconds due to the Cross-encoder rerank).
- Privacy notice shown before upload — see `docs/design/api-contract.md#uploaded-cv-handling-privacy`.
