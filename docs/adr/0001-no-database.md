# No database

The Job corpus (10-20k rows) is queried by vector similarity, not relational queries, and fits entirely in memory (~30 MB). The Job Index and Job Metadata are versioned build artifacts produced by the Offline Pipeline, not live application state. The uploaded CV is never persisted (privacy requirement). We decided against a relational or NoSQL database for the MVP: there's no data that needs one. If optional login/saved-search history ships later, that's the trigger to introduce a real database (e.g. Postgres) — not before.
