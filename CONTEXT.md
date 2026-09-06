# JobFit

A recommendation engine that matches an uploaded CV against a corpus of IT job postings using dense vector embeddings, FAISS similarity search, and cross-encoder reranking.

## Language

**Job**:
An IT job posting from the source dataset, described by a title, company, and description text.
_Avoid_: Vacante, Posting, Listing.

**CV**:
The raw PDF, DOCX, or plain-text document a user uploads. Never persisted to disk or database.
_Avoid_: Resume, Document.

**Candidate Profile**:
The structured text extracted from a CV, used to generate an embedding for matching. Distinct from the CV itself: the CV is the uploaded file, the Candidate Profile is what's derived from it.
_Avoid_: Profile, Candidate.

**Job Index**:
The FAISS vector index over Job embeddings, used for nearest-neighbor similarity search.
_Avoid_: Vector store, Database.

**Job Metadata**:
The non-vector fields of a Job (title, company, experience years, keyword) stored alongside the Job Index, retrieved for display once a match is found.
_Avoid_: Metadata store.

**Artifact**:
An output of the Offline Pipeline (the Job Index or Job Metadata file), stored in R2 at a fixed key and loaded into memory by the backend at startup. Each pipeline run overwrites the previous Artifact in place — see ADR-0006 for why this isn't versioned.
_Avoid_: File, Build output, Versioned artifact.

**Offline Pipeline**:
The batch process that filters the source dataset, generates embeddings, and produces Artifacts. Runs independently of user requests, never in the request path.
_Avoid_: Build script, ETL.

**Bi-encoder**:
The embedding model that encodes a Job or Candidate Profile independently into a vector. Enables fast similarity search over the whole Job Index.
_Avoid_: Embedding model (too generic on its own).

**Cross-encoder**:
The reranking model that scores a (Candidate Profile, Job) pair jointly. More accurate than the Bi-encoder but too slow to run over the full Job Index, so it only scores the Bi-encoder's shortlist.
_Avoid_: Reranker, Rerank model.

**Similarity Score**:
The Bi-encoder's cosine similarity between a Candidate Profile embedding and a Job embedding. Used internally to select the shortlist of candidates for the Cross-encoder; not necessarily shown to the user as-is.
_Avoid_: Score (alone), Relevance.

**Rerank Score**:
The Cross-encoder's score for a (Candidate Profile, Job) pair. Determines the final ranking of Recommendations shown to the user when neither Candidate Signal was declared; otherwise it's the tiebreaker within a Keyword Match/Exp Distance tier, never blended with those signals arithmetically.
_Avoid_: Score (alone), Final score.

**Recommendation**:
A single (Job, Rerank Score) pair returned to the user — one entry in the final ranked list.
_Avoid_: Result, Match.

**Candidate Signals**:
The candidate's own declared experience/skills (`exp_years`, `keywords`), submitted as optional `/recommend` form fields. Distinct from Job Metadata's `exp_years`/`keyword`: Candidate Signals describe the candidate, Job Metadata describes the Job. Either signal may be absent.
_Avoid_: Preferences, Filters.

**Keyword Match**:
Whether any of the candidate's declared Candidate Signals keywords overlaps a Job's `keyword`. Not a `Recommendation` field named after ranking (see `Recommendation`'s avoided "Match") — it's an annotation on a Recommendation, not the Recommendation itself.
_Avoid_: Skill match.

**Exp Distance**:
The symmetric ordinal distance between a candidate's declared `exp_years` bucket and a Job's, on the shared 5-bucket scale. `None` when either side's bucket is absent or unrecognized — never a stand-in for 0.
_Avoid_: Experience gap, Seniority gap.
