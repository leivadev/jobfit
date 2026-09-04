# FAISS in-memory index, not a dedicated vector database

At 10-20k Jobs, a `faiss.IndexFlatIP` loaded into process memory does exact nearest-neighbor search fast enough with zero extra infrastructure. We considered a dedicated vector database (Pinecone, Qdrant, pgvector) but rejected it: it would add a network hop, an operational dependency, and cost, for a search space small enough that brute-force in-memory search is already fast. Revisit only if the Job corpus grows by an order of magnitude or multiple backend instances need a shared index.
