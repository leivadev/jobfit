# bge-small-en-v1.5 replaces all-MiniLM-L6-v2 as the Bi-encoder

Local comparison (30,000-Job sample, 1,000 CVs, 1 thread CPU for latency; `backend/scripts/compare_models.py`): `bge-small-en-v1.5` (33M params, same 384 dims) beat `all-MiniLM-L6-v2` on weak-label P@10 by +0.041 (paired bootstrap 95% CI [+0.024, +0.057]). `bge-base` (109M, 768 dims) added +0.006 over bge-small (CI [−0.006, +0.018], not distinguishable from 0) at double the index size and 3.4× the query latency, so we stopped at bge-small. A manual review of 52 mostly non-IT CVs (judged by the author, not blind) found 204/260 relevant top-5 Jobs for MiniLM and 208/260 for bge-small; bge-small avoided the total misses MiniLM had (e.g. a motion/graphic designer CV returning sales and support Jobs).

Measured cost on 1 CPU thread: RSS after one inference 803 MB (MiniLM) vs 858 MB (bge-small); CV encode 62 ms vs 120 ms. The index stays 384-dim, so its size is unchanged.

Two consequences in code. Chunk size goes from 256 to 480 tokens (512-token model window minus the query prefix and special tokens). And bge is trained asymmetrically: Candidate Profiles are embedded with the prefix `Represent this sentence for searching relevant passages: ` (`BiEncoder.encode_query`) while Jobs are embedded without it (`BiEncoder.encode`). Using plain `encode` for both would silently degrade results.

Keep the Cross-encoder (`ms-marco-MiniLM-L6-v2`). On 100 CVs it added +0.029 weak-label P@10 (no confidence interval, likely within noise) and 85% vs 80% in the manual review, but it also introduced new misses and costs ~4 s per request on 1 CPU thread. Larger rerankers (`ms-marco-L12`, `bge-reranker-base`, `mxbai-rerank-base`) did not beat it, and `bge-reranker-base` needs ~2 GB RSS, which does not fit the 2 GiB container. Revisit if the latency is unacceptable.

Old index artifacts (MiniLM vectors) are incompatible with this encoder: the index must be rebuilt and re-uploaded together with the new code (ADR-0006 fixed keys).
