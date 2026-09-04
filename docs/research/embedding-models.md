# Research: domain-specific embedding models for CV-Job matching (v2)

Not a decision yet — the MVP ships with `sentence-transformers/all-MiniLM-L6-v2` (generic). This is a survey of alternatives for a possible v2, reviewing recent recruitment-domain literature. Promote the relevant section to an ADR once a v2 model is actually chosen.

## Candidates

- **CareerBERT** (2025): SBERT fine-tuned in three stages (general domain → HR domain → matching task), projecting CVs and ESCO job titles into a shared embedding space. Reported to outperform prior pure-embedding approaches in evaluation with human recruiters. Public checkpoints on Hugging Face (`lwolfrum2/careerbert-g`, `lwolfrum2/careerbert-jg`), code on GitHub.
- **JobBERT / JobBERT-v2**: BERT pretrained on job postings, incorporating co-occurring skills extracted from descriptions to improve job title normalization.
- **ESCOXLM-R**: Based on XLM-R with domain-adaptive pretraining on the ESCO taxonomy (skills, competencies, occupations), multilingual (27 languages), with an additional training objective for inducing taxonomic relations.
- **ConFit / ConFit v2**: Not downloadable pretrained models — a contrastive learning method for fine-tuning an encoder on CV-Job pairs, with data augmentation (v1) and hard-negative mining + LLM-generated hypothetical summaries (v2). ConFit v2 reports +13.8% recall and +17.5% nDCG over ConFit v1, BM25, and OpenAI's `text-embedding-003`. This is the approach already referenced in `docs/design/scope.md` as "ConFit-style".

## Reading for the pending decision

CareerBERT is the most direct drop-in replacement for `all-MiniLM-L6-v2` (already fine-tuned and published, no training required). ConFit v2 is the best-reported-performance path, but requires fine-tuning our own model with contrastive learning on the Djinni dataset — more effort, but consistent with what's already noted as a possible v2 approach.

## References

- Rosenberger, J. et al. (2025). *CareerBERT: Matching resumes to ESCO jobs in a shared embedding space for generic job recommendations*. Expert Systems With Applications, 275. https://arxiv.org/abs/2503.02056
- *ConFit: Improving Resume-Job Matching using Data Augmentation and Contrastive Learning* (2024). https://arxiv.org/pdf/2401.16349
- *ConFit v2: Improving Resume-Job Matching using Hypothetical Resume Embedding and Runner-Up Hard-Negative Mining* (2025). ACL Findings. https://arxiv.org/abs/2502.12361
- *Embedding-based Recommender System for Job to Candidate Matching on Scale* (2021). https://arxiv.org/pdf/2107.00221
- *Tripartite Vector Representations for Better Job Recommendation* (2019). https://arxiv.org/pdf/1907.12379
