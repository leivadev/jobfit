from typing import Annotated

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from backend.api.rate_limit import RECOMMEND_RATE_LIMIT, limiter
from backend.api.schemas import Recommendation, RecommendResponse
from backend.api.snippets import build_snippet
from backend.api.state import AppState
from backend.domain.extraction import (
    CvExtractionError,
    UnsupportedCvFormatError,
    extract_text,
)
from backend.domain.matching import (
    CandidateSignals,
    InvalidCandidateSignalError,
    exp_distance,
    keyword_match,
    validate_candidate_signals,
)
from backend.domain.rerank import TOP_K, ShortlistItem, TieredJob, rerank, tier_by_fit

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    """Pure liveness check: no Artifact/model state, no request body."""
    return {"status": "ok"}


@router.post("/recommend", response_model=RecommendResponse)
@limiter.limit(RECOMMEND_RATE_LIMIT)
async def recommend(
    request: Request,
    file: Annotated[UploadFile, File()],
    exp_years: Annotated[str | None, Form()] = None,
    keywords: Annotated[list[str] | None, Form()] = None,
) -> RecommendResponse:
    state: AppState = request.app.state.jobfit
    candidate_signals = CandidateSignals(exp_years=exp_years, keywords=keywords)
    try:
        validate_candidate_signals(candidate_signals)
    except InvalidCandidateSignalError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    content = await file.read()

    try:
        candidate_profile = extract_text(content, filename=file.filename, mime_type=file.content_type)
    except (UnsupportedCvFormatError, CvExtractionError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    query_vector = state.bi_encoder.encode([candidate_profile])[0]
    search_results = state.search_index.search(query_vector)

    shortlist = [
        ShortlistItem(job_row=result.job_row, job_text=state.metadata.iloc[result.job_row]["job_text"])
        for result in search_results
    ]
    # Score the whole shortlist, not just the top 10: tiering below must see
    # every scored Job so it can surface a same-tier Job that raw rerank_score
    # alone had ranked outside the top 10.
    rerank_results = rerank(candidate_profile, shortlist, state.cross_encoder, top_k=len(shortlist))

    tiered_jobs = []
    for result in rerank_results:
        row = state.metadata.iloc[result.job_row]
        tiered_jobs.append(
            TieredJob(
                job_row=result.job_row,
                rerank_score=result.rerank_score,
                keyword_match=keyword_match(candidate_signals.keywords, row["keyword"]),
                exp_distance=exp_distance(candidate_signals.exp_years, row["exp_years"]),
            )
        )
    ranked_jobs = tier_by_fit(tiered_jobs)[:TOP_K]

    results = [_to_recommendation(job, state.metadata) for job in ranked_jobs]
    return RecommendResponse(results=results)


def _to_recommendation(job: TieredJob, metadata: pd.DataFrame) -> Recommendation:
    row = metadata.iloc[job.job_row]
    return Recommendation(
        position=row["position"],
        company=row["company"],
        rerank_score=job.rerank_score,
        exp_years=row["exp_years"],
        keyword=row["keyword"],
        snippet=build_snippet(row["job_text"], row["position"]),
        keyword_match=job.keyword_match,
        exp_distance=job.exp_distance,
    )
