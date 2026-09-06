from pydantic import BaseModel


class Recommendation(BaseModel):
    position: str
    company: str
    rerank_score: float
    exp_years: str
    keyword: str
    snippet: str
    keyword_match: bool
    exp_distance: int | None


class RecommendResponse(BaseModel):
    results: list[Recommendation]
