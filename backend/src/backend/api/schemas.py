from pydantic import BaseModel


class Recommendation(BaseModel):
    position: str
    company: str
    rerank_score: float
    exp_years: str
    keyword: str
    snippet: str


class RecommendResponse(BaseModel):
    results: list[Recommendation]
