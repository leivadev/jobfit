import faiss
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.api.state import AppState
from backend.domain.search import JobSearchIndex

CANDIDATE_PROFILE = "Jane Doe\nBackend Python Developer with 4 years experience."


class FakeBiEncoder:
    """Synthetic Bi-encoder: returns a fixed vector per exact text, keyed by text."""

    def __init__(self, vectors_by_text: dict[str, np.ndarray]) -> None:
        self._vectors_by_text = vectors_by_text

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vectors_by_text[text] for text in texts])


class FakeCrossEncoderModel:
    """Synthetic Cross-encoder: returns a fixed score per pair, keyed by job text."""

    def __init__(self, scores_by_job_text: dict[str, float]) -> None:
        self._scores_by_job_text = scores_by_job_text

    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        return np.array(
            [self._scores_by_job_text[job_text] for _, job_text in pairs],
            dtype=np.float32,
        )


def make_fixture_state() -> AppState:
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    index = faiss.IndexFlatIP(4)
    index.add(embeddings)

    metadata = pd.DataFrame(
        {
            "id": ["job-0", "job-1", "job-2"],
            "position": ["Backend Developer", "QA Engineer", "DevOps Engineer"],
            "company": ["Acme", "Globex", "Initech"],
            "exp_years": ["2-3", "1-3", "5+"],
            "keyword": ["Python", "QA", "DevOps"],
            "job_text": [
                "Backend Developer Build and maintain backend services in Python.",
                "QA Engineer Write and execute test plans.",
                "DevOps Engineer Manage CI/CD pipelines and infrastructure.",
            ],
        }
    )

    bi_encoder = FakeBiEncoder({CANDIDATE_PROFILE: embeddings[0]})
    cross_encoder = FakeCrossEncoderModel(
        {
            metadata.iloc[0]["job_text"]: 0.9,
            metadata.iloc[1]["job_text"]: 0.2,
            metadata.iloc[2]["job_text"]: 0.5,
        }
    )

    return AppState(
        search_index=JobSearchIndex(index),
        metadata=metadata,
        bi_encoder=bi_encoder,
        cross_encoder=cross_encoder,
    )


@pytest.fixture
def client():
    app = create_app(state_factory=make_fixture_state)
    with TestClient(app) as test_client:
        yield test_client


def test_recommend_ranks_by_rerank_score_end_to_end(client):
    response = client.post(
        "/recommend",
        files={"file": ("cv.txt", CANDIDATE_PROFILE.encode("utf-8"), "text/plain")},
    )

    assert response.status_code == 200
    positions = [r["position"] for r in response.json()["results"]]
    assert positions == ["Backend Developer", "DevOps Engineer", "QA Engineer"]


def test_recommend_response_matches_contract_shape(client):
    response = client.post(
        "/recommend",
        files={"file": ("cv.txt", CANDIDATE_PROFILE.encode("utf-8"), "text/plain")},
    )

    result = response.json()["results"][0]
    assert set(result.keys()) == {
        "position",
        "company",
        "rerank_score",
        "exp_years",
        "keyword",
        "snippet",
    }
    assert result["position"] == "Backend Developer"
    assert result["company"] == "Acme"
    assert result["exp_years"] == "2-3"
    assert result["keyword"] == "Python"
    assert result["snippet"] == "Build and maintain backend services in Python."
    assert result["rerank_score"] == pytest.approx(0.9)


def test_recommend_rejects_unsupported_cv_format(client):
    response = client.post(
        "/recommend",
        files={"file": ("cv.exe", b"whatever", "application/x-msdownload")},
    )

    assert response.status_code == 400
    assert "detail" in response.json()


def test_recommend_requires_a_file(client):
    response = client.post("/recommend")

    assert response.status_code == 422
    assert "detail" in response.json()


def test_health_returns_liveness_status(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
