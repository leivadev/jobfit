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
            "exp_years": ["2y", "1y", "5y"],
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
        "keyword_match",
        "exp_distance",
    }
    assert result["position"] == "Backend Developer"
    assert result["company"] == "Acme"
    assert result["exp_years"] == "2y"
    assert result["keyword"] == "Python"
    assert result["snippet"] == "Build and maintain backend services in Python."
    assert result["rerank_score"] == pytest.approx(0.9)


def test_recommend_without_exp_years_or_keywords_leaves_new_fields_absent_signal(client):
    response = client.post(
        "/recommend",
        files={"file": ("cv.txt", CANDIDATE_PROFILE.encode("utf-8"), "text/plain")},
    )

    assert response.status_code == 200
    for result in response.json()["results"]:
        assert result["keyword_match"] is False
        assert result["exp_distance"] is None


def test_recommend_computes_keyword_match_and_exp_distance_without_changing_ranking(client):
    baseline = client.post(
        "/recommend",
        files={"file": ("cv.txt", CANDIDATE_PROFILE.encode("utf-8"), "text/plain")},
    )
    baseline_positions = [r["position"] for r in baseline.json()["results"]]

    response = client.post(
        "/recommend",
        files={"file": ("cv.txt", CANDIDATE_PROFILE.encode("utf-8"), "text/plain")},
        data={"exp_years": "1y", "keywords": ["Python", "DevOps"]},
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert [r["position"] for r in results] == baseline_positions

    by_position = {r["position"]: r for r in results}
    assert by_position["Backend Developer"]["keyword_match"] is True
    assert by_position["Backend Developer"]["exp_distance"] == 1  # 1y -> 2y
    assert by_position["QA Engineer"]["keyword_match"] is False
    assert by_position["QA Engineer"]["exp_distance"] == 0  # 1y -> 1y
    assert by_position["DevOps Engineer"]["keyword_match"] is True
    assert by_position["DevOps Engineer"]["exp_distance"] == 3  # 1y -> 5y


def test_recommend_with_only_exp_years_computes_exp_distance_and_leaves_keyword_match_false(client):
    baseline = client.post(
        "/recommend",
        files={"file": ("cv.txt", CANDIDATE_PROFILE.encode("utf-8"), "text/plain")},
    )
    baseline_positions = [r["position"] for r in baseline.json()["results"]]

    response = client.post(
        "/recommend",
        files={"file": ("cv.txt", CANDIDATE_PROFILE.encode("utf-8"), "text/plain")},
        data={"exp_years": "1y"},
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert [r["position"] for r in results] == baseline_positions

    by_position = {r["position"]: r for r in results}
    assert by_position["Backend Developer"]["exp_distance"] == 1  # 1y -> 2y
    for result in results:
        assert result["keyword_match"] is False


def test_recommend_with_only_keywords_computes_keyword_match_and_leaves_exp_distance_none(client):
    baseline = client.post(
        "/recommend",
        files={"file": ("cv.txt", CANDIDATE_PROFILE.encode("utf-8"), "text/plain")},
    )
    baseline_positions = [r["position"] for r in baseline.json()["results"]]

    response = client.post(
        "/recommend",
        files={"file": ("cv.txt", CANDIDATE_PROFILE.encode("utf-8"), "text/plain")},
        data={"keywords": ["Python"]},
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert [r["position"] for r in results] == baseline_positions

    by_position = {r["position"]: r for r in results}
    assert by_position["Backend Developer"]["keyword_match"] is True
    assert by_position["QA Engineer"]["keyword_match"] is False
    for result in results:
        assert result["exp_distance"] is None


def test_recommend_rejects_exp_years_outside_controlled_vocabulary(client):
    response = client.post(
        "/recommend",
        files={"file": ("cv.txt", CANDIDATE_PROFILE.encode("utf-8"), "text/plain")},
        data={"exp_years": "10y"},
    )

    assert response.status_code == 400
    assert "detail" in response.json()


def test_recommend_rejects_keyword_outside_controlled_vocabulary(client):
    response = client.post(
        "/recommend",
        files={"file": ("cv.txt", CANDIDATE_PROFILE.encode("utf-8"), "text/plain")},
        data={"keywords": ["Python", "Haskell"]},
    )

    assert response.status_code == 400
    assert "detail" in response.json()


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
