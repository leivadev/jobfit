import faiss
import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.api.rate_limit import RECOMMEND_RATE_LIMIT_PER_MINUTE
from backend.api.state import AppState
from backend.api.upload_limits import MAX_UPLOAD_SIZE_BYTES
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


def test_recommend_tiers_by_keyword_match_then_exp_distance_when_both_signals_declared(client):
    response = client.post(
        "/recommend",
        files={"file": ("cv.txt", CANDIDATE_PROFILE.encode("utf-8"), "text/plain")},
        data={"exp_years": "1y", "keywords": ["Python", "DevOps"]},
    )

    assert response.status_code == 200
    results = response.json()["results"]
    # Backend Developer (Python, exp_dist 1) and DevOps Engineer (DevOps, exp_dist
    # 3) both match a declared keyword, so both outrank the non-matching QA
    # Engineer regardless of rerank_score; within the matching tier, the closer
    # exp_distance (Backend Developer) sorts first.
    assert [r["position"] for r in results] == [
        "Backend Developer",
        "DevOps Engineer",
        "QA Engineer",
    ]

    by_position = {r["position"]: r for r in results}
    assert by_position["Backend Developer"]["keyword_match"] is True
    assert by_position["Backend Developer"]["exp_distance"] == 1  # 1y -> 2y
    assert by_position["QA Engineer"]["keyword_match"] is False
    assert by_position["QA Engineer"]["exp_distance"] == 0  # 1y -> 1y
    assert by_position["DevOps Engineer"]["keyword_match"] is True
    assert by_position["DevOps Engineer"]["exp_distance"] == 3  # 1y -> 5y


def test_recommend_with_only_exp_years_tiers_by_exp_distance_ascending(client):
    response = client.post(
        "/recommend",
        files={"file": ("cv.txt", CANDIDATE_PROFILE.encode("utf-8"), "text/plain")},
        data={"exp_years": "1y"},
    )

    assert response.status_code == 200
    results = response.json()["results"]
    # No keywords declared, so keyword_match is False (a no-op tier) for every
    # job; ranking falls back to exp_distance alone: QA (0) < Backend (1) <
    # DevOps (3), overriding the rerank_score-only baseline order.
    assert [r["position"] for r in results] == [
        "QA Engineer",
        "Backend Developer",
        "DevOps Engineer",
    ]

    by_position = {r["position"]: r for r in results}
    assert by_position["Backend Developer"]["exp_distance"] == 1  # 1y -> 2y
    for result in results:
        assert result["keyword_match"] is False


def test_recommend_with_only_keywords_tiers_matches_before_non_matches(client):
    response = client.post(
        "/recommend",
        files={"file": ("cv.txt", CANDIDATE_PROFILE.encode("utf-8"), "text/plain")},
        data={"keywords": ["Python"]},
    )

    assert response.status_code == 200
    results = response.json()["results"]

    by_position = {r["position"]: r for r in results}
    assert by_position["Backend Developer"]["keyword_match"] is True
    assert by_position["QA Engineer"]["keyword_match"] is False
    for result in results:
        assert result["exp_distance"] is None

    # Backend Developer is the only keyword match, so it leads regardless of
    # how the non-matching jobs compare to each other by rerank_score.
    assert results[0]["position"] == "Backend Developer"


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


def test_recommend_rejects_file_over_size_limit_with_413(client):
    # Rejected on declared Content-Length before the body is read at all, so
    # the content itself never needs to be a real CV.
    oversized_content = b"a" * (MAX_UPLOAD_SIZE_BYTES + 1)

    response = client.post(
        "/recommend",
        files={"file": ("cv.txt", oversized_content, "text/plain")},
    )

    assert response.status_code == 413


def test_recommend_caps_bytes_read_when_content_length_is_not_declared(client):
    # Built via a raw httpx.Request/chunked generator rather than `files=`,
    # so the client sends Transfer-Encoding: chunked with no Content-Length
    # header -- exercising the fallback byte-count cap during the read,
    # rather than the declared-length precheck the other 413 test covers.
    oversized_content = b"a" * (MAX_UPLOAD_SIZE_BYTES + 1)
    prepared = httpx.Request(
        "POST",
        "http://testserver/recommend",
        files={"file": ("cv.bin", oversized_content, "application/octet-stream")},
    )
    multipart_body = b"".join(prepared.stream)

    def chunked_body():
        chunk_size = 65536
        for offset in range(0, len(multipart_body), chunk_size):
            yield multipart_body[offset : offset + chunk_size]

    response = client.post(
        "/recommend",
        content=chunked_body(),
        headers={"content-type": prepared.headers["content-type"]},
    )

    assert response.status_code == 413


def test_recommend_rejects_cv_with_no_extractable_text(client):
    response = client.post(
        "/recommend",
        files={"file": ("cv.txt", b"   \n\t  ", "text/plain")},
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


def test_recommend_returns_429_on_the_fourth_request_within_a_minute(client):
    for _ in range(RECOMMEND_RATE_LIMIT_PER_MINUTE):
        response = client.post(
            "/recommend",
            files={"file": ("cv.txt", CANDIDATE_PROFILE.encode("utf-8"), "text/plain")},
        )
        assert response.status_code == 200

    fourth_response = client.post(
        "/recommend",
        files={"file": ("cv.txt", CANDIDATE_PROFILE.encode("utf-8"), "text/plain")},
    )
    assert fourth_response.status_code == 429


def test_health_is_not_rate_limited(client):
    for _ in range(5):
        response = client.get("/health")
        assert response.status_code == 200


LARGE_FIXTURE_CANDIDATE_PROFILE = "Jane Doe\nJunior Python Developer with 1 year of experience."


def make_large_fixture_state() -> AppState:
    """11 Jobs: 10 filler Jobs outscore a Job that only fits on declared signals.

    `Perfect Fit Engineer` has the lowest rerank_score of all 11 -- pure
    rerank_score ordering truncates it out of the top 10 entirely -- but it's
    the candidate's only keyword/exp_years match. Used to prove tiering runs
    over the full pre-truncation shortlist, not just the naive top 10.
    """
    n_jobs = 11
    embeddings = np.eye(n_jobs, dtype=np.float32)
    index = faiss.IndexFlatIP(n_jobs)
    index.add(embeddings)

    positions = [f"Filler {i}" for i in range(10)] + ["Perfect Fit Engineer"]
    job_texts = [f"{position} job description." for position in positions]
    metadata = pd.DataFrame(
        {
            "id": [f"job-{i}" for i in range(n_jobs)],
            "position": positions,
            "company": [f"Company {i}" for i in range(n_jobs)],
            "exp_years": ["5y"] * 10 + ["1y"],
            "keyword": ["SQL"] * 10 + ["Python"],
            "job_text": job_texts,
        }
    )

    bi_encoder = FakeBiEncoder({LARGE_FIXTURE_CANDIDATE_PROFILE: embeddings[0]})
    scores_by_job_text = {job_texts[i]: 1.00 - 0.05 * i for i in range(10)}
    scores_by_job_text[job_texts[10]] = 0.50
    cross_encoder = FakeCrossEncoderModel(scores_by_job_text)

    return AppState(
        search_index=JobSearchIndex(index),
        metadata=metadata,
        bi_encoder=bi_encoder,
        cross_encoder=cross_encoder,
    )


def test_recommend_tiers_the_full_shortlist_before_truncating_to_top_10():
    app = create_app(state_factory=make_large_fixture_state)
    with TestClient(app) as large_client:
        response = large_client.post(
            "/recommend",
            files={
                "file": (
                    "cv.txt",
                    LARGE_FIXTURE_CANDIDATE_PROFILE.encode("utf-8"),
                    "text/plain",
                )
            },
            data={"exp_years": "1y", "keywords": ["Python"]},
        )

    assert response.status_code == 200
    positions = [r["position"] for r in response.json()["results"]]

    assert len(positions) == 10
    assert positions[0] == "Perfect Fit Engineer"
    assert "Filler 9" not in positions
