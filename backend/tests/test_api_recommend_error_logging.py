import structlog
from fastapi.testclient import TestClient
from structlog.testing import capture_logs
from test_api_recommend import CANDIDATE_PROFILE, make_fixture_state

from backend.api.app import create_app


def test_recommend_unexpected_failure_logs_error_entry_with_exception_info(monkeypatch):
    state = make_fixture_state()

    def broken_encode(texts):
        raise RuntimeError("bi-encoder exploded")

    monkeypatch.setattr(state.bi_encoder, "encode", broken_encode)

    app = create_app(state_factory=lambda: state)
    # Runs the real `format_exc_info` processor (wired in `configure_logging`)
    # rather than letting `capture_logs`'s default bypass leave `exc_info` as
    # an unrendered `True`, so this actually proves a traceback is rendered.
    with (
        TestClient(app, raise_server_exceptions=False) as client,
        capture_logs(processors=[structlog.processors.format_exc_info]) as captured,
    ):
        response = client.post(
            "/recommend",
            files={"file": ("cv.txt", CANDIDATE_PROFILE.encode("utf-8"), "text/plain")},
        )

    assert response.status_code == 500

    error_entries = [entry for entry in captured if entry["log_level"] == "error"]
    assert len(error_entries) == 1
    error_entry = error_entries[0]
    assert error_entry["exc_type"] == "RuntimeError"
    assert error_entry["exc_message"] == "bi-encoder exploded"
    assert "Traceback (most recent call last)" in error_entry["exception"]
    assert "RuntimeError: bi-encoder exploded" in error_entry["exception"]
    assert error_entry["file_size_bytes"] == len(CANDIDATE_PROFILE.encode("utf-8"))
    assert isinstance(error_entry["duration_ms"], float)

    for entry in captured:
        for value in entry.values():
            assert "Jane Doe" not in str(value)
            assert CANDIDATE_PROFILE not in str(value)

    aggregate_entries = [entry for entry in captured if entry["event"] == "recommend_request"]
    assert len(aggregate_entries) == 1
    assert aggregate_entries[0]["error"] is True


def test_recommend_intentional_400_produces_no_error_level_entry():
    app = create_app(state_factory=make_fixture_state)
    with TestClient(app) as client, capture_logs() as captured:
        response = client.post(
            "/recommend",
            files={"file": ("cv.txt", CANDIDATE_PROFILE.encode("utf-8"), "text/plain")},
            data={"exp_years": "10y"},
        )

    assert response.status_code == 400

    error_entries = [entry for entry in captured if entry["log_level"] == "error"]
    assert error_entries == []

    aggregate_entries = [entry for entry in captured if entry["event"] == "recommend_request"]
    assert len(aggregate_entries) == 1
    assert aggregate_entries[0]["error"] is True
