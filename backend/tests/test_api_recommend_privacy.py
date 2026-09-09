import tempfile
from pathlib import Path

from fastapi.testclient import TestClient
from structlog.testing import capture_logs
from test_api_recommend import CANDIDATE_PROFILE, FakeBiEncoder, make_fixture_state

from backend.api.app import create_app
from backend.api.state import AppState

# Forces Starlette's multipart parser past its 1 MB in-memory spool
# (see the `spool_max_size` fix in `backend.api.app.create_app`), while
# staying comfortably under the 5 MB accepted-upload limit.
_LARGE_CV_PADDING = "x" * (2 * 1024 * 1024)
_LARGE_CANDIDATE_PROFILE = f"{CANDIDATE_PROFILE}\n{_LARGE_CV_PADDING}"


def _large_upload_fixture_state() -> AppState:
    base_state = make_fixture_state()
    vector = base_state.bi_encoder._vectors_by_text[CANDIDATE_PROFILE]
    return AppState(
        search_index=base_state.search_index,
        metadata=base_state.metadata,
        bi_encoder=FakeBiEncoder({_LARGE_CANDIDATE_PROFILE: vector}),
        cross_encoder=base_state.cross_encoder,
    )


def test_recommend_never_writes_the_cv_to_disk(monkeypatch):
    disk_write_calls: list[str] = []

    original_rollover = tempfile.SpooledTemporaryFile.rollover

    def spy_rollover(self):
        disk_write_calls.append("SpooledTemporaryFile.rollover")
        return original_rollover(self)

    monkeypatch.setattr(tempfile.SpooledTemporaryFile, "rollover", spy_rollover)
    monkeypatch.setattr(
        Path, "write_bytes", lambda self, data: disk_write_calls.append("Path.write_bytes")
    )
    monkeypatch.setattr(
        Path, "write_text", lambda self, data, *a, **kw: disk_write_calls.append("Path.write_text")
    )

    app = create_app(state_factory=_large_upload_fixture_state)
    with TestClient(app) as client:
        response = client.post(
            "/recommend",
            files={"file": ("cv.txt", _LARGE_CANDIDATE_PROFILE.encode("utf-8"), "text/plain")},
        )

    assert response.status_code == 200
    assert disk_write_calls == []


def test_recommend_never_logs_cv_content():
    app = create_app(state_factory=make_fixture_state)
    with TestClient(app) as client, capture_logs() as captured:
        response = client.post(
            "/recommend",
            files={"file": ("cv.txt", CANDIDATE_PROFILE.encode("utf-8"), "text/plain")},
        )

    assert response.status_code == 200
    assert captured
    for entry in captured:
        for value in entry.values():
            assert "Jane Doe" not in str(value)
            assert CANDIDATE_PROFILE not in str(value)


def test_recommend_aggregate_log_entry_has_no_cv_content_on_error():
    app = create_app(state_factory=make_fixture_state)
    with TestClient(app) as client, capture_logs() as captured:
        response = client.post(
            "/recommend",
            files={"file": ("cv.exe", b"whatever", "application/x-msdownload")},
        )

    assert response.status_code == 400
    assert captured
    for entry in captured:
        assert entry["event"] == "recommend_request"
        assert entry["error"] is True
        for value in entry.values():
            assert b"whatever" != value
            assert "whatever" not in str(value)
