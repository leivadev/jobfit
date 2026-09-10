import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.api.usage_guardrail import UsageGuardrailMiddleware


def test_import_disables_joblib_multiprocessing():
    """See #34 and the rationale comment atop app.py. Run in a subprocess
    because joblib decides this once, at its own import time -- a fresh
    interpreter is the only way to check it actually landed before joblib
    was first imported, same as how uvicorn loads this module."""
    script = (
        "import backend.api.app\n"
        "import joblib._multiprocessing_helpers as h\n"
        "assert h.mp is None, h.mp\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode()


def test_startup_failure_propagates_instead_of_serving():
    """A factory failure (unreachable R2, missing Artifact, model load error)
    must fail ASGI startup rather than be swallowed — that's what lets the
    deployment platform's restart loop see a non-zero exit and retry."""

    def failing_factory():
        raise RuntimeError("R2 unreachable")

    app = create_app(state_factory=failing_factory)

    with pytest.raises(RuntimeError, match="R2 unreachable"), TestClient(app):
        pass


def test_cors_allows_configured_frontend_origin():
    app = create_app(state_factory=lambda: None, cors_origins=["http://localhost:5173"])

    with TestClient(app) as client:
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_rejects_unconfigured_origin():
    app = create_app(state_factory=lambda: None, cors_origins=["http://localhost:5173"])

    with TestClient(app) as client:
        response = client.options(
            "/health",
            headers={
                "Origin": "https://evil.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert "access-control-allow-origin" not in response.headers


def _has_usage_guardrail(app) -> bool:
    return any(m.cls is UsageGuardrailMiddleware for m in app.user_middleware)


def test_usage_guardrail_disabled_by_default(monkeypatch):
    monkeypatch.delenv("USAGE_GUARDRAIL_ENABLED", raising=False)

    app = create_app(state_factory=lambda: None)

    assert not _has_usage_guardrail(app)


def test_usage_guardrail_enabled_via_env(monkeypatch):
    monkeypatch.setenv("USAGE_GUARDRAIL_ENABLED", "true")

    app = create_app(state_factory=lambda: None)

    assert _has_usage_guardrail(app)
