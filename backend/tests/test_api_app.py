import pytest
from fastapi.testclient import TestClient

from backend.api.app import create_app


def test_startup_failure_propagates_instead_of_serving():
    """A factory failure (unreachable R2, missing Artifact, model load error)
    must fail ASGI startup rather than be swallowed — that's what lets the
    deployment platform's restart loop see a non-zero exit and retry."""

    def failing_factory():
        raise RuntimeError("R2 unreachable")

    app = create_app(state_factory=failing_factory)

    with pytest.raises(RuntimeError, match="R2 unreachable"), TestClient(app):
        pass
