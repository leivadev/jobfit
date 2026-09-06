from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.api.routes import router
from backend.api.state import AppState, build_state
from backend.config import Settings

StateFactory = Callable[[], AppState]


def create_app(state_factory: StateFactory | None = None) -> FastAPI:
    """Build the FastAPI app, wiring Artifact/model loading into startup.

    `state_factory` defaults to loading real Artifacts from R2 and real
    models (`build_state`); tests inject a factory that builds a fixture
    `AppState` instead, so `TestClient` startup never touches R2 or downloads
    a model. Any exception raised by the factory propagates out of the
    lifespan context manager, which fails ASGI server startup — the process
    exits non-zero before serving a request (no in-process retry; the
    deployment platform's restart loop is the retry mechanism).
    """
    factory = state_factory or (lambda: build_state(Settings()))  # type: ignore[call-arg]

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.jobfit = factory()
        yield

    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()
