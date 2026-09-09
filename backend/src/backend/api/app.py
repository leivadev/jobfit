from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIASGIMiddleware

from backend.api.rate_limit import limiter
from backend.api.routes import router
from backend.api.state import AppState, build_state
from backend.config import Settings, resolve_cors_allowed_origins

StateFactory = Callable[[], AppState]


def create_app(
    state_factory: StateFactory | None = None,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Build the FastAPI app, wiring Artifact/model loading into startup.

    `state_factory` defaults to loading real Artifacts from R2 and real
    models (`build_state`); tests inject a factory that builds a fixture
    `AppState` instead, so `TestClient` startup never touches R2 or downloads
    a model. Any exception raised by the factory propagates out of the
    lifespan context manager, which fails ASGI server startup — the process
    exits non-zero before serving a request (no in-process retry; the
    deployment platform's restart loop is the retry mechanism).

    `cors_origins` defaults to the `CORS_ALLOWED_ORIGINS`-configured
    allowlist (see `backend.config.resolve_cors_allowed_origins`); tests
    override it to check specific origins without depending on env state.
    """
    factory = state_factory or (lambda: build_state(Settings()))  # type: ignore[call-arg]
    origins = cors_origins if cors_origins is not None else resolve_cors_allowed_origins()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        app.state.jobfit = factory()
        yield

    app = FastAPI(lifespan=lifespan)
    app.state.limiter = limiter
    # limiter is a module-level singleton (routes.py decorates with it at
    # import time); reset so each app instance -- notably each test's --
    # starts with a clean counter.
    limiter.reset()
    app.add_middleware(SlowAPIASGIMiddleware)
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST"],
    )
    app.include_router(router)
    return app


app = create_app()
