# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /uvx /bin/

ENV UV_PYTHON_DOWNLOADS=0 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# Deps only, before the source is copied in, so editing app code doesn't
# invalidate this layer's cache.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev --no-editable

COPY README.md pyproject.toml uv.lock ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# Downloads both models at the pinned revisions BiEncoder/CrossEncoder load
# (backend/src/backend/domain/{embeddings,rerank}.py) into HF_HOME, baking
# them into this stage's filesystem.
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token)" \
    /app/.venv/bin/python -c "from backend.domain.embeddings import BiEncoder; from backend.domain.rerank import CrossEncoder; BiEncoder(); CrossEncoder()"

FROM python:3.12-slim AS runtime

RUN groupadd --system app && useradd --system --gid app --no-create-home app

ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/opt/hf-cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /opt/hf-cache /opt/hf-cache

USER app
EXPOSE 8000
CMD ["uvicorn", "backend.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
