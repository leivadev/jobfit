from dataclasses import dataclass

import pandas as pd

from backend.api.artifacts import load_artifacts
from backend.config import Settings
from backend.domain.embeddings import BiEncoder, EmbeddingModel
from backend.domain.rerank import CrossEncoder, CrossEncoderModel
from backend.domain.search import JobSearchIndex


@dataclass(frozen=True)
class AppState:
    """Everything the `/recommend` route needs, loaded once at startup.

    Held as one object on `app.state.jobfit` rather than several attributes so
    tests can substitute a whole fake state (fixture index/metadata, fake
    models) via `create_app(state_factory=...)` without touching R2 or
    downloading real models.
    """

    search_index: JobSearchIndex
    metadata: pd.DataFrame
    bi_encoder: EmbeddingModel
    cross_encoder: CrossEncoderModel


def build_state(settings: Settings) -> AppState:
    """Load Artifacts from R2 and both models into memory. Raises on any failure."""
    search_index, metadata = load_artifacts(settings)
    return AppState(
        search_index=search_index,
        metadata=metadata,
        bi_encoder=BiEncoder(),
        cross_encoder=CrossEncoder(),
    )
