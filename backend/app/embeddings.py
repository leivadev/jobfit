import numpy as np

from backend.domain.embeddings import BiEncoder


def load_model() -> BiEncoder:
    """Instantiate the pipeline's Bi-encoder (pinned revision, chunk+mean-pool internal)."""
    return BiEncoder()


def embed_texts(texts: list[str], model: BiEncoder) -> np.ndarray:
    """Embed each Job text via the Bi-encoder's chunk+mean-pool encode() (ADR-0007)."""
    return model.encode(texts)
