import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from text import chunk_text

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Pinned revision, not the moving default branch — an upstream model update
# must not silently desync old Job vectors from new query vectors (ADR-0007).
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def load_model() -> SentenceTransformer:
    """Load all-MiniLM-L6-v2 pinned to an exact HF revision.

    Raises model_max_length so tokenizer.encode() in chunk_text doesn't warn
    about "sequence length > 256" — that call is intentionally unbounded
    since chunking (not truncation) handles the limit (ADR-0007).
    """
    model = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION)
    model.tokenizer.model_max_length = int(1e9)
    return model


def embed_texts(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """Embed each text as the mean-pooled vector of its chunk embeddings.

    Splits each text into <=256-token chunks via chunk_text (ADR-0007) so
    long descriptions are never silently truncated, encodes every chunk,
    and mean-pools the raw (unnormalized) chunk vectors into one (384,)
    vector per text. Normalization is deferred to build_faiss_index (#5).
    Not pure: moves `model` onto the detected device as a side effect.
    """
    dim = model.get_embedding_dimension()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    chunks_per_text = [
        chunk_text(text, model.tokenizer) for text in tqdm(texts, desc="chunking texts", unit="text")
    ]
    chunk_counts = [len(chunks) for chunks in chunks_per_text]
    all_chunks = [chunk for chunks in chunks_per_text for chunk in chunks]

    chunk_embeddings = (
        _encode_raw(model, all_chunks, device) if all_chunks else np.empty((0, dim), dtype=np.float32)
    )

    boundaries = np.cumsum(chunk_counts)[:-1]
    embeddings = np.zeros((len(texts), dim), dtype=np.float32)
    for i, vectors in enumerate(np.split(chunk_embeddings, boundaries)):
        if len(vectors):
            embeddings[i] = vectors.mean(axis=0)
    return embeddings


def _encode_raw(model: SentenceTransformer, texts: list[str], device: str) -> np.ndarray:
    """Encode texts through the Transformer + Pooling modules only.

    The model's own pipeline appends a Normalize module (module 2), so
    model.encode()'s normalize_embeddings=False flag isn't enough to get
    raw vectors out — it still runs through Normalize. We call modules 0
    and 1 directly to skip it, since normalization is deferred (ADR-0007).
    Moves `model` onto `device` in place.
    """
    transformer, pooling = model[0], model[1]
    model.to(device)
    batches = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), 128), desc="embedding chunks", unit="batch"):
            batch = texts[start : start + 128]
            features = model.preprocess(batch)
            features = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in features.items()
            }
            out = pooling(transformer(features))
            batches.append(out["sentence_embedding"].cpu().numpy())
    return np.concatenate(batches, axis=0).astype(np.float32)
