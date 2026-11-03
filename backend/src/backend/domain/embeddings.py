from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ENCODE_BATCH_SIZE = 128


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool) -> list[int]: ...
    def decode(self, token_ids: list[int]) -> str: ...


class EmbeddingModel(ABC):
    """Strategy interface for turning texts into vectors.

    Chunking/pooling, if any, is fully internal to the implementation — callers
    never see chunk boundaries, only one vector per input text.
    """

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def encode_query(self, texts: list[str]) -> np.ndarray: ...


class BiEncoder(EmbeddingModel):
    """bge-small-en-v1.5 Bi-encoder that chunks + mean-pools long text (ADR-0007, ADR-0009).

    Jobs (Offline Pipeline) go through `encode`, Candidate Profiles (query time)
    through `encode_query`. Both stay comparable under cosine similarity.
    """

    MODEL_NAME = "BAAI/bge-small-en-v1.5"
    # Pinned revision, not the moving default branch — an upstream model update
    # must not silently desync old Job vectors from new query vectors (ADR-0007).
    MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    # 512-token window minus the query prefix (~9 tokens) and [CLS]/[SEP].
    MAX_CHUNK_TOKENS = 480
    # bge is trained asymmetrically: the Candidate Profile is the "query", Jobs are
    # passages without a prefix.
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self) -> None:
        self.model = SentenceTransformer(self.MODEL_NAME, revision=self.MODEL_REVISION)
        # Raises model_max_length so tokenizer.encode() in _chunk_text doesn't warn
        # about "sequence length > 512" — that call is intentionally unbounded
        # since chunking (not truncation) handles the limit (ADR-0007).
        self.model.tokenizer.model_max_length = int(1e9)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Embed Jobs (passages): no prefix."""
        return self._encode_chunked(texts, prefix="")

    def encode_query(self, texts: list[str]) -> np.ndarray:
        """Embed Candidate Profiles (queries): QUERY_PREFIX on every chunk."""
        return self._encode_chunked(texts, prefix=self.QUERY_PREFIX)

    def _encode_chunked(self, texts: list[str], prefix: str) -> np.ndarray:
        """Embed each text as the mean-pooled vector of its chunk embeddings.

        Splits each text into <=MAX_CHUNK_TOKENS chunks (ADR-0007) so long descriptions
        are never silently truncated, encodes every chunk, and mean-pools the raw
        (unnormalized) chunk vectors into one (384,) vector per text. Normalization
        is deferred to the caller (e.g. FAISS index construction).
        """
        dim = self.model.get_embedding_dimension()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        chunks_per_text = [
            _chunk_text(text, self.model.tokenizer, self.MAX_CHUNK_TOKENS)
            for text in tqdm(texts, desc="chunking texts", unit="text")
        ]
        chunk_counts = [len(chunks) for chunks in chunks_per_text]
        all_chunks = [prefix + chunk for chunks in chunks_per_text for chunk in chunks]

        chunk_embeddings = (
            self._encode_raw(all_chunks, device)
            if all_chunks
            else np.empty((0, dim), dtype=np.float32)
        )

        boundaries = np.cumsum(chunk_counts)[:-1]
        embeddings = np.zeros((len(texts), dim), dtype=np.float32)
        for i, vectors in enumerate(np.split(chunk_embeddings, boundaries)):
            if len(vectors):
                embeddings[i] = vectors.mean(axis=0)
        return embeddings

    def _encode_raw(self, texts: list[str], device: str) -> np.ndarray:
        """Encode texts through the Transformer + Pooling modules only.

        The model's own pipeline appends a Normalize module (module 2), so
        model.encode()'s normalize_embeddings=False flag isn't enough to get
        raw vectors out — it still runs through Normalize. We call modules 0
        and 1 directly to skip it, since normalization is deferred (ADR-0007).
        """
        transformer, pooling = self.model[0], self.model[1]
        self.model.to(device)
        batches = []
        with torch.no_grad():
            for start in tqdm(
                range(0, len(texts), ENCODE_BATCH_SIZE),
                desc="embedding chunks",
                unit="batch",
            ):
                batch = texts[start : start + ENCODE_BATCH_SIZE]
                features = self.model.preprocess(batch)
                features = {
                    key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in features.items()
                }
                out = pooling(transformer(features))
                batches.append(out["sentence_embedding"].cpu().numpy())
        return np.concatenate(batches, axis=0).astype(np.float32)


def _chunk_text(text: str, tokenizer: Tokenizer, max_tokens: int) -> list[str]:
    """Split text into non-overlapping windows of at most max_tokens tokens.

    Uses the real model tokenizer rather than a char/word-count heuristic
    (see ADR-0007).
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []

    chunks = []
    start = 0
    while start < len(token_ids):
        window = token_ids[start : start + max_tokens]
        chunk = tokenizer.decode(window)
        # decode() isn't a perfect inverse of encode() (e.g. WordPiece
        # normalization can change token boundaries), so a decoded window
        # can re-encode to more than max_tokens. Shrink until it doesn't,
        # so re-tokenizing this chunk downstream never silently truncates it.
        while (
            len(window) > 1
            and len(tokenizer.encode(chunk, add_special_tokens=False)) > max_tokens
        ):
            window = window[:-1]
            chunk = tokenizer.decode(window)
        chunks.append(chunk)
        start += len(window)
    return chunks
