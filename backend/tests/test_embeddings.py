import numpy as np
import pytest
from embeddings import MODEL_REVISION, _encode_raw, embed_texts, load_model
from text import chunk_text

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def model():
    return load_model()


def test_load_model_pins_exact_revision(model):
    assert model[0].auto_model.config._commit_hash == MODEL_REVISION


def test_embed_texts_returns_one_384_vector_per_text(model):
    texts = ["Python Developer\nBuild backend services.", "QA Engineer\nWrite test plans."]

    embeddings = embed_texts(texts, model)

    assert embeddings.shape == (2, 384)
    assert embeddings.dtype == np.float32


def test_embed_texts_is_not_normalized(model):
    embeddings = embed_texts(["Python Developer\nBuild backend services."], model)

    norm = np.linalg.norm(embeddings[0])
    assert norm != pytest.approx(1.0, abs=1e-3)


def test_embed_texts_mean_pools_chunks_for_long_text(model):
    long_text = "Python Developer\n" + " ".join(f"requirement number {i}" for i in range(1000))
    chunks = chunk_text(long_text, model.tokenizer)
    assert len(chunks) > 1

    # Each chunk is itself under max_tokens, so embed_texts sees it as a
    # single chunk and returns its raw pooled vector unmodified.
    chunk_vectors = np.concatenate([embed_texts([chunk], model) for chunk in chunks], axis=0)
    expected = chunk_vectors.mean(axis=0)
    actual = embed_texts([long_text], model)[0]

    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_embed_texts_matches_raw_encoding_for_short_text(model):
    text = "Python Developer\nShort description."

    actual = embed_texts([text], model)[0]
    expected = _encode_raw(model, [text], device=str(next(model.parameters()).device))[0]

    np.testing.assert_allclose(actual, expected, atol=1e-5)
