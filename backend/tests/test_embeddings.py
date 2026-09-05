import numpy as np
import pytest
from embeddings import embed_texts, load_model

from backend.domain.embeddings import BiEncoder

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def model():
    return load_model()


def test_load_model_returns_a_bi_encoder(model):
    assert isinstance(model, BiEncoder)


def test_embed_texts_delegates_to_bi_encoder_encode(model):
    texts = [
        "Python Developer\nBuild backend services.",
        "QA Engineer\nWrite test plans.",
    ]

    result = embed_texts(texts, model)

    np.testing.assert_array_equal(result, model.encode(texts))
