"""Tests for the embedder module."""

import numpy as np
import pytest

from paperrag.config import EmbedderConfig
from paperrag.embedder import Embedder


@pytest.fixture(scope="module")
def embedder():
    return Embedder(EmbedderConfig())


def test_dimension(embedder: Embedder):
    assert embedder.dimension == 384  # all-MiniLM-L6-v2


def test_embed_single(embedder: Embedder):
    vecs = embedder.embed(["Hello world"])
    assert vecs.shape == (1, 384)
    assert vecs.dtype == np.float32


def test_embed_batch(embedder: Embedder):
    texts = [f"sentence {i}" for i in range(10)]
    vecs = embedder.embed(texts)
    assert vecs.shape == (10, 384)


def test_embed_empty(embedder: Embedder):
    vecs = embedder.embed([])
    assert vecs.shape == (0, 384)


def test_embed_deterministic(embedder: Embedder):
    a = embedder.embed(["deterministic check"])
    b = embedder.embed(["deterministic check"])
    np.testing.assert_array_almost_equal(a, b)


def test_normalised_vectors(embedder: Embedder):
    vecs = embedder.embed(["normalisation test"])
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_array_almost_equal(norms, 1.0, decimal=4)
