"""Embedding module using sentence-transformers."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from paperrag.config import EmbedderConfig

logger = logging.getLogger(__name__)


class Embedder:
    """Wrapper around a SentenceTransformer model with batched encoding
    and deterministic seed control."""

    def __init__(self, config: EmbedderConfig | None = None) -> None:
        self.config = config or EmbedderConfig()
        self._set_seed(self.config.seed)
        device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            "Loading embedding model %s on %s",
            self.config.model_name,
            device,
        )
        self.model = SentenceTransformer(self.config.model_name, device=device)
        self.device = device
        self.dimension: int = self.model.get_sentence_embedding_dimension()  # type: ignore[assignment]
        logger.info("Embedding dimension: %d", self.dimension)

    @staticmethod
    def _set_seed(seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Encode *texts* and return an (N, D) float32 array.

        Uses batched encoding with the configured batch size.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        embeddings = self.model.encode(
            list(texts),
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
        )
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        assert arr.shape[1] == self.dimension, (
            f"Dimension mismatch: got {arr.shape[1]}, expected {self.dimension}"
        )
        return arr
