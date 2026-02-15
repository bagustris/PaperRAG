"""Retriever: ties embedder + vector store for query-time retrieval."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from paper_rag.config import PaperRAGConfig
from paper_rag.embedder import Embedder
from paper_rag.vectorstore import VectorStore

logger = logging.getLogger(__name__)

# Module-level embedder cache to avoid reloading models
_embedder_cache: dict[str, Embedder] = {}


@dataclass
class RetrievalResult:
    """A single retrieval hit."""

    text: str
    score: float
    paper_title: str
    section_name: str
    file_path: str
    chunk_id: int

    def __str__(self) -> str:
        return (
            f"[{self.score:.4f}] {self.paper_title} | "
            f"{self.section_name} (chunk {self.chunk_id})"
        )


class Retriever:
    """High-level retriever that loads an existing index and answers queries."""

    def __init__(self, config: PaperRAGConfig) -> None:
        self.config = config
        index_dir = Path(config.index_dir)

        if not VectorStore.exists(index_dir):
            raise FileNotFoundError(
                f"No index found at {index_dir}. Run `paperrag index` first."
            )

        self.store = VectorStore.load(index_dir)

        # Cache embedder by model name to avoid reloading
        cache_key = config.embedder.model_name
        if cache_key in _embedder_cache:
            logger.info("Using cached embedder for model: %s", cache_key)
            self.embedder = _embedder_cache[cache_key]
        else:
            self.embedder = Embedder(config.embedder)
            _embedder_cache[cache_key] = self.embedder

        if self.embedder.dimension != self.store.dimension:
            raise ValueError(
                f"Embedder dimension ({self.embedder.dimension}) does not match "
                f"index dimension ({self.store.dimension})."
            )

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Embed *query* and return the top-k results from the vector store."""
        k = top_k or self.config.retriever.top_k
        query_vec = self.embedder.embed([query])
        raw = self.store.search(query_vec, top_k=k)

        results: list[RetrievalResult] = []
        for meta, score in raw:
            results.append(
                RetrievalResult(
                    text=meta["text"],
                    score=score,
                    paper_title=meta["paper_title"],
                    section_name=meta["section_name"],
                    file_path=meta["file_path"],
                    chunk_id=meta["chunk_id"],
                )
            )
        logger.info("Retrieved %d results for query: %.80s", len(results), query)
        return results

    def retrieve_file_paths(self, query: str, top_k: int | None = None) -> list[str]:
        """Return list of file_path strings (useful for evaluation)."""
        results = self.retrieve(query, top_k)
        return [r.file_path for r in results]
