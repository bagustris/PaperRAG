"""Retriever: ties embedder + vector store for query-time retrieval."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from paperrag.config import PaperRAGConfig
from paperrag.embedder import Embedder
from paperrag.vectorstore import VectorStore

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
        """Embed *query* and return the top-k results from the vector store.
        
        Results are filtered by score_threshold - only results with similarity
        scores above the threshold are returned.
        
        If use_mmr=True, uses Maximal Marginal Relevance for diversity.
        """
        k = top_k or self.config.retriever.top_k
        query_vec = self.embedder.embed([query])
        
        # Use MMR if configured
        if self.config.retriever.use_mmr:
            return self._retrieve_mmr(query_vec, k)
        
        # Standard similarity search
        raw = self.store.search(query_vec, top_k=k)

        results: list[RetrievalResult] = []
        for meta, score in raw:
            # Filter by score threshold
            if score < self.config.retriever.score_threshold:
                continue
                
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
        
        # Apply per-paper limit (simple re-ranking)
        results = self._rerank_by_paper(results)
        
        logger.info("Retrieved %d results for query: %.80s", len(results), query)
        return results

    def _retrieve_mmr(self, query_vec: np.ndarray, k: int) -> list[RetrievalResult]:
        """Maximal Marginal Relevance retrieval for diverse results.
        
        MMR iteratively selects results that maximize:
        lambda * sim(query, doc) - (1-lambda) * max(sim(doc, selected))
        
        This ensures diverse results while maintaining relevance.
        """
        lambda_param = self.config.retriever.mmr_lambda
        fetch_k = k * 3  # Fetch more candidates for diversity
        
        # Get initial candidates
        raw = self.store.search(query_vec, top_k=fetch_k)
        
        if not raw:
            return []
        
        # Extract vectors and metadata
        candidates = []
        for meta, score in raw:
            if score < self.config.retriever.score_threshold:
                continue
            candidates.append((meta, score))
        
        if not candidates:
            return []
        
        # Get embeddings for all candidates
        candidate_texts = [meta["text"] for meta, _ in candidates]
        candidate_vecs = self.embedder.embed(candidate_texts)
        
        # MMR selection
        selected: list[RetrievalResult] = []
        selected_indices: list[int] = []
        
        for _ in range(min(k, len(candidates))):
            best_idx = -1
            best_score = -float('inf')
            
            for i, (meta, relevance_score) in enumerate(candidates):
                if i in selected_indices:
                    continue
                
                # Compute diversity penalty
                if selected_indices:
                    selected_vecs = candidate_vecs[selected_indices]
                    # Cosine similarity with selected docs
                    similarities = np.dot(selected_vecs, candidate_vecs[i])
                    max_sim = float(np.max(similarities))
                else:
                    max_sim = 0.0
                
                # MMR score: balance relevance and diversity
                mmr_score = lambda_param * relevance_score - (1 - lambda_param) * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            if best_idx == -1:
                break
            
            meta, score = candidates[best_idx]
            selected.append(
                RetrievalResult(
                    text=meta["text"],
                    score=score,  # Keep original relevance score
                    paper_title=meta["paper_title"],
                    section_name=meta["section_name"],
                    file_path=meta["file_path"],
                    chunk_id=meta["chunk_id"],
                )
            )
            selected_indices.append(best_idx)
        
        logger.info("MMR retrieved %d diverse results", len(selected))
        return selected
    
    def _rerank_by_paper(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Limit results per paper to avoid over-representation.
        
        Ensures diversity by limiting how many chunks from same paper.
        """
        max_per_paper = self.config.retriever.max_results_per_paper
        paper_counts: dict[str, int] = {}
        filtered: list[RetrievalResult] = []
        
        for result in results:
            paper = result.paper_title
            count = paper_counts.get(paper, 0)
            
            if count < max_per_paper:
                filtered.append(result)
                paper_counts[paper] = count + 1
        
        if len(filtered) < len(results):
            logger.info("Re-ranking filtered %d → %d results", len(results), len(filtered))
        
        return filtered
    
    def retrieve_file_paths(self, query: str, top_k: int | None = None) -> list[str]:
        """Return list of file_path strings (useful for evaluation)."""
        results = self.retrieve(query, top_k)
        return [r.file_path for r in results]
