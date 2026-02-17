"""Configuration module using Pydantic models."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


def _default_input_dir() -> str:
    return str(Path.home() / "Documents" / "Mendeley Desktop")


class ParserConfig(BaseModel):
    """PDF parsing configuration."""

    extract_tables: bool = False
    fallback_to_raw: bool = True
    ocr_mode: Literal["auto", "always", "never"] = Field(
        default="auto",
        description="OCR strategy: 'auto'=detect per PDF (recommended), 'always'=force OCR, 'never'=skip OCR"
    )
    manifest_file: str | None = Field(
        default=None,
        description="CSV manifest with columns: filename,title,authors,abstract,doi (optional)"
    )


class ChunkerConfig(BaseModel):
    """Chunking configuration."""

    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=200, ge=0)


class EmbedderConfig(BaseModel):
    """Embedding model configuration."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = Field(default=64, ge=1)
    device: str | None = None  # auto-detect if None
    normalize: bool = True
    seed: int = 42


class RetrieverConfig(BaseModel):
    """Retrieval configuration."""

    top_k: int = Field(default=5, ge=1)
    score_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold (0.0 = no filtering)"
    )
    use_mmr: bool = Field(
        default=False,
        description="Use Maximal Marginal Relevance for diverse retrieval"
    )
    mmr_lambda: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="MMR lambda parameter (0=max diversity, 1=max relevance)"
    )
    max_results_per_paper: int = Field(
        default=2,
        ge=1,
        description="Maximum results from same paper (re-ranking)"
    )


class IndexingConfig(BaseModel):
    """Indexing configuration."""

    checkpoint_interval: int = Field(
        default=50,
        ge=0,
        description="Save index every N PDFs during indexing (0 = no checkpoints)"
    )
    n_workers: int = Field(
        default=0,
        ge=0,
        description="Number of parallel PDF processing workers (0 = auto-detect)"
    )
    pdf_timeout: int = Field(
        default=300,
        ge=0,
        description="Timeout in seconds for processing a single PDF (0 = no timeout)"
    )
    enable_gc_per_batch: bool = Field(
        default=True,
        description="Enable garbage collection after each batch"
    )
    log_memory_usage: bool = Field(
        default=False,
        description="Log memory usage during indexing"
    )
    continue_on_error: bool = Field(
        default=True,
        description="Continue indexing even if individual PDFs fail"
    )
    max_failures: int = Field(
        default=-1,
        description="Maximum number of failures before stopping (-1 = unlimited)"
    )

    def get_n_workers(self) -> int:
        """Get actual worker count, auto-detecting if needed.
        
        Uses RAM-aware calculation to prevent OOM kills:
        - Each worker needs ~2GB during peak Docling usage
        - Formula: min(cpu_cores - 1, available_ram_gb // 2)
        """
        import multiprocessing
        if self.n_workers == 0:
            # Auto-detect: balance CPU and RAM constraints
            cpu_count = multiprocessing.cpu_count()
            cpu_workers = max(1, cpu_count - 1)
            
            # RAM-aware calculation (2GB per worker budget)
            try:
                import psutil
                available_gb = psutil.virtual_memory().available / (1024**3)
                # Reserve 2GB for base system + embedding, rest for workers
                ram_workers = max(1, int((available_gb - 2) / 2))
                workers = min(cpu_workers, ram_workers)
                if workers < cpu_workers:
                    import logging
                    logging.getLogger(__name__).info(
                        "Limited workers to %d (from %d) due to RAM constraints (%.1fGB available)",
                        workers, cpu_workers, available_gb
                    )
                return workers
            except ImportError:
                # psutil not available, fall back to CPU-only calculation
                return cpu_workers
        return self.n_workers


class LLMConfig(BaseModel):
    """LLM configuration."""

    model_name: str = "qwen3:1.7b"
    temperature: float = 0.0
    max_tokens: int = 512
    api_base: str = "http://localhost:11434/v1"
    api_key: str | None = None

    @field_validator("api_base", mode="before")
    @classmethod
    def _coerce_api_base(cls, v: object) -> str:
        """Accept None from old config snapshots, fall back to local Ollama default."""
        if v is None:
            return "http://localhost:11434/v1"
        return str(v)

    def resolve_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        env_key = os.getenv("OPENAI_API_KEY", "")
        if env_key:
            return env_key
        return "not-needed"


class PaperRAGConfig(BaseModel):
    """Top-level configuration."""

    input_dir: str = Field(default_factory=_default_input_dir)
    _index_dir: str | None = None  # Private field for custom index directory

    parser: ParserConfig = Field(default_factory=ParserConfig)
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    @property
    def index_dir(self) -> str:
        """Return index directory - custom path if set, otherwise input_dir/.paperrag-index."""
        if self._index_dir is not None:
            return self._index_dir
        return str(Path(self.input_dir) / ".paperrag-index")
    
    @index_dir.setter
    def index_dir(self, value: str) -> None:
        """Set custom index directory."""
        self._index_dir = value

    def snapshot(self) -> dict:
        """Return a JSON-serialisable config snapshot for index metadata."""
        return self.model_dump(mode="json")

    def save_snapshot(self, path: Path) -> None:
        path.write_text(json.dumps(self.snapshot(), indent=2))

    @classmethod
    def load_snapshot(cls, path: Path) -> PaperRAGConfig:
        data = json.loads(path.read_text())
        return cls(**data)
