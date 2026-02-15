"""Configuration module using Pydantic models."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


def _default_input_dir() -> str:
    return str(Path.home() / "Documents" / "Mendeley Desktop")


def _default_index_dir() -> str:
    return str(Path.home() / ".paper_rag" / "index")


class ParserConfig(BaseModel):
    """PDF parsing configuration."""

    extract_tables: bool = False
    fallback_to_raw: bool = True


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
        """Get actual worker count, auto-detecting if needed."""
        import multiprocessing
        if self.n_workers == 0:
            # Auto-detect: use all CPUs - 1 (leave one for system/embedding)
            cpu_count = multiprocessing.cpu_count()
            return max(1, cpu_count - 1)
        return self.n_workers


class LLMConfig(BaseModel):
    """LLM configuration."""

    mode: Literal["local", "openai"] = "openai"
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.0
    max_tokens: int = 1024
    api_base: str | None = None  # for Ollama: http://localhost:11434/v1
    api_key: str | None = None

    def resolve_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        env_key = os.getenv("OPENAI_API_KEY", "")
        if env_key:
            return env_key
        if self.mode == "local":
            return "not-needed"
        return ""


class PaperRAGConfig(BaseModel):
    """Top-level configuration."""

    input_dir: str = Field(default_factory=_default_input_dir)
    index_dir: str = Field(default_factory=_default_index_dir)

    parser: ParserConfig = Field(default_factory=ParserConfig)
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    def snapshot(self) -> dict:
        """Return a JSON-serialisable config snapshot for index metadata."""
        return self.model_dump(mode="json")

    def save_snapshot(self, path: Path) -> None:
        path.write_text(json.dumps(self.snapshot(), indent=2))

    @classmethod
    def load_snapshot(cls, path: Path) -> PaperRAGConfig:
        data = json.loads(path.read_text())
        return cls(**data)
