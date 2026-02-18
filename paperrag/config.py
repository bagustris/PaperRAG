"""Configuration module using Pydantic models."""

from __future__ import annotations

import json
import logging
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

_rc_logger = logging.getLogger(__name__)

# Mapping from .paperragrc keys to config field paths
_RC_KEY_MAP: dict[str, tuple[str, type]] = {
    "model": ("llm.model_name", str),
    "topk": ("retriever.top_k", int),
    "max-tokens": ("llm.max_tokens", int),
    "temperature": ("llm.temperature", float),
    "threshold": ("retriever.score_threshold", float),
    "index-dir": ("index_dir", str),
    "input-dir": ("input_dir", str),
}


def load_rc(path: Path) -> dict:
    """Load a .paperragrc TOML file, returning a flat dict of overrides."""
    if not path.is_file():
        return {}
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception as exc:
        _rc_logger.warning("Failed to parse %s: %s", path, exc)
        return {}


def apply_rc(cfg: "PaperRAGConfig", overrides: dict) -> None:
    """Apply .paperragrc overrides to a PaperRAGConfig instance."""
    for key, value in overrides.items():
        if key not in _RC_KEY_MAP:
            _rc_logger.warning("Unknown .paperragrc key: %s", key)
            continue
        field_path, expected_type = _RC_KEY_MAP[key]
        try:
            casted = expected_type(value)
        except (ValueError, TypeError) as exc:
            _rc_logger.warning("Invalid value for %s in .paperragrc: %s", key, exc)
            continue

        parts = field_path.split(".")
        if len(parts) == 2:
            sub, attr = parts
            setattr(getattr(cfg, sub), attr, casted)
        else:
            setattr(cfg, parts[0], casted)


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

    top_k: int = Field(default=2, ge=1)
    score_threshold: float = Field(
        default=0.1,
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

    model_config = {"extra": "ignore"}  # tolerate old snapshots with api_base/api_key

    model_name: str = "qwen2.5:1.5b"
    temperature: float = 0.0
    max_tokens: int = 128


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
        input_path = Path(self.input_dir)
        if input_path.suffix.lower() == ".pdf":
            return str(input_path.parent / ".paperrag-index")
        return str(input_path / ".paperrag-index")
    
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
