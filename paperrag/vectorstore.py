"""FAISS-backed vector store with persistence and metadata tracking."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import faiss
import numpy as np

from paperrag.chunker import Chunk
from paperrag.config import PaperRAGConfig

logger = logging.getLogger(__name__)

INDEX_FILE = "faiss.index"
METADATA_FILE = "metadata.json"
CONFIG_SNAPSHOT_FILE = "config_snapshot.json"
FILE_HASHES_FILE = "file_hashes.json"
VERSION_FILE = "version.json"


class VectorStore:
    """Manages a FAISS IndexFlatIP index plus chunk metadata on disk."""

    def __init__(self, index_dir: Path, dimension: int) -> None:
        self.index_dir = index_dir
        self.dimension = dimension

        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(dimension)
        self.chunks: list[dict] = []
        self.file_hashes: dict[str, str] = {}
        self.version: int = 0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, config: PaperRAGConfig | None = None) -> None:
        """Write index, metadata, hashes, version with atomic operations."""
        import shutil

        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Write FAISS index atomically
        index_tmp = self.index_dir / f"{INDEX_FILE}.tmp"
        faiss.write_index(self.index, str(index_tmp))
        shutil.move(str(index_tmp), str(self.index_dir / INDEX_FILE))

        # Write metadata as pickle (much faster than JSON)
        metadata_tmp = self.index_dir / f"{METADATA_FILE}.tmp"
        with open(metadata_tmp, 'wb') as f:
            pickle.dump(self.chunks, f, protocol=pickle.HIGHEST_PROTOCOL)
        shutil.move(str(metadata_tmp), str(self.index_dir / METADATA_FILE))

        # Write file hashes as pickle
        hashes_tmp = self.index_dir / f"{FILE_HASHES_FILE}.tmp"
        with open(hashes_tmp, 'wb') as f:
            pickle.dump(self.file_hashes, f, protocol=pickle.HIGHEST_PROTOCOL)
        shutil.move(str(hashes_tmp), str(self.index_dir / FILE_HASHES_FILE))

        # Version file remains JSON (small, human-readable)
        version_tmp = self.index_dir / f"{VERSION_FILE}.tmp"
        version_tmp.write_text(
            json.dumps({"version": self.version, "dimension": self.dimension})
        )
        shutil.move(str(version_tmp), str(self.index_dir / VERSION_FILE))

        # Config snapshot remains JSON (human-readable)
        if config is not None:
            config_tmp = self.index_dir / f"{CONFIG_SNAPSHOT_FILE}.tmp"
            config.save_snapshot(config_tmp)
            shutil.move(str(config_tmp), str(self.index_dir / CONFIG_SNAPSHOT_FILE))

        logger.info(
            "Saved index v%d (%d vectors) to %s",
            self.version,
            self.index.ntotal,
            self.index_dir,
        )

    @classmethod
    def load(cls, index_dir: Path) -> VectorStore:
        """Load an existing index from disk."""
        version_data = json.loads((index_dir / VERSION_FILE).read_text())
        dimension = version_data["dimension"]
        version = version_data["version"]

        store = cls(index_dir, dimension)
        store.index = faiss.read_index(str(index_dir / INDEX_FILE))

        # Load metadata - support both pickle (new) and JSON (legacy)
        metadata_file = index_dir / METADATA_FILE
        try:
            # Try pickle first (new format)
            with open(metadata_file, 'rb') as f:
                store.chunks = pickle.load(f)
            logger.info("Loaded pickle metadata format")
        except (pickle.UnpicklingError, ValueError, EOFError):
            # Fallback to JSON (legacy format)
            logger.info("Falling back to JSON metadata format")
            store.chunks = json.loads(metadata_file.read_text())

        # Load file hashes - support both pickle (new) and JSON (legacy)
        hashes_file = index_dir / FILE_HASHES_FILE
        try:
            # Try pickle first (new format)
            with open(hashes_file, 'rb') as f:
                store.file_hashes = pickle.load(f)
        except (pickle.UnpicklingError, ValueError, EOFError):
            # Fallback to JSON (legacy format)
            store.file_hashes = json.loads(hashes_file.read_text())

        store.version = version

        logger.info(
            "Loaded index v%d (%d vectors, dim=%d)",
            store.version,
            store.index.ntotal,
            store.dimension,
        )
        return store

    @classmethod
    def exists(cls, index_dir: Path) -> bool:
        return (index_dir / INDEX_FILE).exists() and (index_dir / VERSION_FILE).exists()

    # ------------------------------------------------------------------
    # Modification helpers
    # ------------------------------------------------------------------

    def add(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        """Add vectors and their corresponding chunk metadata."""
        assert embeddings.shape[0] == len(chunks)
        assert embeddings.shape[1] == self.dimension
        self.index.add(embeddings)
        self.chunks.extend([c.to_dict() for c in chunks])

    def remove_by_file(self, file_path: str) -> None:
        """Remove all vectors belonging to *file_path*.

        Because FAISS IndexFlatIP does not support selective removal we
        rebuild the index from the remaining vectors.
        """
        keep_indices = [
            i for i, c in enumerate(self.chunks) if c["file_path"] != file_path
        ]
        if len(keep_indices) == len(self.chunks):
            return

        if keep_indices:
            all_vecs = faiss.rev_swig_ptr(
                self.index.get_xb(), self.index.ntotal * self.dimension
            )
            all_vecs = np.array(all_vecs, dtype=np.float32).reshape(-1, self.dimension)
            kept_vecs = all_vecs[keep_indices]
            self.index.reset()
            self.index.add(kept_vecs)
        else:
            self.index.reset()

        self.chunks = [self.chunks[i] for i in keep_indices]

    def set_file_hash(self, file_path: str, file_hash: str) -> None:
        self.file_hashes[file_path] = file_hash

    def get_file_hash(self, file_path: str) -> str | None:
        return self.file_hashes.get(file_path)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query_vec: np.ndarray, top_k: int = 3) -> list[tuple[dict, float]]:
        """Return top-k (chunk_metadata, score) pairs."""
        if self.index.ntotal == 0:
            return []
        query_vec = np.asarray(query_vec, dtype=np.float32)
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        scores, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))
        results: list[tuple[dict, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self.chunks[idx], float(score)))
        return results
