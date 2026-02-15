"""Tests for the FAISS vector store."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from paper_rag.chunker import Chunk
from paper_rag.config import PaperRAGConfig
from paper_rag.vectorstore import VectorStore

DIM = 16
TMP_DIR = Path("/tmp/paper_rag_test_index")


@pytest.fixture(autouse=True)
def cleanup():
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    yield
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)


def _make_chunks(n: int, file_path: str = "/f.pdf") -> list[Chunk]:
    return [
        Chunk(
            chunk_id=i,
            hash_id=f"h{i}",
            text=f"text {i}",
            paper_title="P",
            section_name="S",
            file_path=file_path,
            file_hash="fhash",
        )
        for i in range(n)
    ]


def _make_vecs(n: int) -> np.ndarray:
    rng = np.random.RandomState(42)
    vecs = rng.randn(n, DIM).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def test_add_and_search():
    store = VectorStore(TMP_DIR, DIM)
    vecs = _make_vecs(5)
    chunks = _make_chunks(5)
    store.add(vecs, chunks)
    assert store.index.ntotal == 5

    results = store.search(vecs[0], top_k=2)
    assert len(results) == 2
    assert results[0][0]["chunk_id"] == 0


def test_save_and_load():
    store = VectorStore(TMP_DIR, DIM)
    vecs = _make_vecs(3)
    chunks = _make_chunks(3)
    store.add(vecs, chunks)
    store.set_file_hash("/f.pdf", "abc")
    store.version = 1
    store.save(config=PaperRAGConfig())

    loaded = VectorStore.load(TMP_DIR)
    assert loaded.index.ntotal == 3
    assert loaded.dimension == DIM
    assert loaded.version == 1
    assert loaded.get_file_hash("/f.pdf") == "abc"
    assert len(loaded.chunks) == 3


def test_remove_by_file():
    store = VectorStore(TMP_DIR, DIM)
    vecs_a = _make_vecs(3)
    vecs_b = _make_vecs(2)
    store.add(vecs_a, _make_chunks(3, "/a.pdf"))
    store.add(vecs_b, _make_chunks(2, "/b.pdf"))
    assert store.index.ntotal == 5

    store.remove_by_file("/a.pdf")
    assert store.index.ntotal == 2
    assert all(c["file_path"] == "/b.pdf" for c in store.chunks)


def test_exists():
    assert not VectorStore.exists(TMP_DIR)
    store = VectorStore(TMP_DIR, DIM)
    store.save()
    assert VectorStore.exists(TMP_DIR)


def test_search_empty():
    store = VectorStore(TMP_DIR, DIM)
    results = store.search(np.zeros((1, DIM), dtype=np.float32), top_k=3)
    assert results == []


def test_checkpoint_saves():
    """Test multiple saves simulate checkpointing behavior."""
    store = VectorStore(TMP_DIR, DIM)

    # First checkpoint - add 3 chunks
    vecs1 = _make_vecs(3)
    chunks1 = _make_chunks(3, "/a.pdf")
    store.add(vecs1, chunks1)
    store.set_file_hash("/a.pdf", "hash_a")
    store.save()

    # Second checkpoint - add 2 more chunks
    vecs2 = _make_vecs(2)
    chunks2 = [
        Chunk(
            chunk_id=i + 3,
            hash_id=f"h{i+3}",
            text=f"text {i+3}",
            paper_title="Paper B",
            section_name="Section2",
            file_path="/b.pdf",
            file_hash="hash_b",
        )
        for i in range(2)
    ]
    store.add(vecs2, chunks2)
    store.set_file_hash("/b.pdf", "hash_b")
    store.save()

    # Load and verify all data persisted correctly
    loaded = VectorStore.load(TMP_DIR)
    assert loaded.index.ntotal == 5
    assert len(loaded.chunks) == 5
    assert loaded.get_file_hash("/a.pdf") == "hash_a"
    assert loaded.get_file_hash("/b.pdf") == "hash_b"
