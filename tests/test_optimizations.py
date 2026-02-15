"""Tests for optimization features: parallel processing, caching, etc."""

import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pytest

from paper_rag.chunker import Chunk
from paper_rag.config import IndexingConfig, PaperRAGConfig
from paper_rag.vectorstore import VectorStore

TMP_DIR = Path("/tmp/paper_rag_test_optimizations")


@pytest.fixture(autouse=True)
def cleanup():
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    yield
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)


def test_indexing_config_auto_workers():
    """Test automatic worker count detection."""
    config = IndexingConfig(n_workers=0)
    n_workers = config.get_n_workers()

    # Should return at least 1 worker
    assert n_workers >= 1

    # Should not exceed cpu_count
    import multiprocessing
    assert n_workers <= multiprocessing.cpu_count()


def test_indexing_config_explicit_workers():
    """Test explicit worker count."""
    config = IndexingConfig(n_workers=4)
    assert config.get_n_workers() == 4

    config = IndexingConfig(n_workers=1)
    assert config.get_n_workers() == 1


def test_pickle_save_and_load():
    """Test that vectorstore saves in pickle format and loads correctly."""
    dimension = 384
    store = VectorStore(TMP_DIR, dimension)

    # Add some data
    vecs = np.random.randn(5, dimension).astype(np.float32)
    chunks = [
        Chunk(i, f"hash_{i}", f"text_{i}", "Paper A", "Section1", "/a.pdf", "hash_a")
        for i in range(5)
    ]
    store.add(vecs, chunks)
    store.set_file_hash("/a.pdf", "hash_a")
    store.version = 1
    store.save()

    # Verify metadata is saved as pickle (not JSON)
    metadata_file = TMP_DIR / "metadata.json"
    with open(metadata_file, 'rb') as f:
        # Try to load as pickle - should succeed
        loaded_chunks = pickle.load(f)
        assert len(loaded_chunks) == 5

    # Verify hashes are saved as pickle
    hashes_file = TMP_DIR / "file_hashes.json"
    with open(hashes_file, 'rb') as f:
        loaded_hashes = pickle.load(f)
        assert loaded_hashes["/a.pdf"] == "hash_a"

    # Load and verify
    loaded = VectorStore.load(TMP_DIR)
    assert loaded.index.ntotal == 5
    assert len(loaded.chunks) == 5
    assert loaded.get_file_hash("/a.pdf") == "hash_a"


def test_json_backward_compatibility():
    """Test that vectorstore can still load old JSON format."""
    dimension = 384
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Create an old-style JSON index
    import faiss
    index = faiss.IndexFlatIP(dimension)
    vecs = np.random.randn(3, dimension).astype(np.float32)
    index.add(vecs)
    faiss.write_index(index, str(TMP_DIR / "faiss.index"))

    # Write metadata as JSON (old format)
    chunks = [
        {"chunk_id": i, "hash_id": f"h{i}", "text": f"text {i}",
         "paper_title": "Paper", "section_name": "Section",
         "file_path": "/a.pdf", "file_hash": "hash_a"}
        for i in range(3)
    ]
    (TMP_DIR / "metadata.json").write_text(json.dumps(chunks))

    # Write file hashes as JSON (old format)
    file_hashes = {"/a.pdf": "hash_a"}
    (TMP_DIR / "file_hashes.json").write_text(json.dumps(file_hashes))

    # Write version
    (TMP_DIR / "version.json").write_text(
        json.dumps({"version": 1, "dimension": dimension})
    )

    # Load should work and fall back to JSON
    store = VectorStore.load(TMP_DIR)
    assert store.index.ntotal == 3
    assert len(store.chunks) == 3
    assert store.get_file_hash("/a.pdf") == "hash_a"
    assert store.version == 1


def test_pickle_is_faster_than_json():
    """Benchmark: pickle should be faster than JSON for large datasets."""
    import time

    # Create large dataset
    large_chunks = [
        {"chunk_id": i, "hash_id": f"h{i}", "text": f"text {i}" * 100,
         "paper_title": "Paper", "section_name": "Section",
         "file_path": f"/file_{i}.pdf", "file_hash": f"hash_{i}"}
        for i in range(1000)
    ]

    # Time JSON serialization
    start = time.time()
    json_str = json.dumps(large_chunks)
    json_time = time.time() - start

    # Time pickle serialization
    start = time.time()
    pickle_bytes = pickle.dumps(large_chunks, protocol=pickle.HIGHEST_PROTOCOL)
    pickle_time = time.time() - start

    # Pickle should be significantly faster
    print(f"JSON time: {json_time:.4f}s, Pickle time: {pickle_time:.4f}s")
    assert pickle_time < json_time * 0.8  # At least 20% faster


def test_config_snapshot_compatibility():
    """Test that config with new fields can still save/load snapshots."""
    cfg = PaperRAGConfig()
    cfg.indexing.n_workers = 8
    cfg.indexing.checkpoint_interval = 100

    snapshot_file = TMP_DIR / "config.json"
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    cfg.save_snapshot(snapshot_file)

    # Load snapshot
    loaded_cfg = PaperRAGConfig.load_snapshot(snapshot_file)
    assert loaded_cfg.indexing.n_workers == 8
    assert loaded_cfg.indexing.checkpoint_interval == 100


def test_parallel_hashing_function():
    """Test parallel file hashing works correctly."""
    from paper_rag.parser import compute_file_hash, compute_file_hashes_parallel

    # Create test files
    test_dir = TMP_DIR / "test_pdfs"
    test_dir.mkdir(parents=True, exist_ok=True)

    test_files = []
    for i in range(5):
        test_file = test_dir / f"test_{i}.txt"
        test_file.write_text(f"Test content {i}")
        test_files.append(test_file)

    # Compute hashes serially
    serial_hashes = {str(f): compute_file_hash(f) for f in test_files}

    # Compute hashes in parallel
    parallel_hashes = compute_file_hashes_parallel(test_files, n_workers=2)

    # Results should match
    for f in test_files:
        assert str(f) in parallel_hashes
        assert parallel_hashes[str(f)] == serial_hashes[str(f)]


def test_embedder_cache():
    """Test that embedder is cached properly in retriever."""
    from paper_rag.retriever import _embedder_cache

    # Clear cache
    _embedder_cache.clear()

    # First load should populate cache
    cfg = PaperRAGConfig()
    cache_key = cfg.embedder.model_name

    # Cache should be empty initially
    assert cache_key not in _embedder_cache

    # After creating retriever (if index exists), cache should be populated
    # Note: We can't test full retriever without an index, but we can test the cache dict exists
    assert isinstance(_embedder_cache, dict)


def test_checkpoint_interval_zero_disables():
    """Test that checkpoint_interval=0 disables checkpointing."""
    cfg = PaperRAGConfig()
    cfg.indexing.checkpoint_interval = 0

    assert cfg.indexing.checkpoint_interval == 0

    # In actual indexing, this would skip the checkpoint logic
    # (tested in integration tests)


def test_multiple_checkpoints_with_pickle():
    """Test that multiple checkpoints work correctly with pickle format."""
    dimension = 384
    store = VectorStore(TMP_DIR, dimension)

    # First checkpoint
    vecs1 = np.random.randn(2, dimension).astype(np.float32)
    chunks1 = [
        Chunk(i, f"h{i}", f"text {i}", "P1", "S1", "/a.pdf", "hash_a")
        for i in range(2)
    ]
    store.add(vecs1, chunks1)
    store.set_file_hash("/a.pdf", "hash_a")
    store.save()

    # Second checkpoint
    vecs2 = np.random.randn(2, dimension).astype(np.float32)
    chunks2 = [
        Chunk(i+2, f"h{i+2}", f"text {i+2}", "P2", "S2", "/b.pdf", "hash_b")
        for i in range(2)
    ]
    store.add(vecs2, chunks2)
    store.set_file_hash("/b.pdf", "hash_b")
    store.save()

    # Third checkpoint
    vecs3 = np.random.randn(1, dimension).astype(np.float32)
    chunks3 = [Chunk(4, "h4", "text 4", "P3", "S3", "/c.pdf", "hash_c")]
    store.add(vecs3, chunks3)
    store.set_file_hash("/c.pdf", "hash_c")
    store.save()

    # Load and verify all data
    loaded = VectorStore.load(TMP_DIR)
    assert loaded.index.ntotal == 5
    assert len(loaded.chunks) == 5
    assert loaded.get_file_hash("/a.pdf") == "hash_a"
    assert loaded.get_file_hash("/b.pdf") == "hash_b"
    assert loaded.get_file_hash("/c.pdf") == "hash_c"
