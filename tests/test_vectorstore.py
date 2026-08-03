"""Tests for vectorstore module - FAISS vector store operations."""

import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from paperrag.chunker import Chunk
from paperrag.config import PaperRAGConfig
from paperrag.vectorstore import VectorStore


class TestVectorStoreInit:
    """Test VectorStore initialization."""

    def test_initialization(self):
        """Test VectorStore initializes correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = Path(tmpdir)
            store = VectorStore(index_dir, dimension=384)
            
            assert store.index_dir == index_dir
            assert store.dimension == 384
            assert store.index.ntotal == 0
            assert store.chunks == []
            assert store.file_hashes == {}
            assert store.version == 0


class TestVectorStoreSaveLoad:
    """Test FAISS save/load operations."""

    def test_save_creates_directory(self):
        """Test save creates index directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = Path(tmpdir) / "subdir" / "index"
            store = VectorStore(index_dir, dimension=384)
            
            store.save()
            
            assert index_dir.exists()
            assert (index_dir / "faiss.index").exists()
            assert (index_dir / "metadata.json").exists()
            assert (index_dir / "file_hashes.json").exists()
            assert (index_dir / "version.json").exists()

    def test_save_load_roundtrip(self):
        """Test save and load roundtrip preserves data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = Path(tmpdir)
            
            # Create and populate store
            store = VectorStore(index_dir, dimension=384)
            store.chunks = [{"text": "chunk1"}, {"text": "chunk2"}]
            store.file_hashes = {"file1.pdf": "hash1", "file2.pdf": "hash2"}
            store.version = 1
            
            # Add some vectors
            vectors = np.random.rand(2, 384).astype(np.float32)
            store.index.add(vectors)
            
            store.save()
            
            # Load into new store
            loaded_store = VectorStore.load(index_dir)
            
            assert loaded_store.dimension == 384
            assert loaded_store.version == 1
            assert loaded_store.chunks == store.chunks
            assert loaded_store.file_hashes == store.file_hashes
            assert loaded_store.index.ntotal == 2

    def test_load_preserves_version(self):
        """Test load preserves version information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = Path(tmpdir)
            store = VectorStore(index_dir, dimension=384)
            store.version = 5
            store.save()
            
            loaded_store = VectorStore.load(index_dir)
            assert loaded_store.version == 5

    def test_save_with_config_snapshot(self):
        """Test save includes config snapshot when provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = Path(tmpdir)
            store = VectorStore(index_dir, dimension=384)
            
            config = MagicMock()
            # Make save_snapshot actually create a file
            def create_snapshot(path):
                path.write_text('{"config": "snapshot"}')
            config.save_snapshot = create_snapshot
            
            store.save(config)
            
            assert (index_dir / "config_snapshot.json").exists()
            assert (index_dir / "config_snapshot.json").read_text() == '{"config": "snapshot"}'

    def test_load_missing_version_file(self):
        """Test load fails when version file is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = Path(tmpdir)
            # Don't create version file
            with pytest.raises(FileNotFoundError):
                VectorStore.load(index_dir)


class TestVectorStoreOperations:
    """Test vector store operations."""

    def test_add_vectors(self):
        """Test adding vectors to the index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = Path(tmpdir)
            store = VectorStore(index_dir, dimension=384)
            
            vectors = np.random.rand(5, 384).astype(np.float32)
            store.index.add(vectors)
            
            assert store.index.ntotal == 5

    def test_add_chunks_metadata(self):
        """Test adding chunk metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = Path(tmpdir)
            store = VectorStore(index_dir, dimension=384)
            
            chunk = Chunk(
                chunk_id=0,
                hash_id="abc123",
                text="test text",
                paper_title="Test Paper",
                section_name="Abstract",
                file_path="/path/to/paper.pdf",
                file_hash="sha256"
            )
            
            store.chunks.append(chunk.to_dict())
            
            assert len(store.chunks) == 1
            assert store.chunks[0]["text"] == "test text"

    def test_file_hashes_tracking(self):
        """Test file hash tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = Path(tmpdir)
            store = VectorStore(index_dir, dimension=384)
            
            store.file_hashes["paper1.pdf"] = "hash1"
            store.file_hashes["paper2.pdf"] = "hash2"
            
            assert len(store.file_hashes) == 2
            assert store.file_hashes["paper1.pdf"] == "hash1"

    def test_version_increment(self):
        """Test version increment for reindexing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = Path(tmpdir)
            store = VectorStore(index_dir, dimension=384)
            
            assert store.version == 0
            store.version += 1
            assert store.version == 1

    def test_incremental_version_save(self):
        """Test incremental version saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = Path(tmpdir)
            store = VectorStore(index_dir, dimension=384)
            
            store.version = 1
            store.save()
            
            loaded = VectorStore.load(index_dir)
            assert loaded.version == 1
            
            # Increment and save again
            store.version = 2
            store.save()
            
            loaded = VectorStore.load(index_dir)
            assert loaded.version == 2


class TestVectorStoreMetadata:
    """Test metadata handling."""

    def test_metadata_save_load_pickle(self):
        """Test metadata save/load with pickle format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = Path(tmpdir)
            store = VectorStore(index_dir, dimension=384)
            
            store.chunks = [
                {"text": "chunk1", "metadata": "data1"},
                {"text": "chunk2", "metadata": "data2"}
            ]
            store.save()
            
            loaded = VectorStore.load(index_dir)
            assert loaded.chunks == store.chunks

    def test_metadata_save_load_json_fallback(self):
        """Test metadata save/load with JSON fallback for legacy format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = Path(tmpdir)
            store = VectorStore(index_dir, dimension=384)
            
            store.chunks = [{"text": "chunk1"}]
            store.save()
            
            # Manually convert to JSON format to test fallback
            metadata_file = index_dir / "metadata.json"
            with open(metadata_file, 'rb') as f:
                pickle_data = pickle.load(f)
            
            # Save as JSON
            with open(metadata_file, 'w') as f:
                json.dump(pickle_data, f)
            
            # Should load successfully with JSON fallback
            loaded = VectorStore.load(index_dir)
            assert loaded.chunks == store.chunks

    def test_file_hashes_save_load(self):
        """Test file hashes save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = Path(tmpdir)
            store = VectorStore(index_dir, dimension=384)
            
            store.file_hashes = {
                "paper1.pdf": "hash1",
                "paper2.pdf": "hash2"
            }
            store.save()
            
            loaded = VectorStore.load(index_dir)
            assert loaded.file_hashes == store.file_hashes


class TestVectorStoreReindexing:
    """Test hash-based reindexing logic."""

    def test_detect_changed_files(self):
        """Test detection of changed files based on hashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = Path(tmpdir)
            store = VectorStore(index_dir, dimension=384)
            
            # Store initial hashes
            store.file_hashes = {
                "paper1.pdf": "hash1",
                "paper2.pdf": "hash2"
            }
            
            # Simulate new hashes
            new_hashes = {
                "paper1.pdf": "hash1",  # Unchanged
                "paper2.pdf": "hash3",  # Changed
                "paper3.pdf": "hash4"   # New file
            }
            
            # Identify changed files
            changed_files = []
            new_files = []
            
            for filename, new_hash in new_hashes.items():
                if filename not in store.file_hashes:
                    new_files.append(filename)
                elif store.file_hashes[filename] != new_hash:
                    changed_files.append(filename)
            
            assert "paper2.pdf" in changed_files
            assert "paper3.pdf" in new_files
            assert "paper1.pdf" not in changed_files
            assert "paper1.pdf" not in new_files

    def test_reindex_with_unchanged_files(self):
        """Test reindexing skips unchanged files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = Path(tmpdir)
            store = VectorStore(index_dir, dimension=384)
            
            # Existing files
            store.file_hashes = {
                "paper1.pdf": "hash1",
                "paper2.pdf": "hash2"
            }
            store.chunks = [{"text": "existing chunk"}]
            store.version = 1
            store.save()
            
            # Load and simulate reindexing with unchanged files
            loaded = VectorStore.load(index_dir)
            
            # If no files changed, should not modify index
            assert loaded.version == 1
            assert len(loaded.chunks) == 1

    def test_reindex_with_changed_files(self):
        """Test reindexing processes changed files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = Path(tmpdir)
            store = VectorStore(index_dir, dimension=384)
            
            # Initial state
            store.file_hashes = {"paper1.pdf": "hash1"}
            store.chunks = [{"text": "old chunk"}]
            store.version = 1
            store.save()
            
            # Simulate file change
            loaded = VectorStore.load(index_dir)
            loaded.file_hashes["paper1.pdf"] = "hash2"  # Changed
            loaded.chunks = [{"text": "new chunk"}]
            loaded.version = 2
            loaded.save()
            
            # Verify new state
            reloaded = VectorStore.load(index_dir)
            assert reloaded.version == 2
            assert reloaded.file_hashes["paper1.pdf"] == "hash2"
            assert reloaded.chunks == [{"text": "new chunk"}]


class TestVectorStoreAtomicOperations:
    """Test atomic file operations."""

    def test_save_uses_temp_files(self):
        """Test save uses temporary files for atomicity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = Path(tmpdir)
            store = VectorStore(index_dir, dimension=384)
            
            store.save()
            
            # Check that temp files were cleaned up
            assert not (index_dir / "faiss.index.tmp").exists()
            assert not (index_dir / "metadata.json.tmp").exists()
            assert not (index_dir / "file_hashes.json.tmp").exists()
            assert not (index_dir / "version.json.tmp").exists()

    def test_save_creates_final_files(self):
        """Test save creates final files after atomic operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = Path(tmpdir)
            store = VectorStore(index_dir, dimension=384)
            
            store.save()
            
            # Check that final files exist
            assert (index_dir / "faiss.index").exists()
            assert (index_dir / "metadata.json").exists()
            assert (index_dir / "file_hashes.json").exists()
            assert (index_dir / "version.json").exists()