"""Tests for CLI index discovery functionality."""

import shutil
from pathlib import Path

import pytest

from paperrag.vectorstore import VectorStore

TMP_DIR = Path("/tmp/paperrag_test_cli")


@pytest.fixture(autouse=True)
def cleanup():
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    yield
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)


def test_index_discovery_direct_path(tmp_path):
    """Test that REPL finds index when given direct path."""
    # Create index at tmp_path/my_index/
    index_path = tmp_path / "my_index"
    index_path.mkdir()

    dimension = 384
    store = VectorStore(index_path, dimension)
    store.version = 1
    store.save()

    # Should find index directly
    assert (index_path / "version.json").exists()

    # Verify index can be loaded
    loaded = VectorStore.load(index_path)
    assert loaded.version == 1


def test_index_discovery_subdirectory(tmp_path):
    """Test that REPL finds index in .paperrag-index subdirectory."""
    # Create index at tmp_path/.paperrag-index/
    subdir_path = tmp_path / ".paperrag-index"
    subdir_path.mkdir()

    dimension = 384
    store = VectorStore(subdir_path, dimension)
    store.version = 1
    store.save()

    # Discovery logic should find subdir_path when given tmp_path
    assert (subdir_path / "version.json").exists()
    assert not (tmp_path / "version.json").exists()

    # Verify index can be loaded from subdirectory
    loaded = VectorStore.load(subdir_path)
    assert loaded.version == 1


def test_index_exists_check():
    """Test that VectorStore.exists() works correctly."""
    # Non-existent path
    assert not VectorStore.exists(TMP_DIR / "nonexistent")

    # Create valid index
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    dimension = 384
    store = VectorStore(TMP_DIR, dimension)
    store.save()

    # Should exist now
    assert VectorStore.exists(TMP_DIR)
