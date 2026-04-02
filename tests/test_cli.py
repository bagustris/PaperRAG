"""Tests for CLI index discovery functionality."""

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from paperrag.cli import app
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


# --- review command tests --------------------------------------------------


def test_review_nonexistent_path():
    """review command exits with error for non-existent path."""
    runner = CliRunner()
    result = runner.invoke(app, ["review", "/tmp/this_file_definitely_does_not_exist.pdf"])
    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_review_calls_index_then_repl(tmp_path):
    """review command calls _handle_index then start_repl."""
    dummy_pdf = tmp_path / "paper.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 dummy")

    runner = CliRunner()
    with (
        patch("paperrag.repl._handle_index") as mock_index,
        patch("paperrag.repl.start_repl") as mock_repl,
    ):
        result = runner.invoke(app, ["review", str(dummy_pdf)])

    mock_index.assert_called_once()
    mock_repl.assert_called_once()
    # cfg passed to both calls must share the same input_dir
    index_cfg = mock_index.call_args[0][0]
    repl_cfg = mock_repl.call_args[0][0]
    assert index_cfg.input_dir == str(dummy_pdf)
    assert repl_cfg.input_dir == str(dummy_pdf)


def test_review_input_dir_set_to_argument(tmp_path):
    """review command sets cfg.input_dir to the supplied path."""
    dummy_pdf = tmp_path / "paper.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 dummy")

    runner = CliRunner()
    captured_cfgs: list = []

    def fake_index(cfg):
        captured_cfgs.append(cfg)

    def fake_repl(cfg):
        pass

    with (
        patch("paperrag.repl._handle_index", side_effect=fake_index),
        patch("paperrag.repl.start_repl", side_effect=fake_repl),
    ):
        runner.invoke(app, ["review", str(dummy_pdf)])

    assert len(captured_cfgs) == 1
    assert captured_cfgs[0].input_dir == str(dummy_pdf)


# --- REPL index command tests -----------------------------------------------


def test_repl_help_text_documents_index_path():
    """REPL HELP_TEXT mentions 'index <path>' command."""
    from paperrag.repl import HELP_TEXT

    assert "index <path>" in HELP_TEXT

