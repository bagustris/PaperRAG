"""Tests for CLI index discovery functionality."""

import shutil
from pathlib import Path
from unittest.mock import patch

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
    """REPL HELP_TEXT mentions '/index <path>' command."""
    from paperrag.repl import HELP_TEXT

    assert "/index <path>" in HELP_TEXT


def test_repl_slash_commands_list():
    """Core REPL slash-commands are present in SLASH_COMMANDS."""
    from paperrag.repl import SLASH_COMMANDS

    required = {"/index", "/topk", "/threshold", "/temperature", "/max-tokens",
                "/model", "/config", "/rc", "/help", "/exit", "/quit"}
    assert required.issubset(set(SLASH_COMMANDS))


def test_slash_completer_returns_completions():
    """_SlashCompleter yields completions when input starts with '/'."""
    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document
    from prompt_toolkit.formatted_text import to_plain_text

    from paperrag.repl import _SlashCompleter

    completer = _SlashCompleter()
    doc = Document("/")
    completions = list(completer.get_completions(doc, CompleteEvent()))
    displayed = [to_plain_text(c.display) for c in completions]
    assert "/help" in displayed
    assert "/index" in displayed


def test_slash_completer_no_completions_for_plain_text():
    """_SlashCompleter returns nothing for non-slash input."""
    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document

    from paperrag.repl import _SlashCompleter

    completer = _SlashCompleter()
    doc = Document("what is attention?")
    completions = list(completer.get_completions(doc, CompleteEvent()))
    assert completions == []


def test_slash_completer_partial_match():
    """_SlashCompleter filters to commands matching the typed prefix."""
    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document
    from prompt_toolkit.formatted_text import to_plain_text

    from paperrag.repl import _SlashCompleter

    completer = _SlashCompleter()
    doc = Document("/t")
    completions = list(completer.get_completions(doc, CompleteEvent()))
    displayed = [to_plain_text(c.display) for c in completions]
    # /topk, /threshold, /temperature all start with /t
    assert "/topk" in displayed
    assert "/threshold" in displayed
    assert "/temperature" in displayed
    # /config does not
    assert "/config" not in displayed


def test_slash_completer_no_completions_after_whitespace():
    """_SlashCompleter returns nothing once whitespace is present (e.g. '/t ')."""
    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document

    from paperrag.repl import _SlashCompleter

    completer = _SlashCompleter()
    doc = Document("/t ")
    completions = list(completer.get_completions(doc, CompleteEvent()))
    assert completions == []


# --- review command option tests -------------------------------------------


def test_review_options_propagate_to_config(tmp_path):
    """review CLI options should be reflected in the PaperRAGConfig passed to handlers."""
    dummy_pdf = tmp_path / "paper.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 dummy")

    index_dir = tmp_path / "index"
    index_dir.mkdir()

    runner = CliRunner()
    with (
        patch("paperrag.parser.discover_pdfs", return_value=[dummy_pdf]),
        patch("paperrag.repl._handle_index") as mock_index,
        patch("paperrag.repl.start_repl") as mock_repl,
    ):
        result = runner.invoke(
            app,
            [
                "review",
                str(dummy_pdf),
                "--index-dir",
                str(index_dir),
                "--model",
                "test-model",
                "--topk",
                "7",
                "--threshold",
                "0.42",
                "--temperature",
                "0.7",
                "--max-tokens",
                "512",
            ],
        )

    assert result.exit_code == 0

    mock_index.assert_called_once()
    mock_repl.assert_called_once()

    index_cfg = mock_index.call_args[0][0]
    repl_cfg = mock_repl.call_args[0][0]

    # input_dir and index_dir
    assert index_cfg.input_dir == str(dummy_pdf)
    assert repl_cfg.input_dir == str(dummy_pdf)
    assert index_cfg.index_dir == str(index_dir)
    assert repl_cfg.index_dir == str(index_dir)

    # LLM options
    assert index_cfg.llm.model_name == "test-model"
    assert index_cfg.llm.temperature == 0.7
    assert index_cfg.llm.max_tokens == 512

    # Retriever options
    assert index_cfg.retriever.top_k == 7
    assert index_cfg.retriever.score_threshold == 0.42


def test_review_rc_loading_and_cli_precedence(tmp_path):
    """CLI flags should override rc-loaded values for review command."""
    dummy_pdf = tmp_path / "paper.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 dummy")

    runner = CliRunner()
    with (
        patch("paperrag.cli.load_rc", return_value={"model": "rc-model"}),
        patch("paperrag.parser.discover_pdfs", return_value=[dummy_pdf]),
        patch("paperrag.repl._handle_index") as mock_index,
        patch("paperrag.repl.start_repl") as mock_repl,
    ):
        result = runner.invoke(
            app,
            [
                "review",
                str(dummy_pdf),
                "--model",
                "cli-model",
            ],
        )

    assert result.exit_code == 0

    mock_index.assert_called_once()
    mock_repl.assert_called_once()

    index_cfg = mock_index.call_args[0][0]
    repl_cfg = mock_repl.call_args[0][0]

    # CLI model should win over rc model
    assert index_cfg.llm.model_name == "cli-model"
    assert repl_cfg.llm.model_name == "cli-model"


def test_review_with_directory_input(tmp_path):
    """review command works with a directory containing PDFs."""
    pdf_dir = tmp_path / "papers"
    pdf_dir.mkdir()
    dummy_pdf = pdf_dir / "paper1.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 dummy")

    runner = CliRunner()
    with (
        patch("paperrag.parser.discover_pdfs", return_value=[dummy_pdf]),
        patch("paperrag.repl._handle_index") as mock_index,
        patch("paperrag.repl.start_repl") as mock_repl,
    ):
        result = runner.invoke(app, ["review", str(pdf_dir)])

    assert result.exit_code == 0
    mock_index.assert_called_once()
    mock_repl.assert_called_once()

    cfg = mock_index.call_args[0][0]
    assert cfg.input_dir == str(pdf_dir)


def test_review_no_pdfs_found(tmp_path):
    """review command should exit with error when no PDFs are found."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    runner = CliRunner()
    with patch("paperrag.parser.discover_pdfs", return_value=[]):
        result = runner.invoke(app, ["review", str(empty_dir)])

    assert result.exit_code == 1
    assert "No PDFs found" in result.output


def test_review_resets_index_dir_from_rc(tmp_path):
    """review should clear RC-set index_dir so it auto-derives from input_path."""
    dummy_pdf = tmp_path / "paper.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 dummy")

    runner = CliRunner()
    with (
        patch("paperrag.cli.load_rc", return_value={"index-dir": "/some/rc/path"}),
        patch("paperrag.parser.discover_pdfs", return_value=[dummy_pdf]),
        patch("paperrag.repl._handle_index") as mock_index,
        patch("paperrag.repl.start_repl") as mock_repl,
    ):
        # No --index-dir flag: should auto-derive, NOT use the RC value
        result = runner.invoke(app, ["review", str(dummy_pdf)])

    assert result.exit_code == 0
    cfg = mock_index.call_args[0][0]
    # index_dir should be auto-derived from input_path, not /some/rc/path
    assert cfg.index_dir != "/some/rc/path"
    assert ".paperrag-index" in cfg.index_dir

