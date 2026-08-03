"""Tests for parallel module - parallel PDF processing orchestration.

Note: paperrag.parallel.process_single_pdf does `from paperrag.parser import
parse_pdf` and `from paperrag.chunker import chunk_paper` *inside* the
function body, so patches must target paperrag.parser.parse_pdf /
paperrag.chunker.chunk_paper (not paperrag.parallel.*).

The multi-worker tests rely on multiprocessing's default 'fork' start
method on Linux: forked workers inherit the parent's already-patched
module state, so plain module-level fake functions (not mocks) work
across the process boundary without needing IPC-safe mock objects.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from paperrag.chunker import Chunk
from paperrag.config import ChunkerConfig, ParserConfig
from paperrag.parallel import parallel_process_pdfs, process_single_pdf
from paperrag.parser import ParsedPaper, ParsedSection


def _fake_parse_pdf(path, config=None, manifest=None):
    if "bad" in Path(path).name:
        raise ValueError("simulated parse failure")
    return ParsedPaper(
        file_path=str(path),
        file_hash=f"hash-{Path(path).name}",
        title=f"Title for {Path(path).name}",
        authors="Author",
        sections=[ParsedSection(name="Body", text="content " * 50)],
    )


def _fake_chunk_paper(paper, config):
    return [
        Chunk(
            chunk_id=0,
            hash_id="hash0",
            text="chunk text",
            paper_title=paper.title,
            section_name="Body",
            file_path=paper.file_path,
            file_hash=paper.file_hash,
        )
    ]


def _fake_parse_pdf_slow_for_stuck(path, config=None, manifest=None):
    import time
    if "stuck" in Path(path).name:
        time.sleep(3)
    return _fake_parse_pdf(path, config, manifest)


class TestProcessSinglePdf:
    @patch("paperrag.chunker.chunk_paper")
    @patch("paperrag.parser.parse_pdf")
    def test_success_returns_hash_and_chunks(self, mock_parse, mock_chunk):
        paper = ParsedPaper(
            file_path="/a.pdf", file_hash="abc123", title="T", authors="A",
            sections=[ParsedSection(name="Body", text="x")],
        )
        mock_parse.return_value = paper
        mock_chunk.return_value = [
            Chunk(0, "h0", "text", "T", "Body", "/a.pdf", "abc123")
        ]

        path, file_hash, chunks, error = process_single_pdf(
            Path("/a.pdf"), ParserConfig(), ChunkerConfig()
        )

        assert path == Path("/a.pdf")
        assert file_hash == "abc123"
        assert len(chunks) == 1
        assert error is None

    @patch("paperrag.parser.parse_pdf")
    def test_parse_failure_returns_error_tuple(self, mock_parse):
        mock_parse.side_effect = ValueError("bad PDF")

        path, file_hash, chunks, error = process_single_pdf(
            Path("/broken.pdf"), ParserConfig(), ChunkerConfig()
        )

        assert path == Path("/broken.pdf")
        assert file_hash is None
        assert chunks is None
        assert error == "bad PDF"

    @patch("paperrag.chunker.chunk_paper")
    @patch("paperrag.parser.parse_pdf")
    def test_chunk_failure_returns_error_tuple(self, mock_parse, mock_chunk):
        mock_parse.return_value = ParsedPaper(
            file_path="/a.pdf", file_hash="h", title="T", authors="A",
        )
        mock_chunk.side_effect = RuntimeError("chunking exploded")

        path, file_hash, chunks, error = process_single_pdf(
            Path("/a.pdf"), ParserConfig(), ChunkerConfig()
        )

        assert file_hash is None
        assert chunks is None
        assert error == "chunking exploded"


class TestParallelProcessPdfsSingleThreaded:
    @patch("paperrag.parallel.process_single_pdf")
    def test_n_workers_one_processes_sequentially(self, mock_process):
        pdf_paths = [Path("/a.pdf"), Path("/b.pdf")]
        mock_process.side_effect = lambda pdf, *a, **kw: (pdf, "hash", [], None)

        results = parallel_process_pdfs(
            pdf_paths, ParserConfig(), ChunkerConfig(), n_workers=1
        )

        assert mock_process.call_count == 2
        assert [r[0] for r in results] == pdf_paths

    @patch("paperrag.parallel.process_single_pdf")
    def test_n_workers_one_preserves_errors(self, mock_process):
        pdf_paths = [Path("/a.pdf")]
        mock_process.return_value = (Path("/a.pdf"), None, None, "failed")

        results = parallel_process_pdfs(
            pdf_paths, ParserConfig(), ChunkerConfig(), n_workers=1
        )

        assert results == [(Path("/a.pdf"), None, None, "failed")]


class TestParallelProcessPdfsMultiWorker:
    """paperrag/cli.py forces multiprocessing's start method to 'spawn'
    process-wide (to avoid PyTorch/Docling fork deadlocks), and that choice
    is sticky for the rest of the interpreter once cli.py has been imported
    anywhere - e.g. by tests/test_cli.py, which runs before this file
    alphabetically. Spawned workers are fresh interpreters that re-import
    modules rather than inheriting patched parent state, so monkeypatching
    paperrag.parser.parse_pdf/paperrag.chunker.chunk_paper would silently
    have no effect on the real workers whenever the suite has already
    imported cli.py.

    These tests force a 'fork' context explicitly for parallel_process_pdfs's
    Pool, independent of the process-wide default, so the fakes are reliably
    visible in the worker regardless of what else has run in this suite.
    """

    def test_multiple_workers_process_all_pdfs(self, monkeypatch):
        import multiprocessing
        monkeypatch.setattr("paperrag.parallel.Pool", multiprocessing.get_context("fork").Pool)
        monkeypatch.setattr("paperrag.parser.parse_pdf", _fake_parse_pdf)
        monkeypatch.setattr("paperrag.chunker.chunk_paper", _fake_chunk_paper)

        pdf_paths = [Path("/one.pdf"), Path("/two.pdf"), Path("/three.pdf")]
        results = parallel_process_pdfs(
            pdf_paths, ParserConfig(), ChunkerConfig(), n_workers=2
        )

        assert {r[0] for r in results} == set(pdf_paths)
        for _, file_hash, chunks, error in results:
            assert error is None
            assert file_hash is not None
            assert len(chunks) == 1

    def test_multiple_workers_isolate_per_pdf_failure(self, monkeypatch):
        import multiprocessing
        monkeypatch.setattr("paperrag.parallel.Pool", multiprocessing.get_context("fork").Pool)
        monkeypatch.setattr("paperrag.parser.parse_pdf", _fake_parse_pdf)
        monkeypatch.setattr("paperrag.chunker.chunk_paper", _fake_chunk_paper)

        pdf_paths = [Path("/good.pdf"), Path("/bad.pdf")]
        results = parallel_process_pdfs(
            pdf_paths, ParserConfig(), ChunkerConfig(), n_workers=2
        )
        by_path = {r[0]: r for r in results}

        assert by_path[Path("/good.pdf")][3] is None
        assert by_path[Path("/bad.pdf")][1] is None
        assert "simulated parse failure" in by_path[Path("/bad.pdf")][3]

    def test_timeout_records_failure_without_hanging(self, monkeypatch):
        import multiprocessing
        monkeypatch.setattr("paperrag.parallel.Pool", multiprocessing.get_context("fork").Pool)
        monkeypatch.setattr("paperrag.parser.parse_pdf", _fake_parse_pdf_slow_for_stuck)
        monkeypatch.setattr("paperrag.chunker.chunk_paper", _fake_chunk_paper)

        pdf_paths = [Path("/fast.pdf"), Path("/stuck.pdf")]
        results = parallel_process_pdfs(
            pdf_paths, ParserConfig(), ChunkerConfig(), n_workers=2, timeout=1
        )
        by_path = {r[0]: r for r in results}

        assert by_path[Path("/fast.pdf")][3] is None
        assert "Timeout after 1s" in by_path[Path("/stuck.pdf")][3]
