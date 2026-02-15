"""Tests for the parser module (unit tests that don't require actual PDFs)."""

import hashlib
import tempfile
from pathlib import Path

from paper_rag.parser import compute_file_hash, discover_pdfs


def test_compute_file_hash():
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(b"fake pdf content")
        f.flush()
        path = Path(f.name)

    h = compute_file_hash(path)
    expected = hashlib.sha256(b"fake pdf content").hexdigest()
    assert h == expected
    path.unlink()


def test_compute_file_hash_deterministic():
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(b"hello world")
        f.flush()
        path = Path(f.name)

    assert compute_file_hash(path) == compute_file_hash(path)
    path.unlink()


def test_discover_pdfs_empty(tmp_path):
    pdfs = discover_pdfs(tmp_path)
    assert pdfs == []


def test_discover_pdfs_finds_files(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"")
    (tmp_path / "b.pdf").write_bytes(b"")
    (tmp_path / "c.txt").write_bytes(b"")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "d.pdf").write_bytes(b"")

    pdfs = discover_pdfs(tmp_path)
    assert len(pdfs) == 3
    assert all(p.suffix == ".pdf" for p in pdfs)


def test_discover_pdfs_nonexistent(tmp_path):
    pdfs = discover_pdfs(tmp_path / "does_not_exist")
    assert pdfs == []
