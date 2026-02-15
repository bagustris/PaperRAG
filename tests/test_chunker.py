"""Tests for the chunker module."""

from paper_rag.chunker import Chunk, chunk_paper, chunk_text
from paper_rag.config import ChunkerConfig
from paper_rag.parser import ParsedPaper, ParsedSection


def test_chunk_text_basic():
    text = "A" * 2500
    pieces = chunk_text(text, chunk_size=1000, chunk_overlap=200)
    assert len(pieces) == 4
    assert pieces[0] == "A" * 1000
    assert all(len(p) <= 1000 for p in pieces)


def test_chunk_text_empty():
    assert chunk_text("", 1000, 200) == []


def test_chunk_text_short():
    pieces = chunk_text("hello", 1000, 200)
    assert pieces == ["hello"]


def test_chunk_text_deterministic():
    text = "This is a test document with enough content to chunk. " * 30
    a = chunk_text(text, 1000, 200)
    b = chunk_text(text, 1000, 200)
    assert a == b


def test_chunk_paper_sections():
    paper = ParsedPaper(
        file_path="/tmp/test.pdf",
        file_hash="abc123",
        title="Test Paper",
        authors="Author",
        sections=[
            ParsedSection(name="Abstract", text="Short abstract."),
            ParsedSection(name="Introduction", text="Longer intro. " * 100),
        ],
        raw_text="",
    )
    chunks = chunk_paper(paper, ChunkerConfig(chunk_size=500, chunk_overlap=100))
    assert len(chunks) >= 2
    assert chunks[0].section_name == "Abstract"
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.paper_title == "Test Paper" for c in chunks)
    assert all(c.file_hash == "abc123" for c in chunks)


def test_chunk_ids_are_sequential():
    paper = ParsedPaper(
        file_path="/tmp/test.pdf",
        file_hash="def456",
        title="T",
        authors="",
        sections=[ParsedSection(name="Body", text="word " * 500)],
        raw_text="",
    )
    chunks = chunk_paper(paper)
    ids = [c.chunk_id for c in chunks]
    assert ids == list(range(len(chunks)))


def test_chunk_hash_deterministic():
    paper = ParsedPaper(
        file_path="/tmp/test.pdf",
        file_hash="hash1",
        title="T",
        authors="",
        sections=[ParsedSection(name="Body", text="content " * 200)],
        raw_text="",
    )
    a = chunk_paper(paper)
    b = chunk_paper(paper)
    assert [c.hash_id for c in a] == [c.hash_id for c in b]


def test_chunk_to_dict_roundtrip():
    c = Chunk(
        chunk_id=0,
        hash_id="abc",
        text="hello",
        paper_title="P",
        section_name="S",
        file_path="/f.pdf",
        file_hash="h",
    )
    d = c.to_dict()
    c2 = Chunk.from_dict(d)
    assert c == c2
