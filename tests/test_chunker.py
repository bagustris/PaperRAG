"""Tests for chunker module - deterministic section-aware chunking."""

import pytest
from paperrag.chunker import Chunk, chunk_text, chunk_paper, _deterministic_hash
from paperrag.config import ChunkerConfig
from paperrag.parser import ParsedPaper, ParsedSection


class TestDeterministicHash:
    """Test deterministic hash generation."""

    def test_hash_is_deterministic(self):
        """Same inputs should produce same hash."""
        h1 = _deterministic_hash("test text", "file_hash_123", 0)
        h2 = _deterministic_hash("test text", "file_hash_123", 0)
        assert h1 == h2

    def test_hash_different_text(self):
        """Different text should produce different hash."""
        h1 = _deterministic_hash("text A", "file_hash_123", 0)
        h2 = _deterministic_hash("text B", "file_hash_123", 0)
        assert h1 != h2

    def test_hash_different_file_hash(self):
        """Different file hash should produce different hash."""
        h1 = _deterministic_hash("test text", "file_hash_123", 0)
        h2 = _deterministic_hash("test text", "file_hash_456", 0)
        assert h1 != h2

    def test_hash_different_chunk_id(self):
        """Different chunk ID should produce different hash."""
        h1 = _deterministic_hash("test text", "file_hash_123", 0)
        h2 = _deterministic_hash("test text", "file_hash_123", 1)
        assert h1 != h2

    def test_hash_format(self):
        """Hash should be 16-character hex string."""
        h = _deterministic_hash("test", "hash", 0)
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)


class TestChunkText:
    """Test basic text chunking functionality."""

    def test_empty_text(self):
        """Empty text should return empty list."""
        chunks = chunk_text("", 100, 20)
        assert chunks == []

    def test_single_chunk(self):
        """Text shorter than chunk size should return single chunk."""
        text = "short text"
        chunks = chunk_text(text, 100, 20)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_multiple_chunks(self):
        """Text longer than chunk size should be split into multiple chunks."""
        text = "a" * 150
        chunks = chunk_text(text, 100, 20)
        assert len(chunks) == 2
        assert len(chunks[0]) == 100
        # Second chunk will have the remaining characters (50) plus overlap (20) = 70
        assert len(chunks[1]) == 70

    def test_chunk_overlap(self):
        """Chunks should overlap by specified amount."""
        text = "a" * 150
        chunks = chunk_text(text, 100, 20)
        # Second chunk should start at position 80 (100 - 20)
        assert chunks[1][:20] == chunks[0][-20:]

    def test_chunk_overlap_boundaries(self):
        """Overlap should work correctly at boundaries."""
        text = "abcdefghij" * 15  # 150 chars
        chunks = chunk_text(text, 100, 30)
        # With 150 chars, chunk_size=100, overlap=30:
        # First chunk: 0-100, second: 70-170 (but only 80 chars left)
        # Actually: 0-100, 70-150 (80 chars), 120-220 (but only 30 chars left)
        assert len(chunks) == 3
        # Check overlap between first and second
        overlap1 = chunks[0][-30:]
        assert chunks[1][:30] == overlap1

    def test_chunk_deterministic(self):
        """Same input should always produce same chunks."""
        text = "a" * 200
        chunks1 = chunk_text(text, 100, 20)
        chunks2 = chunk_text(text, 100, 20)
        assert chunks1 == chunks2

    def test_chunk_size_exactly_divisible(self):
        """Handle case where text is exactly divisible by (chunk_size - overlap)."""
        text = "a" * 180  # 180 = 2 * (100 - 20)
        chunks = chunk_text(text, 100, 20)
        # With chunk_size=100, overlap=20, start positions: 0, 80, 160
        # Chunks: 0-100, 80-180, 160-180 (20 chars)
        assert len(chunks) == 3
        assert len(chunks[0]) == 100
        assert len(chunks[1]) == 100
        assert len(chunks[2]) == 20

    def test_chunk_size_with_remainder(self):
        """Handle case where last chunk is smaller."""
        text = "a" * 190  # 190 = 2 * (100 - 20) + 10
        chunks = chunk_text(text, 100, 20)
        # Start positions: 0, 80, 160
        # Chunks: 0-100 (100 chars), 80-180 (100 chars), 160-190 (30 chars)
        assert len(chunks) == 3
        assert len(chunks[0]) == 100
        assert len(chunks[1]) == 100
        assert len(chunks[2]) == 30


class TestChunk:
    """Test Chunk dataclass and serialization."""

    def test_chunk_creation(self):
        """Chunk should be created with all required fields."""
        chunk = Chunk(
            chunk_id=0,
            hash_id="abc123",
            text="test text",
            paper_title="Test Paper",
            section_name="Introduction",
            file_path="/path/to/paper.pdf",
            file_hash="sha256hash"
        )
        assert chunk.chunk_id == 0
        assert chunk.hash_id == "abc123"
        assert chunk.text == "test text"

    def test_chunk_to_dict(self):
        """Chunk should serialize to dict correctly."""
        chunk = Chunk(
            chunk_id=0,
            hash_id="abc123",
            text="test text",
            paper_title="Test Paper",
            section_name="Introduction",
            file_path="/path/to/paper.pdf",
            file_hash="sha256hash"
        )
        d = chunk.to_dict()
        assert d["chunk_id"] == 0
        assert d["hash_id"] == "abc123"
        assert d["text"] == "test text"
        assert d["paper_title"] == "Test Paper"
        assert d["section_name"] == "Introduction"
        assert d["file_path"] == "/path/to/paper.pdf"
        assert d["file_hash"] == "sha256hash"

    def test_chunk_from_dict(self):
        """Chunk should deserialize from dict correctly."""
        d = {
            "chunk_id": 0,
            "hash_id": "abc123",
            "text": "test text",
            "paper_title": "Test Paper",
            "section_name": "Introduction",
            "file_path": "/path/to/paper.pdf",
            "file_hash": "sha256hash"
        }
        chunk = Chunk.from_dict(d)
        assert chunk.chunk_id == 0
        assert chunk.hash_id == "abc123"
        assert chunk.text == "test text"

    def test_chunk_roundtrip(self):
        """Chunk should survive roundtrip through dict."""
        original = Chunk(
            chunk_id=5,
            hash_id="xyz789",
            text="roundtrip test",
            paper_title="Roundtrip Paper",
            section_name="Methods",
            file_path="/path/2.pdf",
            file_hash="hash2"
        )
        d = original.to_dict()
        restored = Chunk.from_dict(d)
        assert restored.chunk_id == original.chunk_id
        assert restored.hash_id == original.hash_id
        assert restored.text == original.text


class TestChunkPaper:
    """Test section-aware paper chunking."""

    def test_chunk_single_section(self):
        """Paper with single section should be chunked correctly."""
        paper = ParsedPaper(
            file_path="/path/to/paper.pdf",
            file_hash="abc123",
            title="Test Paper",
            authors="Author One",
            sections=[ParsedSection(name="Abstract", text="This is the abstract text." * 20)]
        )
        chunks = chunk_paper(paper, ChunkerConfig(chunk_size=100, chunk_overlap=20))
        assert len(chunks) > 0
        assert all(c.paper_title == "Test Paper" for c in chunks)
        assert all(c.section_name == "Abstract" for c in chunks)
        assert all(c.file_path == "/path/to/paper.pdf" for c in chunks)
        assert all(c.file_hash == "abc123" for c in chunks)

    def test_chunk_multiple_sections(self):
        """Paper with multiple sections should chunk each independently."""
        paper = ParsedPaper(
            file_path="/path/to/paper.pdf",
            file_hash="abc123",
            title="Test Paper",
            authors="Author One",
            sections=[
                ParsedSection(name="Abstract", text="Abstract text." * 20),
                ParsedSection(name="Introduction", text="Introduction text." * 20),
                ParsedSection(name="Methods", text="Methods text." * 20),
            ]
        )
        chunks = chunk_paper(paper, ChunkerConfig(chunk_size=100, chunk_overlap=20))
        assert len(chunks) > 0
        # Check that chunks have different section names
        section_names = {c.section_name for c in chunks}
        assert section_names == {"Abstract", "Introduction", "Methods"}

    def test_chunk_empty_section(self):
        """Empty sections should be skipped."""
        paper = ParsedPaper(
            file_path="/path/to/paper.pdf",
            file_hash="abc123",
            title="Test Paper",
            authors="Author One",
            sections=[
                ParsedSection(name="Abstract", text=""),
                ParsedSection(name="Introduction", text="Valid text." * 20),
            ]
        )
        chunks = chunk_paper(paper, ChunkerConfig(chunk_size=100, chunk_overlap=20))
        assert len(chunks) > 0
        assert all(c.section_name == "Introduction" for c in chunks)

    def test_chunk_whitespace_only_section(self):
        """Whitespace-only sections should be skipped after stripping."""
        paper = ParsedPaper(
            file_path="/path/to/paper.pdf",
            file_hash="abc123",
            title="Test Paper",
            authors="Author One",
            sections=[
                ParsedSection(name="Abstract", text="   \n  \t  "),
                ParsedSection(name="Introduction", text="Valid text." * 20),
            ]
        )
        chunks = chunk_paper(paper, ChunkerConfig(chunk_size=100, chunk_overlap=20))
        assert len(chunks) > 0
        assert all(c.section_name == "Introduction" for c in chunks)

    def test_chunk_deterministic_ordering(self):
        """Chunking should produce deterministic ordering."""
        paper = ParsedPaper(
            file_path="/path/to/paper.pdf",
            file_hash="abc123",
            title="Test Paper",
            authors="Author One",
            sections=[
                ParsedSection(name="Abstract", text="Text A" * 20),
                ParsedSection(name="Introduction", text="Text B" * 20),
            ]
        )
        chunks1 = chunk_paper(paper, ChunkerConfig(chunk_size=100, chunk_overlap=20))
        chunks2 = chunk_paper(paper, ChunkerConfig(chunk_size=100, chunk_overlap=20))
        
        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1.chunk_id == c2.chunk_id
            assert c1.hash_id == c2.hash_id
            assert c1.text == c2.text
            assert c1.section_name == c2.section_name

    def test_chunk_global_id_increment(self):
        """Chunk IDs should increment globally across sections."""
        paper = ParsedPaper(
            file_path="/path/to/paper.pdf",
            file_hash="abc123",
            title="Test Paper",
            authors="Author One",
            sections=[
                ParsedSection(name="Abstract", text="Text." * 20),
                ParsedSection(name="Introduction", text="Text." * 20),
            ]
        )
        chunks = chunk_paper(paper, ChunkerConfig(chunk_size=100, chunk_overlap=20))
        chunk_ids = [c.chunk_id for c in chunks]
        assert chunk_ids == sorted(chunk_ids)
        assert chunk_ids[-1] == len(chunks) - 1

    def test_chunk_with_default_config(self):
        """Should use default config when none provided."""
        paper = ParsedPaper(
            file_path="/path/to/paper.pdf",
            file_hash="abc123",
            title="Test Paper",
            authors="Author One",
            sections=[ParsedSection(name="Abstract", text="Text." * 20)]
        )
        chunks = chunk_paper(paper)  # No config provided
        assert len(chunks) > 0
        # Should use default chunk_size=1000, chunk_overlap=200
        # With short text, likely only one chunk
        assert all(c.paper_title == "Test Paper" for c in chunks)

    def test_chunk_strips_whitespace(self):
        """Chunk text should be stripped of leading/trailing whitespace."""
        paper = ParsedPaper(
            file_path="/path/to/paper.pdf",
            file_hash="abc123",
            title="Test Paper",
            authors="Author One",
            sections=[ParsedSection(name="Abstract", text="  Text with spaces  ")]
        )
        chunks = chunk_paper(paper, ChunkerConfig(chunk_size=100, chunk_overlap=20))
        assert len(chunks) == 1
        assert chunks[0].text == "Text with spaces"
        assert not chunks[0].text.startswith(" ")
        assert not chunks[0].text.endswith(" ")