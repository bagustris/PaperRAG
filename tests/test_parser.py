"""Tests for parser module - PDF parsing with Docling."""

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from paperrag.parser import (
    ParsedSection,
    ParsedPaper,
    compute_file_hash,
    has_text_layer,
    compute_file_hashes_parallel,
    _normalise_heading,
    load_manifest,
    _extract_title_from_doc,
    SECTION_NAMES,
)


class TestComputeFileHash:
    """Test file hash computation."""

    def test_hash_consistent(self):
        """Same file should produce same hash."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            temp_path = Path(f.name)
        
        try:
            hash1 = compute_file_hash(temp_path)
            hash2 = compute_file_hash(temp_path)
            assert hash1 == hash2
            assert len(hash1) == 64  # SHA256 hex length
        finally:
            temp_path.unlink()

    def test_hash_different_files(self):
        """Different files should produce different hashes."""
        with tempfile.NamedTemporaryFile(delete=False) as f1:
            f1.write(b"content A")
            temp_path1 = Path(f1.name)
        
        with tempfile.NamedTemporaryFile(delete=False) as f2:
            f2.write(b"content B")
            temp_path2 = Path(f2.name)
        
        try:
            hash1 = compute_file_hash(temp_path1)
            hash2 = compute_file_hash(temp_path2)
            assert hash1 != hash2
        finally:
            temp_path1.unlink()
            temp_path2.unlink()

    def test_hash_empty_file(self):
        """Empty file should produce valid hash."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            hash_value = compute_file_hash(temp_path)
            assert len(hash_value) == 64
            assert hash_value == hashlib.sha256(b"").hexdigest()
        finally:
            temp_path.unlink()

    def test_hash_large_file(self):
        """Large file should be hashed correctly."""
        # Create a larger file to test chunking
        large_content = b"x" * (8192 * 10)  # 10 chunks
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(large_content)
            temp_path = Path(f.name)
        
        try:
            hash_value = compute_file_hash(temp_path)
            assert len(hash_value) == 64
            assert hash_value == hashlib.sha256(large_content).hexdigest()
        finally:
            temp_path.unlink()


class TestHasTextLayer:
    """Test text layer detection in PDFs."""

    def test_text_layer_detection_mock(self):
        """Test text layer detection with mocked PyMuPDF."""
        # Create a temporary PDF file (doesn't need to be valid for mock test)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"fake pdf content")
            temp_path = Path(f.name)
        
        try:
            with patch('paperrag.parser.fitz') as mock_fitz:
                mock_doc = MagicMock()
                mock_page = MagicMock()
                mock_page.get_text.return_value = "This is text content" * 10
                mock_doc.__getitem__.return_value = mock_page
                mock_doc.__len__.return_value = 1
                mock_fitz.open.return_value = mock_doc
                
                result = has_text_layer(temp_path, min_chars=50)
                assert result is True
                
                # Test with insufficient text
                mock_page.get_text.return_value = "short"
                result = has_text_layer(temp_path, min_chars=50)
                assert result is False
        finally:
            temp_path.unlink()

    def test_text_layer_empty_pdf(self):
        """Test with empty PDF (no pages)."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"fake pdf")
            temp_path = Path(f.name)
        
        try:
            with patch('paperrag.parser.fitz') as mock_fitz:
                mock_doc = MagicMock()
                mock_doc.__len__.return_value = 0
                mock_fitz.open.return_value = mock_doc
                
                result = has_text_layer(temp_path)
                assert result is False
        finally:
            temp_path.unlink()

    def test_text_layer_error_handling(self):
        """Test error handling when PDF reading fails."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"corrupted pdf")
            temp_path = Path(f.name)
        
        try:
            with patch('paperrag.parser.fitz') as mock_fitz:
                mock_fitz.open.side_effect = Exception("PDF read error")
                
                result = has_text_layer(temp_path)
                assert result is False  # Should return False on error
        finally:
            temp_path.unlink()


class TestComputeFileHashesParallel:
    """Test parallel file hashing."""

    def test_parallel_hashing(self):
        """Test hashing multiple files in parallel."""
        # Create temporary files
        files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(f"content {i}".encode())
                files.append(Path(f.name))
        
        try:
            hashes = compute_file_hashes_parallel(files, n_workers=2)
            assert len(hashes) == 3
            assert all(isinstance(h, str) and len(h) == 64 for h in hashes.values())
            
            # Verify hashes are different
            hash_values = list(hashes.values())
            assert len(set(hash_values)) == 3
        finally:
            for f in files:
                f.unlink()

    def test_parallel_hashing_with_error(self):
        """Test parallel hashing handles errors gracefully."""
        files = []
        # Create valid file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"valid content")
            files.append(Path(f.name))
        
        # Add non-existent file
        files.append(Path("/nonexistent/file.pdf"))
        
        try:
            hashes = compute_file_hashes_parallel(files, n_workers=2)
            # Should only contain hash for valid file
            assert len(hashes) == 1
            assert len(list(hashes.values())[0]) == 64
        finally:
            for f in files:
                if f.exists():
                    f.unlink()


class TestNormaliseHeading:
    """Test section heading normalization."""

    def test_normalise_known_sections(self):
        """Known section names should be normalized."""
        assert _normalise_heading("Abstract") == "Abstract"
        assert _normalise_heading("abstract") == "Abstract"
        assert _normalise_heading("ABSTRACT") == "Abstract"
        assert _normalise_heading("  Abstract  ") == "Abstract"
        assert _normalise_heading("Introduction") == "Introduction"
        assert _normalise_heading("1. Introduction") == "Introduction"
        assert _normalise_heading("Methods") == "Method"  # Normalized to singular
        assert _normalise_heading("Methodology") == "Method"  # Contains "method" 
        assert _normalise_heading("Results") == "Results"  # Stays as Results (matches SECTION_NAMES)
        assert _normalise_heading("Result") == "Result"  # Matches SECTION_NAMES
        assert _normalise_heading("Conclusion") == "Conclusion"

    def test_normalise_unknown_sections(self):
        """Unknown section names should return None."""
        assert _normalise_heading("Random Text") is None
        assert _normalise_heading("Chapter 1") is None
        assert _normalise_heading("Section 2.3") is None
        assert _normalise_heading("") is None

    def test_normalise_section_names_list(self):
        """SECTION_NAMES should contain expected entries."""
        assert "abstract" in SECTION_NAMES
        assert "introduction" in SECTION_NAMES
        assert "methods" in SECTION_NAMES
        assert "results" in SECTION_NAMES
        assert "conclusion" in SECTION_NAMES


class TestLoadManifest:
    """Test manifest loading from CSV."""

    def test_load_valid_manifest(self):
        """Test loading a valid manifest CSV."""
        manifest_content = """filename,title,authors,abstract,doi
paper1.pdf,Test Paper 1,Author One,Abstract text,10.1234/doi1
paper2.pdf,Test Paper 2,Author Two,Another abstract,10.1234/doi2
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(manifest_content)
            temp_path = Path(f.name)
        
        try:
            manifest = load_manifest(temp_path)
            assert len(manifest) == 2
            assert "paper1.pdf" in manifest
            assert manifest["paper1.pdf"]["title"] == "Test Paper 1"
            assert manifest["paper1.pdf"]["authors"] == "Author One"
            assert manifest["paper1.pdf"]["abstract"] == "Abstract text"
            assert manifest["paper1.pdf"]["doi"] == "10.1234/doi1"
        finally:
            temp_path.unlink()

    def test_load_manifest_missing_optional_fields(self):
        """Test manifest with missing optional fields."""
        manifest_content = """filename,title,authors
paper1.pdf,Test Paper 1,Author One
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(manifest_content)
            temp_path = Path(f.name)
        
        try:
            manifest = load_manifest(temp_path)
            assert len(manifest) == 1
            assert manifest["paper1.pdf"]["title"] == "Test Paper 1"
            assert manifest["paper1.pdf"]["abstract"] == ""  # Empty default
            assert manifest["paper1.pdf"]["doi"] == ""  # Empty default
        finally:
            temp_path.unlink()

    def test_load_manifest_empty_filename(self):
        """Test manifest skips entries with empty filename."""
        manifest_content = """filename,title,authors
,Test Paper 1,Author One
paper2.pdf,Test Paper 2,Author Two
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(manifest_content)
            temp_path = Path(f.name)
        
        try:
            manifest = load_manifest(temp_path)
            assert len(manifest) == 1
            assert "paper2.pdf" in manifest
            assert "" not in manifest
        finally:
            temp_path.unlink()

    def test_load_manifest_nonexistent_file(self):
        """Test loading non-existent manifest returns empty dict."""
        manifest = load_manifest(Path("/nonexistent/manifest.csv"))
        assert manifest == {}

    def test_load_manifest_invalid_csv(self):
        """Test loading invalid CSV returns empty dict."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("invalid csv content")
            temp_path = Path(f.name)
        
        try:
            manifest = load_manifest(temp_path)
            assert manifest == {}
        finally:
            temp_path.unlink()


class TestParsedSection:
    """Test ParsedSection dataclass."""

    def test_section_creation(self):
        """Section should be created with required fields."""
        section = ParsedSection(name="Abstract", text="This is the abstract.")
        assert section.name == "Abstract"
        assert section.text == "This is the abstract."


class TestParsedPaper:
    """Test ParsedPaper dataclass."""

    def test_paper_creation(self):
        """Paper should be created with required fields."""
        paper = ParsedPaper(
            file_path="/path/to/paper.pdf",
            file_hash="abc123",
            title="Test Paper",
            authors="Author One"
        )
        assert paper.file_path == "/path/to/paper.pdf"
        assert paper.file_hash == "abc123"
        assert paper.title == "Test Paper"
        assert paper.authors == "Author One"
        assert paper.sections == []
        assert paper.raw_text == ""
        assert paper.abstract == ""
        assert paper.doi == ""

    def test_paper_with_sections(self):
        """Paper should accept sections."""
        sections = [
            ParsedSection(name="Abstract", text="Abstract text"),
            ParsedSection(name="Introduction", text="Introduction text")
        ]
        paper = ParsedPaper(
            file_path="/path/to/paper.pdf",
            file_hash="abc123",
            title="Test Paper",
            authors="Author One",
            sections=sections
        )
        assert len(paper.sections) == 2
        assert paper.sections[0].name == "Abstract"
        assert paper.sections[1].name == "Introduction"


class TestExtractTitleFromDoc:
    """Test title extraction from Docling document."""

    def test_extract_title_from_doc_attribute(self):
        """Test extraction when doc has title attribute."""
        mock_doc = MagicMock()
        mock_doc.title = "Paper Title"
        
        title = _extract_title_from_doc(mock_doc)
        assert title == "Paper Title"

    def test_extract_title_from_doc_none(self):
        """Test extraction when doc title is None."""
        mock_doc = MagicMock()
        mock_doc.title = None
        mock_doc.export_to_markdown.return_value = "# Alternative Title\n\nContent"
        
        title = _extract_title_from_doc(mock_doc)
        assert title == "Alternative Title"

    def test_extract_title_from_markdown_heading(self):
        """Test extraction from markdown when no title attribute."""
        mock_doc = MagicMock()
        mock_doc.title = None
        mock_doc.export_to_markdown.return_value = "# Markdown Title\n\nSome content"
        
        title = _extract_title_from_doc(mock_doc)
        assert title == "Markdown Title"

    def test_extract_title_skips_artifacts(self):
        """Test extraction skips common artifact patterns."""
        mock_doc = MagicMock()
        mock_doc.title = None
        mock_doc.export_to_markdown.return_value = """<!-- Image comment -->
# Real Title
Content here"""
        
        title = _extract_title_from_doc(mock_doc)
        assert title == "Real Title"

    def test_extract_title_fallback(self):
        """Test extraction falls back to first substantial line."""
        mock_doc = MagicMock()
        mock_doc.title = None
        # Use a line that's likely to be accepted as title (no commas, decent length)
        mock_doc.export_to_markdown.return_value = "This is a reasonable title line\nSecond line\nThird line"
        
        title = _extract_title_from_doc(mock_doc)
        assert title == "This is a reasonable title line"