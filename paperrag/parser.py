"""PDF parsing module using Docling."""

from __future__ import annotations

import csv
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

from paperrag.config import ParserConfig

logger = logging.getLogger(__name__)

SECTION_NAMES = [
    "abstract",
    "introduction",
    "background",
    "related work",
    "method",
    "methods",
    "methodology",
    "approach",
    "experiment",
    "experiments",
    "results",
    "result",
    "discussion",
    "conclusion",
    "conclusions",
    "references",
    "acknowledgement",
    "acknowledgements",
    "appendix",
]


@dataclass
class ParsedSection:
    """A single section extracted from a PDF."""

    name: str
    text: str


@dataclass
class ParsedPaper:
    """Structured representation of a parsed PDF."""

    file_path: str
    file_hash: str
    title: str
    authors: str
    sections: list[ParsedSection] = field(default_factory=list)
    raw_text: str = ""
    abstract: str = ""
    doi: str = ""


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def has_text_layer(pdf_path: Path, min_chars: int = 100) -> bool:
    """Detect if PDF has extractable text (i.e., not a scanned image).
    
    Args:
        pdf_path: Path to PDF file
        min_chars: Minimum characters on first page to consider text-based
        
    Returns:
        True if PDF has text layer, False if likely scanned/image-based
    """
    try:
        # Open PDF and check first page only (fast)
        doc = fitz.open(pdf_path)
        if len(doc) == 0:
            return False
        
        # Extract text from first page
        first_page = doc[0]
        text = first_page.get_text().strip()
        doc.close()
        
        # If first page has substantial text, assume text-based PDF
        return len(text) >= min_chars
    except Exception as e:
        logger.warning("Failed to detect text layer in %s: %s", pdf_path.name, e)
        # On error, assume needs OCR (safe default)
        return False


def compute_file_hashes_parallel(pdf_paths: list[Path], n_workers: int = 4) -> dict[str, str]:
    """Compute hashes for multiple PDFs in parallel.

    Args:
        pdf_paths: List of PDF file paths to hash
        n_workers: Number of parallel worker threads

    Returns:
        Dictionary mapping str(pdf_path) to hash string
    """
    hashes = {}
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_path = {
            executor.submit(compute_file_hash, pdf): pdf
            for pdf in pdf_paths
        }
        for future in as_completed(future_to_path):
            pdf = future_to_path[future]
            try:
                hashes[str(pdf)] = future.result()
            except Exception as e:
                logger.error("Failed to hash %s: %s", pdf.name, e)
    return hashes


def _normalise_heading(heading: str) -> str | None:
    """Return normalised section name if it matches a known heading, else None."""
    lower = heading.strip().lower()
    for sec in SECTION_NAMES:
        if sec in lower:
            return sec.title()
    return None


def load_manifest(manifest_path: Path) -> dict[str, dict[str, str]]:
    """Load CSV manifest with metadata to skip parsing.
    
    Expected columns: filename, title, authors, abstract (optional), doi (optional)
    Returns dict: {filename: {title, authors, abstract, doi}}
    """
    manifest: dict[str, dict[str, str]] = {}
    
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get('filename', '').strip()
                if not filename:
                    continue
                
                manifest[filename] = {
                    'title': row.get('title', 'Unknown').strip(),
                    'authors': row.get('authors', 'Unknown').strip(),
                    'abstract': row.get('abstract', '').strip(),
                    'doi': row.get('doi', '').strip(),
                }
        
        logger.info("Loaded manifest with %d entries from %s", len(manifest), manifest_path)
    except Exception as e:
        logger.warning("Failed to load manifest %s: %s", manifest_path, e)
    
    return manifest


def _extract_title_from_doc(doc) -> str:
    """Best-effort title extraction from a Docling document."""
    if hasattr(doc, "title") and doc.title:
        return str(doc.title).strip()

    md = doc.export_to_markdown()
    
    # Skip common non-title patterns
    skip_patterns = [
        "<!---",
        "--->",
        "<!--",
        "-->",
        "image",
        "figure",
    ]
    
    for line in md.splitlines():
        stripped = line.strip()
        
        # Skip empty or artifact lines
        if not stripped or any(pat in stripped.lower() for pat in skip_patterns):
            continue
            
        # Prefer lines with "#" heading markers
        if stripped.startswith("# "):
            title = stripped.lstrip("# ").strip()
            # Skip if it looks like metadata (has comma-separated names)
            if "," not in title or len(title) > 100:
                return title[:200]
        
        # If no heading found, take first substantial line
        # Skip if it looks like author list (comma-separated, all caps/mixed case)
        if len(stripped) > 10 and not stripped.startswith("#"):
            # Heuristic: if line has multiple commas, likely authors, skip it
            comma_count = stripped.count(",")
            if comma_count == 0 or (comma_count <= 2 and len(stripped) > 50):
                return stripped[:200]
    
    return "Unknown"


def _extract_authors_from_doc(doc) -> str:
    """Best-effort author extraction from Docling document."""
    if hasattr(doc, "authors") and doc.authors:
        if isinstance(doc.authors, list):
            return ", ".join(str(a) for a in doc.authors)
        return str(doc.authors).strip()
    
    # Try to extract from markdown
    md = doc.export_to_markdown()
    for line in md.splitlines()[:10]:  # Check first 10 lines
        stripped = line.strip()
        # Look for lines with multiple commas (author lists)
        if stripped and "," in stripped and not stripped.startswith("#"):
            # Heuristic: if 2+ commas and no year pattern, likely authors
            if stripped.count(",") >= 2 and not any(c.isdigit() for c in stripped[:20]):
                return stripped[:300]
    
    return "Unknown"


def _extract_abstract_from_sections(sections: list[ParsedSection]) -> str:
    """Extract abstract section if present."""
    for sec in sections:
        if sec.name.lower() == "abstract":
            return sec.text[:1000]  # Limit to 1000 chars
    return ""


def parse_pdf(path: Path, config: ParserConfig | None = None, manifest: dict[str, dict[str, str]] | None = None) -> ParsedPaper:
    """Parse a single PDF using Docling and return structured output.
    
    Args:
        path: Path to PDF file
        config: Parser configuration
        manifest: Optional manifest dict for fast metadata lookup
    """
    config = config or ParserConfig()
    file_hash = compute_file_hash(path)
    
    # Check manifest for pre-extracted metadata
    manifest_entry = None
    if manifest:
        manifest_entry = manifest.get(path.name)
        if manifest_entry:
            logger.info("Using manifest metadata for %s", path.name)

    # Determine if OCR should be used
    should_use_ocr = False
    ocr_status = ""
    
    if config.ocr_mode == "always":
        should_use_ocr = True
        ocr_status = " [OCR: forced]"
    elif config.ocr_mode == "never":
        should_use_ocr = False
        ocr_status = " [OCR: disabled]"
    else:  # auto mode
        has_text = has_text_layer(path)
        should_use_ocr = not has_text  # Only use OCR if no text found
        if has_text:
            ocr_status = " [OCR: skipped, text detected]"
        else:
            ocr_status = " [OCR: enabled, scanned PDF]"
    
    logger.info("Parsing %s (hash=%s)%s", path.name, file_hash[:12], ocr_status)

    try:
        if not should_use_ocr:
            # Text-based PDF: disable OCR for speed
            pipeline_options = PdfPipelineOptions(
                do_ocr=False,
                do_table_structure=False,
            )
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options
                    )
                }
            )
        else:
            # Scanned PDF: enable full OCR processing
            converter = DocumentConverter()
        
        result = converter.convert(str(path))
        doc = result.document
    except Exception as exc:
        logger.error("Failed to parse %s: %s", path.name, exc)
        # Use manifest metadata if available even on error
        if manifest_entry:
            return ParsedPaper(
                file_path=str(path),
                file_hash=file_hash,
                title=manifest_entry.get('title', path.stem),
                authors=manifest_entry.get('authors', ''),
                raw_text="",
                abstract=manifest_entry.get('abstract', ''),
                doi=manifest_entry.get('doi', ''),
            )
        return ParsedPaper(
            file_path=str(path),
            file_hash=file_hash,
            title=path.stem,
            authors="",
            raw_text="",
            abstract="",
            doi="",
        )

    # Use manifest metadata if available, otherwise extract
    if manifest_entry:
        title = manifest_entry.get('title', 'Unknown')
        authors = manifest_entry.get('authors', 'Unknown')
        abstract = manifest_entry.get('abstract', '')
        doi = manifest_entry.get('doi', '')
    else:
        title = _extract_title_from_doc(doc)
        authors = _extract_authors_from_doc(doc)
        abstract = ""  # Will extract from sections
        doi = ""

    md_text = doc.export_to_markdown()

    sections: list[ParsedSection] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    for line in md_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            heading_text = stripped.lstrip("#").strip()
            normalised = _normalise_heading(heading_text)
            if normalised:
                if current_heading and current_lines:
                    sections.append(
                        ParsedSection(
                            name=current_heading,
                            text="\n".join(current_lines).strip(),
                        )
                    )
                current_heading = normalised
                current_lines = []
                continue
        current_lines.append(line)

    if current_heading and current_lines:
        sections.append(
            ParsedSection(
                name=current_heading,
                text="\n".join(current_lines).strip(),
            )
        )

    raw_text = md_text

    if not sections and config.fallback_to_raw and raw_text.strip():
        logger.warning("No sections extracted from %s — using raw text fallback", path.name)
        sections = [ParsedSection(name="Full Text", text=raw_text.strip())]
    
    # Extract abstract from sections if not from manifest
    if not abstract and sections:
        abstract = _extract_abstract_from_sections(sections)

    return ParsedPaper(
        file_path=str(path),
        file_hash=file_hash,
        title=title,
        authors=authors,
        sections=sections,
        raw_text=raw_text,
        abstract=abstract,
        doi=doi,
    )


def discover_pdfs(input_dir: Path) -> list[Path]:
    """Recursively find all PDF files under *input_dir*."""
    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        return []
    pdfs = sorted(input_dir.rglob("*.pdf"))
    logger.info("Discovered %d PDF(s) in %s", len(pdfs), input_dir)
    return pdfs
