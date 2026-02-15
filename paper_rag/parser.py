"""PDF parsing module using Docling."""

from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from docling.document_converter import DocumentConverter

from paper_rag.config import ParserConfig

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


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


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


def parse_pdf(path: Path, config: ParserConfig | None = None) -> ParsedPaper:
    """Parse a single PDF using Docling and return structured output."""
    config = config or ParserConfig()
    file_hash = compute_file_hash(path)

    logger.info("Parsing %s (hash=%s)", path.name, file_hash[:12])

    try:
        converter = DocumentConverter()
        result = converter.convert(str(path))
        doc = result.document
    except Exception as exc:
        logger.error("Failed to parse %s: %s", path.name, exc)
        return ParsedPaper(
            file_path=str(path),
            file_hash=file_hash,
            title=path.stem,
            authors="",
            raw_text="",
        )

    title = _extract_title_from_doc(doc)
    authors = ""

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

    return ParsedPaper(
        file_path=str(path),
        file_hash=file_hash,
        title=title,
        authors=authors,
        sections=sections,
        raw_text=raw_text,
    )


def discover_pdfs(input_dir: Path) -> list[Path]:
    """Recursively find all PDF files under *input_dir*."""
    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        return []
    pdfs = sorted(input_dir.rglob("*.pdf"))
    logger.info("Discovered %d PDF(s) in %s", len(pdfs), input_dir)
    return pdfs
