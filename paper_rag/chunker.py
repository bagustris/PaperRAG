"""Section-aware deterministic chunking module."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

from paper_rag.config import ChunkerConfig
from paper_rag.parser import ParsedPaper

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single text chunk with full provenance metadata."""

    chunk_id: int
    hash_id: str
    text: str
    paper_title: str
    section_name: str
    file_path: str
    file_hash: str

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "hash_id": self.hash_id,
            "text": self.text,
            "paper_title": self.paper_title,
            "section_name": self.section_name,
            "file_path": self.file_path,
            "file_hash": self.file_hash,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Chunk:
        return cls(**d)


def _deterministic_hash(text: str, file_hash: str, chunk_id: int) -> str:
    """Produce a deterministic hash for a chunk."""
    payload = f"{file_hash}:{chunk_id}:{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split *text* into overlapping windows of characters.

    Deterministic: same input always produces the same output list.
    """
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


def chunk_paper(paper: ParsedPaper, config: ChunkerConfig | None = None) -> list[Chunk]:
    """Chunk a parsed paper into a list of Chunk objects.

    Chunking is section-aware: each section is chunked independently and
    chunks carry the section name in their metadata.  Ordering is
    deterministic.
    """
    config = config or ChunkerConfig()
    all_chunks: list[Chunk] = []
    global_id = 0

    for section in paper.sections:
        text_pieces = chunk_text(section.text, config.chunk_size, config.chunk_overlap)
        for piece in text_pieces:
            piece = piece.strip()
            if not piece:
                continue
            h = _deterministic_hash(piece, paper.file_hash, global_id)
            all_chunks.append(
                Chunk(
                    chunk_id=global_id,
                    hash_id=h,
                    text=piece,
                    paper_title=paper.title,
                    section_name=section.name,
                    file_path=paper.file_path,
                    file_hash=paper.file_hash,
                )
            )
            global_id += 1

    logger.info(
        "Chunked '%s' into %d chunks (sections=%d)",
        paper.title[:60],
        len(all_chunks),
        len(paper.sections),
    )
    return all_chunks
