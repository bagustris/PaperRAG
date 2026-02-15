"""Parallel PDF processing utilities."""

from __future__ import annotations

import logging
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paper_rag.chunker import Chunk
    from paper_rag.config import ChunkerConfig, ParserConfig

logger = logging.getLogger(__name__)


def process_single_pdf(
    pdf_path: Path,
    parser_config: ParserConfig,
    chunker_config: ChunkerConfig
) -> tuple[Path, str | None, list[Chunk] | None, str | None]:
    """Process one PDF: parse + chunk (NOT embed yet).

    Args:
        pdf_path: Path to PDF file
        parser_config: Parser configuration
        chunker_config: Chunker configuration

    Returns:
        Tuple of (pdf_path, file_hash, chunks, error_message)
        If successful: (pdf_path, file_hash, chunks, None)
        If failed: (pdf_path, None, None, error_message)
    """
    try:
        from paper_rag.chunker import chunk_paper
        from paper_rag.parser import parse_pdf

        paper = parse_pdf(pdf_path, parser_config)
        chunks = chunk_paper(paper, chunker_config)
        return (pdf_path, paper.file_hash, chunks, None)
    except Exception as e:
        logger.error("Failed to process %s: %s", pdf_path.name, e)
        return (pdf_path, None, None, str(e))


def parallel_process_pdfs(
    pdf_paths: list[Path],
    parser_config: ParserConfig,
    chunker_config: ChunkerConfig,
    n_workers: int,
    timeout: int = 0
) -> list[tuple[Path, str | None, list[Chunk] | None, str | None]]:
    """Process PDFs in parallel, return parsed results.

    Args:
        pdf_paths: List of PDF paths to process
        parser_config: Parser configuration
        chunker_config: Chunker configuration
        n_workers: Number of worker processes
        timeout: Timeout in seconds per PDF (0 = no timeout)

    Returns:
        List of tuples: (pdf_path, file_hash, chunks, error_message)
    """
    if n_workers == 1:
        # Single-threaded mode (for debugging or compatibility)
        logger.info("Processing PDFs in single-threaded mode")
        return [
            process_single_pdf(pdf, parser_config, chunker_config)
            for pdf in pdf_paths
        ]

    logger.info("Processing PDFs with %d workers (timeout=%ds)", n_workers, timeout)
    # maxtasksperchild=1 ensures workers are killed after each task, releasing memory (crucial for heavy ML models)
    with Pool(processes=n_workers, maxtasksperchild=1) as pool:
        process_fn = partial(
            process_single_pdf,
            parser_config=parser_config,
            chunker_config=chunker_config
        )
        
        if timeout > 0:
            # Use map_async with timeout for individual PDFs
            results = []
            for pdf_path in pdf_paths:
                try:
                    async_result = pool.apply_async(process_fn, (pdf_path,))
                    result = async_result.get(timeout=timeout)
                    results.append(result)
                except Exception as e:
                    logger.error("Timeout or error processing %s: %s", pdf_path.name, e)
                    results.append((pdf_path, None, None, f"Timeout after {timeout}s: {str(e)}"))
            return results
        else:
            # No timeout, use regular map
            results = pool.map(process_fn, pdf_paths)
            return results

