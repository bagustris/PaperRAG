"""Export module for saving retrieval results in various formats."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from paperrag.retriever import RetrievalResult

logger = logging.getLogger(__name__)


def export_results(
    query_results: list[RetrievalResult],
    output_path: Path,
    format: str = "markdown",
) -> None:
    """Export retrieval results to different file formats.

    Args:
        query_results: List of RetrievalResult objects from the retriever.
        output_path: Path object for the output file.
        format: Output format - 'csv', 'markdown', or 'json'. Default: 'markdown'.

    Raises:
        ValueError: If the specified format is not supported.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    format_lower = format.lower()

    if format_lower == "csv":
        _export_csv(query_results, output_path)
    elif format_lower == "markdown":
        _export_markdown(query_results, output_path)
    elif format_lower == "json":
        _export_json(query_results, output_path)
    else:
        raise ValueError(
            f"Unsupported format: {format}. Supported formats: 'csv', 'markdown', 'json'"
        )


def _export_csv(query_results: list[RetrievalResult], output_path: Path) -> None:
    """Export results to CSV format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "Score",
                "Paper Title",
                "Section Name",
                "File Path",
                "Chunk ID",
                "Text Snippet",
            ]
        )

        for result in query_results:
            writer.writerow(
                [
                    f"{result.score:.6f}",
                    result.paper_title,
                    result.section_name,
                    result.file_path,
                    result.chunk_id,
                    result.text,
                ]
            )

    logger.info("Exported %d results to CSV: %s", len(query_results), output_path)


def _export_markdown(query_results: list[RetrievalResult], output_path: Path) -> None:
    """Export results to Markdown format."""
    filename = output_path.stem
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Query Results - {filename}\n\n")
        f.write(f"**Total Results:** {len(query_results)}\n\n")
        f.write("---\n\n")

        for i, result in enumerate(query_results, start=1):
            f.write(f"## Result {i}\n\n")
            f.write(f"**Score:** `{result.score:.6f}`\n\n")
            f.write(f"**Paper:** {result.paper_title}\n\n")
            f.write(f"**Section:** {result.section_name}\n\n")
            f.write(f"**File:** `{result.file_path}`\n\n")
            f.write(f"**Chunk ID:** `{result.chunk_id}`\n\n")
            f.write(f"**Text:**\n\n")

            text_snippet = result.text.replace("\n", "\n  ")
            f.write(f"```\n{text_snippet}\n```\n\n")

            f.write("---\n\n")

    logger.info("Exported %d results to Markdown: %s", len(query_results), output_path)


def _export_json(query_results: list[RetrievalResult], output_path: Path) -> None:
    """Export results to JSON format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "query_time": None,
        "total_results": len(query_results),
        "results": [
            {
                "text": result.text,
                "score": result.score,
                "paper_title": result.paper_title,
                "section_name": result.section_name,
                "file_path": result.file_path,
                "chunk_id": result.chunk_id,
            }
            for result in query_results
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("Exported %d results to JSON: %s", len(query_results), output_path)


def load_exported_data(file_path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    """Load data back from an exported file.

    Args:
        file_path: Path to the exported file.

    Returns:
        Parsed data from the file. For JSON files, returns dict or list.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(
            f"Unable to load unknown format: {suffix}. Only JSON format is supported for loading."
        )


if __name__ == "__main__":
    logger.info(
        "Export module - provide file export functionality for retrieval results"
    )
