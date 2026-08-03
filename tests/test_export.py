"""Tests for export module - saving retrieval results in various formats."""

import csv
import json

import pytest

from paperrag.export import export_results, load_exported_data
from paperrag.retriever import RetrievalResult


def _sample_results():
    return [
        RetrievalResult(
            text="First chunk text\nwith a newline.",
            score=0.9532,
            paper_title="Paper One",
            section_name="Abstract",
            file_path="/papers/one.pdf",
            chunk_id=0,
        ),
        RetrievalResult(
            text="Second chunk text.",
            score=0.8123,
            paper_title="Paper Two",
            section_name="Introduction",
            file_path="/papers/two.pdf",
            chunk_id=1,
        ),
    ]


class TestExportCSV:
    def test_export_csv_writes_header_and_rows(self, tmp_path):
        out = tmp_path / "results.csv"
        export_results(_sample_results(), out, format="csv")

        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))

        assert rows[0] == [
            "Score", "Paper Title", "Section Name", "File Path", "Chunk ID", "Text Snippet",
        ]
        assert rows[1][0] == "0.953200"
        assert rows[1][1] == "Paper One"
        assert rows[2][4] == "1"

    def test_export_csv_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "nested" / "dir" / "results.csv"
        export_results(_sample_results(), out, format="csv")
        assert out.exists()

    def test_export_csv_empty_results(self, tmp_path):
        out = tmp_path / "empty.csv"
        export_results([], out, format="csv")
        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 1  # header only


class TestExportMarkdown:
    def test_export_markdown_contains_expected_sections(self, tmp_path):
        out = tmp_path / "results.md"
        export_results(_sample_results(), out, format="markdown")

        content = out.read_text(encoding="utf-8")
        assert "# Query Results - results" in content
        assert "**Total Results:** 2" in content
        assert "## Result 1" in content
        assert "## Result 2" in content
        assert "Paper One" in content
        assert "0.953200" in content

    def test_export_markdown_indents_multiline_text(self, tmp_path):
        out = tmp_path / "results.md"
        export_results(_sample_results(), out, format="markdown")
        content = out.read_text(encoding="utf-8")
        assert "First chunk text\n  with a newline." in content


class TestExportJSON:
    def test_export_json_roundtrip(self, tmp_path):
        out = tmp_path / "results.json"
        export_results(_sample_results(), out, format="json")

        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["total_results"] == 2
        assert data["results"][0]["paper_title"] == "Paper One"
        assert data["results"][1]["score"] == pytest.approx(0.8123)

    def test_export_json_is_loadable_via_load_exported_data(self, tmp_path):
        out = tmp_path / "results.json"
        export_results(_sample_results(), out, format="json")

        loaded = load_exported_data(out)
        assert loaded["total_results"] == 2
        assert len(loaded["results"]) == 2


class TestExportFormatHandling:
    def test_export_format_case_insensitive(self, tmp_path):
        out = tmp_path / "results.json"
        export_results(_sample_results(), out, format="JSON")
        assert json.loads(out.read_text(encoding="utf-8"))["total_results"] == 2

    def test_export_unsupported_format_raises(self, tmp_path):
        out = tmp_path / "results.xyz"
        with pytest.raises(ValueError, match="Unsupported format"):
            export_results(_sample_results(), out, format="xyz")

    def test_export_default_format_is_markdown(self, tmp_path):
        out = tmp_path / "results.md"
        export_results(_sample_results(), out)
        assert out.read_text(encoding="utf-8").startswith("# Query Results")


class TestLoadExportedData:
    def test_load_unsupported_extension_raises(self, tmp_path):
        bogus = tmp_path / "results.csv"
        bogus.write_text("Score,Paper Title\n0.9,Paper One\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Unable to load unknown format"):
            load_exported_data(bogus)
