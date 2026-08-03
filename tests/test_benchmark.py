"""Tests for benchmark module - retrieval evaluation metrics."""

import json

import pytest

from paperrag.benchmark import (
    _normalize,
    evaluate,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestNormalize:
    def test_normalize_strips_directory(self):
        assert _normalize("/a/b/c/paper.pdf") == "paper.pdf"

    def test_normalize_already_basename(self):
        assert _normalize("paper.pdf") == "paper.pdf"


class TestPrecisionAtK:
    def test_all_hits(self):
        retrieved = ["a.pdf", "b.pdf", "c.pdf"]
        relevant = ["a.pdf", "b.pdf", "c.pdf"]
        assert precision_at_k(retrieved, relevant, k=3) == 1.0

    def test_no_hits(self):
        retrieved = ["a.pdf", "b.pdf"]
        relevant = ["x.pdf", "y.pdf"]
        assert precision_at_k(retrieved, relevant, k=2) == 0.0

    def test_partial_hits(self):
        retrieved = ["a.pdf", "x.pdf"]
        relevant = ["a.pdf", "y.pdf"]
        assert precision_at_k(retrieved, relevant, k=2) == 0.5

    def test_uses_only_top_k(self):
        retrieved = ["a.pdf", "b.pdf", "c.pdf"]
        relevant = ["c.pdf"]
        # c.pdf is relevant but outside top_k=2
        assert precision_at_k(retrieved, relevant, k=2) == 0.0

    def test_normalizes_paths(self):
        retrieved = ["/dir1/a.pdf"]
        relevant = ["/dir2/a.pdf"]
        assert precision_at_k(retrieved, relevant, k=1) == 1.0


class TestRecallAtK:
    def test_all_relevant_found(self):
        retrieved = ["a.pdf", "b.pdf"]
        relevant = ["a.pdf", "b.pdf"]
        assert recall_at_k(retrieved, relevant, k=2) == 1.0

    def test_some_relevant_missing(self):
        retrieved = ["a.pdf"]
        relevant = ["a.pdf", "b.pdf"]
        assert recall_at_k(retrieved, relevant, k=1) == 0.5

    def test_empty_relevant_set_returns_zero(self):
        assert recall_at_k(["a.pdf"], [], k=1) == 0.0


class TestReciprocalRank:
    def test_first_result_relevant(self):
        assert reciprocal_rank(["a.pdf", "b.pdf"], ["a.pdf"]) == 1.0

    def test_second_result_relevant(self):
        assert reciprocal_rank(["a.pdf", "b.pdf"], ["b.pdf"]) == pytest.approx(0.5)

    def test_no_relevant_found(self):
        assert reciprocal_rank(["a.pdf", "b.pdf"], ["c.pdf"]) == 0.0


class TestEvaluate:
    def test_evaluate_aggregates_metrics(self, tmp_path):
        benchmark_file = tmp_path / "bench.jsonl"
        benchmark_file.write_text(
            "\n".join([
                json.dumps({"question": "q1", "relevant_documents": ["a.pdf"]}),
                json.dumps({"question": "q2", "relevant_documents": ["b.pdf"]}),
            ]),
            encoding="utf-8",
        )

        responses = {
            "q1": ["a.pdf", "x.pdf"],
            "q2": ["y.pdf", "z.pdf"],
        }

        results = evaluate(str(benchmark_file), lambda q: responses[q], k=2)

        assert results["precision@2"] == pytest.approx((0.5 + 0.0) / 2)
        assert results["recall@2"] == pytest.approx((1.0 + 0.0) / 2)
        assert results["mrr"] == pytest.approx((1.0 + 0.0) / 2)
        assert results["avg_latency_ms"] >= 0.0

    def test_evaluate_skips_blank_lines(self, tmp_path):
        benchmark_file = tmp_path / "bench.jsonl"
        benchmark_file.write_text(
            json.dumps({"question": "q1", "relevant_documents": ["a.pdf"]}) + "\n\n",
            encoding="utf-8",
        )
        results = evaluate(str(benchmark_file), lambda q: ["a.pdf"], k=1)
        assert results["precision@1"] == 1.0
