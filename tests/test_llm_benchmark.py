"""Tests for llm_benchmark module - multi-model LLM benchmark harness.

run_benchmark()'s full multi-model loop writes experiment logs under
_PROJECT_ROOT / "experiments" / "logs" and calls out to a live Ollama
instance and retriever/LLM stack. That's out of scope for unit tests here;
we cover the pure helper functions plus the two documented early-exit
paths (Ollama not running, missing index), neither of which reaches the
file-writing code.
"""

import argparse
import json
from unittest.mock import MagicMock, patch

import pytest

from paperrag.llm_benchmark import (
    _generate_results_md,
    _get_hardware_info,
    _has_citation,
    _is_non_empty,
    judge_answer,
    run_benchmark,
)


class TestIsNonEmpty:
    def test_empty_string_is_not_substantive(self):
        assert _is_non_empty("") is False

    def test_whitespace_only_is_not_substantive(self):
        assert _is_non_empty("   \n  ") is False

    def test_refusal_phrase_is_not_substantive(self):
        assert _is_non_empty("I don't know the answer to that.") is False
        assert _is_non_empty("Sorry, I cannot answer this question.") is False

    def test_real_answer_is_substantive(self):
        assert _is_non_empty("The transformer uses self-attention. [1]") is True

    def test_refusal_phrase_is_case_insensitive(self):
        assert _is_non_empty("I DO NOT KNOW.") is False


class TestHasCitation:
    def test_detects_bracketed_number(self):
        assert _has_citation("Attention is all you need [1].") is True

    def test_detects_multiple_citations(self):
        assert _has_citation("See [1] and [23].") is True

    def test_no_citation_returns_false(self):
        assert _has_citation("No citation markers here.") is False

    def test_non_numeric_brackets_not_a_citation(self):
        assert _has_citation("See [Smith et al.]") is False


class TestGetHardwareInfo:
    def test_returns_cpu_and_gpu_keys(self):
        hw = _get_hardware_info()
        assert "cpu" in hw
        assert "gpu" in hw
        assert "ram_gb" in hw

    @patch("psutil.virtual_memory")
    def test_ram_gb_rounded_from_psutil(self, mock_vm):
        mock_vm.return_value = MagicMock(total=8 * 1024**3)
        hw = _get_hardware_info()
        assert hw["ram_gb"] == 8


class TestJudgeAnswer:
    @patch("paperrag.llm_benchmark.generate_answer")
    def test_parses_valid_json_scores(self, mock_generate):
        mock_generate.return_value = 'Some preamble {"relevance": 4, "faithfulness": 5} trailing'

        scores = judge_answer("q", ["ctx"], "answer", judge_model="judge:1b")

        assert scores == {"relevance": 4, "faithfulness": 5}

    @patch("paperrag.llm_benchmark.generate_answer")
    def test_clamps_out_of_range_scores(self, mock_generate):
        mock_generate.return_value = '{"relevance": 9, "faithfulness": 0}'

        scores = judge_answer("q", ["ctx"], "answer", judge_model="judge:1b")

        assert scores == {"relevance": 5, "faithfulness": 1}

    @patch("paperrag.llm_benchmark.generate_answer")
    def test_missing_json_returns_none_scores(self, mock_generate):
        mock_generate.return_value = "The judge refused to respond in JSON."

        scores = judge_answer("q", ["ctx"], "answer", judge_model="judge:1b")

        assert scores == {"relevance": None, "faithfulness": None}

    @patch("paperrag.llm_benchmark.generate_answer")
    def test_generation_exception_returns_none_scores(self, mock_generate):
        mock_generate.side_effect = RuntimeError("judge backend down")

        scores = judge_answer("q", ["ctx"], "answer", judge_model="judge:1b")

        assert scores == {"relevance": None, "faithfulness": None}

    @patch("paperrag.llm_benchmark.generate_answer")
    def test_non_numeric_scores_return_none(self, mock_generate):
        mock_generate.return_value = '{"relevance": "high", "faithfulness": "low"}'

        scores = judge_answer("q", ["ctx"], "answer", judge_model="judge:1b")

        assert scores == {"relevance": None, "faithfulness": None}


class TestGenerateResultsMd:
    def test_writes_table_without_judge(self, tmp_path):
        args = argparse.Namespace(
            benchmark_file="bench.jsonl", top_k=3, max_tokens=256,
            judge_model=None, output_dir=str(tmp_path),
        )
        model_results = [{
            "model": "qwen2.5:1.5b", "avg_latency_ms": 123.4,
            "avg_answer_length": 200.0, "non_empty_rate": 0.9, "citation_rate": 0.8,
        }]
        retrieval_metrics = {"precision@3": 0.75, "recall@3": 0.5, "mrr": 0.6, "avg_latency_ms": 10.0}

        _generate_results_md(model_results, retrieval_metrics, args)

        content = (tmp_path / "results.md").read_text(encoding="utf-8")
        assert "qwen2.5:1.5b" in content
        assert "default model" in content
        assert "Relevance" not in content

    def test_writes_table_with_judge(self, tmp_path):
        args = argparse.Namespace(
            benchmark_file="bench.jsonl", top_k=3, max_tokens=256,
            judge_model="judge:1b", output_dir=str(tmp_path),
        )
        model_results = [{
            "model": "llama3.2:3b", "avg_latency_ms": 200.0,
            "avg_answer_length": 150.0, "non_empty_rate": 1.0, "citation_rate": 0.5,
            "avg_relevance": 4.2, "avg_faithfulness": None,
        }]
        retrieval_metrics = {"precision@3": 0.5, "recall@3": 0.5, "mrr": 0.5, "avg_latency_ms": 5.0}

        _generate_results_md(model_results, retrieval_metrics, args)

        content = (tmp_path / "results.md").read_text(encoding="utf-8")
        assert "Relevance" in content
        assert "4.2" in content
        assert "N/A" in content  # missing faithfulness


class TestRunBenchmarkEarlyExits:
    @patch("paperrag.llm_benchmark._check_ollama_running", return_value=False)
    def test_exits_when_ollama_not_running(self, mock_check, capsys):
        args = argparse.Namespace(index_dir="/tmp/whatever")

        with pytest.raises(SystemExit) as exc_info:
            run_benchmark(args)

        assert exc_info.value.code == 1
        assert "Ollama is not running" in capsys.readouterr().out

    @patch("paperrag.llm_benchmark.Retriever")
    @patch("paperrag.llm_benchmark._check_ollama_running", return_value=True)
    def test_exits_when_index_missing(self, mock_check, mock_retriever, capsys):
        mock_retriever.side_effect = FileNotFoundError("No index found at /tmp/missing.")
        args = argparse.Namespace(index_dir="/tmp/missing")

        with pytest.raises(SystemExit) as exc_info:
            run_benchmark(args)

        assert exc_info.value.code == 1
        assert "No index found" in capsys.readouterr().out
