#!/usr/bin/env python3
"""Multi-model LLM benchmark for PaperRAG.

Compares how different small models perform on RAG questions using
a shared retrieval index and standardized evaluation metrics.

Usage::

    python -m paperrag.llm_benchmark --index-dir /tmp/test-index
    python -m paperrag.llm_benchmark --index-dir /tmp/test-index --models qwen2.5:1.5b,llama3.2:3b
    python -m paperrag.llm_benchmark --index-dir /tmp/test-index --judge-model llama3.3:70b
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from paperrag.benchmark import evaluate as evaluate_retrieval
from paperrag.config import LLMConfig, PaperRAGConfig
from paperrag.llm import (
    _check_ollama_model_available,
    describe_llm_error,
    generate_answer,
)
from paperrag.retriever import Retriever

DEFAULT_MODELS = [
    "qwen2.5:1.5b",
    "llama3.2:3b",
    "qwen3:1.7b",
    "phi4-mini-reasoning:latest",
    "gemma3:1b",
]

DEFAULT_BENCHMARK_FILE = str(Path(__file__).parent / "sample_questions.jsonl")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_JUDGE_PROMPT = """\
Score the following answer on two criteria (1-5 each):
- Relevance: Does the answer address the question?
- Faithfulness: Is the answer supported by the provided context?

Question: {question}

Context: {context}

Answer: {answer}

Respond with JSON only: {{"relevance": <1-5>, "faithfulness": <1-5>}}"""

_CITATION_PATTERN = re.compile(r"\[\d+\]")
_NO_ANSWER_PHRASES = [
    "i don't know",
    "i do not know",
    "no context available",
    "not in context",
    "cannot answer",
    "no relevant information",
]


def _is_non_empty(answer: str) -> bool:
    """Return True if the answer is a substantive response."""
    lower = answer.strip().lower()
    if not lower:
        return False
    return not any(phrase in lower for phrase in _NO_ANSWER_PHRASES)


def _has_citation(answer: str) -> bool:
    """Return True if the answer contains citation markers like [1], [2]."""
    return bool(_CITATION_PATTERN.search(answer))


def _get_hardware_info() -> dict:
    """Collect basic hardware information."""
    hw: dict = {"cpu": platform.processor() or platform.machine(), "gpu": None}
    try:
        import psutil
        hw["ram_gb"] = round(psutil.virtual_memory().total / (1024**3))
    except ImportError:
        hw["ram_gb"] = None
    return hw


def _check_ollama_running() -> bool:
    """Check if Ollama service is reachable."""
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def judge_answer(
    question: str,
    context_chunks: list[str],
    answer: str,
    judge_model: str,
) -> dict[str, int | None]:
    """Score an answer using an LLM judge.

    Returns dict with 'relevance' and 'faithfulness' scores (1-5),
    or None values on parse failure.
    """
    context = "\n\n".join(context_chunks)
    prompt = _JUDGE_PROMPT.format(question=question, context=context, answer=answer)

    judge_config = LLMConfig(model_name=judge_model, temperature=0.0, max_tokens=64)
    try:
        response = generate_answer(
            question=prompt,
            context_chunks=[""],  # prompt already contains context
            config=judge_config,
        )
        match = re.search(r"\{[^}]+\}", response)
        if match:
            scores = json.loads(match.group())
            relevance = scores.get("relevance")
            faithfulness = scores.get("faithfulness")
            if isinstance(relevance, (int, float)) and isinstance(faithfulness, (int, float)):
                return {
                    "relevance": max(1, min(5, int(relevance))),
                    "faithfulness": max(1, min(5, int(faithfulness))),
                }
    except Exception:
        pass
    return {"relevance": None, "faithfulness": None}


def run_benchmark(args: argparse.Namespace) -> None:
    """Execute the multi-model benchmark."""
    if not _check_ollama_running():
        print("ERROR: Ollama is not running. Start it with: ollama serve")
        sys.exit(1)

    cfg = PaperRAGConfig()
    cfg.index_dir = args.index_dir

    try:
        retriever = Retriever(cfg)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    benchmark_file = args.benchmark_file
    questions = []
    with open(benchmark_file) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    if not questions:
        print("ERROR: No questions found in benchmark file.")
        sys.exit(1)

    print(f"Loaded {len(questions)} questions from {benchmark_file}")

    # Retrieval evaluation (shared across all models)
    print("\n--- Retrieval Evaluation ---")
    retrieval_metrics = evaluate_retrieval(
        benchmark_file,
        lambda q: retriever.retrieve_file_paths(q, top_k=args.top_k),
        k=args.top_k,
    )
    for metric, value in retrieval_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Index metadata for experiment logs
    store = retriever.store
    num_chunks = store.index.ntotal if hasattr(store.index, "ntotal") else 0
    unique_files = {m.get("file_path", "") for m in store.chunks if m.get("file_path")}
    num_documents = len(unique_files)

    models = [m.strip() for m in args.models.split(",")]
    all_model_results = []
    hardware = _get_hardware_info()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    print(f"\n--- Benchmarking {len(models)} models ---")

    for model in models:
        print(f"\n=== Model: {model} ===")

        if not _check_ollama_model_available(model):
            print(f"  SKIP: Model '{model}' not found in Ollama. Run: ollama pull {model}")
            continue

        model_config = LLMConfig(
            model_name=model, temperature=0.0, max_tokens=args.max_tokens,
        )

        per_question_results = []
        latencies = []
        answer_lengths = []
        non_empty_count = 0
        citation_count = 0

        for i, q in enumerate(questions):
            question_text = q["question"]
            print(f"  Q{i+1}/{len(questions)}: {question_text[:60]}...", end=" ", flush=True)

            results = retriever.retrieve(question_text, top_k=args.top_k)
            context_chunks = [r.text for r in results]

            try:
                start = time.perf_counter()
                answer = generate_answer(question_text, context_chunks, model_config)
                latency_ms = (time.perf_counter() - start) * 1000
            except Exception as e:
                err_msg, hint = describe_llm_error(e, model)
                print(f"FAILED: {err_msg}")
                if hint:
                    print(f"    Hint: {hint}")
                per_question_results.append({
                    "question": question_text, "answer": None,
                    "latency_ms": None, "error": err_msg,
                })
                continue

            is_non_empty = _is_non_empty(answer)
            has_cite = _has_citation(answer)

            latencies.append(latency_ms)
            answer_lengths.append(len(answer))
            if is_non_empty:
                non_empty_count += 1
            if has_cite:
                citation_count += 1

            qr: dict = {
                "question": question_text,
                "answer": answer,
                "latency_ms": round(latency_ms, 1),
                "answer_length": len(answer),
                "non_empty": is_non_empty,
                "has_citation": has_cite,
            }

            if args.judge_model:
                scores = judge_answer(question_text, context_chunks, answer, args.judge_model)
                qr["relevance"] = scores["relevance"]
                qr["faithfulness"] = scores["faithfulness"]

            per_question_results.append(qr)
            print(f"{latency_ms:.0f}ms, {len(answer)} chars")

        # Aggregate
        n_answered = len(latencies)
        if n_answered == 0:
            print(f"  No successful answers for {model}.")
            continue

        avg_latency = sum(latencies) / n_answered
        avg_length = sum(answer_lengths) / n_answered
        non_empty_rate = non_empty_count / len(questions)
        citation_rate = citation_count / len(questions)

        model_summary: dict = {
            "model": model,
            "avg_latency_ms": round(avg_latency, 1),
            "avg_answer_length": round(avg_length, 1),
            "non_empty_rate": round(non_empty_rate, 3),
            "citation_rate": round(citation_rate, 3),
        }

        avg_relevance = None
        avg_faithfulness = None
        if args.judge_model:
            rel_scores = [r["relevance"] for r in per_question_results if r.get("relevance") is not None]
            faith_scores = [r["faithfulness"] for r in per_question_results if r.get("faithfulness") is not None]
            if rel_scores:
                avg_relevance = round(sum(rel_scores) / len(rel_scores), 2)
            if faith_scores:
                avg_faithfulness = round(sum(faith_scores) / len(faith_scores), 2)
            model_summary["avg_relevance"] = avg_relevance
            model_summary["avg_faithfulness"] = avg_faithfulness

        all_model_results.append(model_summary)

        print(f"\n  Summary for {model}:")
        print(f"    Avg latency:    {avg_latency:.1f} ms")
        print(f"    Avg answer len: {avg_length:.0f} chars")
        print(f"    Non-empty rate: {non_empty_rate:.0%}")
        print(f"    Citation rate:  {citation_rate:.0%}")
        if avg_relevance is not None:
            print(f"    Avg relevance:    {avg_relevance}")
        if avg_faithfulness is not None:
            print(f"    Avg faithfulness: {avg_faithfulness}")

        # Write experiment log
        safe_name = model.replace(":", "_").replace("/", "_").replace(".", "_")
        ts_compact = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        experiment_id = f"benchmark-{ts_compact}-{safe_name}"

        experiment_log = {
            "experiment_id": experiment_id,
            "timestamp": timestamp,
            "embedding_model": cfg.embedder.model_name,
            "embedding_dimension": retriever.embedder.dimension,
            "index_version": cfg.index_dir,
            "chunk_size": cfg.chunker.chunk_size,
            "chunk_overlap": cfg.chunker.chunk_overlap,
            "top_k": args.top_k,
            f"precision@{args.top_k}": retrieval_metrics.get(f"precision@{args.top_k}", 0),
            f"recall@{args.top_k}": retrieval_metrics.get(f"recall@{args.top_k}", 0),
            "mrr": retrieval_metrics.get("mrr", 0),
            "avg_latency_ms": retrieval_metrics.get("avg_latency_ms", 0),
            "num_documents_indexed": num_documents,
            "num_chunks": num_chunks,
            "hardware": hardware,
            "llm_model": model,
            "llm_temperature": 0.0,
            "llm_max_tokens": args.max_tokens,
            "llm_avg_latency_ms": round(avg_latency, 1),
            "llm_non_empty_rate": round(non_empty_rate, 3),
            "llm_citation_rate": round(citation_rate, 3),
            "llm_avg_relevance": avg_relevance,
            "llm_avg_faithfulness": avg_faithfulness,
            "judge_model": args.judge_model,
            "per_question_results": per_question_results,
        }

        log_dir = _PROJECT_ROOT / "experiments" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{experiment_id}.json"
        with open(log_path, "w") as f:
            json.dump(experiment_log, f, indent=2)
        print(f"  Log saved: {log_path}")

    # Generate results.md
    if all_model_results:
        _generate_results_md(all_model_results, retrieval_metrics, args)
        print(f"\nResults written to {args.output_dir}/results.md")


def _generate_results_md(
    model_results: list[dict],
    retrieval_metrics: dict,
    args: argparse.Namespace,
) -> None:
    """Generate a Markdown results table."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# PaperRAG Benchmark Results",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## Configuration",
        "",
        f"- Benchmark file: `{args.benchmark_file}`",
        f"- Top-k: {args.top_k}",
        f"- Max tokens: {args.max_tokens}",
        f"- Judge model: {args.judge_model or 'N/A'}",
        "",
        "## Retrieval Metrics (shared across all models)",
        "",
    ]
    for metric, value in retrieval_metrics.items():
        lines.append(f"- {metric}: {value:.4f}")
    lines.append("")

    has_judge = args.judge_model is not None
    header = "| Model | Avg Latency (ms) | Avg Answer Len | Non-Empty Rate | Citation Rate |"
    sep = "|-------|-------------------|----------------|----------------|---------------|"
    if has_judge:
        header += " Relevance | Faithfulness | Note |"
        sep += "-----------|--------------|------|"
    else:
        header += " Note |"
        sep += "------|"

    lines.extend(["## Model Comparison", "", header, sep])

    default_model = "qwen2.5:1.5b"
    for r in model_results:
        row = (
            f"| {r['model']} "
            f"| {r['avg_latency_ms']:.0f} "
            f"| {r['avg_answer_length']:.0f} "
            f"| {r['non_empty_rate']:.0%} "
            f"| {r['citation_rate']:.0%} "
        )
        if has_judge:
            rel = r.get("avg_relevance")
            faith = r.get("avg_faithfulness")
            row += f"| {rel if rel is not None else 'N/A'} "
            row += f"| {faith if faith is not None else 'N/A'} "
        note = "default model" if r["model"] == default_model else ""
        row += f"| {note} |"
        lines.append(row)

    lines.append("")
    (output_dir / "results.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-model LLM benchmark for PaperRAG",
    )
    parser.add_argument(
        "--index-dir", required=True,
        help="Path to the PaperRAG index directory",
    )
    parser.add_argument(
        "--benchmark-file", default=DEFAULT_BENCHMARK_FILE,
        help="JSONL file with benchmark questions (default: paperrag/sample_questions.jsonl)",
    )
    parser.add_argument(
        "--models", default=",".join(DEFAULT_MODELS),
        help=f"Comma-separated list of Ollama models (default: {','.join(DEFAULT_MODELS)})",
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Number of chunks to retrieve per question (default: 3)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256,
        help="Maximum tokens for LLM generation (default: 256)",
    )
    parser.add_argument(
        "--judge-model", default=None,
        help="Ollama model for LLM-as-judge scoring (e.g., llama3.3:70b). Omit to skip.",
    )
    parser.add_argument(
        "--output-dir", default=str(Path(__file__).parent),
        help="Directory for results.md output (default: paperrag/)",
    )

    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
