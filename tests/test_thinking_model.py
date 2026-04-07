"""Integration tests for thinking/reasoning model support.

These tests verify that PaperRAG correctly handles models with thinking
capabilities (e.g. Qwen3, phi4-mini-reasoning) — specifically that
<think>...</think> blocks are stripped and the visible answer is non-empty.

The Ollama-based tests require a running Ollama instance with the model
already pulled and are skipped automatically when Ollama is not available.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from paperrag.config import LLMConfig, PaperRAGConfig
from paperrag.llm import generate_answer, stream_answer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEST_PDFS_DIR = Path(__file__).resolve().parent.parent / "test_pdfs"
# Pick the smallest PDF in test_pdfs for fast indexing
SMALL_PDF = TEST_PDFS_DIR / "Halevy, Norvig, Pereira - 2009 - The unreasonable effectiveness of data.pdf"


def _ollama_available() -> bool:
    """Return True if Ollama is reachable on localhost:11434."""
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2):
            return True
    except Exception:
        return False


def _ollama_model_available(model: str) -> bool:
    """Return True if a specific model is available in Ollama."""
    if not _ollama_available():
        return False
    try:
        import json
        import urllib.request
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as resp:
            data = json.loads(resp.read())
            available = [m["name"] for m in data.get("models", [])]
            base = model.split(":")[0]
            return any(base == a.split(":")[0] for a in available)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Unit-level: mocked thinking-model integration test with context from PDF
# ---------------------------------------------------------------------------


def test_thinking_model_non_empty_answer_mocked():
    """Simulate a thinking model answering a reasoning question about a PDF.

    Uses a realistic response that includes <think> blocks, verifying that
    the final answer is non-empty after stripping.
    """
    cfg = LLMConfig(model_name="qwen3:1.7b", think=False)

    # Simulate a response a thinking model might produce even with /no_think
    think_response = (
        "<think>\n"
        "The user is asking about the main argument of the paper.\n"
        "From the context, the paper argues that simple models with\n"
        "more data outperform complex models with less data.\n"
        "Let me formulate a concise answer.\n"
        "</think>\n"
        "The paper argues that the effectiveness of simple statistical "
        "models can be dramatically improved by using very large datasets, "
        "often outperforming more sophisticated algorithms trained on "
        "smaller data [1]."
    )

    fake_response = MagicMock()
    fake_response.choices[0].message.content = think_response

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    with patch("paperrag.llm._prepare", return_value=(fake_client, [])):
        result = generate_answer(
            "What is the main argument of the paper?",
            ["Simple models with more data outperform complex models..."],
            cfg,
        )

    # The answer MUST be non-empty (this is the core of issue #7)
    assert result != "", "Answer should not be empty after stripping think blocks"
    assert "<think>" not in result
    assert "</think>" not in result
    assert "effectiveness" in result.lower()


def test_thinking_model_stream_non_empty_mocked():
    """Simulate streaming from a thinking model, verifying non-empty output."""
    cfg = LLMConfig(model_name="qwen3:1.7b", think=True)

    chunk1 = MagicMock()
    chunk1.choices[0].delta.content = "<think>\nLet me analyze the context.\n</think>\n"
    chunk2 = MagicMock()
    chunk2.choices[0].delta.content = "Data quantity matters more than algorithm complexity [1]."

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = iter([chunk1, chunk2])

    with patch("paperrag.llm._prepare", return_value=(fake_client, [])):
        chunks = list(stream_answer(
            "Why does data matter?",
            ["More data leads to better results..."],
            cfg,
        ))

    assert len(chunks) > 0, "stream_answer should yield at least one chunk"
    full_answer = "".join(chunks)
    assert full_answer != "", "Streamed answer should not be empty"
    assert "<think>" not in full_answer


# ---------------------------------------------------------------------------
# Live Ollama integration tests (skipped when Ollama or model unavailable)
# ---------------------------------------------------------------------------

THINKING_MODEL = "qwen3:1.7b"


@pytest.mark.skipif(
    not _ollama_model_available(THINKING_MODEL),
    reason=f"Ollama not running or '{THINKING_MODEL}' model not available",
)
class TestThinkingModelLiveOllama:
    """Live integration tests using Ollama with a thinking-capable model.

    These tests index a small PDF from test_pdfs and query it with a simple
    reasoning question, asserting the answer is non-empty.
    """

    @pytest.fixture(autouse=True)
    def _setup_index(self, tmp_path):
        """Index the small test PDF into a temporary directory."""
        from paperrag.chunker import chunk_paper
        from paperrag.embedder import Embedder
        from paperrag.parser import parse_pdf
        from paperrag.vectorstore import VectorStore

        self.index_dir = tmp_path / ".paperrag-index"
        self.index_dir.mkdir()

        cfg = PaperRAGConfig(input_dir=str(SMALL_PDF))
        embedder = Embedder(cfg.embedder)
        store = VectorStore(self.index_dir, embedder.dimension)

        paper = parse_pdf(SMALL_PDF, cfg.parser)
        chunks = chunk_paper(paper, cfg.chunker)
        assert len(chunks) > 0, f"No chunks produced from {SMALL_PDF.name}"

        embeddings = embedder.embed([c.text for c in chunks])
        store.add(embeddings, chunks)
        store.save(config=cfg)

        self.cfg = PaperRAGConfig(input_dir=str(SMALL_PDF))
        self.cfg.index_dir = str(self.index_dir)
        self.cfg.llm.model_name = THINKING_MODEL
        self.cfg.llm.think = False  # test with /no_think (the default)
        self.store = store
        self.embedder = embedder

    def test_query_non_empty_answer(self):
        """Query with a simple reasoning question and verify non-empty answer."""
        from paperrag.retriever import Retriever

        retriever = Retriever(self.cfg, store=self.store)
        results = retriever.retrieve("What is the main argument of the paper?", top_k=2)
        assert len(results) > 0, "Retriever should return results"

        context_chunks = [r.text for r in results]
        answer = generate_answer(
            "What is the main argument of the paper?",
            context_chunks,
            self.cfg.llm,
        )

        assert answer != "", (
            f"Answer should not be empty! Model '{THINKING_MODEL}' returned an "
            f"empty answer. This is the bug described in issue #7."
        )
        assert "<think>" not in answer, "Think blocks should be stripped"

    def test_stream_query_non_empty(self):
        """Stream a query and verify the answer is non-empty."""
        from paperrag.retriever import Retriever

        retriever = Retriever(self.cfg, store=self.store)
        results = retriever.retrieve("Why is more data helpful?", top_k=2)
        assert len(results) > 0

        context_chunks = [r.text for r in results]
        chunks = list(stream_answer(
            "Why is more data helpful?",
            context_chunks,
            self.cfg.llm,
        ))

        full_answer = "".join(chunks)
        assert full_answer != "", (
            f"Streamed answer should not be empty! Model '{THINKING_MODEL}' "
            f"returned an empty answer."
        )
        assert "<think>" not in full_answer

    def test_thinking_enabled_non_empty(self):
        """Even with thinking mode ON, the visible answer should be non-empty."""
        from paperrag.retriever import Retriever

        self.cfg.llm.think = True
        retriever = Retriever(self.cfg, store=self.store)
        results = retriever.retrieve("Summarize the key finding of this paper.", top_k=2)
        assert len(results) > 0

        context_chunks = [r.text for r in results]
        answer = generate_answer(
            "Summarize the key finding of this paper.",
            context_chunks,
            self.cfg.llm,
        )

        assert answer != "", (
            f"Answer should not be empty even with think=True! "
            f"The <think> blocks should be stripped from the response."
        )
        assert "<think>" not in answer
