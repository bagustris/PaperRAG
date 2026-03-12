"""Tests for the llm module — backend routing logic (no live Ollama or GGUF file needed)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from paperrag.config import LLMConfig
from paperrag.llm import (
    _build_messages,
    _build_prompt,
    _is_gguf_model,
    _llama_cache,
    describe_llm_error,
    generate_answer,
    stream_answer,
)


# ---------------------------------------------------------------------------
# _is_gguf_model
# ---------------------------------------------------------------------------


def test_is_gguf_model_with_gguf_extension():
    assert _is_gguf_model("model.gguf") is True


def test_is_gguf_model_with_gguf_uppercase():
    assert _is_gguf_model("/path/to/Model.GGUF") is True


def test_is_gguf_model_with_ollama_name():
    assert _is_gguf_model("qwen2.5:1.5b") is False


def test_is_gguf_model_with_bare_name():
    assert _is_gguf_model("llama3") is False


# ---------------------------------------------------------------------------
# _build_messages
# ---------------------------------------------------------------------------


def test_build_messages_structure():
    msgs = _build_messages("What is X?", ["Some context about X."], "llama3")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "What is X?" in msgs[1]["content"]


def test_build_messages_qwen3_no_think():
    msgs = _build_messages("Q?", ["ctx"], "qwen3:1.7b")
    assert msgs[1]["content"].endswith("/no_think")


def test_build_messages_non_qwen3_no_trailing_no_think():
    msgs = _build_messages("Q?", ["ctx"], "llama3")
    assert not msgs[1]["content"].endswith("/no_think")


# ---------------------------------------------------------------------------
# generate_answer — no context
# ---------------------------------------------------------------------------


def test_generate_answer_no_context():
    result = generate_answer("Q?", [])
    assert result == "No context available to answer the question."


def test_stream_answer_no_context():
    chunks = list(stream_answer("Q?", []))
    assert chunks == ["No context available to answer the question."]


# ---------------------------------------------------------------------------
# generate_answer — llama.cpp path (mocked)
# ---------------------------------------------------------------------------


def test_generate_answer_gguf_file_not_found():
    cfg = LLMConfig(model_name="/nonexistent/model.gguf")
    fake_llama_cpp = MagicMock()
    with patch.dict("sys.modules", {"llama_cpp": fake_llama_cpp}):
        with patch("paperrag.llm._llama_cache", {}):
            with pytest.raises(FileNotFoundError):
                generate_answer("Q?", ["context"], cfg)


def test_generate_answer_gguf_missing_package():
    cfg = LLMConfig(model_name="/some/model.gguf")
    with patch("paperrag.llm._llama_cache", {}):
        # Simulate llama_cpp not installed
        with patch.dict("sys.modules", {"llama_cpp": None}):
            with pytest.raises(ImportError, match="llama-cpp-python"):
                generate_answer("Q?", ["context"], cfg)


def test_generate_answer_gguf_mocked(tmp_path):
    """Full generate_answer path through llama.cpp using a mocked Llama."""
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"fake gguf")

    cfg = LLMConfig(model_name=str(model_file))

    fake_llm = MagicMock()
    fake_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "  answer text  "}}]
    }

    fake_llama_cpp = MagicMock()
    fake_llama_cpp.Llama.return_value = fake_llm

    # Clear the module-level cache so the mock is used
    _llama_cache.clear()
    try:
        with patch.dict("sys.modules", {"llama_cpp": fake_llama_cpp}):
            result = generate_answer("What?", ["some context"], cfg)

        assert result == "answer text"
        fake_llm.create_chat_completion.assert_called_once()
    finally:
        _llama_cache.clear()


def test_stream_answer_gguf_mocked(tmp_path):
    """Full stream_answer path through llama.cpp using a mocked Llama."""
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"fake gguf")

    cfg = LLMConfig(model_name=str(model_file))

    def _fake_stream(**kwargs):
        for token in ["Hello", " world"]:
            yield {"choices": [{"delta": {"content": token}}]}

    fake_llm = MagicMock()
    fake_llm.create_chat_completion.side_effect = _fake_stream

    fake_llama_cpp = MagicMock()
    fake_llama_cpp.Llama.return_value = fake_llm

    _llama_cache.clear()
    try:
        with patch.dict("sys.modules", {"llama_cpp": fake_llama_cpp}):
            chunks = list(stream_answer("What?", ["ctx"], cfg))

        assert chunks == ["Hello", " world"]
    finally:
        _llama_cache.clear()


# ---------------------------------------------------------------------------
# describe_llm_error
# ---------------------------------------------------------------------------


def test_describe_llm_error_gguf_file_not_found():
    exc = FileNotFoundError("GGUF model file not found: /x/y.gguf")
    msg, hint = describe_llm_error(exc, "/x/y.gguf")
    assert "not found" in msg
    assert hint is None


def test_describe_llm_error_gguf_import_error():
    exc = ImportError("The 'llama-cpp-python' package is required")
    msg, hint = describe_llm_error(exc, "/x/y.gguf")
    assert hint == "uv pip install llama-cpp-python"


def test_describe_llm_error_gguf_generic():
    exc = RuntimeError("some llama.cpp failure")
    msg, hint = describe_llm_error(exc, "model.gguf")
    assert "llama.cpp error" in msg
    assert hint is None


def test_describe_llm_error_ollama_generic():
    exc = RuntimeError("connection refused")
    msg, hint = describe_llm_error(exc, "qwen2.5:1.5b")
    assert "LLM error" in msg
    assert hint is None
