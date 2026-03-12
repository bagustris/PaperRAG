"""Tests for the llm module — backend routing logic (no live Ollama or llama-server needed)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from paperrag.config import LLMConfig
from paperrag.llm import (
    _build_messages,
    _build_prompt,
    _is_gguf_model,
    _is_hf_model,
    _is_llama_backend,
    _llama_server_clients,
    _llama_server_procs,
    _resolve_model_path,
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
# _is_hf_model
# ---------------------------------------------------------------------------


def test_is_hf_model_valid_repo():
    assert _is_hf_model("Qwen/Qwen3-1.7B-GGUF") is True


def test_is_hf_model_valid_repo_with_org():
    assert _is_hf_model("microsoft/phi-3-mini-4k-instruct-gguf") is True


def test_is_hf_model_local_absolute_path():
    assert _is_hf_model("/local/path/model.gguf") is False


def test_is_hf_model_local_relative_path():
    assert _is_hf_model("./model.gguf") is False


def test_is_hf_model_tilde_path():
    assert _is_hf_model("~/models/model.gguf") is False


def test_is_hf_model_ollama_name():
    assert _is_hf_model("qwen2.5:1.5b") is False


def test_is_hf_model_bare_name():
    assert _is_hf_model("llama3") is False


# ---------------------------------------------------------------------------
# _is_llama_backend
# ---------------------------------------------------------------------------


def test_is_llama_backend_local_gguf():
    assert _is_llama_backend("/path/to/model.gguf") is True


def test_is_llama_backend_hf_repo():
    assert _is_llama_backend("Qwen/Qwen3-1.7B-GGUF") is True


def test_is_llama_backend_ollama_model():
    assert _is_llama_backend("qwen2.5:1.5b") is False


def test_is_llama_backend_bare_ollama():
    assert _is_llama_backend("llama3") is False


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
# _resolve_model_path
# ---------------------------------------------------------------------------


def test_resolve_model_path_local_gguf(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fake gguf")
    assert _resolve_model_path(str(model)) == str(model)


def test_resolve_model_path_local_gguf_missing():
    with pytest.raises(FileNotFoundError, match="GGUF model file not found"):
        _resolve_model_path("/nonexistent/path/model.gguf")


def test_resolve_model_path_hf_model_mocked():
    """_resolve_model_path delegates to _download_hf_gguf for HF repo IDs."""
    with patch("paperrag.llm._download_hf_gguf", return_value="/cache/model.gguf") as mock_dl:
        result = _resolve_model_path("Qwen/Qwen3-1.7B-GGUF")
    assert result == "/cache/model.gguf"
    mock_dl.assert_called_once_with("Qwen/Qwen3-1.7B-GGUF")


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
# generate_answer — llama-server path (mocked)
# ---------------------------------------------------------------------------


def test_generate_answer_llama_server_mocked(tmp_path):
    """generate_answer routes GGUF models through _get_or_start_llama_server."""
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"fake gguf")

    cfg = LLMConfig(model_name=str(model_file))

    fake_response = MagicMock()
    fake_response.choices[0].message.content = "  answer text  "

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    with patch("paperrag.llm._get_or_start_llama_server", return_value=fake_client) as mock_start:
        result = generate_answer("What?", ["some context"], cfg)

    assert result == "answer text"
    mock_start.assert_called_once_with(str(model_file), cfg.n_ctx, cfg.n_gpu_layers)


def test_generate_answer_hf_model_mocked():
    """generate_answer downloads HF models then routes through llama-server."""
    cfg = LLMConfig(model_name="Qwen/Qwen3-1.7B-GGUF")

    fake_response = MagicMock()
    fake_response.choices[0].message.content = "hf answer"

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    with patch("paperrag.llm._resolve_model_path", return_value="/cache/model.gguf") as mock_res:
        with patch("paperrag.llm._get_or_start_llama_server", return_value=fake_client):
            result = generate_answer("What?", ["ctx"], cfg)

    assert result == "hf answer"
    mock_res.assert_called_once_with("Qwen/Qwen3-1.7B-GGUF")


def test_stream_answer_llama_server_mocked(tmp_path):
    """stream_answer yields tokens from llama-server via OpenAI streaming."""
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"fake gguf")

    cfg = LLMConfig(model_name=str(model_file))

    chunk1 = MagicMock()
    chunk1.choices[0].delta.content = "Hello"
    chunk2 = MagicMock()
    chunk2.choices[0].delta.content = " world"

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = iter([chunk1, chunk2])

    with patch("paperrag.llm._get_or_start_llama_server", return_value=fake_client):
        chunks = list(stream_answer("What?", ["ctx"], cfg))

    assert chunks == ["Hello", " world"]


def test_generate_answer_gguf_file_not_found():
    cfg = LLMConfig(model_name="/nonexistent/model.gguf")
    with pytest.raises(FileNotFoundError):
        generate_answer("Q?", ["context"], cfg)


def test_generate_answer_hf_missing_package():
    """ImportError for huggingface-hub is propagated clearly."""
    cfg = LLMConfig(model_name="Qwen/Qwen3-1.7B-GGUF")
    with patch(
        "paperrag.llm._resolve_model_path",
        side_effect=ImportError("huggingface-hub is required"),
    ):
        with pytest.raises(ImportError, match="huggingface-hub"):
            generate_answer("Q?", ["context"], cfg)


# ---------------------------------------------------------------------------
# describe_llm_error
# ---------------------------------------------------------------------------


def test_describe_llm_error_gguf_file_not_found():
    exc = FileNotFoundError("GGUF model file not found: /x/y.gguf")
    msg, hint = describe_llm_error(exc, "/x/y.gguf")
    assert "not found" in msg
    assert hint is None


def test_describe_llm_error_llama_server_not_found():
    exc = FileNotFoundError("llama-server not found. Install with: brew install llama-cpp")
    msg, hint = describe_llm_error(exc, "/x/y.gguf")
    assert hint == "brew install llama-cpp"


def test_describe_llm_error_hf_import_error():
    exc = ImportError("huggingface-hub is required to download HuggingFace models.")
    msg, hint = describe_llm_error(exc, "Qwen/Qwen3-1.7B-GGUF")
    assert hint == "uv pip install huggingface-hub"


def test_describe_llm_error_llama_server_runtime():
    exc = RuntimeError("llama-server failed to start on port 12345")
    msg, hint = describe_llm_error(exc, "model.gguf")
    assert hint == "brew install llama-cpp"


def test_describe_llm_error_gguf_generic():
    exc = RuntimeError("some other failure")
    msg, hint = describe_llm_error(exc, "model.gguf")
    assert "llama.cpp error" in msg
    assert hint is None


def test_describe_llm_error_ollama_generic():
    exc = RuntimeError("connection refused")
    msg, hint = describe_llm_error(exc, "qwen2.5:1.5b")
    assert "LLM error" in msg
    assert hint is None

