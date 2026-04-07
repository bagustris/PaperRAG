"""Tests for the llm module — backend routing logic (no live Ollama or llama-server needed)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from paperrag.config import LLMConfig
from paperrag.llm import (
    _build_messages,
    _build_prompt,
    _cleanup_llama_servers,
    _extract_answer,
    _is_gguf_model,
    _is_hf_model,
    _is_llama_backend,
    _sanitize_stream,
    _strip_think_blocks,
    _strip_trailing_source_footers,
    _thinking_max_tokens,
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


def test_is_hf_model_ollama_namespaced():
    """Ollama library/model:tag format must NOT be treated as a HF repo ID."""
    assert _is_hf_model("library/llama3:latest") is False


def test_is_hf_model_relative_parent_path():
    """Relative parent paths like ../parent/child must NOT be treated as HF repo IDs."""
    assert _is_hf_model("../parent/child") is False


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
    prompt = "You are a research assistant."
    msgs = _build_messages("What is X?", ["Some context about X."], "llama3", prompt)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == prompt
    assert msgs[1]["role"] == "user"
    assert "What is X?" in msgs[1]["content"]


def test_build_messages_qwen3_no_think():
    msgs = _build_messages("Q?", ["ctx"], "qwen3:1.7b", "System")
    assert msgs[1]["content"].endswith("/no_think")


def test_build_messages_non_qwen3_no_trailing_no_think():
    msgs = _build_messages("Q?", ["ctx"], "llama3", "System")
    assert not msgs[1]["content"].endswith("/no_think")


def test_build_messages_custom_prompt():
    custom = "You are a pirate."
    msgs = _build_messages("Q?", ["ctx"], "llama3", custom)
    assert msgs[0]["content"] == custom


# ---------------------------------------------------------------------------
# _strip_think_blocks
# ---------------------------------------------------------------------------


def test_strip_think_blocks_complete_block():
    text = "<think>\nLet me reason about this...\n</think>\nThe answer is 42."
    assert _strip_think_blocks(text) == "The answer is 42."


def test_strip_think_blocks_no_blocks():
    text = "The answer is 42."
    assert _strip_think_blocks(text) == "The answer is 42."


def test_strip_think_blocks_only_think_block():
    text = "<think>Only reasoning here, no answer.</think>"
    assert _strip_think_blocks(text) == ""


def test_strip_think_blocks_unclosed():
    text = "<think>This reasoning was truncated by max_tokens and never closed"
    assert _strip_think_blocks(text) == ""


def test_strip_think_blocks_multiple_blocks():
    text = (
        "<think>First reasoning</think>\n"
        "Part one.\n"
        "<think>Second reasoning</think>\n"
        "Part two."
    )
    result = _strip_think_blocks(text)
    assert "Part one." in result
    assert "Part two." in result
    assert "<think>" not in result


def test_strip_think_blocks_leading_text_before_think():
    """Visible text before the first <think> block should be preserved."""
    text = "Intro. <think>reasoning</think> Answer."
    result = _strip_think_blocks(text)
    assert "Intro." in result
    assert "Answer." in result
    assert "<think>" not in result


def test_strip_think_blocks_multiline_reasoning():
    text = (
        "<think>\n"
        "Step 1: Consider the context.\n"
        "Step 2: The paper discusses attention mechanisms.\n"
        "Step 3: Formulate the answer.\n"
        "</think>\n"
        "Attention is a mechanism that allows models to focus on relevant parts of the input [1]."
    )
    result = _strip_think_blocks(text)
    assert result == "Attention is a mechanism that allows models to focus on relevant parts of the input [1]."
    assert "<think>" not in result


def test_strip_think_blocks_empty_string():
    assert _strip_think_blocks("") == ""


def test_strip_think_blocks_whitespace_after_close():
    text = "<think>reasoning</think>   \n\n  The answer."
    assert _strip_think_blocks(text) == "The answer."


# ---------------------------------------------------------------------------
# _thinking_max_tokens
# ---------------------------------------------------------------------------


def test_thinking_max_tokens_think_off():
    cfg = LLMConfig(max_tokens=256, think=False)
    assert _thinking_max_tokens(cfg) == 256


def test_thinking_max_tokens_think_on():
    cfg = LLMConfig(max_tokens=256, think=True)
    assert _thinking_max_tokens(cfg) == 256 * 4


def test_thinking_max_tokens_think_on_custom():
    cfg = LLMConfig(max_tokens=512, think=True)
    assert _thinking_max_tokens(cfg) == 512 * 4


# ---------------------------------------------------------------------------
# _extract_answer
# ---------------------------------------------------------------------------


def test_extract_answer_content_present():
    """_extract_answer returns content when it contains the answer."""
    msg = MagicMock()
    msg.content = "The answer is 42."
    assert _extract_answer(msg) == "The answer is 42."


def test_extract_answer_content_with_think_blocks():
    """_extract_answer strips <think> blocks from content."""
    msg = MagicMock()
    msg.content = "<think>reasoning</think>\nThe answer is 42."
    assert _extract_answer(msg) == "The answer is 42."


def test_extract_answer_content_empty_reasoning_content_present():
    """_extract_answer falls back to reasoning_content when content is empty."""
    msg = MagicMock()
    msg.content = ""
    msg.model_dump.return_value = {
        "content": "",
        "reasoning_content": "The model reasoned and concluded: the answer is 42.",
        "role": "assistant",
    }
    result = _extract_answer(msg)
    assert "the answer is 42" in result.lower()


def test_extract_answer_content_none_reasoning_present():
    """_extract_answer falls back to reasoning field (Ollama) when content is None."""
    msg = MagicMock()
    msg.content = None
    msg.model_dump.return_value = {
        "content": None,
        "reasoning": "After analysis, the paper discusses attention mechanisms.",
        "role": "assistant",
    }
    result = _extract_answer(msg)
    assert "attention" in result.lower()


def test_extract_answer_both_empty():
    """_extract_answer returns empty string when both content and reasoning are empty."""
    msg = MagicMock()
    msg.content = ""
    msg.model_dump.return_value = {"content": "", "role": "assistant"}
    assert _extract_answer(msg) == ""


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


def test_strip_trailing_source_footers_removes_footer_lines_only():
    text = "Answer sentence [1].\n\nSource: [1]\nSource: [2]\n"
    assert _strip_trailing_source_footers(text) == "Answer sentence [1]."


def test_strip_trailing_source_footers_keeps_inline_source_word():
    text = "The source separation method improves SER [1]."
    assert _strip_trailing_source_footers(text) == text


def test_sanitize_stream_removes_trailing_source_footer():
    chunks = ["Answer text [1].\n", "\nSource: [1]\n", "Source: [2]\n"]
    assert list(_sanitize_stream(chunks)) == ["Answer text [1]."]


def test_sanitize_stream_strips_think_blocks():
    """_sanitize_stream should strip <think> blocks from streamed output."""
    chunks = [
        "<think>\nLet me reason",
        " about this.\n</think>\n",
        "The answer is attention [1].",
    ]
    result = list(_sanitize_stream(chunks))
    assert result == ["The answer is attention [1]."]


def test_sanitize_stream_strips_think_blocks_and_source_footers():
    """_sanitize_stream should strip both <think> blocks and trailing source footers."""
    chunks = [
        "<think>reasoning</think>\n",
        "Answer text [1].\n",
        "\nSource: [1]\n",
    ]
    result = list(_sanitize_stream(chunks))
    assert result == ["Answer text [1]."]


def test_sanitize_stream_think_only_yields_nothing():
    """If the entire response is a think block, _sanitize_stream should yield nothing."""
    chunks = ["<think>Only internal reasoning, no answer.</think>"]
    result = list(_sanitize_stream(chunks))
    assert result == []


def test_cleanup_llama_servers_swallows_keyboard_interrupt():
    fake_proc = MagicMock()
    fake_proc.poll.side_effect = [None, None]
    fake_proc.wait.side_effect = KeyboardInterrupt()

    original_procs = dict(_llama_server_procs)
    try:
        _llama_server_procs.clear()
        _llama_server_procs[("model.gguf", 2048, 0, 4)] = fake_proc
        _cleanup_llama_servers()
    finally:
        _llama_server_procs.clear()
        _llama_server_procs.update(original_procs)

    fake_proc.terminate.assert_called_once()
    fake_proc.kill.assert_called_once()


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
    mock_start.assert_called_once_with(str(model_file), cfg.ctx_size, cfg.n_gpu_layers, cfg.n_threads)


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


def _make_stream_chunk(content: str | None = None, reasoning: str | None = None) -> MagicMock:
    """Create a fake streaming chunk with proper model_dump() support."""
    chunk = MagicMock()
    chunk.choices[0].delta.content = content
    dump = {"content": content, "role": "assistant"}
    if reasoning is not None:
        dump["reasoning"] = reasoning
    chunk.choices[0].delta.model_dump.return_value = dump
    return chunk


def test_stream_answer_llama_server_mocked(tmp_path):
    """stream_answer yields tokens from llama-server via OpenAI streaming."""
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"fake gguf")

    cfg = LLMConfig(model_name=str(model_file))

    chunk1 = _make_stream_chunk(content="Hello")
    chunk2 = _make_stream_chunk(content=" world")

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = iter([chunk1, chunk2])

    with patch("paperrag.llm._get_or_start_llama_server", return_value=fake_client):
        chunks = list(stream_answer("What?", ["ctx"], cfg))

    assert chunks == ["Hello world"]


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


# ---------------------------------------------------------------------------
# generate_answer / stream_answer — thinking model responses (mocked)
# ---------------------------------------------------------------------------


def test_generate_answer_strips_think_blocks_ollama():
    """generate_answer should strip <think> blocks from Ollama thinking model responses."""
    cfg = LLMConfig(model_name="qwen3:1.7b", think=True)

    think_response = (
        "<think>\n"
        "The user asks about attention mechanisms.\n"
        "Let me check the context provided.\n"
        "</think>\n"
        "Attention allows models to focus on relevant input parts [1]."
    )

    fake_response = MagicMock()
    fake_response.choices[0].message.content = think_response

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    with patch("paperrag.llm._prepare", return_value=(fake_client, [])):
        result = generate_answer("What is attention?", ["context about attention"], cfg)

    assert result != ""
    assert "<think>" not in result
    assert "Attention allows models to focus on relevant input parts [1]." == result


def test_generate_answer_strips_think_blocks_llama_server(tmp_path):
    """generate_answer should strip <think> blocks from llama-server thinking model responses."""
    model_file = tmp_path / "qwen3-model.gguf"
    model_file.write_bytes(b"fake gguf")

    cfg = LLMConfig(model_name=str(model_file), think=True)

    think_response = (
        "<think>\n"
        "Step 1: Read context about the paper.\n"
        "Step 2: The paper is about speech processing.\n"
        "</think>\n"
        "The paper discusses speech chain mechanisms [1]."
    )

    fake_response = MagicMock()
    fake_response.choices[0].message.content = think_response

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    with patch("paperrag.llm._get_or_start_llama_server", return_value=fake_client):
        result = generate_answer("What is the paper about?", ["context about speech"], cfg)

    assert result != ""
    assert "<think>" not in result
    assert "The paper discusses speech chain mechanisms [1]." == result


def test_stream_answer_strips_think_blocks_ollama():
    """stream_answer should strip <think> blocks from Ollama thinking model responses."""
    cfg = LLMConfig(model_name="qwen3:1.7b", think=True)

    # Simulate streamed chunks that include think blocks in content
    chunk1 = _make_stream_chunk(content="<think>\nReasoning")
    chunk2 = _make_stream_chunk(content=" here.\n</think>\n")
    chunk3 = _make_stream_chunk(content="The answer is 42.")

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = iter([chunk1, chunk2, chunk3])

    with patch("paperrag.llm._prepare", return_value=(fake_client, [])):
        chunks = list(stream_answer("What is the answer?", ["some context"], cfg))

    assert len(chunks) == 1
    assert chunks[0] == "The answer is 42."
    assert "<think>" not in chunks[0]


def test_generate_answer_thinking_only_response():
    """generate_answer returns empty string when model only outputs thinking, no answer."""
    cfg = LLMConfig(model_name="qwen3:1.7b", think=True)

    # Model produces only a think block with no visible answer
    think_only = "<think>\nI thought about this but produced no answer.\n</think>"

    fake_msg = MagicMock()
    fake_msg.content = think_only
    fake_msg.model_dump.return_value = {
        "content": think_only,
        "role": "assistant",
    }

    fake_response = MagicMock()
    fake_response.choices[0].message = fake_msg

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    with patch("paperrag.llm._prepare", return_value=(fake_client, [])):
        result = generate_answer("Q?", ["ctx"], cfg)

    assert result == ""
