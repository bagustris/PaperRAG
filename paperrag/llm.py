"""LLM module for local inference via Ollama (OpenAI-compatible API) or llama.cpp (GGUF files)."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator

from paperrag.config import LLMConfig

logger = logging.getLogger(__name__)

# Module-level client cache to avoid reconnection overhead per query
_client_cache: object | None = None
_model_checked: set[str] = set()

# Cache for loaded llama-cpp-python Llama instances (keyed by (model_path, n_ctx, n_gpu_layers))
_llama_cache: dict[tuple, object] = {}

SYSTEM_PROMPT = (
    "You are a helpful research assistant. "
    "Answer strictly using the provided context. "
    "If the answer is not in context, respond: 'I don't know.' "
    "Be concise and direct."
)

# Maximum characters per context chunk sent to the LLM.
# Longer chunks are truncated to keep prompt size manageable for small models.
_MAX_CHUNK_CHARS = 750


def _build_prompt(question: str, context_chunks: list[str]) -> str:
    context_lines = []
    for i, chunk in enumerate(context_chunks):
        # Truncate overly long chunks to keep prompt compact
        text = chunk[:_MAX_CHUNK_CHARS] + "..." if len(chunk) > _MAX_CHUNK_CHARS else chunk
        context_lines.append(f"[{i+1}] {text}")
    context_block = "\n\n---\n\n".join(context_lines)

    return (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        f"Answer concisely using ONLY the context. Cite sources as [1], [2], etc.\n\n"
        f"Answer:"
    )


_OLLAMA_BASE_URL = "http://localhost:11434"
_OLLAMA_API_URL = f"{_OLLAMA_BASE_URL}/v1"


def _check_ollama_model_available(model_name: str) -> bool:
    """Check if a model is available in Ollama.

    Returns True if the model is available, False otherwise.
    Uses fuzzy matching to handle version differences (e.g., llama3.2:3b vs llama3.2:2b).
    """
    try:
        import requests
        response = requests.get(f"{_OLLAMA_BASE_URL}/api/tags", timeout=2)
        if response.status_code == 200:
            data = response.json()
            available_models = [model["name"] for model in data.get("models", [])]

            # Exact match first
            if model_name in available_models:
                return True

            # Fuzzy match: check if base model name (without version tag) exists
            # e.g., "llama3.2:3b" -> check for any "llama3.2:*"
            base_model = model_name.split(':')[0] if ':' in model_name else model_name
            for available in available_models:
                available_base = available.split(':')[0] if ':' in available else available
                if base_model == available_base:
                    return True

            return False
        return False
    except Exception:
        # If we can't check (network error, Ollama not running, etc.), assume it's available
        # The actual API call will fail with a better error message
        return True


def _is_gguf_model(model_name: str) -> bool:
    """Return True if *model_name* refers to a local GGUF file (ends with .gguf)."""
    return model_name.lower().endswith(".gguf")


def _get_llama_model(model_path: str, n_ctx: int = 2048, n_gpu_layers: int = 0) -> object:
    """Load and cache a llama-cpp-python ``Llama`` instance for *model_path*.

    The instance is keyed by ``(model_path, n_ctx, n_gpu_layers)`` so that
    changing context size or GPU offload creates a fresh instance.

    Raises ``ImportError`` if llama-cpp-python is not installed.
    Raises ``FileNotFoundError`` if the GGUF file does not exist.
    """
    try:
        from llama_cpp import Llama  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'llama-cpp-python' package is required for GGUF models. "
            "Install with: uv pip install llama-cpp-python"
        ) from exc

    cache_key = (model_path, n_ctx, n_gpu_layers)
    if cache_key not in _llama_cache:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"GGUF model file not found: {model_path}")
        logger.info("Loading GGUF model from %s (n_ctx=%d, n_gpu_layers=%d)", model_path, n_ctx, n_gpu_layers)
        _llama_cache[cache_key] = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
    return _llama_cache[cache_key]


def _build_messages(question: str, context_chunks: list[str], model_name: str) -> list[dict]:
    """Build the chat messages list from question, context chunks and model name."""
    user_prompt = _build_prompt(question, context_chunks)

    # For Qwen3 models, append /no_think to disable the slow "thinking" mode.
    # Thinking mode generates a long internal reasoning chain before answering,
    # which is unnecessary for RAG Q&A and adds ~30-50s of latency.
    model_lower = model_name.lower()
    if "qwen3" in model_lower or "qwen-3" in model_lower:
        user_prompt += " /no_think"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _prepare(
    question: str,
    context_chunks: list[str],
    config: LLMConfig,
) -> tuple:
    """Ollama-specific setup: build messages, get/cache OpenAI client, return (client, messages)."""
    global _client_cache
    from openai import OpenAI

    messages = _build_messages(question, context_chunks, config.model_name)

    if _client_cache is not None:
        client = _client_cache
    else:
        client = OpenAI(api_key="not-needed", base_url=_OLLAMA_API_URL)
        _client_cache = client

    # Check if Ollama model is available (only once per model)
    if config.model_name not in _model_checked:
        if not _check_ollama_model_available(config.model_name):
            logger.warning(
                "Model '%s' not found in Ollama. Available models can be listed with: ollama list\n"
                "To pull this model, run: ollama pull %s",
                config.model_name,
                config.model_name,
            )
        _model_checked.add(config.model_name)

    logger.info("Calling Ollama LLM (model=%s, temp=%.2f)", config.model_name, config.temperature)
    return client, messages


def generate_answer(
    question: str,
    context_chunks: list[str],
    config: LLMConfig | None = None,
) -> str:
    """Generate an answer using the configured LLM backend (blocking).

    Uses llama.cpp when *config.model_name* ends with ``.gguf``;
    otherwise delegates to Ollama via the OpenAI-compatible API.
    """
    config = config or LLMConfig()

    if not context_chunks:
        return "No context available to answer the question."

    if _is_gguf_model(config.model_name):
        llm = _get_llama_model(config.model_name, config.n_ctx, config.n_gpu_layers)
        messages = _build_messages(question, context_chunks, config.model_name)
        logger.info("Calling llama.cpp (model=%s, temp=%.2f)", config.model_name, config.temperature)
        response = llm.create_chat_completion(  # type: ignore[union-attr]
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        return (response["choices"][0]["message"]["content"] or "").strip()

    try:
        client, messages = _prepare(question, context_chunks, config)
    except ImportError:
        raise ImportError(
            "The 'openai' package is required. Install with: uv pip install openai"
        )

    response = client.chat.completions.create(
        model=config.model_name,
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        extra_body={"num_ctx": 2048, "keep_alive": "30m"},
    )
    return (response.choices[0].message.content or "").strip()


def stream_answer(
    question: str,
    context_chunks: list[str],
    config: LLMConfig | None = None,
) -> Iterator[str]:
    """Yield text chunks as they arrive from the LLM (streaming).

    Uses llama.cpp when *config.model_name* ends with ``.gguf``;
    otherwise delegates to Ollama via the OpenAI-compatible API.

    Usage::

        for chunk in stream_answer(question, chunks, cfg.llm):
            sys.stdout.write(chunk)
            sys.stdout.flush()
    """
    config = config or LLMConfig()

    if not context_chunks:
        yield "No context available to answer the question."
        return

    if _is_gguf_model(config.model_name):
        llm = _get_llama_model(config.model_name, config.n_ctx, config.n_gpu_layers)
        messages = _build_messages(question, context_chunks, config.model_name)
        logger.info(
            "Calling llama.cpp streaming (model=%s, temp=%.2f)",
            config.model_name,
            config.temperature,
        )
        response = llm.create_chat_completion(  # type: ignore[union-attr]
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stream=True,
        )
        for chunk in response:
            delta = chunk["choices"][0]["delta"].get("content", "")
            if delta:
                yield delta
        return

    try:
        client, messages = _prepare(question, context_chunks, config)
    except ImportError:
        raise ImportError(
            "The 'openai' package is required. Install with: uv pip install openai"
        )

    response = client.chat.completions.create(
        model=config.model_name,
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        stream=True,
        extra_body={"num_ctx": 2048, "keep_alive": "30m"},
    )
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def describe_llm_error(exc: Exception, model_name: str) -> tuple[str, str | None]:
    """Return (short_error, optional_hint) for a human-readable LLM error message.

    The hint is non-None when there's a concrete remediation action.
    """
    msg = str(exc)

    if _is_gguf_model(model_name):
        if isinstance(exc, FileNotFoundError):
            return (str(exc), None)
        if isinstance(exc, ImportError):
            return (msg, "uv pip install llama-cpp-python")
        return (f"llama.cpp error for '{model_name}': {msg}", None)

    try:
        from openai import APIStatusError
        if isinstance(exc, APIStatusError) and exc.status_code == 500:
            if "missing tensor" in msg or "failed to load model" in msg:
                return (
                    f"Model '{model_name}' failed to load (corrupted download).",
                    f"ollama pull {model_name}",
                )
            return (f"Ollama returned a server error for '{model_name}'.", None)
        if isinstance(exc, APIStatusError):
            return (f"API error {exc.status_code}: {exc.message}", None)
    except ImportError:
        pass
    return (f"LLM error: {msg}", None)
