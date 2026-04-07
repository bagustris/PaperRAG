"""LLM module for local inference via Ollama or llama.cpp (GGUF / HuggingFace models).

Backend selection rules
-----------------------
* Local ``*.gguf`` file path   → **llama-server** (from ``brew install llama-cpp``)
* HuggingFace repo ID          → download GGUF + **llama-server**
  (e.g. ``Qwen/Qwen3-1.7B-GGUF``)
* All other names              → **Ollama** (unchanged)

Example usage
-------------
::

    paperrag query "What is X?" --model qwen2.5:1.5b         # Ollama
    paperrag query "What is X?" --model Qwen/Qwen3-1.7B-GGUF # HF download + llama-server
    paperrag query "What is X?" --model /path/to/model.gguf  # local GGUF + llama-server
"""

from __future__ import annotations

import atexit
import logging
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request
from collections.abc import Iterator

from paperrag.config import LLMConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------

# Ollama OpenAI client cache
_client_cache: object | None = None
_model_checked: set[str] = set()

# llama-server process and client caches (keyed by (model_path, ctx_size, n_gpu_layers))
_llama_server_procs: dict[tuple, subprocess.Popen] = {}
_llama_server_clients: dict[tuple, object] = {}

# Seconds to wait for a llama-server process to exit cleanly before assuming port conflict
_PROC_WAIT_TIMEOUT = 5


def _cleanup_llama_servers() -> None:
    """Terminate all managed llama-server processes at interpreter exit."""
    for proc in list(_llama_server_procs.values()):
        try:
            if proc.poll() is not None:
                continue
            proc.terminate()
            proc.wait(timeout=_PROC_WAIT_TIMEOUT)
        except BaseException:
            try:
                if proc.poll() is None:
                    proc.kill()
            except BaseException:
                pass


atexit.register(_cleanup_llama_servers)

# Maximum characters per context chunk sent to the LLM.
# Longer chunks are truncated to keep prompt size manageable for small models.
_MAX_CHUNK_CHARS = 750
_TRAILING_SOURCE_LINE_RE = re.compile(r"^\s*Sources?:\s*\[[0-9,\s-]+\]\s*$", re.IGNORECASE)

# Regex to strip <think>...</think> blocks produced by reasoning/thinking models
# (e.g. Qwen3, Qwen3.5, phi4-mini-reasoning).  Handles both complete and
# unclosed blocks (the latter can appear when the response is truncated).
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*", re.DOTALL)


def _build_prompt(question: str, context_chunks: list[str], source_labels: list[int] | None = None) -> str:
    context_lines = []
    for i, chunk in enumerate(context_chunks):
        label = source_labels[i] if source_labels else i + 1
        # Truncate overly long chunks to keep prompt compact
        text = chunk[:_MAX_CHUNK_CHARS] + "..." if len(chunk) > _MAX_CHUNK_CHARS else chunk
        context_lines.append(f"[{label}] {text}")
    context_block = "\n\n---\n\n".join(context_lines)

    unique_labels = sorted(set(source_labels)) if source_labels else list(range(1, len(context_chunks) + 1))
    n = len(unique_labels)
    cite_instruction = (
        "Use inline citation [1] within your answer." if n == 1
        else f"Use inline citations [1]–[{n}] within your answer. Only cite sources from [1] to [{n}]."
    )
    return (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        f"Answer concisely using ONLY the context. {cite_instruction} Do not add a separate 'Source:' or 'Sources:' list at the end."
    )


def _strip_think_blocks(text: str) -> str:
    """Remove ``<think>…</think>`` blocks from model output.

    Thinking/reasoning models (e.g. Qwen3, Qwen3.5, phi4-mini-reasoning) may
    wrap internal chain-of-thought reasoning in ``<think>`` tags.  This helper
    strips those blocks so only the visible answer remains.

    Handles both complete ``<think>…</think>`` blocks and unclosed ``<think>``
    blocks (which can occur when the response is truncated by *max_tokens*).
    """
    # Strip complete think blocks first
    result = _THINK_BLOCK_RE.sub("", text)
    # Then strip any unclosed think block (e.g. truncated response)
    result = _UNCLOSED_THINK_RE.sub("", result)
    return result.strip()


def _strip_trailing_source_footers(text: str) -> str:
    """Remove standalone trailing ``Source: [n]`` footer lines from model output."""
    lines = text.splitlines()
    end = len(lines)

    while end > 0 and not lines[end - 1].strip():
        end -= 1

    while end > 0 and _TRAILING_SOURCE_LINE_RE.match(lines[end - 1]):
        end -= 1
        while end > 0 and not lines[end - 1].strip():
            end -= 1

    return "\n".join(lines[:end]).strip()


def _sanitize_stream(chunks: Iterator[str]) -> Iterator[str]:
    """Buffer streamed text, strip think blocks and trailing source footers."""
    text = "".join(chunks)
    cleaned = _strip_think_blocks(text)
    cleaned = _strip_trailing_source_footers(cleaned)
    if cleaned:
        yield cleaned


_OLLAMA_BASE_URL = "http://localhost:11434"
_OLLAMA_API_URL = f"{_OLLAMA_BASE_URL}/v1"


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


def _is_gguf_model(model_name: str) -> bool:
    """Return True if *model_name* is a local path to a GGUF file (ends with ``.gguf``)."""
    return model_name.lower().endswith(".gguf")


def _is_hf_model(model_name: str) -> bool:
    """Return True if *model_name* is a HuggingFace repo ID (e.g. ``Qwen/Qwen3-1.7B-GGUF``).

    HF repo IDs have the form ``owner/repo`` with no path prefix and no Ollama tag syntax.
    Excluded patterns:

    * Absolute or relative paths: ``/…``, ``./…``, ``~/…``, or any path containing ``..``
    * Ollama namespaced models with a tag: ``library/llama3:latest``, ``org/model:tag``
    """
    if model_name.startswith(("/", "./", "~/")):
        return False
    if ".." in model_name.split("/"):  # catch ../, ../../, etc.
        return False
    if ":" in model_name:  # Ollama tag syntax, e.g. qwen2.5:1.5b or library/model:tag
        return False
    parts = model_name.split("/")
    return len(parts) == 2 and all(parts)


def _is_llama_backend(model_name: str) -> bool:
    """Return True if this model should use the llama.cpp (``llama-server``) backend.

    Dispatch rules:

    * Local ``.gguf`` file path   → **llama-server**
    * HuggingFace repo ID (``org/repo``) → download GGUF + **llama-server**
    * All other names (e.g. ``qwen2.5:1.5b``) → **Ollama**
    """
    return _is_gguf_model(model_name) or _is_hf_model(model_name)


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------


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


def prewarm_ollama(config: LLMConfig) -> bool:
    """Send a minimal 1-token request to load the Ollama model into memory.

    Returns True if successful, False if Ollama is unreachable or llama-server backend.
    Only applies to the Ollama backend; llama-server has its own startup mechanism.
    """
    if _is_llama_backend(config.model_name):
        return False

    try:
        from openai import OpenAI
        client = OpenAI(api_key="not-needed", base_url=_OLLAMA_API_URL)
        client.chat.completions.create(
            model=config.model_name,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            stream=False,
            extra_body={"num_ctx": config.ctx_size, "keep_alive": "30m"},
        )
        _model_checked.add(config.model_name)  # skip redundant check on first real query
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# HuggingFace model download
# ---------------------------------------------------------------------------


def _download_hf_gguf(repo_id: str) -> str:
    """Download a GGUF file from *repo_id* on HuggingFace Hub.

    Prefers Q4_K_M quantization; falls back to the first ``.gguf`` file found.
    Models are cached in HuggingFace's default cache (``~/.cache/huggingface/hub/``),
    so re-runs will reuse the download without hitting the network again.

    Raises ``ImportError`` if ``huggingface-hub`` is not installed.
    Raises ``ValueError`` if no GGUF files are found in the repository.
    Raises ``RuntimeError`` if the repository cannot be listed.
    """
    try:
        from huggingface_hub import hf_hub_download, list_repo_files  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "huggingface-hub is required to download HuggingFace models. "
            "Install with: uv pip install huggingface-hub"
        ) from exc

    logger.info("Listing GGUF files in %s ...", repo_id)
    try:
        all_files = list(list_repo_files(repo_id))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to list files in HuggingFace repo '{repo_id}': {exc}"
        ) from exc

    gguf_files = [f for f in all_files if f.lower().endswith(".gguf")]
    if not gguf_files:
        raise ValueError(f"No GGUF files found in HuggingFace repo: {repo_id}")

    # Prefer Q4_K_M (good quality/speed tradeoff), then Q4_0, then first available
    chosen = None
    for pattern in ("Q4_K_M", "q4_k_m", "Q4_0", "q4_0"):
        matches = [f for f in gguf_files if pattern in f]
        if matches:
            chosen = matches[0]
            break
    if chosen is None:
        chosen = gguf_files[0]

    logger.info("Downloading '%s' from %s ...", chosen, repo_id)
    local_path = hf_hub_download(repo_id=repo_id, filename=chosen)
    logger.info("Model ready at %s", local_path)
    return local_path


def _resolve_model_path(model_name: str) -> str:
    """Resolve a model name to a local GGUF file path.

    * Local ``.gguf`` path → validated and returned as-is.
    * HuggingFace repo ID → GGUF downloaded via ``huggingface-hub`` and path returned.
    """
    if _is_gguf_model(model_name):
        if not os.path.isfile(model_name):
            raise FileNotFoundError(f"GGUF model file not found: {model_name}")
        return model_name

    if _is_hf_model(model_name):
        return _download_hf_gguf(model_name)

    raise ValueError(f"Cannot resolve model path for: {model_name!r}")


# ---------------------------------------------------------------------------
# llama-server lifecycle
# ---------------------------------------------------------------------------


def _find_llama_server_binary() -> str:
    """Return the path to the ``llama-server`` binary.

    Checks ``PATH`` first, then common Homebrew install locations.

    Raises ``FileNotFoundError`` if not found with an actionable install hint.
    """
    if bin_path := shutil.which("llama-server"):
        return bin_path

    for candidate in (
        "/opt/homebrew/bin/llama-server",            # Homebrew on Apple Silicon
        "/usr/local/bin/llama-server",               # Homebrew on Intel Mac
        "/home/linuxbrew/.linuxbrew/bin/llama-server",  # Linuxbrew
    ):
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        "llama-server not found.\n"
        "Install with:  brew install llama-cpp\n"
        "See also: https://github.com/ggerganov/llama.cpp"
    )


def _find_free_port() -> int:
    """Return a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_llama_server(port: int, timeout: float = 120.0) -> bool:
    """Poll ``http://localhost:{port}/health`` until the server is ready."""
    url = f"http://localhost:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return True
        except Exception:
            time.sleep(0.5)
    return False


def _get_or_start_llama_server(model_path: str, ctx_size: int, n_gpu_layers: int, n_threads: int = 0) -> object:
    """Return a cached OpenAI client connected to a running ``llama-server``.

    Starts a new ``llama-server`` process on a free port if none is cached for
    the given ``(model_path, ctx_size, n_gpu_layers, n_threads)`` combination, or if the
    previously started process has exited unexpectedly.

    The server process is registered with :func:`atexit` and terminated when
    the Python interpreter exits.

    Install ``llama-server`` with: ``brew install llama-cpp``
    """
    from openai import OpenAI

    effective_threads = n_threads if n_threads > 0 else (os.cpu_count() or 4)
    cache_key = (model_path, ctx_size, n_gpu_layers, effective_threads)

    # Re-use existing server if still alive
    if cache_key in _llama_server_clients:
        proc = _llama_server_procs.get(cache_key)
        if proc is None or proc.poll() is None:  # no proc tracked, or still running
            return _llama_server_clients[cache_key]
        # Process died — remove stale entries and fall through to restart
        logger.warning("llama-server process exited unexpectedly; restarting ...")
        del _llama_server_clients[cache_key]
        _llama_server_procs.pop(cache_key, None)

    llama_server = _find_llama_server_binary()

    # Retry port selection to guard against the TOCTOU window between
    # _find_free_port() releasing the ephemeral port and llama-server binding it.
    _MAX_PORT_RETRIES = 3
    _server_timeout = 120.0
    for attempt in range(1, _MAX_PORT_RETRIES + 1):
        port = _find_free_port()

        cmd = [
            llama_server,
            "--model", model_path,
            "--port", str(port),
            "--ctx-size", str(ctx_size),
            "--n-gpu-layers", str(n_gpu_layers),
            "--threads", str(effective_threads),
            "--parallel", "1",
        ]

        model_label = os.path.basename(model_path)
        log_path = os.path.join(
            tempfile.gettempdir(), f"paperrag-llama-server-{port}.log"
        )
        logger.info(
            "Starting llama-server for '%s' on port %d (attempt %d/%d, stderr → %s)",
            model_label,
            port,
            attempt,
            _MAX_PORT_RETRIES,
            log_path,
        )
        logger.info("llama-server cmd: %s", " ".join(cmd))

        with open(log_path, "w") as log_fh:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=log_fh
            )

        if _wait_for_llama_server(port, timeout=_server_timeout):
            # Register in the cache — cleanup is handled by _cleanup_llama_servers
            _llama_server_procs[cache_key] = proc
            logger.info("llama-server ready on port %d", port)
            client = OpenAI(api_key="not-needed", base_url=f"http://localhost:{port}/v1")
            _llama_server_clients[cache_key] = client
            return client

        # Server did not become ready — check if it died immediately (port race)
        # or just timed out (model loading issue)
        proc.terminate()
        try:
            return_code = proc.wait(timeout=_PROC_WAIT_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            return_code = proc.wait()
        server_failed_immediately = return_code != 0
        if server_failed_immediately and attempt < _MAX_PORT_RETRIES:
            logger.warning(
                "llama-server exited (rc=%d) on port %d; retrying with a new port ...",
                return_code,
                port,
            )
            # Clean up the log file for this failed attempt
            try:
                os.unlink(log_path)
            except OSError:
                pass
            continue
        # Either timed out (model issue) or exhausted retries
        raise RuntimeError(
            f"llama-server failed to start on port {port} within "
            f"{int(_server_timeout)}s. "
            f"Check server log for details: {log_path}\n"
            "Also verify the model file is valid and llama-server is installed "
            "(brew install llama-cpp)."
        )

    # Should be unreachable (loop always returns or raises), but satisfies type checkers
    raise RuntimeError("llama-server could not be started after retries.")


# ---------------------------------------------------------------------------
# Shared message builder
# ---------------------------------------------------------------------------


def _build_messages(question: str, context_chunks: list[str], model_name: str, system_prompt: str, source_labels: list[int] | None = None, think: bool = False) -> list[dict]:
    """Build the chat messages list from question, context chunks, and model name."""
    user_prompt = _build_prompt(question, context_chunks, source_labels=source_labels)

    # For Qwen3/Qwen3.5 models, suppress thinking mode unless explicitly enabled.
    # Thinking mode generates a long internal reasoning chain before answering,
    # which is unnecessary for RAG Q&A and adds ~30-50s of latency.
    model_lower = model_name.lower()
    if "qwen3" in model_lower or "qwen-3" in model_lower:
        if not think:
            user_prompt += " /no_think"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _prepare(
    question: str,
    context_chunks: list[str],
    config: LLMConfig,
    source_labels: list[int] | None = None,
) -> tuple:
    """Ollama-specific setup: build messages, get/cache OpenAI client, return (client, messages)."""
    global _client_cache
    from openai import OpenAI

    messages = _build_messages(question, context_chunks, config.model_name, config.system_prompt, source_labels=source_labels, think=config.think)

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_answer(
    question: str,
    context_chunks: list[str],
    config: LLMConfig | None = None,
) -> str:
    """Generate an answer using the configured LLM backend (blocking).

    Backend selection:

    * HuggingFace repo IDs (``org/repo``) and local ``.gguf`` file paths use
      **llama.cpp** via ``llama-server`` (install: ``brew install llama-cpp``).
    * All other model names delegate to **Ollama**.

    Examples::

        # Ollama (unchanged)
        paperrag query "What is X?" --model qwen2.5:1.5b

        # llama.cpp — download Qwen3 GGUF from HuggingFace automatically
        paperrag query "What is X?" --model Qwen/Qwen3-1.7B-GGUF

        # llama.cpp — use a local GGUF file
        paperrag query "What is X?" --model /path/to/model.gguf
    """
    config = config or LLMConfig()

    if not context_chunks:
        return "No context available to answer the question."

    if _is_llama_backend(config.model_name):
        model_path = _resolve_model_path(config.model_name)
        client = _get_or_start_llama_server(model_path, config.ctx_size, config.n_gpu_layers, config.n_threads)
        messages = _build_messages(question, context_chunks, config.model_name, config.system_prompt, think=config.think)
        logger.info(
            "Calling llama-server (model=%s, temp=%.2f)", config.model_name, config.temperature
        )
        response = client.chat.completions.create(  # type: ignore[union-attr]
            model=os.path.basename(model_path),
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        raw = response.choices[0].message.content or ""
        return _strip_trailing_source_footers(_strip_think_blocks(raw))

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
        extra_body={"num_ctx": config.ctx_size, "keep_alive": "30m"},
    )
    raw = response.choices[0].message.content or ""
    return _strip_trailing_source_footers(_strip_think_blocks(raw))


def stream_answer(
    question: str,
    context_chunks: list[str],
    config: LLMConfig | None = None,
    source_files: list[str] | None = None,
) -> Iterator[str]:
    """Yield text chunks as they arrive from the LLM (streaming).

    Backend selection:

    * HuggingFace repo IDs (``org/repo``) and local ``.gguf`` file paths use
      **llama.cpp** via ``llama-server``.
    * All other model names delegate to **Ollama**.

    Usage::

        for chunk in stream_answer(question, chunks, cfg.llm):
            sys.stdout.write(chunk)
            sys.stdout.flush()
    """
    config = config or LLMConfig()

    if not context_chunks:
        yield "No context available to answer the question."
        return

    # Compute per-chunk source labels: chunks from the same file get the same number
    source_labels: list[int] | None = None
    if source_files:
        file_to_label: dict[str, int] = {}
        source_labels = []
        for f in source_files:
            if f not in file_to_label:
                file_to_label[f] = len(file_to_label) + 1
            source_labels.append(file_to_label[f])

    if _is_llama_backend(config.model_name):
        model_path = _resolve_model_path(config.model_name)
        client = _get_or_start_llama_server(model_path, config.ctx_size, config.n_gpu_layers, config.n_threads)
        messages = _build_messages(question, context_chunks, config.model_name, config.system_prompt, source_labels=source_labels, think=config.think)
        logger.info(
            "Calling llama-server streaming (model=%s, temp=%.2f)",
            config.model_name,
            config.temperature,
        )
        response = client.chat.completions.create(  # type: ignore[union-attr]
            model=os.path.basename(model_path),
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stream=True,
        )
        yield from _sanitize_stream(
            delta
            for chunk in response
            if (delta := chunk.choices[0].delta.content)
        )
        return

    try:
        client, messages = _prepare(question, context_chunks, config, source_labels=source_labels)
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
        extra_body={"num_ctx": config.ctx_size, "keep_alive": "30m"},
    )
    yield from _sanitize_stream(
        delta
        for chunk in response
        if (delta := chunk.choices[0].delta.content)
    )


def describe_llm_error(exc: Exception, model_name: str) -> tuple[str, str | None]:
    """Return (short_error, optional_hint) for a human-readable LLM error message.

    The hint is non-None when there's a concrete remediation action.
    """
    msg = str(exc)

    if _is_llama_backend(model_name):
        if isinstance(exc, FileNotFoundError):
            # Missing llama-server binary vs missing GGUF file
            if "llama-server" in msg:
                return (msg, "brew install llama-cpp")
            return (msg, None)
        if isinstance(exc, ImportError):
            return (msg, "uv pip install huggingface-hub")
        if isinstance(exc, RuntimeError) and "llama-server" in msg:
            return (msg, "brew install llama-cpp")
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
