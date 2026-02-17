"""LLM module supporting local (Ollama) and OpenAI-compatible APIs."""

from __future__ import annotations

import logging

from paperrag.config import LLMConfig

logger = logging.getLogger(__name__)

# Module-level client cache to avoid reconnection overhead per query
_client_cache: dict[str, object] = {}
_model_checked: set[str] = set()

SYSTEM_PROMPT = (
    "You are a helpful research assistant. "
    "Answer strictly using the provided context. "
    "If the answer is not in context, respond: 'I don't know.' "
    "Be concise and direct."
)

# Maximum characters per context chunk sent to the LLM.
# Longer chunks are truncated to keep prompt size manageable for small models.
_MAX_CHUNK_CHARS = 1500


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
        f"Instructions: Answer using ONLY the provided context above. "
        f"Use numbered citations [1], [2], etc. to reference specific context sources. "
        f"Do NOT use author-year citations like (Smith et al., 2020). "
        f"Only use the numbered format [1], [2], [3], etc.\n\n"
        f"Answer:"
    )


def _check_ollama_model_available(model_name: str, base_url: str = "http://localhost:11434") -> bool:
    """Check if a model is available in Ollama.
    
    Returns True if the model is available, False otherwise.
    Uses fuzzy matching to handle version differences (e.g., llama3.2:3b vs llama3.2:2b).
    """
    try:
        import requests
        response = requests.get(f"{base_url}/api/tags", timeout=2)
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



def generate_answer(
    question: str,
    context_chunks: list[str],
    config: LLMConfig | None = None,
) -> str:
    """Generate an answer using the configured LLM backend.

    Supports two modes:
    - **openai**: Any OpenAI-compatible API (OpenAI, Ollama via /v1, etc.)
    - **local**: Same as openai but defaults api_base to localhost Ollama.
    """
    config = config or LLMConfig()

    if not context_chunks:
        return "No context available to answer the question."

    user_prompt = _build_prompt(question, context_chunks)

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "The 'openai' package is required for LLM support. "
            "Install with: uv pip install paper-rag[llm]"
        )

    api_key = config.resolve_api_key()
    base_url = config.api_base
    if config.mode == "local" and not base_url:
        base_url = "http://localhost:11434/v1"
    if not api_key and config.mode == "local":
        api_key = "not-needed"

    # Validate API key for OpenAI mode
    if config.mode == "openai" and not api_key:
        raise ValueError(
            "LLM answer generation skipped (no API key configured). To enable:\n"
            "  • Set OPENAI_API_KEY environment variable, or\n"
            "  • Use --model <name> for local Ollama models, or\n"
            "  • Use --no-llm to suppress this message"
        )

    # Reuse cached client for same base_url to avoid connection overhead
    cache_key = f"{base_url}|{api_key}"
    if cache_key in _client_cache:
        client = _client_cache[cache_key]
    else:
        client = OpenAI(api_key=api_key, base_url=base_url)
        _client_cache[cache_key] = client

    # Check if local model is available in Ollama (only once per model)
    if config.mode == "local" and config.model_name not in _model_checked:
        # Strip /v1 suffix from base_url for Ollama API check
        check_url = (base_url or "http://localhost:11434").replace("/v1", "").rstrip("/")
        if not _check_ollama_model_available(config.model_name, check_url):
            logger.warning(
                "Model '%s' not found in Ollama. Available models can be listed with: ollama list\n"
                "To pull this model, run: ollama pull %s",
                config.model_name,
                config.model_name,
            )
        _model_checked.add(config.model_name)

    logger.info(
        "Calling LLM (%s, model=%s, temp=%.2f)",
        config.mode,
        config.model_name,
        config.temperature,
    )

    # For Qwen3 models, append /no_think to disable the slow "thinking" mode.
    # Thinking mode generates a long internal reasoning chain before answering,
    # which is unnecessary for RAG Q&A and adds ~30-50s of latency.
    model_lower = config.model_name.lower()
    if "qwen3" in model_lower or "qwen-3" in model_lower:
        user_prompt += " /no_think"

    response = client.chat.completions.create(
        model=config.model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    answer = response.choices[0].message.content or ""
    return answer.strip()
