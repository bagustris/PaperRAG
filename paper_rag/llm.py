"""LLM module supporting local (Ollama) and OpenAI-compatible APIs."""

from __future__ import annotations

import logging

from paper_rag.config import LLMConfig

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful research assistant. "
    "Answer strictly using the provided context. "
    "If the answer is not in context, respond: 'I don't know.'"
)


def _build_prompt(question: str, context_chunks: list[str]) -> str:
    context_lines = []
    for i, chunk in enumerate(context_chunks):
        context_lines.append(f"[{i+1}] {chunk}")
    context_block = "\n\n---\n\n".join(context_lines)
    
    return (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        f"Answer (use inline citations like [1], [2] where appropriate):"
    )


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
            "  • Use --llm-mode local with Ollama, or\n"
            "  • Use --no-llm to suppress this message"
        )

    client = OpenAI(api_key=api_key, base_url=base_url)

    logger.info(
        "Calling LLM (%s, model=%s, temp=%.2f)",
        config.mode,
        config.model_name,
        config.temperature,
    )

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
