"""
src/llm/client.py — Google Gemini API wrapper (via OpenAI-compatible endpoint).
Free tier: 1,500 requests/day. Get a key at https://aistudio.google.com
"""
from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import GEMINI_MODEL, CLAUDE_MAX_TOKENS, get_api_key

# Ordered list of models to try — each has its own free-tier quota pool
_MODEL_FALLBACK_ORDER = [
    GEMINI_MODEL,
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
]


def _is_quota_error(exc: Exception) -> bool:
    return "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc) or "quota" in str(exc).lower()


def _rag_fallback(messages: list[dict]) -> str:
    """Return formatted RAG context when all Gemini models are quota-exhausted."""
    user_content = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
    )
    return (
        "⚠️ **AI quota temporarily exhausted** — here is the relevant medical literature "
        "retrieved from my knowledge base:\n\n"
        + user_content.replace("Question:", "**Question:**", 1)
        + "\n\n*Please try again in a few minutes when the API quota resets, "
        "or visit the BreastCare AI documentation for more information.*"
    )


def call_llm(messages: list[dict], system: str | None = None) -> str:
    """
    Call Google Gemini via OpenAI-compatible endpoint with given messages.
    Tries multiple models in fallback order if quota is exceeded.
    Returns raw response text, or a formatted RAG context string on total failure.
    Raises ValueError if GEMINI_API_KEY is not set.
    """
    from openai import OpenAI
    from src.rag.prompt_builder import SYSTEM_PROMPT

    api_key = get_api_key()
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not set. Add it to your farwa_env activate script.\n"
            "Get a free key (no credit card) at: https://aistudio.google.com"
        )

    effective_system = system or SYSTEM_PROMPT
    full_messages = [{"role": "system", "content": effective_system}] + messages

    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    tried = set()
    for model in _MODEL_FALLBACK_ORDER:
        if model in tried:
            continue
        tried.add(model)
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=CLAUDE_MAX_TOKENS,
                messages=full_messages,
            )
            return response.choices[0].message.content
        except Exception as exc:
            if _is_quota_error(exc):
                continue  # try next model
            raise  # non-quota errors bubble up immediately

    # All models exhausted — return useful RAG fallback
    return _rag_fallback(full_messages)
