"""
src/llm/client.py — Google Gemini through its OpenAI-compatible endpoint.

One entry point, complete(), used by the pipeline, the agents and the evaluations:
- model fallback: on a daily-quota error, or a model that is retired (404) or keeps
  failing, the next model in FALLBACK_MODELS is tried;
- per-model parameters (LLM_MODEL_PARAMS) keep "thinking" minimal, because thinking
  tokens count against max_tokens and used to truncate answers;
- retries with exponential backoff for transient errors (5xx, timeouts) and, when
  wait_on_rate_limit=True (evaluations), for per-minute rate limits; interactive
  calls get a short retry budget and an HTTP timeout so a request cannot hang;
- an optional JSONL response cache (evaluations), keyed by a hash of the request
  (including the per-model parameters), so re-running an evaluation spends no quota;
- every result reports the model used, token usage, latency and finish reason;
  finish_reason "length" means the answer was cut off by max_tokens.

Get a free-tier key at https://aistudio.google.com
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (GEMINI_MODEL, LLM_FALLBACK_MODELS, LLM_MAX_TOKENS, LLM_MODEL_PARAMS,
                        get_api_key)

logger = logging.getLogger(__name__)
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
FALLBACK_MODELS = list(dict.fromkeys([GEMINI_MODEL, *LLM_FALLBACK_MODELS]))


class QuotaExhausted(RuntimeError):
    """Every model in the fallback list is out of quota."""


@dataclass
class LLMResult:
    text: str
    model: str
    usage: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    cached: bool = False
    finish_reason: str | None = None     # "stop", or "length" when max_tokens cut the answer off

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


def _classify(exc: Exception) -> tuple[str, float | None]:
    """('daily_quota' | 'rate_limit' | 'unavailable' | 'transient' | 'fatal', wait in seconds)."""
    msg = str(exc)
    wait = None
    m = re.search(r"retry(?:Delay)?\D{0,12}?(\d+(?:\.\d+)?)\s*s", msg, re.I)
    if m:
        wait = float(m.group(1))
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
        return ("daily_quota" if "PerDay" in msg or "per day" in msg.lower() else "rate_limit"), wait
    if "404" in msg or "no longer available" in msg.lower() or "not found" in msg.lower():
        return "unavailable", wait
    # Network-level failures (dropped TLS connections, DNS hiccups, timeouts) are worth retrying.
    network = type(exc).__name__ in ("APIConnectionError", "APITimeoutError", "ConnectError", "ReadTimeout")
    if (network or any(code in msg for code in ("500", "502", "503", "504"))
            or any(w in msg.lower() for w in ("timeout", "timed out", "connection error", "connection reset"))):
        return "transient", wait
    return "fatal", wait


_CACHES: dict[Path, dict] = {}


def _cache(path: Path) -> dict:
    if path not in _CACHES:
        entries = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    entries[rec["key"]] = rec
        _CACHES[path] = entries
    return _CACHES[path]


def complete(messages: list[dict], system: str | None = None, *, model: str | None = None,
             max_tokens: int = LLM_MAX_TOKENS, temperature: float | None = 0.2,
             json_mode: bool = False, cache_path: Path | None = None,
             wait_on_rate_limit: bool = False, max_retries: int | None = None) -> LLMResult:
    full = ([{"role": "system", "content": system}] if system else []) + messages
    models = [model] if model else FALLBACK_MODELS
    key = hashlib.sha256(json.dumps(
        {"m": models, "msg": full, "t": temperature, "max": max_tokens, "json": json_mode,
         "params": {m: LLM_MODEL_PARAMS.get(m, {}) for m in models}},
        sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if cache_path is not None and key in _cache(cache_path):
        rec = _cache(cache_path)[key]
        return LLMResult(rec["text"], rec["model"], rec.get("usage", {}), rec.get("latency_ms", 0.0), True,
                         rec.get("finish_reason"))

    api_key = get_api_key()
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set. Add it to a .env file (see .env.example).\n"
                         "Get a free key (no credit card) at: https://aistudio.google.com")
    from openai import OpenAI
    # Evaluations may wait out rate limits; an interactive request must answer within the
    # host's time limit (Vercel: 60 s), so it gets a short timeout and few retries.
    if max_retries is None:
        max_retries = 6 if wait_on_rate_limit else 2
    client = OpenAI(api_key=api_key, base_url=BASE_URL, max_retries=0,
                    timeout=120.0 if wait_on_rate_limit else 25.0)
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    if temperature is not None:
        extra["temperature"] = temperature

    for m in models:
        for attempt in range(max_retries):
            try:
                t0 = time.perf_counter()
                resp = client.chat.completions.create(model=m, messages=full, max_tokens=max_tokens,
                                                      **extra, **LLM_MODEL_PARAMS.get(m, {}))
                usage = resp.usage.model_dump() if getattr(resp, "usage", None) else {}
                usage = {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
                finish = resp.choices[0].finish_reason
                if finish == "length":
                    logger.warning("%s stopped at max_tokens=%d; the answer is truncated", m, max_tokens)
                result = LLMResult(resp.choices[0].message.content or "", m, usage,
                                   round((time.perf_counter() - t0) * 1000, 1), False, finish)
                if cache_path is not None:
                    rec = {"key": key, **asdict(result)}
                    rec.pop("cached")
                    _cache(cache_path)[key] = rec
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_path, "a", encoding="utf-8", newline="\n") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                return result
            except Exception as exc:
                kind, wait = _classify(exc)
                if kind == "fatal":
                    raise
                if kind in ("daily_quota", "unavailable") or (kind == "rate_limit" and not wait_on_rate_limit):
                    logger.warning("%s on %s; trying the next model", kind, m)
                    break
                delay = min(wait if wait is not None else 2 ** attempt * 2, 90 if wait_on_rate_limit else 4)
                logger.warning("%s on %s; retrying in %.0f s", kind, m, delay)
                time.sleep(delay)
    raise QuotaExhausted("all Gemini models are out of quota or unavailable")
