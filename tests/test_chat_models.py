"""
/api/chat model choice (backend 2.4.0): /health lists the models a request may name, a request may
name one, and the evaluated default chain stays the default. Every model call is stubbed (the
Gemini and extra-provider functions in server.py, and openai.OpenAI); no test reaches a provider.
"""
import types

import httpx
import openai
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import server
import src.llm.client as llm_client
import src.pipeline as pipeline
from src.config import GEMINI_MODEL, LLM_FALLBACK_MODELS
from src.guardrails import CRISIS_REPLY
from src.llm.client import FALLBACK_MODELS, LLMResult, QuotaExhausted

client = TestClient(server.app)
CASE = {"age": 60, "shape": 4, "margin": 5, "density": 3}
CRISIS = "I keep having thoughts about killing myself while I wait for results."
FALLBACK = LLM_FALLBACK_MODELS[0]
EXTRA_ENV = ("EXTRA_LLM_BASE_URL", "EXTRA_LLM_MODELS", "EXTRA_LLM_API_KEY", "EXTRA_LLM_PROVIDER")
REAL_EXTRA_COMPLETE = server._extra_complete            # saved before any test replaces it
GEMINI_KEY_MARKER = "gemini-key-marker-for-tests-0000"
EXTRA_KEY_MARKER = "extra-key-marker-for-tests-0000"


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    """Fresh rate limiter, a fake Gemini key, no extra provider, and nothing that can reach a
    provider: unstubbed model calls are recorded and fail the test."""
    monkeypatch.setattr(server, "chat_limiter", server._RateLimiter(per_minute=1000, per_day=1000))
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY_MARKER)
    for name in EXTRA_ENV:
        monkeypatch.delenv(name, raising=False)
    stray = []

    def unexpected(name):
        def call(*args, **kwargs):
            stray.append(name)
            raise AssertionError(f"unstubbed call to {name}")
        return call
    monkeypatch.setattr(server, "_gemini_complete", unexpected("_gemini_complete"))
    monkeypatch.setattr(server, "_extra_complete", unexpected("_extra_complete"))
    monkeypatch.setattr(openai, "OpenAI", unexpected("openai.OpenAI"))
    yield
    assert stray == [], f"a model call was not stubbed: {stray}"


def _stub(monkeypatch, name, reply="A cyst is a fluid-filled sac [1].", reports=None, error=None):
    """Replace server.<name> with a recorder that answers as `reports` (default: the model it was
    asked for, else GEMINI_MODEL), or raises `error`."""
    calls = []

    def fake(messages, system=None, **kwargs):
        calls.append({"system": system, **kwargs})
        if error is not None:
            raise error
        return LLMResult(reply, reports or kwargs.get("model") or GEMINI_MODEL, {}, 1.0, finish_reason="stop")
    monkeypatch.setattr(server, name, fake)
    return calls


def _extra_env(monkeypatch, base_url="http://localhost:11434/v1", models="llama-test,other-test",
               provider="ollama", key=None):
    for name, value in (("EXTRA_LLM_BASE_URL", base_url), ("EXTRA_LLM_MODELS", models),
                        ("EXTRA_LLM_PROVIDER", provider), ("EXTRA_LLM_API_KEY", key)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


def _chat(question="What is a cyst?", **extra):
    return client.post("/api/chat", json={**CASE, "question": question, **extra})


def _ok(question="What is a cyst?", **extra):
    r = _chat(question, **extra)
    assert r.status_code == 200, r.text
    return r.json()


def _models():
    return client.get("/health").json()["llm"]["models"]


def _fake_openai(monkeypatch, reply="Answer from the extra provider [1].", error=None):
    """openai.OpenAI stand-in: records the client settings and each request."""
    record = {"init": [], "create": []}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            record["init"].append(kwargs)
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            record["create"].append(kwargs)
            if error is not None:
                raise error
            usage = types.SimpleNamespace(model_dump=lambda: {"prompt_tokens": 11, "completion_tokens": 7,
                                                              "total_tokens": 18, "extra_field": 1})
            choice = types.SimpleNamespace(message=types.SimpleNamespace(content=reply), finish_reason="stop")
            return types.SimpleNamespace(choices=[choice], usage=usage, model="provider-side-name")
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    return record


def _status_error(cls, code):
    response = httpx.Response(code, request=httpx.Request("POST", "http://localhost:11434/v1/chat/completions"))
    return cls(f"Error code: {code}", response=response, body=None)


# ── /health ───────────────────────────────────────────────────────────────────
def test_router_is_installed_in_the_pipeline_and_the_client():
    assert pipeline.complete is server._routed_complete
    assert llm_client.complete is server._routed_complete
    assert server._ORIGINAL_COMPLETE.__module__ == "src.llm.client"


def test_health_lists_the_gemini_models_with_exactly_one_evaluated():
    body = client.get("/health").json()
    assert body["version"] == "2.4.0" and client.get("/openapi.json").json()["info"]["version"] == "2.4.0"
    llm = body["llm"]
    # the 2.3.1 fields are unchanged
    assert llm["provider"] == "gemini" and llm["model"] == GEMINI_MODEL
    assert llm["fallback_models"] == list(LLM_FALLBACK_MODELS) and llm["key_set"] is True
    models = llm["models"]
    assert [m["id"] for m in models] == FALLBACK_MODELS and models[0]["id"] == GEMINI_MODEL
    assert all(set(m) == {"id", "provider", "evaluated", "available"} for m in models)
    assert all(m["provider"] == "gemini" and m["available"] is True for m in models)
    assert [m["id"] for m in models if m["evaluated"]] == [GEMINI_MODEL]
    assert GEMINI_KEY_MARKER not in str(body)


def test_gemini_models_are_unavailable_without_a_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    models = _models()
    assert models and all(m["available"] is False for m in models)
    assert [m["id"] for m in models if m["evaluated"]] == [GEMINI_MODEL]


def test_extra_provider_is_listed_only_when_its_env_is_set(monkeypatch):
    assert {m["provider"] for m in _models()} == {"gemini"}
    _extra_env(monkeypatch, models=None)                      # a base URL alone names no model
    assert {m["provider"] for m in _models()} == {"gemini"}
    _extra_env(monkeypatch, base_url=None, provider=None)     # models without a base URL: not usable
    extra = [m for m in _models() if m["provider"] != "gemini"]
    assert extra == [{"id": "llama-test", "provider": "custom", "evaluated": False, "available": False},
                     {"id": "other-test", "provider": "custom", "evaluated": False, "available": False}]
    _extra_env(monkeypatch, models=f"llama-test, {GEMINI_MODEL}, llama-test", key=EXTRA_KEY_MARKER)
    body = client.get("/health").json()
    extra = [m for m in body["llm"]["models"] if m["provider"] != "gemini"]
    # a Gemini id is not listed twice; duplicates are dropped; the key is never served
    assert extra == [{"id": "llama-test", "provider": "ollama", "evaluated": False, "available": True}]
    assert [m["id"] for m in body["llm"]["models"] if m["evaluated"]] == [GEMINI_MODEL]
    assert EXTRA_KEY_MARKER not in str(body) and "localhost:11434" not in str(body)


# ── Validation ────────────────────────────────────────────────────────────────
def _assert_model_422(r, model_id):
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert detail[0]["loc"] == ["body", "model"] and model_id in detail[0]["msg"]
    assert GEMINI_KEY_MARKER not in r.text and EXTRA_KEY_MARKER not in r.text


def test_unknown_or_unavailable_model_is_refused_with_422(monkeypatch):
    _assert_model_422(_chat(model="no-such-model"), "no-such-model")
    _extra_env(monkeypatch, base_url=None, key=EXTRA_KEY_MARKER)      # listed, but not usable
    r = _chat(model="llama-test")
    _assert_model_422(r, "llama-test")
    assert "not available" in r.json()["detail"][0]["msg"]
    monkeypatch.delenv("GEMINI_API_KEY")
    _assert_model_422(_chat(model=GEMINI_MODEL), GEMINI_MODEL)
    assert _chat(model="").status_code == 422                          # empty id: plain validation error


def test_crisis_reply_never_depends_on_the_model_choice(monkeypatch):
    gemini = _stub(monkeypatch, "_gemini_complete")
    for model in (FALLBACK, "no-such-model"):                          # a valid pick and a stale one
        body = _ok(CRISIS, model=model)
        assert body["response"] == CRISIS_REPLY and body["guardrail"] == "crisis"
        g = body["generation"]
        assert g["status"] == "fixed" and g["model"] is None and g["evaluated"] is False
        assert g["requested_model"] == model and body["passages"] == []
    assert gemini == []


# ── Routing ───────────────────────────────────────────────────────────────────
def test_no_model_uses_the_default_chain_and_is_evaluated_only_for_the_default(monkeypatch):
    calls = _stub(monkeypatch, "_gemini_complete", reports=GEMINI_MODEL)
    g = _ok()["generation"]
    assert len(calls) == 1 and calls[0].get("model") is None and calls[0]["system"]
    assert g == {"provider": "gemini", "configured_model": GEMINI_MODEL, "model": GEMINI_MODEL,
                 "requested_model": None, "evaluated": True, "status": "generated"}
    # the default chain fell back to another model: generated, but not the evaluated model
    _stub(monkeypatch, "_gemini_complete", reports=FALLBACK)
    g = _ok()["generation"]
    assert g["status"] == "generated" and g["model"] == FALLBACK and g["evaluated"] is False


def test_picking_a_gemini_fallback_model_uses_it_alone(monkeypatch):
    calls = _stub(monkeypatch, "_gemini_complete")
    body = _ok(model=FALLBACK)
    assert [c["model"] for c in calls] == [FALLBACK]
    g = body["generation"]
    assert g["status"] == "generated" and g["provider"] == "gemini" and g["model"] == FALLBACK
    assert g["requested_model"] == FALLBACK and g["evaluated"] is False
    assert body["trace"]["model"] == FALLBACK


def test_picking_the_default_model_by_name_is_evaluated(monkeypatch):
    calls = _stub(monkeypatch, "_gemini_complete")
    g = _ok(model=GEMINI_MODEL)["generation"]
    assert [c["model"] for c in calls] == [GEMINI_MODEL]
    assert g["model"] == GEMINI_MODEL and g["requested_model"] == GEMINI_MODEL and g["evaluated"] is True


def test_extra_provider_model_routes_to_the_extra_function(monkeypatch):
    _extra_env(monkeypatch)
    gemini = _stub(monkeypatch, "_gemini_complete")
    extra = _stub(monkeypatch, "_extra_complete", reply="A cyst is a fluid-filled sac [1].")
    body = _ok(model="llama-test")
    assert gemini == [] and [c["model"] for c in extra] == ["llama-test"] and extra[0]["system"]
    g = body["generation"]
    assert g == {"provider": "ollama", "configured_model": GEMINI_MODEL, "model": "llama-test",
                 "requested_model": "llama-test", "evaluated": False, "status": "generated"}
    assert [c["n"] for c in body["citations"]] == [1]         # the same output checks apply


def test_extra_complete_sends_one_openai_compatible_request(monkeypatch):
    _extra_env(monkeypatch)
    record = _fake_openai(monkeypatch)
    res = REAL_EXTRA_COMPLETE([{"role": "user", "content": "hi"}], "SYSTEM", model="llama-test",
                              max_tokens=321, temperature=0.4, cache_path="ignored", wait_on_rate_limit=True)
    assert record["init"] == [{"base_url": "http://localhost:11434/v1", "api_key": "not-needed",
                               "timeout": 25, "max_retries": 1}]
    sent = record["create"][0]
    assert sent["model"] == "llama-test" and sent["max_tokens"] == 321 and sent["temperature"] == 0.4
    assert sent["messages"] == [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "hi"}]
    assert isinstance(res, LLMResult) and res.text == "Answer from the extra provider [1]."
    assert res.model == "llama-test" and res.finish_reason == "stop" and res.cached is False
    assert res.usage == {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
    assert res.latency_ms >= 0
    _extra_env(monkeypatch, key=EXTRA_KEY_MARKER)
    REAL_EXTRA_COMPLETE([{"role": "user", "content": "hi"}], None, model="other-test")
    assert record["init"][-1]["api_key"] == EXTRA_KEY_MARKER
    assert record["create"][-1]["messages"] == [{"role": "user", "content": "hi"}]


def test_extra_provider_429_is_quota_exhausted(monkeypatch):
    _extra_env(monkeypatch)
    monkeypatch.setattr(server, "_extra_complete", REAL_EXTRA_COMPLETE)
    _fake_openai(monkeypatch, error=_status_error(openai.RateLimitError, 429))
    with pytest.raises(QuotaExhausted):
        REAL_EXTRA_COMPLETE([{"role": "user", "content": "hi"}], None, model="llama-test")
    body = _ok(model="llama-test")
    g = body["generation"]
    assert g["status"] == "quota_exhausted" and g["provider"] == "ollama" and g["model"] is None
    assert g["requested_model"] == "llama-test" and g["evaluated"] is False and body["passages"]


def test_other_extra_provider_errors_are_reported_as_unavailable(monkeypatch):
    _extra_env(monkeypatch)
    monkeypatch.setattr(server, "_extra_complete", REAL_EXTRA_COMPLETE)
    _fake_openai(monkeypatch, error=_status_error(openai.InternalServerError, 500))
    body = _ok(model="llama-test")
    assert body["generation"]["status"] == "unavailable" and "Error code" not in str(body)


def test_agent_modes_use_the_picked_model_and_are_never_evaluated(monkeypatch):
    calls = _stub(monkeypatch, "_gemini_complete", reply="## FINAL ANSWER\nA cyst is a fluid-filled sac.")
    body = _ok(mode="cot", model=FALLBACK)
    assert body["mode"] == "cot" and calls and all(c["model"] == FALLBACK for c in calls)
    assert body["generation"]["model"] == FALLBACK and body["generation"]["evaluated"] is False
    # the default model in an unevaluated mode is not "evaluated" either
    calls = _stub(monkeypatch, "_gemini_complete", reply="## FINAL ANSWER\nA cyst is a fluid-filled sac.")
    body = _ok(mode="cot")
    assert body["mode"] == "cot" and all(c.get("model") is None for c in calls)
    assert body["generation"]["model"] == GEMINI_MODEL and body["generation"]["evaluated"] is False


# ── The requested model does not outlive its request ──────────────────────────
def test_requested_model_is_cleared_after_each_request(monkeypatch):
    calls = _stub(monkeypatch, "_gemini_complete")
    assert _ok(model=FALLBACK)["generation"]["model"] == FALLBACK
    second = _ok()
    assert [c.get("model") for c in calls] == [FALLBACK, None]
    assert second["generation"]["model"] == GEMINI_MODEL and second["generation"]["requested_model"] is None
    assert server._requested_model.get() is None


def test_requested_model_is_reset_in_the_calling_context_even_on_errors(monkeypatch):
    calls = _stub(monkeypatch, "_gemini_complete")
    req = server.ChatRequest(**CASE, question="What is a cyst?", model=FALLBACK)
    assert server.chat(req).status_code == 200 and [c["model"] for c in calls] == [FALLBACK]
    assert server._requested_model.get() is None

    def broken(*args, **kwargs):
        raise RuntimeError("pipeline failed")
    monkeypatch.setattr(pipeline, "run_pipeline", broken)
    with pytest.raises(HTTPException) as err:
        server.chat(req)
    assert err.value.status_code == 500 and server._requested_model.get() is None
