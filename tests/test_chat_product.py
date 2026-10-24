"""
/api/chat product behaviour added in 2.6.0, with every model call stubbed: small talk gets a
fixed reply with no retrieval; choosing a model is an administrative option; a failing model is
never replaced by another one; reasoning blocks from OpenAI-compatible models are removed.
"""
import pytest
from fastapi.testclient import TestClient

import server
import src.pipeline as pipeline
from src.config import GEMINI_MODEL, LLM_FALLBACK_MODELS
from src.guardrails import CRISIS_REPLY
from src.llm.client import LLMResult

client = TestClient(server.app)
CASE = {"age": 60, "shape": 4, "margin": 5, "density": 3}


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setattr(server, "chat_limiter", server._RateLimiter(per_minute=1000, per_day=1000))
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key-marker-for-tests-0000")
    for name in ("LLM_USER_MODEL_SELECTION", "LLM_DEFAULT_MODEL", "EXTRA_LLM_BASE_URL",
                 "EXTRA_LLM_MODELS", "EXTRA_LLM_API_KEY", "EXTRA_LLM_PROVIDER"):
        monkeypatch.delenv(name, raising=False)


def _stub(monkeypatch, name, reply="A cyst is a fluid-filled sac [1].", error=None):
    calls = []

    def fake(messages, system=None, **kwargs):
        calls.append(kwargs)
        if error is not None:
            raise error
        return LLMResult(reply, kwargs.get("model") or GEMINI_MODEL, {}, 1.0, finish_reason="stop")
    monkeypatch.setattr(server, name, fake)
    return calls


def _chat(question, **extra):
    return client.post("/api/chat", json={**CASE, "question": question, **extra})


@pytest.mark.parametrize("greeting", ["hi", "Hello!", "hey there", "good morning", "Thanks!",
                                      "thank you so much", "ok", "Assalam o alaikum"])
def test_small_talk_gets_a_fixed_reply_without_retrieval_or_a_model_call(monkeypatch, greeting):
    gemini = _stub(monkeypatch, "_gemini_complete")
    monkeypatch.setattr(pipeline, "rag_query", lambda *a, **k: pytest.fail("retrieval ran"))
    r = _chat(greeting)
    assert r.status_code == 200
    body = r.json()
    assert body["response"] == server.SMALL_TALK_REPLY and body["route"] == "small_talk"
    assert body["passages"] == [] and body["citations"] == [] and body["sources"] == []
    assert body["generation"]["status"] == "fixed" and body["generation"]["model"] is None
    assert body["prediction"]["probability"] > 0 and gemini == []


@pytest.mark.parametrize("message", ["hi, what does a spiculated margin mean?",
                                     "hello, is BI-RADS 4 bad?", "thanks, and what is a cyst?"])
def test_a_greeting_with_a_question_takes_the_guarded_pipeline(monkeypatch, message):
    gemini = _stub(monkeypatch, "_gemini_complete")
    body = _chat(message).json()
    assert "route" not in body and len(gemini) == 1 and body["generation"]["status"] == "generated"


def test_a_greeting_with_a_crisis_message_gets_the_crisis_reply(monkeypatch):
    _stub(monkeypatch, "_gemini_complete")
    body = _chat("hi, I keep having thoughts about killing myself").json()
    assert body["response"] == CRISIS_REPLY and "route" not in body


def test_model_choice_is_off_by_default_and_said_so(monkeypatch):
    _stub(monkeypatch, "_gemini_complete")
    llm = client.get("/health").json()["llm"]
    assert llm["user_selectable"] is False and llm["default_model"] is None
    r = _chat("What is a cyst?", model=LLM_FALLBACK_MODELS[0])
    assert r.status_code == 422 and "turned off" in r.text


def test_the_administrator_default_model_answers_when_no_model_is_named(monkeypatch):
    monkeypatch.setenv("LLM_DEFAULT_MODEL", LLM_FALLBACK_MODELS[0])
    gemini = _stub(monkeypatch, "_gemini_complete")
    body = _chat("What is a cyst?").json()
    assert [c.get("model") for c in gemini] == [LLM_FALLBACK_MODELS[0]]
    assert body["generation"]["model"] == LLM_FALLBACK_MODELS[0] and body["generation"]["evaluated"] is False
    assert client.get("/health").json()["llm"]["default_model"] == LLM_FALLBACK_MODELS[0]


def test_a_misconfigured_default_is_reported_and_never_replaced(monkeypatch):
    monkeypatch.setenv("LLM_DEFAULT_MODEL", "no-such-model")
    gemini = _stub(monkeypatch, "_gemini_complete")
    r = _chat("What is a cyst?")
    assert r.status_code == 503 and "no-such-model" in r.text and gemini == []
    # crisis replies never depend on the model configuration
    crisis = _chat("I keep having thoughts about killing myself.")
    assert crisis.status_code == 200 and crisis.json()["response"] == CRISIS_REPLY


def test_a_failing_extra_provider_is_not_replaced_by_gemini(monkeypatch):
    monkeypatch.setenv("LLM_USER_MODEL_SELECTION", "1")
    monkeypatch.setenv("EXTRA_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("EXTRA_LLM_MODELS", "llama-test")
    gemini = _stub(monkeypatch, "_gemini_complete")
    _stub(monkeypatch, "_extra_complete", error=RuntimeError("provider down"))
    body = _chat("What is a cyst?", model="llama-test").json()
    assert body["generation"]["status"] == "unavailable" and gemini == []
    assert "provider down" not in str(body)


def test_reasoning_blocks_are_removed_from_extra_provider_answers():
    assert server._strip_reasoning("<think>step 1, step 2</think>\nA cyst is fluid [1].") == "A cyst is fluid [1]."
    assert server._strip_reasoning("<THINKING>x</THINKING>Answer") == "Answer"
    assert server._strip_reasoning("No tags [2].") == "No tags [2]."
