"""
/api/chat response contract (backend 2.3.x): every request runs the guarded pipeline, the
response says who produced the reply (`generation.status`) and lists the retrieved passages
(`passages`, with the passage text). The language model is stubbed; no test calls Gemini.
"""
import pytest
from fastapi.testclient import TestClient

import server
import src.pipeline as pipeline
from src.guardrails import CRISIS_REPLY
from src.llm.client import LLMResult, QuotaExhausted

client = TestClient(server.app)
CASE = {"age": 60, "shape": 4, "margin": 5, "density": 3}


@pytest.fixture(autouse=True)
def fresh_limiter(monkeypatch):
    monkeypatch.setattr(server, "chat_limiter", server._RateLimiter(per_minute=1000, per_day=1000))


def _ask(question):
    r = client.post("/api/chat", json={**CASE, "question": question})
    assert r.status_code == 200
    return r.json()


def _model(monkeypatch, behaviour):
    calls = []

    def fake_complete(messages, system=None, **kw):
        calls.append(messages)
        if isinstance(behaviour, Exception):
            raise behaviour
        return LLMResult(behaviour, "gemini-test-model", {}, 1.0, finish_reason="stop")
    monkeypatch.setattr(pipeline, "complete", fake_complete)
    return calls


def test_generated_answer_lists_retrieved_passages_with_cited_flags(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-used")
    _model(monkeypatch, "A spiculated margin has lines radiating from the mass [1].")
    body = _ask("What is a spiculated margin?")
    assert body["generation"]["status"] == "generated" and body["generation"]["model"] == "gemini-test-model"
    assert body["generation"]["provider"] == "gemini"
    assert [c["n"] for c in body["citations"]] == [1]
    passages = body["passages"]
    assert passages and [p["n"] for p in passages] == list(range(1, len(passages) + 1))
    assert all(p["text"] and p["source"] and "section" in p for p in passages)
    assert [p["n"] for p in passages if p["cited"]] == [1]
    # the cited passage is the one the citation names
    assert (passages[0]["source"], passages[0]["section"]) == (body["citations"][0]["source"],
                                                               body["citations"][0]["section"])


def test_without_a_key_crisis_messages_still_get_the_fixed_crisis_reply(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    calls = _model(monkeypatch, ValueError("GEMINI_API_KEY not set"))
    body = _ask("I keep having thoughts about killing myself while I wait for results.")
    assert body["response"] == CRISIS_REPLY and body["guardrail"] == "crisis"
    assert body["generation"]["status"] == "fixed" and body["passages"] == [] and not calls


def test_without_a_key_other_questions_are_retrieval_only_and_say_so(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    _model(monkeypatch, ValueError("GEMINI_API_KEY not set"))
    body = _ask("What does BI-RADS category 4 mean?")
    g = body["generation"]
    assert g["status"] == "not_configured" and g["model"] is None
    assert body["response"] == server.NOT_CONFIGURED_REPLY and "not an answer" in body["response"]
    assert body["citations"] == [] and body["passages"] and not any(p["cited"] for p in body["passages"])
    # retrieval used the question, not only the case summary
    assert any("4" in (p["section"] or "") or "BI-RADS" in p["text"] for p in body["passages"])
    assert "GEMINI_API_KEY not set" not in str(body["trace"])        # no exception text


def test_quota_and_provider_failures_are_labelled(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-used")
    _model(monkeypatch, QuotaExhausted("all models out of quota"))
    q = _ask("What is a cyst?")
    assert q["generation"]["status"] == "quota_exhausted" and q["passages"]
    _model(monkeypatch, RuntimeError("Error code: 500 provider down"))
    e = _ask("What is a cyst?")
    assert e["generation"]["status"] == "unavailable" and "500" not in str(e)


def test_crisis_backstop_reply_is_fixed_with_no_passages(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-used")
    _model(monkeypatch, "Please call or text 988. Your estimate is 81.5% [0].")
    body = _ask("Part of me thinks this whole situation is pointless.")
    assert body["response"] == CRISIS_REPLY and body["generation"]["status"] == "fixed"
    assert body["passages"] == [] and body["citations"] == [] and body["input_guardrail"] == "ok"
    assert "81.5" not in body["response"]


def test_health_names_the_language_model_but_never_the_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-tests-0000")
    body = client.get("/health").json()
    assert body["llm"]["provider"] == "gemini" and body["llm"]["key_set"] is True and body["llm"]["model"]
    assert "fake-key-for-tests-0000" not in str(body)
