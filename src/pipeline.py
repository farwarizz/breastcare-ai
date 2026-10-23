"""
src/pipeline.py — Central orchestrator for BreastCare AI.

Supports three pipeline modes:
    'single'       — Original 8-step pipeline (ML → SHAP → RAG → LLM → Safety)
    'multi_agent'  — 3 specialized LLM agents (Analyst → Clinician → Safety)
    'cot'          — Chain-of-Thought (structured step-by-step reasoning)

All modes share steps 1-5. Modes differ only in LLM generation (steps 6-8).
Audit logging is automatic when enabled.

The UI calls run_pipeline() only; it never touches ML/SHAP/RAG/agents directly.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    HIGH_RISK_THRESHOLD, CHUNK_TOP_K,
    AUDIT_ENABLED, MULTI_AGENT_ENABLED,
)
from src.ml.predict  import predict
from src.ml.explain  import explain
from src.rag.retriever      import rag_query, query_by_source
from src.rag.prompt_builder import (build_prompt, build_retrieval_query, parse_citations,
                                    select_retrieval_query, SYSTEM_PROMPT)
from src.llm.client  import QuotaExhausted, complete
from src.safety      import (HIGH_RISK_DISCLAIMER, STANDARD_DISCLAIMER, apply_patterns, filter_response,
                             has_substance, remove_internal_text, remove_unrequested_estimate,
                             strip_invalid_citations)
from src.guardrails  import CRISIS_REPLY, check_input

UNAVAILABLE_REPLY = ("The assistant could not generate an answer just now. The estimate, the "
                     "prediction set and the explanation above are unaffected; please try the "
                     "question again in a moment.")
NO_ANSWER_REPLY = ("I can't help with that part. I can explain breast-imaging terms, what the estimate "
                   "shown on screen means, or what usually happens after a mammogram.")
# A model answer that gives crisis lines is a crisis reply written by the model: the input
# guard missed the risk but the model saw it. Such answers are replaced as a whole by the
# fixed crisis reply, because in held-out v2 they also quoted the estimate, cited a missing
# document or repeated internal instructions.
SUPPORT_LINES_PRESENT = re.compile(r"\b988\b|116 123|findahelpline|suicide (&|and) crisis|crisis (line|lifeline)",
                                   re.IGNORECASE)

logger = logging.getLogger(__name__)

# Lazy-loaded singleton audit DB
_audit_db = None


def _get_audit_db():
    """Lazy-load the audit database."""
    global _audit_db
    if _audit_db is None and AUDIT_ENABLED:
        try:
            from src.audit.database import AuditDB
            _audit_db = AuditDB()
        except Exception as exc:
            logger.warning("Audit DB init failed: %s", exc)
    return _audit_db


def run_pipeline(
    features: dict,
    user_question: str | None = None,
    mode: str = "single",
    session_id: str = "",
    llm_kwargs: dict | None = None,
) -> dict:
    """
    Run the complete BreastCare AI pipeline.

    Order: input guardrail → prediction + SHAP → retrieval → generation → output
    filter. The guardrail answers crisis, injection, privacy and off-topic
    messages with fixed text and never calls the LLM for them.

    llm_kwargs is passed to the LLM client (the evaluations use it for their
    response cache and to wait out rate limits).

    Returns a dict with prediction, explanation, llm_response, citations,
    sources_used, chunks, query_used, guardrail, trace (stage timings, model,
    token usage) and, for agent modes, reasoning_trace / agent_traces.
    """
    timer = _Timer()
    decision = check_input(user_question)
    with timer("predict"):
        prediction = predict(features)
    with timer("explain"):
        explanation = explain(features)

    if not decision.calls_llm:
        result = {
            "prediction": prediction, "explanation": explanation,
            "llm_response": {"main_text": decision.fixed_reply,
                             "disclaimer": STANDARD_DISCLAIMER.strip(),
                             "high_risk": False, "filters_fired": 0},
            "citations": [], "sources_used": [], "chunks": [], "query_used": None,
            "guardrail": decision.category, "mode": "guardrail",
        }
    else:
        retrieval_query = select_retrieval_query(explanation, user_question)
        with timer("retrieve"):
            chunks = rag_query(retrieval_query, k=CHUNK_TOP_K)
            if prediction["probability"] >= HIGH_RISK_THRESHOLD:
                existing = {c.get("chunk_id") for c in chunks}
                chunks += [c for c in query_by_source("safety_disclaimer", retrieval_query, k=1)
                           if c.get("chunk_id") not in existing]
        # The chain-of-thought and multi-agent modes are not covered by the red-team
        # evaluation and do not carry the guardrail's policy notes, so a question the
        # guardrail flagged always takes the evaluated single-call path.
        requested = mode
        if mode in ("multi_agent", "cot") and decision.policy_note:
            mode = "single"
        with timer("generate"):
            if mode in ("multi_agent", "cot"):
                result = _run_agent_pipeline(features, prediction, explanation, chunks,
                                             retrieval_query, user_question, mode, llm_kwargs)
            else:
                result = _run_single_pipeline(features, prediction, explanation, chunks,
                                              retrieval_query, user_question, decision.policy_note,
                                              llm_kwargs)
        result["guardrail"] = decision.category
        if requested != result["mode"]:
            result["requested_mode"] = requested
        llm_info = result.get("_llm") or {}
        answered = bool(llm_info.get("model")) and not llm_info.get("empty_response")
        if answered:
            _postprocess_answer(result, decision.category, user_question, prediction, len(chunks))
        text = (result.get("llm_response") or {}).get("main_text", "")
        # Citations come only from a generated answer (not from the quota/error fallbacks or a
        # replaced crisis reply).
        cites = (parse_citations(text, len(chunks)) if answered and result["guardrail"] != "crisis"
                 else {"cited": [], "invalid": []})
        result["citations"] = [{"n": n, "source": chunks[n - 1].get("source"),
                                "section": chunks[n - 1].get("section")} for n in cites["cited"]]
        result.setdefault("invalid_citations", cites["invalid"])

    result["trace"] = {"timings_ms": timer.times, **result.pop("_llm", {})}
    _log_to_audit(session_id, features, prediction, explanation, result, result["mode"])
    return result


def _postprocess_answer(result: dict, input_category: str, question: str | None,
                        prediction: dict, n_documents: int) -> None:
    """Deterministic checks on a model answer (single or agent mode), in place:
    1. an answer that gives crisis lines is replaced as a whole by the fixed crisis reply;
    2. otherwise sentences repeating internal instructions, and sentences about the case
       estimate when the question did not ask about it, are removed, and citation markers
       pointing to no provided document are stripped. Nothing is added back afterwards."""
    lr = result["llm_response"]
    text = lr.get("main_text", "")
    if SUPPORT_LINES_PRESENT.search(text):
        result["llm_response"] = {**_fixed_reply(CRISIS_REPLY, 0.0), "crisis_backstop": True}
        result["guardrail"], result["input_guardrail"] = "crisis", input_category
        result["invalid_citations"] = parse_citations(text, n_documents)["invalid"]
        result.pop("reasoning_trace", None)
        result.pop("agent_traces", None)
        return
    invalid = parse_citations(text, n_documents)["invalid"]
    text, leaked = remove_internal_text(text)
    text, unrequested = remove_unrequested_estimate(text, question, prediction["probability"])
    text = strip_invalid_citations(text, n_documents)
    if not has_substance(text):
        text = NO_ANSWER_REPLY
    lr["main_text"] = text
    lr["full_text"] = text + "\n\n" + lr.get("disclaimer", "")
    lr["sanitised"] = {"internal_sentences_removed": leaked, "estimate_sentences_removed": unrequested,
                       "invalid_citations_removed": len(invalid)}
    result["invalid_citations"] = invalid


class _Timer:
    """Collects wall-clock milliseconds per pipeline stage."""
    def __init__(self):
        self.times: dict[str, float] = {}

    def __call__(self, name: str):
        import contextlib, time

        @contextlib.contextmanager
        def block():
            t0 = time.perf_counter()
            yield
            self.times[name] = round((time.perf_counter() - t0) * 1000, 1)
        return block()


def _run_single_pipeline(
    features, prediction, explanation, chunks,
    retrieval_query, user_question, policy_note=None, llm_kwargs=None,
) -> dict:
    """Single LLM call with the grounded prompt, then the output filter."""
    messages = build_prompt(prediction, explanation, chunks, features, user_question, policy_note)
    try:
        res = complete(messages, SYSTEM_PROMPT, **(llm_kwargs or {}))
        llm = {"model": res.model, "usage": res.usage, "llm_latency_ms": res.latency_ms,
               "finish_reason": res.finish_reason}
        if not res.text.strip():                  # an empty completion is not an answer
            llm["empty_response"] = True
            safe_response = _fixed_reply(UNAVAILABLE_REPLY, prediction["probability"])
        else:
            safe_response = filter_response(res.text, prediction["probability"])
            if res.truncated:
                safe_response["main_text"] += " …\n\n(The answer was cut off; please ask again for the rest.)"
    except QuotaExhausted:
        llm = {"model": None, "quota_exhausted": True}
        safe_response = _passages_reply(chunks, prediction["probability"])
    except Exception as exc:                      # auth, bad request, network: never show raw errors
        # The details stay in the server log; the response (and its trace) only says the
        # provider failed, without exception names or messages.
        logger.exception("LLM call failed: %s", exc)
        llm = {"model": None, "error": "provider_error"}
        safe_response = _fixed_reply(UNAVAILABLE_REPLY, prediction["probability"])

    return {
        "prediction":   prediction,
        "explanation":  explanation,
        "llm_response": safe_response,
        "sources_used": list(dict.fromkeys(c.get("source", "") for c in chunks)),
        "chunks":       chunks,
        "query_used":   retrieval_query,
        "mode":         "single",
        "_llm":         llm,
    }


def _fixed_reply(text: str, probability: float) -> dict:
    """A reply that did not come from the LLM, with the usual disclaimers attached."""
    high = probability >= HIGH_RISK_THRESHOLD
    disclaimer = (HIGH_RISK_DISCLAIMER + STANDARD_DISCLAIMER) if high else STANDARD_DISCLAIMER
    return {"main_text": text, "disclaimer": disclaimer.strip(), "high_risk": high,
            "filters_fired": 0, "full_text": text + disclaimer}


def _passages_reply(chunks: list[dict], probability: float) -> dict:
    """When every model is out of quota: the retrieved passages verbatim (never the prompt,
    which holds internal instructions), labelled as unsummarised knowledge-base text."""
    passages = "\n\n".join(f"**{c.get('source', '').replace('_', ' ')} › {c.get('section', '')}**\n{c.get('text', '')}"
                           for c in chunks) or "No passages were retrieved."
    text = ("⚠️ The AI model's free quota is used up for now, so here are the most relevant "
            "knowledge-base passages, unedited:\n\n" + passages +
            "\n\nPlease try again later for a written answer.")
    return _fixed_reply(text, probability)


def _run_agent_pipeline(
    features, prediction, explanation, chunks,
    retrieval_query, user_question, mode, llm_kwargs=None,
) -> dict:
    """Multi-agent or Chain-of-Thought pipeline (optional, unevaluated). Falls back to the
    evaluated single-call pipeline whenever an agent step fails."""
    from src.agents.orchestrator import MultiAgentPipeline

    context = {
        "features": features,
        "prediction": prediction,
        "explanation": explanation,
        "chunks": chunks,
        "user_question": user_question or "",
    }
    try:
        agent_result = MultiAgentPipeline().run(context, mode=mode)
    except Exception as exc:
        agent_result = {"success": False, "reasoning_trace": f"{type(exc).__name__}"}
    if not agent_result.get("success"):
        logger.error("Agent pipeline (%s) failed, using the single-call pipeline", mode)
        return _run_single_pipeline(features, prediction, explanation, chunks,
                                    retrieval_query, user_question, None, llm_kwargs)

    return {
        "prediction":       prediction,
        "explanation":      explanation,
        "llm_response": {
            "main_text":  agent_result.get("main_text", ""),
            "disclaimer": agent_result.get("disclaimer", ""),
            "high_risk":  agent_result.get("high_risk", False),
        },
        "sources_used":     list(dict.fromkeys(c.get("source", "") for c in chunks)),
        "chunks":           chunks,
        "query_used":       retrieval_query,
        "mode":             mode,
        # The step-by-step trace is shown in the UI, so it passes the same output filter.
        "reasoning_trace":  apply_patterns(agent_result.get("reasoning_trace", ""))[0],
        "agent_traces":     agent_result.get("agent_traces", []),
        "_llm":             {"model": agent_result.get("model") or "agents"},
    }


def run_pipeline_ml_only(features: dict, session_id: str = "") -> dict:
    """
    Run only ML + SHAP — no LLM call.
    Used when API key is not set or in fast preview mode.
    """
    prediction  = predict(features)
    explanation = explain(features)
    query       = build_retrieval_query(explanation)
    chunks      = rag_query(query, k=CHUNK_TOP_K)

    # Audit
    _log_to_audit(session_id, features, prediction, explanation, {}, "ml_only")

    return {
        "prediction":   prediction,
        "explanation":  explanation,
        "llm_response": None,
        "sources_used": [c.get("source", "") for c in chunks],
        "chunks":       chunks,
        "query_used":   query,
        "mode":         "ml_only",
    }


def _log_to_audit(
    session_id: str, features: dict, prediction: dict,
    explanation: dict, result: dict, mode: str,
) -> None:
    """Log prediction to audit database if enabled."""
    if not AUDIT_ENABLED or not session_id:
        return

    try:
        db = _get_audit_db()
        if db:
            db.log_prediction(
                session_id=session_id,
                features=features,
                prediction=prediction,
                explanation=explanation,
                model_used=prediction.get("model", "unknown"),
                pipeline_mode=mode,
            )
    except Exception as exc:
        logger.warning("Audit logging failed: %s", exc)
