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
from src.rag.prompt_builder import build_retrieval_query, build_prompt
from src.llm.client  import call_llm
from src.safety      import filter_response

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
    use_hybrid_rag: bool = False,
) -> dict:
    """
    Run the complete BreastCare AI pipeline.

    Parameters
    ----------
    features : dict — {age, shape, margin, density}
    user_question : str or None — optional follow-up from chat panel
    mode : 'single' | 'multi_agent' | 'cot'
    session_id : str — audit session ID
    use_hybrid_rag : bool — use GraphRAG hybrid retrieval

    Returns
    -------
    dict with prediction, explanation, llm_response, sources_used, chunks,
    query_used, and optional reasoning_trace / agent_traces.
    """
    # ── Steps 1-2: ML prediction + SHAP explanation ─────────────
    prediction = predict(features)
    explanation = explain(features)

    # ── Step 3: Build retrieval query from SHAP output ──────────
    retrieval_query = build_retrieval_query(explanation, user_question)

    # ── Step 4: Retrieve relevant chunks ────────────────────────
    if use_hybrid_rag:
        try:
            from src.rag.graph_retriever import hybrid_retrieve
            chunks = hybrid_retrieve(retrieval_query, k=CHUNK_TOP_K)
        except Exception as exc:
            logger.warning("Hybrid RAG failed, falling back to vector: %s", exc)
            chunks = rag_query(retrieval_query, k=CHUNK_TOP_K)
    else:
        chunks = rag_query(retrieval_query, k=CHUNK_TOP_K)

    # ── Step 5: Inject safety disclaimer for high-risk ──────────
    if prediction["probability"] >= HIGH_RISK_THRESHOLD:
        safety_chunks = query_by_source("safety_disclaimer", retrieval_query, k=1)
        existing_ids = {c.get("chunk_id") for c in chunks}
        for sc in safety_chunks:
            if sc.get("chunk_id") not in existing_ids:
                chunks.append(sc)

    # ── Steps 6-8: LLM generation (mode-dependent) ─────────────
    if mode in ("multi_agent", "cot"):
        result = _run_agent_pipeline(
            features, prediction, explanation, chunks,
            retrieval_query, user_question, mode,
        )
    else:
        result = _run_single_pipeline(
            features, prediction, explanation, chunks,
            retrieval_query, user_question,
        )

    # ── Audit logging ───────────────────────────────────────────
    _log_to_audit(session_id, features, prediction, explanation, result, mode)

    return result


def _run_single_pipeline(
    features, prediction, explanation, chunks,
    retrieval_query, user_question,
) -> dict:
    """Original 8-step single-pass pipeline."""
    messages = build_prompt(prediction, explanation, chunks, features, user_question)
    raw_response = call_llm(messages)
    safe_response = filter_response(raw_response, prediction["probability"])

    return {
        "prediction":   prediction,
        "explanation":  explanation,
        "llm_response": safe_response,
        "sources_used": list({c.get("source", "") for c in chunks}),
        "chunks":       chunks,
        "query_used":   retrieval_query,
        "mode":         "single",
    }


def _run_agent_pipeline(
    features, prediction, explanation, chunks,
    retrieval_query, user_question, mode,
) -> dict:
    """Multi-agent or Chain-of-Thought pipeline."""
    try:
        from src.agents.orchestrator import MultiAgentPipeline

        context = {
            "features": features,
            "prediction": prediction,
            "explanation": explanation,
            "chunks": chunks,
            "user_question": user_question or "",
        }

        pipeline = MultiAgentPipeline()
        agent_result = pipeline.run(context, mode=mode)

        return {
            "prediction":       prediction,
            "explanation":      explanation,
            "llm_response": {
                "main_text":  agent_result.get("main_text", ""),
                "disclaimer": agent_result.get("disclaimer", ""),
                "high_risk":  agent_result.get("high_risk", False),
            },
            "sources_used":     list({c.get("source", "") for c in chunks}),
            "chunks":           chunks,
            "query_used":       retrieval_query,
            "mode":             mode,
            "reasoning_trace":  agent_result.get("reasoning_trace", ""),
            "agent_traces":     agent_result.get("agent_traces", []),
        }

    except Exception as exc:
        logger.error("Agent pipeline failed, falling back to single: %s", exc)
        return _run_single_pipeline(
            features, prediction, explanation, chunks,
            retrieval_query, user_question,
        )


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
