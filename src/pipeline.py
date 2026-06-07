"""
src/pipeline.py — Single orchestrator entry point.
The UI calls run_pipeline() only; it never touches ML/SHAP/RAG directly.

8-step execution:
    1. ML prediction
    2. SHAP explanation
    3. Build retrieval query from SHAP output
    4. Retrieve relevant chunks
    5. Inject safety disclaimer for high-risk predictions
    6. Assemble LLM prompt
    7. Call Claude API
    8. Safety filter
"""
from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import HIGH_RISK_THRESHOLD, CHUNK_TOP_K
from src.ml.predict  import predict
from src.ml.explain  import explain
from src.rag.retriever      import rag_query, query_by_source
from src.rag.prompt_builder import build_retrieval_query, build_prompt
from src.llm.client  import call_llm
from src.safety      import filter_response


def run_pipeline(features: dict, user_question: str | None = None) -> dict:
    """
    Run the complete BreastCare AI pipeline.

    Parameters
    ----------
    features : dict — {age: int, shape: int, margin: int, density: int}
    user_question : str or None — optional follow-up from chat panel

    Returns
    -------
    dict:
        prediction    dict  — from predict.py
        explanation   dict  — from explain.py
        llm_response  dict  — from safety.py (main_text, disclaimer, high_risk, ...)
        sources_used  list  — source names from retrieved chunks
        chunks        list  — full retrieved chunk dicts
        query_used    str   — retrieval query
    """
    # Step 1 — ML prediction
    prediction = predict(features)

    # Step 2 — SHAP explanation
    explanation = explain(features)

    # Step 3 — Build retrieval query from SHAP output
    retrieval_query = build_retrieval_query(explanation, user_question)

    # Step 4 — Retrieve relevant chunks
    chunks = rag_query(retrieval_query, k=CHUNK_TOP_K)

    # Step 5 — Inject safety disclaimer chunk for high-risk predictions
    if prediction["probability"] >= HIGH_RISK_THRESHOLD:
        safety_chunks = query_by_source("safety_disclaimer", retrieval_query, k=1)
        existing_ids = {c.get("chunk_id") for c in chunks}
        for sc in safety_chunks:
            if sc.get("chunk_id") not in existing_ids:
                chunks.append(sc)

    # Step 6 — Assemble LLM prompt
    messages = build_prompt(prediction, explanation, chunks, features, user_question)

    # Step 7 — Call Claude API
    raw_response = call_llm(messages)

    # Step 8 — Safety filter
    safe_response = filter_response(raw_response, prediction["probability"])

    return {
        "prediction":   prediction,
        "explanation":  explanation,
        "llm_response": safe_response,
        "sources_used": list({c.get("source", "") for c in chunks}),
        "chunks":       chunks,
        "query_used":   retrieval_query,
    }


def run_pipeline_ml_only(features: dict) -> dict:
    """
    Run only ML + SHAP — no LLM call.
    Used when API key is not set or in fast preview mode.
    """
    prediction  = predict(features)
    explanation = explain(features)
    query       = build_retrieval_query(explanation)
    chunks      = rag_query(query, k=CHUNK_TOP_K)

    return {
        "prediction":   prediction,
        "explanation":  explanation,
        "llm_response": None,
        "sources_used": [c.get("source", "") for c in chunks],
        "chunks":       chunks,
        "query_used":   query,
    }
