"""
src/model_info.py — Facts about the trained models and the knowledge base, read
from saved artifacts (models/model_metadata.json, knowledge_base/index/).

This is the single source of truth for numbers shown in the UI
(GET /api/model-info) and in the docs (scripts/update_readme_metrics.py), so
nothing displayed is typed by hand.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (MODEL_METADATA_PATH, KB_SOURCES_DIR, CHUNKS_JSONL, KB_MANIFEST_PATH,
                        RETRIEVAL_MODE, RETRIEVAL_QUERY, EVAL_REPORTS_DIR, GEMINI_MODEL, JUDGE_MODEL)

EVALUATION_REPORTS = {   # Phase 2 evaluations (src/evals/*); each is optional until it has been run
    "retrieval": EVAL_REPORTS_DIR / "rag_evaluation.json",
    "answers":   EVAL_REPORTS_DIR / "answer_evaluation.json",
    "safety":    EVAL_REPORTS_DIR / "safety_evaluation.json",
}


def load_metadata() -> dict:
    """Metrics and dataset provenance written by `python -m src.ml.train`."""
    with open(MODEL_METADATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def knowledge_base_stats() -> dict:
    """What the index actually contains (not just what the sources folder holds)."""
    with open(CHUNKS_JSONL, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    manifest = json.loads(KB_MANIFEST_PATH.read_text(encoding="utf-8"))
    indexed = sorted({c["source"] for c in chunks})
    available = sorted(p.stem for p in KB_SOURCES_DIR.glob("*.md"))
    mode = RETRIEVAL_MODE
    base = mode.removesuffix("+rerank")
    embedder = base.split(":", 1)[1] if ":" in base else None
    return {
        "n_chunks":            len(chunks),
        "indexed_sources":     indexed,
        "n_indexed_sources":   len(indexed),
        "available_sources":   available,
        "n_available_sources": len(available),
        "retrieval_mode":      mode,
        "retrieval_query":     RETRIEVAL_QUERY,
        "rerank":              mode.endswith("+rerank"),
        "chunks_sha256":       manifest.get("chunks_sha256"),
        "embedder":          manifest["embedders"].get(embedder, {}).get("model") if embedder else "none (bm25)",
        "embedding_dim":       manifest["embedders"].get(embedder, {}).get("dim") if embedder else None,
    }


def evaluation_summaries() -> dict:
    """Headline numbers of the RAG, answer-quality and red-team evaluations (no per-item rows).
    A report that has not been produced yet is simply absent."""
    out = {}
    for name, path in EVALUATION_REPORTS.items():
        if not path.exists():
            continue
        rep = json.loads(path.read_text(encoding="utf-8"))
        rep.pop("rows", None)
        out[name] = rep
    return out


def get_model_info() -> dict:
    return {"metadata": load_metadata(), "knowledge_base": knowledge_base_stats(),
            "llm": {"model": GEMINI_MODEL, "judge_model": JUDGE_MODEL},
            "evaluations": evaluation_summaries()}
