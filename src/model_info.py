"""
src/model_info.py — Facts about the trained models and the knowledge base, read
from saved artifacts (models/model_metadata.json and knowledge_base/vectorstore/).

This is the single source of truth for numbers shown in the UI
(GET /api/model-info) and in the README (scripts/update_readme_metrics.py),
so nothing displayed is typed by hand.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    MODEL_METADATA_PATH, CHUNK_METADATA_PATH, FAISS_INDEX_PATH,
    EMBEDDER_FLAG_PATH, KB_SOURCES_DIR,
)


def load_metadata() -> dict:
    """Metrics and dataset provenance written by `python -m src.ml.train`."""
    with open(MODEL_METADATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def knowledge_base_stats() -> dict:
    """What the FAISS index actually contains (not what the sources folder holds)."""
    import faiss

    with open(CHUNK_METADATA_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    indexed = sorted({c.get("source", "") for c in chunks.values()})
    available = sorted(p.stem for p in KB_SOURCES_DIR.glob("*.md"))
    embedder = (EMBEDDER_FLAG_PATH.read_text().strip()
                if EMBEDDER_FLAG_PATH.exists() else "tfidf")
    return {
        "n_chunks":            len(chunks),
        "indexed_sources":     indexed,
        "n_indexed_sources":   len(indexed),
        "available_sources":   available,
        "n_available_sources": len(available),
        "embedder":            embedder,
        "embedding_dim":       int(faiss.read_index(str(FAISS_INDEX_PATH)).d),
    }


def get_model_info() -> dict:
    return {"metadata": load_metadata(), "knowledge_base": knowledge_base_stats()}
