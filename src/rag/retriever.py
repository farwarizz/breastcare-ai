"""
src/rag/retriever.py — Query the FAISS vectorstore and return top-k chunks.
Auto-detects TF-IDF vs dense embedder from the flag file.
"""
from __future__ import annotations
import json
import pickle
import numpy as np
import faiss
from functools import lru_cache
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    FAISS_INDEX_PATH, CHUNK_METADATA_PATH, TFIDF_PATH, SVD_PATH,
    EMBEDDER_FLAG_PATH, CHUNK_TOP_K,
)


@lru_cache(maxsize=1)
def _load_index_and_meta():
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_INDEX_PATH}. "
            "Run `python scripts/build_knowledge_base.py` first."
        )
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(CHUNK_METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)
    embedder_type = "tfidf"
    if EMBEDDER_FLAG_PATH.exists():
        embedder_type = EMBEDDER_FLAG_PATH.read_text().strip()
    return index, metadata, embedder_type


@lru_cache(maxsize=1)
def _load_tfidf_models():
    with open(TFIDF_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(SVD_PATH, "rb") as f:
        svd = pickle.load(f)
    return vectorizer, svd


def _encode_query(query: str, embedder_type: str) -> np.ndarray:
    if embedder_type == "dense":
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            vec = model.encode([query], convert_to_numpy=True).astype("float32")
            faiss.normalize_L2(vec)
            return vec
        except Exception:
            pass
    # Fallback: TF-IDF + SVD
    vectorizer, svd = _load_tfidf_models()
    tfidf_vec = vectorizer.transform([query])
    vec = svd.transform(tfidf_vec).astype("float32")
    faiss.normalize_L2(vec)
    return vec


def rag_query(query: str, k: int = CHUNK_TOP_K) -> list[dict]:
    """
    Search FAISS for top-k most similar chunks.

    Returns list of dicts with keys:
        chunk_id, source, section, text, score
    """
    index, metadata, embedder_type = _load_index_and_meta()
    query_vec = _encode_query(query, embedder_type)

    distances, indices = index.search(query_vec, k)
    results = []
    seen = set()
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        chunk = metadata.get(str(idx), {})
        chunk_id = chunk.get("chunk_id", str(idx))
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        results.append({
            **chunk,
            "score": float(dist),
        })
    return results


def query_by_source(source_name: str, query: str, k: int = 1) -> list[dict]:
    """
    Retrieve chunks filtered to a specific source document.
    Used for safety disclaimer injection on high-risk predictions.
    """
    _, metadata, _ = _load_index_and_meta()
    # Filter chunks by source, then return up to k
    source_chunks = [
        {**c, "score": 1.0}
        for c in metadata.values()
        if c.get("source") == source_name
    ]
    return source_chunks[:k]
