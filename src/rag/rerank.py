"""
src/rag/rerank.py — Optional cross-encoder re-ranking (ONNX via fastembed, no PyTorch).

A cross-encoder reads the query and each candidate together, which is more
accurate than comparing independent embeddings but slower, so it only
re-orders a small candidate pool from the first-stage retriever.
"""
from __future__ import annotations

import importlib.util
import os
from functools import lru_cache

RERANKER = "Xenova/ms-marco-MiniLM-L-6-v2"
POOL = 20          # first-stage candidates passed to the cross-encoder


def available() -> bool:
    return importlib.util.find_spec("fastembed") is not None


@lru_cache(maxsize=1)
def _model():
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    return TextCrossEncoder(model_name=RERANKER, cache_dir=os.environ.get("FASTEMBED_CACHE_PATH"))


def rerank(query: str, chunks: list[dict], k: int) -> list[dict]:
    if not chunks:
        return []
    passages = [f"{c['section']}: {c['text']}" for c in chunks]
    scores = list(_model().rerank(query, passages))
    order = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))[:k]
    return [{**chunks[i], "score": round(float(scores[i]), 6), "rank": r + 1}
            for r, i in enumerate(order)]
