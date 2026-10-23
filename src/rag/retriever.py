"""
src/rag/retriever.py — Search the knowledge base.

Modes
  bm25            keyword search (src/rag/bm25.py)
  dense:<name>    cosine similarity with a torch-free embedder (src/rag/embedders.py)
  hybrid:<name>   reciprocal rank fusion, score = Σ 1 / (60 + rank), of bm25 and dense:<name>
  <mode>+rerank   re-order the top 20 of <mode> with a cross-encoder (src/rag/rerank.py)

The default mode is config.RETRIEVAL_MODE, chosen by the retrieval evaluation
(src/evals/retrieval.py). If a mode's embedder is not installed or its vectors
do not match the current chunks, search falls back to bm25 and records the mode
it actually used. It never compares vectors from different models.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import CHUNK_TOP_K, CHUNKS_JSONL, KB_INDEX_DIR, KB_MANIFEST_PATH, RETRIEVAL_MODE
from src.rag.bm25 import BM25
from src.rag import rerank as reranker
from src.rag.embedders import EMBEDDERS, available, encode

logger = logging.getLogger(__name__)
RRF_K = 60


@lru_cache(maxsize=1)
def _index():
    if not CHUNKS_JSONL.exists() or not KB_MANIFEST_PATH.exists():
        raise FileNotFoundError("Knowledge-base index missing. Run `python scripts/build_knowledge_base.py`.")
    with open(CHUNKS_JSONL, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    manifest = json.loads(KB_MANIFEST_PATH.read_text(encoding="utf-8"))
    from src.rag.index import chunks_hash
    if chunks_hash(chunks) != manifest.get("chunks_sha256"):
        raise ValueError("chunks.jsonl does not match the index manifest; rebuild the knowledge base.")
    return chunks, manifest, BM25([c["index_text"] for c in chunks])


@lru_cache(maxsize=None)
def _vectors(name: str) -> np.ndarray | None:
    """The stored vectors for an embedder, or None if they cannot be used safely."""
    chunks, manifest, _ = _index()
    info = manifest["embedders"].get(name)
    if info is None or not available(name):
        return None
    path = KB_INDEX_DIR / info["file"]
    if not path.exists():
        return None
    vectors = np.load(path)
    return vectors if vectors.shape == (len(chunks), info["dim"]) else None


def resolve_mode(mode: str | None = None) -> str:
    """The mode that will actually run (falls back to bm25 if dense vectors are unusable)."""
    mode = mode or RETRIEVAL_MODE
    if mode.endswith("+rerank"):
        base = resolve_mode(mode[: -len("+rerank")])
        if not reranker.available():
            logger.warning("Re-ranker unavailable (fastembed missing); using %s", base)
            return base
        return f"{base}+rerank"
    if mode == "bm25":
        return mode
    kind, _, name = mode.partition(":")
    if kind not in ("dense", "hybrid") or name not in EMBEDDERS:
        raise ValueError(f"unknown retrieval mode {mode!r}")
    if _vectors(name) is None:
        logger.warning("Retrieval mode %s unavailable (embedder or vectors missing); using bm25", mode)
        return "bm25"
    return mode


def _order(scores: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Candidate indices sorted by score (descending), ties broken by position."""
    return candidates[np.lexsort((candidates, -scores[candidates]))]


def search(query: str, k: int = CHUNK_TOP_K, mode: str | None = None,
           source: str | None = None) -> list[dict]:
    chunks, _, bm25 = _index()
    used = resolve_mode(mode)
    if used.endswith("+rerank"):
        pool = search(query, reranker.POOL, used[: -len("+rerank")], source)
        return [{**c, "retrieval_mode": used} for c in reranker.rerank(query, pool, k)]
    candidates = np.array([i for i, c in enumerate(chunks) if source is None or c["source"] == source])
    if len(candidates) == 0:
        return []

    def dense_scores(name: str) -> np.ndarray:
        return _vectors(name) @ encode(name, [query])[0]

    if used == "bm25":
        scores = bm25.scores(query)
    elif used.startswith("dense:"):
        scores = dense_scores(used.split(":", 1)[1])
    else:
        # Reciprocal-rank fusion. BM25 ranks only chunks that share a term with the query:
        # a chunk with BM25 score 0 was not retrieved by it and gets no credit from that
        # list (otherwise ties would be broken by file position and leak into the fusion).
        scores = np.zeros(len(chunks))
        lexical = bm25.scores(query)
        for ranked in (_order(lexical, candidates[lexical[candidates] > 0]),
                       _order(dense_scores(used.split(":", 1)[1]), candidates)):
            for rank, idx in enumerate(ranked, start=1):
                scores[idx] += 1.0 / (RRF_K + rank)

    out = []
    for rank, idx in enumerate(_order(scores, candidates)[:k], start=1):
        chunk = {key: v for key, v in chunks[idx].items() if key != "index_text"}
        out.append({**chunk, "score": round(float(scores[idx]), 6), "rank": rank,
                    "retrieval_mode": used})
    return out


def rag_query(query: str, k: int = CHUNK_TOP_K) -> list[dict]:
    return search(query, k)


def query_by_source(source_name: str, query: str, k: int = 1) -> list[dict]:
    """Best-matching chunk(s) from one document (used for the safety notice on high-risk results)."""
    return search(query, k, source=source_name)


def stats() -> dict:
    chunks, manifest, _ = _index()
    return {
        "n_chunks": len(chunks),
        "n_sources": len({c["source"] for c in chunks}),
        "embedders": {n: {"model": i["model"], "dim": i["dim"]} for n, i in manifest["embedders"].items()},
        "configured_mode": RETRIEVAL_MODE,
        "active_mode": resolve_mode(),
    }
