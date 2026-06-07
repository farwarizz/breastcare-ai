"""
src/rag/build_vectorstore.py — Embed chunks and build FAISS index.

Primary:  sentence-transformers all-MiniLM-L6-v2 (384-dim dense embeddings)
Fallback: TF-IDF + TruncatedSVD (128-dim) — used when HuggingFace unreachable.
A flag file (embedder_type.txt) records which was used so retriever.py loads correctly.
"""
from __future__ import annotations
import json
import pickle
import numpy as np
import faiss
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    CHUNKS_JSONL, FAISS_INDEX_PATH, CHUNK_METADATA_PATH,
    TFIDF_PATH, SVD_PATH, EMBEDDER_FLAG_PATH,
    EMBED_DIM_DENSE, EMBED_DIM_FALLBACK,
    KB_VECTORSTORE_DIR,
)


def _load_chunks() -> list[dict]:
    chunks = []
    with open(CHUNKS_JSONL, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def _embed_tfidf(texts: list[str]) -> np.ndarray:
    """TF-IDF + SVD (128-dim) fallback embedder."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    svd = TruncatedSVD(n_components=EMBED_DIM_FALLBACK, random_state=42)
    embeddings = svd.fit_transform(tfidf_matrix).astype("float32")

    # Save for query-time reuse
    with open(TFIDF_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(SVD_PATH, "wb") as f:
        pickle.dump(svd, f)

    return embeddings


def _embed_dense(texts: list[str]) -> np.ndarray:
    """sentence-transformers all-MiniLM-L6-v2 (384-dim)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True,
                               batch_size=32, convert_to_numpy=True)
    return embeddings.astype("float32")


def build_vectorstore(chunks: list[dict]) -> None:
    KB_VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    texts = [c["text"] for c in chunks]

    # Try dense embeddings first; fall back to TF-IDF if unavailable
    embedder_type = "dense"
    try:
        embeddings = _embed_dense(texts)
        print(f"  Embedded {len(texts)} chunks with all-MiniLM-L6-v2 "
              f"({embeddings.shape[1]}-dim)")
    except Exception as e:
        print(f"  Dense embedding unavailable ({e}). Using TF-IDF+SVD fallback.")
        embeddings = _embed_tfidf(texts)
        embedder_type = "tfidf"
        print(f"  Embedded {len(texts)} chunks with TF-IDF+SVD "
              f"({embeddings.shape[1]}-dim)")

    # L2-normalise so IndexFlatIP = cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"  FAISS index saved: {FAISS_INDEX_PATH} ({index.ntotal} vectors, dim={dim})")

    # Save metadata mapping vector_id → chunk
    metadata = {str(i): chunks[i] for i in range(len(chunks))}
    with open(CHUNK_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"  Metadata saved: {CHUNK_METADATA_PATH}")

    # Record which embedder was used
    with open(EMBEDDER_FLAG_PATH, "w") as f:
        f.write(embedder_type)
    print(f"  Embedder type: {embedder_type}")
