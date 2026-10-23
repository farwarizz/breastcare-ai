"""
src/rag/embedders.py — Sentence embedders that do not need PyTorch.

- minilm: sentence-transformers/all-MiniLM-L6-v2 through fastembed (ONNX
  Runtime). Same model the project used before, without the torch dependency.
- potion: minishlab/potion-base-8M through model2vec, a static embedding model
  distilled from a sentence transformer. Plain numpy at query time and about
  30 MB, which suits small serverless deployments.

Models download from the Hugging Face Hub on first use and are cached
(FASTEMBED_CACHE_PATH / HF_HOME control where). Vectors are L2-normalised, so a
dot product is cosine similarity.
"""
from __future__ import annotations

import importlib.util
import os
from functools import lru_cache

import numpy as np

EMBEDDERS = {
    "minilm": {"library": "fastembed", "model": "sentence-transformers/all-MiniLM-L6-v2"},
    "potion": {"library": "model2vec", "model": "minishlab/potion-base-8M"},
}


def available(name: str) -> bool:
    return importlib.util.find_spec(EMBEDDERS[name]["library"]) is not None


def _normalise(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


@lru_cache(maxsize=None)
def _model(name: str):
    spec = EMBEDDERS[name]
    if spec["library"] == "fastembed":
        from fastembed import TextEmbedding
        return TextEmbedding(model_name=spec["model"],
                             cache_dir=os.environ.get("FASTEMBED_CACHE_PATH"))
    from model2vec import StaticModel
    return StaticModel.from_pretrained(spec["model"])


def encode(name: str, texts: list[str]) -> np.ndarray:
    """Normalised float32 embeddings, one row per text."""
    model = _model(name)
    if EMBEDDERS[name]["library"] == "fastembed":
        return _normalise(np.stack(list(model.embed(texts))))
    return _normalise(model.encode(texts))
