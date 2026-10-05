"""
src/ml/artifacts.py — Load the trained model bundles and their metadata (cached).

A bundle (joblib) holds the fitted estimator plus what is needed to explain it.
Calibration, the decision threshold and the conformal cut-offs are plain
numbers kept in models/model_metadata.json.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
import sys

import joblib

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import SERVED_MODEL_PATH, CHALLENGER_MODEL_PATH, MODEL_METADATA_PATH

_PATHS = {"served": SERVED_MODEL_PATH, "challenger": CHALLENGER_MODEL_PATH}


@lru_cache(maxsize=2)
def load_bundle(which: str = "served") -> dict:
    path = _PATHS[which]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.ml.train` first.")
    return joblib.load(path)


@lru_cache(maxsize=1)
def load_metadata() -> dict:
    if not MODEL_METADATA_PATH.exists():
        raise FileNotFoundError(f"{MODEL_METADATA_PATH} not found. Run `python -m src.ml.train` first.")
    with open(MODEL_METADATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def model_meta(which: str = "served") -> dict:
    return load_metadata()[f"{which}_model"]


def clear_cache() -> None:
    load_bundle.cache_clear()
    load_metadata.cache_clear()
