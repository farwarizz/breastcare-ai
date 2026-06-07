"""
src/ml/predict.py — Load LightGBM and return prediction dict.
Model is loaded once and cached with lru_cache for the session lifetime.
"""
from __future__ import annotations
import numpy as np
import joblib
from functools import lru_cache
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    LGBM_MODEL_PATH, FEATURE_NAMES,
    LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD, DECISION_THRESHOLD,
    MODEL_METADATA_PATH,
)


@lru_cache(maxsize=1)
def _load_model():
    if not LGBM_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {LGBM_MODEL_PATH}. "
            "Run `python -m src.ml.train` first."
        )
    return joblib.load(LGBM_MODEL_PATH)


@lru_cache(maxsize=1)
def _load_threshold() -> float:
    """Load Youden-calibrated threshold from metadata, fallback to 0.50."""
    try:
        import json
        with open(MODEL_METADATA_PATH) as f:
            meta = json.load(f)
        return float(meta.get("lgbm_optimal_threshold", DECISION_THRESHOLD))
    except Exception:
        return DECISION_THRESHOLD


def predict(features: dict) -> dict:
    """
    Run the LightGBM clinical tool on four mammographic features.

    Parameters
    ----------
    features : dict with keys: age (int), shape (int), margin (int), density (int)

    Returns
    -------
    dict:
        probability     float   — P(malignant), 0–1
        predicted_class str     — "Malignant" | "Benign"
        risk_category   str     — "Low" | "Moderate" | "Elevated"
        confidence_band str     — e.g. "72% ± 8%"
        feature_array   list    — [age, shape, margin, density]
    """
    model = _load_model()

    import pandas as pd
    X = pd.DataFrame([[
        features["age"],
        features["shape"],
        features["margin"],
        features["density"],
    ]], columns=FEATURE_NAMES)

    prob = float(model.predict_proba(X)[0, 1])
    threshold = _load_threshold()
    pred = "Malignant" if prob >= threshold else "Benign"

    if prob < LOW_RISK_THRESHOLD:
        risk_cat = "Low"
    elif prob < HIGH_RISK_THRESHOLD:
        risk_cat = "Moderate"
    else:
        risk_cat = "Elevated"

    # Simple ±8 pp confidence band (research model; not calibrated)
    pct = round(prob * 100, 1)
    band = f"{pct}% ± 8%"

    return {
        "probability":     prob,
        "predicted_class": pred,
        "risk_category":   risk_cat,
        "confidence_band": band,
        "feature_array":   [features["age"], features["shape"],
                             features["margin"], features["density"]],
    }
