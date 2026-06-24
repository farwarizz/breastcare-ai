"""
src/analytics/model_comparison.py — A/B model comparison across all available models.

Runs the same input through LightGBM, XGBoost, Ensemble, and (optionally)
the image model, then produces a comparison table and agreement metrics.

Usage:
    from src.analytics.model_comparison import compare_models
    result = compare_models(features)
"""
from __future__ import annotations

import logging
import numpy as np
import joblib
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    LGBM_MODEL_PATH, XGBOOST_MODEL_PATH, ENSEMBLE_MODEL_PATH,
    FEATURE_NAMES, DECISION_THRESHOLD,
    LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD,
)

logger = logging.getLogger(__name__)


def compare_models(features: dict) -> dict[str, Any]:
    """
    Run all available models on the same input and compare results.

    Returns a comparison dict with per-model predictions and agreement metrics.
    """
    X = np.array([[
        features["age"], features["shape"],
        features["margin"], features["density"],
    ]], dtype=float)

    results = {}

    # LightGBM (mammographic)
    if LGBM_MODEL_PATH.exists():
        try:
            model = joblib.load(LGBM_MODEL_PATH)
            prob = float(model.predict_proba(X)[:, 1][0])
            results["lightgbm"] = _format_result("LightGBM (Mammographic)", prob)
        except Exception as e:
            logger.warning("LightGBM comparison failed: %s", e)

    # Stacking Ensemble
    if ENSEMBLE_MODEL_PATH.exists():
        try:
            from src.ml.ensemble import StackingEnsemble
            ensemble = StackingEnsemble.load()
            prob = float(ensemble.predict_proba(X)[:, 1][0])
            results["ensemble"] = _format_result("Stacking Ensemble", prob)

            # Also get individual base model predictions
            base_preds = ensemble.get_base_predictions(X)
            for name, base_prob in base_preds.items():
                results[f"ensemble_{name}"] = _format_result(
                    f"Ensemble/{name.title()}", base_prob
                )
        except Exception as e:
            logger.warning("Ensemble comparison failed: %s", e)

    if not results:
        return {"available": False, "message": "No models available for comparison."}

    # Agreement metrics
    probs = [r["probability"] for r in results.values()]
    agreement = _compute_agreement(probs)

    return {
        "available": True,
        "models": results,
        "agreement": agreement,
        "features": features,
    }


def _format_result(model_name: str, probability: float) -> dict[str, Any]:
    """Format a single model's result."""
    return {
        "model_name": model_name,
        "probability": round(probability, 4),
        "probability_pct": round(probability * 100, 1),
        "predicted_class": "Malignant" if probability >= DECISION_THRESHOLD else "Benign",
        "risk_category": _risk_category(probability),
    }


def _risk_category(prob: float) -> str:
    if prob < LOW_RISK_THRESHOLD:
        return "Low"
    elif prob < HIGH_RISK_THRESHOLD:
        return "Moderate"
    return "Elevated"


def _compute_agreement(probabilities: list[float]) -> dict[str, Any]:
    """Compute agreement metrics across model predictions."""
    if len(probabilities) < 2:
        return {"sufficient_models": False}

    probs = np.array(probabilities)
    classes = (probs >= DECISION_THRESHOLD).astype(int)

    return {
        "sufficient_models": True,
        "n_models": len(probs),
        "mean_probability": round(float(probs.mean()), 4),
        "std_probability": round(float(probs.std()), 4),
        "min_probability": round(float(probs.min()), 4),
        "max_probability": round(float(probs.max()), 4),
        "spread": round(float(probs.max() - probs.min()), 4),
        "class_agreement": bool(classes.min() == classes.max()),
        "unanimous_class": "Malignant" if classes.all() else ("Benign" if not classes.any() else "Split"),
        "agreement_pct": round(
            float(max(classes.mean(), 1 - classes.mean()) * 100), 1
        ),
    }
