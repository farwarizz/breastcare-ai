"""
src/analytics/batch_predict.py — Batch prediction on CSV uploads.

Runs the served model on every row: calibrated probability, conformal
prediction set, class at the development-chosen threshold, risk category and
the feature with the largest contribution.

Usage:
    from src.analytics.batch_predict import batch_predict
    results_df = batch_predict(input_df)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import FEATURE_NAMES, HIGH_RISK_THRESHOLD, SHAPE_MAP, MARGIN_MAP, DENSITY_MAP
from src.ml.artifacts import model_meta
from src.ml.explain import contributions
from src.ml.predict import predict_frame

logger = logging.getLogger(__name__)

_VALID = {"shape": set(SHAPE_MAP), "margin": set(MARGIN_MAP), "density": set(DENSITY_MAP)}



class BatchInputError(ValueError):
    """A problem with the uploaded rows; its message is safe to show to the user."""

def _validate(df: pd.DataFrame) -> None:
    missing = set(FEATURE_NAMES) - set(df.columns)
    if missing:
        raise BatchInputError(f"Missing required columns: {sorted(missing)}")
    if df[FEATURE_NAMES].isna().any().any():
        raise BatchInputError("Missing values in age/shape/margin/density are not allowed")
    for col, valid in _VALID.items():
        bad = sorted(set(df[col].astype(int)) - valid)
        if bad:
            raise BatchInputError(f"Column {col!r} has undocumented codes {bad}; valid: {sorted(valid)}")
    if not df["age"].between(18, 96).all():
        raise BatchInputError("Column 'age' must be between 18 and 96")


def batch_predict(df: pd.DataFrame) -> pd.DataFrame:
    """Original columns + probability, prediction set, class, risk category, dominant feature."""
    _validate(df)
    pred = predict_frame(df)
    _, contrib = contributions(df)
    result = df.copy()
    result["probability"] = np.round(pred["probability"].to_numpy(), 4)
    result["probability_pct"] = np.round(pred["probability"].to_numpy() * 100, 1)
    result["prediction_set"] = [" + ".join(s) if s else "(empty)" for s in pred["prediction_set"]]
    result["predicted_class"] = pred["predicted_class"].to_numpy()
    result["risk_category"] = pred["risk_category"].to_numpy()
    result["dominant_feature"] = [FEATURE_NAMES[i] for i in np.abs(contrib).argmax(axis=1)]
    result["model_used"] = model_meta("served")["label"]
    logger.info("Batch prediction complete: %d rows", len(result))
    return result


def batch_summary(result_df: pd.DataFrame) -> dict[str, Any]:
    """Summary statistics for a batch result."""
    return {
        "total_rows": len(result_df),
        "avg_probability": round(float(result_df["probability"].mean()), 4),
        "median_probability": round(float(result_df["probability"].median()), 4),
        "std_probability": round(float(result_df["probability"].std()), 4) if len(result_df) > 1 else 0.0,
        "risk_distribution": result_df["risk_category"].value_counts().to_dict(),
        "class_distribution": result_df["predicted_class"].value_counts().to_dict(),
        "prediction_set_distribution": result_df["prediction_set"].value_counts().to_dict(),
        "dominant_feature_counts": result_df["dominant_feature"].value_counts().to_dict(),
        "high_risk_count": int((result_df["probability"] >= HIGH_RISK_THRESHOLD).sum()),
        "high_risk_pct": round(float((result_df["probability"] >= HIGH_RISK_THRESHOLD).mean() * 100), 1),
    }
