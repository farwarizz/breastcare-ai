"""
src/analytics/batch_predict.py — Batch prediction on CSV uploads.

Runs the ML model across an entire DataFrame, computing predictions,
SHAP explanations, and risk categories for every row.

Usage:
    from src.analytics.batch_predict import batch_predict
    results_df = batch_predict(input_df)
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Callable
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    LGBM_MODEL_PATH, LGBM_SHAP_PATH,
    ENSEMBLE_MODEL_PATH, FEATURE_NAMES,
    LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD, DECISION_THRESHOLD,
)

logger = logging.getLogger(__name__)


def batch_predict(
    df: pd.DataFrame,
    use_ensemble: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """
    Run predictions on every row in a DataFrame.

    Parameters
    ----------
    df : DataFrame with columns: age, shape, margin, density
    use_ensemble : use stacking ensemble instead of LightGBM
    progress_callback : optional fn(current, total) for UI progress

    Returns
    -------
    DataFrame with original columns + prediction columns appended.
    """
    import joblib

    # Validate input
    required = {"age", "shape", "margin", "density"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[FEATURE_NAMES].values.astype(float)
    n_rows = len(X)

    if use_ensemble and ENSEMBLE_MODEL_PATH.exists():
        from src.ml.ensemble import StackingEnsemble
        model = StackingEnsemble.load()
        probs = model.predict_proba(X)[:, 1]
        model_name = "stacking_ensemble"
    else:
        if not LGBM_MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {LGBM_MODEL_PATH}")
        model = joblib.load(LGBM_MODEL_PATH)
        probs = model.predict_proba(X)[:, 1]
        model_name = "lightgbm"

    # Compute SHAP if explainer is available
    shap_values = None
    dominant_features = []
    try:
        shap_path = LGBM_SHAP_PATH if not use_ensemble else ENSEMBLE_MODEL_PATH
        if LGBM_SHAP_PATH.exists() and not use_ensemble:
            explainer = joblib.load(LGBM_SHAP_PATH)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # malignant class

            for i in range(n_rows):
                idx = np.argmax(np.abs(shap_values[i]))
                dominant_features.append(FEATURE_NAMES[idx])

                if progress_callback and (i + 1) % 50 == 0:
                    progress_callback(i + 1, n_rows)
    except Exception as e:
        logger.warning("SHAP batch computation failed: %s", e)
        dominant_features = ["N/A"] * n_rows

    if not dominant_features:
        dominant_features = ["N/A"] * n_rows

    # Build results
    result_df = df.copy()
    result_df["probability"] = np.round(probs, 4)
    result_df["probability_pct"] = np.round(probs * 100, 1)
    result_df["predicted_class"] = np.where(probs >= DECISION_THRESHOLD, "Malignant", "Benign")
    result_df["risk_category"] = pd.cut(
        probs,
        bins=[0, LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD, 1.0],
        labels=["Low", "Moderate", "Elevated"],
        include_lowest=True,
    )
    result_df["dominant_feature"] = dominant_features
    result_df["model_used"] = model_name

    logger.info("Batch prediction complete: %d rows, model=%s", n_rows, model_name)
    return result_df


def batch_summary(result_df: pd.DataFrame) -> dict[str, Any]:
    """Generate summary statistics from batch prediction results."""
    return {
        "total_rows": len(result_df),
        "avg_probability": round(result_df["probability"].mean(), 4),
        "median_probability": round(result_df["probability"].median(), 4),
        "std_probability": round(result_df["probability"].std(), 4),
        "risk_distribution": result_df["risk_category"].value_counts().to_dict(),
        "class_distribution": result_df["predicted_class"].value_counts().to_dict(),
        "dominant_feature_counts": result_df["dominant_feature"].value_counts().to_dict(),
        "high_risk_count": int((result_df["probability"] >= HIGH_RISK_THRESHOLD).sum()),
        "high_risk_pct": round(
            (result_df["probability"] >= HIGH_RISK_THRESHOLD).mean() * 100, 1
        ),
    }
