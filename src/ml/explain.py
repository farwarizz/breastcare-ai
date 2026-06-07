"""
src/ml/explain.py — SHAP TreeExplainer wrapper for LightGBM.

Key architectural point: shap_text is used as the RAG retrieval query
so retrieved documents are always aligned with what drove the prediction,
not just with the user's question.
"""
from __future__ import annotations
import numpy as np
import joblib
from functools import lru_cache
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    LGBM_SHAP_PATH, FEATURE_NAMES,
    SHAPE_MAP, MARGIN_MAP, DENSITY_MAP,
)


@lru_cache(maxsize=1)
def _load_explainer():
    if not LGBM_SHAP_PATH.exists():
        raise FileNotFoundError(
            f"SHAP explainer not found at {LGBM_SHAP_PATH}. "
            "Run `python -m src.ml.train` first."
        )
    return joblib.load(LGBM_SHAP_PATH)


def _feature_label(name: str, value) -> str:
    """Return human-readable label for a feature value."""
    val = int(value)
    if name == "age":
        return f"{val} years"
    if name == "shape":
        return SHAPE_MAP.get(val, str(val))
    if name == "margin":
        return MARGIN_MAP.get(val, str(val))
    if name == "density":
        return DENSITY_MAP.get(val, str(val))
    return str(val)


def explain(features: dict) -> dict:
    """
    Compute SHAP feature attributions for a single prediction.

    Parameters
    ----------
    features : dict — same as predict() input

    Returns
    -------
    dict:
        base_value           float  — E[f(x)]
        feature_attributions dict   — {feature_name: shap_value}
        dominant_feature     str    — feature with largest |shap|
        risk_direction       str    — "elevated" | "reduced"
        shap_text            str    — natural language for RAG query
        shap_values          list   — raw SHAP values in feature order
        feature_values       list   — raw input values in feature order
        feature_labels       dict   — human-readable labels per feature
    """
    explainer = _load_explainer()

    import pandas as pd
    import warnings
    X = pd.DataFrame([[
        features["age"],
        features["shape"],
        features["margin"],
        features["density"],
    ]], columns=FEATURE_NAMES)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = explainer.shap_values(X)
    if isinstance(raw, list):
        sv = raw[1][0]   # malignant class attributions
    else:
        sv = raw[0]

    base_value = float(explainer.expected_value
                       if not hasattr(explainer.expected_value, '__len__')
                       else explainer.expected_value[1])

    attributions = {name: float(sv[i]) for i, name in enumerate(FEATURE_NAMES)}

    # Dominant feature = largest absolute SHAP value
    dominant = max(attributions, key=lambda k: abs(attributions[k]))
    risk_dir = "elevated" if attributions[dominant] > 0 else "reduced"

    # Build natural-language shap_text for RAG query
    # "margin (Spiculated) elevated risk; shape (Irregular) elevated risk; ..."
    ordered = sorted(attributions.items(), key=lambda x: abs(x[1]), reverse=True)
    parts = []
    for feat, val in ordered:
        label = _feature_label(feat, features[feat])
        direction = "elevated" if val > 0 else "reduced"
        parts.append(f"{feat} ({label}) {direction} risk")

    shap_text = (
        f"The strongest predictor was {dominant} ({_feature_label(dominant, features[dominant])}) "
        f"{'increasing' if risk_dir == 'elevated' else 'decreasing'} risk. "
        + "; ".join(parts) + "."
    )

    feature_labels = {
        name: _feature_label(name, features[name]) for name in FEATURE_NAMES
    }

    return {
        "base_value":           base_value,
        "feature_attributions": attributions,
        "dominant_feature":     dominant,
        "risk_direction":       risk_dir,
        "shap_text":            shap_text,
        "shap_values":          list(sv),
        "feature_values":       [features[n] for n in FEATURE_NAMES],
        "feature_labels":       feature_labels,
    }
