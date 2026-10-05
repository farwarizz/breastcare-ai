"""
src/ml/explain.py — Per-feature contributions (SHAP values) for a trained model.

- Logistic regression: exact SHAP values in log-odds space. For a linear model
  with features treated as independent (interventional SHAP), the contribution
  of an input column is coef · (x − mean over the training data); one-hot
  columns are summed back into their original feature. shap.LinearExplainer
  computes the same quantity.
- LightGBM: TreeSHAP (shap.TreeExplainer), also in log-odds space.

In both cases base_value + sum(contributions) equals the model's log-odds
before calibration. Calibration is a monotone map, so the signs and ranking of
contributions carry over to the calibrated probability.

The text summary (shap_text) also drives the RAG retrieval query, so retrieved
documents relate to what moved the estimate.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import FEATURE_NAMES, SHAPE_MAP, MARGIN_MAP, DENSITY_MAP
from src.ml.artifacts import load_bundle


def model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The four model inputs with the dtypes used in training."""
    X = df[FEATURE_NAMES].copy()
    X["age"] = X["age"].astype(float)
    for col in ("shape", "margin", "density"):
        X[col] = X[col].astype(int)
    return X


def linear_background(pipeline, X_train: pd.DataFrame) -> dict:
    """Stored in the bundle at training time: column means and column→feature map."""
    pre = pipeline.named_steps["pre"]
    Z = pre.transform(model_frame(X_train))
    names = pre.get_feature_names_out()          # e.g. num__age, cat__shape_3
    groups = [next(i for i, f in enumerate(FEATURE_NAMES) if n.split("__", 1)[1].startswith(f))
              for n in names]
    return {"background_mean": Z.mean(axis=0), "column_groups": groups}


@lru_cache(maxsize=2)
def _tree_explainer(which: str):
    import shap
    return shap.TreeExplainer(load_bundle(which)["estimator"])


def contributions(df: pd.DataFrame, which: str = "served") -> tuple[float, np.ndarray]:
    """(base value, contributions [n_rows × 4]) in log-odds space."""
    bundle = load_bundle(which)
    X = model_frame(df)
    if bundle["family"] == "logistic_regression":
        pipe = bundle["estimator"]
        Z = pipe.named_steps["pre"].transform(X)
        lr = pipe.named_steps["lr"]
        coef, mu = lr.coef_[0], bundle["background_mean"]
        cols = coef * (Z - mu)
        out = np.zeros((len(X), len(FEATURE_NAMES)))
        for j, g in enumerate(bundle["column_groups"]):
            out[:, g] += cols[:, j]
        base = float(lr.intercept_[0] + coef @ mu)
        return base, out
    explainer = _tree_explainer(which)
    sv = explainer.shap_values(X)
    sv = sv[1] if isinstance(sv, list) else sv
    ev = explainer.expected_value
    base = float(ev[1] if np.ndim(ev) else ev)
    return base, np.asarray(sv, dtype=float)


def method_name(which: str = "served") -> str:
    return ("Exact linear SHAP (interventional), log-odds"
            if load_bundle(which)["family"] == "logistic_regression"
            else "TreeSHAP, log-odds")


def _feature_label(name: str, value) -> str:
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
    Contributions for a single case.

    Returns base_value, feature_attributions {feature: value}, dominant_feature,
    risk_direction ("elevated" | "reduced"), shap_text (drives RAG retrieval),
    shap_values, feature_values, feature_labels and method.
    """
    base, contrib = contributions(pd.DataFrame([features]))
    sv = contrib[0]
    attributions = {name: float(sv[i]) for i, name in enumerate(FEATURE_NAMES)}

    dominant = max(attributions, key=lambda k: abs(attributions[k]))
    risk_dir = "elevated" if attributions[dominant] > 0 else "reduced"

    ordered = sorted(attributions.items(), key=lambda x: abs(x[1]), reverse=True)
    parts = [f"{feat} ({_feature_label(feat, features[feat])}) "
             f"{'elevated' if val > 0 else 'reduced'} risk" for feat, val in ordered]
    shap_text = (
        f"The strongest predictor was {dominant} ({_feature_label(dominant, features[dominant])}) "
        f"{'increasing' if risk_dir == 'elevated' else 'decreasing'} risk. "
        + "; ".join(parts) + "."
    )

    return {
        "base_value":           base,
        "feature_attributions": attributions,
        "dominant_feature":     dominant,
        "risk_direction":       risk_dir,
        "shap_text":            shap_text,
        "shap_values":          [float(v) for v in sv],
        "feature_values":       [features[n] for n in FEATURE_NAMES],
        "feature_labels":       {n: _feature_label(n, features[n]) for n in FEATURE_NAMES},
        "method":               method_name(),
    }
