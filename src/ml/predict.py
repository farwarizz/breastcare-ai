"""
src/ml/predict.py — Risk estimate plus a conformal prediction set.

For each case:
- probability       P(malignant): model score → calibration map chosen on
                    out-of-fold predictions (see src/ml/calibration.py). The rule may
                    choose "none" (the served model's raw probability); `calibrated`
                    is true only when a calibration map was actually applied.
- prediction_set    Mondrian conformal set at the stored coverage target
                    (90% per true class; see src/ml/conformal.py)
- predicted_class   at a threshold chosen on out-of-fold development data
- risk_category     Low < 0.35 ≤ Moderate < 0.65 ≤ Elevated (served probability)
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import FEATURE_NAMES, LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD
from src.ml.artifacts import load_bundle, model_meta
from src.ml.calibration import Calibrator
from src.ml.conformal import MondrianConformal
from src.ml.explain import model_frame


def risk_category(p: float) -> str:
    if p < LOW_RISK_THRESHOLD:
        return "Low"
    if p < HIGH_RISK_THRESHOLD:
        return "Moderate"
    return "Elevated"


def predict_frame(df: pd.DataFrame, which: str = "served") -> pd.DataFrame:
    """Vectorised prediction for any number of rows (used by batch and comparison)."""
    bundle, meta = load_bundle(which), model_meta(which)
    raw = bundle["estimator"].predict_proba(model_frame(df))[:, 1]
    prob = Calibrator.from_dict(meta["calibration"]).transform(raw)
    threshold = meta["threshold"]["value"]
    out = pd.DataFrame(index=df.index)
    out["raw_probability"] = raw
    out["probability"] = prob
    out["predicted_class"] = np.where(prob >= threshold, "Malignant", "Benign")
    out["risk_category"] = [risk_category(p) for p in prob]
    if "conformal" in meta:
        out["prediction_set"] = MondrianConformal.from_dict(meta["conformal"]).predict_sets(prob)
    return out


def predict(features: dict) -> dict:
    """Single-case prediction for the served model."""
    meta = model_meta("served")
    row = predict_frame(pd.DataFrame([features])).iloc[0]
    return {
        "probability":           float(row["probability"]),
        "raw_model_probability": float(row["raw_probability"]),
        "calibrated":            meta["calibration"]["method"] != "none",
        "calibration_method":    meta["calibration"]["method"],
        "prediction_set":        list(row["prediction_set"]),
        "set_coverage":          meta["conformal"]["coverage_target"],
        "predicted_class":       row["predicted_class"],
        "decision_threshold":    meta["threshold"]["value"],
        "risk_category":         row["risk_category"],
        "model":                 meta["label"],
        "feature_array":         [features[n] for n in FEATURE_NAMES],
    }
