"""
src/analytics/model_comparison.py — Served model vs challenger on the same input.

Both are trained on the same fit split and evaluated on the same held-out test
set (see models/model_metadata.json for the paired comparison with CIs). This
module shows their predictions for one case side by side.

Usage:
    from src.analytics.model_comparison import compare_models
    result = compare_models(features)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.ml.artifacts import load_metadata, model_meta
from src.ml.predict import predict_frame


def compare_models(features: dict) -> dict[str, Any]:
    df = pd.DataFrame([features])
    results = {}
    for which in ("served", "challenger"):
        meta = model_meta(which)
        row = predict_frame(df, which).iloc[0]
        results[which] = {
            "model_name": meta["label"] + (" (served)" if which == "served" else " (challenger)"),
            "probability": round(float(row["probability"]), 4),
            "probability_pct": round(float(row["probability"]) * 100, 1),
            "predicted_class": row["predicted_class"],
            "risk_category": row["risk_category"],
            "test_auroc": meta["test"]["auroc"],
        }
    probs = np.array([r["probability"] for r in results.values()])
    classes = sorted({r["predicted_class"] for r in results.values()})
    agree = len(classes) == 1
    diff = load_metadata()["comparisons"]["auroc_difference_served_minus_challenger"]
    return {
        "available": True,
        "models": results,
        "agreement": {
            "sufficient_models": True,
            "n_models": 2,
            "spread": round(float(probs.max() - probs.min()), 4),
            "class_agreement": agree,
            "unanimous_class": classes[0] if agree else "Split",
            "agreement_pct": 100.0 if agree else 50.0,
            "test_auroc_difference": diff,
        },
        "features": features,
    }
