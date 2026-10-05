"""
src/analytics/global_shap.py — Dataset-wide feature contributions for the served model.

Computes per-feature contributions (SHAP values, log-odds) for every cleaned
case in data/processed/cleaned.csv and summarises global importance. If that
file is missing it reports "unavailable"; it never substitutes generated data.

Usage:
    from src.analytics.global_shap import compute_global_shap
    data = compute_global_shap()
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import CLEANED_CSV, FEATURE_NAMES
from src.ml.explain import contributions, method_name

logger = logging.getLogger(__name__)


def compute_global_shap(df: pd.DataFrame | None = None) -> dict[str, Any]:
    if df is None:
        if not CLEANED_CSV.exists():
            return {"available": False,
                    "message": f"{CLEANED_CSV.name} not found; run `python -m src.ml.train`."}
        df = pd.read_csv(CLEANED_CSV)
    try:
        base, sv = contributions(df)
    except FileNotFoundError as exc:
        return {"available": False, "message": str(exc)}

    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    X = df[FEATURE_NAMES].to_numpy(dtype=float)
    return {
        "available": True,
        "method": method_name(),
        "n_samples": int(len(df)),
        "base_value": round(base, 4),
        "global_importance": [
            {"feature": FEATURE_NAMES[i], "mean_abs_shap": round(float(mean_abs[i]), 4), "rank": r + 1}
            for r, i in enumerate(order)
        ],
        "feature_stats": {
            name: {
                "mean": round(float(sv[:, i].mean()), 4),
                "std": round(float(sv[:, i].std()), 4),
                "min": round(float(sv[:, i].min()), 4),
                "max": round(float(sv[:, i].max()), 4),
                "pct_positive": round(float((sv[:, i] > 0).mean() * 100), 1),
            }
            for i, name in enumerate(FEATURE_NAMES)
        },
        "dependence_data": {
            name: {"feature_values": X[:200, i].tolist(),
                   "shap_values": np.round(sv[:200, i], 4).tolist()}
            for i, name in enumerate(FEATURE_NAMES)
        },
    }
