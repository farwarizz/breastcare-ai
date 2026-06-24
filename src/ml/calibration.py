"""
src/ml/calibration.py — Probability calibration and uncertainty quantification.

Ensures model probabilities are clinically meaningful:
- Platt scaling (sigmoid): parametric calibration via LogisticRegression
- Isotonic regression: non-parametric calibration
- Conformal prediction: coverage-guaranteed prediction intervals

Usage:
    from src.ml.calibration import CalibratedModel, conformal_predict
    cal_model = CalibratedModel(base_model, method="platt")
    cal_model.fit(X_val, y_val)
    prob = cal_model.predict_proba(X_test)
    intervals = conformal_predict(model, X_cal, y_cal, X_test, alpha=0.1)
"""
from __future__ import annotations

import json
import logging
import numpy as np
from pathlib import Path
from typing import Any
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import CALIBRATION_PATH

logger = logging.getLogger(__name__)


class CalibratedModel:
    """
    Wrapper that adds probability calibration to any sklearn-compatible classifier.

    Methods:
        'platt'    — Platt scaling (sigmoid/logistic)
        'isotonic' — Isotonic regression (non-parametric)
        'auto'     — Tries both, keeps the one with lower Brier score
    """

    def __init__(self, base_model: Any, method: str = "auto"):
        self.base_model = base_model
        self.method = method
        self.calibrator_ = None
        self.method_used_ = None
        self.metrics_: dict[str, Any] = {}

    def fit(self, X_val: np.ndarray, y_val: np.ndarray) -> "CalibratedModel":
        """
        Fit the calibration model on a held-out validation set.

        The base model should already be trained. This only learns
        the probability mapping.
        """
        raw_probs = self.base_model.predict_proba(X_val)[:, 1]

        if self.method == "auto":
            # Try both, pick lower Brier score
            platt = LogisticRegression(max_iter=1000)
            platt.fit(raw_probs.reshape(-1, 1), y_val)
            platt_probs = platt.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
            platt_brier = brier_score_loss(y_val, platt_probs)

            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            iso.fit(raw_probs, y_val)
            iso_probs = iso.predict(raw_probs)
            iso_brier = brier_score_loss(y_val, iso_probs)

            if platt_brier <= iso_brier:
                self.calibrator_ = platt
                self.method_used_ = "platt"
            else:
                self.calibrator_ = iso
                self.method_used_ = "isotonic"

            self.metrics_ = {
                "platt_brier": round(platt_brier, 6),
                "isotonic_brier": round(iso_brier, 6),
                "uncalibrated_brier": round(brier_score_loss(y_val, raw_probs), 6),
                "method_selected": self.method_used_,
            }
        elif self.method == "platt":
            self.calibrator_ = LogisticRegression(max_iter=1000)
            self.calibrator_.fit(raw_probs.reshape(-1, 1), y_val)
            self.method_used_ = "platt"
        elif self.method == "isotonic":
            self.calibrator_ = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            self.calibrator_.fit(raw_probs, y_val)
            self.method_used_ = "isotonic"

        logger.info("Calibration fitted with method=%s", self.method_used_)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities."""
        raw_probs = self.base_model.predict_proba(X)[:, 1]

        if self.method_used_ == "platt":
            cal_probs = self.calibrator_.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
        elif self.method_used_ == "isotonic":
            cal_probs = self.calibrator_.predict(raw_probs)
        else:
            cal_probs = raw_probs

        # Stack into (n, 2) array
        return np.column_stack([1 - cal_probs, cal_probs])


def conformal_predict(
    model: Any,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 0.1,
) -> list[dict[str, Any]]:
    """
    Split conformal prediction for coverage-guaranteed intervals.

    Uses a calibration set to compute nonconformity scores, then
    constructs prediction sets at (1-alpha) coverage level.

    Parameters
    ----------
    model : fitted classifier with predict_proba
    X_cal, y_cal : calibration set (held-out from training)
    X_test : test samples to produce intervals for
    alpha : significance level (0.1 = 90% coverage)

    Returns
    -------
    List of dicts with keys: lower, upper, width, coverage_level
    """
    cal_probs = model.predict_proba(X_cal)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    # Nonconformity scores: |p - y|
    scores = np.abs(cal_probs - y_cal)

    # Quantile threshold
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_hat = np.quantile(scores, min(q_level, 1.0))

    intervals = []
    for p in test_probs:
        lower = max(0.0, float(p - q_hat))
        upper = min(1.0, float(p + q_hat))
        intervals.append({
            "point_estimate": round(float(p), 4),
            "lower": round(lower, 4),
            "upper": round(upper, 4),
            "width": round(upper - lower, 4),
            "coverage_level": round(1 - alpha, 2),
        })

    return intervals


def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """
    Compute reliability diagram data for calibration visualization.
    """
    fraction_pos, mean_predicted = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    brier = brier_score_loss(y_true, y_prob)

    return {
        "fraction_positive": fraction_pos.tolist(),
        "mean_predicted": mean_predicted.tolist(),
        "brier_score": round(brier, 6),
        "n_bins": n_bins,
    }


def save_calibration_results(results: dict) -> None:
    """Persist calibration results to disk."""
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Calibration results saved: %s", CALIBRATION_PATH)
