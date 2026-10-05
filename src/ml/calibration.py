"""
src/ml/calibration.py — Probability calibration fitted on out-of-fold predictions.

The calibration map is learned from out-of-fold (OOF) predictions on the fit
split, which behave like predictions on unseen data, and then applied to the
final model trained on the whole fit split (the same scheme as scikit-learn's
CalibratedClassifierCV with ensemble=False).

Candidates: "none", "sigmoid" (Platt scaling on the logit) and "isotonic".
The method is chosen by cross-validated Brier score on the OOF predictions,
with the simpler method winning ties. Parameters are plain numbers, so the
calibrator is stored in model_metadata.json rather than as a pickle.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold

METHODS = ("none", "sigmoid", "isotonic")   # simplest first
_EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


@dataclass
class Calibrator:
    method: str = "none"
    params: dict = field(default_factory=dict)

    def transform(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if self.method == "none":
            return p
        if self.method == "sigmoid":
            z = self.params["a"] * logit(p) + self.params["b"]
            return 1.0 / (1.0 + np.exp(-z))
        if self.method == "isotonic":
            return np.interp(p, self.params["x"], self.params["y"])
        raise ValueError(f"unknown calibration method {self.method!r}")

    def to_dict(self) -> dict:
        return {"method": self.method, "params": self.params}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibrator":
        return cls(method=d["method"], params=d.get("params", {}))


def fit_calibrator(method: str, p, y) -> Calibrator:
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=int)
    if method == "none":
        return Calibrator("none")
    if method == "sigmoid":
        lr = LogisticRegression(penalty=None, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
        return Calibrator("sigmoid", {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])})
    if method == "isotonic":
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)
        return Calibrator("isotonic", {"x": iso.X_thresholds_.tolist(),
                                       "y": iso.y_thresholds_.tolist()})
    raise ValueError(f"unknown calibration method {method!r}")


def select_calibrator(p_oof, y, n_splits: int = 5, seed: int = 42):
    """
    Choose the method with the lowest cross-validated Brier score on the OOF
    predictions (ties → the simpler method), then fit it on all OOF predictions.
    Returns (calibrator, {method: mean CV Brier}).
    """
    p_oof, y = np.asarray(p_oof, dtype=float), np.asarray(y, dtype=int)
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    scores = {}
    for method in METHODS:
        fold_scores = []
        for tr, va in cv.split(p_oof, y):
            cal = fit_calibrator(method, p_oof[tr], y[tr])
            fold_scores.append(brier_score_loss(y[va], cal.transform(p_oof[va])))
        scores[method] = float(np.mean(fold_scores))
    best = min(METHODS, key=lambda m: (round(scores[m], 6), METHODS.index(m)))
    return fit_calibrator(best, p_oof, y), scores
