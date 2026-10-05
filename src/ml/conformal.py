"""
src/ml/conformal.py — Class-conditional (Mondrian) split conformal prediction.

Instead of an invented "± x%" band, each prediction comes with a prediction
set — {Benign}, {Malignant}, both, or (rarely) neither — that contains the true
class with probability at least 1 − alpha *for each true class separately*,
assuming the calibration and new cases are exchangeable. Class-conditional
coverage matters here: the guarantee holds for malignant cases on their own,
not only on average.

Method: LAC nonconformity score s = 1 − p̂(true class). For each class k with
n_k calibration cases, q̂_k is the ⌈(n_k + 1)(1 − alpha)⌉-th smallest score.
A class enters the set when 1 − p̂(k) ≤ q̂_k. For a binary model with
p = P(malignant) this reduces to two cut-offs on p:
    Benign    ∈ set  ⇔  p ≤ q̂_benign
    Malignant ∈ set  ⇔  p ≥ 1 − q̂_malignant

References: Vovk, Gammerman & Shafer (2005), Algorithmic Learning in a Random
World; Angelopoulos & Bates (2021), A Gentle Introduction to Conformal
Prediction and Distribution-Free Uncertainty Quantification.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

BENIGN, MALIGNANT = "Benign", "Malignant"


def conformal_quantile(scores, alpha: float) -> float:
    """Finite-sample-corrected quantile; 1.0 (include always) if n is too small."""
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    return 1.0 if k > n else float(scores[k - 1])


@dataclass
class MondrianConformal:
    alpha: float
    benign_if_p_le: float
    malignant_if_p_ge: float
    n_benign: int
    n_malignant: int

    def predict_sets(self, p) -> list[list[str]]:
        p = np.atleast_1d(np.asarray(p, dtype=float))
        sets = []
        for v in p:
            s = []
            if v <= self.benign_if_p_le:
                s.append(BENIGN)
            if v >= self.malignant_if_p_ge:
                s.append(MALIGNANT)
            sets.append(s)
        return sets

    def covers(self, p, y) -> np.ndarray:
        """Boolean per case: is the true label inside its prediction set?"""
        p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=int)
        return np.where(y == 1, p >= self.malignant_if_p_ge, p <= self.benign_if_p_le)

    def set_sizes(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        return (p <= self.benign_if_p_le).astype(int) + (p >= self.malignant_if_p_ge).astype(int)

    def to_dict(self) -> dict:
        return {
            "method": "Mondrian (class-conditional) split conformal, LAC score",
            "alpha": self.alpha,
            "coverage_target": round(1 - self.alpha, 4),
            "benign_if_p_le": self.benign_if_p_le,
            "malignant_if_p_ge": self.malignant_if_p_ge,
            "n_calibration": {"benign": self.n_benign, "malignant": self.n_malignant},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MondrianConformal":
        return cls(alpha=d["alpha"], benign_if_p_le=d["benign_if_p_le"],
                   malignant_if_p_ge=d["malignant_if_p_ge"],
                   n_benign=d["n_calibration"]["benign"],
                   n_malignant=d["n_calibration"]["malignant"])


def fit_mondrian(p_cal, y_cal, alpha: float = 0.10) -> MondrianConformal:
    """Fit on a calibration split never used for training, calibration or thresholds."""
    p_cal, y_cal = np.asarray(p_cal, dtype=float), np.asarray(y_cal, dtype=int)
    q_benign = conformal_quantile(p_cal[y_cal == 0], alpha)              # 1 − (1 − p)
    q_malignant = conformal_quantile(1 - p_cal[y_cal == 1], alpha)
    return MondrianConformal(alpha=alpha, benign_if_p_le=q_benign,
                             malignant_if_p_ge=1 - q_malignant,
                             n_benign=int((y_cal == 0).sum()),
                             n_malignant=int((y_cal == 1).sum()))
