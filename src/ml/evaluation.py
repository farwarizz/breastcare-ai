"""
src/ml/evaluation.py — Held-out evaluation with bootstrap confidence intervals.

Every headline number is reported as {"value": x, "ci": [low, high]}: a 95%
percentile interval from B bootstrap resamples of the test set. The same
resample indices are reused across metrics and models, so differences between
models are paired.

Metrics
- Discrimination: AUROC, AUPRC (average precision).
- Calibration: Brier score; expected calibration error (10 equal-width bins);
  calibration intercept ("calibration-in-the-large", slope fixed at 1) and
  calibration slope from a logistic regression of the outcome on logit(p).
  Ideal: intercept 0, slope 1.
- Operating point: sensitivity, specificity, PPV, NPV, accuracy at a threshold
  chosen beforehand on development data.
- Decision curve analysis: net benefit = TP/n − FP/n · pt / (1 − pt)
  (Vickers & Elkin, 2006).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

from src.ml.calibration import logit


# ── Point metrics ────────────────────────────────────────────────
def _two_classes(y) -> bool:
    return 0 < int(np.sum(y)) < len(y)


def auroc(y, p) -> float:
    return float(roc_auc_score(y, p)) if _two_classes(y) else float("nan")


def auprc(y, p) -> float:
    return float(average_precision_score(y, p)) if _two_classes(y) else float("nan")


def brier(y, p) -> float:
    return float(brier_score_loss(y, p))


def ece(y, p, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width probability bins."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def calibration_intercept(y, p) -> float:
    """Calibration-in-the-large: intercept a in logit P(y) = a + logit(p)."""
    y, z = np.asarray(y, dtype=float), logit(p)

    def nll(a):
        q = 1.0 / (1.0 + np.exp(-(a + z)))
        q = np.clip(q, 1e-12, 1 - 1e-12)
        return -np.sum(y * np.log(q) + (1 - y) * np.log(1 - q))

    return float(minimize_scalar(nll, bounds=(-10, 10), method="bounded").x)


def calibration_slope(y, p) -> float:
    if not _two_classes(y):
        return float("nan")
    lr = LogisticRegression(penalty=None, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
    return float(lr.coef_[0, 0])


def threshold_counts(y, p, t: float) -> dict:
    y, pred = np.asarray(y, dtype=int), (np.asarray(p) >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _ratio(a: int, b: int) -> float:
    return a / b if b else float("nan")


def threshold_metrics(y, p, t: float) -> dict:
    c = threshold_counts(y, p, t)
    return {
        "sensitivity": _ratio(c["tp"], c["tp"] + c["fn"]),
        "specificity": _ratio(c["tn"], c["tn"] + c["fp"]),
        "ppv":         _ratio(c["tp"], c["tp"] + c["fp"]),
        "npv":         _ratio(c["tn"], c["tn"] + c["fn"]),
        "accuracy":    _ratio(c["tp"] + c["tn"], len(np.asarray(y))),
    }


# ── Thresholds (chosen on development data, never on test) ───────
def youden_threshold(y, p) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    return float(min(thr[int(np.argmax(j))], 1.0))


def sensitivity_threshold(y, p, target: float = 0.95) -> float:
    """Highest threshold whose sensitivity reaches the target."""
    fpr, tpr, thr = roc_curve(y, p)
    idx = int(np.argmax(tpr >= target))
    return float(min(thr[idx], 1.0))


# ── Decision curve analysis ──────────────────────────────────────
def net_benefit(y, p, thresholds) -> np.ndarray:
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    n = len(y)
    out = []
    for pt in thresholds:
        treat = p >= pt
        tp = np.sum(treat & (y == 1)); fp = np.sum(treat & (y == 0))
        out.append(tp / n - fp / n * pt / (1 - pt))
    return np.asarray(out)


def net_benefit_treat_all(y, thresholds) -> np.ndarray:
    prev = float(np.mean(y))
    return np.asarray([prev - (1 - prev) * pt / (1 - pt) for pt in thresholds])


# ── Bootstrap ────────────────────────────────────────────────────
class Bootstrap:
    """Percentile bootstrap over a fixed set of resample indices (paired across models)."""

    def __init__(self, n: int, n_resamples: int = 2000, seed: int = 2026, level: float = 0.95):
        rng = np.random.default_rng(seed)
        self.idx = rng.integers(0, n, size=(n_resamples, n))
        self.n, self.level = n, level

    def _interval(self, values) -> list[float]:
        v = np.asarray([x for x in values if np.isfinite(x)])
        lo, hi = np.percentile(v, [50 * (1 - self.level), 50 * (1 + self.level)])
        return [round(float(lo), 4), round(float(hi), 4)]

    def ci(self, stat) -> dict:
        """stat(indices) -> float, evaluated on the full sample and every resample."""
        value = stat(np.arange(self.n))
        return {"value": round(float(value), 4), "ci": self._interval(stat(i) for i in self.idx)}

    def paired(self, stat_a, stat_b) -> dict:
        diff = lambda i: stat_a(i) - stat_b(i)
        return self.ci(diff)


# ── Summaries ────────────────────────────────────────────────────
def summarize(y, p, threshold: float, boot: Bootstrap, conformal=None) -> dict:
    """All test metrics for one model, each with a 95% bootstrap CI."""
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    tm = lambda key: (lambda i: threshold_metrics(y[i], p[i], threshold)[key])
    out = {
        "n": int(len(y)), "n_malignant": int(y.sum()),
        "auroc": boot.ci(lambda i: auroc(y[i], p[i])),
        "auprc": boot.ci(lambda i: auprc(y[i], p[i])),
        "brier": boot.ci(lambda i: brier(y[i], p[i])),
        "ece": boot.ci(lambda i: ece(y[i], p[i])),
        "calibration_intercept": boot.ci(lambda i: calibration_intercept(y[i], p[i])),
        "calibration_slope": boot.ci(lambda i: calibration_slope(y[i], p[i])),
        "threshold": round(float(threshold), 4),
        "sensitivity": boot.ci(tm("sensitivity")),
        "specificity": boot.ci(tm("specificity")),
        "ppv": boot.ci(tm("ppv")),
        "npv": boot.ci(tm("npv")),
        "accuracy": boot.ci(tm("accuracy")),
        "counts": threshold_counts(y, p, threshold),
    }
    if conformal is not None:
        covered = conformal.covers(p, y)
        sizes = conformal.set_sizes(p)
        cov_class = lambda k: (lambda i: covered[i][y[i] == k].mean() if (y[i] == k).any() else np.nan)
        out["conformal"] = {
            "coverage": boot.ci(lambda i: covered[i].mean()),
            "coverage_benign": boot.ci(cov_class(0)),
            "coverage_malignant": boot.ci(cov_class(1)),
            "share_single_label": boot.ci(lambda i: (sizes[i] == 1).mean()),
            "share_both_labels": boot.ci(lambda i: (sizes[i] == 2).mean()),
            "share_empty": boot.ci(lambda i: (sizes[i] == 0).mean()),
        }
    return out


def subgroup_summary(y, p, groups, threshold: float, n_resamples: int = 2000,
                     seed: int = 2026) -> dict:
    """Per-subgroup AUROC, sensitivity, specificity and calibration intercept with CIs."""
    y, p, groups = np.asarray(y, dtype=int), np.asarray(p, dtype=float), np.asarray(groups)
    out = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        yg, pg = y[m], p[m]
        boot = Bootstrap(len(yg), n_resamples, seed)
        tm = lambda key: (lambda i: threshold_metrics(yg[i], pg[i], threshold)[key])
        out[str(g)] = {
            "n": int(m.sum()), "n_malignant": int(yg.sum()),
            "auroc": boot.ci(lambda i: auroc(yg[i], pg[i])),
            "sensitivity": boot.ci(tm("sensitivity")),
            "specificity": boot.ci(tm("specificity")),
            "calibration_intercept": boot.ci(lambda i: calibration_intercept(yg[i], pg[i])),
        }
    return out
