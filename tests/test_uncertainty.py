"""Unit tests for calibration, conformal prediction and evaluation maths."""
import math

import numpy as np
import pytest

from src.ml.calibration import Calibrator, fit_calibrator, select_calibrator
from src.ml.conformal import BENIGN, MALIGNANT, conformal_quantile, fit_mondrian
from src.ml.evaluation import (
    Bootstrap, calibration_intercept, calibration_slope, ece, net_benefit,
    net_benefit_treat_all, sensitivity_threshold, threshold_metrics, youden_threshold,
)

RNG = np.random.default_rng(0)


def _calibrated_sample(n=4000):
    p = RNG.uniform(0.02, 0.98, n)
    y = (RNG.uniform(size=n) < p).astype(int)
    return p, y


# ── Conformal ────────────────────────────────────────────────────
def test_conformal_quantile_uses_finite_sample_correction():
    scores = np.arange(1, 10) / 10           # n = 9
    # ceil((9 + 1) * 0.9) = 9 → the 9th smallest score
    assert conformal_quantile(scores, 0.10) == pytest.approx(0.9)
    # too few points for the requested level → include always
    assert conformal_quantile([0.2, 0.4], 0.10) == 1.0


def test_mondrian_coverage_holds_per_class():
    alpha, cover_b, cover_m = 0.10, [], []
    for seed in range(500):
        rng = np.random.default_rng(seed)
        p = rng.uniform(0.02, 0.98, 600)
        y = (rng.uniform(size=600) < p).astype(int)
        cp = fit_mondrian(p[:300], y[:300], alpha)
        covered = cp.covers(p[300:], y[300:])
        cover_b.append(covered[y[300:] == 0].mean())
        cover_m.append(covered[y[300:] == 1].mean())
    # Finite-sample guarantee: expected per-class coverage >= 1 - alpha.
    # Allow 3 Monte-Carlo standard errors of slack.
    for cov in (np.array(cover_b), np.array(cover_m)):
        se = cov.std(ddof=1) / np.sqrt(len(cov))
        assert cov.mean() >= 1 - alpha - 3 * se


def test_prediction_sets_follow_the_cutoffs():
    cp = fit_mondrian(*_calibrated_sample(), alpha=0.10)
    lo, hi = cp.benign_if_p_le, cp.malignant_if_p_ge
    assert cp.predict_sets([0.0])[0] == [BENIGN]
    assert cp.predict_sets([1.0])[0] == [MALIGNANT]
    mid = (lo + hi) / 2
    expected = ([BENIGN] if mid <= lo else []) + ([MALIGNANT] if mid >= hi else [])
    assert cp.predict_sets([mid])[0] == expected
    assert cp.from_dict(cp.to_dict()) == cp


# ── Calibration ──────────────────────────────────────────────────
def test_sigmoid_calibration_recovers_a_known_distortion():
    p_true, y = _calibrated_sample()
    z = np.log(p_true / (1 - p_true))
    p_overconfident = 1 / (1 + np.exp(-2.5 * z))      # too extreme
    cal = fit_calibrator("sigmoid", p_overconfident, y)
    assert cal.params["a"] == pytest.approx(0.4, abs=0.08)
    assert ece(y, cal.transform(p_overconfident)) < ece(y, p_overconfident)


def test_selection_keeps_well_calibrated_scores_untouched():
    p, y = _calibrated_sample()
    cal, scores = select_calibrator(p, y)
    assert set(scores) == {"none", "sigmoid", "isotonic"}
    # recalibrating an already calibrated score cannot help much
    assert min(scores.values()) == pytest.approx(scores["none"], abs=0.002)


def test_calibrator_roundtrip_through_json():
    p, y = _calibrated_sample(500)
    for method in ("none", "sigmoid", "isotonic"):
        cal = fit_calibrator(method, p, y)
        again = Calibrator.from_dict(cal.to_dict())
        assert np.allclose(cal.transform(p), again.transform(p))


# ── Evaluation ───────────────────────────────────────────────────
def test_perfectly_calibrated_scores_have_ideal_calibration_metrics():
    p, y = _calibrated_sample(20000)
    assert calibration_intercept(y, p) == pytest.approx(0.0, abs=0.06)
    assert calibration_slope(y, p) == pytest.approx(1.0, abs=0.06)
    assert ece(y, p) < 0.02


def test_threshold_metrics_on_a_hand_checked_example():
    y = np.array([1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.6, 0.2, 0.7, 0.3, 0.1, 0.05])
    m = threshold_metrics(y, p, 0.5)          # TP=2 FN=1 FP=1 TN=3
    assert m["sensitivity"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(3 / 4)
    assert m["ppv"] == pytest.approx(2 / 3)
    assert m["npv"] == pytest.approx(3 / 4)


def test_threshold_helpers():
    p, y = _calibrated_sample()
    t95 = sensitivity_threshold(y, p, 0.95)
    assert threshold_metrics(y, p, t95)["sensitivity"] >= 0.95
    assert 0.0 < youden_threshold(y, p) < 1.0


def test_net_benefit_formula():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.8, 0.6, 0.4, 0.1])
    # at pt = 0.5: treat the first two → TP = 1, FP = 1 → 1/4 − 1/4 · 1 = 0
    assert net_benefit(y, p, [0.5])[0] == pytest.approx(0.0)
    assert net_benefit_treat_all(y, [0.5])[0] == pytest.approx(0.5 - 0.5)


def test_bootstrap_is_reproducible_and_brackets_the_estimate():
    p, y = _calibrated_sample(300)
    a = Bootstrap(len(y), 500, seed=1).ci(lambda i: float(np.mean(y[i])))
    b = Bootstrap(len(y), 500, seed=1).ci(lambda i: float(np.mean(y[i])))
    assert a == b
    assert a["ci"][0] <= a["value"] <= a["ci"][1]
    assert not math.isnan(a["ci"][0])
