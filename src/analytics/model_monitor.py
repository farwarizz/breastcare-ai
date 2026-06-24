"""
src/analytics/model_monitor.py — Model monitoring and data drift detection.

Tracks prediction distributions over time and detects:
- Data drift: input feature distributions vs training baseline (PSI, KS test)
- Prediction drift: prediction distribution shifts
- Performance degradation signals

Usage:
    from src.analytics.model_monitor import PredictionMonitor
    monitor = PredictionMonitor()
    monitor.record_prediction(features, probability)
    report = monitor.get_drift_report()
"""
from __future__ import annotations

import json
import logging
import numpy as np
from collections import deque
from typing import Any
from scipy import stats

logger = logging.getLogger(__name__)

# Training distribution baselines (from UCI Mammographic Masses dataset)
TRAINING_BASELINES = {
    "age": {"mean": 55.5, "std": 14.5, "min": 18, "max": 96},
    "shape": {"distribution": {1: 0.15, 2: 0.25, 3: 0.30, 4: 0.30}},
    "margin": {"distribution": {1: 0.25, 2: 0.10, 3: 0.20, 4: 0.25, 5: 0.20}},
    "density": {"distribution": {1: 0.20, 2: 0.35, 3: 0.30, 4: 0.15}},
    "probability": {"mean": 0.48, "std": 0.22},
}


class PredictionMonitor:
    """
    Real-time model monitoring for drift detection.

    Maintains a rolling window of recent predictions and compares
    against training-time baselines.
    """

    def __init__(self, window_size: int = 200):
        self.window_size = window_size
        self.recent_predictions: deque = deque(maxlen=window_size)
        self.recent_features: deque = deque(maxlen=window_size)
        self.total_predictions = 0

    def record_prediction(
        self, features: dict, probability: float
    ) -> None:
        """Record a prediction for monitoring."""
        self.recent_predictions.append(probability)
        self.recent_features.append(features)
        self.total_predictions += 1

    def get_drift_report(self) -> dict[str, Any]:
        """
        Generate a comprehensive drift report.

        Returns dict with:
        - feature_drift: per-feature drift metrics
        - prediction_drift: prediction distribution drift
        - alerts: list of triggered alerts
        - sample_size: number of predictions in current window
        """
        n = len(self.recent_predictions)
        if n < 10:
            return {
                "status": "insufficient_data",
                "sample_size": n,
                "message": f"Need at least 10 predictions, have {n}",
            }

        report = {
            "status": "ok",
            "sample_size": n,
            "total_predictions": self.total_predictions,
            "alerts": [],
            "feature_drift": {},
            "prediction_drift": {},
        }

        # Prediction drift
        probs = np.array(list(self.recent_predictions))
        report["prediction_drift"] = {
            "current_mean": round(float(probs.mean()), 4),
            "current_std": round(float(probs.std()), 4),
            "baseline_mean": TRAINING_BASELINES["probability"]["mean"],
            "baseline_std": TRAINING_BASELINES["probability"]["std"],
            "mean_shift": round(
                float(probs.mean()) - TRAINING_BASELINES["probability"]["mean"], 4
            ),
        }

        # PSI for prediction distribution
        pred_psi = self._compute_psi(
            probs,
            TRAINING_BASELINES["probability"]["mean"],
            TRAINING_BASELINES["probability"]["std"],
        )
        report["prediction_drift"]["psi"] = round(pred_psi, 4)
        if pred_psi > 0.2:
            report["alerts"].append({
                "type": "prediction_drift",
                "severity": "high",
                "message": f"Prediction drift detected (PSI={pred_psi:.3f} > 0.2)",
            })
        elif pred_psi > 0.1:
            report["alerts"].append({
                "type": "prediction_drift",
                "severity": "medium",
                "message": f"Minor prediction drift (PSI={pred_psi:.3f})",
            })

        # Feature drift (age)
        ages = np.array([f.get("age", 50) for f in self.recent_features])
        age_baseline = TRAINING_BASELINES["age"]
        ks_stat, ks_pval = self._ks_test_vs_normal(
            ages, age_baseline["mean"], age_baseline["std"]
        )
        report["feature_drift"]["age"] = {
            "current_mean": round(float(ages.mean()), 1),
            "baseline_mean": age_baseline["mean"],
            "ks_statistic": round(ks_stat, 4),
            "ks_pvalue": round(ks_pval, 4),
            "drift_detected": ks_pval < 0.05,
        }
        if ks_pval < 0.01:
            report["alerts"].append({
                "type": "feature_drift",
                "feature": "age",
                "severity": "high",
                "message": f"Age distribution drift (KS p={ks_pval:.4f})",
            })

        # Feature drift (categorical features)
        for feat in ["shape", "margin", "density"]:
            values = [f.get(feat, 1) for f in self.recent_features]
            baseline_dist = TRAINING_BASELINES.get(feat, {}).get("distribution", {})
            if baseline_dist:
                cat_drift = self._categorical_drift(values, baseline_dist)
                report["feature_drift"][feat] = cat_drift
                if cat_drift.get("chi2_pvalue", 1.0) < 0.05:
                    report["alerts"].append({
                        "type": "feature_drift",
                        "feature": feat,
                        "severity": "medium",
                        "message": f"{feat.title()} distribution changed (p={cat_drift['chi2_pvalue']:.4f})",
                    })

        if report["alerts"]:
            report["status"] = "drift_detected"

        return report

    def _compute_psi(
        self, values: np.ndarray, baseline_mean: float, baseline_std: float
    ) -> float:
        """
        Population Stability Index (PSI).
        Compares current distribution to baseline using histogram binning.
        PSI < 0.1: no significant change
        PSI 0.1-0.2: moderate change
        PSI > 0.2: significant change
        """
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)

        # Current distribution
        actual_hist, _ = np.histogram(values, bins=bins, density=True)
        actual_hist = actual_hist / actual_hist.sum() if actual_hist.sum() > 0 else actual_hist

        # Expected distribution (from baseline normal)
        expected_samples = np.random.RandomState(42).normal(
            baseline_mean, baseline_std, size=1000
        )
        expected_samples = np.clip(expected_samples, 0, 1)
        expected_hist, _ = np.histogram(expected_samples, bins=bins, density=True)
        expected_hist = expected_hist / expected_hist.sum() if expected_hist.sum() > 0 else expected_hist

        # PSI calculation with smoothing
        eps = 1e-4
        actual_hist = np.maximum(actual_hist, eps)
        expected_hist = np.maximum(expected_hist, eps)

        psi = np.sum((actual_hist - expected_hist) * np.log(actual_hist / expected_hist))
        return float(psi)

    def _ks_test_vs_normal(
        self, values: np.ndarray, baseline_mean: float, baseline_std: float
    ) -> tuple[float, float]:
        """Kolmogorov-Smirnov test against baseline normal distribution."""
        ks_stat, p_val = stats.kstest(values, "norm", args=(baseline_mean, baseline_std))
        return float(ks_stat), float(p_val)

    def _categorical_drift(
        self, values: list, baseline_dist: dict
    ) -> dict[str, Any]:
        """Chi-squared test for categorical feature drift."""
        from collections import Counter

        counts = Counter(values)
        categories = sorted(set(list(baseline_dist.keys()) + list(counts.keys())))
        n = len(values)

        observed = np.array([counts.get(c, 0) for c in categories])
        expected = np.array([baseline_dist.get(c, 0.01) * n for c in categories])
        expected = np.maximum(expected, 0.5)  # avoid zero

        chi2, p_val = stats.chisquare(observed, expected)

        return {
            "observed_distribution": {str(k): v for k, v in counts.items()},
            "chi2_statistic": round(float(chi2), 4),
            "chi2_pvalue": round(float(p_val), 4),
            "drift_detected": p_val < 0.05,
        }
