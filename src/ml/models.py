"""
src/ml/models.py — Candidate models, cross-validation and the selection rule.

Candidates (all on the four features age, shape, margin, density):
- logistic_regression — standardised age + one-hot shape, margin, density.
  One-hot because shape and margin are nominal codes (UCI documentation).
- lightgbm_monotone — LightGBM constrained so risk never decreases with age or
  with shape (round < oval < lobular < irregular). Margin is left unconstrained
  because its codes are not in risk order (obscured is less suspicious than
  microlobulated); density is unconstrained because 90% of masses share one
  value and the remaining groups are tiny.
- lightgbm — unconstrained LightGBM.

Selection rule (fixed before looking at the test set): pick the configuration
with the lowest mean 5-fold CV log-loss on the fit split; if a simpler family
is within one standard error of that best score, prefer the simplest such
family (logistic_regression < lightgbm_monotone < lightgbm).

LightGBM's tree count comes from early stopping inside each CV fold. That makes
its CV score slightly optimistic, which biases selection towards LightGBM.
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import FEATURE_NAMES

RANDOM_STATE = 42
N_FOLDS = 5

FAMILIES = {   # in order of simplicity
    "logistic_regression": "Logistic regression",
    "lightgbm_monotone":   "LightGBM (monotone in age and shape)",
    "lightgbm":            "LightGBM",
}
GRIDS = {
    "logistic_regression": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "lightgbm_monotone":   [{"num_leaves": nl, "min_child_samples": m}
                            for nl in (4, 8, 16) for m in (10, 20, 40)],
    "lightgbm":            [{"num_leaves": nl, "min_child_samples": m}
                            for nl in (4, 8, 16) for m in (10, 20, 40)],
}
MONOTONE = {"age": 1, "shape": 1, "margin": 0, "density": 0}
LGBM_FIXED = {"learning_rate": 0.03, "random_state": RANDOM_STATE, "verbose": -1}


def make_estimator(family: str, params: dict, n_estimators: int = 2000):
    if family == "logistic_regression":
        pre = ColumnTransformer([
            ("num", StandardScaler(), ["age"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, categories=[
                [1, 2, 3, 4], [1, 2, 3, 4, 5], [1, 2, 3, 4]]), ["shape", "margin", "density"]),
        ])
        return Pipeline([("pre", pre),
                         ("lr", LogisticRegression(C=params["C"], max_iter=5000))])
    if family in ("lightgbm", "lightgbm_monotone"):
        mono = [MONOTONE[f] for f in FEATURE_NAMES] if family == "lightgbm_monotone" else None
        return lgb.LGBMClassifier(n_estimators=n_estimators, monotone_constraints=mono,
                                  **LGBM_FIXED, **params)
    raise ValueError(f"unknown family {family!r}")


def _folds(y):
    return StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(np.zeros(len(y)), y)


def cross_validate(family: str, params: dict, X: pd.DataFrame, y: np.ndarray) -> dict:
    """5-fold CV log-loss (mean ± SE) and AUROC; LightGBM also returns its tree count."""
    losses, aucs, iters = [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, va in _folds(y):
            est = make_estimator(family, params)
            if family == "logistic_regression":
                est.fit(X.iloc[tr], y[tr])
            else:
                est.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
                iters.append(int(est.best_iteration_))
            p = est.predict_proba(X.iloc[va])[:, 1]
            losses.append(log_loss(y[va], p))
            aucs.append(roc_auc_score(y[va], p))
    result = {
        "family": family, "params": params,
        "cv_logloss": round(float(np.mean(losses)), 4),
        "cv_logloss_se": round(float(np.std(losses, ddof=1) / np.sqrt(len(losses))), 4),
        "cv_auroc": round(float(np.mean(aucs)), 4),
    }
    if iters:
        result["n_estimators"] = int(np.median(iters))
    return result


def final_params(result: dict) -> tuple[dict, int]:
    return result["params"], result.get("n_estimators", 0)


def build(result: dict):
    """Unfitted estimator for a CV result, with LightGBM's tree count fixed."""
    params, n_est = final_params(result)
    return make_estimator(result["family"], params, n_estimators=n_est or 2000)


def out_of_fold(result: dict, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """OOF probabilities with the configuration fixed (no early stopping on the fold)."""
    oof = np.zeros(len(y))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, va in _folds(y):
            est = build(result).fit(X.iloc[tr], y[tr])
            oof[va] = est.predict_proba(X.iloc[va])[:, 1]
    return oof


def select(results: list[dict]) -> dict:
    """
    Apply the selection rule. Returns {"best_per_family": ..., "selected": family,
    "best_overall": family, "rule": text}.
    """
    best = {}
    for r in results:
        if r["family"] not in best or r["cv_logloss"] < best[r["family"]]["cv_logloss"]:
            best[r["family"]] = r
    overall = min(best.values(), key=lambda r: r["cv_logloss"])
    bound = overall["cv_logloss"] + overall["cv_logloss_se"]
    selected = next(f for f in FAMILIES if f in best and best[f]["cv_logloss"] <= bound)
    return {
        "best_per_family": best,
        "best_overall": overall["family"],
        "selected": selected,
        "one_se_bound": round(bound, 4),
        "rule": ("Lowest mean 5-fold CV log-loss on the fit split; a simpler family within "
                 "one standard error of the best is preferred "
                 "(logistic regression < monotone LightGBM < LightGBM)."),
    }
