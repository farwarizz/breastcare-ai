"""
src/ml/hyperopt.py — Optuna-based hyperparameter optimization for all models.

Runs Bayesian optimization to find optimal hyperparameters for LightGBM,
XGBoost, and the stacking ensemble. Results cached in models/optuna_results.json.

Usage:
    from src.ml.hyperopt import optimize_lgbm, optimize_xgboost
    best_params = optimize_lgbm(X, y, n_trials=50)
"""
from __future__ import annotations

import json
import logging
import numpy as np
from pathlib import Path
from typing import Any
from sklearn.model_selection import StratifiedKFold, cross_val_score
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import OPTUNA_RESULTS_PATH

logger = logging.getLogger(__name__)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_OK = True
except ImportError:
    _OPTUNA_OK = False


def _check_optuna() -> None:
    if not _OPTUNA_OK:
        raise ImportError("optuna is required: pip install optuna>=3.0")


def optimize_lgbm(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 50,
    n_folds: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Optimize LightGBM hyperparameters using Optuna.

    Returns dict with best_params, best_auc, and study summary.
    """
    _check_optuna()
    import lightgbm as lgb

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "min_child_samples": trial.suggest_int("min_child_samples", 3, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": random_state,
            "verbose": -1,
        }
        model = lgb.LGBMClassifier(**params)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize", study_name="lgbm_optimization")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    result = {
        "model": "lightgbm",
        "best_params": study.best_params,
        "best_auc": round(study.best_value, 4),
        "n_trials": n_trials,
        "n_folds": n_folds,
    }
    logger.info("LightGBM Optuna best AUC: %.4f", study.best_value)
    return result


def optimize_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 50,
    n_folds: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Optimize XGBoost hyperparameters using Optuna.
    """
    _check_optuna()
    import xgboost as xgb_lib

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "eval_metric": "logloss",
            "random_state": random_state,
            "verbosity": 0,
        }
        model = xgb_lib.XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize", study_name="xgboost_optimization")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    result = {
        "model": "xgboost",
        "best_params": study.best_params,
        "best_auc": round(study.best_value, 4),
        "n_trials": n_trials,
        "n_folds": n_folds,
    }
    logger.info("XGBoost Optuna best AUC: %.4f", study.best_value)
    return result


def run_full_optimization(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 50,
) -> dict[str, Any]:
    """
    Run Optuna optimization for all models and save results.
    """
    results = {}

    logger.info("Starting full hyperparameter optimization (%d trials per model)", n_trials)

    results["lightgbm"] = optimize_lgbm(X, y, n_trials=n_trials)
    results["xgboost"] = optimize_xgboost(X, y, n_trials=n_trials)

    # Save results
    OPTUNA_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OPTUNA_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Optuna results saved: %s", OPTUNA_RESULTS_PATH)
    return results
