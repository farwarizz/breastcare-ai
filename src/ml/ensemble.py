"""
src/ml/ensemble.py — Stacking ensemble combining XGBoost, LightGBM, Random Forest, and SVM.

Architecture:
    Base models (Level 0): XGBoost, LightGBM, RandomForest, SVM
    Meta-learner (Level 1): LogisticRegression on out-of-fold predictions
    Calibration: Optional Platt scaling for clinical-grade probabilities

The stacking approach produces a single, calibrated probability that benefits
from model diversity — each base model captures different decision boundaries.

Usage:
    from src.ml.ensemble import StackingEnsemble
    ensemble = StackingEnsemble()
    ensemble.fit(X_train, y_train)
    prob = ensemble.predict_proba(X_test)
"""
from __future__ import annotations

import logging
import numpy as np
import joblib
from pathlib import Path
from typing import Any
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score
import lightgbm as lgb
import xgboost as xgb
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import ENSEMBLE_MODEL_PATH, DECISION_THRESHOLD

logger = logging.getLogger(__name__)


class StackingEnsemble:
    """
    Two-level stacking ensemble for binary classification.

    Level 0: Four diverse base models trained with out-of-fold predictions.
    Level 1: LogisticRegression meta-learner combining OOF predictions.
    """

    def __init__(
        self,
        n_folds: int = 5,
        random_state: int = 42,
        xgb_params: dict | None = None,
        lgbm_params: dict | None = None,
    ):
        self.n_folds = n_folds
        self.random_state = random_state

        # Base model configurations
        self.xgb_params = xgb_params or {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "eval_metric": "logloss",
            "random_state": random_state,
            "verbosity": 0,
        }
        self.lgbm_params = lgbm_params or {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": random_state,
            "verbose": -1,
        }

        # Models (populated after fit)
        self.base_models_: dict[str, list] = {}
        self.meta_learner_: LogisticRegression | None = None
        self.is_fitted_ = False
        self.metrics_: dict[str, Any] = {}

    def _create_base_models(self) -> dict[str, Any]:
        """Instantiate fresh base model instances."""
        return {
            "xgboost": xgb.XGBClassifier(**self.xgb_params),
            "lightgbm": lgb.LGBMClassifier(**self.lgbm_params),
            "random_forest": RandomForestClassifier(
                n_estimators=200, random_state=self.random_state, n_jobs=-1,
            ),
            "svm": Pipeline([
                ("scaler", StandardScaler()),
                ("svm", SVC(probability=True, random_state=self.random_state)),
            ]),
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackingEnsemble":
        """
        Train the stacking ensemble using out-of-fold predictions.

        1. For each fold, train all base models on the training portion
           and predict on the held-out portion → OOF predictions.
        2. Train meta-learner on the OOF prediction matrix.
        3. Retrain all base models on the full dataset for inference.
        """
        logger.info("Training stacking ensemble (n_folds=%d)", self.n_folds)
        kfold = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )
        model_names = list(self._create_base_models().keys())
        n_models = len(model_names)

        # OOF prediction matrix
        oof_predictions = np.zeros((len(y), n_models))
        fold_models = {name: [] for name in model_names}

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold = y[train_idx]

            models = self._create_base_models()
            for i, (name, model) in enumerate(models.items()):
                model.fit(X_train_fold, y_train_fold)
                prob = model.predict_proba(X_val_fold)[:, 1]
                oof_predictions[val_idx, i] = prob
                fold_models[name].append(model)

            logger.debug("Fold %d/%d complete", fold_idx + 1, self.n_folds)

        # Train meta-learner on OOF predictions
        self.meta_learner_ = LogisticRegression(
            random_state=self.random_state, max_iter=1000
        )
        self.meta_learner_.fit(oof_predictions, y)

        # OOF ensemble performance
        meta_prob = self.meta_learner_.predict_proba(oof_predictions)[:, 1]
        self.metrics_ = {
            "oof_auc": round(roc_auc_score(y, meta_prob), 4),
            "oof_accuracy": round(accuracy_score(y, (meta_prob >= 0.5).astype(int)) * 100, 2),
            "oof_recall": round(recall_score(y, (meta_prob >= 0.5).astype(int)) * 100, 2),
            "n_folds": self.n_folds,
            "n_base_models": n_models,
            "base_model_names": model_names,
            "meta_learner_weights": dict(zip(
                model_names,
                self.meta_learner_.coef_[0].round(4).tolist(),
            )),
        }
        logger.info(
            "Ensemble OOF AUC: %.4f | Accuracy: %.2f%% | Recall: %.2f%%",
            self.metrics_["oof_auc"],
            self.metrics_["oof_accuracy"],
            self.metrics_["oof_recall"],
        )

        # Retrain base models on full dataset for inference
        self.base_models_ = {}
        final_models = self._create_base_models()
        for name, model in final_models.items():
            model.fit(X, y)
            self.base_models_[name] = model

        self.is_fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities using the stacking ensemble.

        Returns (n_samples, 2) array of [P(benign), P(malignant)].
        """
        if not self.is_fitted_:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        base_predictions = np.column_stack([
            model.predict_proba(X)[:, 1]
            for model in self.base_models_.values()
        ])

        meta_probs = self.meta_learner_.predict_proba(base_predictions)
        return meta_probs

    def predict(self, X: np.ndarray, threshold: float = DECISION_THRESHOLD) -> np.ndarray:
        """Predict binary class labels."""
        prob = self.predict_proba(X)[:, 1]
        return (prob >= threshold).astype(int)

    def get_base_predictions(self, X: np.ndarray) -> dict[str, float]:
        """Get individual predictions from each base model (for comparison)."""
        if not self.is_fitted_:
            raise RuntimeError("Ensemble not fitted.")
        return {
            name: float(model.predict_proba(X)[:, 1][0])
            for name, model in self.base_models_.items()
        }

    def save(self, path: Path | None = None) -> None:
        """Save the ensemble to disk."""
        path = path or ENSEMBLE_MODEL_PATH
        joblib.dump(self, path, compress=3)
        logger.info("Ensemble saved: %s", path)

    @staticmethod
    def load(path: Path | None = None) -> "StackingEnsemble":
        """Load a saved ensemble from disk."""
        path = path or ENSEMBLE_MODEL_PATH
        ensemble = joblib.load(path)
        logger.info("Ensemble loaded: %s", path)
        return ensemble
