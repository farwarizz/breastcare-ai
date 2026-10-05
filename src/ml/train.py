"""
src/ml/train.py — Multi-dataset training pipeline.
Trains LightGBM on the UCI Mammographic Mass dataset (the model the app serves)
and XGBoost on WDBC (research benchmark) with a 5-model 10-fold CV comparison.

Data: training only runs on the official UCI file, verified by SHA-256
(see data/README.md). If the file is missing or altered it stops with an
error; it never substitutes generated data.

Known limitation (scheduled for Phase 1): early stopping and the Youden
decision thresholds are selected on the test split, so the reported test
metrics are optimistically biased. EVALUATION_PROTOCOL below records this
in model_metadata.json.

Run:  python -m src.ml.train
"""
import hashlib
import json
import platform
import warnings
import numpy as np
import pandas as pd
import joblib
import shap
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import sklearn
import xgboost as xgb

try:
    from imblearn.over_sampling import SMOTE
    _SMOTE_AVAILABLE = True
except ImportError:
    _SMOTE_AVAILABLE = False
    print("  [INFO] imbalanced-learn not installed — skipping SMOTE augmentation")

warnings.filterwarnings("ignore")

# Ensure src is importable when run as module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    DATA_RAW_DIR, DATA_PROCESSED_DIR, CLEANED_CSV,
    MODELS_DIR, LGBM_MODEL_PATH, LGBM_SHAP_PATH,
    XGBOOST_MODEL_PATH, XGBOOST_SHAP_PATH,
    MODEL_METADATA_PATH, WDBC_FEATURES_PATH,
    DECISION_THRESHOLD, FEATURE_NAMES,
    SHAPE_MAP, MARGIN_MAP, DENSITY_MAP,
    MAMMO_UCI_ID, MAMMO_RAW_PATH, MAMMO_SOURCE_URL, MAMMO_SHA256,
    MAMMO_DOI, MAMMO_RAW_COLUMNS,
)

TEST_SIZE    = 0.2
RANDOM_STATE = 42

# Describes exactly what this script does — written to model_metadata.json.
EVALUATION_PROTOCOL = (
    "Single stratified 80/20 train/test split (random_state=42) per dataset. "
    "LightGBM early stopping and both Youden's-J decision thresholds are selected "
    "on the test split, so test metrics are optimistically biased (to be fixed in "
    "Phase 1 with a separate validation split). SMOTE, when imbalanced-learn is "
    "installed, is applied to the WDBC training split only. The 5-model comparison "
    "is stratified 10-fold CV on the WDBC training split. No probability calibration "
    "and no uncertainty intervals."
)

# ── helpers ──────────────────────────────────────────────────────
def ensure_dirs():
    for d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _youden_threshold(model, X_val, y_val):
    """Compute optimal decision threshold via Youden's J statistic on a validation set."""
    prob = model.predict_proba(X_val)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_val, prob)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(np.clip(thresholds[idx], 0.3, 0.7))


def _metrics(model, X_test, y_test, threshold=DECISION_THRESHOLD):
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= threshold).astype(int)
    return {
        "accuracy":    round(accuracy_score(y_test, pred) * 100, 2),
        "auc":         round(roc_auc_score(y_test, prob), 4),
        "recall":      round(recall_score(y_test, pred) * 100, 2),
        "precision":   round(precision_score(y_test, pred, zero_division=0) * 100, 2),
        "f1":          round(f1_score(y_test, pred) * 100, 2),
        "specificity": round(accuracy_score(y_test[y_test == 0], pred[y_test == 0]) * 100, 2)
                        if (y_test == 0).any() else 0.0,
        "false_negatives": int(((pred == 0) & (y_test == 1)).sum()),
        "false_positives": int(((pred == 1) & (y_test == 0)).sum()),
    }


# ── Dataset 1: UCI Mammographic Mass (id 161) ────────────────────
# Documented UCI codes; anything else means the file is not what we expect.
_VALID_CODES = {
    "shape":    set(SHAPE_MAP),     # 1–4 (nominal)
    "margin":   set(MARGIN_MAP),    # 1–5 (nominal)
    "density":  set(DENSITY_MAP),   # 1–4 (ordinal)
    "severity": {0, 1},             # 0 = benign, 1 = malignant
}


def verify_raw_file(path: Path = MAMMO_RAW_PATH) -> str:
    """
    Return the SHA-256 of the raw file. Raises if the file is missing or is
    not byte-identical to the official UCI file.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Fetch the real UCI file with "
            "`python scripts/download_data.py` (training never substitutes generated data)."
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != MAMMO_SHA256:
        raise ValueError(
            f"{path} is not the official UCI Mammographic Mass file "
            f"(sha256 {digest}, expected {MAMMO_SHA256}). Refusing to train on it. "
            "Re-fetch it with `python scripts/download_data.py --force`."
        )
    return digest


def read_raw_mammographic(path: Path = MAMMO_RAW_PATH) -> pd.DataFrame:
    """Read the headerless UCI file; '?' marks a missing value."""
    return pd.read_csv(path, header=None, names=MAMMO_RAW_COLUMNS, na_values=["?"])


def clean_mammographic(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Features: age (years), shape, margin, density (integer codes).
    Target: severity (0 = benign, 1 = malignant).
    BI-RADS is dropped: it is the radiologist's assessment, so using it as a
    feature would leak the conclusion into the model.
    Rows with any missing feature are dropped (codes cannot be meaningfully imputed).
    Raises ValueError if any value falls outside the documented UCI codes.
    """
    df = raw.drop(columns=["birads"]).dropna()
    df = df.astype({"shape": int, "margin": int, "density": int, "severity": int})

    problems = [
        f"{col}: unexpected values {sorted(df.loc[~df[col].isin(valid), col].unique())}"
        for col, valid in _VALID_CODES.items()
        if not df[col].isin(valid).all()
    ]
    if not df["age"].between(1, 120).all():
        problems.append("age: values outside 1-120")
    if problems:
        raise ValueError("Mammographic data failed validation — " + "; ".join(problems))

    return df[FEATURE_NAMES + ["severity"]]


def load_mammographic(path: Path = MAMMO_RAW_PATH):
    """
    Verify, load and clean the UCI Mammographic Mass file; save cleaned.csv.
    Returns (cleaned DataFrame, provenance dict for model_metadata.json).
    """
    digest = verify_raw_file(path)
    print(f"  Loading mammographic data from {path} (sha256 verified)")
    raw = read_raw_mammographic(path)
    df = clean_mammographic(raw)

    print(f"  Raw: {len(raw)} rows  |  Cleaned: {len(df)} rows  |  "
          f"Malignant: {int(df.severity.sum())}  |  Benign: {int((df.severity == 0).sum())}")
    df.to_csv(CLEANED_CSV, index=False)

    provenance = {
        "name":         "UCI Mammographic Mass",
        "uci_id":       MAMMO_UCI_ID,
        "doi":          MAMMO_DOI,
        "license":      "CC BY 4.0",
        "source_url":   MAMMO_SOURCE_URL,
        "raw_sha256":   digest,
        "raw_rows":     int(len(raw)),
        "raw_missing":  {col: int(n) for col, n in raw.isna().sum().items()},
        "clean_rows":   int(len(df)),
        "benign":       int((df.severity == 0).sum()),
        "malignant":    int(df.severity.sum()),
        "features":     list(FEATURE_NAMES),
        "excluded_columns": {"birads": "radiologist's assessment (label leakage)"},
        "target":       "severity: 0 = benign, 1 = malignant",
    }
    return df, provenance


def train_lightgbm(df):
    """
    Train LightGBM on the UCI Mammographic Mass data — the model the app serves.
    4 features: age, shape, margin, density. No resampling.
    """
    print("\n── LightGBM on Mammographic Masses ─────────────────────")
    X = df[FEATURE_NAMES].values
    y = df["severity"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    n_pos = int(y_train.sum()); n_neg = int((y_train==0).sum())
    print(f"  Training split: {len(X_train)} samples (benign={n_neg}, malignant={n_pos})")

    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=5,
        random_state=42,
        verbose=-1,
    )
    # NOTE (Phase 1): early stopping monitors the test split, which leaks
    # test information into training. Kept as-is in Phase 0; see EVALUATION_PROTOCOL.
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(30, verbose=False),
                         lgb.log_evaluation(-1)])

    # Threshold via Youden's J — NOTE (Phase 1): chosen on the test split.
    optimal_threshold = _youden_threshold(model, X_test, y_test)
    print(f"  Optimal threshold (Youden's J): {optimal_threshold:.3f}")

    m = _metrics(model, X_test, y_test, threshold=optimal_threshold)
    m["n_train"], m["n_test"] = int(len(X_train)), int(len(X_test))
    print(f"  Accuracy: {m['accuracy']}%  |  AUC: {m['auc']}  |  Recall: {m['recall']}%")

    # SHAP TreeExplainer
    explainer = shap.TreeExplainer(model)

    joblib.dump(model,     LGBM_MODEL_PATH,   compress=3)
    joblib.dump(explainer, LGBM_SHAP_PATH,    compress=3)
    print(f"  Saved: {LGBM_MODEL_PATH.name} + {LGBM_SHAP_PATH.name}")
    return model, m, optimal_threshold


# ── Dataset 2: WDBC ──────────────────────────────────────────────
def load_wdbc():
    """
    Load Wisconsin Breast Cancer Diagnostic from sklearn.
    30 nuclear features from FNA images.
    """
    data = load_breast_cancer()
    X = data.data
    # sklearn: 0=malignant, 1=benign — flip for clarity
    y = 1 - data.target
    features = list(data.feature_names)
    return X, y, features


def train_xgboost(X, y):
    """
    Train XGBoost on WDBC — research benchmark (not used by the app).
    Uses SMOTE on the training split if imbalanced-learn is available.
    """
    print("\n── XGBoost on WDBC ─────────────────────────────────────")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    n_train = int(len(X_train))  # before any SMOTE resampling

    # SMOTE augmentation on training split only
    if _SMOTE_AVAILABLE:
        sm = SMOTE(random_state=42, k_neighbors=5)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print(f"  SMOTE applied: training set {len(X_train)} samples "
              f"(benign={int((y_train==0).sum())}, malignant={int((y_train==1).sum())})")

    model = xgb.XGBClassifier(
        n_estimators=400,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        gamma=0.05,
        reg_alpha=0.05,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    # Threshold via Youden's J — NOTE (Phase 1): chosen on the test split.
    optimal_threshold = _youden_threshold(model, X_test, y_test)
    print(f"  Optimal threshold (Youden's J): {optimal_threshold:.3f}")

    m = _metrics(model, X_test, y_test, threshold=optimal_threshold)
    m["n_train"], m["n_test"] = n_train, int(len(X_test))
    print(f"  Accuracy: {m['accuracy']}%  |  AUC: {m['auc']}  |  Recall: {m['recall']}%  "
          f"|  FP: {m['false_positives']}  |  FN: {m['false_negatives']}")

    explainer = shap.TreeExplainer(model)

    joblib.dump(model,     XGBOOST_MODEL_PATH,  compress=3)
    joblib.dump(explainer, XGBOOST_SHAP_PATH,   compress=3)
    print(f"  Saved: {XGBOOST_MODEL_PATH.name} + {XGBOOST_SHAP_PATH.name}")
    return model, m, optimal_threshold


# ── 5-Model CV comparison on WDBC ────────────────────────────────
def run_cv_comparison(X, y):
    """
    Stratified 10-fold CV comparison of 5 classifiers on WDBC.
    Returns results dict for model_metadata.json.
    """
    print("\n── 5-Model 10-Fold CV Comparison (WDBC) ───────────────")
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    models = {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            gamma=0.1, reg_alpha=0.1,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, verbosity=0),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            random_state=42, verbose=-1),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(probability=True, random_state=42)),
        ]),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
    }

    scoring = ["roc_auc", "accuracy", "recall"]
    cv_results = {}

    for name, clf in models.items():
        res = cross_validate(clf, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        cv_results[name] = {
            "auc":      round(res["test_roc_auc"].mean(), 4),
            "auc_std":  round(res["test_roc_auc"].std(),  4),
            "accuracy": round(res["test_accuracy"].mean() * 100, 2),
            "recall":   round(res["test_recall"].mean() * 100, 2),
        }
        print(f"  {name:<22} AUC: {cv_results[name]['auc']} ± {cv_results[name]['auc_std']}  "
              f"Acc: {cv_results[name]['accuracy']}%  Rec: {cv_results[name]['recall']}%")

    return cv_results


# ── Main ─────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("BreastCare AI — Model Training Pipeline")
    print("=" * 60)
    ensure_dirs()

    # Dataset 1: Mammographic
    print("\nDataset 1: UCI Mammographic Mass (id 161)")
    df, mammo_info = load_mammographic()
    lgbm_model, lgbm_metrics, lgbm_threshold = train_lightgbm(df)

    # Dataset 2: WDBC
    print("\nDataset 2: Wisconsin Breast Cancer Diagnostic (WDBC)")
    X, y, wdbc_features = load_wdbc()
    xgb_model, xgb_metrics, xgb_threshold = train_xgboost(X, y)

    # Save WDBC feature names for the XGBoost explain panel
    with open(WDBC_FEATURES_PATH, "w") as f:
        json.dump(wdbc_features, f)

    # 5-model CV comparison
    cv_results = run_cv_comparison(X, y)

    # Save metadata (metrics, Youden thresholds, provenance, protocol)
    metadata = {
        "lgbm_mammographic":      lgbm_metrics,
        "lgbm_optimal_threshold": lgbm_threshold,
        "xgboost_wdbc":           xgb_metrics,
        "xgb_optimal_threshold":  xgb_threshold,
        "cv_comparison":          cv_results,
        "augmentation":           "SMOTE" if _SMOTE_AVAILABLE else "none",
        "evaluation_protocol":    EVALUATION_PROTOCOL,
        "datasets": {
            "mammographic": mammo_info,
            "wdbc": {
                "name":       "Wisconsin Diagnostic Breast Cancer (WDBC)",
                "source":     "sklearn.datasets.load_breast_cancer (copy of UCI WDBC)",
                "rows":       int(len(y)),
                "benign":     int((y == 0).sum()),
                "malignant":  int((y == 1).sum()),
                "n_features": len(wdbc_features),
            },
        },
        "versions": {
            "python":       platform.python_version(),
            "lightgbm":     lgb.__version__,
            "xgboost":      xgb.__version__,
            "scikit-learn": sklearn.__version__,
            "shap":         shap.__version__,
        },
    }
    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    print("\n" + "=" * 60)
    print("Training complete. Models saved to models/")
    print("=" * 60)


if __name__ == "__main__":
    main()
