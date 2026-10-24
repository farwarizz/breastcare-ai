"""
src/ml/zoo.py — Research comparison models ("model zoo") for the risk estimate.

Five more classifier families, trained under exactly the protocol of the served
model (src/ml/train.py) and saved separately. They do not replace or change the
served model, its metadata or its selection rule: they were added after the
served model was chosen, so their test results are reported, never used to pick
a model (choosing by those numbers would tune on the test split).

Protocol (same data, split, folds, calibration and threshold rules as train.py)
 1. Verify and clean the UCI file and draw the same stratified split (seed 2026);
    the split ids are checked against models/splits.json and the result recorded.
 2. Per family, a small fixed grid scored by 5-fold CV log-loss on the fit split
    (the folds of src/ml/models.py). The grid point with the lowest mean CV
    log-loss is kept (ties: the first in grid order). No choice between families.
 3. Out-of-fold predictions of that configuration → calibrator
    (src/ml/calibration.select_calibrator) → Youden threshold and the highest
    threshold with out-of-fold sensitivity ≥ 95% → final model on the whole fit split.
 4. Test split, for reporting only: the metrics of src/ml/evaluation.summarize
    with the same 2,000 bootstrap resamples (seed 2026) as the served model, and
    a paired bootstrap AUROC difference against the served model.
 5. Save models/zoo/<family>.joblib and models/zoo_metadata.json.

Features are the served model's four inputs. Non-tree models use the logistic
regression's preprocessing (standardised age + one-hot shape, margin, density);
tree models use the raw codes. The conformal split is not used: these models
have no conformal sets and no explanations.

Run:  python -m src.ml.zoo
"""
from __future__ import annotations

import datetime
import functools
import json
import platform
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import FEATURE_NAMES, MODELS_DIR, SERVED_MODEL_PATH, SPLITS_PATH
from src.ml import data as D
from src.ml import models as M
from src.ml.calibration import Calibrator, select_calibrator
from src.ml.evaluation import (
    Bootstrap, auroc, brier, sensitivity_threshold, summarize, youden_threshold,
)
from src.ml.explain import model_frame

ZOO_DIR = MODELS_DIR / "zoo"
ZOO_METADATA_PATH = MODELS_DIR / "zoo_metadata.json"

# Same constants as src/ml/train.py (kept in step by tests/test_model_zoo.py).
HIGH_SENSITIVITY = 0.95
N_BOOTSTRAP = 2000
BOOT_SEED = 2026

FAMILIES = {   # report order
    "random_forest":     "Random forest",
    "gradient_boosting": "Gradient boosting (histogram)",
    "svm_rbf":           "Support vector machine (RBF)",
    "knn":               "k-nearest neighbours",
    "mlp":               "Neural network (MLP)",
}
FIXED = {
    "random_forest":     {"n_estimators": 300, "random_state": M.RANDOM_STATE, "n_jobs": 1},
    "gradient_boosting": {"learning_rate": 0.05, "min_samples_leaf": 20, "random_state": M.RANDOM_STATE},
    "svm_rbf":           {"kernel": "rbf", "gamma": "scale", "probability": True,
                          "random_state": M.RANDOM_STATE},
    "knn":               {},
    "mlp":               {"hidden_layer_sizes": (16,), "max_iter": 2000, "random_state": M.RANDOM_STATE},
}
GRIDS = {
    "random_forest":     [{"max_depth": d, "min_samples_leaf": m} for d in (3, 5, None) for m in (5, 20)],
    "gradient_boosting": [{"max_depth": d, "max_iter": n} for d in (2, 3) for n in (100, 200)],
    "svm_rbf":           [{"C": c} for c in (0.3, 1.0, 3.0)],
    "knn":               [{"n_neighbors": k} for k in (15, 31, 51)],
    "mlp":               [{"alpha": a} for a in (0.001, 0.01, 0.1)],
}
TREES = {"random_forest", "gradient_boosting"}
_CLASSES = {"random_forest": RandomForestClassifier, "gradient_boosting": HistGradientBoostingClassifier,
            "svm_rbf": SVC, "knn": KNeighborsClassifier, "mlp": MLPClassifier}
PREPROCESSING = {
    True:  "raw codes (age in years; shape, margin and density as integer codes)",
    False: "the logistic regression's preprocessing: standardised age + one-hot shape, margin, density",
}

RULE = ("Per family, the grid point with the lowest mean 5-fold cross-validated log-loss on the fit "
        "split (the folds of src/ml/models.py; ties go to the first point in grid order). Calibration "
        "method (cross-validated Brier score) and both thresholds come from that configuration's "
        "out-of-fold predictions on the fit split. Families are not compared with each other and "
        "none replaces the served model.")
CAVEAT = ("These models were added after the served model was chosen. Their test results are reported, "
          "not used to choose: picking a model by these numbers would tune on the test split. With 167 "
          "test cases, differences are mostly within the intervals.")


# ── Estimators ───────────────────────────────────────────────────
def _linear_preprocessing():
    """A fresh copy of the logistic regression's preprocessing step (src/ml/models.py)."""
    return clone(M.make_estimator("logistic_regression", {"C": 1.0}).named_steps["pre"])


def make_estimator(family: str, params: dict):
    est = _CLASSES[family](**FIXED[family], **params)
    if family in TREES:
        return est
    return Pipeline([("pre", _linear_preprocessing()), ("clf", est)])


def cross_validate(family: str, params: dict, X: pd.DataFrame, y: np.ndarray) -> tuple[dict, np.ndarray]:
    """5-fold CV log-loss (mean ± SE) and AUROC, as src/ml/models.cross_validate, plus the
    out-of-fold probabilities (no early stopping here, so they are the OOF predictions too)."""
    losses, aucs = [], []
    oof = np.zeros(len(y))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, va in M._folds(y):
            est = make_estimator(family, params).fit(X.iloc[tr], y[tr])
            p = est.predict_proba(X.iloc[va])[:, 1]
            oof[va] = p
            losses.append(log_loss(y[va], p, labels=[0, 1]))
            aucs.append(roc_auc_score(y[va], p))
    result = {
        "params": params,
        "cv_logloss": round(float(np.mean(losses)), 4),
        "cv_logloss_se": round(float(np.std(losses, ddof=1) / np.sqrt(len(losses))), 4),
        "cv_auroc": round(float(np.mean(aucs)), 4),
    }
    return result, oof


def pick(grid_results: list[dict]) -> int:
    """Index of the lowest mean CV log-loss; the first in grid order wins a tie."""
    best = 0
    for i, r in enumerate(grid_results):
        if r["cv_logloss"] < grid_results[best]["cv_logloss"]:
            best = i
    return best


def develop(family: str, Xf: pd.DataFrame, yf: np.ndarray) -> dict:
    """Grid → chosen configuration → calibrator and thresholds from OOF → final model."""
    runs = [cross_validate(family, params, Xf, yf) for params in GRIDS[family]]
    grid = [r for r, _ in runs]
    best = pick(grid)
    result, oof = runs[best]
    calibrator, cal_scores = select_calibrator(oof, yf)
    oof_cal = calibrator.transform(oof)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator = make_estimator(family, result["params"]).fit(Xf, yf)
    return {
        "family": family, "result": result, "grid": grid, "estimator": estimator,
        "calibrator": calibrator,
        "calibration_cv_brier": {k: round(v, 5) for k, v in cal_scores.items()},
        "threshold": youden_threshold(yf, oof_cal),
        "high_sensitivity_threshold": sensitivity_threshold(yf, oof_cal, HIGH_SENSITIVITY),
        "oof_auroc": round(auroc(yf, oof_cal), 4),
        "oof_brier": round(brier(yf, oof_cal), 4),
    }


def calibrated(estimator, calibration: dict, X: pd.DataFrame) -> np.ndarray:
    return Calibrator.from_dict(calibration).transform(estimator.predict_proba(model_frame(X))[:, 1])


# ── Metadata ─────────────────────────────────────────────────────
def _protocol(n: dict) -> str:
    total = sum(n.values())
    return (f"Same data, split and protocol as the served model (src/ml/train.py). The {total} cleaned UCI "
            f"Mammographic Mass rows are split with seed {D.SPLIT_SEED} into a fit split ({n['fit']} rows), "
            f"a conformal split ({n['conformal']}) and a test split ({n['test']}); the split ids are checked "
            f"against models/splits.json. For each family a small fixed grid is scored by {M.N_FOLDS}-fold "
            f"cross-validated log-loss on the fit split, using the same folds as the served model. The out-of-"
            f"fold predictions of the chosen configuration pick the calibration method (cross-validated Brier "
            f"score) and two thresholds: Youden's J, and the highest threshold that keeps out-of-fold "
            f"sensitivity at {HIGH_SENSITIVITY:.0%} or more. The final model is then trained on the whole fit "
            f"split. The test split is used only to report results, with 95% percentile bootstrap intervals "
            f"from the same {N_BOOTSTRAP:,} resamples (seed {BOOT_SEED}) as the served model, so each "
            f"difference from the served model is paired. The conformal split is not used: these models "
            f"have no conformal sets and no explanations.")


def model_entry(dev: dict, test_summary: dict, auroc_diff: dict, n_fit: int) -> dict:
    family = dev["family"]
    r = dev["result"]
    fixed = {k: (list(v) if isinstance(v, tuple) else v) for k, v in FIXED[family].items()}
    return {
        "family": family,
        "label": FAMILIES[family],
        "params": r["params"],
        "fixed_params": fixed,
        "preprocessing": PREPROCESSING[family in TREES],
        "trained_on": f"fit split (n={n_fit})",
        "cv": {k: r[k] for k in ("cv_logloss", "cv_logloss_se", "cv_auroc")},
        "cv_grid": dev["grid"],
        "oof": {"auroc": dev["oof_auroc"], "brier": dev["oof_brier"]},
        "calibration": {**dev["calibrator"].to_dict(),
                        "cv_brier_by_method": dev["calibration_cv_brier"],
                        "fitted_on": "out-of-fold predictions, fit split"},
        "threshold": {"value": round(dev["threshold"], 4),
                      "method": "Youden's J on out-of-fold calibrated predictions (fit split)"},
        "high_sensitivity_threshold": {
            "value": round(dev["high_sensitivity_threshold"], 4),
            "method": f"highest threshold with out-of-fold sensitivity ≥ {HIGH_SENSITIVITY:.0%}"},
        "test": test_summary,
        "vs_served": {"auroc_difference": auroc_diff,
                      "definition": "test AUROC of this model minus the served model's, paired bootstrap "
                                    "(same resamples)"},
    }


def save_bundle(dev: dict, entry: dict) -> Path:
    path = ZOO_DIR / f"{dev['family']}.joblib"
    joblib.dump({"family": dev["family"], "label": FAMILIES[dev["family"]], "estimator": dev["estimator"],
                 "params": dev["result"]["params"], "calibration": dev["calibrator"].to_dict(),
                 "threshold": entry["threshold"]["value"],
                 "high_sensitivity_threshold": entry["high_sensitivity_threshold"]["value"],
                 "feature_names": list(FEATURE_NAMES)}, path, compress=3)
    return path


# ── Training run ─────────────────────────────────────────────────
def main() -> None:
    print("=" * 64)
    print("BreastCare AI - research comparison models (model zoo)")
    print("=" * 64)

    print("\n1. Data and split")
    # The steps of D.load_mammographic without rewriting data/processed/cleaned.csv.
    D.verify_raw_file()
    df = D.clean_mammographic(D.read_raw_mammographic())
    splits = D.split_rows(df)
    saved = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    matches = {k: sorted(int(i) for i in v) for k, v in splits.items()} == saved
    print(f"  split ids match models/splits.json: {matches}")
    X, y = df[FEATURE_NAMES], df["severity"]
    Xf, yf = X.loc[splits["fit"]], y.loc[splits["fit"]].to_numpy()
    Xt, yt = X.loc[splits["test"]], y.loc[splits["test"]].to_numpy()
    n = {k: len(v) for k, v in splits.items()}
    print(f"  fit {n['fit']} - conformal {n['conformal']} - test {n['test']}")

    print("\n2. Served model on the test split (read-only reference)")
    md = json.loads((MODELS_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    served_meta = md["served_model"]
    served = joblib.load(SERVED_MODEL_PATH)
    ps = calibrated(served["estimator"], served_meta["calibration"], Xt)
    served_auroc = round(auroc(yt, ps), 4)
    served_ok = served_auroc == served_meta["test"]["auroc"]["value"]
    print(f"  {served_meta['label']}: recomputed test AUROC {served_auroc} "
          f"(matches model_metadata.json: {served_ok})")

    print("\n3. Families (5-fold CV on the fit split, OOF calibration and thresholds)")
    boot = Bootstrap(len(yt), N_BOOTSTRAP, BOOT_SEED)
    entries = []
    ZOO_DIR.mkdir(parents=True, exist_ok=True)
    for family in FAMILIES:
        dev = develop(family, Xf, yf)
        r = dev["result"]
        print(f"  {FAMILIES[family]:<32} {r['params']}  CV log-loss {r['cv_logloss']:.4f} "
              f"+/- {r['cv_logloss_se']:.4f}  calibration {dev['calibrator'].method}  "
              f"threshold {dev['threshold']:.3f}")
        p = calibrated(dev["estimator"], dev["calibrator"].to_dict(), Xt)
        test_summary = summarize(yt, p, dev["threshold"], boot)
        diff = boot.paired(lambda i, p=p: auroc(yt[i], p[i]), lambda i: auroc(yt[i], ps[i]))
        entry = model_entry(dev, test_summary, diff, n["fit"])
        path = save_bundle(dev, entry)
        print(f"  {'':<32} test AUROC {test_summary['auroc']['value']:.4f} {test_summary['auroc']['ci']}"
              f"  minus served {diff['value']:+.4f} {diff['ci']}  ({path.stat().st_size:,} bytes)")
        entries.append(entry)

    metadata = {
        "schema_version": 1,
        "created_at": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat(),
        "protocol": _protocol(n),
        "rule": RULE,
        "split_check": {"matches_saved_splits": matches, "saved_splits": "models/splits.json",
                        "split_seed": D.SPLIT_SEED, "n": n},
        "caveat": CAVEAT,
        "served_reference": {"family": served_meta["family"], "label": served_meta["label"],
                             "test_auroc_recomputed": served_auroc,
                             "matches_model_metadata": served_ok,
                             "note": "The served model and models/model_metadata.json are unchanged; its "
                                     "test predictions are recomputed only for the paired comparisons."},
        "cv": {"folds": M.N_FOLDS, "shuffle": True, "random_state": M.RANDOM_STATE},
        "bootstrap": {"resamples": N_BOOTSTRAP, "seed": BOOT_SEED, "interval": "95% percentile"},
        "models": entries,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__},
    }
    with open(ZOO_METADATA_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print(f"\n4. Saved {len(entries)} bundles to {ZOO_DIR.relative_to(MODELS_DIR.parent)} "
          f"and {ZOO_METADATA_PATH.relative_to(MODELS_DIR.parent)}")


# ── Serving (research comparisons; never the assessment) ─────────
@functools.lru_cache(maxsize=1)
def load_zoo() -> tuple | None:
    """(metadata, [(model entry, bundle), ...]) in the metadata's order, or None when
    models/zoo_metadata.json does not exist. A listed bundle that is missing or does not
    match its entry is an error (no silent substitute)."""
    if not ZOO_METADATA_PATH.exists():
        return None
    meta = json.loads(ZOO_METADATA_PATH.read_text(encoding="utf-8"))
    pairs = []
    for entry in meta["models"]:
        bundle = joblib.load(ZOO_DIR / f"{entry['family']}.joblib")
        if (bundle["family"] != entry["family"] or bundle["feature_names"] != list(FEATURE_NAMES)
                or bundle["threshold"] != entry["threshold"]["value"]):
            raise ValueError(f"models/zoo/{entry['family']}.joblib does not match models/zoo_metadata.json")
        pairs.append((entry, bundle))
    return meta, tuple(pairs)


def predict_case(features: dict) -> list[dict]:
    """Every zoo model's calibrated probability for one case, in the metadata's order.
    predicted_class uses the served model's convention (probability ≥ threshold → Malignant)."""
    loaded = load_zoo()
    if loaded is None:
        return []
    X = pd.DataFrame([features])
    out = []
    for entry, bundle in loaded[1]:
        p = float(calibrated(bundle["estimator"], bundle["calibration"], X)[0])
        t = bundle["threshold"]
        out.append({"family": entry["family"], "label": entry["label"], "probability": p,
                    "threshold": t, "predicted_class": "Malignant" if p >= t else "Benign",
                    "test_auroc": entry.get("test", {}).get("auroc")})
    return out


if __name__ == "__main__":
    main()
