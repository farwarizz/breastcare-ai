"""
src/ml/train.py — Training and evaluation pipeline (UCI Mammographic Mass).

Protocol. Every decision is made without the test split.
 1. Verify and clean the UCI file (src/ml/data.py).
 2. Stratified split (seed 2026): fit 60% · conformal 20% · test 20%.
 3. Model selection by 5-fold CV log-loss on the fit split, using a fixed rule
    (src/ml/models.py).
 4. For every candidate family: out-of-fold predictions on the fit split →
    choose and fit a calibrator → choose the decision threshold (Youden's J)
    and a high-sensitivity threshold (sensitivity ≥ 95%) → train the final
    model on the whole fit split.
 5. Mondrian split conformal on the conformal split (served model only).
 6. Test split, used once: metrics with 95% bootstrap CIs, a paired comparison
    with the challenger, the radiologists' BI-RADS assessment as a reference,
    a monotonic-constraint ablation, age subgroups and decision curves.
 7. Save model bundles, metadata, split ids and figures; log to MLflow if installed.

Run:  python -m src.ml.train
"""
from __future__ import annotations

import json
import platform
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import shap
import sklearn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    FEATURE_NAMES, MODELS_DIR, MODEL_METADATA_PATH, SERVED_MODEL_PATH,
    CHALLENGER_MODEL_PATH, SPLITS_PATH, FIGURES_DIR,
)
from src.ml import data as D
from src.ml import models as M
from src.ml.calibration import select_calibrator
from src.ml.conformal import fit_mondrian
from src.ml.evaluation import (
    Bootstrap, auroc, brier, net_benefit, net_benefit_treat_all, sensitivity_threshold,
    subgroup_summary, summarize, threshold_metrics, youden_threshold,
)
from src.ml.explain import linear_background

ALPHA = 0.10                     # conformal miscoverage → 90% sets
HIGH_SENSITIVITY = 0.95
N_BOOTSTRAP = 2000
BOOT_SEED = 2026
DCA_TABLE = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
AGE_SPLIT = 50

PROTOCOL = (
    "Stratified split of the 831 cleaned rows (seed 2026): fit 60%, conformal 20%, "
    "test 20%. Model selection by 5-fold CV log-loss on the fit split with a "
    "one-standard-error rule favouring simpler models. Calibration method and decision "
    "thresholds are chosen on out-of-fold predictions of the fit split; final models "
    "are trained on the whole fit split. Conformal cut-offs use only the conformal "
    "split. The test split is used once, for the reported metrics (95% percentile "
    "bootstrap CIs, 2,000 resamples, paired across models)."
)


def develop(result: dict, Xf, yf) -> dict:
    """OOF predictions → calibrator → thresholds → final model on the whole fit split."""
    oof = M.out_of_fold(result, Xf, yf)
    calibrator, cal_scores = select_calibrator(oof, yf)
    oof_cal = calibrator.transform(oof)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator = M.build(result).fit(Xf, yf)
    return {
        "result": result,
        "estimator": estimator,
        "calibrator": calibrator,
        "calibration_cv_brier": {k: round(v, 5) for k, v in cal_scores.items()},
        "threshold": youden_threshold(yf, oof_cal),
        "high_sensitivity_threshold": sensitivity_threshold(yf, oof_cal, HIGH_SENSITIVITY),
        "oof_auroc": round(auroc(yf, oof_cal), 4),
        "oof_brier": round(brier(yf, oof_cal), 4),
    }


def calibrated(dev: dict, X) -> np.ndarray:
    return dev["calibrator"].transform(dev["estimator"].predict_proba(X)[:, 1])


def model_entry(family: str, dev: dict, test_summary: dict, n_fit: int) -> dict:
    r = dev["result"]
    entry = {
        "family": family,
        "label": M.FAMILIES[family],
        "params": r["params"],
        "trained_on": f"fit split (n={n_fit})",
        "cv": {k: r[k] for k in ("cv_logloss", "cv_logloss_se", "cv_auroc")},
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
    }
    if "n_estimators" in r:
        entry["params"] = {**r["params"], "n_estimators": r["n_estimators"]}
    return entry


def save_bundle(path: Path, family: str, dev: dict, Xf) -> None:
    bundle = {"family": family, "label": M.FAMILIES[family],
              "estimator": dev["estimator"], "feature_names": list(FEATURE_NAMES)}
    if family == "logistic_regression":
        bundle.update(linear_background(dev["estimator"], Xf))
    joblib.dump(bundle, path, compress=3)


def log_mlflow(metadata: dict) -> str:
    """Log the run to a local MLflow store (mlflow.db + mlartifacts/). Optional."""
    try:
        import mlflow
    except ImportError:
        return "skipped (mlflow not installed)"
    try:
        return _log_mlflow(mlflow, metadata)
    except Exception as exc:          # tracking must never break training
        return f"FAILED ({type(exc).__name__}: {exc})"


def _log_mlflow(mlflow, metadata: dict) -> str:
    root = MODELS_DIR.parent
    mlflow.set_tracking_uri(f"sqlite:///{(root / 'mlflow.db').as_posix()}")
    name = "breastcare-mammographic"
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(root / "mlartifacts").resolve().as_uri())
    mlflow.set_experiment(name)
    served = metadata["served_model"]
    with mlflow.start_run(run_name=served["family"]) as run:
        mlflow.log_params({"served_family": served["family"], **{f"served_{k}": v for k, v in served["params"].items()},
                           "calibration": served["calibration"]["method"],
                           "threshold": served["threshold"]["value"], "alpha": ALPHA})
        for r in metadata["selection"]["cv_results"]:
            tag = f"{r['family']}_{'_'.join(f'{k}{v}' for k, v in r['params'].items())}"
            mlflow.log_metric(f"cv_logloss.{tag}", r["cv_logloss"])
        for key in ("auroc", "auprc", "brier", "ece", "calibration_slope", "calibration_intercept",
                    "sensitivity", "specificity"):
            mlflow.log_metric(f"test_{key}", served["test"][key]["value"])
        mlflow.log_metric("test_conformal_coverage", served["test"]["conformal"]["coverage"]["value"])
        mlflow.log_artifact(str(MODEL_METADATA_PATH))
        mlflow.log_artifacts(str(FIGURES_DIR), artifact_path="figures")
        return f"logged run {run.info.run_id}"


def main() -> None:
    print("=" * 64)
    print("BreastCare AI — training and evaluation")
    print("=" * 64)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n1. Data")
    df, provenance = D.load_mammographic()
    raw = D.read_raw_mammographic()
    splits = D.split_rows(df)
    X, y = df[FEATURE_NAMES], df["severity"]
    Xf, yf = X.loc[splits["fit"]], y.loc[splits["fit"]].to_numpy()
    Xc, yc = X.loc[splits["conformal"]], y.loc[splits["conformal"]].to_numpy()
    Xt, yt = X.loc[splits["test"]], y.loc[splits["test"]].to_numpy()
    print(f"  fit {len(yf)} · conformal {len(yc)} · test {len(yt)}")

    print("\n2. Model selection (5-fold CV on the fit split)")
    cv_results = [M.cross_validate(fam, params, Xf, yf) for fam in M.FAMILIES for params in M.GRIDS[fam]]
    sel = M.select(cv_results)
    for fam, r in sel["best_per_family"].items():
        print(f"  {M.FAMILIES[fam]:<40} log-loss {r['cv_logloss']:.4f} ± {r['cv_logloss_se']:.4f}"
              f"  AUROC {r['cv_auroc']:.4f}  {r['params']}")
    served = sel["selected"]
    challenger = min((f for f in sel["best_per_family"] if f != served),
                     key=lambda f: sel["best_per_family"][f]["cv_logloss"])
    print(f"  → served: {M.FAMILIES[served]} · challenger: {M.FAMILIES[challenger]}")

    print("\n3. Calibration and thresholds (out-of-fold, fit split)")
    devs = {fam: develop(r, Xf, yf) for fam, r in sel["best_per_family"].items()}
    for fam, d in devs.items():
        print(f"  {M.FAMILIES[fam]:<40} calibration {d['calibrator'].method:<8} "
              f"threshold {d['threshold']:.3f}  high-sens {d['high_sensitivity_threshold']:.3f}")

    print("\n4. Conformal calibration (conformal split)")
    conformal = fit_mondrian(calibrated(devs[served], Xc), yc, ALPHA)
    print(f"  Benign in set if p ≤ {conformal.benign_if_p_le:.3f}; "
          f"Malignant in set if p ≥ {conformal.malignant_if_p_ge:.3f}")

    print("\n5. Test evaluation (used once)")
    boot = Bootstrap(len(yt), N_BOOTSTRAP, BOOT_SEED)
    test_p = {fam: calibrated(d, Xt) for fam, d in devs.items()}
    summaries = {fam: summarize(yt, test_p[fam], devs[fam]["threshold"], boot,
                                conformal if fam == served else None) for fam in devs}
    ps, pc = test_p[served], test_p[challenger]
    t_hs = devs[served]["high_sensitivity_threshold"]
    high_sens = {"threshold": round(t_hs, 4),
                 **{k: boot.ci(lambda i, k=k: threshold_metrics(yt[i], ps[i], t_hs)[k])
                    for k in ("sensitivity", "specificity", "npv")}}

    birads_all = raw.loc[splits["test"], "birads"]
    valid = birads_all.between(1, 5).to_numpy()
    yb, sb, pb = yt[valid], birads_all.to_numpy()[valid], ps[valid]
    bboot = Bootstrap(int(valid.sum()), N_BOOTSTRAP, BOOT_SEED)
    birads_ref = {
        "n": int(valid.sum()), "n_excluded": int((~valid).sum()),
        "excluded_reason": "BI-RADS missing or outside the documented 1–5 range (0, 6, 55)",
        "auroc_birads": bboot.ci(lambda i: auroc(yb[i], sb[i])),
        "auroc_model_same_rows": bboot.ci(lambda i: auroc(yb[i], pb[i])),
        "auroc_difference_model_minus_birads": bboot.paired(lambda i: auroc(yb[i], pb[i]),
                                                            lambda i: auroc(yb[i], sb[i])),
    }
    comparisons = {
        "auroc_difference_served_minus_challenger": boot.paired(
            lambda i: auroc(yt[i], ps[i]), lambda i: auroc(yt[i], pc[i])),
        "brier_difference_served_minus_challenger": boot.paired(
            lambda i: brier(yt[i], ps[i]), lambda i: brier(yt[i], pc[i])),
        "birads_reference": birads_ref,
        "monotone_ablation": {
            "auroc_difference_monotone_minus_unconstrained": boot.paired(
                lambda i: auroc(yt[i], test_p["lightgbm_monotone"][i]),
                lambda i: auroc(yt[i], test_p["lightgbm"][i])),
            "brier_difference_monotone_minus_unconstrained": boot.paired(
                lambda i: brier(yt[i], test_p["lightgbm_monotone"][i]),
                lambda i: brier(yt[i], test_p["lightgbm"][i])),
        },
    }
    age_groups = np.where(Xt["age"].to_numpy() < AGE_SPLIT, f"age < {AGE_SPLIT}", f"age ≥ {AGE_SPLIT}")
    subgroups = subgroup_summary(yt, ps, age_groups, devs[served]["threshold"], N_BOOTSTRAP, BOOT_SEED)
    thresholds = np.round(np.arange(0.01, 0.61, 0.01), 2)
    nb = {fam: net_benefit(yt, test_p[fam], thresholds) for fam in (served, challenger)}
    nb_all = net_benefit_treat_all(yt, thresholds)
    pick = lambda arr, t: round(float(arr[int(np.argmin(np.abs(thresholds - t)))]), 4)
    dca = {"thresholds": DCA_TABLE,
           "net_benefit": {M.FAMILIES[f]: [pick(nb[f], t) for t in DCA_TABLE] for f in nb},
           "treat_all": [pick(nb_all, t) for t in DCA_TABLE], "treat_none": [0.0] * len(DCA_TABLE)}
    s = summaries[served]
    print(f"  served AUROC {s['auroc']['value']:.3f} {s['auroc']['ci']} · Brier {s['brier']['value']:.3f}"
          f" · conformal coverage {s['conformal']['coverage']['value']:.3f}")
    print(f"  BI-RADS AUROC {birads_ref['auroc_birads']['value']:.3f} on {birads_ref['n']} rows")

    print("\n6. Figures")
    from src.ml.figures import render_all
    figure_models = [{"label": M.FAMILIES[f], "p": test_p[f], "auroc": summaries[f]["auroc"],
                      "auprc": summaries[f]["auprc"], "slope": summaries[f]["calibration_slope"],
                      "intercept": summaries[f]["calibration_intercept"]} for f in (served, challenger)]
    figures = render_all(FIGURES_DIR, y=yt, models=figure_models,
                         birads={"y": yb, "score": sb, "auroc": birads_ref["auroc_birads"]},
                         dca_thresholds=thresholds,
                         dca_curves=[{"label": M.FAMILIES[f], "nb": nb[f]} for f in (served, challenger)],
                         dca_treat_all=nb_all, conformal=conformal)
    print(f"  wrote {len(figures)} files to {FIGURES_DIR}")

    print("\n7. Save")
    save_bundle(SERVED_MODEL_PATH, served, devs[served], Xf)
    save_bundle(CHALLENGER_MODEL_PATH, challenger, devs[challenger], Xf)
    with open(SPLITS_PATH, "w", encoding="utf-8") as f:
        json.dump({k: [int(i) for i in v] for k, v in splits.items()}, f)
        f.write("\n")

    served_entry = model_entry(served, devs[served], summaries[served], len(yf))
    served_entry["conformal"] = {**conformal.to_dict(), "fitted_on": f"conformal split (n={len(yc)})"}
    served_entry["test"]["high_sensitivity_operating_point"] = high_sens
    metadata = {
        "schema_version": 2,
        "protocol": {
            "description": PROTOCOL,
            "split_seed": D.SPLIT_SEED,
            "n": {"fit": len(yf), "conformal": len(yc), "test": len(yt)},
            "bootstrap": {"resamples": N_BOOTSTRAP, "seed": BOOT_SEED, "interval": "95% percentile"},
            "risk_categories": "Low < 0.35 ≤ Moderate < 0.65 ≤ Elevated (calibrated probability)",
        },
        "datasets": {"mammographic": provenance},
        "selection": {
            "rule": sel["rule"], "selected": served, "challenger": challenger,
            "best_overall_by_cv": sel["best_overall"], "one_se_bound": sel["one_se_bound"],
            "cv_results": cv_results,
        },
        "served_model": served_entry,
        "challenger_model": model_entry(challenger, devs[challenger], summaries[challenger], len(yf)),
        "ablation_models": {f: model_entry(f, devs[f], summaries[f], len(yf))
                            for f in devs if f not in (served, challenger)},
        "comparisons": comparisons,
        "subgroups": {"by": f"age, split at {AGE_SPLIT}", "groups": subgroups},
        "decision_curve": dca,
        "figures": sorted(figures),
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "lightgbm": lgb.__version__, "shap": shap.__version__,
                     "numpy": np.__version__},
    }
    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"  saved {SERVED_MODEL_PATH.name}, {CHALLENGER_MODEL_PATH.name}, "
          f"{MODEL_METADATA_PATH.name}, {SPLITS_PATH.name}")
    print(f"  MLflow: {log_mlflow(metadata)}")
    print("\nDone. Next: python scripts/update_readme_metrics.py")


if __name__ == "__main__":
    main()
