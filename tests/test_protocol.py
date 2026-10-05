"""The evaluation protocol as recorded in the artifacts: no leakage, rule-based selection."""
import json

import joblib

from src import config
from src.ml import data, models

MD = json.loads(config.MODEL_METADATA_PATH.read_text(encoding="utf-8"))
SPLITS = json.loads(config.SPLITS_PATH.read_text(encoding="utf-8"))


def test_splits_are_disjoint_complete_and_reproducible():
    fit, conf, test = (set(SPLITS[k]) for k in ("fit", "conformal", "test"))
    assert not (fit & conf) and not (fit & test) and not (conf & test)
    df = data.clean_mammographic(data.read_raw_mammographic())
    assert fit | conf | test == set(df.index)
    again = data.split_rows(df)
    assert {k: sorted(map(int, v)) for k, v in again.items()} == SPLITS
    assert MD["protocol"]["n"] == {k: len(v) for k, v in SPLITS.items()}


def test_tuning_used_only_development_data():
    for key in ("served_model", "challenger_model"):
        m = MD[key]
        assert "out-of-fold" in m["calibration"]["fitted_on"]
        assert "out-of-fold" in m["threshold"]["method"]
        assert m["trained_on"] == f"fit split (n={len(SPLITS['fit'])})"
    conf = MD["served_model"]["conformal"]
    assert conf["fitted_on"] == f"conformal split (n={len(SPLITS['conformal'])})"
    assert sum(conf["n_calibration"].values()) == len(SPLITS["conformal"])
    assert MD["served_model"]["test"]["n"] == len(SPLITS["test"])


def test_served_model_follows_the_selection_rule():
    sel = models.select(MD["selection"]["cv_results"])
    assert sel["selected"] == MD["selection"]["selected"] == MD["served_model"]["family"]
    assert MD["challenger_model"]["family"] == MD["selection"]["challenger"]
    bundle = joblib.load(config.SERVED_MODEL_PATH)
    assert bundle["family"] == MD["served_model"]["family"]
    assert joblib.load(config.CHALLENGER_MODEL_PATH)["family"] == MD["challenger_model"]["family"]


def test_every_reported_metric_has_a_confidence_interval():
    test = MD["served_model"]["test"]
    for key in ("auroc", "auprc", "brier", "ece", "calibration_intercept", "calibration_slope",
                "sensitivity", "specificity", "ppv", "npv"):
        lo, hi = test[key]["ci"]
        assert lo <= test[key]["value"] <= hi, key


def test_runtime_pins_match_the_versions_that_trained_the_artifacts():
    """Pickled pipelines must be loaded by the library versions that wrote them."""
    reqs = (config.ROOT / "requirements.txt").read_text(encoding="utf-8")
    for package in ("scikit-learn", "lightgbm"):
        pin = next(line.split("#")[0].strip() for line in reqs.splitlines()
                   if line.startswith(package + "=="))
        assert pin == f"{package}=={MD['versions'][package]}", pin


def test_birads_is_only_a_reference():
    assert "birads" not in MD["datasets"]["mammographic"]["features"]
    assert MD["comparisons"]["birads_reference"]["n"] <= len(SPLITS["test"])
