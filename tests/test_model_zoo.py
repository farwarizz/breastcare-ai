"""Research comparison models (src/ml/zoo.py, backend 2.5.0): trained under the served model's
protocol, saved separately, reported but never used to choose; /api/assess-models shows every model
on one case with the served model first; the served model and the frozen files are unchanged."""
import hashlib
import json

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

import server
from src import config
from src.ml import calibration, data, models, train, zoo
from src.ml.evaluation import auroc

client = TestClient(server.app)
CASE = {"age": 66, "shape": 4, "margin": 5, "density": 3}
MD = json.loads(config.MODEL_METADATA_PATH.read_text(encoding="utf-8"))
ZOO = json.loads(zoo.ZOO_METADATA_PATH.read_text(encoding="utf-8"))
SPLITS = json.loads(config.SPLITS_PATH.read_text(encoding="utf-8"))
SERVED_ONLY_TEST_KEYS = {"conformal", "high_sensitivity_operating_point"}


def _test_split():
    df = data.clean_mammographic(data.read_raw_mammographic())
    Xt = df.loc[SPLITS["test"], config.FEATURE_NAMES]
    return Xt, df.loc[SPLITS["test"], "severity"].to_numpy()


@pytest.fixture
def no_zoo(monkeypatch, tmp_path):
    monkeypatch.setattr(zoo, "ZOO_METADATA_PATH", tmp_path / "zoo_metadata.json")
    zoo.load_zoo.cache_clear()
    yield
    zoo.load_zoo.cache_clear()


# ── Artifacts ─────────────────────────────────────────────────────────────────
def test_zoo_metadata_has_the_five_families_under_the_served_protocol():
    assert ZOO["schema_version"] == 1 and ZOO["created_at"]
    assert ZOO["split_check"]["matches_saved_splits"] is True
    assert ZOO["split_check"]["n"] == {k: len(v) for k, v in SPLITS.items()}
    assert ZOO["caveat"].startswith("These models were added after the served model was chosen.")
    assert ZOO["protocol"] and ZOO["rule"]
    assert set(ZOO["versions"]) >= {"python", "scikit-learn"}
    assert ZOO["served_reference"]["family"] == MD["served_model"]["family"]
    assert ZOO["served_reference"]["matches_model_metadata"] is True
    assert [m["family"] for m in ZOO["models"]] == list(zoo.FAMILIES) == [
        "random_forest", "gradient_boosting", "svm_rbf", "knn", "mlp"]
    served_test_keys = set(MD["served_model"]["test"]) - SERVED_ONLY_TEST_KEYS
    for m in ZOO["models"]:
        fam = m["family"]
        assert m["label"] == zoo.FAMILIES[fam]
        assert m["trained_on"] == f"fit split (n={len(SPLITS['fit'])})"
        for key in ("threshold", "high_sensitivity_threshold"):
            assert 0 < m[key]["value"] < 1, (fam, key)
            assert "out-of-fold" in m[key]["method"]
        assert m["calibration"]["method"] in calibration.METHODS
        assert "out-of-fold" in m["calibration"]["fitted_on"]
        # the chosen grid point is the lowest mean CV log-loss (first in grid order on a tie)
        assert [g["params"] for g in m["cv_grid"]] == json.loads(json.dumps(zoo.GRIDS[fam]))
        assert m["params"] == m["cv_grid"][zoo.pick(m["cv_grid"])]["params"]
        assert m["cv"] == {k: m["cv_grid"][zoo.pick(m["cv_grid"])][k]
                           for k in ("cv_logloss", "cv_logloss_se", "cv_auroc")}
        # test summary: the same shape as served_model.test (without the served-only parts)
        assert set(m["test"]) == served_test_keys, fam
        assert m["test"]["n"] == len(SPLITS["test"]) == MD["served_model"]["test"]["n"]
        assert m["test"]["threshold"] == m["threshold"]["value"]
        for key in ("auroc", "auprc", "brier", "ece", "calibration_intercept", "calibration_slope",
                    "sensitivity", "specificity", "ppv", "npv", "accuracy"):
            assert set(m["test"][key]) == {"value", "ci"}, (fam, key)
            lo, hi = m["test"][key]["ci"]
            assert lo <= hi, (fam, key)
        # A percentile interval need not contain a biased statistic's point estimate (ECE is
        # biased upward in resamples), so containment is checked only for AUROC.
        lo, hi = m["test"]["auroc"]["ci"]
        assert lo <= m["test"]["auroc"]["value"] <= hi, fam
        diff = m["vs_served"]["auroc_difference"]
        assert set(diff) == {"value", "ci"} and diff["ci"][0] <= diff["value"] <= diff["ci"][1]


def test_zoo_uses_the_training_constants_and_folds():
    assert (zoo.HIGH_SENSITIVITY, zoo.N_BOOTSTRAP, zoo.BOOT_SEED) == \
        (train.HIGH_SENSITIVITY, train.N_BOOTSTRAP, train.BOOT_SEED)
    assert ZOO["bootstrap"]["resamples"] == MD["protocol"]["bootstrap"]["resamples"]
    assert ZOO["bootstrap"]["seed"] == MD["protocol"]["bootstrap"]["seed"]
    assert ZOO["cv"] == {"folds": models.N_FOLDS, "shuffle": True, "random_state": models.RANDOM_STATE}
    # non-tree models use exactly the logistic regression's preprocessing
    pre = zoo.make_estimator("knn", {"n_neighbors": 15}).named_steps["pre"]
    lr_pre = models.make_estimator("logistic_regression", {"C": 1.0}).named_steps["pre"]
    assert repr(pre) == repr(lr_pre)


def test_zoo_cross_validation_is_reproducible():
    df = data.clean_mammographic(data.read_raw_mammographic())
    Xf = df.loc[SPLITS["fit"], config.FEATURE_NAMES]
    yf = df.loc[SPLITS["fit"], "severity"].to_numpy()
    knn = next(m for m in ZOO["models"] if m["family"] == "knn")
    again = [zoo.cross_validate("knn", params, Xf, yf)[0] for params in zoo.GRIDS["knn"]]
    assert again == knn["cv_grid"]


def test_zoo_bundles_load_and_predict_deterministically():
    Xt, yt = _test_split()
    for m in ZOO["models"]:
        path = zoo.ZOO_DIR / f"{m['family']}.joblib"
        assert path.stat().st_size < 1_000_000, path.name
        first, second = joblib.load(path), joblib.load(path)
        assert first["family"] == m["family"] and first["feature_names"] == config.FEATURE_NAMES
        assert first["threshold"] == m["threshold"]["value"]
        assert first["high_sensitivity_threshold"] == m["high_sensitivity_threshold"]["value"]
        assert first["calibration"] == {k: m["calibration"][k] for k in ("method", "params")}
        p1 = zoo.calibrated(first["estimator"], first["calibration"], Xt)
        p2 = zoo.calibrated(second["estimator"], second["calibration"], Xt)
        assert np.array_equal(p1, p2)
        assert ((p1 >= 0) & (p1 <= 1)).all()
        # the saved model reproduces the reported test AUROC
        assert round(auroc(yt, p1), 4) == m["test"]["auroc"]["value"], m["family"]


def test_served_model_is_unchanged_and_frozen_files_match_their_hashes():
    freeze = json.loads((config.ROOT / "evals" / "heldout_v3_freeze.json").read_text(encoding="utf-8"))
    changed = [rel for rel, digest in freeze["sha256"].items()
               if hashlib.sha256((config.ROOT / rel).read_bytes()).hexdigest() != digest]
    assert changed == []
    assert joblib.load(config.SERVED_MODEL_PATH)["family"] == MD["served_model"]["family"]
    assert MD["served_model"]["family"] not in zoo.FAMILIES


# ── API ───────────────────────────────────────────────────────────────────────
def test_model_info_serves_the_zoo_beside_the_unchanged_metadata():
    body = client.get("/api/model-info").json()
    assert body["zoo"] == ZOO
    assert body["metadata"] == MD
    assert set(body) == {"metadata", "knowledge_base", "llm", "evaluations", "zoo"}


def test_model_info_zoo_is_null_without_the_zoo(no_zoo):
    body = client.get("/api/model-info").json()
    assert body["zoo"] is None and body["metadata"] == MD


def test_health_lists_assess_models_and_the_new_version():
    body = client.get("/health").json()
    assert body["version"] == server.API_VERSION == "2.5.0"
    assert client.get("/openapi.json").json()["info"]["version"] == "2.5.0"
    assert "/api/assess-models" in body["endpoints"]["extended"]


def test_assess_models_matches_assess_and_compare_models():
    r = client.post("/api/assess-models", json=CASE)
    assert r.status_code == 200
    body = r.json()
    assert body["inputs"] == CASE
    assert body["note"] == server.ASSESS_MODELS_NOTE
    roles = [m["role"] for m in body["models"]]
    assert roles == ["served", "challenger"] + ["research"] * len(ZOO["models"])
    assert [m["family"] for m in body["models"]] == \
        [MD["served_model"]["family"], MD["challenger_model"]["family"]] + [m["family"] for m in ZOO["models"]]

    served, challenger = body["models"][0], body["models"][1]
    assess = client.post("/api/assess", json=CASE).json()["prediction"]
    assert served["probability"] == assess["probability"]
    assert served["threshold"] == assess["decision_threshold"]
    assert served["predicted_class"] == assess["predicted_class"]
    assert served["label"] == assess["model"] == MD["served_model"]["label"]
    assert served["test_auroc"] == MD["served_model"]["test"]["auroc"]

    compare = client.post("/api/compare-models", json=CASE).json()["models"]
    assert round(served["probability"], 4) == compare["served"]["probability"]
    assert round(challenger["probability"], 4) == compare["challenger"]["probability"]
    assert challenger["predicted_class"] == compare["challenger"]["predicted_class"]
    assert challenger["test_auroc"] == compare["challenger"]["test_auroc"]
    assert challenger["threshold"] == MD["challenger_model"]["threshold"]["value"]

    for m, entry in zip(body["models"][2:], ZOO["models"]):
        assert m["label"] == entry["label"] and m["threshold"] == entry["threshold"]["value"]
        assert m["test_auroc"] == entry["test"]["auroc"]
    for m in body["models"]:
        assert set(m) == {"family", "label", "role", "probability", "threshold", "predicted_class", "test_auroc"}
        assert 0.0 <= m["probability"] <= 1.0
        # the served model's convention: probability >= threshold -> Malignant
        assert m["predicted_class"] == ("Malignant" if m["probability"] >= m["threshold"] else "Benign")
    assert body["above_threshold"] == {
        "k": sum(m["predicted_class"] == "Malignant" for m in body["models"]), "n": len(body["models"])}


def test_assess_models_is_deterministic():
    a = client.post("/api/assess-models", json=CASE).json()
    b = client.post("/api/assess-models", json={"density": 3, "margin": 5, "shape": 4, "age": 66}).json()
    assert a == b


@pytest.mark.parametrize("body", [
    {**CASE, "age": 17}, {**CASE, "age": 97}, {**CASE, "shape": 5}, {**CASE, "margin": 0},
    {**CASE, "density": 5}, {"age": 66, "shape": 4, "margin": 5}, {**CASE, "age": "old"}, {},
])
def test_assess_models_rejects_invalid_input(body):
    r = client.post("/api/assess-models", json=body)
    assert r.status_code == 422
    assert "models" not in r.json()


def test_assess_models_without_the_zoo_returns_served_and_challenger(no_zoo):
    body = client.post("/api/assess-models", json=CASE).json()
    assert [m["role"] for m in body["models"]] == ["served", "challenger"]
    assert body["above_threshold"]["n"] == 2
