"""Served-model outputs: calibrated, with a valid conformal set, exact explanations,
clinically sensible behaviour, and no heuristic image score."""
import itertools

import numpy as np
import pandas as pd
import pytest

from src import config
from src.ml import image_model
from src.ml.artifacts import model_meta
from src.ml.explain import contributions, explain
from src.ml.predict import predict, predict_frame

CASES = [
    {"age": 30, "shape": 1, "margin": 1, "density": 3},
    {"age": 52, "shape": 3, "margin": 4, "density": 3},
    {"age": 70, "shape": 4, "margin": 5, "density": 3},
]
GRID = pd.DataFrame(
    [{"age": a, "shape": s, "margin": m, "density": d}
     for a, s, m, d in itertools.product(range(20, 91, 10), (1, 2, 3, 4), (1, 2, 3, 4, 5), (1, 2, 3, 4))])


@pytest.mark.parametrize("features", CASES)
def test_prediction_has_a_conformal_set_and_an_honest_calibration_flag(features):
    out = predict(features)
    assert "confidence_band" not in out
    # true only when the training rule chose a calibration map (it chose "none" for the served model)
    assert out["calibrated"] is (model_meta()["calibration"]["method"] != "none")
    assert 0.0 <= out["probability"] <= 1.0
    assert set(out["prediction_set"]) <= {"Benign", "Malignant"}
    assert out["set_coverage"] == model_meta()["conformal"]["coverage_target"]
    assert out["model"] == model_meta()["label"]
    assert not any("±" in str(v) for v in out.values())


def test_prediction_set_follows_the_stored_cutoffs():
    conf = model_meta()["conformal"]
    pred = predict_frame(GRID)
    for p, s in zip(pred["probability"], pred["prediction_set"]):
        assert ("Benign" in s) == (p <= conf["benign_if_p_le"])
        assert ("Malignant" in s) == (p >= conf["malignant_if_p_ge"])


def test_contributions_add_up_to_the_model_log_odds():
    base, contrib = contributions(GRID)
    raw = predict_frame(GRID)["raw_probability"].to_numpy()
    logit = np.log(raw / (1 - raw))
    assert np.allclose(base + contrib.sum(axis=1), logit, atol=1e-6)


def test_explanation_shape():
    e = explain(CASES[2])
    assert set(e["feature_attributions"]) == set(config.FEATURE_NAMES)
    assert e["dominant_feature"] in config.FEATURE_NAMES
    assert e["method"]


@pytest.mark.parametrize("which", ["served", "challenger"])
def test_risk_never_decreases_with_age(which):
    pred = predict_frame(GRID, which)["probability"].to_numpy()
    by_age = pred.reshape(8, -1)                 # rows: age 20..90, cols: other features
    assert np.all(np.diff(by_age, axis=0) >= -1e-9)


@pytest.mark.parametrize("which", ["served", "challenger"])
def test_clinical_partial_orders(which):
    """Irregular ≥ round shape and spiculated ≥ circumscribed margin, all else equal."""
    p = predict_frame(GRID, which).assign(**GRID)
    key = ["age", "margin", "density"]
    irregular = p[p["shape"] == 4].set_index(key)["probability"]
    round_ = p[p["shape"] == 1].set_index(key)["probability"]
    assert (irregular - round_ >= -1e-9).all()
    key = ["age", "shape", "density"]
    spiculated = p[p["margin"] == 5].set_index(key)["probability"]
    circumscribed = p[p["margin"] == 1].set_index(key)["probability"]
    assert (spiculated - circumscribed >= -1e-9).all()


def test_image_demo_fallback_is_gone():
    assert not hasattr(image_model, "predict_image_demo")
    for path in [config.ROOT / "server.py", config.ROOT / "src" / "ml" / "image_model.py"]:
        text = path.read_text(encoding="utf-8")
        for marker in ("predict_image_demo", "dark_ratio", "brightness", "Statistical Image Analysis"):
            assert marker not in text, f"{marker!r} still in {path.name}"


def test_real_image_model_output_has_no_fabricated_interval():
    if not image_model.is_available():
        pytest.skip("real image-model weights or torch/timm not installed")
    from PIL import Image
    out = image_model.predict_image(Image.new("RGB", (64, 64), (120, 120, 120)))
    assert "confidence_band" not in out
    assert out["calibrated"] is False
    assert 0.0 <= out["probability"] <= 1.0
