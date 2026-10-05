"""Predictions carry no fabricated uncertainty, and no heuristic image score exists."""
import pytest

from src import config
from src.ml import image_model
from src.ml.predict import predict

CASES = [
    {"age": 30, "shape": 1, "margin": 1, "density": 3},
    {"age": 70, "shape": 4, "margin": 5, "density": 3},
]


@pytest.mark.parametrize("features", CASES)
def test_predict_returns_uncalibrated_probability_only(features):
    out = predict(features)
    assert "confidence_band" not in out
    assert out["calibrated"] is False
    assert 0.0 <= out["probability"] <= 1.0
    assert not any("±" in str(v) for v in out.values())


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
