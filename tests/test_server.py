"""API checks: the single supported frontend, honest image endpoint, no fake outputs."""
import io
import json

from fastapi.testclient import TestClient

import server
from src import config
from src.ml import image_model

client = TestClient(server.app)
FEATURES = {"age": 60, "shape": 4, "margin": 5, "density": 3}


def _png() -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (128, 128, 128)).save(buf, format="PNG")
    return buf.getvalue()


def test_root_serves_index_html():
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert r.content == (config.ROOT / "app" / "index.html").read_bytes()


def test_assess_image_returns_503_when_model_unavailable(monkeypatch):
    monkeypatch.setattr(image_model, "is_available", lambda: False)
    r = client.post("/api/assess-image", files={"file": ("m.png", _png(), "image/png")})
    assert r.status_code == 503
    body = r.json()
    assert "image model unavailable" in body["detail"].lower()
    assert "probability" not in body


def test_assess_image_returns_503_when_weights_fail_to_load(monkeypatch):
    def broken():
        raise RuntimeError("corrupt weights")
    monkeypatch.setattr(image_model, "is_available", lambda: True)
    monkeypatch.setattr(image_model, "ensure_loaded", broken)
    r = client.post("/api/assess-image", files={"file": ("m.png", _png(), "image/png")})
    assert r.status_code == 503
    assert "probability" not in r.json()


def test_health_does_not_claim_a_missing_image_model(monkeypatch):
    monkeypatch.setattr(image_model, "is_available", lambda: False)
    assert client.get("/health").json()["image_model"] is False


def test_assess_returns_prediction_without_fabricated_band():
    r = client.post("/api/assess", json=FEATURES)
    assert r.status_code == 200
    pred = r.json()["prediction"]
    assert "confidence_band" not in pred
    assert pred["calibrated"] is False


def test_compare_models_only_lists_trained_models():
    r = client.post("/api/compare-models", json=FEATURES)
    assert r.status_code == 200
    names = [m["model_name"] for m in r.json()["models"].values()]
    assert not any("Logistic" in n for n in names)   # the hand-set "baseline" is gone


def test_model_info_serves_saved_metadata():
    r = client.get("/api/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["metadata"] == json.loads(config.MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    chunks = json.loads(config.CHUNK_METADATA_PATH.read_text(encoding="utf-8"))
    assert body["knowledge_base"]["n_chunks"] == len(chunks)
