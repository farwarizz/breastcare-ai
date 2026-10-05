"""API checks: the single supported frontend, honest image endpoint, real model outputs."""
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


def test_health_reports_models_truthfully(monkeypatch):
    monkeypatch.setattr(image_model, "is_available", lambda: False)
    body = client.get("/health").json()
    assert body["image_model"] is False
    assert body["served_model"] == json.loads(
        config.MODEL_METADATA_PATH.read_text(encoding="utf-8"))["served_model"]["label"]


def test_assess_returns_calibrated_probability_and_prediction_set():
    r = client.post("/api/assess", json=FEATURES)
    assert r.status_code == 200
    pred = r.json()["prediction"]
    assert "confidence_band" not in pred
    assert pred["calibrated"] is True
    assert set(pred["prediction_set"]) <= {"Benign", "Malignant"}


def test_compare_models_shows_served_and_challenger():
    r = client.post("/api/compare-models", json=FEATURES)
    assert r.status_code == 200
    body = r.json()
    md = json.loads(config.MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    names = [m["model_name"] for m in body["models"].values()]
    assert names == [md["served_model"]["label"] + " (served)",
                     md["challenger_model"]["label"] + " (challenger)"]
    assert body["agreement"]["test_auroc_difference"] == \
        md["comparisons"]["auroc_difference_served_minus_challenger"]


def test_batch_predict_validates_and_predicts():
    csv = "age,shape,margin,density\n45,1,1,3\n70,4,5,3\n"
    r = client.post("/api/batch-predict", files={"file": ("b.csv", csv.encode(), "text/csv")})
    assert r.status_code == 200
    rows = r.json()["results"]
    assert len(rows) == 2 and all("prediction_set" in row for row in rows)
    bad = "age,shape,margin,density\n45,9,1,3\n"
    r = client.post("/api/batch-predict", files={"file": ("b.csv", bad.encode(), "text/csv")})
    assert r.status_code == 400


def test_model_info_serves_saved_metadata():
    r = client.get("/api/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["metadata"] == json.loads(config.MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    chunks = json.loads(config.CHUNK_METADATA_PATH.read_text(encoding="utf-8"))
    assert body["knowledge_base"]["n_chunks"] == len(chunks)


def test_evaluation_figures_are_served():
    md = json.loads(config.MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    for name in md["figures"]:
        r = client.get(f"/figures/{name}")
        assert r.status_code == 200 and r.headers["content-type"] == "image/png", name
