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


def test_health_check_does_not_import_torch():
    """The image model's torch stack loads only when an image is analysed."""
    import subprocess, sys
    code = ("import sys, server; from fastapi.testclient import TestClient; "
            "c = TestClient(server.app); c.get('/health'); "
            "print('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], cwd=config.ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "False"


def test_health_reports_models_truthfully(monkeypatch):
    monkeypatch.setattr(image_model, "is_available", lambda: False)
    body = client.get("/health").json()
    assert body["image_model"] is False
    assert body["served_model"] == json.loads(
        config.MODEL_METADATA_PATH.read_text(encoding="utf-8"))["served_model"]["label"]


def test_assess_returns_probability_and_prediction_set():
    r = client.post("/api/assess", json=FEATURES)
    assert r.status_code == 200
    pred = r.json()["prediction"]
    assert "confidence_band" not in pred
    # `calibrated` reports whether a calibration map was applied, not a constant
    assert pred["calibrated"] is (pred["calibration_method"] != "none")
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
    with open(config.CHUNKS_JSONL, encoding="utf-8") as f:
        n_chunks = sum(1 for line in f if line.strip())
    kb = body["knowledge_base"]
    assert kb["n_chunks"] == n_chunks
    assert kb["retrieval_mode"] == config.RETRIEVAL_MODE and kb["retrieval_query"] == config.RETRIEVAL_QUERY
    assert body["llm"]["model"] == config.GEMINI_MODEL
    # Evaluation summaries come straight from reports/*.json, without the per-item rows.
    for name, path in {"retrieval": "rag_evaluation.json", "answers": "answer_evaluation.json",
                       "safety": "safety_evaluation.json"}.items():
        report = config.EVAL_REPORTS_DIR / path
        if report.exists():
            saved = json.loads(report.read_text(encoding="utf-8"))
            saved.pop("rows", None)
            assert body["evaluations"][name] == saved
        else:
            assert name not in body["evaluations"]


def test_evaluation_figures_are_served():
    md = json.loads(config.MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    for name in md["figures"]:
        r = client.get(f"/figures/{name}")
        assert r.status_code == 200 and r.headers["content-type"] == "image/png", name
