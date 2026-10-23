"""
The five deployment blockers from the 2026-10-20 backend audit, as tests:
audit endpoints need a token; /api/chat is rate limited; uploads are size-limited; error
responses carry no exception text; CORS allows only configured origins. Plus the
calibration-metadata consistency finding. No test calls Gemini.
"""
import json

import pytest
from fastapi.testclient import TestClient

import server
from src import config

client = TestClient(server.app)
FEATURES = {"age": 60, "shape": 4, "margin": 5, "density": 3}


# ── 1. Audit endpoints ────────────────────────────────────────────────────────
@pytest.mark.parametrize("method,path", [("get", "/api/audit/history"), ("get", "/api/audit/stats"),
                                         ("get", "/api/audit/export"), ("post", "/api/audit/session")])
def test_audit_endpoints_are_disabled_without_a_configured_token(monkeypatch, method, path):
    monkeypatch.delenv("AUDIT_API_TOKEN", raising=False)
    r = getattr(client, method)(path)
    assert r.status_code == 403 and "history" not in r.text


def test_audit_endpoints_need_the_right_token(monkeypatch):
    monkeypatch.setenv("AUDIT_API_TOKEN", "test-token-123")
    assert client.get("/api/audit/stats").status_code == 401
    assert client.get("/api/audit/stats", headers={"Authorization": "Bearer wrong"}).status_code == 401
    ok = client.get("/api/audit/stats", headers={"Authorization": "Bearer test-token-123"})
    assert ok.status_code == 200


# ── 2. Chat rate limit ────────────────────────────────────────────────────────
def test_chat_is_rate_limited_per_client(monkeypatch):
    import src.pipeline as pipeline
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-used")
    monkeypatch.setattr(server, "chat_limiter", server._RateLimiter(per_minute=2, per_day=100))
    monkeypatch.setattr(pipeline, "run_pipeline", lambda *a, **k: {
        "llm_response": {"main_text": "stub", "disclaimer": ""}, "prediction": {}, "explanation": {},
        "mode": "single", "citations": [], "guardrail": "ok", "trace": {}})
    body = {**FEATURES, "question": "What is a cyst?"}
    assert [client.post("/api/chat", json=body).status_code for _ in range(2)] == [200, 200]
    r = client.post("/api/chat", json=body)
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1


def test_chat_daily_cap_applies_across_clients():
    limiter = server._RateLimiter(per_minute=100, per_day=3)
    assert [limiter.check(c) for c in ("a", "b", "c")] == [None, None, None]
    assert limiter.check("d") is not None


# ── 3. Upload limits ──────────────────────────────────────────────────────────
def test_batch_upload_size_and_row_limits(monkeypatch):
    monkeypatch.setattr(server, "MAX_CSV_BYTES", 64)
    big = ("age,shape,margin,density\n" + "45,1,1,3\n" * 20).encode()
    assert client.post("/api/batch-predict", files={"file": ("b.csv", big, "text/csv")}).status_code == 413
    monkeypatch.setattr(server, "MAX_CSV_BYTES", 1024 * 1024)
    monkeypatch.setattr(server, "MAX_BATCH_ROWS", 5)
    assert client.post("/api/batch-predict", files={"file": ("b.csv", big, "text/csv")}).status_code == 413


def test_image_upload_size_limit(monkeypatch):
    from src.ml import image_model
    monkeypatch.setattr(image_model, "is_available", lambda: True)
    monkeypatch.setattr(image_model, "ensure_loaded", lambda: None)
    monkeypatch.setattr(server, "MAX_IMAGE_BYTES", 100)
    r = client.post("/api/assess-image", files={"file": ("m.png", b"\x89PNG" + b"0" * 500, "image/png")})
    assert r.status_code == 413


# ── 4. No exception text in responses ─────────────────────────────────────────
def test_unreadable_csv_gets_a_generic_message():
    r = client.post("/api/batch-predict", files={"file": ("b.csv", b"\x00\x01\x02,\"unclosed\n", "text/csv")})
    assert r.status_code == 400
    assert "Error tokenizing" not in r.text and "line" not in r.json()["detail"].lower()


def test_batch_validation_messages_are_still_shown():
    bad = b"age,shape,margin,density\n45,9,1,3\n"
    r = client.post("/api/batch-predict", files={"file": ("b.csv", bad, "text/csv")})
    assert r.status_code == 400 and "shape" in r.json()["detail"]


def test_image_model_errors_are_generic(monkeypatch):
    from src.ml import image_model

    def broken():
        raise RuntimeError("CUDA error at C:/secret/path/model.pth")
    monkeypatch.setattr(image_model, "is_available", lambda: True)
    monkeypatch.setattr(image_model, "ensure_loaded", broken)
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    r = client.post("/api/assess-image", files={"file": ("m.png", buf.getvalue(), "image/png")})
    assert r.status_code == 503 and "secret" not in r.text and "CUDA" not in r.text

    def predict_fails(img):
        raise ValueError("tensor shape mismatch in /srv/model.py")
    monkeypatch.setattr(image_model, "ensure_loaded", lambda: None)
    monkeypatch.setattr(image_model, "predict_image", predict_fails)
    r = client.post("/api/assess-image", files={"file": ("m.png", buf.getvalue(), "image/png")})
    assert r.status_code == 500 and "tensor" not in r.text and "/srv" not in r.text


# ── 5. CORS ───────────────────────────────────────────────────────────────────
def test_cors_allows_only_configured_origins():
    allowed = client.options("/api/assess", headers={"Origin": "http://localhost:5199",
                                                     "Access-Control-Request-Method": "POST"})
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:5199"
    other = client.options("/api/assess", headers={"Origin": "https://evil.example",
                                                   "Access-Control-Request-Method": "POST"})
    assert other.headers.get("access-control-allow-origin") is None
    assert "*" not in server.ALLOWED_ORIGINS


# ── Calibration metadata (audit finding 6) ────────────────────────────────────
def test_calibration_claims_match_the_calibration_step():
    md = json.loads(config.MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    method = md["served_model"]["calibration"]["method"]
    pred = client.post("/api/assess", json=FEATURES).json()["prediction"]
    assert pred["calibration_method"] == method and pred["calibrated"] is (method != "none")
    if method == "none":
        assert "calibrated probability" not in md["protocol"]["risk_categories"]
        ui = (config.ROOT / "app" / "index.html").read_text(encoding="utf-8").lower()
        assert "calibrated" not in ui.replace("uncalibrated", "")
        assert "calibrated probability" not in (config.ROOT / "README.md").read_text(encoding="utf-8").lower()
    info = client.get("/api/model-info").json()["metadata"]
    assert info["served_model"]["calibration"]["method"] == method
