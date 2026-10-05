"""Docs match the code: generated metrics, Gemini-only setup, one frontend."""
import os
import re
import subprocess
import sys

from src import config

ROOT = config.ROOT


def test_doc_metrics_match_model_metadata():
    result = subprocess.run(
        [sys.executable, "scripts/update_readme_metrics.py", "--check"],
        cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_no_stale_anthropic_or_claude_references():
    files = [
        ROOT / "README.md", ROOT / "docs" / "DESIGN.md", ROOT / "data" / "README.md",
        ROOT / "requirements.txt", ROOT / ".env.example", ROOT / "Procfile",
        ROOT / "server.py", ROOT / "app" / "index.html",
        *(ROOT / "src").rglob("*.py"), *(ROOT / "scripts").glob("*.py"),
        *(ROOT / "knowledge_base" / "sources").glob("*.md"),
        ROOT / "knowledge_base" / "chunks" / "chunks.jsonl",
        ROOT / "knowledge_base" / "vectorstore" / "chunk_metadata.json",
    ]
    stale = re.compile(r"anthropic|claude", re.IGNORECASE)
    offenders = [str(p.relative_to(ROOT)) for p in files
                 if stale.search(p.read_text(encoding="utf-8"))]
    assert offenders == []


def test_fastapi_html_is_the_only_frontend():
    assert not (ROOT / "app" / "streamlit_app.py").exists()
    assert not (ROOT / ".streamlit").exists()
    for name in ("requirements.txt", "README.md"):
        assert "streamlit" not in (ROOT / name).read_text(encoding="utf-8").lower()
    procfile = (ROOT / "Procfile").read_text(encoding="utf-8").strip()
    assert procfile == "web: uvicorn server:app --host 0.0.0.0 --port $PORT"


def test_readme_setup_matches_the_app():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "uvicorn server:app" in readme
    assert "GEMINI_API_KEY" in readme
    assert "GEMINI_API_KEY" in (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "not a medical device" in readme.lower()


def test_frontend_has_no_hardcoded_metrics_or_invented_estimates():
    html = (ROOT / "app" / "index.html").read_text(encoding="utf-8")
    assert "function sim(" not in html             # client-side fake estimate removed
    for stale in ("0.9924", "0.7251", "849 real", "60 curated", "10,000+"):
        assert stale not in html, stale
