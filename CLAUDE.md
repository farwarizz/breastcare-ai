# BreastCare AI: notes for coding assistants

Research and educational prototype (not a medical device). It estimates breast-mass malignancy
risk from four BI-RADS descriptors on the real UCI Mammographic Mass data: a calibrated
probability, a Mondrian conformal prediction set, SHAP contributions and a Gemini RAG assistant.
It is served by FastAPI (`server.py`) with one frontend (`app/index.html`).
Design: `docs/DESIGN.md` · model card: `docs/MODEL_CARD.md` · results: `reports/evaluation.md` ·
data provenance: `data/README.md`.

## Commands

- Run the app: `uvicorn server:app --reload`, then open http://localhost:8000
- Tests: `pytest`
- Retrain and evaluate: `python -m src.ml.train`, then `python scripts/update_readme_metrics.py`
  (deterministic; logs to MLflow if installed; dev dependencies in `requirements-dev.txt`)
- Verify the raw data: `python scripts/download_data.py` (needs internet and `ucimlrepo`)
- WDBC appendix: `python experiments/wdbc_benchmark.py`
- Rebuild the knowledge base: `python scripts/build_knowledge_base.py` (indexes every file in `knowledge_base/sources/`)
- On Windows, set `PYTHONUTF8=1` when piping script output.

## Rules

- Never add synthetic, heuristic or client-side fallbacks that produce a probability. Fail loudly,
  or return HTTP 503.
- The test split (`models/splits.json`) is used once, for reporting. Never tune anything on it:
  model choice, calibration, thresholds and conformal cut-offs all come from the fit and
  conformal splits.
- The served model is chosen by the rule in `src/ml/models.py`, not by hand.
- BI-RADS is never a model feature (label leakage); it is only a reference in evaluation.
- Doc numbers come only from `models/model_metadata.json` via `scripts/update_readme_metrics.py`
  (generated blocks). Static prose must not state results. The UI reads `/api/model-info`.
- The LLM provider is Gemini (`GEMINI_API_KEY`, OpenAI-compatible endpoint in `src/llm/client.py`).

## Roadmap

Phase 0 (honesty cleanup) and Phase 1 (ML rigor) are done. Next: Phase 2, RAG and safety
evaluation (index the full knowledge base, retrieval/faithfulness eval set, red-team suite);
Phase 3, packaging, CI and deployment; Phase 4, presentation.
