# BreastCare AI: notes for coding assistants

Research and educational prototype (not a medical device). It estimates breast-mass malignancy
risk from four BI-RADS descriptors on the real UCI Mammographic Mass data: a calibrated
probability, a Mondrian conformal prediction set, SHAP contributions and a Gemini RAG assistant
with input guardrails, numbered citations and an output filter.
It is served by FastAPI (`server.py`) with one frontend (`app/index.html`).
Design: `docs/DESIGN.md` · model card: `docs/MODEL_CARD.md` · results: `reports/evaluation.md` and
`reports/rag_evaluation.md` · data provenance: `data/README.md`.

## Commands

- Run the app: `uvicorn server:app --reload`, then open http://localhost:8000
- Tests: `pytest`
- Retrain and evaluate: `python -m src.ml.train`, then `python scripts/update_readme_metrics.py`
  (deterministic; logs to MLflow if installed; dev dependencies in `requirements-dev.txt`)
- Verify the raw data: `python scripts/download_data.py` (needs internet and `ucimlrepo`)
- WDBC appendix: `python experiments/wdbc_benchmark.py`
- Rebuild the knowledge base: `python scripts/build_knowledge_base.py` (chunks every file in
  `knowledge_base/sources/` and writes the `fastembed`/`model2vec` embeddings and manifest to
  `knowledge_base/index/`; BM25 is rebuilt in memory at load time)
- Assistant evaluations: `python -m src.evals.retrieval` (no LLM), `python -m src.evals.answers`,
  `python -m src.evals.redteam` (both call Gemini; responses are cached in `evals/cache/`, commit the cache),
  then `python scripts/update_readme_metrics.py`
- On Windows, set `PYTHONUTF8=1` when piping script output.

## Rules

- Never add synthetic, heuristic or client-side fallbacks that produce a probability. Fail loudly,
  or return HTTP 503.
- The test split (`models/splits.json`) is used once, for reporting. Never tune anything on it:
  model choice, calibration, thresholds and conformal cut-offs all come from the fit and
  conformal splits.
- The served model is chosen by the rule in `src/ml/models.py`, not by hand.
- BI-RADS is never a model feature (label leakage); it is only a reference in evaluation.
- Doc numbers come only from `models/model_metadata.json` and `reports/*_evaluation.json` via
  `scripts/update_readme_metrics.py` (generated blocks). Static prose must not state results. The UI
  reads `/api/model-info`.
- `RETRIEVAL_MODE` and `RETRIEVAL_QUERY` in `src/config.py` are copied from the choice that
  `src/evals/retrieval.py`'s fixed rule makes, never picked by taste; `tests/test_evals.py` fails if they
  disagree with the saved report, or if the report describes a different index.
- The input-guardrail regexes are frozen against `evals/redteam_heldout.jsonl`: never change
  `src/guardrails.py` because of a held-out result. Add the case to the development set and write new
  held-out prompts instead, or the held-out number stops meaning anything.
- Knowledge-base documents state only what their `## Sources` section supports. Keep section headings
  stable: `evals/retrieval_questions.jsonl` and the evaluation's `SECTION_FOR` map refer to them.
- The LLM provider is Gemini (`GEMINI_API_KEY`, OpenAI-compatible endpoint in `src/llm/client.py`).
  Gemini 2.5+/3.x models think by default and the hidden tokens count against `max_tokens`, so
  `LLM_MODEL_PARAMS` keeps `reasoning_effort` minimal; keep that when adding a model.

## Roadmap

Phase 0 (honesty cleanup) and Phase 1 (ML rigor) are done. Phase 2 (fact-checked knowledge base,
evaluated retrieval, cited answers, guardrails, red team) is implemented; held-out v1 showed the safety
layers did not generalise (crisis misses), so Phase 2.1 fixed them and is re-evaluated on held-out v2.
Phase 3 (packaging, CI, deployment, audit trail in the UI, real drift baselines) starts only when the
GO conditions are met: no crisis misses and attack success below the no-safety baseline on held-out v2,
the judge-agreement sheet labelled by a person, and the commits approved. Then Phase 4, presentation.
Nothing is published or deployed before that.
