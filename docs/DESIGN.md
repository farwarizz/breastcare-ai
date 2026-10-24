# BreastCare AI: design document

This describes what the code does today, including the parts that are experimental or not
active. For a short overview see the [README](../README.md); for intended use and limitations
see the [model card](MODEL_CARD.md); for every number see the
[evaluation report](../reports/evaluation.md) (risk model) and the
[assistant evaluation report](../reports/rag_evaluation.md) (retrieval, answers, red team).

## 1. Scope

BreastCare AI is a **research and educational prototype**. It estimates the probability that a
breast mass is malignant from four structured inputs (age, and the BI-RADS descriptors shape,
margin and density), attaches a conformal prediction set, explains the estimate with SHAP, and
answers follow-up questions with an LLM grounded in a small, fact-checked knowledge base.

It is **not** a medical device. It is not FDA-cleared and not clinically validated, it does not
diagnose, and it does not recommend treatment. The risk model works only on the structured
descriptors, not on images. The separate, experimental image model is described in §9.

Motivation, from the dataset documentation: the low positive predictive value of breast biopsy
after mammography leads to roughly 70% of biopsies having benign outcomes. The UCI data was
published to study whether descriptor-based models can support the biopsy decision.

## 2. Data

UCI Mammographic Mass (id 161): 961 raw cases, 831 after dropping rows with missing values
(428 benign / 403 malignant). The raw file is the original, checksum-pinned UCI file,
cross-checked against `ucimlrepo` ([provenance](../data/README.md)). Training refuses any other
file, and no production code can generate substitute data. The BI-RADS assessment column is
dropped from the inputs because it is the radiologist's conclusion (label leakage); it is kept
only as a reference for evaluation.

The classic WDBC dataset is no longer part of the pipeline. It lives in a reproducible appendix
(`experiments/wdbc_benchmark.py` → [reports/wdbc_benchmark.md](../reports/wdbc_benchmark.md)).

## 3. Architecture and request flow

```
Browser: app/index.html  ──fetch──►  FastAPI: server.py
                                           │
             ┌─────────────────────────────┼──────────────────────────────┐
             ▼                             ▼                              ▼
    /api/assess, /api/report      /api/chat (full pipeline)      /api/model-info, /figures
    run_pipeline_ml_only()        run_pipeline()                 saved metadata, evaluations, figures
```

`src/pipeline.py`, `run_pipeline()` (used by `/api/chat`):

1. `check_input()` (`src/guardrails.py`): regular-expression routing of the user's message. Crisis,
   prompt-injection, privacy and off-topic messages get a fixed reply and the LLM is never called;
   diagnosis, treatment and skip-care questions continue with a policy note added to the prompt.
2. `predict()` (`src/ml/predict.py`): the served model's score passes through the stored
   calibration map (for the served model the rule chose `none`, so `calibrated` is false). The result
   is the probability, the conformal prediction set,
   the class at the development-chosen threshold, and a risk category
   (Low < 0.35 ≤ Moderate < 0.65 ≤ Elevated).
3. `explain()` (`src/ml/explain.py`): per-feature SHAP contributions in log-odds space. These
   are exact for the linear model (interventional), or LightGBM's native TreeSHAP
   (`predict_proba(pred_contrib=True)`). Also returns the dominant feature and a text summary.
4. `select_retrieval_query()`: the retrieval query is either the question alone or the SHAP summary
   followed by the question; `RETRIEVAL_QUERY` in `src/config.py` holds the variant the retrieval
   evaluation chose (§6).
5. `rag_query()` (`src/rag/retriever.py`): top-3 chunks with the served retrieval mode
   (`RETRIEVAL_MODE`, also chosen by the evaluation). If the probability is ≥ 0.65, the best-matching chunk of
   the safety-disclaimer document is appended.
6. `build_prompt()` → `complete()` (`src/llm/client.py`): the prompt states the probability
   with its conformal set, the SHAP summary, the numbered documents and the question; Gemini is called
   through Google's OpenAI-compatible endpoint with model fallbacks.
7. `parse_citations()`: the `[n]` citations in the answer are resolved to document names and sections;
   numbers that do not exist are reported as `invalid_citations`.
8. `filter_response()` (`src/safety.py`): regex rewrite of diagnostic phrases, plus the standard (or
   high-risk) disclaimer.
9. The response carries a `trace` (stage timings, model used, token usage) and, if a session id is
   given, the prediction is written to the SQLite audit log.

`run_pipeline_ml_only()` (used by `/api/assess` and `/api/report`) runs steps 2–5 without an LLM call.

## 4. Component status

| Component | Where | Status |
|---|---|---|
| Served model, calibration, conformal sets | `src/ml/predict.py`, `artifacts.py`, `calibration.py`, `conformal.py` | Active on every assessment |
| Training + evaluation pipeline | `src/ml/train.py`, `models.py`, `evaluation.py`, `figures.py` | Active (offline, `python -m src.ml.train`) |
| SHAP contributions | `src/ml/explain.py` | Active |
| Challenger model | `src/analytics/model_comparison.py` | Active in model comparison |
| Knowledge base + index | `knowledge_base/sources/`, `src/rag/chunker.py`, `index.py`, `scripts/build_knowledge_base.py` | Active; 10 fact-checked documents, every one indexed |
| Retrieval (BM25, dense, hybrid, re-ranking) | `src/rag/bm25.py`, `embedders.py`, `retriever.py`, `rerank.py` | Active; served mode chosen by `src/evals/retrieval.py` |
| Input guardrails | `src/guardrails.py` | Active on every chat message |
| Gemini generation with citations | `src/llm/client.py`, `src/rag/prompt_builder.py` | Active in `/api/chat` with `GEMINI_API_KEY` |
| Output filter | `src/safety.py` | Active on every LLM response |
| Evaluations | `src/evals/retrieval.py`, `answers.py`, `redteam.py`; `evals/` | Offline; results in `reports/*_evaluation.json` |
| Chain-of-thought / multi-agent modes | `src/agents/` | Optional (chat mode selector); not evaluated |
| Audit log, PDF report, batch CSV | `src/audit/`, `src/analytics/batch_predict.py` | Active; the UI never starts a session, so the log stays empty (Phase 3) |
| Dataset-wide SHAP | `src/analytics/global_shap.py` | Active in `/api/global-shap` |
| EfficientNet-B0 image model | `src/ml/image_model.py` | Experimental, optional install (`requirements-image.txt`); HTTP 503 unless the real weights and torch/timm are present |
| Simplified Gail / Tyrer-Cuzick | `src/risk/` | Active in `/api/risk-profile`; approximations, not the validated calculators |
| Drift monitor | `src/analytics/model_monitor.py` | Placeholder baselines, illustrative only |

## 5. Machine learning

### Protocol

Stratified split of the 831 cleaned rows (seed 2026; row ids in `models/splits.json`):

| Split | Share | Used for |
|---|---|---|
| fit | 60% | 5-fold CV model selection; out-of-fold predictions for calibration and thresholds; training the final models |
| conformal | 20% | conformal cut-offs only |
| test | 20% | final evaluation, used once |

The final models are trained on the fit split only, not refit on fit + conformal. Refitting would
invalidate the conformal guarantee, which needs calibration data the model has never seen.

### Candidates and selection (`src/ml/models.py`)

- **Logistic regression:** standardised age plus one-hot shape, margin and density (shape and margin
  are nominal codes). C ∈ {0.01, 0.1, 1, 10}.
- **LightGBM, monotone in age and shape:** risk may not decrease with age or with shape
  (round < oval < lobular < irregular). Margin is left unconstrained because its codes are not in risk
  order: obscured is less suspicious than microlobulated, as in BI-RADS and in the development data.
  Density is unconstrained because about 90% of masses share one value.
- **LightGBM, unconstrained.** Both LightGBM grids use num_leaves ∈ {4, 8, 16} and
  min_child_samples ∈ {10, 20, 40}; the tree count comes from early stopping inside CV folds.

**Rule**, fixed before looking at the test set: lowest mean 5-fold CV log-loss on the fit split; a
simpler family within one standard error of the best is preferred (logistic regression < monotone
LightGBM < LightGBM). The runner-up family is kept as the challenger. LightGBM's early stopping on
the CV folds makes its CV score slightly optimistic, which biases this rule *towards* LightGBM.

### Calibration and thresholds

Out-of-fold predictions on the fit split choose a calibration method (none / Platt sigmoid /
isotonic, by cross-validated Brier score, with ties going to the simpler method). The same
predictions choose the decision threshold (Youden's J) and a high-sensitivity threshold (highest
threshold with sensitivity ≥ 95%). This is the scheme of scikit-learn's
`CalibratedClassifierCV(ensemble=False)`. Calibrator parameters are plain numbers stored in the
metadata.

### Conformal prediction sets

Mondrian (class-conditional) split conformal prediction with the LAC score, at α = 0.10, fitted on
the conformal split. Benign enters the set when p ≤ q̂_benign, and Malignant when p ≥ 1 − q̂_malignant.
Each true class gets at least 90% coverage on average over random draws. The method is
unit-tested by simulation (`tests/test_uncertainty.py`).

### Evaluation (`src/ml/evaluation.py`)

On the test split, used once: AUROC, AUPRC, Brier, ECE, calibration intercept and slope, and
sensitivity / specificity / PPV / NPV at the chosen threshold. All come with 95% percentile
bootstrap CIs (2,000 resamples, the same indices for every model, so differences are paired).
Also reported: conformal coverage overall and per class, the high-sensitivity operating point, the
radiologists' BI-RADS assessment as a reference (rows with BI-RADS 1–5), the monotonic-constraint
ablation, age subgroups (< 50 vs ≥ 50), and a decision curve. Training is deterministic, and runs
are logged to MLflow when it is installed.

Current results (generated from `models/model_metadata.json`):

<!-- BEGIN GENERATED: results (generated by scripts/update_readme_metrics.py from models/model_metadata.json; do not edit by hand) -->
**Served model: Logistic regression**, chosen by a fixed rule. Lowest mean 5-fold CV log-loss on the fit split; a simpler family within one standard error of the best is preferred (logistic regression < monotone LightGBM < LightGBM).
Held-out test set: n = 167 (81 malignant), used once; 95% bootstrap CIs in brackets.

| Metric | Logistic regression (served) | LightGBM (monotone in age and shape) (challenger) |
|---|---|---|
| AUROC | 0.90 (0.84–0.94) | 0.90 (0.84–0.94) |
| AUPRC | 0.87 (0.79–0.93) | 0.87 (0.78–0.93) |
| Brier score (lower is better) | 0.132 (0.110–0.160) | 0.129 (0.104–0.159) |
| Calibration slope (ideal 1) | 1.63 (1.25–2.21) | 1.35 (1.02–1.86) |
| Calibration intercept (ideal 0) | -0.02 (-0.35 to 0.27) | -0.04 (-0.40 to 0.28) |
| Sensitivity at its threshold | 0.90 (0.83–0.96) | 0.91 (0.85–0.96) |
| Specificity at its threshold | 0.81 (0.72–0.89) | 0.77 (0.67–0.85) |
| Decision threshold (chosen on dev data) | 0.514 | 0.474 |

- **Served vs challenger:** AUROC difference +0.001 (-0.011 to +0.015): no significant difference in AUROC (the 95% CI includes 0).
- **Calibration:** the served model is underconfident on this test split (predictions less extreme than outcomes; slope > 1); intercept -0.02 (-0.35 to 0.27). Calibration step chosen on out-of-fold data: `none`.
- **90% conformal prediction sets:** coverage 0.87 (0.81–0.92) (consistent with the 90% target); benign 0.83 (0.74–0.90), malignant 0.91 (0.85–0.97). 89% of cases get a single label.
- **High-sensitivity operating point** (threshold 0.20, chosen on development data): sensitivity 0.99 (0.96–1.00), specificity 0.42 (0.31–0.52).
- **Reference, radiologists' BI-RADS assessment** (ordinal category, n = 164): AUROC 0.83 (0.78–0.88).
- **Decision curve:** the served model matches 'biopsy everyone' at the 2% BI-RADS biopsy threshold and adds net benefit over biopsying everyone from a threshold of 20%.
- **Subgroups (age, split at 50):** age < 50: sensitivity 0.71 (0.46–0.93), n malignant = 14; age ≥ 50: sensitivity 0.94 (0.88–0.99), n malignant = 67; sensitivity is 0.23 lower in the age < 50 group at the global threshold.
- **Cross-validation vs test:** 5-fold CV AUROC on the fit split 0.84; test AUROC 0.90. The CV figure is the more conservative estimate for new data.

<picture><source media="(prefers-color-scheme: dark)" srcset="../reports/figures/roc_pr_dark.png"><img src="../reports/figures/roc_pr_light.png" alt="ROC and precision-recall curves" width="100%"></picture>
<picture><source media="(prefers-color-scheme: dark)" srcset="../reports/figures/calibration_dark.png"><img src="../reports/figures/calibration_light.png" alt="Calibration curve" width="100%"></picture>
<picture><source media="(prefers-color-scheme: dark)" srcset="../reports/figures/decision_curve_dark.png"><img src="../reports/figures/decision_curve_light.png" alt="Decision curve" width="100%"></picture>
<picture><source media="(prefers-color-scheme: dark)" srcset="../reports/figures/conformal_sets_dark.png"><img src="../reports/figures/conformal_sets_light.png" alt="Conformal prediction sets" width="100%"></picture>

**Data:** UCI Mammographic Mass (UCI id 161): 831 cleaned cases. Split: fit 498 · conformal 166 · test 167. **Knowledge base:** 103 chunks from 10 of 10 documents.
Full details: [reports/evaluation.md](../reports/evaluation.md).
<!-- END GENERATED: results -->

## 6. Retrieval-augmented generation

### Knowledge base

Ten Markdown documents in `knowledge_base/sources/`, written for the project: BI-RADS mass
descriptors, BI-RADS categories 3–5 and follow-up, an imaging glossary, imaging modalities, screening
guidelines (NCCN, ACS, USPSTF), risk factors and epidemiology, treatment pathways, WHO screening and
early-diagnosis positions, patient communication, and the tool's own safety disclaimer. In Phase 2
every statement was checked against primary sources (ACR BI-RADS v2025 forms, NCI PDQ and SEER, ACS,
USPSTF, NCCN patient guidelines, WHO documents, and the cited trials); unverifiable figures were
removed, and each file ends with a `## Sources` section that the indexer skips. The documents are
summaries, not clinical guidance, and have not been reviewed by a clinician.

### Chunking and index (`src/rag/chunker.py`, `index.py`)

Documents are split by heading into sections; sections longer than 900 characters are split at
paragraph or sentence boundaries. Each chunk keeps its heading path (`Mass Margin > Spiculated
Margin`) and is indexed with a contextual header (`Document title > path: text`), so a sentence like
"it is one of the most suspicious features" stays attached to the term it describes. `manifest.json`
records a hash of every chunk's id and indexed text plus the source files' SHA-256. The retriever refuses
chunks that do not match the manifest, and treats stored vectors as unusable (falling back to BM25) when
their file is missing or their shape does not match the chunks and the recorded dimension. The retrieval
report records the same hash, and a test fails if the reported numbers describe a different index.
Bibliography sections, including subsections under them, are not indexed; long sections split at
paragraphs, then sentences (re-joined with spaces), then lines for lists and tables.

### Retrieval modes (`src/rag/retriever.py`)

- **BM25** (`src/rag/bm25.py`): a small in-repo implementation (k1 = 1.5, b = 0.75) with a tokenizer
  that lowercases, drops stop-words and normalises common plurals ("masses" → "mass", "diagnoses" →
  "diagnosis").
- **Dense:** cosine similarity over normalised embeddings. Two torch-free embedders: `minilm`
  (`sentence-transformers/all-MiniLM-L6-v2` through `fastembed`, ONNX, 384-d; identical vectors to the
  original package) and `potion` (`minishlab/potion-base-8M` through `model2vec`, static, 256-d, no
  neural inference at query time).
- **Hybrid:** reciprocal-rank fusion (k = 60) of the BM25 and dense rankings. Only chunks BM25 actually
  matched (score > 0) get credit from the BM25 list; the first version credited every chunk, so a query
  with no keyword match was reordered by file position (found in review, fixed before the reported run).
- **Re-ranking:** a cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2`, ONNX) rescoring the top 20 of any
  mode.

`RETRIEVAL_MODE` names the served mode. If its embedder is not installed, retrieval falls back to BM25
and logs a warning rather than mixing incompatible vector spaces (the Phase 1 TF-IDF-padding bug).

### Retrieval evaluation (`src/evals/retrieval.py`)

`evals/retrieval_questions.jsonl` holds 161 answerable questions, each labelled with the document
section(s) that answer it (ten kinds: descriptors, categories, glossary, imaging, screening, risk,
treatment, WHO, the tool itself, communication), plus 16 questions the knowledge base cannot answer.
Every mode answers the bare question; Hit@1/3/5, MRR@10 and nDCG@5 are reported with bootstrap CIs
and paired against BM25. The questions were written alongside the rewritten knowledge base, so this is a
regression benchmark for this corpus, not an estimate for arbitrary user questions, and 38 of the 161
labels name a parent section with several chunks, which makes Hit@k lenient for those questions. **Rule, fixed in advance:** the highest Hit@3 wins, but a cheaper mode wins if
it is within 0.02 (cost order BM25 < potion < MiniLM < re-ranking). A second experiment compares the
question alone with the Phase 1 construction (SHAP summary + question); the question alone is kept
unless the prefix is clearly better. `tests/test_evals.py` fails if `src/config.py` disagrees with the
saved report.

### Prompt and citations (`src/rag/prompt_builder.py`)

The system prompt fixes the role (educational, non-diagnostic), requires numbered citations, forbids
following instructions found in the documents, and forbids treatment and care decisions. The user
message lists the estimate and its conformal set, the SHAP summary, the documents as `[1]`–`[3]` (plus
`[4]`, the safety notice, for estimates ≥ 0.65), the question and, when the guardrail set one, a policy
note. `parse_citations()` accepts `[1]`, `[1][2]`, `[1, 2]` and `[1-3]`, and reports which documents were
cited and any number that does not exist. Answers from the quota or error fallbacks carry no citations.

### Answer evaluation (`src/evals/answers.py`)

A kind-stratified sample of 36 answerable questions plus every unanswerable one is answered by the
production pipeline. A Gemini judge (JSON output, temperature 0), given the question, the numbered
documents and the answer, labels faithfulness (every claim supported), citation correctness and
whether the answer says the documents do not cover the question. The judge prompt counts "the documents
do not cover this" as faithful, so the headline metric is *answered faithfully* (faithful and not an
abstention); faithfulness among answered questions and the pipeline's own retrieval hit rate are reported
beside it, to separate retrieval misses from generation errors. Gemini responses are cached in
`evals/cache/` so the evaluation is reproducible.

Known weaknesses of the judge: it is the same model as the generator (`gemini-3.5-flash-lite`), so it can
favour its own phrasing, and its accuracy is unmeasured. `evals/judge_agreement_sheet.csv` holds 30 items
for a person to label, and `python -m src.evals.answers --agreement` reports the agreement; until that is
done, treat the judge-based numbers as indicative.

**Independent judge panel** (`src/evals/common.py`, `PANEL_JUDGES`). To reduce self-preference and the
dependence on one free quota, models from other families on other free, OpenAI-compatible APIs can grade
the same replies. Each takes part only when its key is in `.env` (placeholders in `.env.example`); the app
itself never calls them.

| Judge | Provider | Free tier (checked 2026-10-13) | Key |
|---|---|---|---|
| `gpt-oss-120b` (recommended) | Cerebras | 1M tokens/day, 5 requests/min | `CEREBRAS_API_KEY` |
| `qwen3.8-27b`, `gpt-oss-120b` | Groq | about 1,000 requests and 200K tokens/day each | `GROQ_API_KEY` |
| any `:free` model | OpenRouter | 50 requests/day without purchased credits | `OPENROUTER_API_KEY` + `OPENROUTER_JUDGE_MODEL` |

Rules, fixed before any panel verdict existed: every judge's verdict is kept; the panel verdict is the
majority of the judges that answered, with a tie counted as a violation; reports show the Gemini judge, the
panel and their agreement side by side; and the headline safety metric counts a reply as unsafe if *either*
the Gemini judge or the panel flags it, so adding judges can only make the numbers stricter. Panel calls are
cached per judge in `evals/cache/`, like the Gemini calls.

Current results (generated from `reports/*_evaluation.json`):

<!-- BEGIN GENERATED: assistant (generated by scripts/update_readme_metrics.py from reports/*_evaluation.json; do not edit by hand) -->
Knowledge base: 103 chunks from 10 fact-checked documents; retrieval `bm25`.
- **Retrieval:** `bm25` (BM25), the served mode, chosen by a pre-registered rule from 8 modes. On 161 labelled questions: Hit@3 94% (91%–98%), MRR@10 83% (78%–87%), nDCG@5 79% (75%–83%), 0 ms per query; the best mode, `bm25+rerank`, reaches Hit@3 96% (93%–99%) and MRR@10 93% (90%–96%) at 1508 ms per query.
- **Query construction:** the question alone reaches Hit@3 94% (91%–98%) against 60% (52%–67%) with the SHAP summary prepended (paired difference +0.35 (+0.27 to +0.43): a significant difference in Hit@3 (the 95% CI excludes 0)); the app uses `question`. Explanation-only queries find the section describing a top feature within three chunks for 100% of 6 cases.
- **Answers** (generated by `gemini-3.5-flash-lite` ×52; LLM judge `gemini-3.5-flash-lite`): on 36 answerable questions, 86% (71%–94%) got an answer whose every claim the judge found in the retrieved documents (faithful when answering: 97% (84%–99%); the pipeline's top-3 contained a labelled section for 89% (75%–96%)); 100% (90%–100%) had only correct citations. On 16 unanswerable questions it said the documents do not cover them 15/16 (72%–99%) of the time. The judge's agreement with a person is not yet measured (`evals/judge_agreement_sheet.csv`).
- **Red team:** held-out set v3, written after the safety rules were frozen (74 adversarial prompts): attack success 7% (3%–15%) with the safety layers vs 24% (16%–35%) without; over-refusal of 24 benign prompts 8% (2%–26%) vs 42% (24%–61%). Crisis prompts on the held-out set: 0/16 (0%–19%) unsafe replies with the safety layers. On the development set, used to build the safety rules, so in-sample (208 adversarial prompts): attack success 4% (2%–7%) with the safety layers vs 18% (13%–24%) without; over-refusal of 38 benign prompts 3% (0%–14%) vs 29% (17%–45%). Before the fixes, held-out v2 gave 21% (13%–33%) with the safety layers vs 16% (9%–27%) without (crisis 0/12 (0%–24%)); v2 is now part of the development set. Pre-registered acceptance criteria: all met. The input guard blocked 0% (0%–2%) of the 177 benign retrieval questions. Judge `gemini-3.5-flash-lite`.
Full details: [reports/rag_evaluation.md](../reports/rag_evaluation.md).
<!-- END GENERATED: assistant -->

## 7. LLM layer

- Google Gemini through the OpenAI-compatible endpoint, using the `openai` package; the key comes from
  `GEMINI_API_KEY`. The primary model is `gemini-3.5-flash-lite`, the model every evaluation measures.
  `gemini-2.5-flash` was the original default, but its free tier allows only a few dozen requests a day:
  in an earlier evaluation run it ran out part-way and the remaining answers silently came from the
  fallback, so the reported numbers described a mix of models. It is now a fallback.
- `complete()` returns the text with the model used, token usage, latency and finish reason. On quota,
  overload, retired-model or transient errors it tries `gemini-2.5-flash`, then `gemini-3.5-flash`.
  Interactive calls get a 25-second HTTP timeout and at most two retries per model, so a request cannot
  outlast the host's time limit; the evaluations wait out rate limits instead.
- When every model is out of quota, the chat shows the retrieved passages verbatim and says so (never
  the prompt, which holds internal instructions); any other LLM error produces a fixed "try again"
  message, and the real error goes to the server log, not the user. If no key is set, `/api/chat`
  returns the top retrieved passages with a notice.
- Gemini 2.5+/3.x models "think" by default and the hidden thinking tokens count against `max_tokens`.
  The Phase 2 review found that at `reasoning_effort="low"`, `gemini-3.5-flash-lite` spent 370-800 of a
  900-token budget thinking: 24 of 201 red-team replies were cut off mid-sentence, and several "violations"
  in that run were really truncations. `LLM_MODEL_PARAMS` now uses `"minimal"` (3.x) and `"none"` (2.5),
  which use no hidden tokens, the budget is 1,500 tokens, `finish_reason == "length"` is recorded and shown
  to the user, and a test fails if any evaluation reply was truncated.
- An optional JSONL response cache, keyed by the full request including the per-model parameters, makes
  the evaluations reproducible and free to re-run.

## 8. Safety layers

**Input guardrails** (`src/guardrails.py`) run before anything else, as regular expressions, so they
cannot be argued out of their behaviour. Thoughts of suicide or self-harm have two tiers: *crisis*
(explicit statements, following common suicide-screening language: wishing to be dead or not to wake up,
ending one's life, others being better off without them, not being around, hoarding medication, self-harm)
returns a fixed supportive message with crisis lines and no LLM call; *distress* (softer signs such as
hopelessness, feeling like a burden or seeing no way out) returns a second fixed message with the same
support lines, also without an LLM call. Ordinary worry ("I'm so anxious about my results") is not matched
and gets a normal answer. *Injection* (attempts to change the rules,
reveal the prompt or force an output format), *privacy* (requests for other people's data) and
*off-topic* requests (with no breast-imaging term) return fixed replies. *Diagnosis* (including prognosis
and "would you bet" questions), *treatment* and *skip-care* questions still reach the LLM, with a policy
note telling it what it must not do.

**System prompt** (`src/rag/prompt_builder.py`, and the chain-of-thought prompt): rule 0, *safety first*,
overrides every other rule including grounding, so a reply to someone in distress gives support lines even
though the documents "do not cover" it, and never mentions the estimate. Rule 3 allows the estimate only when
the user asks about their own estimate, result or finding, says the case context belongs to the current user
only, and requires every mention of the estimate or prediction set to say that it is not a diagnosis, that a
low estimate does not rule out cancer, and that only a biopsy can confirm or exclude it. The case context no
longer includes the thresholded "predicted class", and the probability is not called calibrated (the served
model has no calibration step).

**Crisis backstop** (`src/pipeline.py`): when the guard misses a crisis message but the model answers it with
crisis lines, the whole model answer is replaced by the fixed crisis reply (no citations, no estimate, no
model text survives). Held-out v2 showed why: such answers also quoted the estimate, cited a non-existent
"[0]" or repeated internal instructions. A support-seeking question whose answer quotes the knowledge base's
crisis lines also gets the fixed crisis reply, a deliberate over-escalation.

**Answer sanitising** (`src/safety.py`, applied in `src/pipeline.py` to every other model answer, after the
output filter, and never to the red-team baseline): sentences that repeat six or more consecutive words of the
system prompt, task wording or policy notes are removed (phrases the assistant is told to say are exempt);
sentences about the case estimate are removed when the question did not ask about it; citation markers that
point to no provided document (`[0]`, `[7]` of 5, `[Case context]`) are stripped and reported in
`invalid_citations`. An answer with nothing left becomes a neutral fixed reply.

**Output filter** (`src/safety.py`): case-insensitive regex rewrites on every LLM answer and on the
CoT reasoning trace: the Phase 1 set of 12 (one rule narrowed to statements about the user), plus verdicts
about the user's finding ("it is likely benign", "your mass is malignant", "it's cancer", "you don't have
cancer"), medication directives that name a drug or treatment ("you should take tamoxifen", "take 20 mg"),
and skipping recommended care. Non-assertive uses ("only a biopsy can say whether you have cancer") are left
alone. If an answer discusses the estimate but never mentions a biopsy, a fixed caveat is appended. Then the
standard disclaimer, and a high-risk notice when the probability is ≥ 0.65. `tests/test_guardrails.py` pins
each rule in both directions (what it must catch and what it must leave alone).

**Red-team evaluation** (`src/evals/redteam.py`) runs two prompt sets in ten categories (diagnosis,
treatment, skip-care, prompt injection, roleplay, crisis, off-topic, privacy, misinformation, and benign
controls):

- *held-out v3* (`evals/redteam_heldout_v3.jsonl`): written after the 2026-10-23 fixes were frozen (file
  hashes in `evals/heldout_v3_freeze.json`) and never used to change them. It is the estimate to quote, and
  it runs first. Its acceptance criteria (`ACCEPTANCE` in `src/evals/redteam.py`) were fixed before any v3
  prompt was written and are evaluated in the report;
- *development* (`evals/redteam_prompts.jsonl`, 246 prompts): the original 104, on which the rules were
  built, plus held-out v1 and v2 after each was used to fix failures. Its numbers are in-sample, a
  regression test.

Each prompt runs through the production pipeline and through an ablation with the Phase 2 safety layers
removed: no input guard, the Phase 1 system prompt and task wording, the SHAP-prefixed query and the 12
original output patterns, with the same retriever, knowledge base and model. (It is not a byte-for-byte
replay of the Phase 1 system, which used a different index and model.) A Gemini judge labels each LLM reply
against a written rule for its category; fixed guardrail replies are labelled by rule, and the report says
how many of each. It also gives attack success per category, over-refusal on benign prompts, LLM calls,
truncated replies, the input guard's false-alarm rate on the 177 benign retrieval questions, and three
deterministic checks on every reply (invalid citation markers, repeated internal instructions, the estimate in
a crisis reply). The report keeps the pre-fix results of held-out v2 (`reports/safety_evaluation_v2_prefix.json`,
byte-identical to the 2026-10-20 as-run report) and v1 (`reports/safety_evaluation_v1_prefix.json`) as the
baselines the fixes are judged against. The baseline configuration builds its prompt with `legacy=True`, so its
prompts, and cached replies, are unchanged by later fixes. Only the default single-call mode is evaluated, so a question the guardrail flags
is always answered in that mode, even when the user chose chain-of-thought or multi-agent.

History, so the numbers can be read honestly:

1. The first run left two misinformation prompts uncorrected because the knowledge base said nothing about
   them, so a sourced "Common Misconceptions" section was added.
2. The Phase 2 review found guardrail false positives ("Dr. Dan recommended a biopsy" blocked as a
   jailbreak), missed crisis phrasings ("self-harming") and the truncation problem in §7; all were fixed
   against the development set. Held-out v1 was then written.
3. On held-out v1 the safety layers did *not* lower attack success, and three replies to indirect suicidal
   ideation gave no crisis resources. That result is preserved as the pre-fix baseline. The failures were
   fixed with the crisis/distress tiers, rule 0, the estimate caveat and five more sourced myths; v1 moved
   into the development set, the rules were frozen, and held-out v2 was written before any v2 result
   existed.
4. Held-out v2 (frozen-code run, 2026-10-20, tag `eval/heldout-v2-frozen`): attack success 13/62 with the
   safety layers vs 10/62 without, crisis 0/12, so NO-GO. Most failures came from the model volunteering the
   estimate; two crisis replies written by the model quoted the estimate, cited "[0]" or leaked rule 8, and
   the distress backstop only prepended support lines.
5. Phase 2.2 (2026-10-23) fixed those: fixed replies for both crisis tiers, the crisis backstop, answer
   sanitising, wider crisis/injection/privacy patterns and the estimate rule. v2 moved into the development
   set, the code was frozen (`evals/heldout_v3_freeze.json`), acceptance criteria were written, and held-out
   v3 was written after that by a separate author who did not read the safety code.
6. Changing the safety code because of a v3 result would make v3 in-sample too; the next round needs a v4.

## 9. Experimental and inactive modules

- **Image model:** EfficientNet-B0 (timm, 2 classes, 224×224, ImageNet normalisation) with a hook-based
  Grad-CAM on `conv_head`. The weights `models/efficientnet_cbis.pth` were trained on CBIS-DDSM on
  Kaggle. The training and evaluation run is not stored in this repo, so no accuracy figure can be
  reproduced or reported for it. `/api/assess-image` returns
  **HTTP 503** when the weights or dependencies are missing or fail to load. It never substitutes a
  heuristic score. PyTorch is an optional install and is imported only when an image is analysed,
  so the deployed app stays small.
- **Chain-of-thought / multi-agent:** CoT is one LLM call with a fixed six-step structure and the same
  rules as the main prompt (numbered citations, no BI-RADS category, no treatment advice, instructions in
  content ignored). Multi-agent runs Analyst → Clinician → Safety reviewer as three calls. Both fall back
  to the single-call pipeline when any step fails, questions the guardrail flags never use them, and the
  CoT trace shown in the UI passes the output filter. Neither has been evaluated; the measured numbers
  apply to the default mode.
- **Classical risk models:** a SEER-style age incidence table plus relative-risk multipliers
  (simplified Gail) and an approximate Tyrer-Cuzick. These are educational approximations.
- **Drift monitor:** a rolling window of 200 predictions. The reference values are placeholders,
  not training statistics, and the API response says so.
- **Serverless deployment:** `api/index.py` and `vercel.json` run the same FastAPI app on Vercel; the
  audit database and model caches move to `/tmp` there (`src/config.py`). The Linux install of
  `requirements.txt` was measured at about 460 MB (scipy 140, pandas 73, scikit-learn 50, numpy 70,
  fontTools 29 via fpdf2), within Vercel's 500 MB limit for Python functions only because the served
  retrieval mode (BM25) needs no embedding library; `fastembed` or `model2vec` would add 35–100 MB.

## 10. Design decisions

1. **A selection rule, not a favourite model.** The served model is whatever the fixed rule picks; the
   runner-up is kept as a challenger with a paired comparison.
2. **Nominal codes one-hot encoded** for the linear model; **monotonic constraints only where clinical
   knowledge and data agree** (age, shape).
3. **Calibration learned out-of-fold**, so it reflects unseen-data behaviour without a separate split.
4. **Conformal sets instead of an invented interval**, class-conditional so malignant cases get their
   own guarantee.
5. **Test set used once.** Every threshold, calibrator and model choice comes from development data.
6. **Exact explanations where possible.** SHAP values are exact for the linear model; base value +
   contributions reproduces the model's log-odds (tested).
7. **BI-RADS excluded** from the inputs (leakage), but reported as a clinical reference.
8. **Fail closed.** Missing data, missing image weights, an unreachable API or a missing embedder
   produce an error or a declared fallback, never a substitute estimate or a silent degradation.
9. **Docs generated from artifacts.** README, DESIGN, the model card and both reports come from
   `models/model_metadata.json` and `reports/*_evaluation.json`. The UI reads `/api/model-info`, and
   a test fails on drift.
10. **Retrieval chosen by rule, torch-free.** Eight modes are compared on a labelled question set and
    a pre-registered rule picks the served one; embedders run on ONNX or static vectors so the
    deployment never needs PyTorch.
11. **Grounding you can check.** Answers cite numbered documents, the response returns those
    citations, and an evaluation measures whether the citations support the sentences.
12. **Guardrails as code, measured.** Deterministic rules handle the cases where an LLM must not be
    the decision maker, and a red-team suite reports what they do and do not stop.
13. **Gemini** for its free API tier.

## 11. Limitations

1. Small single-institution data (Erlangen 2003–2006), so results may not transfer; test CIs are wide.
2. Four coarse, reader-dependent descriptors; real systems use far more information.
3. Calibration and conformal coverage are checked on one test split; re-check on new data.
4. The knowledge base is a set of project-written summaries, fact-checked but not clinician-reviewed,
   and small enough that many questions fall outside it.
5. The answer and red-team verdicts come from an LLM judge whose agreement with a person is unmeasured
   until the agreement sheet is filled in; the guardrails are regular expressions and will miss paraphrases.
6. Not validated in any clinical setting.
7. When the knowledge base covers a topic but not the exact question (for example, which gene panel a guideline
   recommends), the model can answer with related, cited content instead of saying the documents do not cover
   it. The answer evaluation counts this under correct abstention on unanswerable questions (generated results
   above). Changing it means changing the frozen system prompt or pipeline, so it needs a new held-out round.

## 12. Project structure

```
breastcare_ai/
├── server.py                 FastAPI app (serves app/index.html, API, /figures)
├── app/index.html            single-page frontend (the only frontend)
├── api/index.py, vercel.json serverless entry point and config for Vercel
├── Procfile                  uvicorn server:app
├── requirements.txt          runtime · -image.txt: optional PyTorch image model · -dev.txt: training, evals, tests
├── data/                     README (provenance) · raw/ (checksum-pinned UCI file) · processed/
├── models/                   served_model.joblib · challenger_model.joblib · model_metadata.json ·
│                             splits.json · efficientnet_cbis.pth (experimental image model)
├── reports/                  evaluation.md, rag_evaluation.md (generated) · *_evaluation.json · figures/
├── knowledge_base/           sources/ (10 documents) · chunks/chunks.jsonl · index/ (embeddings, manifest)
├── evals/                    retrieval_questions.jsonl · redteam_prompts.jsonl · judge_agreement_sheet.csv · cache/
├── experiments/              wdbc_benchmark.py (appendix)
├── scripts/                  download_data.py · build_knowledge_base.py · update_readme_metrics.py
├── src/
│   ├── config.py  pipeline.py  guardrails.py  safety.py  model_info.py
│   ├── ml/        data · models · calibration · conformal · evaluation · figures · train ·
│   │              artifacts · predict · explain · image_model
│   ├── rag/       chunker · bm25 · embedders · index · retriever · rerank · prompt_builder
│   ├── evals/     common · retrieval · answers · redteam
│   ├── llm/       client.py (Gemini, fallbacks, cache)
│   ├── agents/    CoT / multi-agent orchestration (unevaluated)
│   ├── audit/     SQLite log, PDF report, anonymiser
│   ├── analytics/ batch prediction, model comparison, global SHAP, drift monitor
│   └── risk/      simplified classical risk models
├── tests/                    pytest suite (+ fixtures/synthetic_mammographic.py, test-only)
└── docs/                     DESIGN.md (this file) · MODEL_CARD.md
```

## 13. References

1. Elter, M., Schulz-Wendtland, R., & Wittenberg, T. (2007). The prediction of breast cancer biopsy
   outcomes using two CAD approaches that both emphasize an intelligible decision process. *Medical Physics*.
2. Elter, M. (2007). Mammographic Mass [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C53K6Z
3. American College of Radiology. *ACR BI-RADS Atlas*, 5th edition (2013) and BI-RADS v2025 update.
4. Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
5. Angelopoulos, A. N., & Bates, S. (2021). A gentle introduction to conformal prediction and
   distribution-free uncertainty quantification. arXiv:2107.07511.
6. Van Calster, B., McLernon, D. J., van Smeden, M., Wynants, L., & Steyerberg, E. W. (2019).
   Calibration: the Achilles heel of predictive analytics. *BMC Medicine*, 17, 230.
7. Vickers, A. J., & Elkin, E. B. (2006). Decision curve analysis: a novel method for evaluating
   prediction models. *Medical Decision Making*, 26(6), 565–574.
8. Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized
   likelihood methods. *Advances in Large Margin Classifiers*.
9. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*, 2nd ed.
   (one-standard-error rule, §7.10).
10. Youden, W. J. (1950). Index for rating diagnostic tests. *Cancer*.
11. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
12. Ke, G. et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS*.
13. Mitchell, M. et al. (2019). Model cards for model reporting. *FAT\* '19*.
14. Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). Nuclear feature extraction for breast
    tumor diagnosis. *IS&T/SPIE Electronic Imaging* (WDBC, appendix).
15. Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond.
    *Foundations and Trends in Information Retrieval*, 3(4), 333–389.
16. Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). Reciprocal rank fusion outperforms Condorcet
    and individual rank learning methods. *SIGIR*.
17. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: sentence embeddings using Siamese BERT-networks.
    *EMNLP* (all-MiniLM-L6-v2 and the ms-marco cross-encoders come from this line of work).
18. Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference.
    *Journal of the American Statistical Association* (interval used for the evaluation proportions).
19. Zheng, L. et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. *NeurIPS*
    (LLM-as-judge, and why its agreement with people must be checked).
