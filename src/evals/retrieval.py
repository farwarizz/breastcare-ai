"""
src/evals/retrieval.py — Retrieval evaluation on a labelled question set.

Question set: evals/retrieval_questions.jsonl, one JSON object per line:
  {"id", "question", "kind", "relevant": [{"source": doc, "section": heading path}]}
A retrieved chunk is relevant when it comes from a listed document and its
section equals the listed section or sits under it. Questions with an empty
"relevant" list are unanswerable from the knowledge base; they are skipped here
and used by the answer-quality evaluation instead.

Two experiments:
1. Modes: every retrieval mode answers the bare question. Metrics: Hit@1/3/5,
   MRR@10 and nDCG@5 (binary relevance), with 95% bootstrap CIs over questions.
2. Query construction: the pipeline used to prepend the SHAP summary of a
   representative case to every user question. Both variants are compared on
   the chosen mode, and SHAP-only queries (explanations without a user question)
   are scored against the sections describing the case's top-2 features.

Selection rule (fixed in advance): the highest Hit@3 (three chunks reach the
LLM); a cheaper mode wins if its Hit@3 is within 0.02 of the best.

Run:  python -m src.evals.retrieval   → reports/rag_evaluation.json
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import EVALS_DIR, EVAL_REPORTS_DIR, SHAPE_MAP, MARGIN_MAP, DENSITY_MAP
from src.ml.evaluation import Bootstrap

QUESTIONS_PATH = EVALS_DIR / "retrieval_questions.jsonl"
RESULTS_PATH = EVAL_REPORTS_DIR / "rag_evaluation.json"
# Cheapest first: bm25 needs nothing; potion is numpy + a 30 MB model; minilm needs
# ONNX Runtime + a 90 MB model; re-ranking adds an 80 MB cross-encoder.
MODES = ["bm25", "dense:potion", "hybrid:potion", "dense:minilm", "hybrid:minilm",
         "bm25+rerank", "hybrid:potion+rerank", "hybrid:minilm+rerank"]
TOLERANCE = 0.02
SECTION_FOR = {   # feature → value → knowledge-base section describing it
    "shape":   {k: f"Mass Shape > {v} Shape" for k, v in SHAPE_MAP.items()},
    "margin":  {k: f"Mass Margin > {v} Margin" for k, v in MARGIN_MAP.items()},
    "density": {1: "Mass Density > High Density", 2: "Mass Density > Iso-density",
                3: "Mass Density > Low Density", 4: "Mass Density > Fat-containing"},
}
CASES = [  # representative inputs for the query-construction experiment
    {"age": 45, "shape": 1, "margin": 1, "density": 3},
    {"age": 52, "shape": 3, "margin": 4, "density": 3},
    {"age": 66, "shape": 4, "margin": 5, "density": 3},
    {"age": 71, "shape": 4, "margin": 4, "density": 2},
    {"age": 38, "shape": 2, "margin": 3, "density": 3},
    {"age": 59, "shape": 4, "margin": 2, "density": 1},
]


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_relevant(chunk: dict, relevant: list[dict]) -> bool:
    return any(chunk["source"] == r["source"] and
               (chunk["section"] == r["section"] or chunk["section"].startswith(r["section"] + " >"))
               for r in relevant)


def scores_for(ranked: list[dict], relevant: list[dict], n_relevant: int) -> dict:
    flags = [is_relevant(c, relevant) for c in ranked]
    first = next((i + 1 for i, f in enumerate(flags) if f), None)
    dcg = sum(1 / math.log2(i + 2) for i, f in enumerate(flags[:5]) if f)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(n_relevant, 5))) or 1.0
    return {"hit@1": float(first == 1), "hit@3": float(first is not None and first <= 3),
            "hit@5": float(first is not None and first <= 5),
            "mrr@10": 1 / first if first and first <= 10 else 0.0, "ndcg@5": dcg / idcg}


def _n_relevant(chunks: list[dict], relevant: list[dict]) -> int:
    return sum(is_relevant(c, relevant) for c in chunks)


def run_mode(mode: str, items: list[dict], all_chunks: list[dict]) -> tuple[dict, list[dict], float]:
    from src.rag.retriever import search
    search(items[0]["query"], 10, mode)                 # warm-up (model loading)
    per, t0 = [], time.perf_counter()
    for it in items:
        ranked = search(it["query"], 10, mode)
        per.append(scores_for(ranked, it["relevant"], _n_relevant(all_chunks, it["relevant"])))
    latency_ms = (time.perf_counter() - t0) / len(items) * 1000
    return {k: np.array([p[k] for p in per]) for k in per[0]}, per, latency_ms


def summarise(arrays: dict, boot: Bootstrap) -> dict:
    return {k: boot.ci(lambda i, a=a: float(a[i].mean())) for k, a in arrays.items()}


def shap_query(case: dict, question: str | None) -> tuple[str, dict]:
    from src.ml.explain import explain
    from src.rag.prompt_builder import build_retrieval_query
    e = explain(case)
    return build_retrieval_query(e, question), e


def evaluate() -> dict:
    from src.rag.retriever import _index, resolve_mode
    chunks = _index()[0]
    questions = load_questions()
    answerable = [q for q in questions if q["relevant"]]
    items = [{"query": q["question"], "relevant": q["relevant"], "kind": q["kind"]} for q in answerable]
    boot = Bootstrap(len(items), 2000, 2026)

    modes, arrays_by_mode = {}, {}
    for mode in MODES:
        used = resolve_mode(mode)
        if used != mode:
            modes[mode] = {"skipped": f"unavailable here (would run as {used})"}
            continue
        arrays, _, latency = run_mode(mode, items, chunks)
        arrays_by_mode[mode] = arrays
        modes[mode] = {"metrics": summarise(arrays, boot), "latency_ms": round(latency, 1)}
        print(f"  {mode:<22} hit@3 {arrays['hit@3'].mean():.3f}  mrr {arrays['mrr@10'].mean():.3f}  "
              f"{latency:.0f} ms/query")

    hit3 = {m: float(a["hit@3"].mean()) for m, a in arrays_by_mode.items()}
    best = max(hit3, key=hit3.get)
    chosen = next(m for m in MODES if m in hit3 and hit3[m] >= hit3[best] - TOLERANCE)
    paired = {m: boot.paired(lambda i, a=arrays_by_mode[m]: float(a["hit@3"][i].mean()),
                             lambda i, b=arrays_by_mode["bm25"]: float(b["hit@3"][i].mean()))
              for m in arrays_by_mode if m != "bm25"}
    kinds = sorted({it["kind"] for it in items})
    by_kind = {m: {k: round(float(np.mean([a for a, it in zip(arrays_by_mode[m]["hit@3"], items)
                                           if it["kind"] == k])), 3) for k in kinds}
               for m in arrays_by_mode}

    # Query construction on the chosen mode: bare question vs SHAP summary + question.
    prefixed = []
    for i, q in enumerate(answerable):
        query, _ = shap_query(CASES[i % len(CASES)], q["question"])
        prefixed.append({"query": query, "relevant": q["relevant"], "kind": q["kind"]})
    arr_prefixed, _, _ = run_mode(chosen, prefixed, chunks)
    query_construction = {
        "bare_question": summarise(arrays_by_mode[chosen], boot),
        "shap_prefix_plus_question": summarise(arr_prefixed, boot),
        "hit3_difference_bare_minus_prefixed": boot.paired(
            lambda i: float(arrays_by_mode[chosen]["hit@3"][i].mean()),
            lambda i: float(arr_prefixed["hit@3"][i].mean())),
    }

    # SHAP-only queries: do they find the sections describing the top-2 features?
    explanation_items = []
    for case in CASES:
        query, e = shap_query(case, None)
        top = sorted(e["feature_attributions"], key=lambda f: -abs(e["feature_attributions"][f]))
        rel = [{"source": "birads_terminology", "section": SECTION_FOR[f][case[f]]}
               for f in top if f in SECTION_FOR][:2]
        explanation_items.append({"query": query, "relevant": rel, "kind": "explanation"})
    arr_expl, _, _ = run_mode(chosen, explanation_items, chunks)

    results = {
        "n_questions": len(questions), "n_answerable": len(answerable),
        "n_unanswerable": len(questions) - len(answerable),
        "kinds": {k: sum(1 for it in items if it["kind"] == k) for k in kinds},
        "n_chunks": len(chunks),
        "chunks_sha256": _index()[1]["chunks_sha256"],   # the exact index these numbers describe
        "modes": modes, "selection": {
            "rule": f"highest Hit@3; a cheaper mode wins if within {TOLERANCE} of the best "
                    f"(cost order: {', '.join(MODES)})",
            "best": best, "chosen": chosen},
        "paired_hit3_vs_bm25": paired, "hit3_by_kind": by_kind,
        "query_construction": query_construction,
        "explanation_queries": {"n_cases": len(CASES),
                                "hit@3": round(float(arr_expl["hit@3"].mean()), 3),
                                "mrr@10": round(float(arr_expl["mrr@10"].mean()), 3)},
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"  chosen: {chosen} (best {best}) → {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    evaluate()
