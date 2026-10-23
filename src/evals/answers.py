"""
src/evals/answers.py — Answer-quality evaluation of the RAG assistant.

Sample: every unanswerable question in evals/retrieval_questions.jsonl plus a
fixed, kind-stratified sample of answerable ones. Each question is answered by
the production pipeline (input guard → retrieval → grounded prompt → Gemini →
output filter) in the context of a representative assessment.

Measured
- Automatic: share of answers with at least one valid citation; share citing a
  number that was not provided.
- LLM judge (JUDGE_MODEL, JSON), given the question, the numbered documents and
  the answer: faithful (every factual claim is supported by the documents),
  citations_correct (each cited document supports its sentence), and
  says_not_covered (the answer states the documents do not cover the question).
- Answerable questions: faithfulness and citation correctness.
- Unanswerable questions: correct abstention (says_not_covered) and, among
  answers that did not abstain, unsupported content.

The judge itself needs checking by a person: evals/judge_agreement_sheet.csv
holds 30 items with the judge's verdict and an empty column for a human label;
`python -m src.evals.answers --agreement` reports agreement once it is filled.

Run:  python -m src.evals.answers   → reports/answer_evaluation.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import EVALS_DIR, EVAL_REPORTS_DIR, GEMINI_MODEL, JUDGE_MODEL
from src.evals.common import CACHE_DIR, case_for, judge, wilson

RESULTS_PATH = EVAL_REPORTS_DIR / "answer_evaluation.json"
SHEET_PATH = EVALS_DIR / "judge_agreement_sheet.csv"
N_ANSWERABLE = 36
JUDGE_SYSTEM = ("You are a strict evaluator of grounded answers from a breast-imaging education chatbot. "
                "Base every judgement only on the numbered documents provided, not on your own knowledge. "
                "Output only a JSON object.")


def sample_questions() -> list[dict]:
    from src.evals.retrieval import load_questions
    questions = load_questions()
    unanswerable = [q for q in questions if not q["relevant"]]
    answerable = [q for q in questions if q["relevant"]]
    rng = random.Random(2026)
    by_kind: dict[str, list] = {}
    for q in answerable:
        by_kind.setdefault(q["kind"], []).append(q)
    picked, kinds = [], sorted(by_kind)
    while len(picked) < min(N_ANSWERABLE, len(answerable)):
        for k in kinds:
            pool = [q for q in by_kind[k] if q not in picked]
            if pool and len(picked) < N_ANSWERABLE:
                picked.append(rng.choice(pool))
    return sorted(picked, key=lambda q: q["id"]) + unanswerable


def answer(question: str, case: dict) -> dict:
    from src.pipeline import run_pipeline
    result = run_pipeline(case, question, llm_kwargs={
        "cache_path": CACHE_DIR / "answers_generation.jsonl", "wait_on_rate_limit": True})
    trace = result.get("trace") or {}
    if result.get("mode") != "guardrail" and not trace.get("model"):
        # Never score the app's quota or error fallback as the model's answer; abort and resume later.
        raise RuntimeError(f"no model answer (quota_exhausted={trace.get('quota_exhausted')}, "
                           f"error={trace.get('error')}); not scoring a fallback reply")
    return result


def judge_answer(question: str, chunks: list[dict], text: str) -> dict:
    docs = "\n\n".join(f"[{i}] ({c['source']} › {c['section']})\n{c['text']}" for i, c in enumerate(chunks, 1))
    return judge(JUDGE_SYSTEM,
                 f"Question:\n\"\"\"{question}\"\"\"\n\nNumbered documents:\n{docs}\n\nAnswer:\n\"\"\"{text}\"\"\"\n\n"
                 "Return JSON with these fields:\n"
                 '{"says_not_covered": true if the answer states the documents do not cover the question (or a '
                 'central part of it), "faithful": true only if every factual claim in the answer is supported by '
                 'the documents (statements that the documents do not cover something, and generic advice to '
                 'consult a clinician, count as supported), "unsupported_claims": [up to 3 short quotes], '
                 '"citations_correct": true if every bracketed citation points to a document that supports the '
                 'sentence it is attached to (true if there are no citations and no factual claims)}',
                 "answers_judge.jsonl")


def answers_provenance() -> dict:
    """The index and system prompt these answers were generated with, so stale reports are detectable."""
    import hashlib
    from src.rag.prompt_builder import SYSTEM_PROMPT
    from src.rag.retriever import _index
    return {"chunks_sha256": _index()[1]["chunks_sha256"],
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()}


def evaluate() -> dict:
    from src.evals.retrieval import is_relevant
    rows = []
    from src.llm.client import _cache
    gen_path, judge_path = CACHE_DIR / "answers_generation.jsonl", CACHE_DIR / "answers_judge.jsonl"
    for i, q in enumerate(sample_questions()):
        n_gen = len(_cache(gen_path))
        result = answer(q["question"], case_for(i))
        gen_fresh = len(_cache(gen_path)) > n_gen
        n_judge = len(_cache(judge_path))
        text = result["llm_response"]["main_text"]
        chunks = result.get("chunks", [])
        verdict = judge_answer(q["question"], chunks, text) if chunks else {
            "says_not_covered": False, "faithful": True, "unsupported_claims": [], "citations_correct": True}
        trace = result.get("trace") or {}
        rows.append({
            "id": q["id"], "kind": q["kind"], "answerable": bool(q["relevant"]), "question": q["question"],
            "guardrail": result.get("guardrail"), "n_chunks": len(chunks),
            # did the pipeline's retrieval find a labelled section? (separates retrieval misses from LLM errors)
            "retrieval_hit": any(is_relevant(c, q["relevant"]) for c in chunks) if q["relevant"] else None,
            "truncated": trace.get("finish_reason") == "length",
            # null when no model call was involved (a fixed guardrail reply needs no generation or judge)
            "reply_fresh": gen_fresh if result.get("mode") != "guardrail" else None,
            "judge_fresh": (len(_cache(judge_path)) > n_judge) if chunks else None,
            "model": trace.get("model"),   # the model that actually answered (fallbacks)
            "valid_citations": len(result.get("citations", [])), "invalid_citations": result.get("invalid_citations", []),
            "says_not_covered": bool(verdict.get("says_not_covered")), "faithful": bool(verdict.get("faithful")),
            "citations_correct": bool(verdict.get("citations_correct")),
            "unsupported_claims": verdict.get("unsupported_claims", [])[:3],
            "answer": text, "documents": [f"{c['source']} › {c['section']}" for c in chunks],
        })
        print(f"  {q['id']} {'A' if q['relevant'] else 'U'} faithful={rows[-1]['faithful']} "
              f"not_covered={rows[-1]['says_not_covered']} cites={rows[-1]['valid_citations']}")

    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    answered_una = [r for r in una if not r["says_not_covered"]]
    answered = [r for r in ans if not r["says_not_covered"]]
    results = {
        "generation_model": GEMINI_MODEL, "judge_model": JUDGE_MODEL,
        # Which models produced the answers after quota fallbacks (configured model first in the chain).
        "generation_models_used": dict(Counter(r["model"] or "none (quota exhausted)" for r in rows)),
        "n": len(rows), "n_answerable": len(ans), "n_unanswerable": len(una),
        "truncated_answers": sum(r["truncated"] for r in rows),
        # Where each answer and verdict came from in this run: a new API call or evals/cache/.
        "replies_fresh": sum(r["reply_fresh"] is True for r in rows),
        "replies_cached": sum(r["reply_fresh"] is False for r in rows),
        "replies_without_model": sum(r["reply_fresh"] is None for r in rows),   # fresh + cached + this = n
        "judge_verdicts_fresh": sum(r["judge_fresh"] is True for r in rows),
        "judge_verdicts_cached": sum(r["judge_fresh"] is False for r in rows),
        "judge_verdicts_none": sum(r["judge_fresh"] is None for r in rows),
        "provenance": answers_provenance(),
        "answerable": {
            # The judge counts "the documents do not cover this" as faithful, so "faithful" alone
            # flatters abstentions; "answered_faithfully" requires an actual, supported answer.
            "answered_faithfully": wilson(sum(r["faithful"] and not r["says_not_covered"] for r in ans), len(ans)),
            "faithful_when_answering": wilson(sum(r["faithful"] for r in answered), len(answered)),
            "retrieval_hit": wilson(sum(bool(r["retrieval_hit"]) for r in ans), len(ans)),
            "faithful": wilson(sum(r["faithful"] for r in ans), len(ans)),
            "citations_correct": wilson(sum(r["citations_correct"] for r in ans), len(ans)),
            "has_valid_citation": wilson(sum(r["valid_citations"] > 0 for r in ans), len(ans)),
            "cites_missing_document": wilson(sum(bool(r["invalid_citations"]) for r in ans), len(ans)),
            "abstained": wilson(sum(r["says_not_covered"] for r in ans), len(ans)),
        },
        "unanswerable": {
            "correct_abstention": wilson(sum(r["says_not_covered"] for r in una), len(una)),
            "unsupported_when_answering": wilson(sum(not r["faithful"] for r in answered_una), len(answered_una)),
        },
        "rows": rows,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        f.write("\n")
    write_agreement_sheet(rows)
    a, u = results["answerable"], results["unanswerable"]
    print(f"  answered faithfully {a['answered_faithfully']['value']} · faithful {a['faithful']['value']} · "
          f"citations correct {a['citations_correct']['value']} · abstention on unanswerable "
          f"{u['correct_abstention']['value']} · truncated {results['truncated_answers']}")
    return results


def write_agreement_sheet(rows: list[dict], n: int = 30) -> None:
    """Items for a person to label, so the judge's accuracy can be measured."""
    rng = random.Random(7)
    picked = rng.sample(rows, min(n, len(rows)))
    with open(SHEET_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "documents", "answer", "judge_faithful", "human_faithful (yes/no)"])
        for r in picked:
            w.writerow([r["id"], r["question"], " | ".join(r["documents"]), r["answer"],
                        "yes" if r["faithful"] else "no", ""])


def agreement() -> None:
    with open(SHEET_PATH, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["human_faithful (yes/no)"].strip()]
    if not rows:
        print("No human labels yet: fill the last column of", SHEET_PATH)
        return
    agree = sum(r["judge_faithful"] == r["human_faithful (yes/no)"].strip().lower() for r in rows)
    print(f"judge–human agreement on faithfulness: {agree}/{len(rows)} ({agree / len(rows):.0%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agreement", action="store_true")
    agreement() if parser.parse_args().agreement else evaluate()
