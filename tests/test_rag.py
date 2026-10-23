"""Retrieval engine: tokenizer, BM25, chunking, index integrity, search modes, citations."""
import json

import numpy as np
import pytest

from src import config
from src.rag.bm25 import BM25, tokenize
from src.rag.chunker import chunk_document, parse_sections
from src.rag.embedders import available
from src.rag.index import chunks_hash
from src.rag.prompt_builder import parse_citations, select_retrieval_query
from src.rag.retriever import resolve_mode, search

SAMPLE = """# Guide

## Margins

### Spiculated Margin
Lines radiate from the mass.

### Circumscribed Margin
Sharp border.

## Sources
- Some reference
"""


def _embedder_ready(name: str) -> bool:
    """Installed and loadable (the model is cached or downloadable). Offline machines without
    the cached model skip the dense tests instead of failing."""
    if not available(name):
        return False
    try:
        from src.rag.embedders import encode
        encode(name, ["probe"])
        return True
    except Exception:
        return False


def test_tokenizer_plural_rules():
    assert tokenize("Masses biopsies margins diagnosis status") == ["mass", "biopsy", "margin", "diagnosis", "status"]
    assert tokenize("diagnoses metastases") == ["diagnosis", "metastasis"]
    assert "the" not in tokenize("the mass")


def test_chunker_splits_long_sections_and_skips_nested_sources():
    long_para = " ".join(f"Sentence number {i} about breast density." for i in range(60))
    table = "\n".join(f"| row {i} | value {i} | another column with text |" for i in range(60))
    doc = f"# T\n\n## Long\n{long_para}\n\n## Table\n{table}\n\n## Sources\n### Guidelines\n- ref\n"
    chunks = chunk_document("t", doc)
    from src.rag.chunker import MAX_CHARS
    assert all(len(c["text"]) <= MAX_CHARS for c in chunks)
    assert not any("Sources" in c["section"] or "Guidelines" in c["section"] for c in chunks)
    long_parts = [c["text"] for c in chunks if c["section"] == "Long"]
    assert len(long_parts) > 1 and all("\n\n" not in p for p in long_parts)   # one paragraph stays one paragraph


def test_hybrid_fusion_ignores_chunks_bm25_did_not_retrieve(monkeypatch):
    import src.rag.retriever as r
    chunks = r._index()[0]
    rng = np.random.default_rng(0)
    fake = rng.normal(size=(len(chunks), 8)).astype(np.float32)
    query_vec = rng.normal(size=8).astype(np.float32)
    monkeypatch.setattr(r, "_vectors", lambda name: fake)
    monkeypatch.setattr(r, "encode", lambda name, texts: query_vec[None, :])
    q = "zebra quokka"                                  # shares no term with any chunk
    assert r._index()[2].scores(q).max() == 0
    dense_ids = [c["chunk_id"] for c in r.search(q, 5, mode="dense:potion")]
    hybrid_ids = [c["chunk_id"] for c in r.search(q, 5, mode="hybrid:potion")]
    assert hybrid_ids == dense_ids                      # no positional credit from an empty BM25 list


def test_unusable_vectors_fall_back_to_bm25(monkeypatch):
    import src.rag.retriever as r
    monkeypatch.setattr(r, "_vectors", lambda name: None)
    assert r.resolve_mode("hybrid:minilm") == "bm25"
    assert r.search("spiculated margin", 2, mode="dense:minilm")[0]["retrieval_mode"] == "bm25"


def test_reranker_reorders_the_pool(monkeypatch):
    import src.rag.rerank as rr
    import src.rag.retriever as r

    class FakeCrossEncoder:
        def rerank(self, query, passages):             # prefer passages that mention "density"
            return [float("density" in p.lower()) for p in passages]

    monkeypatch.setattr(rr, "_model", lambda: FakeCrossEncoder())
    monkeypatch.setattr(rr, "available", lambda: True)
    out = r.search("what does high density mean for a mass", 3, mode="bm25+rerank")
    assert out[0]["retrieval_mode"] == "bm25+rerank" and "density" in (out[0]["section"] + out[0]["text"]).lower()
    assert [c["rank"] for c in out] == [1, 2, 3]


def test_bm25_ranks_the_matching_document_first():
    bm25 = BM25(["spiculated margin radiating lines", "fat containing mass benign", "breast density categories"])
    assert int(np.argmax(bm25.scores("what is a spiculated margin"))) == 0
    assert bm25.scores("zebra")[0] == 0


def test_chunker_keeps_heading_paths_and_skips_sources():
    sections = parse_sections(SAMPLE)
    assert (["Guide", "Margins", "Spiculated Margin"], "Lines radiate from the mass.") in sections
    chunks = chunk_document("guide", SAMPLE)
    assert [c["section"] for c in chunks] == ["Margins > Spiculated Margin", "Margins > Circumscribed Margin"]
    assert chunks[0]["index_text"].startswith("Guide > Margins > Spiculated Margin:")
    assert all("Sources" not in c["section"] for c in chunks)


def test_index_manifest_matches_the_chunks():
    with open(config.CHUNKS_JSONL, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    manifest = json.loads(config.KB_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["chunks_sha256"] == chunks_hash(chunks)
    assert {c["source"] for c in chunks} == {p.stem for p in config.KB_SOURCES_DIR.glob("*.md")}
    for name, info in manifest["embedders"].items():
        assert np.load(config.KB_INDEX_DIR / info["file"]).shape == (len(chunks), info["dim"])


@pytest.mark.parametrize("mode", ["bm25", "dense:minilm", "hybrid:minilm", "dense:potion", "hybrid:potion"])
def test_search_modes(mode):
    if mode != "bm25" and not _embedder_ready(mode.split(":")[1]):
        pytest.skip(f"{mode}: embedder not installed or model not cached (offline)")
    out = search("What does a spiculated margin mean?", 3, mode=mode)
    assert len(out) == 3 and out[0]["retrieval_mode"] == mode
    assert [c["rank"] for c in out] == [1, 2, 3]
    assert "index_text" not in out[0]


def test_source_filter_and_unknown_mode():
    out = search("high probability notice", 2, source="safety_disclaimer")
    assert out and all(c["source"] == "safety_disclaimer" for c in out)
    with pytest.raises(ValueError):
        resolve_mode("telepathy")


def test_citation_parsing():
    assert parse_citations("Yes [1]. Also [3][2]; not [7].", 3) == {"cited": [1, 2, 3], "invalid": [7]}
    assert parse_citations("No citations.", 3) == {"cited": [], "invalid": []}
    assert parse_citations("Lists [1, 2] and ranges [2-3] count too.", 3) == {"cited": [1, 2, 3], "invalid": []}
    assert parse_citations("Zero [0] and [12] are invalid.", 4)["invalid"] == [0, 12]


def test_retrieval_query_strategy():
    expl = {"dominant_feature": "margin", "risk_direction": "elevated", "shap_text": "margin (Spiculated)"}
    assert select_retrieval_query(expl, "What is a cyst?", "question") == "What is a cyst?"
    assert "margin" in select_retrieval_query(expl, "What is a cyst?", "shap_prefix")
    assert "margin" in select_retrieval_query(expl, None, "question")   # explanation request
