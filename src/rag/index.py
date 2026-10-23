"""
src/rag/index.py — Build the knowledge-base index.

Writes knowledge_base/chunks/chunks.jsonl, one embeddings_<name>.npy per
available embedder, and manifest.json (chunker settings, source file hashes,
embedder models and dimensions, and a hash of every chunk's id and text so the
retriever can refuse vectors that no longer line up with the chunks).
BM25 statistics are cheap and are rebuilt from the chunks at load time.

Run:  python scripts/build_knowledge_base.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import KB_INDEX_DIR, KB_MANIFEST_PATH, KB_SOURCES_DIR
from src.rag.chunker import MAX_CHARS, build_chunks
from src.rag.embedders import EMBEDDERS, available, encode


def chunks_hash(chunks: list[dict]) -> str:
    """Hash of every chunk's id and indexed text, in order: any edit to a chunk invalidates
    the stored embeddings, not only a change to the list of ids."""
    h = hashlib.sha256()
    for c in chunks:
        h.update(f"{c['chunk_id']}\x1f{c['index_text']}\x1e".encode())
    return h.hexdigest()


def build(embedders: tuple[str, ...] = tuple(EMBEDDERS)) -> dict:
    chunks = build_chunks()
    texts = [c["index_text"] for c in chunks]
    KB_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "n_chunks": len(chunks),
        "chunks_sha256": chunks_hash(chunks),
        "chunker": {"max_chars": MAX_CHARS, "contextual_header": True},
        "sources": {p.stem: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(KB_SOURCES_DIR.glob("*.md"))},
        "embedders": {},
    }
    for name in embedders:
        if not available(name):
            print(f"  {name}: {EMBEDDERS[name]['library']} not installed, skipped")
            continue
        vectors = encode(name, texts)
        np.save(KB_INDEX_DIR / f"embeddings_{name}.npy", vectors)
        manifest["embedders"][name] = {"model": EMBEDDERS[name]["model"],
                                       "library": EMBEDDERS[name]["library"],
                                       "dim": int(vectors.shape[1]),
                                       "file": f"embeddings_{name}.npy"}
        print(f"  {name}: {vectors.shape[0]} × {vectors.shape[1]} ({EMBEDDERS[name]['model']})")
    with open(KB_MANIFEST_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest
