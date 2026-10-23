"""
scripts/build_knowledge_base.py — Chunk every source document and build the index.

Run:  python scripts/build_knowledge_base.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.rag.index import build

print("BreastCare AI — knowledge-base builder")
manifest = build()
print(f"Done: {manifest['n_chunks']} chunks from {len(manifest['sources'])} documents; "
      f"embedders: {', '.join(manifest['embedders']) or 'none (bm25 only)'}")
