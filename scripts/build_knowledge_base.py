"""
scripts/build_knowledge_base.py — Chunks all source documents and builds the FAISS index.
Run:  python scripts/build_knowledge_base.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.chunker         import build_chunks
from src.rag.build_vectorstore import build_vectorstore

print("=" * 60)
print("BreastCare AI — Knowledge Base Builder")
print("=" * 60)

print("\nStep 1: Chunking source documents...")
chunks = build_chunks()

print(f"\nStep 2: Building FAISS vectorstore ({len(chunks)} chunks)...")
build_vectorstore(chunks)

print("\nKnowledge base ready.")
print("=" * 60)
