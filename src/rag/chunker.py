"""
src/rag/chunker.py — Paragraph-level chunking of knowledge base source documents.
Splits at double-newline boundaries; oversized paragraphs split at sentence level.
Minimum chunk: CHUNK_MIN_LEN chars. Maximum: CHUNK_MAX_LEN chars.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    KB_SOURCES_DIR, CHUNKS_JSONL,
    CHUNK_MIN_LEN, CHUNK_MAX_LEN,
)


def _split_sentences(text: str) -> list[str]:
    """Split text at sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _chunk_paragraph(para: str) -> list[str]:
    """
    Return one or more chunks from a paragraph.
    If paragraph <= CHUNK_MAX_LEN: return as-is.
    If larger: split into sentence-level sub-chunks,
    accumulating up to CHUNK_MAX_LEN chars before cutting.
    """
    if len(para) <= CHUNK_MAX_LEN:
        return [para]

    sentences = _split_sentences(para)
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 > CHUNK_MAX_LEN and current:
            chunks.append(current.strip())
            current = sent
        else:
            current = (current + " " + sent).strip()
    if current:
        chunks.append(current)
    return chunks


def chunk_document(source_name: str, text: str) -> list[dict]:
    """
    Chunk a single document into paragraph-level chunks.

    Returns list of dicts:
        source   str  — filename without extension
        section  str  — nearest preceding heading, or ""
        text     str  — chunk content
        chunk_id str  — "{source}_{idx}"
    """
    chunks = []
    current_section = ""
    paragraphs = re.split(r"\n{2,}", text.strip())

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Detect Markdown heading → update section label
        if para.startswith("#"):
            current_section = re.sub(r"^#+\s*", "", para).strip()
            continue

        # Skip too-short fragments
        if len(para) < CHUNK_MIN_LEN:
            continue

        for sub_chunk in _chunk_paragraph(para):
            if len(sub_chunk) < CHUNK_MIN_LEN:
                continue
            idx = len(chunks)
            chunks.append({
                "source":   source_name,
                "section":  current_section,
                "text":     sub_chunk,
                "chunk_id": f"{source_name}_{idx}",
            })

    return chunks


def build_chunks() -> list[dict]:
    """
    Process all four source documents and write chunks.jsonl.
    Returns list of all chunk dicts.
    """
    CHUNKS_JSONL.parent.mkdir(parents=True, exist_ok=True)

    source_files = sorted(KB_SOURCES_DIR.glob("*.md"))
    if not source_files:
        raise FileNotFoundError(f"No .md files found in {KB_SOURCES_DIR}")

    all_chunks = []
    for path in source_files:
        source_name = path.stem
        text = path.read_text(encoding="utf-8")
        doc_chunks = chunk_document(source_name, text)
        all_chunks.extend(doc_chunks)
        print(f"  {source_name:<40} → {len(doc_chunks)} chunks")

    with open(CHUNKS_JSONL, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\n  Total chunks: {len(all_chunks)}  →  {CHUNKS_JSONL}")
    return all_chunks


if __name__ == "__main__":
    chunks = build_chunks()
