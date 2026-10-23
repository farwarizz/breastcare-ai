"""
src/rag/chunker.py — Heading-aware chunking of the knowledge-base Markdown files.

Each chunk is the text under one Markdown heading, labelled with its heading
path ("Document > Section > Subsection"). Short paragraphs stay attached to
their heading instead of being dropped. Sections longer than MAX_CHARS are
split at paragraph boundaries, then at sentence boundaries. A trailing
"Sources" section is a bibliography and is not indexed.

For retrieval, each chunk's `index_text` is prefixed with its heading path
(a contextual chunk header), which helps short sections match queries that
use the section's topic words.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import KB_SOURCES_DIR, CHUNKS_JSONL

MAX_CHARS = 900
SKIP_SECTIONS = {"sources", "references"}
_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "section"


def parse_sections(text: str) -> list[tuple[list[str], str]]:
    """[(heading path, body text)] in document order; text before any heading is skipped."""
    path: list[str] = []
    sections: list[tuple[list[str], list[str]]] = []
    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            level, title = len(m.group(1)), m.group(2).strip()
            path = path[: level - 1] + [title]
            sections.append((list(path), []))
        elif sections and line.strip() != "---":
            sections[-1][1].append(line)
    return [(p, "\n".join(body).strip()) for p, body in sections]


def _units(paragraph: str) -> list[tuple[str, str]]:
    """Split one over-long paragraph into (unit, joiner) pairs: sentences joined by a space;
    list items or table rows (lines with no sentence break) joined by a newline; and, as a
    last resort, hard slices of MAX_CHARS."""
    if len(paragraph) <= MAX_CHARS:
        return [(paragraph, "\n\n")]
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    if all(len(s) <= MAX_CHARS for s in sentences) and len(sentences) > 1:
        return [(s, " ") for s in sentences]
    lines = [ln for ln in paragraph.splitlines() if ln.strip()]
    if len(lines) > 1 and all(len(ln) <= MAX_CHARS for ln in lines):
        return [(ln, "\n") for ln in lines]
    return [(paragraph[i:i + MAX_CHARS], "") for i in range(0, len(paragraph), MAX_CHARS)]


def _split_long(body: str) -> list[str]:
    if len(body) <= MAX_CHARS:
        return [body]
    pieces, current = [], ""
    for paragraph in (p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()):
        for j, (unit, joiner) in enumerate(_units(paragraph)):
            sep = "\n\n" if j == 0 else joiner        # paragraphs apart, a paragraph's parts together
            if current and len(current) + len(sep) + len(unit) > MAX_CHARS:
                pieces.append(current)
                current = unit
            else:
                current = f"{current}{sep}{unit}" if current else unit
    if current:
        pieces.append(current)
    return pieces


def chunk_document(doc_id: str, text: str) -> list[dict]:
    chunks = []
    for path, body in parse_sections(text):
        # Skip bibliographies, including subsections under them ("Sources > Guidelines").
        if not body or any(p.strip().lower() in SKIP_SECTIONS for p in path):
            continue
        title = path[0]
        section = " > ".join(path[1:]) if len(path) > 1 else path[0]
        for i, piece in enumerate(_split_long(body)):
            chunks.append({
                "chunk_id": f"{doc_id}::{_slug(section)}::{i}",
                "source": doc_id,
                "title": title,
                "section": section,
                "text": piece,
                "index_text": f"{' > '.join(path)}: {piece}",
            })
    return chunks


def build_chunks(sources_dir: Path = KB_SOURCES_DIR, out: Path = CHUNKS_JSONL) -> list[dict]:
    files = sorted(sources_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No .md files found in {sources_dir}")
    chunks = []
    for path in files:
        doc_chunks = chunk_document(path.stem, path.read_text(encoding="utf-8"))
        chunks += doc_chunks
        print(f"  {path.stem:<32} {len(doc_chunks):>3} chunks")
    ids = [c["chunk_id"] for c in chunks]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"duplicate chunk ids (repeated headings?): {dupes[:5]}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"  total {len(chunks)} chunks → {out}")
    return chunks
