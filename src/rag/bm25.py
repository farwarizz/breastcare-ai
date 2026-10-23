"""
src/rag/bm25.py — Okapi BM25 keyword retrieval (no dependencies beyond numpy).

Tokens: lower-cased alphanumeric words, common English stop words removed, and
a light plural strip ("margins" → "margin", "masses" → "mass", "biopsies" →
"biopsy"). IDF uses the non-negative form
log(1 + (N − df + 0.5) / (df + 0.5)).
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does for from had has have how i if in
into is it its me my no not of on or our should so than that the their them then there these
they this to was we were what when where which who why will with would you your about
""".split())


# Greek-derived plurals the suffix rules below would get wrong (diagnoses → diagnose).
IRREGULAR = {"diagnoses": "diagnosis", "metastases": "metastasis", "analyses": "analysis",
             "prognoses": "prognosis", "axes": "axis", "neoplasms": "neoplasm"}


def tokenize(text: str) -> list[str]:
    tokens = []
    for t in re.findall(r"[a-z0-9]+", text.lower()):
        if t in STOPWORDS:
            continue
        if t in IRREGULAR:
            t = IRREGULAR[t]
        elif len(t) > 3:
            if t.endswith("sses"):                       # masses → mass
                t = t[:-2]
            elif t.endswith("ies") and len(t) > 4:      # biopsies → biopsy
                t = t[:-3] + "y"
            elif t.endswith("s") and not t.endswith(("ss", "us", "is")):
                t = t[:-1]                              # margins → margin
        tokens.append(t)
    return tokens


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d)) for d in documents]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avg_len = float(self.lengths.mean()) if len(self.docs) else 0.0
        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.docs))
        norm = self.k1 * (1 - self.b + self.b * self.lengths / (self.avg_len or 1.0))
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([d.get(term, 0) for d in self.docs], dtype=float)
            out += idf * tf * (self.k1 + 1) / (tf + norm)
        return out
