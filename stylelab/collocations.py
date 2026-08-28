"""Match the phrasing of papers, not only their vocabulary.

A vocabulary check catches the wrong word. It does not catch the right words joined in a
way papers never join them, and it says nothing about the formulas papers rely on and a
draft omits.

Measured against 6.0 million words: of the 60 most common connective formulas in
published work, five drafts used four. "due to the" occurs 729 times per million in
papers, "as shown in" 689, "the presence of" 601, "in order to" 380. The drafts wrote
around every one of them.

That inverts the usual advice. The humanization literature says to strip stock phrases,
and academic prose is built out of stock phrases. It simply uses a different set from the
one those lists target: papers say "attributed to the" and "than that of", not "delve
into" and "a testament to". Avoiding formula altogether is itself a departure from the
register.

Build the reference with scripts/build_ngrams.py, then call check_collocations().
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .features import FUNCTION_WORDS, words

DEFAULT_PATH = "results/academic_ngrams.json"
_CACHE: dict = {}


def load_ngrams(path: str = DEFAULT_PATH) -> dict:
    if path in _CACHE:
        return _CACHE[path]
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    _CACHE[path] = data
    return data


def _is_connective(gram: str, fw: set) -> bool:
    """A discourse formula is mostly function words holding one content word together."""
    parts = gram.split()
    return sum(w in fw for w in parts) >= 2 and all(len(w) < 14 for w in parts)


def missing_formulas(text: str, top_n: int = 60,
                     path: str = DEFAULT_PATH) -> list[tuple[str, float]]:
    """Standard connective formulas the corpus uses heavily and this draft does not.

    Returned most-frequent first, as (phrase, occurrences per million in the corpus).
    These are suggestions rather than faults: a short draft cannot contain sixty formulas.
    A draft containing almost none of them is writing around the register.
    """
    ng = load_ngrams(path)
    if not ng:
        return []
    fw = set(FUNCTION_WORDS)
    total = ng["total_words"]
    ranked = sorted(((c, g) for g, c in ng["trigrams"].items() if _is_connective(g, fw)),
                    reverse=True)[:top_n]

    w = [x.lower() for x in words(text)]
    present = set(" ".join(g) for g in zip(w, w[1:], w[2:]))
    return [(g, 1e6 * c / total) for c, g in ranked if g not in present]


def unattested_pairs(text: str, min_count: int = 2,
                     path: str = DEFAULT_PATH) -> list[tuple[str, int]]:
    """Word pairs in the draft that never occur in the corpus.

    Pairs of two function words are skipped, as are very short tokens, because those
    combinations are grammar rather than phrasing. Technical pairs will still appear and
    are expected; a narrow topic produces collocations a broad corpus has not seen.
    """
    ng = load_ngrams(path)
    if not ng:
        return []
    fw = set(FUNCTION_WORDS)
    bi = ng["bigrams"]
    w = [x.lower() for x in words(text)]
    seen = Counter(" ".join(g) for g in zip(w, w[1:]))

    out = []
    for gram, c in seen.items():
        if c < min_count:
            continue
        a, b = gram.split()
        if len(a) < 3 or len(b) < 3 or (a in fw and b in fw):
            continue
        if bi.get(gram, 0) == 0:
            out.append((gram, c))
    return sorted(out, key=lambda t: -t[1])


def formula_coverage(text: str, top_n: int = 60, path: str = DEFAULT_PATH) -> float:
    """Share of the corpus's most common connective formulas that the draft uses."""
    ng = load_ngrams(path)
    if not ng:
        return float("nan")
    missing = missing_formulas(text, top_n=top_n, path=path)
    return (top_n - len(missing)) / top_n
