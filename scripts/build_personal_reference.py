"""Build a personal style reference from an author's own papers.

STYLE.md says a writer's own habits beat a corpus median, and this is how that gets
made. The generic PMC profile is a stand-in for a personal one; where the author's own
published work is available, it is the better target by a wide margin.

PDF text needs harder cleaning than JATS XML. Extraction interleaves running heads,
page numbers, figure captions and equation fragments with the prose, and none of that
is writing. Equations are the worst offender: a line of symbols contributes no words but
wrecks sentence segmentation, so lines that are mostly non-alphabetic are dropped.

Usage:
    python scripts/build_personal_reference.py --pdf-dir data/group_pdfs --name group
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stylelab.corpus import clean_text  # noqa: E402
from stylelab.features import extract_features, load_spacy  # noqa: E402

# Everything from here on is not prose the author composed as argument.
_RE_TAIL = re.compile(
    r"\n\s*(?:REFERENCES?|Bibliography|ACKNOWLEDG(?:E)?MENTS?|Supporting Information"
    r"|AUTHOR INFORMATION|Notes\s*\n)\b", re.IGNORECASE)
_RE_CAPTION = re.compile(r"^\s*(?:Figure|Fig\.|Table|Scheme|Eq\.?)\s*\d+", re.IGNORECASE)
_RE_HEADER = re.compile(
    r"^\s*(?:https?://|doi:|DOI|Journal of|The Journal of|Langmuir|ACS |Received|"
    r"Published|Accepted|Revised|Downloaded|\d+\s*$|[A-Z][a-z]+ et al\.)", re.IGNORECASE)
_RE_HYPHEN_BREAK = re.compile(r"([a-z])-\s*\n\s*([a-z])")
# Publisher boilerplate that PDF extraction interleaves with the prose: copyright and
# licence lines, permission notices, download stamps, peer-review watermarks, running
# heads. None of it is writing, and left in it becomes "attested" phrasing.
_RE_BOILERPLATE = re.compile(
    r"this journal is|royal society of chemistry|american chemical society|all rights reserved|"
    r"with permission from|reproduced from ref|adapted from ref|copyright \d{4}|\(c\) \d{4}|"
    r"downloaded (?:on|from|by)|for peer review|peer review of|creative commons|licensed under|"
    r"this article is|published on \w+ \d|see https?://|\bdoi\b|department of|university of|"
    r"school of|institute of|corresponding author|e-?mail|received:|accepted:|revised:|"
    r"supporting information|electronic supplementary|\bissn\b|\bvol\.? \d|\bpp\.? \d|"
    r"cite this|citation:|author contributions|conflicts? of interest|acknowledg", re.IGNORECASE)
# Some PDF producers emit fi/fl/ff as a ligature glyph preceded by a stray space, so
# the raw text holds "di <ff>erent" rather than "different". Unicode normalisation then
# expands the glyph and yields "di fferent".
#
# The space is not always spurious. "the <fi>ber" is two words and must stay two. A
# naive join produced thefiber, thefluid and thefirst at the top of a term list, so the
# decision is delegated to the corpus: join only when the joined form is attested and
# the left fragment is not itself a common word.
_RE_LIGATURE_SPACE = re.compile(
    r"([A-Za-z]{1,6})\s+([\ufb00-\ufb06][a-z]{1,14})")

_LIG_MAP = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
            "\ufb03": "ffi", "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st"}
_VOCAB_CACHE: set[str] = set()


def _corpus_vocab() -> dict:
    """Word rates per million in the reference corpus, for judging a candidate join.

    Rates rather than membership. The corpus itself contains the artefacts this repair
    exists to remove, so "speci" and "modi" are present in it and a membership test
    rejects every correct join. Their rates give them away instead: 0.0 and 0.0 per
    million against 84,836 for "the".
    """
    global _VOCAB_CACHE
    if _VOCAB_CACHE:
        return _VOCAB_CACHE
    import json
    f = Path("results/academic_vocab.json")
    if f.exists():
        d = json.loads(f.read_text(encoding="utf-8"))
        n = d["total_words"]
        _VOCAB_CACHE = {w: 1e6 * c / n for w, c in d["freq"].items()}
    return _VOCAB_CACHE


def _join_ligature(m) -> str:
    left, right = m.group(1), m.group(2)
    expanded = _LIG_MAP.get(right[0], right[0]) + right[1:]
    vocab = _corpus_vocab()
    if not vocab:
        return m.group(0)
    joined = (left + expanded).lower()
    # Join when the result is an attested word and the left piece is too rare to be a
    # word in its own right. 500 per million separates the artefacts cleanly: "di" sits
    # at 45, "speci" and "modi" at 0, while "the" is at 84,836 and "of" comparable.
    if vocab.get(joined, 0.0) > 0.0 and vocab.get(left.lower(), 0.0) < 500.0:
        return left + right
    return m.group(0)


# Characters that mark a line as equation debris. Deliberately excludes digits: an
# alphabetic-ratio filter drops number-dense prose along with the equations, and numeric
# density is the single feature this project most needs to measure correctly. "adsorbed
# 4.2 mmol/g at 298 K" and "∫∂ρ/∂r dr = α²" differ in symbols, not in digits.
_MATH_CHARS = set("=+<>∑∫∂∇√∞≈≠≤≥±×⋅→←⇒αβγδεζηθλμνξπρστφχψωΓΔΘΛΞΠΣΦΨΩ|^_{}[]\\")


def math_ratio(line: str) -> float:
    """Share of characters that belong to mathematical notation rather than prose."""
    if not line.strip():
        return 1.0
    return sum(c in _MATH_CHARS for c in line) / len(line)


def alpha_ratio(line: str) -> float:
    """Share of characters that are letters, spaces or digits.

    Digits count as prose here. A methods sentence reporting measurements is prose; only
    lines that are mostly notation are not.
    """
    if not line.strip():
        return 0.0
    return sum(c.isalnum() or c.isspace() for c in line) / len(line)


def clean_pdf_text(raw: str, min_alpha: float = 0.80) -> str:
    """Strip PDF furniture and equations, keep argued prose."""
    m = _RE_TAIL.search(raw)
    if m:
        raw = raw[:m.start()]
    raw = _RE_HYPHEN_BREAK.sub(r"\1\2", raw)
    raw = _RE_LIGATURE_SPACE.sub(_join_ligature, raw)

    kept = []
    for line in raw.split("\n"):
        s = line.strip()
        if len(s) < 25:
            continue
        if _RE_CAPTION.match(s) or _RE_HEADER.match(s) or _RE_BOILERPLATE.search(s):
            continue
        # Equation and symbol debris. Judged by notation density, not by letter count,
        # so that quantity-bearing sentences survive.
        if math_ratio(s) > 0.06 or alpha_ratio(s) < min_alpha:
            continue
        kept.append(s)

    # PDF extraction breaks lines mid-sentence, so rejoin and let sentence splitting
    # work on continuous text rather than on layout artefacts.
    return clean_text(" ".join(kept))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf-dir", required=True)
    ap.add_argument("--name", required=True, help="label for the reference file")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--min-words", type=int, default=1500)
    ap.add_argument("--excerpt-words", type=int, default=0,
                    help="split each paper into excerpts of this length instead of "
                         "profiling whole documents. Over a third of features drift "
                         "with measured length, so a reference built from 7000 word "
                         "papers does not describe an 800 word draft. Match this to the "
                         "length you intend to score.")
    args = ap.parse_args()

    try:
        from pypdf import PdfReader
    except ImportError:
        print("needs pypdf: pip install pypdf", file=sys.stderr)
        return 1

    nlp = load_spacy()
    docs = []
    for pdf in sorted(Path(args.pdf_dir).glob("*.pdf")):
        try:
            raw = "\n".join((pg.extract_text() or "") for pg in PdfReader(str(pdf)).pages)
        except Exception as exc:  # noqa: BLE001
            print(f"  {pdf.name}: extraction failed ({type(exc).__name__})")
            continue
        text = clean_pdf_text(raw)
        n = len(text.split())
        if n < args.min_words:
            print(f"  {pdf.name}: only {n} words after cleaning, skipped")
            continue
        if args.excerpt_words:
            # Split by word count, not by paragraph. clean_pdf_text joins the extracted
            # lines into continuous prose because PDF extraction breaks sentences across
            # layout lines, so there are no paragraph boundaries left to split on. The
            # consequence is that paragraph-shape features cannot be profiled from PDFs
            # at all; they come out undefined and are excluded downstream.
            w = text.split()
            k = 0
            for start in range(0, len(w), args.excerpt_words):
                piece = w[start:start + args.excerpt_words]
                if len(piece) < args.excerpt_words // 2:
                    break
                k += 1
                docs.append((f"{pdf.name}#{k}", " ".join(piece)))
            print(f"  {pdf.name}: {n} words -> {k} excerpts")
        else:
            docs.append((pdf.name, text))
            print(f"  {pdf.name}: {n} words kept")

    if len(docs) < 3:
        print("need at least 3 usable papers for a stable profile", file=sys.stderr)
        return 1

    feats = [extract_features(t, nlp=nlp) for _, t in docs]
    keys = sorted(set(feats[0]))
    import numpy as np
    import statistics as _st
    ref = {
        "n_documents": len(docs),
        "median_words": float(_st.median(len(t.split()) for _, t in docs)),
        "sources": [n for n, _ in docs],
        "features": {},
    }
    for k in keys:
        vals = np.array([f[k] for f in feats if np.isfinite(f.get(k, np.nan))])
        if vals.size < 3:
            continue
        ref["features"][k] = {
            "mean": float(vals.mean()), "sd": float(vals.std(ddof=1)),
            "p05": float(np.percentile(vals, 5)), "p25": float(np.percentile(vals, 25)),
            "p50": float(np.percentile(vals, 50)), "p75": float(np.percentile(vals, 75)),
            "p95": float(np.percentile(vals, 95)),
        }
    # score.py ranks by this; with no AI corpus to compare against there are no effect
    # sizes, so every feature is weighted equally and the ranking is by raw deviation.
    ref["separating_features"] = []

    out = Path(args.outdir) / f"{args.name}_reference.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ref, indent=2), encoding="utf-8")
    print(f"\nwrote {out} from {len(docs)} papers, {len(ref['features'])} features")
    print(f"score a draft against it with:")
    print(f"  python scripts/score.py DRAFT --reference {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
