"""Build the reference bands that ship inside the skill.

The bands are computed with check.py's own measure() so that a draft and its target
are the same quantity. The full pipeline in stylelab/ counts passive voice from a
dependency parse and clauses from dependency labels; check.py approximates both with
regular expressions. A band from one applied to a number from the other is not a
comparison, which is what an earlier build did.

Excerpts are 1200 words, one per paper, taken from the start of the cleaned body, so
the bands describe manuscript-length prose and not whole papers. Over a third of the
measures drift with length.

    python scripts/build_skill_reference.py
"""
from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / ".claude/skills/writing-style"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from stylelab.corpus import clean_text  # noqa: E402

spec = importlib.util.spec_from_file_location("check", SKILL / "check.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)

EXCERPT = 1200
MIN_WORDS = 600
KEYS = [k for k, _ in check.SHOWN]


def excerpt(text: str) -> str | None:
    w = text.split()
    if len(w) < MIN_WORDS:
        return None
    return " ".join(w[:EXCERPT])


def pmc_docs() -> list[str]:
    out = []
    with open(ROOT / "data/human_pmc_sim.jsonl", encoding="utf-8") as fh:
        for line in fh:
            e = excerpt(clean_text(json.loads(line)["text"]))
            if e:
                out.append(e)
    return out


def group_docs() -> list[str]:
    from pypdf import PdfReader
    from build_personal_reference import clean_pdf_text
    out = []
    for pdf in sorted((ROOT / "data/group_pdfs").glob("*.pdf")):
        try:
            raw = "\n".join((pg.extract_text() or "") for pg in PdfReader(str(pdf)).pages)
        except Exception:  # noqa: BLE001
            continue
        w = clean_pdf_text(raw).split()
        # Several excerpts per paper, since there are only 19 of them.
        for start in range(0, len(w), EXCERPT):
            piece = w[start:start + EXCERPT]
            if len(piece) < EXCERPT // 2:
                break
            out.append(" ".join(piece))
    return out


def percentile(vals: list[float], q: float) -> float:
    vals = sorted(vals)
    k = (len(vals) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def build(docs: list[str], source: str) -> dict:
    feats = [check.measure(d) for d in docs]
    ref = {"n_documents": len(docs), "source": source,
           "median_words": statistics.median(len(d.split()) for d in docs),
           "measured_with": "check.py measure()", "features": {}}
    for k in KEYS:
        vals = [f[k] for f in feats]
        ref["features"][k] = {
            "mean": statistics.fmean(vals), "sd": statistics.stdev(vals),
            **{f"p{int(q*100):02d}": percentile(vals, q) for q in (.05, .25, .50, .75, .95)}}
    return ref


def main() -> int:
    for name, docs, source in (("reference.json", pmc_docs(), "615 adsorption and simulation papers, PMC OA"),
                               ("group_reference.json", group_docs(), "19 papers by one research group")):
        ref = build(docs, source)
        (SKILL / "data" / name).write_text(json.dumps(ref, indent=1), encoding="utf-8")
        print(f"{name}: {ref['n_documents']} excerpts, median {ref['median_words']:.0f} words")
        for k in KEYS:
            f = ref["features"][k]
            print(f"  {k:26s} {f['p25']:8.2f} - {f['p75']:<8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
