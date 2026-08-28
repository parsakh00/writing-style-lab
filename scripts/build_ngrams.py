"""Build the academic vocabulary and n-gram references from the human corpora.

Both files are derived and regenerable, so neither is versioned. Rebuild after adding
papers:

    python scripts/build_ngrams.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stylelab.corpus import read_jsonl  # noqa: E402
from stylelab.features import words  # noqa: E402


def stream_corpora(jsonl_paths: list[str], pdf_dirs: list[str]):
    for path in jsonl_paths:
        if Path(path).exists():
            for d in read_jsonl(path):
                yield d.text
    if pdf_dirs:
        try:
            from pypdf import PdfReader
        except ImportError:
            return
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from build_personal_reference import clean_pdf_text
        for d in pdf_dirs:
            for pdf in sorted(Path(d).glob("*.pdf")):
                try:
                    raw = "\n".join((pg.extract_text() or "")
                                    for pg in PdfReader(str(pdf)).pages)
                except Exception:
                    continue
                yield clean_pdf_text(raw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", nargs="*",
                    default=["data/human_pmc800.jsonl", "data/human_pmc_sim.jsonl"])
    ap.add_argument("--pdf-dir", nargs="*", default=["data/group_pdfs"])
    ap.add_argument("--min-bigram", type=int, default=5)
    ap.add_argument("--min-trigram", type=int, default=4)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    vocab, bi, tri, n = Counter(), Counter(), Counter(), 0
    for text in stream_corpora(args.jsonl, args.pdf_dir):
        w = [x.lower() for x in words(text)]
        n += len(w)
        vocab.update(w)
        bi.update(zip(w, w[1:]))
        tri.update(zip(w, w[1:], w[2:]))

    if n == 0:
        print("no corpus found; check --jsonl and --pdf-dir", file=sys.stderr)
        return 1

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    # Counts of one are mostly typographic noise from PDF extraction, and dropping them
    # roughly halves both files without losing anything a draft would match against.
    (out / "academic_vocab.json").write_text(json.dumps(
        {"total_words": n, "freq": {w: c for w, c in vocab.items() if c >= 2}}),
        encoding="utf-8")
    (out / "academic_ngrams.json").write_text(json.dumps({
        "total_words": n,
        "bigrams": {" ".join(k): v for k, v in bi.items() if v >= args.min_bigram},
        "trigrams": {" ".join(k): v for k, v in tri.items() if v >= args.min_trigram},
    }), encoding="utf-8")

    print(f"{n:,} words")
    print(f"  vocabulary  {sum(1 for c in vocab.values() if c >= 2):,} types")
    print(f"  bigrams     {sum(1 for c in bi.values() if c >= args.min_bigram):,}")
    print(f"  trigrams    {sum(1 for c in tri.values() if c >= args.min_trigram):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
