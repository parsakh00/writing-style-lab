"""Find what GPTZero's score actually tracks, by correlating it against measured features.

Five drafts were called machine-written while real papers of the same register were not,
and four hypotheses drawn from the tool's own explanations were wrong. Those explanations
are written after the fact and describe formal technical prose rather than the mechanism.
This script replaces that guesswork with a correlation over hundreds of documents.

Scores are measured, never optimised against. Nothing here feeds a revision loop.

Usage:
    python scripts/gptzero_probe.py --per-group 40 --excerpt-words 1000
    python scripts/gptzero_probe.py --dry-run          # cost only, no calls
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stylelab.corpus import clean_text, read_jsonl  # noqa: E402
from stylelab.features import extract_features, load_spacy  # noqa: E402
from stylelab.gptzero import GPTZero  # noqa: E402


def excerpt(text: str, n: int) -> str:
    w = text.split()
    return " ".join(w[:n])


def gather(data_dir: str, per_group: int, n_words: int) -> list[tuple[str, str, str]]:
    """(group, label, text) for each document to score."""
    items: list[tuple[str, str, str]] = []

    human = [d.text for d in read_jsonl(Path(data_dir) / "human_pmc.jsonl")]
    for i, t in enumerate(human[:per_group]):
        items.append(("real_paper", f"pmc:{i}", excerpt(t, n_words)))

    ai: list[str] = []
    for g in sorted(glob.glob(str(Path(data_dir) / "ai_Qwen*.jsonl"))):
        ai += [d.text for d in read_jsonl(g)]
    for i, t in enumerate(ai[:per_group]):
        items.append(("qwen_paper", f"qwen:{i}", excerpt(t, n_words)))

    for p in sorted(Path("drafts").glob("*.md")):
        raw = p.read_text(encoding="utf-8")
        items.append(("my_draft", p.name,
                      clean_text(re.sub(r"^#.*$", "", raw, flags=re.M))))
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--per-group", type=int, default=40)
    ap.add_argument("--excerpt-words", type=int, default=1000)
    ap.add_argument("--budget", type=int, default=150_000,
                    help="hard cap on words sent. The account is metered.")
    ap.add_argument("--out", default="results/gptzero_scores.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    items = gather(args.data_dir, args.per_group, args.excerpt_words)
    total = sum(len(t.split()) for _, _, t in items)
    print(f"{len(items)} documents, {total:,} words")
    for g in ("real_paper", "qwen_paper", "my_draft"):
        n = sum(1 for x, _, _ in items if x == g)
        print(f"  {g:12s} {n:4d}")
    if args.dry_run:
        print(f"\nDRY RUN. Would send {total:,} words. Nothing called, nothing spent.")
        return 0
    if total > args.budget:
        print(f"\n{total:,} words exceeds the budget of {args.budget:,}. Lower "
              f"--per-group or --excerpt-words, or raise --budget deliberately.",
              file=sys.stderr)
        return 1

    client = GPTZero(budget_words=args.budget)
    nlp = load_spacy()
    rows = []
    for i, (group, doc_id, text) in enumerate(items, 1):
        try:
            score = client.score(text)
        except Exception as exc:  # noqa: BLE001
            print(f"\n  {doc_id}: {type(exc).__name__}: {exc}")
            if i == 1:
                print("  first call failed; check the key and the response shape")
                return 1
            continue
        feats = extract_features(text, nlp=nlp)
        rows.append({"group": group, "doc_id": doc_id, **score,
                     "features": {k: v for k, v in feats.items() if v == v}})
        print(f"  {i}/{len(items)}  {client.spent_words:,} words spent", end="\r")

    print()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"wrote {args.out}: {len(rows)} scored, {client.spent_words:,} words used")

    # Immediate sanity check: does it separate the groups we already know the truth for?
    import statistics as s
    print(f"\n{'group':12s}{'n':>4s}{'median prob_ai':>16s}{'median sent>0.5':>18s}")
    for g in ("real_paper", "qwen_paper", "my_draft"):
        vals = [r for r in rows if r["group"] == g]
        if not vals:
            continue
        pa = [v["prob_ai"] for v in vals if v.get("prob_ai") is not None]
        sf = [v.get("sent_frac_over_half") for v in vals
              if v.get("sent_frac_over_half") is not None]
        print(f"{g:12s}{len(vals):4d}"
              f"{(s.median(pa) if pa else float('nan')):16.3f}"
              f"{(s.median(sf) if sf else float('nan')):18.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
