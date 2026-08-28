"""Score a draft against the human corpus distribution.

Reports, per feature, how far the draft sits from where human papers sit, in standard
deviations, ranked worst first. Then it points at the specific passages and phrases
responsible, because "your sentence rhythm is uniform" is not actionable and "sentences
14 through 21 are all between 22 and 26 words" is.

The comparison is against the human corpus. No detector is involved. Editing a draft
while watching a detector's score is a different activity from editing it to read more
like the reference corpus, and this tool does the second one.

A z score is a description, not a verdict. Plenty of good writing sits two standard
deviations from a corpus median on some axis, usually on purpose. Read the report as a
list of things to look at, not a list of things to fix.

Usage:
    python scripts/score.py draft.md
    python scripts/score.py draft.md --top 15 --show-passages 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stylelab.corpus import clean_text  # noqa: E402
from stylelab.features import (  # noqa: E402
    _TELL_PATTERNS,
    extract_features,
    feature_group,
    load_spacy,
    out_of_vocabulary,
)
from stylelab.collocations import (  # noqa: E402
    formula_coverage,
    missing_formulas,
)
from stylelab.windows import window_profile, windowed_features  # noqa: E402

# Features that drift with how much text was measured, not with how it was written.
# Measured directly: the same papers at an 800 word excerpt versus in full give MTLD 81.6
# against 59.1, and rare-phrase counts of zero against non-zero simply because a short
# text has fewer chances to contain a rare phrase. Comparing these across a length gap
# reports a difference in sample size as a difference in style.
LENGTH_SENSITIVE = {"mtld", "tell_total_rate", "tell_distinct_count", "hapax_rate",
                    "punct_quote_rate", "opener_entropy", "opener_top1_frac"}

# Features whose z score says nothing about style, only about how long the draft is.
LENGTH_FEATURES = {
    "n_words", "n_sentences", "type_count", "n_windows", "lm_n_tokens",
    "w_n_words_mean", "w_n_words_sd", "w_n_words_worst", "w_n_words_p25",
    "w_n_words_p75", "longest_uniform_run",
}


def strip_markdown(text: str) -> str:
    """Remove markup so the measures see prose, not syntax.

    Code blocks, tables, headings, list markers and link syntax would all distort
    sentence length and punctuation rates, and none of them are the prose being judged.
    """
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"~~~.*?~~~", " ", text, flags=re.S)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"^\s*\|.*\|\s*$", " ", text, flags=re.M)      # tables
    text = re.sub(r"^\s{0,3}#{1,6}\s+.*$", " ", text, flags=re.M)  # headings
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.M)          # bullets
    text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.M)        # numbered lists
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)        # links and images
    text = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", text)     # emphasis
    text = re.sub(r"\$[^$]*\$", " ", text)                        # inline maths
    return clean_text(text)


def load_reference(path: Path) -> dict:
    if not path.exists():
        print(f"{path} not found. Run scripts/analyze.py first.", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def z_scores(feats: dict[str, float], ref: dict,
             include_function: bool = False) -> list[dict]:
    """Signed distance from the human median, in human standard deviations.

    Function word features are excluded from the ranked report by default, for the same
    reason STYLE.md leaves them out: they carry real signal in the aggregate analysis
    but "use 'its' 30% less often" is not something a writer can act on, and letting
    them fill the top of the list buries the findings that are actionable. They still
    count toward the summary tally at the bottom.
    """
    # Weight by how well each feature separated the corpora, so a large deviation on a
    # feature that never distinguished anything is not reported above a moderate
    # deviation on one that did.
    weights = {r["feature"]: abs(r["cohens_d"])
               for r in ref.get("separating_features", [])}

    rows = []
    for name, value in feats.items():
        if name in LENGTH_FEATURES or name not in ref["features"]:
            continue
        stat = ref["features"][name]
        sd = stat["sd"]
        if not sd or not np.isfinite(sd) or sd == 0:
            continue
        z = (value - stat["p50"]) / sd
        if not np.isfinite(z):
            continue
        rows.append({
            "feature": name,
            "value": value,
            "human_p25": stat["p25"],
            "human_p50": stat["p50"],
            "human_p75": stat["p75"],
            "z": z,
            "weight": weights.get(name, 0.0),
            "priority": abs(z) * (0.25 + weights.get(name, 0.0)),
            "actionable": include_function or feature_group(name) != "function",
        })
    return sorted(rows, key=lambda r: r["priority"], reverse=True)


def find_tells(text: str) -> list[tuple[int, str, str]]:
    """Locate stock phrases with their line numbers."""
    lines = text.split("\n")
    hits = []
    for i, line in enumerate(lines, 1):
        for phrase, pat in _TELL_PATTERNS:
            for m in pat.finditer(line):
                start = max(0, m.start() - 35)
                context = line[start:m.end() + 35].strip()
                hits.append((i, phrase, context))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("draft")
    ap.add_argument("--reference", default="results/human_reference.json")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--show-passages", type=int, default=3)
    ap.add_argument("--no-spacy", action="store_true")
    ap.add_argument("--include-function", action="store_true",
                    help="also rank individual function word rates, which are real "
                         "signal but not something a writer can act on directly")
    ap.add_argument("--lm", action="store_true",
                    help="add GPT-2 features; slower, needs the model downloaded")
    args = ap.parse_args()

    draft_path = Path(args.draft)
    if not draft_path.exists():
        print(f"{draft_path} not found", file=sys.stderr)
        return 1

    raw = draft_path.read_text(encoding="utf-8", errors="replace")
    text = strip_markdown(raw) if draft_path.suffix.lower() in {".md", ".markdown"} else clean_text(raw)

    n_words = len(text.split())
    if n_words < 300:
        print(f"warning: {n_words} words. Below about 300 words these measures are "
              f"dominated by sampling noise and the report will not mean much.\n")

    ref = load_reference(Path(args.reference))
    nlp = None if args.no_spacy else load_spacy()

    feats = extract_features(text, nlp=nlp)
    feats.update(windowed_features(text, nlp=nlp))
    if args.lm:
        from stylelab.lm import LMScorer
        feats.update(LMScorer().load().features(text))

    print(f"{draft_path.name}: {n_words:,} words, "
          f"{int(feats['n_sentences'])} sentences")
    print(f"reference: {ref['n_documents']} human documents "
          f"({', '.join(ref['sources'])})\n")

    # A reference built from full papers does not describe a short draft. Warn before
    # the table rather than after it, since the table is what gets acted on.
    ref_len = ref.get("median_words", 0.0)
    length_gap = bool(ref_len) and not (0.5 <= n_words / ref_len <= 2.0)
    if length_gap:
        print(f"WARNING: this draft is {n_words:,} words; the reference was built from "
              f"documents of about {ref_len:,.0f}.")
        print(f"  Over a third of features drift with measured length. The "
              f"length-sensitive ones are excluded below:")
        print(f"  {', '.join(sorted(LENGTH_SENSITIVE))}")
        print("  Compare a passage of similar length, or rebuild the "
              "reference at this scale.")
        print()
    rows = z_scores(feats, ref, include_function=args.include_function)
    if length_gap:
        rows = [r for r in rows if r["feature"] not in LENGTH_SENSITIVE]
    if not rows:
        print("no comparable features; is the reference file from the same pipeline?")
        return 1

    shown = [r for r in rows if r["actionable"]]
    print(f"{'feature':40s} {'draft':>9s} {'human range':>18s} {'z':>7s}")
    print("-" * 78)
    for r in shown[:args.top]:
        rng = f"{r['human_p25']:.3g}-{r['human_p75']:.3g}"
        flag = "  <<" if abs(r["z"]) > 2 else ""
        print(f"{r['feature'][:40]:40s} {r['value']:9.3g} {rng:>18s} "
              f"{r['z']:+7.2f}{flag}")

    inside = sum(1 for r in rows if abs(r["z"]) <= 1)
    print(f"\n{inside}/{len(rows)} features within one sd of the human median")
    n_hidden = len(rows) - len(shown)
    if n_hidden:
        print(f"({n_hidden} function-word features counted above but not listed; "
              f"pass --include-function to see them)")

    # Uniform passages, located.
    profile = window_profile(text, nlp=nlp)
    if profile and args.show_passages:
        worst = sorted(profile, key=lambda w: w["w_sent_len_cv"])[:args.show_passages]
        print(f"\nmost uniform passages (lowest sentence length variation):")
        for w in worst:
            print(f"  sentence {w['start_sentence'] + 1}+  "
                  f"cv={w['w_sent_len_cv']:.2f}  mean={w['w_sent_len_mean']:.0f} words")
            print(f"    \"{w['preview']}\"")

    # Words the corpus effectively never uses. Derived from 6 million words of papers
    # rather than from a hand-written list, which is why it catches judgement vocabulary
    # ("silently", "defensible", "admits") that no phrase list would think to include.
    oov = out_of_vocabulary(text)
    if oov:
        absent = [x for x in oov if x[2] == 0.0]
        print()
        print(f"words the corpus does not use ({len(oov)} below 1 per million, "
              f"{len(absent)} absent entirely):")
        for w, c, r in oov[:12]:
            mark = "absent" if r == 0.0 else f"{r:.2f}/M"
            print(f"  {w:20s} used {c}x   corpus {mark}")
        if len(oov) > 12:
            print(f"  and {len(oov) - 12} more")
        print("  Technical terms are expected here. General-purpose words are not:")
        print("  papers quantify where these characterise.")

    # Phrasing, not only word choice. Papers are built out of connective formulas;
    # writing around all of them is itself a departure from the register.
    cov = formula_coverage(text)
    if cov == cov:
        verdict = "low" if cov < 0.07 else "in range"
        print()
        print(f"connective formula coverage: {cov:.0%} of the 60 most common "
              f"({verdict}; real papers 7-13%)")
        if cov < 0.07:
            miss = missing_formulas(text)[:8]
            print("  formulas papers use heavily and this draft does not:")
            for g, r in miss:
                print(f"    {g:24s}{r:6.0f} per million in papers")

    tells = find_tells(raw)
    if tells:
        print(f"\nstock phrases ({len(tells)} found):")
        for line_no, phrase, context in tells[:12]:
            print(f"  line {line_no}: \"{phrase}\"")
            print(f"    ...{context}...")
        if len(tells) > 12:
            print(f"  and {len(tells) - 12} more")
    else:
        print("\nno stock phrases from the list found")

    print("\nThese are descriptions, not corrections. A deviation is a place to look,")
    print("and some of them will be the right call already.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
