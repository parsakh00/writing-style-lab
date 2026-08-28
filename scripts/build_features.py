"""Turn the corpora into a feature table.

Reads every *.jsonl shard under data/, extracts the full feature vector for each
document, and writes results/features.csv.

The language model and detector blocks are optional because they are the slow part.
Surface, lexical, punctuation, function word, discourse and tell features run in
seconds; GPT-2 scoring runs in minutes to hours depending on corpus size and whether a
GPU is present. Build the cheap table first, look at it, then add the expensive columns.

Usage:
    python scripts/build_features.py
    python scripts/build_features.py --lm --detectors
    python scripts/build_features.py --lm --device cuda --limit 200
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stylelab.corpus import load_all  # noqa: E402
from stylelab.features import extract_features, load_spacy, parse_once  # noqa: E402
from stylelab.windows import windowed_features  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results/features.csv")
    ap.add_argument("--min-words", type=int, default=300,
                    help="matches the shortest text a detector will score at all")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap documents per label, for quick iteration")
    ap.add_argument("--no-spacy", action="store_true",
                    help="skip the syntax block if the parser is unavailable")
    ap.add_argument("--lm", action="store_true", help="add GPT-2 perplexity features")
    ap.add_argument("--lm-model", default="gpt2")
    ap.add_argument("--detectors", action="store_true",
                    help="add open detector scores, for validation only")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    docs = load_all(args.data_dir, min_words=args.min_words)
    if not docs:
        print(f"no documents under {args.data_dir}/. Run the fetch scripts first.",
              file=sys.stderr)
        return 1

    if args.limit:
        capped: dict[str, int] = {}
        kept = []
        for d in docs:
            key = f"{d.label}:{d.source}"
            if capped.get(key, 0) < args.limit:
                kept.append(d)
                capped[key] = capped.get(key, 0) + 1
        docs = kept

    counts = pd.Series([f"{d.label}/{d.source}" for d in docs]).value_counts()
    print("corpus:")
    for name, n in counts.items():
        print(f"  {name:28s} {n:5d}")

    nlp = None if args.no_spacy else load_spacy()
    if nlp is None and not args.no_spacy:
        print("\nspaCy model unavailable, syntax features will be skipped.")
        print("  install with: python -m spacy download en_core_web_sm")

    scorer = None
    if args.lm:
        from stylelab.lm import LMScorer
        print(f"\nloading language model {args.lm_model}...")
        scorer = LMScorer(model_name=args.lm_model, device=args.device).load()
        print(f"  running on {scorer.device}")

    detectors = {}
    if args.detectors:
        from stylelab.detectors import load_detectors
        print()
        detectors = load_detectors(device=args.device)

    rows = []
    t0 = time.time()
    # Per-stage totals, so a slow run reports where its time went instead of leaving it
    # to be guessed at afterwards.
    stage = {"parse": 0.0, "features": 0.0, "windows": 0.0, "lm": 0.0, "detectors": 0.0}

    for i, doc in enumerate(docs, 1):
        ts = time.perf_counter()
        # Parsed once and shared. Letting extract_features and windowed_features each
        # parse independently costs a second full spaCy pass, which on a 4000 word paper
        # is about 940 ms against 145 ms for every other CPU feature combined.
        parsed = parse_once(doc.text, nlp)
        stage["parse"] += time.perf_counter() - ts

        ts = time.perf_counter()
        feats = extract_features(doc.text, nlp=nlp, parsed=parsed)
        stage["features"] += time.perf_counter() - ts

        ts = time.perf_counter()
        feats.update(windowed_features(doc.text, nlp=nlp, sentences=parsed[1]))
        stage["windows"] += time.perf_counter() - ts

        if scorer is not None:
            ts = time.perf_counter()
            feats.update(scorer.features(doc.text))
            stage["lm"] += time.perf_counter() - ts
        if detectors:
            from stylelab.detectors import detector_features
            ts = time.perf_counter()
            feats.update(detector_features(doc.text, detectors))
            stage["detectors"] += time.perf_counter() - ts

        rows.append({
            "doc_id": doc.doc_id,
            "label": doc.label,
            "source": doc.source,
            "year": doc.year,
            # Which human document this one was generated to mirror, if any. Carried
            # through so analyze.py can compare against exactly the papers that were
            # mirrored rather than against the whole human corpus.
            "mirrors": doc.meta.get("mirrors"),
            **feats,
        })

        if i % 10 == 0 or i == len(docs):
            rate = i / max(time.time() - t0, 1e-6)
            eta = (len(docs) - i) / rate if rate > 0 else 0
            print(f"  {i}/{len(docs)} docs  {rate:.1f}/s  eta {eta / 60:.1f} min",
                  end="\r")

    print()
    df = pd.DataFrame(rows)

    # A feature that is constant across the whole corpus carries no information and
    # would only add noise to the ranking in analyze.py. Drop it here and say so, rather
    # than letting it appear as a zero-effect row later.
    meta_cols = {"doc_id", "label", "source", "year", "mirrors"}
    numeric = [c for c in df.columns if c not in meta_cols]
    dead = [c for c in numeric if df[c].nunique(dropna=False) <= 1]
    if dead:
        df = df.drop(columns=dead)
        print(f"dropped {len(dead)} constant features "
              f"(e.g. {', '.join(dead[:4])}{'...' if len(dead) > 4 else ''})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows x {len(df.columns) - len(meta_cols)} features "
          f"to {args.out}")

    total = time.time() - t0
    print(f"\nelapsed {total / 60:.1f} min over {len(docs)} documents "
          f"({total / max(len(docs), 1):.1f} s/doc)")
    print("  stage breakdown:")
    for name, secs in sorted(stage.items(), key=lambda kv: -kv[1]):
        if secs <= 0:
            continue
        print(f"    {name:12s} {secs / 60:7.1f} min  {secs / len(docs):6.2f} s/doc  "
              f"{100 * secs / total:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
