"""Pull open human/AI corpora that other people built.

These are secondary corpora, and the distinction matters for how they are used.

The primary comparison in this repository is PMC full text against generated full text
on matched titles. Everything about that pairing is controlled: same topics, same
lengths, same genre.

The corpora here are not matched to it. HC3 is questions and answers. RAID's abstract
domain is abstracts, which are a few hundred words of unusually dense prose. Gutenberg
is nineteenth and early twentieth century literary writing. All three are genuinely
human, and all three differ from a chemistry paper in ways that have nothing to do with
whether a machine wrote them.

So they are tagged by source and analysed separately, never pooled into one "human"
pile. Pooling them would reintroduce exactly the register confound the PMC design was
built to avoid, and the resulting effect sizes would mostly be measuring genre.

What they are good for:

  HC3, RAID   independent human/AI pairs built by other people, with other generators.
              If a feature separates human from AI here as well, it is not an artefact
              of my prompt or my one generator.
  Gutenberg   an upper bound on human variance. Not a target to imitate. It shows how
              much sentence rhythm can vary before academic convention clamps it.

Usage:
    python scripts/fetch_datasets.py --hc3 500 --raid 500
    python scripts/fetch_datasets.py --gutenberg 40
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stylelab.corpus import Document, clean_text, write_jsonl  # noqa: E402

GUTENDEX = "https://gutendex.com/books"


def length_match(docs: list[Document], n_bins: int = 10,
                 verbose: bool = True) -> list[Document]:
    """Equalise the word-count distributions of the two labels.

    This is not optional hygiene, it is the difference between a result and an artefact.

    In these corpora the two labels differ systematically in length. HC3 human answers
    run to a median of 430 words against 324 for the model answers, and they are far
    more variable. RAID is worse: its human abstracts have a median of 170 words while
    the machine generations sit at 251, so a 300 word floor keeps 38% of the machine
    text and 0.1% of the human text.

    Left alone, a length-sensitive feature would separate the labels perfectly and the
    finding would be about length. Worse, it would look like a strong stylistic result,
    because plenty of style features correlate with length.

    The fix is a stratified subsample: bin by word count, and in each bin keep the same
    number of documents from each label. That equalises the marginal length
    distributions at the cost of discarding some documents, which is the right trade.
    """
    import numpy as np

    human = [d for d in docs if d.label == "human"]
    ai = [d for d in docs if d.label == "ai"]
    if not human or not ai:
        return docs

    all_w = np.array([d.word_count() for d in docs], dtype=float)
    # Quantile edges over the pooled lengths, so bins carry comparable mass.
    edges = np.unique(np.quantile(all_w, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return docs

    def binned(group: list[Document]) -> dict[int, list[Document]]:
        out: dict[int, list[Document]] = {}
        for d in group:
            b = int(np.clip(np.searchsorted(edges, d.word_count(), side="right") - 1,
                            0, len(edges) - 2))
            out.setdefault(b, []).append(d)
        return out

    hb, ab = binned(human), binned(ai)
    rng = random.Random(0)
    kept: list[Document] = []
    for b in sorted(set(hb) | set(ab)):
        h, a = hb.get(b, []), ab.get(b, [])
        k = min(len(h), len(a))
        if k == 0:
            continue
        # Shuffle before truncating so the kept subset is not ordered by fetch order,
        # which correlates with topic in both HC3 and RAID.
        rng.shuffle(h)
        rng.shuffle(a)
        kept.extend(h[:k])
        kept.extend(a[:k])

    if verbose:
        import statistics as s
        def med(g, lab):
            w = [d.word_count() for d in g if d.label == lab]
            return s.median(w) if w else 0
        print(f"  length matching: {len(docs)} -> {len(kept)} documents")
        print(f"    before  human median {med(docs, 'human'):.0f}w, "
              f"ai median {med(docs, 'ai'):.0f}w")
        print(f"    after   human median {med(kept, 'human'):.0f}w, "
              f"ai median {med(kept, 'ai'):.0f}w")
        n_h = sum(1 for d in kept if d.label == "human")
        print(f"    kept    {n_h} human, {len(kept) - n_h} ai")
        if len(kept) < 40:
            print("    WARNING: too few documents survive matching to support a "
                  "comparison. This corpus is not usable at this length floor.")
    return kept


def require_datasets():
    try:
        import datasets  # noqa: F401
        return True
    except ImportError:
        print("this source needs the datasets package: pip install datasets",
              file=sys.stderr)
        return False


def fetch_hc3(n: int, min_words: int) -> list[Document]:
    """Human ChatGPT Comparison Corpus: paired answers to the same questions.

    The raw all.jsonl is downloaded and parsed directly rather than going through
    load_dataset. The HC3 repository still contains a loading script, and datasets 5
    refuses any repository containing one, even when asked for a specific data file.
    Reading the jsonl is both simpler and immune to that machinery changing again.
    """
    import json

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("this source needs huggingface_hub: pip install huggingface_hub",
              file=sys.stderr)
        return []

    print(f"loading HC3 (target {n} per side)...")
    try:
        path = hf_hub_download("Hello-SimpleAI/HC3", "all.jsonl", repo_type="dataset")
    except Exception as exc:  # noqa: BLE001
        print(f"  could not download HC3: {type(exc).__name__}: {exc}")
        return []

    with open(path, "r", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]

    docs: list[Document] = []
    n_h = n_a = 0
    for i, row in enumerate(rows):
        if n_h >= n and n_a >= n:
            break
        # Answers are stored as lists; each entry is one full answer.
        for j, ans in enumerate(row.get("human_answers") or []):
            if n_h >= n:
                break
            t = clean_text(ans)
            if len(t.split()) >= min_words:
                docs.append(Document(doc_id=f"hc3:h:{i}:{j}", label="human",
                                     source="hc3", text=t,
                                     meta={"domain": row.get("source", "")}))
                n_h += 1
        for j, ans in enumerate(row.get("chatgpt_answers") or []):
            if n_a >= n:
                break
            t = clean_text(ans)
            if len(t.split()) >= min_words:
                docs.append(Document(doc_id=f"hc3:a:{i}:{j}", label="ai",
                                     source="hc3", text=t,
                                     meta={"domain": row.get("source", ""),
                                           "model": "chatgpt"}))
                n_a += 1
    print(f"  {n_h} human, {n_a} ai")
    return docs


def fetch_raid(n: int, min_words: int, domain: str) -> list[Document]:
    """RAID detection benchmark, restricted to unattacked text in one domain.

    Rows carrying an adversarial attack are excluded. Those exist to test detector
    robustness against deliberate evasion, which is not what is being measured here,
    and including them would put deliberately perturbed text into the AI distribution.
    """
    if not require_datasets():
        return []
    from datasets import load_dataset

    print(f"loading RAID (domain={domain}, target {n} per side, streaming)...")
    # RAID ships as plain CSV, so the data file is named explicitly. The full train
    # split is many gigabytes and only a small slice of one domain is wanted, so it is
    # streamed rather than downloaded.
    try:
        ds = load_dataset("liamdugan/raid", data_files="train.csv",
                          split="train", streaming=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  could not load RAID: {type(exc).__name__}: {exc}")
        return []

    docs: list[Document] = []
    n_h = n_a = 0
    seen = 0
    for row in ds:
        seen += 1
        if (n_h >= n and n_a >= n) or seen > 400_000:
            break
        if row.get("domain") != domain:
            continue
        if (row.get("attack") or "none") != "none":
            continue
        text = clean_text(row.get("generation") or "")
        if len(text.split()) < min_words:
            continue

        model = row.get("model") or ""
        if model == "human":
            if n_h >= n:
                continue
            docs.append(Document(doc_id=f"raid:h:{row.get('id', n_h)}", label="human",
                                 source="raid", text=text,
                                 meta={"domain": domain}))
            n_h += 1
        else:
            if n_a >= n:
                continue
            docs.append(Document(doc_id=f"raid:a:{row.get('id', n_a)}", label="ai",
                                 source="raid", text=text,
                                 meta={"domain": domain, "model": model}))
            n_a += 1

    print(f"  {n_h} human, {n_a} ai (scanned {seen:,} rows)")
    return docs


def _strip_gutenberg_boilerplate(text: str) -> str:
    """Cut the Project Gutenberg licence header and footer."""
    start = re.search(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG[^*]*\*\*\*",
                      text, re.I)
    end = re.search(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG[^*]*\*\*\*",
                    text, re.I)
    if start:
        text = text[start.end():]
    if end:
        # The end marker was found in the original string; re-find it after slicing.
        end2 = re.search(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG[^*]*\*\*\*",
                         text, re.I)
        if end2:
            text = text[:end2.start()]
    return text


def fetch_gutenberg(n: int, min_words: int, max_words: int) -> list[Document]:
    """Public domain literary prose via the Gutendex index."""
    print(f"fetching {n} Gutenberg texts...")
    docs: list[Document] = []
    page = 1

    while len(docs) < n and page <= 20:
        try:
            r = requests.get(GUTENDEX, params={"languages": "en", "topic": "fiction",
                                               "page": page}, timeout=60)
            r.raise_for_status()
            results = r.json().get("results", [])
        except Exception as exc:  # noqa: BLE001
            print(f"  index page {page} failed: {type(exc).__name__}")
            break
        if not results:
            break

        for book in results:
            if len(docs) >= n:
                break
            formats = book.get("formats", {})
            url = next((u for k, u in formats.items()
                        if k.startswith("text/plain") and not u.endswith(".zip")), None)
            if not url:
                continue
            try:
                rr = requests.get(url, timeout=90)
                rr.raise_for_status()
                body = _strip_gutenberg_boilerplate(rr.text)
            except Exception:  # noqa: BLE001
                continue

            text = clean_text(body)
            wc = len(text.split())
            if wc < min_words:
                continue
            # Books run to hundreds of thousands of words. Truncating to roughly the
            # length of a paper keeps document length from becoming the thing that
            # separates this corpus from every other one.
            if wc > max_words:
                text = " ".join(text.split()[:max_words])

            docs.append(Document(
                doc_id=f"gutenberg:{book.get('id')}",
                label="human", source="gutenberg", text=text,
                title=book.get("title", "")[:200],
                meta={"authors": [a.get("name", "") for a in book.get("authors", [])]},
            ))
            print(f"  {len(docs)}/{n}", end="\r")
            time.sleep(0.4)
        page += 1

    print()
    return docs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hc3", type=int, default=0, help="documents per side from HC3")
    ap.add_argument("--raid", type=int, default=0, help="documents per side from RAID")
    ap.add_argument("--raid-domain", default="abstracts",
                    help="RAID domain; 'abstracts' is the closest to academic prose")
    ap.add_argument("--gutenberg", type=int, default=0)
    ap.add_argument("--min-words", type=int, default=300)
    ap.add_argument("--no-length-match", action="store_true",
                    help="skip the stratified length match between labels. Only do "
                         "this if you intend to handle the length confound yourself; "
                         "in HC3 and RAID the two labels differ in length enough to "
                         "produce a strong result that is entirely about length.")
    ap.add_argument("--gutenberg-max-words", type=int, default=6000)
    ap.add_argument("--outdir", default="data")
    args = ap.parse_args()

    if not (args.hc3 or args.raid or args.gutenberg):
        ap.print_help()
        print("\nnothing requested; pass at least one of --hc3, --raid, --gutenberg")
        return 1

    outdir = Path(args.outdir)
    wrote = 0

    def finish(docs: list[Document], name: str) -> int:
        if not docs:
            return 0
        if not args.no_length_match:
            docs = length_match(docs)
        return write_jsonl(docs, outdir / name)

    if args.hc3:
        wrote += finish(fetch_hc3(args.hc3, args.min_words), "sec_hc3.jsonl")
    if args.raid:
        wrote += finish(fetch_raid(args.raid, args.min_words, args.raid_domain),
                        "sec_raid.jsonl")
    if args.gutenberg:
        docs = fetch_gutenberg(args.gutenberg, args.min_words,
                               args.gutenberg_max_words)
        if docs:
            wrote += write_jsonl(docs, outdir / "sec_gutenberg.jsonl")

    print(f"\nwrote {wrote} documents to {outdir}/")
    print("These are tagged by source. Keep them out of the primary comparison:")
    print("  python scripts/analyze.py --sources pmc,generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
