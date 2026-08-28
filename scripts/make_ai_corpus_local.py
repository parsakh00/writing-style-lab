"""Generate the AI half of the matched corpus with a local open model.

The generator runs on a local GPU rather than behind an API.
Free, unlimited, and it adds a second independent
generator, which is worth more than it might sound: a feature that separates human prose
from two unrelated model families is a property of machine writing, while one that shows
up for a single model is that model's habit.

A note on decoding, because it is the choice most likely to invalidate the result.

Sampling settings change the measured statistics substantially and in exactly the
direction the study is looking at. Greedy decoding produces unusually repetitive,
low-perplexity, low-variance text, so a greedy corpus would inflate every effect this
project measures and the finding would be about the decoding configuration rather than
about the model. Temperature, top-p and repetition penalty all move these numbers.

So the defaults here are the model publisher's own recommended generation settings, not
a choice of mine, and every setting is written into each document's metadata. If a
result later looks surprising, the decoding configuration is the first thing to check
and it is recorded rather than remembered.

Usage:
    python scripts/make_ai_corpus_local.py --n 300
    python scripts/make_ai_corpus_local.py --n 50 --model Qwen/Qwen2.5-14B-Instruct
    python scripts/make_ai_corpus_local.py --n 300 --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stylelab.corpus import (  # noqa: E402
    Document,
    clean_title,
    load_all,
    read_jsonl,
    write_jsonl,
)

SYSTEM = (
    "You are writing the body text of a peer-reviewed research article in chemistry "
    "and materials science, for submission to a journal such as Chemical Science or "
    "Journal of Materials Chemistry A."
)

PROMPT = """Write the main body of a research article titled:

"{title}"

Write roughly {n_words} words. Include the introduction, the results and discussion,
and a short conclusion, as continuous prose.

Do not include a reference list, figures, tables, equations, or citation markers.
Do not include section headings. Return only the article text.
"""


def build_prompts(human_docs, n: int, seed: int) -> list[tuple[str, str, int]]:
    """Pick human documents to mirror. Returns (doc_id, title, target_words).

    Titles are re-cleaned here rather than trusted from the corpus file, since corpora
    collected before clean_title existed still carry publisher footnote markers, and a
    title is the prompt in this script.
    """
    rng = random.Random(seed)
    usable = []
    for d in human_docs:
        title = clean_title(d.title or "")
        if title and d.word_count() >= 800:
            usable.append((d.doc_id, title, d.word_count()))
    rng.shuffle(usable)
    return usable[:n]

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Publisher-recommended decoding for the Qwen2.5 instruct family. Deliberately not tuned
# by me: a hand-picked configuration would make the corpus a statement about my taste in
# sampling parameters.
DEFAULT_GEN = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.05,
}


def load_model(name: str, device: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    # Decoder-only models must be left padded for batched generation. With right
    # padding the pad tokens sit between the prompt and the first generated token, and
    # the model continues from padding instead of from the prompt, which produces
    # fluent-looking output that does not answer the prompt at all.
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=torch.bfloat16,
        device_map=device or "auto",
    )
    model.eval()
    return tok, model


CONTINUE = (
    "Continue the article from exactly where it stopped, mid-argument if necessary. "
    "Do not repeat or summarise anything already written, do not restate the "
    "introduction, and do not start a new conclusion unless the article is finished. "
    "Write approximately {n_words} more words."
)


def _generate_raw(tok, model, chats: list[list[dict]], gen_cfg: dict,
                  max_new: int) -> list[str]:
    """One batched generation pass over a list of chat histories."""
    import torch

    texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
             for c in chats]
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=True,
            pad_token_id=tok.pad_token_id,
            **gen_cfg,
        )

    # With left padding every sequence begins generating at the same index, so one
    # slice is correct for the whole batch.
    prompt_len = enc["input_ids"].shape[1]
    return [tok.decode(seq[prompt_len:], skip_special_tokens=True).strip()
            for seq in out]


def _looks_repeated(existing: str, addition: str) -> bool:
    """Detect a continuation that restarts rather than continues.

    Asked to continue, an instruct model will sometimes re-open the article or paraphrase
    what it already wrote. Splicing that in produces a document with the introduction
    twice, which reads as pathologically repetitive and would show up as unusually low
    lexical diversity, a feature this study measures. Cheaper to detect and stop.
    """
    head = " ".join(addition.split()[:12]).lower()
    return bool(head) and head in existing.lower()


def generate_documents(tok, model, batch: list[tuple[str, str, int]], gen_cfg: dict,
                       max_new_cap: int, max_rounds: int,
                       target_ratio: float) -> list[str]:
    """Generate a batch to full target length using multi-round continuation.

    A single pass will not do it. Qwen2.5-7B writes about 44% of any requested length
    and plateaus near 900 words regardless of the number asked for, while the PMC papers
    it is mirroring have a median of 3822. Comparing full-length human papers against
    900-word model pieces would reintroduce exactly the length confound that already had
    to be removed from HC3 and RAID, and the resulting effects would be about length.

    So each document is continued across several turns until it reaches its target. Items
    finish at different rounds, so each round re-batches only the ones still short.
    """
    states = [{"chat": [{"role": "system", "content": SYSTEM},
                        {"role": "user",
                         "content": PROMPT.format(title=t, n_words=n)}],
               "text": "", "target": n}
              for _, t, n in batch]

    for _ in range(max_rounds):
        active = [i for i, s in enumerate(states)
                  if len(s["text"].split()) < s["target"] * target_ratio]
        if not active:
            break

        remaining = max(states[i]["target"] - len(states[i]["text"].split())
                        for i in active)
        max_new = min(int(remaining * 1.6), max_new_cap)
        outs = _generate_raw(tok, model, [states[i]["chat"] for i in active],
                             gen_cfg, max_new)

        for i, out in zip(active, outs):
            s = states[i]
            if not out or _looks_repeated(s["text"], out):
                # Mark finished: further rounds would only add more repetition.
                s["target"] = 0
                continue
            s["text"] = (s["text"] + "\n\n" + out).strip() if s["text"] else out
            s["chat"] = s["chat"] + [
                {"role": "assistant", "content": out},
                {"role": "user", "content": CONTINUE.format(
                    n_words=max(200, s["target"] - len(s["text"].split())))},
            ]

    return [s["text"] for s in states]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-cap", type=int, default=2048,
                    help="token budget per continuation round")
    ap.add_argument("--max-rounds", type=int, default=8,
                    help="continuation rounds. The model writes ~900 words per round "
                         "and PMC papers have a median of 3822, so several are needed")
    ap.add_argument("--min-ratio", type=float, default=0.6,
                    help="discard a generation shorter than this fraction of its "
                         "target. Roughly one in eight documents stops early when the "
                         "continuation guard trips, and keeping those would leave the "
                         "AI corpus with a short tail the human corpus does not have")
    ap.add_argument("--target-ratio", type=float, default=0.85,
                    help="stop continuing once this fraction of the target is reached")
    ap.add_argument("--min-words", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0,
                    help="index of this worker, 0-based")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="split the work across this many concurrent GPUs. Shards take "
                         "a strided slice of the length-sorted job list so each gets a "
                         "comparable mix of short and long papers, rather than one "
                         "worker taking every long one")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry-run", action="store_true")
    for key, val in DEFAULT_GEN.items():
        ap.add_argument(f"--{key.replace('_', '-')}", type=type(val), default=val)
    args = ap.parse_args()

    short = args.model.split("/")[-1]
    if args.out:
        out_path = args.out
    elif args.num_shards > 1:
        out_path = f"data/ai_{short}_s{args.shard}.jsonl"
    else:
        out_path = f"data/ai_{short}.jsonl"

    human = [d for d in load_all(args.data_dir, min_words=args.min_words)
             if d.label == "human" and d.source == "pmc"]
    if not human:
        print("no PMC documents found. Run scripts/fetch_pmc.py first.", file=sys.stderr)
        return 1

    jobs = build_prompts(human, args.n, args.seed)
    if not jobs:
        print("no usable titles in the human corpus", file=sys.stderr)
        return 1

    gen_cfg = {k: getattr(args, k) for k in DEFAULT_GEN}

    # Resume scans every shard file, not just this worker's own output. Workers run
    # concurrently and a restart must not regenerate what another shard already produced,
    # nor what a previous unsharded run produced.
    done: set[str] = set()
    for shard_file in sorted(Path(args.data_dir).glob(f"ai_{short}*.jsonl")):
        for d in read_jsonl(shard_file):
            src = d.meta.get("mirrors")
            if src:
                done.add(src)
    if done:
        print(f"resuming: {len(done)} documents already generated across all shards")
    jobs = [j for j in jobs if j[0] not in done]

    if args.num_shards > 1:
        # Sort before striding so shard k takes every k-th document by length. Slicing a
        # length-sorted list into contiguous blocks instead would hand one worker all the
        # long papers and leave it running hours after the others finished.
        jobs.sort(key=lambda j: j[2])
        jobs = jobs[args.shard::args.num_shards]
        print(f"shard {args.shard} of {args.num_shards}: {len(jobs)} documents")

    total_words = sum(j[2] for j in jobs)
    print(f"generator     {args.model}")
    print(f"to generate   {len(jobs)} documents")
    print(f"target words  {total_words:,} "
          f"(mean {total_words // max(len(jobs), 1):,})")
    print(f"decoding      {json.dumps(gen_cfg)}")

    if args.dry_run:
        print("\nDRY RUN, model not loaded and nothing generated.")
        if jobs:
            print("\nsample prompt:")
            _, title, n_words = jobs[0]
            for line in PROMPT.format(title=title, n_words=n_words).strip().split("\n")[:5]:
                print(f"  {line}")
        return 0

    if not jobs:
        print("nothing left to generate")
        return 0

    print(f"\nloading {args.model}...")
    t_load = time.time()
    tok, model = load_model(args.model, args.device)
    print(f"  loaded in {time.time() - t_load:.0f}s on {model.device}")

    # Batch documents of similar target length together. Generation runs until the
    # longest member of a batch finishes, so mixing a 1600 word target with a 6000 word
    # one makes the short one cost as much as the long one.
    jobs.sort(key=lambda j: j[2])

    docs: list[Document] = []
    t0 = time.time()
    n_short = 0

    for i in range(0, len(jobs), args.batch_size):
        batch = jobs[i:i + args.batch_size]
        try:
            outputs = generate_documents(tok, model, batch, gen_cfg,
                                         args.max_new_cap, args.max_rounds,
                                         args.target_ratio)
        except Exception as exc:  # noqa: BLE001
            print(f"\n  batch at {i} failed: {type(exc).__name__}: {exc}")
            continue

        for (src_id, title, n_words), text in zip(batch, outputs):
            wc = len(text.split())
            # Two floors. The absolute one keeps documents too short to measure at all;
            # the relative one keeps the AI length distribution aligned with the human
            # papers it mirrors, which is the whole point of generating to a target.
            if wc < args.min_words or wc < n_words * args.min_ratio:
                n_short += 1
                continue
            docs.append(Document(
                doc_id=f"ai:{short}:{len(docs)}",
                label="ai",
                source="generated",
                text=text,
                title=title,
                year=None,
                meta={"model": args.model, "mirrors": src_id,
                      "target_words": n_words, "decoding": gen_cfg},
            ))

        elapsed = time.time() - t0
        rate = (i + len(batch)) / max(elapsed, 1e-6)
        eta = (len(jobs) - i - len(batch)) / rate if rate > 0 else 0
        print(f"  {len(docs)} kept / {i + len(batch)} attempted  "
              f"{rate * 60:.1f} docs/min  eta {eta / 60:.0f} min", end="\r")

        # Append-as-we-go so a crash or a walltime kill does not lose the work.
        if docs:
            existing = list(read_jsonl(out_path)) if Path(out_path).exists() else []
            keep = [d for d in existing if d.meta.get("mirrors") not in
                    {x.meta.get("mirrors") for x in docs}]
            write_jsonl(keep + docs, out_path)

    print()
    if n_short:
        print(f"{n_short} generations fell short of {args.min_words} words and were "
              f"discarded")

    final = list(read_jsonl(out_path)) if Path(out_path).exists() else []
    if not final:
        print("no usable generations", file=sys.stderr)
        return 1
    words = sum(d.word_count() for d in final)
    print(f"{len(final)} documents in {out_path}")
    print(f"  {words:,} words, mean {words // len(final):,} per document")

    # Report how close the corpus got to its targets. This is the number that decides
    # whether the corpus is usable: if the model lands well short, the AI side is
    # systematically shorter than the human side and the comparison measures length.
    ratios = sorted(d.word_count() / d.meta["target_words"]
                    for d in final if d.meta.get("target_words"))
    if ratios:
        mid = ratios[len(ratios) // 2]
        print(f"  achieved/target length: median {mid:.2f}, "
              f"p10 {ratios[len(ratios) // 10]:.2f}, "
              f"p90 {ratios[9 * len(ratios) // 10]:.2f}")
        if mid < 0.7:
            print("  WARNING: the AI corpus is materially shorter than the human "
                  "corpus it mirrors.")
            print("  Raise --max-rounds, or expect analyze.py to be comparing length.")
    print(f"  elapsed {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
