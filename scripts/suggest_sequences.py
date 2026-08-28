"""For every word triple in a draft that no paper uses, show what papers put after the
same two words. Needs the full n-gram index (results/academic_ngrams.json), so this
lives in the repository rather than in the skill.

    python scripts/suggest_sequences.py DRAFT.md
"""
import json, re, sys
sys.stdout.reconfigure(errors="replace")
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
ng = json.load(open(ROOT / "results/academic_ngrams.json", encoding="utf-8"))
tri, big = ng["trigrams"], ng["bigrams"]
cont = defaultdict(Counter)
for g, c in tri.items():
    a, b, d = g.split(); cont[(a, b)][d] += c
RE_W = re.compile(r"[A-Za-z][A-Za-z'-]*"); RE_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
text = Path(sys.argv[1]).read_text(encoding="utf-8")
sents = [s for s in RE_SENT.split(re.sub(r"\s*\n\s*", " ", text)) if len(s.split()) > 3]
total = unatt = 0
for i, s in enumerate(sents, 1):
    w = [x.lower() for x in RE_W.findall(s)]
    grams = [" ".join(w[j:j + 3]) for j in range(len(w) - 2)]
    miss = [g for g in grams if g not in tri]
    total += len(grams); unatt += len(miss)
    print(f"\n[{i}] {s[:140]}\n     unattested {len(miss)}/{len(grams)}")
    for g in miss:
        a, b, c = g.split(); top = cont.get((a, b))
        if top:
            print(f"       '{g}'  ->  after '{a} {b}' papers write: " + ", ".join(f"{k} ({v})" for k, v in top.most_common(4)))
        else:
            print(f"       '{g}'  ->  '{a} {b}' is {'' if f'{a} {b}' in big else 'not '}a paper bigram")
print(f"\nunattested: {unatt}/{total} = {unatt / total:.0%}")
