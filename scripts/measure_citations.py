"""How papers cite: density, placement, cluster size, the words around a marker, and
whether anything is quoted directly. Runs on data/human_pmc_cited.jsonl, built by
fetch_pmc with citations kept as bracketed markers.

    python scripts/measure_citations.py
"""
import json, re, random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
docs = [json.loads(l)["text"].replace("\n", " ") for l in open(ROOT / "data/human_pmc_cited.jsonl", encoding="utf-8")]
CITE = re.compile(r"\[(?:\d{1,3}|n)(?:\s?[,–-]\s?\d{1,3})*\]")
RE_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
RE_W = re.compile(r"[A-Za-z][A-Za-z'-]*")
INTEG = re.compile(r"\b[A-Z][a-z]+ (?:et al\.?|and (?:co-?workers|colleagues))")
VERBS = ("showed|shown|reported|found|proposed|developed|observed|demonstrated|suggested|noted|derived|studied|used|"
         "calculated|computed|measured|obtained|presented|investigated|described|prepared|synthesized|synthesised|"
         "examined|explored|introduced|performed|carried out|employed|applied|revealed|concluded|argued|pointed out|"
         "attributed|achieved|fabricated|designed|reviewed|discussed|compared|tested|evaluated|confirmed|established|"
         "identified|determined|predicted|estimated|modeled|modelled|simulated")
INTEG_V = re.compile(r"\b[A-Z][a-z]+ (?:et al\.?|and (?:co-?workers|colleagues))\s*(?:\[[^\]]+\])?\s*(?:have |has |also |recently |first )?(?:" + VERBS + r")\b")
INTRO = re.compile(r"\b(previous(?:ly)?|earlier|recent(?:ly)?|prior (?:work|studies|study|reports?)|several (?:studies|works|groups|authors|reports)|"
                   r"many (?:studies|works|groups|authors|reports)|numerous|various|extensive(?:ly)?|well[- ]known|well[- ]established|widely|"
                   r"commonly|often|typically|for example|for instance|e\.g\.|see|as (?:reported|shown|described|discussed|demonstrated|noted|observed|suggested|proposed|expected) (?:in|by|previously|elsewhere)|"
                   r"according to|in the literature|in (?:ref|refs)\.?|literature|reported (?:to|by|in)|"
                   r"has been (?:shown|reported|demonstrated|found|observed|proposed|suggested|used|studied|described|attributed|widely)|"
                   r"have been (?:shown|reported|demonstrated|found|observed|proposed|suggested|used|studied|described|widely)|it is (?:well )?known|is known to|are known to)\b", re.I)

dens, share, ex = [], [], []
pos, clus, pre, post, intro = Counter(), Counter(), Counter(), Counter(), Counter()
integ = integ_v = quotes = 0
for d in docs:
    S = [s for s in RE_SENT.split(d) if 5 <= len(s.split()) <= 90]
    w = len(RE_W.findall(d)); c = len(CITE.findall(d)); dens.append(1000 * c / w)
    cs = [s for s in S if CITE.search(s)]; share.append(len(cs) / max(len(S), 1))
    integ += len(INTEG.findall(d)); integ_v += len(INTEG_V.findall(d))
    quotes += len(re.findall(r"[\"“][^\"”]{30,}[\"”]", d))
    for s in cs:
        for i in INTRO.findall(s): intro[i.lower()] += 1
        ms = list(CITE.finditer(s))
        for m in ms:
            clus[min(len(re.findall(r"\d+", m.group())), 5)] += 1
            tail = s[m.end():].strip()
            pos["sentence-final" if not tail.strip(" .") else ("before , ; or )" if tail[:1] in ",;)" else "mid-sentence")] += 1
            pw = RE_W.findall(s[:m.start()]); pre[" ".join(pw[-2:]).lower()] += 1
            if tail.strip(" .") and tail[:1] not in ",;)":
                t = RE_W.findall(tail); post[t[0].lower() if t else ""] += 1
        if 8 <= len(s.split()) <= 38 and len(ms) <= 2: ex.append(s.strip())

q = lambda a, f: sorted(a)[int(f * (len(a) - 1))]
W = sum(len(RE_W.findall(d)) for d in docs); C = sum(len(CITE.findall(d)) for d in docs)
print(f"{len(docs)} papers, {W:,} words, {C:,} citation markers")
print(f"density per 1000 words: p05 {q(dens,.05):.1f}  p25 {q(dens,.25):.1f}  p50 {q(dens,.5):.1f}  p75 {q(dens,.75):.1f}  p95 {q(dens,.95):.1f}")
print(f"sentences carrying a citation: p05 {q(share,.05):.0%}  p25 {q(share,.25):.0%}  p50 {q(share,.5):.0%}  p75 {q(share,.75):.0%}  p95 {q(share,.95):.0%}")
tot = sum(pos.values()); print("\nmarker position:"); [print(f"  {k:22s}{v:6d} {v/tot:5.0%}") for k, v in pos.most_common()]
tot = sum(clus.values()); print("references per marker:"); [print(f"  {k}{'+' if k == 5 else ' '} {v:6d} {v/tot:5.0%}") for k, v in sorted(clus.items())]
print(f"\nauthor named: {integ} ({1000*integ/W:.2f}/1000w); named with a reporting verb: {integ_v} ({1000*integ_v/W:.2f}/1000w) -> {integ_v/(integ_v+C):.1%} of citations")
print(f"direct quotations of 30+ characters: {quotes} in {len(docs)} papers")
print("\ntwo words before the marker (top 40):"); [print(f"  {k:26s}{v:5d}") for k, v in pre.most_common(40)]
print("\nfirst word after a mid-sentence marker (top 20):"); [print(f"  {k:14s}{v:5d}") for k, v in post.most_common(20)]
print("\nintroducers in citing sentences (count, per 1000 words):"); [print(f"  {k:34s}{v:5d}  {1000*v/W:5.2f}") for k, v in intro.most_common(40)]
random.seed(7); print("\n40 citing sentences:"); [print("  - " + s[:200]) for s in random.sample(ex, min(40, len(ex)))]
json.dump({"n_papers": len(docs), "words": W, "markers": C, "density_p25": q(dens,.25), "density_p75": q(dens,.75),
           "share_p25": q(share,.25), "share_p75": q(share,.75), "position": dict(pos), "cluster": dict(clus),
           "integral": integ, "integral_verb": integ_v, "quotes": quotes, "intro": intro.most_common(40), "pre": pre.most_common(60)},
          open(ROOT / "results/citation_measured.json", "w"), indent=1)
