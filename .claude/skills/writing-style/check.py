"""Score a draft against published scientific prose. No dependencies beyond the stdlib.

Reads the files in data/ next to this script, so the whole skill directory can be
copied anywhere and will work on its own.

    python check.py draft.md
    python check.py draft.md --register letter
    python check.py draft.md --reference group

Reports three things:

  targets     where the draft sits against measured ranges from published papers
  vocabulary  general-purpose words the corpus effectively never uses
  sequences   share of the draft's connective word sequences that papers have used
  structure   words before the verb, sentences that open on old information,
              clauses per sentence, and authors named to introduce a finding
  phrasing    share of the connective formulas papers rely on

Numbers describe, they do not judge. A deviation is a place to look. Some of them will
already be the right call.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

RE_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
RE_SENT = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9(\"'])")
RE_NUMERIC = re.compile(r"\S*\d\S*")
RE_CONCESSIVE = re.compile(
    r"\b(?:though|although|however|whereas|albeit|nonetheless|nevertheless|"
    r"admittedly|granted|conversely|whilst)\b", re.I)
# Contrastive specification, which papers use at 0.90 per 1000 words and drafts at 5.11.
RE_CONTRASTIVE = re.compile(
    r"(?:rather than|while (?:it|this|that|the)|on the other hand|whereas)", re.I)
# Stock hedges, measured at 0.000 across 786,313 words of papers.
RE_COUNTERWEIGHT = re.compile(
    r"(?:at the expense of|is not to say|trade-?off|in exchange for|that said|"
    r"to be fair|the flip side|cuts both ways)", re.I)
RE_ATTENUATOR = re.compile(
    r"\b(?:for (?:now|the moment)|to some extent|in practice|arguably|"
    r"it (?:remains|is worth|should be noted)|we would not)\b", re.I)

# Text inside quotation marks is being discussed, not used. A document explaining that
# "arguably" is an attenuator would otherwise be scored as using one, and a draft naming
# "In conclusion" as a stock phrase scores as containing it. Measured once at z = +15.99
# on exactly that mistake.
_RE_QUOTED = re.compile(r"\"[^\"]{1,120}\"|'[^']{1,120}'|“[^”]{1,120}”")


def drop_quoted(text: str) -> str:
    """Blank out quoted spans so mentions are not counted as uses."""
    return _RE_QUOTED.sub(" ", text)


FUNCTION_WORDS = set(
    "a about above after again against all also although always an and another any are "
    "as at be because been before being below between both but by can cannot could did "
    "do does down during each either few for from further had has have he her here his "
    "how however i if in into is it its itself just may might more most much must my "
    "neither no nor not of off on once one only or other our out over own same she "
    "should since so some such than that the their them then there these they this "
    "those though through thus to too under until up upon very was we were what when "
    "where whether which while who whose why will with within would you your".split())

FIRST_PERSON = ("we", "our", "us", "ours", "ourselves", "i", "my", "me", "mine")

# Which targets apply where.
#
# The reference is built from journal articles, and several of its numbers describe that
# genre rather than good writing. A cover letter is written in "we" and scores 22 first
# person per 1000 words against a paper range of 0.9 to 8.9; documentation is imperative
# and scores 0.19 passive against 0.38 to 0.70. Reporting those as failures taught nobody
# anything and had to be waved away by hand three times before this existed.
#
# UNIVERSAL features describe the instruction-tuned register and hold in any prose.
# PAPER features describe the genre and apply only to journal-article text.
UNIVERSAL = {"counterweight_rate", "attenuator_rate", "contrastive_rate",
             "concessive_rate", "balanced_sentence_frac", "long_word_rate",
             "numeric_token_rate"}

REGISTERS = {
    "paper":  None,            # every target applies
    "letter": UNIVERSAL,       # plus nothing: a letter is first-person and shorter
    "docs":   UNIVERSAL,       # documentation is imperative and scannable
}

# Reported ranges are p25 to p75 of the corpus. Only features a writer can act on.
SHOWN = [
    ("counterweight_rate", "stock hedges /1000w"),
    ("contrastive_rate", "contrastive constructions /1000w"),
    ("attenuator_rate", "attenuators /1000w"),
    ("concessive_rate", "concessives /1000w"),
    ("balanced_sentence_frac", "self-qualifying sentences"),
    ("passive_per_clause", "passive per clause"),
    ("first_person_rate", "first person /1000w"),
    ("sent_len_mean", "mean sentence length"),
    ("sent_len_iqr", "sentence length IQR"),
    ("punct_comma_rate", "commas /1000w"),
    ("numeric_token_rate", "numeric tokens /1000w"),
    ("long_word_rate", "long words /1000w"),
    ("nominalisation_rate", "nominalisation /1000w"),
]


def load(name: str) -> dict:
    p = DATA / name
    if not p.exists():
        sys.exit(f"missing {p}. The data/ directory must travel with this script.")
    return json.loads(p.read_text(encoding="utf-8"))


def strip_markup(text: str) -> str:
    """Remove markup so the measures see prose rather than syntax."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"^\s*\|.*\|\s*$", " ", text, flags=re.M)
    text = re.sub(r"^\s{0,3}#{1,6}\s+.*$", " ", text, flags=re.M)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.M)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", text)
    return re.sub(r"[ \t]+", " ", text)


def sentences(text: str) -> list[str]:
    flat = re.sub(r"\s*\n\s*", " ", text).strip()
    return [s for s in RE_SENT.split(flat) if len(RE_WORD.findall(s)) >= 3]


def measure(text: str) -> dict:
    # Register measures run on the unquoted text; length and word measures
    # run on the whole thing, since a quotation is still words on the page.
    unquoted = drop_quoted(text)
    w = RE_WORD.findall(text)
    n = len(w) or 1
    sents = sentences(text)
    lens = sorted(len(RE_WORD.findall(s)) for s in sents) or [0]
    mid = len(lens) // 2

    def rate(c):
        return 1000.0 * c / n

    low = [x.lower() for x in w]
    balanced = sum(1 for s in sents
                   if RE_CONCESSIVE.search(drop_quoted(s))
                   or RE_COUNTERWEIGHT.search(drop_quoted(s)))
    # Passive is approximated without a parser: a form of "be" followed by a past
    # participle. Cruder than a dependency parse and close enough to place a draft.
    passive = len(re.findall(r"\b(?:is|are|was|were|be|been|being)\s+\w+ed\b", text, re.I))
    clauses = max(len(sents), 1)
    return {
        "counterweight_rate": rate(len(RE_COUNTERWEIGHT.findall(unquoted))),
        "contrastive_rate": rate(len(RE_CONTRASTIVE.findall(unquoted))),
        "attenuator_rate": rate(len(RE_ATTENUATOR.findall(unquoted))),
        "concessive_rate": rate(len(RE_CONCESSIVE.findall(unquoted))),
        "balanced_sentence_frac": balanced / clauses,
        "passive_per_clause": passive / clauses,
        "first_person_rate": rate(sum(low.count(x) for x in FIRST_PERSON)),
        "sent_len_mean": sum(lens) / len(lens),
        "sent_len_iqr": lens[3 * len(lens) // 4] - lens[len(lens) // 4],
        "punct_comma_rate": rate(text.count(",")),
        "numeric_token_rate": rate(len(RE_NUMERIC.findall(text))),
        "long_word_rate": rate(sum(1 for x in w if len(x) >= 8)),
        "nominalisation_rate": rate(len(re.findall(
            r"\b\w{4,}(?:tion|sion|ment|ness|ity|ance|ence|ism)s?\b", text, re.I))),
        "_n_words": len(w),
        "_n_sentences": len(sents),
        "_median_len": lens[mid],
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("draft")
    ap.add_argument("--top", type=int, default=13)
    ap.add_argument("--reference", choices=("papers", "corpus", "group"), default="papers",
                    help="corpus: 615 excerpts from 615 adsorption and simulation "
                         "papers. group: one research group and its coauthors. papers "
                         "(default): where the two bands overlap, the stricter target")
    ap.add_argument("--caption", action="store_true",
                    help="treat the draft as a figure caption and score its shape")
    ap.add_argument("--intro", action="store_true",
                    help="treat the draft as an introduction and score its shape: "
                         "opener, gap, purpose statement, questions, citation density")
    ap.add_argument("--suggest", action="store_true",
                    help="for every word triple no paper uses, show what papers write "
                         "after the same two words, sentence by sentence")
    ap.add_argument("--register", choices=sorted(REGISTERS), default="paper",
                    help="paper applies every target; letter and docs apply "
                         "only the register measures, since sentence length, "
                         "passive voice and first person are genre features "
                         "of journal articles rather than of good writing")
    return ap


def report(text: str, register: str = "paper", reference: str = "papers",
           top: int = 13, name: str = "draft", suggest: bool = False,
           intro: bool = False, caption: bool = False) -> str:
    """Score a text and return the report as a string.

    This is the library entry point: pass prose, get the report. The command line
    wraps it with a file path.
    """
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _run(strip_markup(text), argparse.Namespace(
            register=register, reference=reference, top=top, suggest=suggest,
            intro=intro, caption=caption), name)
    return buf.getvalue()


def suggest_sequences(text: str, sents: list[str]) -> None:
    """For each word triple no paper uses, what papers write after the same two words.

    This is how a sentence gets rebuilt: not by guessing a better phrase but by reading
    off the continuation papers actually use. Real papers themselves run 69 to 76%
    unattested on all triples, since technical names are unattested by nature, so the
    aim is to reach that range and stop.
    """
    from collections import defaultdict
    tri = load("trigrams.json")["trigrams"]
    cont: dict = defaultdict(Counter)
    for g, c in tri.items():
        a, b, d = g.split()
        cont[(a, b)][d] += c
    total = unatt = 0
    print("\nsequence suggestions (papers 69-76% unattested on all triples)")
    for i, snt in enumerate(sents, 1):
        w = [x.lower() for x in RE_WORD.findall(snt)]
        grams = [" ".join(w[j:j + 3]) for j in range(len(w) - 2)]
        miss = [g for g in grams if g not in tri]
        total += len(grams); unatt += len(miss)
        useful = [(g, cont.get(tuple(g.split()[:2]))) for g in miss]
        useful = [(g, t) for g, t in useful if t]
        if not useful:
            continue
        print(f"  [{i}] {snt[:110]}")
        for g, top in useful[:6]:
            a, b, _ = g.split()
            alts = ", ".join(f"{k} ({v})" for k, v in top.most_common(3))
            print(f"        '{g}' -> after '{a} {b}' papers write: {alts}")
    if total:
        print(f"  unattested triples: {unatt}/{total} = {unatt / total:.0%}")



def intro_shape(text: str, sents: list[str], n_words: int) -> None:
    """Score an introduction against 266 published ones; see SKILL.md, Introductions."""
    n = len(sents)
    if n < 5:
        print("\nintroduction: too short to score (under 5 sentences)")
        return
    first = sents[0]
    OP = [("importance or wide use (papers 11%)",
           r"(?:play(?:s)? (?:a|an) (?:key|important|crucial|central|vital|major)|(?:has|have) (?:attracted|received|gained|drawn|garnered)|(?:is|are) (?:one of the most|widely|among the most|of (?:great|considerable))|great (?:attention|interest|potential)|(?:emerged|promising) (?:as|candidates?)|extensively (?:studied|investigated|used)|(?:has|have) been (?:widely|extensively))"),
          ("recent growth (papers 5%)", r"(?:in recent (?:years|decades)|recently|over the (?:past|last)|the last (?:decade|few years))"),
          ("societal need (papers 6%)", r"(?:the (?:need|demand|challenge)|increasing (?:demand|concern|levels?)|global (?:warming|energy|climate)|greenhouse gas|energy crisis|co2 emissions?|climate change|environmental|air pollution|clean(?:er)? energy)"),
          ("definition (papers 4%)", r"(?:(?:are|is) a (?:class|family|type|group|series) of|consist(?:s|ing)? of|composed of|constructed (?:from|by)|known as|refers? to)")]
    kind = "plain factual claim (papers 74%)"
    for label, pat in OP:
        if re.search(pat, first, re.I):
            kind = label
            break
    GAP = re.compile(r"(?:remains? (?:unclear|unknown|challenging|elusive|an open|a challenge|poorly understood|limited|scarce|to be)|little (?:is known|attention|work|information)|few (?:studies|reports|works|attempts)|no (?:study|report|systematic|general)|has not (?:been|yet)|have not (?:been|yet)|not (?:yet|fully|well) (?:been )?(?:understood|explored|studied|established|investigated|addressed|clear)|is still (?:lacking|missing|unclear|unknown|debated)|lack of|open question|to date|(?:however|but|yet|unfortunately|despite)[^.]{0,120}(?:difficult|challeng|hinder|limit|problem|unclear|unknown|remains?|not been|little|few|scarce|hamper|suffer))", re.I)
    PUR = re.compile(r"(?:in this (?:work|paper|study|article|contribution|letter)|here,? we|in the present (?:work|study|paper)|the (?:aim|purpose|goal|objective) of (?:this|the present)|this (?:work|paper|study) (?:presents|reports|describes|examines|investigates|addresses|focuses|aims)|we (?:report|present|propose|develop|investigate|examine|study|demonstrate|show|introduce|address|extend|apply|use|explore)\b)", re.I)
    gi = next((i for i, x in enumerate(sents) if GAP.search(x)), None)
    pi = next((i for i, x in enumerate(sents) if PUR.search(x)), None)
    ann = len(re.findall(r"(?:we (?:find|found|show|demonstrate|observe) that|our (?:results|findings|calculations|simulations) (?:show|reveal|indicate|suggest|demonstrate))", text, flags=re.I))
    MARK = re.compile(r"\[(?:\d{1,3})(?:\s?[,\u2013-]\s?\d{1,3})*\]|\((?:[A-Z][A-Za-z-]+(?: et al\.?)?(?:,| and [A-Z][A-Za-z-]+)? ?\d{4}[a-z]?(?:; ?)?)+\)")
    dens = 1000 * len(MARK.findall(text)) / max(n_words, 1)
    print("\nintroduction shape (266 published introductions)")
    print(f"  opener: {kind}")
    if gi is None:
        print("  gap: none found (papers state one in 51%, as a concrete lack)")
    else:
        print(f"  gap: sentence {gi + 1}, {gi / n:.0%} of the way in (papers 18-68%)")
    if pi is None:
        print("  purpose statement: none found (papers 53%: 'In this work, we...')  <<")
    else:
        print(f"  purpose statement: sentence {pi + 1}, {pi / n:.0%} of the way in (papers 36-77%)"
              f"{'  <<' if pi / n < 0.25 else ''}")
    print(f"  length: {n_words} words (papers 515-779); citations {dens:.1f}/1000w (papers 9.7-44.2)"
          f"{'  <<' if dens < 5 else ''}")
    if "?" in text:
        print("  contains a literal question (papers 3%): the gap implies the question instead")
    if ann:
        print(f"  announces findings {ann}x (papers do this in 8% of introductions)")



def caption_shape(text: str, n_words: int) -> None:
    """Score a figure caption against 1,837 published ones; see SKILL.md."""
    text = text.strip()
    sents = [x for x in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text) if x.strip()]
    if not sents:
        print("\ncaption: nothing to measure")
        return
    VERB1 = re.compile(r"\b(?:is|are|was|were|shows?|show|illustrates?|presents?|depicts?|displays?|compares?|represents?|gives?|indicates?|has|have|can|reveals?)\b")
    frag = not VERB1.search(sents[0])
    print("\ncaption shape (1,837 published captions)")
    print(f"  opening: {'verbless fragment (papers 91%)' if frag else 'full sentence (papers 9%)  <<'}")
    mark = "" if 10 <= n_words <= 120 else "  <<"
    print(f"  length: {n_words} words in {len(sents)} sentence(s) (figure captions 24-92 words; table captions 11-30){mark}")
    if re.search(r"\(\s*[a-d]\s*\)", text, re.I):
        print("  panel labels present (papers 38% when multi-panel)")
    if re.search(r"\d+\s*(?:K|bar|kPa|MPa|atm|mol|wt%|nm)\b|\bat \d|\bT\s*=|\bP\s*=", text):
        print("  numeric conditions present (papers 31%)")
    else:
        print("  no numeric conditions: add temperature, pressure or composition if the figure depends on them")
    if re.search(r"\b(?:solid|dashed|dotted|open|filled|closed) (?:lines?|circles?|symbols?|squares?|triangles?|curves?)|\bsymbols? (?:are|represent|denote|show)|\blines? (?:are|represent|denote|show|correspond)", text, re.I):
        print("  line and symbol key present (papers 16%)")
    if re.search(r"\bshow(?:s|ing)? that\b|\bdemonstrat|\bindicating that\b|\bconfirming\b|\bsuggesting that\b", text, re.I):
        print("  states a conclusion (papers 2%): the caption identifies, the text interprets  <<")
    if re.search(r"\bfigure \d+ shows\b|\bfig\.? \d+ shows\b", text, re.I):
        print("  'Figure N shows' belongs to the text, not the caption  <<")


def main() -> int:
    args = build_parser().parse_args()
    path = Path(args.draft)
    if not path.exists():
        sys.exit(f"{path} not found")
    text = strip_markup(path.read_text(encoding="utf-8", errors="replace"))
    _run(text, args, path.name)
    return 0


def _run(text: str, args: argparse.Namespace, name: str) -> None:

    refname = {"papers": "combined_reference.json", "corpus": "reference.json", "group": "group_reference.json"}[args.reference]
    ref = load(refname)["features"]
    vocab = load("vocab.json")
    formulas = load("formulas.json")
    m = measure(text)
    sents = sentences(text)

    print(f"{name}: {m['_n_words']:,} words, {m['_n_sentences']} sentences\n")
    if m["_n_words"] < 300:
        print("under 300 words; these measures are noisy at this length\n")

    print(f"{'':38s}{'draft':>9s}{args.reference:>16s}")
    print("-" * 64)
    allowed = REGISTERS[args.register]
    if allowed is not None:
        print(f"register: {args.register}. Genre-specific targets are not applied; "
              f"{len(allowed)} universal measures shown.")
        print()
    off, shown = [], 0
    for key, label in SHOWN[: args.top]:
        if key not in ref:
            continue
        if allowed is not None and key not in allowed:
            continue
        lo, hi = ref[key]["p25"], ref[key]["p75"]
        v = m[key]
        shown += 1
        mark = "" if lo <= v <= hi else "  <<"
        if mark:
            off.append(label)
        print(f"{label:38s}{v:9.2f}{lo:8.2f}-{hi:<7.2f}{mark}")
    print(f"\n{shown - len(off)}/{shown} inside the published range")

    total = vocab["total_words"]
    freq = vocab["freq"]
    seen = Counter(x.lower() for x in RE_WORD.findall(text))
    rare = [(w, c) for w, c in seen.items()
            if len(w) >= 4 and w not in FUNCTION_WORDS
            and 1e6 * freq.get(w, 0) / total < 1.0]
    if rare:
        rare.sort(key=lambda t: (freq.get(t[0], 0), -t[1]))
        print(f"\nwords the corpus does not use ({len(rare)}):")
        for w, c in rare[:12]:
            tag = "absent" if freq.get(w, 0) == 0 else "rare"
            print(f"  {w:24s} used {c}x   {tag}")
        print("  Technical terms belong here. General-purpose words do not:")
        print("  papers quantify where those characterise.")

    # Word sequences. Trigrams carrying at least two function words are phrasing rather
    # than content, and papers build almost entirely from ones other papers have used.
    # Measured on 20 held-out papers: 47 to 58% of a paper's connective trigrams appear
    # in the corpus set, p05 39%. Drafts written in the default register run 24
    # to 39%. This is the measure closest to "the word order reads machine-written".
    low = [x.lower() for x in RE_WORD.findall(text)]
    if (DATA / "sequences.json").exists():
        known = set(load("sequences.json")["trigrams"])
    else:
        # The page ships trigrams.json only; the connective subset is derived here.
        known = {g for g in load("trigrams.json")["trigrams"]
                 if sum(x in FUNCTION_WORDS for x in g.split()) >= 2}
    conn = [g for g in zip(low, low[1:], low[2:])
            if sum(x in FUNCTION_WORDS for x in g) >= 2]
    conn = [" ".join(g) for g in conn]
    if conn:
        hit = sum(g in known for g in conn) / len(conn)
        mark = "" if hit >= 0.39 else "  <<"
        print(f"\nconnective sequences papers have used: {hit:.0%} (papers 39-65%, p05-p95){mark}")
        if hit < 0.39:
            missing = [g for g in dict.fromkeys(conn) if g not in known]
            print(f"  sequences no paper in 6.4M words makes ({len(missing)}), first 15:")
            for g in missing[:15]:
                print(f"    {g}")
            print("  Rebuild the sentence around a sequence papers use; the formulas below are a start.")


    # Structure, after Gopen and Swan (1990). Two of their seven principles are
    # checkable without a parser. "Follow a grammatical subject as soon as possible with
    # its verb": the share of sentences with more than 12 words before the first finite
    # verb runs 15 to 29% in 19 published papers; a draft written to sound academic hit
    # 60%. "Place old information in the topic position": papers open 5 to 19% of
    # sentences with This, These, Such, Here or It, pointing back to the previous
    # sentence; drafts written in the default register open none that way.
    VERB = re.compile(r"\b(?:is|are|was|were|has|have|had|can|may|could|should|will|"
                      r"shows?|gives?|leads?|results?|depends?|becomes?|remains?|"
                      r"increases?|decreases?|follows?|corresponds?|indicates?|"
                      r"suggests?|requires?|provides?|yields?|occurs?|forms?)\b")
    gaps = []
    for snt in sents:
        mv = VERB.search(snt)
        if mv:
            gaps.append(len(snt[:mv.start()].split()))
    if gaps:
        late = sum(g > 12 for g in gaps) / len(gaps)
        back = sum(1 for snt in sents if re.match(r"(?:This|These|Such|Here|It)\b", snt)) / len(sents)
        print(f"\nstructure (Gopen and Swan)")
        print(f"  sentences with >12 words before the verb: {late:.0%} (papers 15-29%){'  <<' if late > 0.35 else ''}")
        print(f"  sentences opening on This/These/Such/Here/It: {back:.0%} (papers 5-19%){'  <<' if back < 0.03 else ''}")


    # Citation. In 19 published papers, 1,749 citations are a number attached to a claim
    # in the author's words; 199 name an author, and 4 name one with a reporting verb
    # ("X et al. showed that"). A name is used when the name is the topic (a method
    # called after its authors), not to introduce a finding.
    INTEGRAL = re.compile(r"\b(?:[A-Z][a-z]+ et al\.?|[A-Z][a-z]+ and (?:co-?workers|colleagues))"
                          r"[^.]{0,40}?\b(?:showed|show|reported|report|found|find|proposed|"
                          r"developed|observed|demonstrated|suggested|noted|derived|studied|"
                          r"calculated|computed|measured|obtained|presented|pointed out)\b"
                          r"|\b(?:work|study|results|decomposition|analysis|model) of [A-Z][a-z]+ and (?:co-?workers|colleagues)")
    cites = INTEGRAL.findall(text)
    # Density, measured on 245 papers with markers kept: 9.9 to 17.2 per 1000 words.
    MARK = re.compile(r"\[(?:\d{1,3}|n)(?:\s?[,–-]\s?\d{1,3})*\]|\((?:[A-Z][A-Za-z-]+(?: et al\.?)?(?:,| and [A-Z][A-Za-z-]+)? ?\d{4}[a-z]?(?:; ?)?)+\)")
    n_marks = len(MARK.findall(text))
    dens = 1000 * n_marks / max(m["_n_words"], 1)
    if args.register == "paper" and m["_n_words"] >= 300:
        print(f"  citations: {n_marks} = {dens:.1f} per 1000 words (papers 9.9-17.2){'  <<' if dens < 5 else ''}")
    CL = re.compile(r"\b(?:which|that|because|although|whereas|while|when|if|since|so that|as|where)\b|, and\b|, but\b|; ")
    if sents:
        cl = [1 + len(CL.findall(snt)) for snt in sents]
        mean_cl = sum(cl) / len(cl); three = sum(x >= 3 for x in cl) / len(cl)
        print(f"  clauses per sentence: {mean_cl:.2f} (papers 1.6-2.2); three or more: {three:.0%} (papers 13-33%)"
              f"{'  <<' if mean_cl > 2.3 or three > 0.36 else ''}")
    if cites:
        print(f"  author named to introduce a finding ({len(cites)}; papers 4 in 103,000 words):")
        for hit in cites[:5]:
            print(f"    {hit.strip()[:70]}")
        print("  State the finding in your words and attach the reference.")


    # Four habits of drafts, each measured at 3 to 60 times the paper rate on 245
    # papers: a colon inside the sentence introducing evidence (papers 1.1% of
    # sentences, drafts 7.2%), a passive with its agent attached by "by" (1.4% against
    # 4.3%), a trailing "depending on" (0.10 against 1.24 per 1000 words), and
    # "is therefore" inside the verb (0.03 against 1.85). The paper pattern is the
    # quantity as subject, the comparison on "while" or "whereas", and the conclusion
    # in its own sentence.
    if sents:
        n_s = len(sents)
        colon = sum(1 for x in sents if re.search(r"[a-z\]]:\s+[a-z]", x)) / n_s
        byag = sum(1 for x in sents if re.search(
            r"\b(?:is|are|was|were)\s+(?:\w+ly\s+)?\w+ed\s+by\s+(?:the|a|an|every|each|all|most|any)\b", x)) / n_s
        dep = 1000 * len(re.findall(r"\bdepending on\b", text)) / max(m["_n_words"], 1)
        thf = 1000 * len(re.findall(r"\b(?:is|are|was|were)\s+(?:therefore|thus)\b", text)) / max(m["_n_words"], 1)
        flags = []
        if colon > 0.03: flags.append(f"colon inside the sentence {colon:.0%} (papers 1%)")
        if byag > 0.03: flags.append(f"passive with a by-agent {byag:.0%} (papers 1%)")
        if dep > 0.4: flags.append(f"'depending on' {dep:.1f}/1000w (papers 0.1)")
        if thf > 0.3: flags.append(f"'is therefore' {thf:.1f}/1000w (papers 0.03)")
        if flags:
            print("  draft habits papers do not share:")
            for f in flags:
                print(f"    {f}")


    # General rules derived from the suggestions and verified across the corpus. Each
    # is a construction papers use at or near zero against a form they use hundreds of
    # times; see SKILL.md, "General rules from the suggestions".
    RULES = [
        ("quantity noun + at/from/among (papers: of)",
         r"\b(?:uptake|heat|spread|deviation|amount|density|capacity|loading|enthalpy|rate|value|values|distribution|coefficient|fraction|ratio|number)\s+(?:at|from|among)\b(?!\s+(?:which|that))"),
        ("comparison with bare 'experiment' (papers: the experimental value/data)",
         r"\b(?:than|with|to|from)\s+experiment\b(?!al|s)"),
        ("a factor without a number",
         r"\bby an? (?:similar|comparable|large|small|considerable) factor\b"),
        ("abstract-noun claim papers do not make",
         r"\bis (?:a|an|the) (?:source|property|limitation|indication|reflection|feature|hallmark|sign) of\b"),
        ("'therefore' inside the verb (papers open the sentence with it)",
         r"\b(?:is|are|was|were|has|have|can|could|would|should)\s+therefore\b"),
        ("non-canonical noun (training data, test data, Henry constant, Boltzmann weight)",
         r"\b(?:training data|test data|henry constant|boltzmann weight)\b"),
        ("'at low/high coverage' or 'in the ... limit' where a variable was set",
         r"\bat (?:low|high) coverage\b|\bin the (?:low|high|zero)[- ]\w+ limit\b"),
    ]
    hits = []
    for label, pat in RULES:
        found = re.findall(pat, text, re.I)
        if found:
            hits.append((label, len(found)))
    if hits:
        print("  general rules (SKILL.md, from the suggestions):")
        for label, k in hits:
            print(f"    {k}x  {label}")

    if getattr(args, "caption", False):
        caption_shape(text, m["_n_words"])

    if getattr(args, "intro", False):
        intro_shape(text, sents, m["_n_words"])

    if getattr(args, "suggest", False):
        suggest_sequences(text, sents)

    ftotal = formulas["total_words"]
    fset = formulas["formulas"]
    present = set(" ".join(g) for g in zip(low, low[1:], low[2:]))
    top60 = sorted(fset.items(), key=lambda kv: -kv[1])[:60]
    used = sum(1 for g, _ in top60 if g in present)
    print(f"\nconnective formulas: {used}/60 used ({used/60:.0%}; papers 7-13%)")
    if used / 60 < 0.07:
        print("  heavily used in papers and missing here:")
        for g, c in [(g, c) for g, c in top60 if g not in present][:8]:
            print(f"    {g:26s}{1e6 * c / ftotal:6.0f} per million")


if __name__ == "__main__":
    raise SystemExit(main())
