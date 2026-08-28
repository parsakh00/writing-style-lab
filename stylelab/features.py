"""Per-document style features.

Every function here returns numbers that can be checked by hand on a short passage.
That constraint is deliberate. The point of the project is to end up with a policy that
states *why* it asks for something, and a feature nobody can verify by counting cannot
support a rule anybody should follow.

Features are grouped:

  surface     sentence and paragraph length, and how much those lengths vary
  lexical     vocabulary richness, word length
  punctuation rates per 1000 words for each mark
  function    frequencies of ~90 function words, the classic Burrows Delta vector
  syntax      passive voice, clause depth, opener variety, part of speech mix
  discourse   hedges, boosters, transitions, first person, list structures
  tells       phrase level habits reported as characteristic of model prose

All rates are per 1000 words unless the name says otherwise, so documents of different
lengths are directly comparable. The length-sensitive vocabulary measures use MTLD and a
fixed-window type token ratio for the same reason: raw TTR falls as text gets longer, so
comparing it across corpora with different document lengths measures length, not range.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Sequence

import numpy as np

# --- tokenisation ---------------------------------------------------------

_RE_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_ABBREV = (
    r"Fig|Figs|Eq|Eqs|Ref|Refs|Tab|approx|ca|cf|vs|etc|al|e\.g|i\.e|Dr|Prof|Mr|Mrs|Ms|"
    r"St|Inc|Ltd|Co|No|vol|pp|wt|at|min|max|Sec|Chap|resp|viz"
)
# Split on terminal punctuation followed by whitespace and something that can open a
# sentence. Python's re has no variable-width lookbehind, so abbreviations cannot be
# excluded in the pattern; fragments that ended on one are merged back afterwards.
_RE_SENT_CAND = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9(\"'])")
_RE_ENDS_ABBREV = re.compile(r"(?:^|\s)(?:" + _ABBREV + r")\.[\"')\]]*$", re.IGNORECASE)
# A lone initial, as in "J. Smith", is also not a sentence end.
_RE_ENDS_INITIAL = re.compile(r"(?:^|\s)[A-Z]\.$")

# A sentence needs this many real words to count. Stripping figure callouts can leave
# stubs like "See." behind, and counting those as very short sentences would inflate
# the length variance of whichever corpus cites figures more heavily.
MIN_SENT_WORDS = 3


def words(text: str) -> list[str]:
    return _RE_WORD.findall(text)


def split_sentences(text: str) -> list[str]:
    """Regex sentence splitter with scientific-abbreviation guards.

    Used when spaCy is unavailable. spaCy's parser-based boundaries are better and are
    preferred when a pipeline is passed in, but the two agree closely enough on academic
    prose that results stay comparable.
    """
    flat = re.sub(r"\s*\n\s*", " ", text).strip()
    if not flat:
        return []

    merged: list[str] = []
    for piece in _RE_SENT_CAND.split(flat):
        piece = piece.strip()
        if not piece:
            continue
        # The previous fragment ended on "Fig." or "et al." or an initial, so this was
        # never a sentence boundary. Glue it back on.
        if merged and (_RE_ENDS_ABBREV.search(merged[-1])
                       or _RE_ENDS_INITIAL.search(merged[-1])):
            merged[-1] = merged[-1] + " " + piece
        else:
            merged.append(piece)

    return [s for s in merged if len(words(s)) >= MIN_SENT_WORDS]


def split_paragraphs(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return [p for p in paras if len(words(p)) >= MIN_SENT_WORDS]


# --- helpers --------------------------------------------------------------

def _safe_stats(values: Sequence[float], prefix: str) -> dict[str, float]:
    """Mean, spread and shape for a list of lengths.

    The coefficient of variation and the burstiness index are the two that matter for
    this study. Both are scale free, so they compare a corpus of long review articles
    against a corpus of short papers without the mean contaminating the answer.
    """
    if not values:
        return {
            f"{prefix}_mean": 0.0, f"{prefix}_sd": 0.0, f"{prefix}_cv": 0.0,
            f"{prefix}_burstiness": 0.0, f"{prefix}_p10": 0.0, f"{prefix}_p50": 0.0,
            f"{prefix}_p90": 0.0, f"{prefix}_iqr": 0.0, f"{prefix}_neighbour_delta": 0.0,
        }
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    cv = sd / mean if mean > 0 else 0.0
    # Goh and Barabasi burstiness: -1 fully regular, 0 Poisson, +1 highly bursty.
    burst = (sd - mean) / (sd + mean) if (sd + mean) > 0 else 0.0
    # Mean absolute step between consecutive sentences. Two documents can share a length
    # distribution while one alternates long and short and the other drifts slowly; this
    # separates them and the distribution statistics cannot.
    neighbour = float(np.abs(np.diff(arr)).mean()) if arr.size > 1 else 0.0
    q1, q3 = np.percentile(arr, [25, 75])
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_sd": sd,
        f"{prefix}_cv": cv,
        f"{prefix}_burstiness": burst,
        f"{prefix}_p10": float(np.percentile(arr, 10)),
        f"{prefix}_p50": float(np.percentile(arr, 50)),
        f"{prefix}_p90": float(np.percentile(arr, 90)),
        f"{prefix}_iqr": float(q3 - q1),
        f"{prefix}_neighbour_delta": neighbour,
    }


def _rate(count: int, n_words: int) -> float:
    return 1000.0 * count / n_words if n_words else 0.0


def mtld(tokens: Sequence[str], threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity, McCarthy and Jarvis 2010.

    Chosen over type token ratio because it is length invariant by construction: it
    counts how many words it takes for a running TTR to fall below the threshold, then
    averages that run length forward and backward. Raw TTR would just tell us which
    corpus has longer documents.
    """
    def _pass(seq: Sequence[str]) -> float:
        factors, types, count = 0.0, set(), 0
        for tok in seq:
            types.add(tok)
            count += 1
            if count > 0 and len(types) / count <= threshold:
                factors += 1
                types, count = set(), 0
        if count > 0:
            ttr = len(types) / count
            # Partial factor for the trailing run.
            factors += (1 - ttr) / (1 - threshold) if threshold < 1 else 0
        return len(seq) / factors if factors > 0 else float(len(seq))

    low = [t.lower() for t in tokens]
    if len(low) < 50:
        return 0.0
    return (_pass(low) + _pass(low[::-1])) / 2.0


# --- word lists -----------------------------------------------------------

FUNCTION_WORDS = [
    "a", "about", "above", "after", "again", "against", "all", "also", "although",
    "always", "an", "and", "another", "any", "are", "as", "at", "be", "because",
    "been", "before", "being", "below", "between", "both", "but", "by", "can",
    "cannot", "could", "did", "do", "does", "down", "during", "each", "either",
    "few", "for", "from", "further", "had", "has", "have", "he", "her", "here",
    "his", "how", "however", "i", "if", "in", "into", "is", "it", "its", "itself",
    "just", "may", "might", "more", "most", "much", "must", "my", "neither", "no",
    "nor", "not", "of", "off", "on", "once", "one", "only", "or", "other", "our",
    "out", "over", "own", "same", "she", "should", "since", "so", "some", "such",
    "than", "that", "the", "their", "them", "then", "there", "these", "they",
    "this", "those", "though", "through", "thus", "to", "too", "under", "until",
    "up", "upon", "very", "was", "we", "were", "what", "when", "where", "whether",
    "which", "while", "who", "whose", "why", "will", "with", "within", "would",
    "you", "your",
]

HEDGES = [
    "may", "might", "could", "possibly", "perhaps", "probably", "likely", "unlikely",
    "suggest", "suggests", "suggested", "appear", "appears", "appeared", "seem",
    "seems", "seemed", "indicate", "indicates", "indicated", "tend", "tends",
    "relatively", "somewhat", "generally", "typically", "often", "usually",
    "approximately", "roughly", "presumably", "apparently", "potentially",
    "assume", "assumed", "estimate", "estimated", "suppose", "arguably",
]

BOOSTERS = [
    "clearly", "obviously", "certainly", "definitely", "undoubtedly", "evidently",
    "significantly", "substantially", "considerably", "notably", "remarkably",
    "markedly", "dramatically", "strongly", "highly", "extremely", "crucial",
    "critical", "essential", "vital", "key", "important", "profound", "compelling",
    "robust", "unprecedented", "striking",
]

TRANSITIONS = [
    "however", "moreover", "furthermore", "additionally", "therefore", "thus",
    "consequently", "nevertheless", "nonetheless", "conversely", "similarly",
    "accordingly", "hence", "meanwhile", "subsequently", "overall", "importantly",
    "notably", "specifically", "particularly", "indeed", "instead", "besides",
]

FIRST_PERSON = ["we", "our", "us", "ours", "ourselves", "i", "my", "me", "mine"]

# Phrases repeatedly reported in the literature and in practitioner accounts as
# characteristic of instruction-tuned model prose. This list is a hypothesis to be
# tested against the corpora, not an assumption. analyze.py reports the measured rate in
# both corpora so entries that turn out not to separate can be dropped.
MODEL_TELLS = [
    "it is important to note", "it is worth noting", "it should be noted",
    "plays a crucial role", "plays a significant role", "plays a vital role",
    "a wide range of", "a wide array of", "a myriad of", "in the realm of",
    "delve into", "delving into", "underscores the", "underscoring the",
    "highlights the importance", "highlighting the importance",
    "sheds light on", "shed light on", "pave the way", "paves the way",
    "paving the way", "a testament to", "in today's", "ever-evolving",
    "comprehensive understanding", "holistic approach", "multifaceted",
    "significant implications", "far-reaching", "at the forefront",
    "cutting-edge", "state-of-the-art", "seamlessly", "intricate",
    "intricacies", "navigate the", "navigating the", "harness the",
    "harnessing the", "unlock the", "unlocking the", "leverage the",
    "the landscape of", "a cornerstone", "in conclusion", "in summary",
    "it is essential to", "not only", "but also", "as we", "let us",
    "rich tapestry", "meticulous", "meticulously", "realm", "pivotal",
    "profound implications", "invaluable", "robust framework",
]

_TELL_PATTERNS = [(t, re.compile(r"\b" + re.escape(t).replace(r"\ ", r"\s+") + r"\b", re.I))
                  for t in MODEL_TELLS]

# "A, B, and C" three-item lists. Model prose leans on these heavily.
_RE_TRICOLON = re.compile(
    r"\b\w+(?:\s+\w+){0,2},\s+\w+(?:\s+\w+){0,2},\s+and\s+\w+", re.I
)
_RE_NOMINALISATION = re.compile(
    r"\b\w{4,}(?:tion|sion|ment|ness|ity|ance|ence|ism|isation|ization)s?\b", re.I
)


# --- feature blocks -------------------------------------------------------

def surface_features(text: str, sentences: Sequence[str]) -> dict[str, float]:
    sent_lens = [len(words(s)) for s in sentences]
    paras = split_paragraphs(text)

    feats: dict[str, float] = {}
    feats.update(_safe_stats(sent_lens, "sent_len"))

    # Paragraph features are undefined, not zero, for text with no paragraph breaks.
    #
    # Treating an unbroken document as a single enormous paragraph silently turns a
    # formatting difference into a style measurement. In HC3 every human answer is a
    # single block while a quarter of the model answers use blank lines, which made
    # "sentences per paragraph" read 15 against 9 and produced four confident rules
    # that were about markdown, not prose.
    #
    # NaN propagates correctly: the separation table drops it, and a feature that is
    # undefined for most of one side is excluded from the comparison rather than
    # compared against a fabricated value.
    if len(paras) >= 2:
        para_sent_counts = [max(1, len(split_sentences(p))) for p in paras]
        para_word_counts = [len(words(p)) for p in paras]
        feats.update(_safe_stats(para_sent_counts, "para_sents"))
        feats.update(_safe_stats(para_word_counts, "para_words"))
    else:
        nan = float("nan")
        for prefix in ("para_sents", "para_words"):
            feats.update({k: nan for k in _safe_stats([], prefix)})
    feats["has_paragraph_breaks"] = float(len(paras) >= 2)

    feats["n_sentences"] = float(len(sentences))
    feats["n_words"] = float(len(words(text)))

    # Share of sentences at the extremes. A corpus can match on mean and sd while never
    # actually risking a five-word sentence or a forty-word one.
    n = len(sent_lens) or 1
    feats["short_sent_frac"] = sum(1 for x in sent_lens if x <= 8) / n
    feats["long_sent_frac"] = sum(1 for x in sent_lens if x >= 35) / n
    return feats


def lexical_features(text: str) -> dict[str, float]:
    toks = words(text)
    n = len(toks)
    low = [t.lower() for t in toks]
    counts = Counter(low)
    types = len(counts)

    # Fixed 500-word window so the comparison is not just a length comparison.
    window = low[:500]
    win_ttr = len(set(window)) / len(window) if window else 0.0

    lens = [len(t) for t in toks] or [0]
    return {
        "ttr_500": win_ttr,
        "mtld": mtld(toks),
        "hapax_rate": sum(1 for c in counts.values() if c == 1) / types if types else 0.0,
        "type_count": float(types),
        "word_len_mean": float(np.mean(lens)),
        "word_len_sd": float(np.std(lens)),
        "long_word_rate": _rate(sum(1 for t in toks if len(t) >= 8), n),
        "nominalisation_rate": _rate(len(_RE_NOMINALISATION.findall(text)), n),
    }


def punctuation_features(text: str) -> dict[str, float]:
    n = len(words(text))
    marks = {
        "comma": text.count(","),
        "semicolon": text.count(";"),
        "colon": text.count(":"),
        "em_dash": text.count("—"),
        "en_dash": text.count("–"),
        "hyphen": text.count("-"),
        "paren": text.count("("),
        "question": text.count("?"),
        "exclaim": text.count("!"),
        "quote": text.count('"'),
        "apostrophe": text.count("'"),
    }
    feats = {f"punct_{k}_rate": _rate(v, n) for k, v in marks.items()}
    # Any dash at all, since the em/en distinction is partly a house-style artefact.
    feats["punct_long_dash_rate"] = _rate(marks["em_dash"] + marks["en_dash"], n)
    return feats


def function_word_features(text: str) -> dict[str, float]:
    toks = [t.lower() for t in words(text)]
    n = len(toks)
    counts = Counter(toks)
    return {f"fw_{w}": _rate(counts.get(w, 0), n) for w in FUNCTION_WORDS}


def discourse_features(text: str, sentences: Sequence[str]) -> dict[str, float]:
    toks = [t.lower() for t in words(text)]
    n = len(toks)
    counts = Counter(toks)

    def group_rate(group: Sequence[str]) -> float:
        return _rate(sum(counts.get(w, 0) for w in group), n)

    feats = {
        "hedge_rate": group_rate(HEDGES),
        "booster_rate": group_rate(BOOSTERS),
        "transition_rate": group_rate(TRANSITIONS),
        "first_person_rate": group_rate(FIRST_PERSON),
        "tricolon_rate": _rate(len(_RE_TRICOLON.findall(text)), n),
    }

    # Sentence-initial transitions specifically. Model prose front-loads connectives far
    # more than it uses them mid sentence, and the two positions carry different weight.
    initial = 0
    initial_this = 0
    for s in sentences:
        w = words(s)
        if not w:
            continue
        head = w[0].lower()
        if head in TRANSITIONS:
            initial += 1
        if head == "this":
            initial_this += 1
    ns = len(sentences) or 1
    feats["sent_initial_transition_frac"] = initial / ns
    feats["sent_initial_this_frac"] = initial_this / ns

    # Opener variety: entropy over the first word of each sentence. Low entropy means
    # the text keeps starting sentences the same handful of ways.
    heads = Counter(words(s)[0].lower() for s in sentences if words(s))
    total = sum(heads.values())
    if total:
        probs = np.array([c / total for c in heads.values()])
        feats["opener_entropy"] = float(-(probs * np.log2(probs)).sum())
        feats["opener_top1_frac"] = max(heads.values()) / total
    else:
        feats["opener_entropy"] = 0.0
        feats["opener_top1_frac"] = 0.0
    return feats


def tell_features(text: str) -> dict[str, float]:
    # Quoted mentions do not count. A draft naming "In conclusion" as a stock
    # phrase previously scored z = +15.99 for containing it.
    text = drop_quoted(text)
    n = len(words(text))
    total = 0
    per_phrase: dict[str, float] = {}
    for phrase, pat in _TELL_PATTERNS:
        c = len(pat.findall(text))
        total += c
        key = "tell_" + re.sub(r"[^a-z0-9]+", "_", phrase.lower()).strip("_")
        per_phrase[key] = _rate(c, n)
    # Counted before the summary entries are added, otherwise tell_total_rate counts
    # itself as a distinct phrase and every document with any tell is off by one.
    per_phrase["tell_distinct_count"] = float(sum(1 for v in per_phrase.values() if v > 0))
    per_phrase["tell_total_rate"] = _rate(total, n)
    return per_phrase


# --- specificity ----------------------------------------------------------

# Units common in physical-science writing. Not exhaustive by design: the point is to
# detect the presence of measured quantities, not to parse them.
_UNITS = (r"nm|mm|cm|km|m|A|K|C|F|eV|meV|keV|kJ|J|cal|mol|kg|g|mg|ug|mL|L|bar|atm|Pa|"
          r"kPa|MPa|GPa|Torr|Hz|kHz|MHz|GHz|s|ms|ns|ps|min|h|wt|vol|ppm|ppb|%|deg")
_RE_QUANTITY = re.compile(r"\b\d+(?:\.\d+)?\s*(?:" + _UNITS + r")\b")
_RE_NUMERIC_TOKEN = re.compile(r"\S*\d\S*")
_RE_DECIMAL = re.compile(r"\b\d+\.\d+\b")
_RE_ACRONYM = re.compile(r"\b[A-Z]{2,}[0-9]*\b")


def specificity_features(text: str) -> dict[str, float]:
    """How densely the text carries concrete, checkable detail.

    Added after a draft that matched the human corpus on every other feature was still
    called machine-written by an external detector, while three real papers of the same
    length were not. The draft held 1.2 numeric tokens per thousand words; the papers
    held 51 to 109.

    This block was missing for a structural reason worth recording: the word tokeniser is
    [A-Za-z][A-Za-z'-]*, so every digit in every corpus had been invisible to all other
    features. Measured on the full corpora, numeric token density separates real papers
    from generated ones at d = -1.94, which would place it among the ten strongest
    features in the study.

    Counted on the raw text rather than on the word list, precisely because the word list
    is what was dropping the evidence.
    """
    n_words = len(text.split())
    if n_words == 0:
        return {}
    return {
        "numeric_token_rate": 1000.0 * len(_RE_NUMERIC_TOKEN.findall(text)) / n_words,
        "quantity_rate": 1000.0 * len(_RE_QUANTITY.findall(text)) / n_words,
        "decimal_rate": 1000.0 * len(_RE_DECIMAL.findall(text)) / n_words,
        "acronym_rate": 1000.0 * len(_RE_ACRONYM.findall(text)) / n_words,
        "digit_char_rate": 1000.0 * sum(c.isdigit() for c in text) / n_words,
    }


# Text inside quotation marks is being discussed, not used. A document explaining that
# "arguably" is an attenuator would otherwise be scored as using one, and a draft naming
# "In conclusion" as a stock phrase scores as containing it. Measured once at z = +15.99
# on exactly that mistake.
_RE_QUOTED = re.compile(r"\"[^\"]{1,120}\"|'[^']{1,120}'|“[^”]{1,120}”")


def drop_quoted(text: str) -> str:
    """Blank out quoted spans so mentions are not counted as uses."""
    return _RE_QUOTED.sub(" ", text)

# --- register -------------------------------------------------------------

# Instruction-tuned prose qualifies as it asserts: every claim arrives with its own
# counterweight. Measured across corpora, drafts written in that register carry
# counterweight constructions at 6.3 per 1000 words against 0.68 in published papers,
# and nearly one sentence in five contains its own rebuttal against one in sixteen.
#
# This block exists because a corpus comparison could not find it. The generated corpus
# was a model told to write a paper, not a model addressing a reader, so both sides of
# that comparison lacked the register entirely and the difference cancelled to nothing.
_RE_CONCESSIVE = re.compile(
    r"\b(?:though|although|however|whereas|albeit|nonetheless|nevertheless|"
    r"admittedly|granted|conversely|whilst)\b", re.I)
# Two kinds of contrast, and only one of them is a tell.
#
# Papers use contrastive specification constantly: "attributed to accessibility rather
# than to number", "physisorption rather than covalent attachment". Measured over 786,313
# words, "rather than" appears at 0.079 per 1000, "while the" at 0.413, "on the other
# hand" at 0.139. Those are precise writing and must not be flagged.
#
# What papers never do is hedge a claim with a stock concessive. "At the expense of",
# "that said", "in exchange for", "is not to say" and "to be fair" all measure exactly
# 0.000 across the same corpus. An earlier version of this pattern merged the two groups
# and reported drafts at 3 to 6 per 1000 mostly on the strength of legitimate contrast.
_RE_CONTRASTIVE = re.compile(
    r"(?:rather than|while (?:it|this|that|the)|on the other hand|whereas)", re.I)
_RE_COUNTERWEIGHT = re.compile(
    r"(?:at the expense of|is not to say|comes at (?:a|the) (?:cost|price)|"
    r"trade-?off|in exchange for|that said|to be fair|for its part|"
    r"the flip side|cuts both ways)", re.I)

_RE_ATTENUATOR = re.compile(
    r"\b(?:for (?:now|the moment)|to some extent|in practice|arguably|"
    r"it (?:remains|is worth|should be noted)|we would not|"
    r"not (?:deep|fatal|serious))\b", re.I)


def register_features(text: str, sentences: Sequence[str]) -> dict[str, float]:
    """Density of self-qualifying, balanced-tradeoff constructions.

    Quoted spans are excluded: a document explaining that "arguably" is an
    attenuator is discussing one, not using one.
    """
    unquoted = drop_quoted(text)
    n = len(words(text))
    if n == 0:
        return {}
    n_sent = len(sentences) or 1
    balanced = sum(1 for x in sentences
                   if _RE_CONCESSIVE.search(drop_quoted(x))
                   or _RE_COUNTERWEIGHT.search(drop_quoted(x)))
    return {
        "concessive_rate": _rate(len(_RE_CONCESSIVE.findall(unquoted)), n),
        "counterweight_rate": _rate(len(_RE_COUNTERWEIGHT.findall(unquoted)), n),
        "contrastive_rate": _rate(len(_RE_CONTRASTIVE.findall(unquoted)), n),
        "attenuator_rate": _rate(len(_RE_ATTENUATOR.findall(unquoted)), n),
        "balanced_sentence_frac": balanced / n_sent,
    }


# --- vocabulary -----------------------------------------------------------

_VOCAB_CACHE: dict = {}


def load_academic_vocab(path: str = "results/academic_vocab.json") -> dict:
    """Word frequencies from the human corpora, cached after first load."""
    import json
    from pathlib import Path as _P
    if path in _VOCAB_CACHE:
        return _VOCAB_CACHE[path]
    f = _P(path)
    if not f.exists():
        _VOCAB_CACHE[path] = {}
        return {}
    data = json.loads(f.read_text(encoding="utf-8"))
    _VOCAB_CACHE[path] = data
    return data


def out_of_vocabulary(text: str, min_rate_per_million: float = 1.0,
                      vocab_path: str = "results/academic_vocab.json"
                      ) -> list[tuple[str, int, float]]:
    """Ordinary words in the text that published papers effectively never use.

    Built after a vocabulary comparison found that the words separating a draft from
    published work were not technical terms but judgements: silently, honoured,
    defensible, admits, badly, visibly, wrong, young, ordinary. None appears in six
    million words of papers, or appears far below the rate a real author would use it.
    Papers quantify where those words characterise.

    Returns (word, count in text, rate per million in the corpus), rarest first. Words
    absent from the corpus come back with a rate of 0.0.
    """
    data = load_academic_vocab(vocab_path)
    if not data:
        return []
    freq = data["freq"]
    total = data["total_words"]
    seen = Counter(w.lower() for w in words(text))
    out = []
    for w, c in seen.items():
        if w in FUNCTION_WORDS or len(w) < 4:
            continue
        rate = 1e6 * freq.get(w, 0) / total
        if rate < min_rate_per_million:
            out.append((w, c, rate))
    return sorted(out, key=lambda t: (t[2], -t[1]))


def syntax_features(doc: Any) -> dict[str, float]:
    """Part of speech and dependency features. Requires a parsed spaCy Doc."""
    n_tok = sum(1 for t in doc if not t.is_punct and not t.is_space)
    if n_tok == 0:
        return {}

    pos = Counter(t.pos_ for t in doc)
    deps = Counter(t.dep_ for t in doc)

    # Dependency depth per sentence, averaged. Deeper trees mean more subordination.
    depths: list[float] = []
    for sent in doc.sents:
        d = 0
        for tok in sent:
            depth, guard = 0, 0
            cur = tok
            while cur.head is not cur and guard < 100:
                cur = cur.head
                depth += 1
                guard += 1
            d = max(d, depth)
        depths.append(d)

    n_clauses = deps.get("ROOT", 0) + deps.get("ccomp", 0) + deps.get("xcomp", 0) \
        + deps.get("advcl", 0) + deps.get("relcl", 0) + deps.get("acl", 0)
    n_passive = deps.get("nsubjpass", 0) + deps.get("auxpass", 0) + deps.get("csubjpass", 0)

    feats = {
        "dep_depth_mean": float(np.mean(depths)) if depths else 0.0,
        "dep_depth_sd": float(np.std(depths)) if depths else 0.0,
        "passive_per_clause": n_passive / n_clauses if n_clauses else 0.0,
        "subordination_rate": _rate(
            deps.get("advcl", 0) + deps.get("relcl", 0) + deps.get("acl", 0), n_tok
        ),
        "coordination_rate": _rate(deps.get("cc", 0), n_tok),
        "noun_rate": _rate(pos.get("NOUN", 0) + pos.get("PROPN", 0), n_tok),
        "verb_rate": _rate(pos.get("VERB", 0), n_tok),
        "adj_rate": _rate(pos.get("ADJ", 0), n_tok),
        "adv_rate": _rate(pos.get("ADV", 0), n_tok),
        "adp_rate": _rate(pos.get("ADP", 0), n_tok),
    }
    feats["noun_verb_ratio"] = (
        feats["noun_rate"] / feats["verb_rate"] if feats["verb_rate"] else 0.0
    )

    # Variety of grammatical openers, distinct from the lexical opener entropy above.
    opener_pos = Counter(
        next((t.pos_ for t in sent if not t.is_punct and not t.is_space), "X")
        for sent in doc.sents
    )
    total = sum(opener_pos.values())
    if total:
        probs = np.array([c / total for c in opener_pos.values()])
        feats["opener_pos_entropy"] = float(-(probs * np.log2(probs)).sum())
    else:
        feats["opener_pos_entropy"] = 0.0
    return feats


# --- entry point ----------------------------------------------------------

def load_spacy(model: str = "en_core_web_sm"):
    """Load the parser, or return None so callers can fall back to surface features."""
    try:
        import spacy
    except ImportError:
        return None
    try:
        return spacy.load(model, disable=["ner", "lemmatizer"])
    except OSError:
        return None


def parse_once(text: str, nlp: Any | None = None,
               max_chars: int = 200_000) -> tuple[Any | None, list[str]]:
    """Parse a document a single time and return (spacy_doc, sentences).

    Parsing dominates the cost of the CPU feature blocks: on a 4000 word paper it is
    roughly 940 ms against 145 ms for everything else combined. The pipeline needs both
    document features and windowed features, and computing them independently parses the
    same text twice, so nearly half the CPU time was spent producing a result that
    already existed.

    Callers parse here once and pass the result to both.
    """
    text = text[:max_chars]
    if nlp is not None:
        doc = nlp(text)
        sentences = [s.text.strip() for s in doc.sents
                     if len(words(s.text)) >= MIN_SENT_WORDS]
        return doc, sentences
    return None, split_sentences(text)


def extract_features(text: str, nlp: Any | None = None,
                     max_chars: int = 200_000,
                     parsed: tuple[Any | None, list[str]] | None = None
                     ) -> dict[str, float]:
    """All features for one document.

    Passing a loaded spaCy pipeline adds the syntax block. Without it the surface,
    lexical, punctuation, function word, discourse and tell blocks are still produced,
    which is most of the feature set.

    Pass `parsed` from parse_once() to avoid re-parsing when windowed features are also
    being computed for the same text.
    """
    text = text[:max_chars]
    doc, sentences = parsed if parsed is not None else parse_once(text, nlp, max_chars)

    feats: dict[str, float] = {}
    feats.update(surface_features(text, sentences))
    feats.update(lexical_features(text))
    feats.update(punctuation_features(text))
    feats.update(function_word_features(text))
    feats.update(discourse_features(text, sentences))
    feats.update(tell_features(text))
    feats.update(specificity_features(text))
    feats.update(register_features(text, sentences))
    if doc is not None:
        feats.update(syntax_features(doc))

    # Infinities become NaN, and NaN is preserved rather than flattened to zero. A
    # feature that could not be measured for this document must stay distinguishable
    # from one measured as zero; collapsing the two is how an undefined value turns
    # into a confident finding.
    out: dict[str, float] = {}
    for k, v in feats.items():
        if v is None:
            out[k] = float("nan")
        else:
            fv = float(v)
            out[k] = fv if math.isfinite(fv) else float("nan")
    return out


# Feature groups, used by analyze.py to report separation by category.
FEATURE_GROUPS = {
    "surface": ("sent_len", "para_sents", "para_words", "n_sentences", "n_words",
                "short_sent_frac", "long_sent_frac"),
    # Kept separate because it is a property of the file, not of the writing. It is
    # reported so a formatting difference between corpora is visible rather than
    # silently feeding the style features, but it never becomes a style rule.
    "formatting": ("has_paragraph_breaks",),
    "lexical": ("ttr_500", "mtld", "hapax_rate", "type_count", "word_len",
                "long_word_rate", "nominalisation_rate"),
    "punctuation": ("punct_",),
    "function": ("fw_",),
    "discourse": ("hedge_rate", "booster_rate", "transition_rate", "first_person_rate",
                  "tricolon_rate", "sent_initial_", "opener_entropy", "opener_top1_frac"),
    "tells": ("tell_",),
    "specificity": ("numeric_token_rate", "quantity_rate", "decimal_rate",
                    "acronym_rate", "digit_char_rate"),
    "syntax": ("dep_depth", "passive_per_clause", "subordination_rate",
               "coordination_rate", "noun_rate", "verb_rate", "adj_rate", "adv_rate",
               "adp_rate", "noun_verb_ratio", "opener_pos_entropy"),
    "lm": ("lm_",),
    "window": ("w_", "n_windows", "frac_uniform_windows", "longest_uniform_run"),
    "detector": ("det_",),
    "register": ("concessive_rate", "counterweight_rate", "contrastive_rate",
                 "attenuator_rate", "balanced_sentence_frac"),
}


def feature_group(name: str) -> str:
    for group, prefixes in FEATURE_GROUPS.items():
        if any(name.startswith(p) or name == p for p in prefixes):
            return group
    return "other"
