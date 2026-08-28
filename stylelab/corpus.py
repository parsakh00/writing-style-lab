"""Corpus loading and cleaning.

A Document is the unit everything else operates on. Text arrives from very different
places, JATS XML from PubMed Central, plain abstracts from arXiv, rows from a Hugging
Face dataset, and it all has to reach the feature extractor in the same shape or the
comparison is measuring provenance instead of style.

The cleaning is deliberately aggressive about anything that is an artefact of academic
publishing rather than a choice the writer made. Inline citations, figure callouts,
equation debris and DOIs all carry no stylistic signal and their density varies by
journal, so leaving them in would let journal conventions masquerade as authorship.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable, Iterator


@dataclass
class Document:
    """One text under measurement."""

    doc_id: str
    label: str  # "human" or "ai"
    source: str  # "pmc", "arxiv", "hc3", "raid", "gutenberg", "generated", "draft"
    text: str
    title: str = ""
    year: int | None = None
    meta: dict = field(default_factory=dict)

    def word_count(self) -> int:
        return len(self.text.split())


# --- cleaning -------------------------------------------------------------

# Numeric citations: [1], [1,2], [1-3], [1, 2, 5-9]
_RE_NUM_CITE = re.compile(r"\[\s*\d+(\s*[-,–]\s*\d+)*\s*\]")
# Author-year citations: (Smith et al., 2019), (Smith and Jones 2019; Lee 2020)
_RE_AUTHOR_CITE = re.compile(
    r"\((?:[^()]*?\b(?:et al\.?|and|&)\b[^()]*?\d{4}[a-z]?|[A-Z][A-Za-z'`-]+,?\s+\d{4}[a-z]?)"
    r"(?:\s*;\s*[^()]*?\d{4}[a-z]?)*\)"
)
# Figure / table / scheme / equation callouts
_RE_CALLOUT = re.compile(
    r"\(?\b(?:Fig(?:ure)?s?|Tables?|Schemes?|Eq(?:uation|n)?s?|Supplementary\s+\w+)\.?\s*"
    r"S?\d+[a-zA-Z]?(?:\s*[-–,]\s*S?\d+[a-zA-Z]?)*\)?",
    re.IGNORECASE,
)
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_DOI = re.compile(r"\b(?:doi:\s*)?10\.\d{4,9}/\S+", re.IGNORECASE)
_RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
# Leftover brackets after citation removal. Dropping <xref> children from "[1, 2]"
# leaves "[, ]", and from "(Smith 2019; Lee 2020)" leaves "(; )", so the separators have
# to be allowed inside as well as pure whitespace.
_RE_EMPTY_BRACKET = re.compile(r"\(\s*[;,\-–—\s]*\)|\[\s*[;,\-–—\s]*\]")
_RE_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")
_RE_MULTISPACE = re.compile(r"[ \t]+")
_RE_MULTINEWLINE = re.compile(r"\n{3,}")

# Boilerplate sections that are formulaic and would dilute the style signal.
_DROP_SECTIONS = {
    "acknowledgment",
    "acknowledgments",
    "acknowledgement",
    "acknowledgements",
    "references",
    "bibliography",
    "supporting information",
    "supplementary material",
    "supplementary materials",
    "conflict of interest",
    "conflicts of interest",
    "competing interests",
    "author contributions",
    "funding",
    "data availability",
    "abbreviations",
    "orcid",
}


def normalise_unicode(text: str) -> str:
    """Fold typographic variants so punctuation counts mean one thing.

    Curly quotes, exotic spaces and the several hyphen-like characters are folded to
    their plain equivalents, and zero width characters are deleted outright. The em
    dash and en dash are deliberately left alone: dash usage is one of the features
    under measurement, so collapsing them would erase signal.

    Every mapping is written as an escape sequence rather than a literal character.
    These characters are invisible or nearly invisible in a source file, and a silent
    copy-paste corruption here would quietly change the punctuation counts for the
    whole study.
    """
    text = unicodedata.normalize("NFKC", text)

    fold = {
        # single quotes and apostrophes
        "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'",
        # double quotes
        "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"', "\u2033": '"',
        # spaces of various widths -> plain space
        "\u00a0": " ", "\u2002": " ", "\u2003": " ", "\u2004": " ", "\u2005": " ",
        "\u2006": " ", "\u2007": " ", "\u2008": " ", "\u2009": " ", "\u200a": " ",
        "\u202f": " ", "\u205f": " ", "\u3000": " ",
        # zero width and byte order mark -> deleted
        "\u200b": "", "\u200c": "", "\u200d": "", "\u2060": "", "\ufeff": "",
        # hyphen-like -> ASCII hyphen (en dash \u2013 and em dash \u2014 kept distinct)
        "\u2010": "-", "\u2011": "-", "\u2212": "-", "\u00ad": "",
        # ligatures
        "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi",
        "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st",
        # ellipsis -> three dots so sentence splitting sees it consistently
        "\u2026": "...",
    }
    for bad, good in fold.items():
        if bad in text:
            text = text.replace(bad, good)
    return text


def clean_text(text: str, keep_citations: bool = False) -> str:
    """Strip publishing artefacts, keep the prose and its paragraph structure."""
    text = normalise_unicode(text)
    text = _RE_URL.sub(" ", text)
    text = _RE_DOI.sub(" ", text)
    text = _RE_EMAIL.sub(" ", text)
    if not keep_citations:
        text = _RE_AUTHOR_CITE.sub(" ", text)
        text = _RE_NUM_CITE.sub(" ", text)
    text = _RE_CALLOUT.sub(" ", text)
    text = _RE_EMPTY_BRACKET.sub(" ", text)

    # Collapse whitespace without destroying paragraph breaks.
    lines = [_RE_MULTISPACE.sub(" ", ln).strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = _RE_MULTINEWLINE.sub("\n\n", text)
    text = _RE_SPACE_BEFORE_PUNCT.sub(r"\1", text)
    return text.strip()


# Footnote markers publishers append to titles: dagger, double dagger, asterisk, section
# and pilcrow signs, sometimes several together. In PMC chemistry journals these are
# extremely common, and 43 of 60 sampled titles ended with one.
_RE_TITLE_FOOTNOTE = re.compile(r"[\s†‡§¶*★٭]+$")


def clean_title(title: str) -> str:
    """Normalise a title and strip publisher footnote markers.

    Titles are not just metadata here: they are the prompt that generates the matched
    AI corpus. A trailing dagger meaning "supplementary information available" is an
    artefact of the journal's typesetting, and passing it to a generator asks the model
    to account for a symbol that has nothing to do with the paper.
    """
    title = normalise_unicode(title)
    title = re.sub(r"\s+", " ", title).strip()
    return _RE_TITLE_FOOTNOTE.sub("", title).strip()


def is_boilerplate_heading(heading: str) -> bool:
    key = heading.strip().lower().rstrip(":.").strip()
    key = re.sub(r"^\d+[.)]?\s*", "", key)
    return key in _DROP_SECTIONS


# --- JATS full text -------------------------------------------------------

def _remove_preserving_tail(parent, child) -> None:
    """Drop an element without swallowing the prose that follows it.

    ElementTree stores the text between an element's close tag and its next sibling on
    the element itself, as .tail. A plain parent.remove(child) therefore deletes that
    text too. For an inline <xref> citation the tail is the rest of the sentence, so the
    naive removal quietly truncates prose at every citation in the corpus.

    The tail is reattached to the previous sibling, or to the parent's own text if the
    removed element was first.
    """
    tail = child.tail or ""
    if tail:
        siblings = list(parent)
        idx = siblings.index(child)
        if idx == 0:
            parent.text = (parent.text or "") + tail
        else:
            prev = siblings[idx - 1]
            prev.tail = (prev.tail or "") + tail
    parent.remove(child)


def jats_to_text(xml_string: str, drop_boilerplate: bool = True,
                 keep_citations: bool = False) -> tuple[str, str]:
    """Pull title and body prose out of a PMC JATS record.

    Returns (title, body_text). Tables, figures, formulae and reference lists are
    dropped entirely rather than flattened, because their text is not prose and would
    skew every length and lexical measure.
    """
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError:
        return "", ""

    title_el = root.find(".//article-title")
    title = clean_title("".join(title_el.itertext())) if title_el is not None else ""

    body = root.find(".//body")
    if body is None:
        return title, ""

    # Remove non-prose subtrees in place. ElementTree has no parent pointers, so walk
    # every element and drop offending children by identity.
    drop_tags = {
        "table-wrap", "table", "fig", "graphic", "disp-formula", "inline-formula",
        "media", "supplementary-material", "ref-list", "xref", "tex-math", "mml:math",
        "array", "alternatives",
    }
    for parent in body.iter():
        for child in list(parent):
            tag = child.tag.split("}")[-1]
            if tag == "xref" and keep_citations and child.get("ref-type", "bibr") == "bibr":
                # Keep a bibliographic citation as a bracketed marker so that where
                # papers cite, and how densely, can be measured. Figure and table
                # cross-references are still dropped.
                label = "".join(child.itertext()).strip()
                if not label.replace(",", "").replace("-", "").replace("–", "").replace(" ", "").isdigit():
                    label = "".join(ch for ch in (child.get("rid") or "") if ch.isdigit()) or "n"
                for sub in list(child):
                    child.remove(sub)
                child.text = f"[{label}]"
                continue
            if tag in drop_tags:
                _remove_preserving_tail(parent, child)

    chunks: list[str] = []
    for sec in body.iter():
        tag = sec.tag.split("}")[-1]
        if tag != "sec":
            continue
        head = sec.find("title")
        head_text = "".join(head.itertext()).strip() if head is not None else ""
        if drop_boilerplate and head_text and is_boilerplate_heading(head_text):
            continue
        for p in sec.findall("./p"):
            para = "".join(p.itertext()).strip()
            if para:
                chunks.append(para)

    # Some records put paragraphs straight under <body> with no sections.
    for p in body.findall("./p"):
        para = "".join(p.itertext()).strip()
        if para:
            chunks.append(para)

    return title, clean_text("\n\n".join(chunks), keep_citations=keep_citations)


# --- persistence ----------------------------------------------------------

def write_jsonl(docs: Iterable[Document], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[Document]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield Document(**json.loads(line))


def load_all(data_dir: str | Path, min_words: int = 200) -> list[Document]:
    """Load every corpus shard under data_dir, filtering out fragments.

    Documents shorter than min_words are dropped. Most of the stylometric measures,
    type token ratio and the function word vector especially, are unstable on short
    text and would add noise that looks like signal.
    """
    data_dir = Path(data_dir)
    docs: list[Document] = []
    for shard in sorted(data_dir.glob("*.jsonl")):
        for doc in read_jsonl(shard):
            if doc.word_count() >= min_words:
                docs.append(doc)
    return docs
