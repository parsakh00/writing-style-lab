"""Build the human corpus from the PubMed Central Open Access subset.

Two constraints define this corpus and both matter.

Pre-2020 only. The cutoff is not about the release date of any particular model, it is
about when LLM assisted drafting became common enough to contaminate a sample. Anything
published before 2020 was written before that was possible at scale, which makes the
human label trustworthy by construction rather than by assumption.

Chemistry, materials and adsorption topics only. The comparison downstream is against
model written text on the same subjects. Matching the register is what makes the
measured difference attributable to authorship rather than to genre, and a corpus of
clinical trial reports would fail that test even though it is equally human.

Usage:
    python scripts/fetch_pmc.py --max-docs 800
    python scripts/fetch_pmc.py --max-docs 2000 --api-key $NCBI_API_KEY
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stylelab.corpus import Document, jats_to_text, write_jsonl  # noqa: E402

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Topic terms chosen to match the register of porous materials and gas adsorption work.
#
# Kept deliberately narrow. An earlier, looser list included generic method words like
# "spectroscopy", "crystal structure" and "catalysis", which match a large fraction of
# PMC and pulled in clinical and nutrition papers. Those are equally human, but they are
# a different register, and register drift in the reference corpus would show up
# downstream as a difference the model supposedly has.
DEFAULT_TOPICS = [
    "metal-organic framework", "metal organic framework", "metal-organic frameworks",
    "covalent organic framework", "zeolite", "zeolitic imidazolate framework",
    "gas adsorption", "adsorption isotherm", "adsorbent", "physisorption",
    "chemisorption", "gas separation", "carbon capture", "CO2 capture",
    "porous material", "porous materials", "microporous", "mesoporous",
    "molecular sieve", "activated carbon", "pore volume", "surface area BET",
    "gas storage", "hydrogen storage", "methane storage", "framework material",
]


def build_query(topics: list[str], year_min: int, year_max: int) -> str:
    topic_clause = " OR ".join(f'"{t}"[Title/Abstract]' for t in topics)
    return (
        f'("open access"[filter]) AND '
        f'("{year_min}/01/01"[PDAT] : "{year_max}/12/31"[PDAT]) AND '
        f'({topic_clause})'
    )


def _params(api_key: str | None, **kw) -> dict:
    p = {"tool": "writing-style-lab", "email": "", **kw}
    if api_key:
        p["api_key"] = api_key
    return {k: v for k, v in p.items() if v != ""}


def esearch(query: str, retmax: int, api_key: str | None) -> list[str]:
    """Collect PMC ids, paging because esearch caps retmax at 10000 per call."""
    ids: list[str] = []
    retstart = 0
    page = min(retmax, 5000)

    while len(ids) < retmax:
        r = requests.get(
            f"{EUTILS}/esearch.fcgi",
            params=_params(api_key, db="pmc", term=query, retmode="json",
                           retmax=page, retstart=retstart, sort="pub+date"),
            timeout=60,
        )
        r.raise_for_status()
        result = r.json().get("esearchresult", {})
        batch = result.get("idlist", [])
        if not batch:
            break
        ids.extend(batch)
        retstart += len(batch)
        if retstart >= int(result.get("count", 0)):
            break
        time.sleep(0.12 if api_key else 0.35)

    return ids[:retmax]


def efetch_batch(pmcids: list[str], api_key: str | None) -> str:
    r = requests.get(
        f"{EUTILS}/efetch.fcgi",
        params=_params(api_key, db="pmc", id=",".join(pmcids), retmode="xml"),
        timeout=120,
    )
    r.raise_for_status()
    return r.text


def split_articles(xml: str) -> list[str]:
    """Split a multi-article efetch response into individual article records."""
    import re

    parts = re.split(r"(?=<article[\s>])", xml)
    return [p for p in parts if p.strip().startswith("<article")]


def extract_year(article_xml: str) -> int | None:
    import re

    m = re.search(r"<pub-date[^>]*>.*?<year>(\d{4})</year>", article_xml, re.S)
    return int(m.group(1)) if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-docs", type=int, default=800)
    ap.add_argument("--year-min", type=int, default=2005)
    ap.add_argument("--year-max", type=int, default=2019,
                    help="hard upper bound; keep below 2020 to guarantee clean labels")
    ap.add_argument("--min-words", type=int, default=600)
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--out", default="data/human_pmc.jsonl")
    ap.add_argument("--api-key", default=os.environ.get("NCBI_API_KEY"),
                    help="NCBI key raises the rate limit from 3/s to 10/s")
    args = ap.parse_args()

    if args.year_max >= 2020:
        print(f"refusing year-max={args.year_max}: the human label is only trustworthy "
              f"before 2020. Pass --year-max 2019 or lower.", file=sys.stderr)
        return 2

    query = build_query(DEFAULT_TOPICS, args.year_min, args.year_max)
    print(f"query: {query[:160]}...")

    # Over-fetch ids: many records have no retrievable full text body, and some come
    # back too short once tables and references are stripped.
    want_ids = min(args.max_docs * 3, 20000)
    print(f"searching for up to {want_ids} ids...")
    ids = esearch(query, want_ids, args.api_key)
    print(f"got {len(ids)} ids")
    if not ids:
        print("no results; check the query or your network", file=sys.stderr)
        return 1

    docs: list[Document] = []
    delay = 0.12 if args.api_key else 0.35

    for i in range(0, len(ids), args.batch):
        if len(docs) >= args.max_docs:
            break
        batch = ids[i:i + args.batch]
        try:
            xml = efetch_batch(batch, args.api_key)
        except Exception as exc:  # noqa: BLE001
            print(f"  batch {i} failed ({type(exc).__name__}), skipping")
            time.sleep(2.0)
            continue

        for pmcid, article in zip(batch, split_articles(xml)):
            title, body = jats_to_text(article)
            if not body or len(body.split()) < args.min_words:
                continue
            year = extract_year(article)
            if year is not None and year > args.year_max:
                continue
            docs.append(Document(
                doc_id=f"pmc:{pmcid}",
                label="human",
                source="pmc",
                text=body,
                title=title,
                year=year,
                meta={"pmcid": pmcid},
            ))
            if len(docs) >= args.max_docs:
                break

        print(f"  {len(docs)}/{args.max_docs} kept "
              f"(scanned {min(i + args.batch, len(ids))} ids)", end="\r")
        time.sleep(delay)

    print()
    if not docs:
        print("nothing usable retrieved", file=sys.stderr)
        return 1

    n = write_jsonl(docs, args.out)
    years = [d.year for d in docs if d.year]
    total_words = sum(d.word_count() for d in docs)
    print(f"wrote {n} documents to {args.out}")
    print(f"  years {min(years)}-{max(years)}" if years else "  years unknown")
    print(f"  {total_words:,} words, median {total_words // n:,} per document")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
