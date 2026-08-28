"""Client for the GPTZero detection API.

This exists because five drafts were called machine-written while three real papers of
the same register were not, and four hypotheses drawn from the tool's own explanations
were all wrong. Those explanations are generated after the classifier returns a score;
they describe formal technical writing, which is what the text is, and they are not the
mechanism. Reasoning from them produced four failures in a row.

The point of an API is to stop reasoning from examples and correlate the actual score
against features measured across hundreds of documents.

Two things are deliberate. Scores are never fed back into a revision loop; the client
measures and returns. And every call is counted against an explicit word budget, because
the account is metered and a careless sweep over the full corpus would spend it.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

ENDPOINT = "https://api.gptzero.me/v2/predict/text"
# Per-document ceiling on the paid plans. Roughly 25,000 words, so a whole paper fits in
# one call and the 1000-word excerpts used here sit at about 4% of the limit.
MAX_CHARS_PER_DOC = 150_000
KEY_FILE = ".gptzero_key"


def load_key(explicit: str | None = None) -> str:
    """Find the key without ever printing it."""
    if explicit:
        return explicit.strip()
    env = os.environ.get("GPTZERO_API_KEY")
    if env:
        return env.strip()
    for p in (Path(KEY_FILE), Path(__file__).resolve().parents[1] / KEY_FILE):
        if p.exists():
            key = p.read_text(encoding="utf-8").strip()
            if key:
                return key
    raise RuntimeError(
        f"no API key found. Write it to {KEY_FILE} in the repository root, "
        f"or set GPTZERO_API_KEY."
    )


@dataclass
class GPTZero:
    """Minimal client with a hard word budget."""

    api_key: str = ""
    budget_words: int = 150_000
    pause: float = 0.6
    spent_words: int = 0
    calls: int = 0
    _last_raw: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if not self.api_key:
            self.api_key = load_key()

    def score(self, text: str) -> dict:
        """Score one document. Returns a flat dict; raises if the budget is exhausted."""
        import requests

        if len(text) > MAX_CHARS_PER_DOC:
            raise ValueError(
                f"document is {len(text):,} characters, over the {MAX_CHARS_PER_DOC:,} "
                f"per-document limit. Split it before scoring."
            )
        n = len(text.split())
        if self.spent_words + n > self.budget_words:
            raise RuntimeError(
                f"word budget exhausted: {self.spent_words:,} spent, {n:,} more "
                f"requested, limit {self.budget_words:,}. Raise --budget deliberately."
            )

        r = requests.post(
            ENDPOINT,
            headers={"x-api-key": self.api_key,
                     "Content-Type": "application/json",
                     "Accept": "application/json"},
            json={"document": text},
            timeout=120,
        )
        r.raise_for_status()
        payload = r.json()
        self._last_raw = payload
        self.spent_words += n
        self.calls += 1
        time.sleep(self.pause)
        return self._flatten(payload, n)

    @staticmethod
    def _flatten(payload: dict, n_words: int) -> dict:
        """Pull the fields we need without assuming the whole schema.

        The response shape is documented but versioned, so each field is looked up
        defensively and a miss becomes None rather than an exception. The raw payload is
        kept on the client for inspection when something is missing.
        """
        docs = payload.get("documents") or []
        d = docs[0] if docs else {}
        probs = d.get("class_probabilities") or {}
        sents = d.get("sentences") or []
        sent_scores = [s.get("generated_prob") for s in sents
                       if isinstance(s.get("generated_prob"), (int, float))]
        out = {
            "n_words": n_words,
            "completely_generated_prob": d.get("completely_generated_prob"),
            "prob_ai": probs.get("ai"),
            "prob_human": probs.get("human"),
            "prob_mixed": probs.get("mixed"),
            "predicted_class": d.get("predicted_class"),
            "n_sentences_scored": len(sent_scores),
        }
        if sent_scores:
            ordered = sorted(sent_scores)
            k = len(ordered)
            out.update({
                "sent_prob_mean": sum(ordered) / k,
                "sent_prob_median": ordered[k // 2],
                "sent_prob_p90": ordered[int(0.9 * (k - 1))],
                # GPTZero's own document rule is a share of flagged sentences, so the
                # share above threshold is closer to what it reports than the mean.
                "sent_frac_over_half": sum(1 for x in ordered if x > 0.5) / k,
            })
        return out

    def budget_left(self) -> int:
        return max(0, self.budget_words - self.spent_words)
