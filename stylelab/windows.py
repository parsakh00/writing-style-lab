"""Sliding-window features.

Document level averages are the wrong unit of analysis for this problem, and the reason
is architectural rather than a matter of taste.

Turnitin's published model architecture scores a segment window of roughly a few hundred
words, about five to ten sentences, striding across the document one sentence at a time.
Each sentence inherits the score of every window containing it, those scores are pooled,
and the document is called AI generated when more than a fifth of its sentences clear a
threshold. Their document verdict is therefore a statement about the *upper tail* of a
window distribution, not about the document mean.

That geometry is replicated here because it is the scale at which the phenomenon lives,
not because anything is being fitted to their labels. A paper can have an unremarkable
mean and still contain long stretches of very uniform prose, and a document level mean
averages exactly that away.

The summary statistics that matter are therefore the tail ones: worst_* and
frac_below_* rather than the mean.
"""

from __future__ import annotations

from typing import Any, Iterator, Sequence

import numpy as np

from .features import parse_once, words

# Matches the published segment geometry: a few hundred words, five to ten sentences.
DEFAULT_WINDOW_SENTS = 8
DEFAULT_STRIDE_SENTS = 1


def iter_windows(
    sentences: Sequence[str],
    window: int = DEFAULT_WINDOW_SENTS,
    stride: int = DEFAULT_STRIDE_SENTS,
) -> Iterator[tuple[int, list[str]]]:
    """Yield (start_index, sentences) for each overlapping window.

    A document shorter than one window yields a single truncated window rather than
    nothing, so short drafts still produce a score instead of silently disappearing.
    """
    n = len(sentences)
    if n == 0:
        return
    if n <= window:
        yield 0, list(sentences)
        return
    for start in range(0, n - window + 1, stride):
        yield start, list(sentences[start : start + window])


def window_metrics(sents: Sequence[str]) -> dict[str, float]:
    """The handful of measures that are stable on a few hundred words.

    Deliberately a small set. MTLD, the function word vector and the dependency
    statistics all need far more text than one window to be meaningful, and computing
    them here would produce numbers that look like features but are mostly sampling
    noise.
    """
    lens = [len(words(s)) for s in sents]
    text = " ".join(sents)
    n_words = len(words(text))
    if n_words == 0:
        return {}

    arr = np.asarray(lens, dtype=float)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0

    toks = [t.lower() for t in words(text)]
    ttr = len(set(toks)) / len(toks) if toks else 0.0

    # Opener repetition inside the window. Several sentences in a row opening the same
    # way is one of the most visible forms of local uniformity.
    heads = [words(s)[0].lower() for s in sents if words(s)]
    head_repeat = 1.0 - (len(set(heads)) / len(heads)) if heads else 0.0

    return {
        "w_sent_len_mean": mean,
        "w_sent_len_sd": sd,
        "w_sent_len_cv": sd / mean if mean > 0 else 0.0,
        "w_sent_len_range": float(arr.max() - arr.min()),
        "w_neighbour_delta": float(np.abs(np.diff(arr)).mean()) if arr.size > 1 else 0.0,
        "w_ttr": ttr,
        "w_head_repeat": head_repeat,
        "w_comma_rate": 1000.0 * text.count(",") / n_words,
        "w_n_words": float(n_words),
    }


def _tail_summary(values: Sequence[float], name: str, low_is_uniform: bool) -> dict[str, float]:
    """Collapse a per-window series into distribution and tail statistics.

    low_is_uniform says which end of the distribution represents machine-like
    uniformity, so the "worst window" is picked from the correct tail. For variation
    measures like sentence length CV, low means uniform; for repetition measures like
    head_repeat, high means uniform.
    """
    if not values:
        return {}
    arr = np.asarray(values, dtype=float)
    out = {
        f"{name}_mean": float(arr.mean()),
        f"{name}_sd": float(arr.std()),
    }
    if low_is_uniform:
        out[f"{name}_worst"] = float(np.percentile(arr, 5))
        out[f"{name}_p25"] = float(np.percentile(arr, 25))
    else:
        out[f"{name}_worst"] = float(np.percentile(arr, 95))
        out[f"{name}_p75"] = float(np.percentile(arr, 75))
    return out


# Which direction means "uniform" for each window metric.
_LOW_IS_UNIFORM = {
    "w_sent_len_sd": True,
    "w_sent_len_cv": True,
    "w_sent_len_range": True,
    "w_neighbour_delta": True,
    "w_ttr": True,
    "w_head_repeat": False,
    "w_sent_len_mean": True,
    "w_comma_rate": True,
    "w_n_words": True,
}


def windowed_features(
    text: str,
    nlp: Any | None = None,
    window: int = DEFAULT_WINDOW_SENTS,
    stride: int = DEFAULT_STRIDE_SENTS,
    cv_uniform_threshold: float = 0.35,
    sentences: Sequence[str] | None = None,
) -> dict[str, float]:
    """Per-document features derived from the sliding window series.

    cv_uniform_threshold marks a window as uniform when its sentence length coefficient
    of variation falls below it. The default is a placeholder to be replaced by a
    quantile of the measured human distribution once analyze.py has run; hard coding a
    guess would bake an assumption into the very thing being measured.

    Pass `sentences` from parse_once() to reuse an existing parse instead of paying for
    a second one.
    """
    if sentences is None:
        _, sentences = parse_once(text, nlp)

    series: dict[str, list[float]] = {}
    n_windows = 0
    for _, sents in iter_windows(sentences, window, stride):
        m = window_metrics(sents)
        if not m:
            continue
        n_windows += 1
        for k, v in m.items():
            series.setdefault(k, []).append(v)

    feats: dict[str, float] = {"n_windows": float(n_windows)}
    if n_windows == 0:
        return feats

    for k, vals in series.items():
        feats.update(_tail_summary(vals, k, _LOW_IS_UNIFORM.get(k, True)))

    # The direct analogue of the document verdict: what share of windows are uniform.
    cvs = series.get("w_sent_len_cv", [])
    if cvs:
        arr = np.asarray(cvs)
        feats["frac_uniform_windows"] = float((arr < cv_uniform_threshold).mean())
        # Longest unbroken run of uniform windows. A fifth of a document being uniform
        # reads very differently when it is one contiguous block than when it is
        # scattered, and the fraction alone cannot tell those apart.
        longest, current = 0, 0
        for v in arr:
            current = current + 1 if v < cv_uniform_threshold else 0
            longest = max(longest, current)
        feats["longest_uniform_run"] = float(longest)
        feats["longest_uniform_run_frac"] = longest / len(arr)

    return feats


def window_profile(
    text: str,
    nlp: Any | None = None,
    window: int = DEFAULT_WINDOW_SENTS,
    stride: int = DEFAULT_STRIDE_SENTS,
    sentences: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Per-window records for a single document, for score.py to point at line numbers.

    Returns the window's start sentence index, its metrics, and its opening text, so a
    report can say which passage is uniform rather than only that some passage is.
    """
    if sentences is None:
        _, sentences = parse_once(text, nlp)

    out: list[dict[str, Any]] = []
    for start, sents in iter_windows(sentences, window, stride):
        m = window_metrics(sents)
        if not m:
            continue
        out.append({
            "start_sentence": start,
            "n_sentences": len(sents),
            "preview": (sents[0][:110] + "...") if sents and len(sents[0]) > 110
            else (sents[0] if sents else ""),
            **m,
        })
    return out
