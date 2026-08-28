"""Open detectors, run locally, used as measuring instruments.

These exist to answer one question: does an independent detector agree with the feature
analysis about which corpus is which, and which features move together with its score?
That is validation. If the discriminative features found in analyze.py also predict what
these detectors do, the finding is about the prose. If they do not, the finding is about
this particular corpus and should be distrusted.

They are deliberately not wired into score.py's revision loop. Scoring a draft, editing,
rescoring and repeating until a detector's number falls is a different activity from
measuring, and it is not what this repository does.

Three methods, chosen because they are open, run locally, and their mechanisms are
inspectable:

  Binoculars      Hans et al. 2024. Ratio of a text's perplexity under one model to the
                  cross perplexity between two models. The denominator normalises away
                  "this text is about an unusual topic", which is the failure mode that
                  makes raw perplexity biased against non-native and technical writing.
                  Currently the strongest zero shot method.
  Fast-DetectGPT  Bao et al. 2024. Conditional probability curvature, computed
                  analytically from one model's own distribution rather than by
                  perturbing and rescoring, which is what made the original DetectGPT
                  expensive.
  RoBERTa         The OpenAI released detector. A trained classifier rather than a zero
                  shot statistic, included as a contrasting baseline.

Model choices here are smaller than the papers use. Binoculars was published with a
Falcon 7B pair; this uses a Pythia pair, which shares a tokenizer and runs on a laptop.
Absolute scores therefore will not match published thresholds. Relative separation
between two corpora, which is all this is used for, survives that substitution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .lm import dtype_kwargs

# Both models must share a tokenizer for cross perplexity to be defined at all.
BINOCULARS_OBSERVER = "EleutherAI/pythia-160m"
BINOCULARS_PERFORMER = "EleutherAI/pythia-410m"
FASTDETECT_MODEL = "EleutherAI/pythia-160m"
ROBERTA_DETECTOR = "openai-community/roberta-base-openai-detector"

MAX_LEN = 512


def _device(explicit: str | None = None) -> str:
    import torch

    if explicit:
        return explicit
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Binoculars:
    """Perplexity over cross perplexity. Lower means more machine-like."""

    observer_name: str = BINOCULARS_OBSERVER
    performer_name: str = BINOCULARS_PERFORMER
    device: str | None = None
    _obs: object | None = field(default=None, repr=False)
    _perf: object | None = field(default=None, repr=False)
    _tok: object | None = field(default=None, repr=False)

    def load(self) -> "Binoculars":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = _device(self.device)
        kw = dtype_kwargs(self.device)
        self._tok = AutoTokenizer.from_pretrained(self.observer_name)
        self._obs = AutoModelForCausalLM.from_pretrained(
            self.observer_name, **kw).to(self.device).eval()
        self._perf = AutoModelForCausalLM.from_pretrained(
            self.performer_name, **kw).to(self.device).eval()

        # Cross perplexity is only defined if both models score the same token stream.
        # Two checkpoints from the same family usually share a tokenizer, but a swapped
        # model name would otherwise produce confidently meaningless numbers instead of
        # an error.
        other = AutoTokenizer.from_pretrained(self.performer_name)
        if other.get_vocab() != self._tok.get_vocab():
            raise ValueError(
                f"observer ({self.observer_name}) and performer ({self.performer_name}) "
                f"do not share a tokenizer; Binoculars cross perplexity is undefined"
            )
        return self

    def score(self, text: str) -> float:
        import torch

        if self._obs is None:
            self.load()

        ids = self._tok(text, return_tensors="pt", truncation=True,
                        max_length=MAX_LEN).input_ids.to(self.device)
        if ids.shape[1] < 2:
            return float("nan")

        with torch.no_grad():
            obs_logits = self._obs(ids).logits[0, :-1].float()
            perf_logits = self._perf(ids).logits[0, :-1].float()

        targets = ids[0, 1:]
        perf_logprobs = torch.log_softmax(perf_logits, dim=-1)

        # Numerator: the text's own negative log likelihood under the performer.
        log_ppl = -perf_logprobs.gather(1, targets.unsqueeze(1)).squeeze(1).mean()

        # Denominator: expected surprise of the observer's *distribution* under the
        # performer. This is the whole trick. It measures how surprising this context
        # is in general, so dividing by it removes the topic difficulty that would
        # otherwise dominate the numerator.
        obs_probs = torch.softmax(obs_logits, dim=-1)
        x_ppl = -(obs_probs * perf_logprobs).sum(dim=-1).mean()

        if x_ppl.item() == 0:
            return float("nan")
        return float(log_ppl.item() / x_ppl.item())


@dataclass
class FastDetectGPT:
    """Conditional probability curvature. Higher means more machine-like."""

    model_name: str = FASTDETECT_MODEL
    device: str | None = None
    _model: object | None = field(default=None, repr=False)
    _tok: object | None = field(default=None, repr=False)

    def load(self) -> "FastDetectGPT":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = _device(self.device)
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **dtype_kwargs(self.device)).to(self.device).eval()
        return self

    def score(self, text: str) -> float:
        import torch

        if self._model is None:
            self.load()

        ids = self._tok(text, return_tensors="pt", truncation=True,
                        max_length=MAX_LEN).input_ids.to(self.device)
        if ids.shape[1] < 2:
            return float("nan")

        with torch.no_grad():
            logits = self._model(ids).logits[0, :-1].float()

        targets = ids[0, 1:]
        logprobs = torch.log_softmax(logits, dim=-1)
        probs = logprobs.exp()

        observed = logprobs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Under the model's own distribution, the expected log probability of a sampled
        # token is the negative entropy, and its variance follows in closed form. The
        # z-score of what the text actually did against that reference is the curvature.
        # Machine text sits near the model's own expectation; human text sits below it.
        mu = (probs * logprobs).sum(dim=-1)
        var = (probs * logprobs.pow(2)).sum(dim=-1) - mu.pow(2)

        denom = float(var.sum().clamp(min=1e-9).sqrt().item())
        return float((observed.sum().item() - mu.sum().item()) / denom)


@dataclass
class RobertaDetector:
    """OpenAI's released classifier. Returns the probability of class index 1.

    The label orientation of this checkpoint is a recurring source of confusion, so no
    orientation is assumed here. The raw class 1 probability is reported and analyze.py
    determines the direction empirically from the corpora, whose labels are known. A
    detector whose orientation has to be inferred is still usable; one whose orientation
    is guessed wrongly silently inverts every conclusion drawn from it.
    """

    model_name: str = ROBERTA_DETECTOR
    device: str | None = None
    _model: object | None = field(default=None, repr=False)
    _tok: object | None = field(default=None, repr=False)

    def load(self) -> "RobertaDetector":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = _device(self.device)
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name).to(self.device).eval()
        return self

    def score(self, text: str) -> float:
        import torch

        if self._model is None:
            self.load()

        # Long documents are chunked and averaged rather than truncated, so a verdict
        # reflects the whole text instead of its first few hundred tokens.
        ids = self._tok(text, return_tensors="pt", truncation=False).input_ids[0]
        chunks = [ids[i:i + MAX_LEN] for i in range(0, ids.shape[0], MAX_LEN)]
        chunks = [c for c in chunks if c.shape[0] >= 16] or [ids[:MAX_LEN]]

        probs: list[float] = []
        for chunk in chunks:
            with torch.no_grad():
                logits = self._model(chunk.unsqueeze(0).to(self.device)).logits
            probs.append(float(torch.softmax(logits, dim=-1)[0, 1].item()))
        return float(np.mean(probs)) if probs else float("nan")


def detector_features(text: str, detectors: dict[str, object]) -> dict[str, float]:
    """Run each loaded detector over one text, tolerating individual failures."""
    out: dict[str, float] = {}
    for name, det in detectors.items():
        try:
            out[f"det_{name}"] = det.score(text)
        except Exception as exc:  # noqa: BLE001 - one bad model must not lose the run
            out[f"det_{name}"] = float("nan")
            out[f"det_{name}_error"] = 1.0
            print(f"  detector {name} failed: {type(exc).__name__}: {exc}")
    return out


def load_detectors(which: tuple[str, ...] = ("binoculars", "fastdetectgpt", "roberta"),
                   device: str | None = None) -> dict[str, object]:
    available = {
        "binoculars": lambda: Binoculars(device=device).load(),
        "fastdetectgpt": lambda: FastDetectGPT(device=device).load(),
        "roberta": lambda: RobertaDetector(device=device).load(),
    }
    loaded: dict[str, object] = {}
    for name in which:
        if name not in available:
            raise ValueError(f"unknown detector: {name}")
        print(f"loading detector: {name}")
        loaded[name] = available[name]()
    return loaded
