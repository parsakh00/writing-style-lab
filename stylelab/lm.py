"""Language model derived features: perplexity, surprise dispersion, token rank bins.

This is the block that most directly corresponds to what a learned detector responds to.
Turnitin's classifier is a fine tuned transformer rather than a hand built feature model,
so nothing here reproduces it. What these features do capture is the underlying quantity
their architecture description points at: how predictable each token is given its
context, and how much that predictability varies across the text.

Three things are computed per document:

  perplexity          the classic aggregate. Low means the text is easy to predict.
  log prob dispersion how uneven the surprise is. Two texts can share a mean perplexity
                      while one is uniformly mid-surprise and the other alternates
                      between the obvious and the unexpected. The second is what human
                      prose tends to look like, and the mean cannot see the difference.
  rank bins           the GLTR view (Gehrmann, Strobelt and Rush 2019). For each token,
                      where did it sit in the model's ranked prediction? Sampled text
                      lives disproportionately in the top 10.

GPT-2 small is the default because it is what the detection literature standardised on,
which makes numbers here comparable to published work. It is a weak model by current
standards and its surprise estimates are correspondingly coarse. Swap it with --lm-model
if you want a sharper instrument; the features keep their meaning, the absolute values
do not transfer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

DEFAULT_LM = "gpt2"
# GPT-2's context is 1024. Windows stride with overlap so that tokens near a window
# boundary are still scored with real left context rather than from a cold start, which
# would otherwise register as spurious surprise at fixed intervals through the document.
DEFAULT_MAX_LEN = 512
DEFAULT_STRIDE = 256

GLTR_BINS = (10, 100, 1000)


def dtype_kwargs(device: str) -> dict:
    """Weight dtype argument, named correctly for the installed transformers.

    The keyword was renamed from torch_dtype to dtype in transformers 5. Passing the
    old name still works but warns, and will eventually stop working; passing the new
    name outright breaks on 4.x. Picking by version keeps both usable, which matters
    because the cluster and the laptop are rarely on the same release.

    Half precision only on GPU. On CPU, float16 matmuls are emulated and can be slower
    than float32 as well as less accurate, and accuracy is the point here.
    """
    import torch
    import transformers

    dtype = torch.float32 if device == "cpu" else torch.float16
    try:
        major = int(transformers.__version__.split(".")[0])
    except (ValueError, IndexError):
        major = 4
    return {"dtype": dtype} if major >= 5 else {"torch_dtype": dtype}


@dataclass
class LMScorer:
    """Holds a causal LM and produces token level statistics for a text."""

    model_name: str = DEFAULT_LM
    device: str | None = None
    max_length: int = DEFAULT_MAX_LEN
    stride: int = DEFAULT_STRIDE
    _model: object | None = None
    _tok: object | None = None

    def load(self) -> "LMScorer":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **dtype_kwargs(self.device)
        )
        self._model.to(self.device)
        self._model.eval()
        return self

    def token_stats(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (log_probs, ranks) for every scored token in the text.

        Both arrays are aligned and cover each token once. Overlapping windows are used
        for context but only the newly exposed tokens in each window are recorded, so no
        token contributes twice and none is scored without left context except the very
        first few.
        """
        import torch

        if self._model is None:
            self.load()

        ids = self._tok(text, return_tensors="pt").input_ids[0]
        n = ids.shape[0]
        if n < 2:
            return np.array([]), np.array([])

        all_lp: list[np.ndarray] = []
        all_rank: list[np.ndarray] = []
        prev_end = 0
        start = 0

        while start < n - 1:
            end = min(start + self.max_length, n)
            chunk = ids[start:end].unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self._model(chunk).logits[0].float()

            # Position i predicts token i+1.
            logits = logits[:-1]
            targets = chunk[0][1:]

            # The log probability of the realised token, without materialising the full
            # log-softmax. That tensor is [sequence, vocabulary], about 100 MB per
            # window at this context length, and only one column of it is ever read.
            # Subtracting the log-sum-exp gives the same number from two [sequence]
            # vectors.
            tok_logit = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
            tok_lp = tok_logit - torch.logsumexp(logits, dim=-1)

            # Rank of the realised token, 1 = most likely. Counting how many tokens beat
            # it is far cheaper than a full argsort, and the comparison runs on the raw
            # logits because softmax is monotonic and therefore rank preserving.
            better = (logits > tok_logit.unsqueeze(1)).sum(dim=1)
            tok_rank = (better + 1).float()

            # Absolute index of the token each prediction refers to.
            first_target = start + 1
            keep_from = max(prev_end, first_target)
            offset = keep_from - first_target
            if offset < tok_lp.shape[0]:
                all_lp.append(tok_lp[offset:].cpu().numpy())
                all_rank.append(tok_rank[offset:].cpu().numpy())
                prev_end = end

            if end == n:
                break
            start += self.stride

        if not all_lp:
            return np.array([]), np.array([])
        return np.concatenate(all_lp), np.concatenate(all_rank)

    def features(self, text: str) -> dict[str, float]:
        lp, rank = self.token_stats(text)
        if lp.size == 0:
            return {}

        nll = -lp
        feats = {
            "lm_ppl": float(np.exp(nll.mean())),
            "lm_logprob_mean": float(lp.mean()),
            "lm_logprob_sd": float(lp.std()),
            # Dispersion normalised by level, so it is not just a restatement of ppl.
            "lm_logprob_cv": float(lp.std() / abs(lp.mean())) if lp.mean() != 0 else 0.0,
            "lm_logprob_p10": float(np.percentile(lp, 10)),
            "lm_logprob_p50": float(np.percentile(lp, 50)),
            "lm_logprob_p90": float(np.percentile(lp, 90)),
            "lm_logprob_iqr": float(np.percentile(lp, 75) - np.percentile(lp, 25)),
            # How often the text does something the model finds genuinely unlikely.
            "lm_surprise_frac": float((nll > 5.0).mean()),
            "lm_n_tokens": float(lp.size),
        }

        # Mean absolute step between consecutive tokens' surprise. This is the token
        # level analogue of sentence length burstiness: it asks whether the text moves
        # between predictable and unpredictable, or sits at one level throughout.
        if lp.size > 1:
            feats["lm_logprob_neighbour_delta"] = float(np.abs(np.diff(lp)).mean())
        else:
            feats["lm_logprob_neighbour_delta"] = 0.0

        # GLTR rank bins.
        prev = 0
        for b in GLTR_BINS:
            feats[f"lm_rank_top{b}_frac"] = float(((rank > prev) & (rank <= b)).mean())
            prev = b
        feats[f"lm_rank_beyond{GLTR_BINS[-1]}_frac"] = float((rank > GLTR_BINS[-1]).mean())
        feats["lm_log_rank_mean"] = float(np.log(rank).mean())
        feats["lm_log_rank_sd"] = float(np.log(rank).std())

        return feats

    def batch_features(self, texts: Iterable[str]) -> list[dict[str, float]]:
        if self._model is None:
            self.load()
        return [self.features(t) for t in texts]
