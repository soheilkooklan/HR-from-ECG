"""
Training objective for HR-from-ECG v2.

    L = L_beat + a*L_offset + b*L_quality + c*L_rhythm + d*L_quantile

The first three terms are conventional. The fourth is the contribution that
makes the model physiology-informed, and it needs justifying.

**The rhythm prior.** The obvious way to inject cardiac structure is to
penalise implausible RR intervals -- but RR intervals are computed from
*discrete* peak positions, so that penalty has no gradient. The trick used here
is to impose the constraint one level earlier, on the continuous beat
probability map itself. A healthy beat train is pseudo-periodic, so the
normalised autocorrelation of the probability map must have a sharp peak at the
lag corresponding to the mean RR interval. The loss rewards exactly that:

    L_rhythm = 1 - max_{lag in physiological range} rho(lag)

This is fully differentiable, requires no labels, and turns out to matter most
precisely where the supervised term is weakest: under heavy artefact, where
false detections are spread randomly in time and therefore destroy periodicity,
whereas true beats preserve it. Because it needs no ground truth it also works
unchanged as a semi-supervised term on unlabelled records.

**Why focal + soft Dice for the beat head.** The infrastructure study showed
that the failure mode of classical detection under noise is false positives,
not missed beats. Dice on soft Gaussian targets penalises precision and recall
symmetrically, and focal weighting concentrates capacity on the ambiguous
candidates rather than the thousands of trivially-negative samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["HRLoss", "LossWeights", "soft_dice_loss", "focal_bce",
           "rhythm_prior_loss", "pinball_loss"]


@dataclass
class LossWeights:
    beat: float = 1.0
    offset: float = 0.5
    quality: float = 1.0
    rhythm: float = 0.2
    quantile: float = 0.05


def soft_dice_loss(prob: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Dice on soft targets, computed per example then averaged."""
    num = 2.0 * (prob * target).sum(dim=-1) + eps
    den = (prob * prob).sum(dim=-1) + (target * target).sum(dim=-1) + eps
    return (1.0 - num / den).mean()


def focal_bce(logit: torch.Tensor, target: torch.Tensor,
              alpha: float = 0.75, gamma: float = 2.0) -> torch.Tensor:
    """Focal binary cross-entropy against soft targets."""
    p = torch.sigmoid(logit)
    ce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    a_t = alpha * target + (1 - alpha) * (1 - target)
    return (a_t * (1 - p_t).pow(gamma) * ce).mean()


def rhythm_prior_loss(
    prob: torch.Tensor,
    fs: float,
    hr_range: tuple[float, float] = (30.0, 220.0),
    temperature: float = 20.0,
) -> torch.Tensor:
    """
    Penalise loss of pseudo-periodicity in the beat probability map.

    The autocorrelation is evaluated over lags corresponding to the given
    heart-rate range, and a soft maximum (log-sum-exp) is used instead of a
    hard max so that the gradient reaches every plausible lag early in
    training rather than only the current argmax.
    """
    B, L = prob.shape
    x = prob - prob.mean(dim=-1, keepdim=True)
    denom = (x * x).sum(dim=-1, keepdim=True) + 1e-8

    lag_min = max(int(fs * 60.0 / hr_range[1]), 1)
    lag_max = min(int(fs * 60.0 / hr_range[0]), L - 1)
    if lag_max <= lag_min:
        return prob.new_zeros(())

    # FFT-based autocorrelation: O(L log L) instead of O(L * n_lags)
    n = 1
    while n < 2 * L:
        n *= 2
    X = torch.fft.rfft(x, n=n)
    ac = torch.fft.irfft(X * torch.conj(X), n=n)[:, :L]
    rho = ac / denom
    band = rho[:, lag_min:lag_max + 1]
    soft_max = torch.logsumexp(band * temperature, dim=-1) / temperature
    return (1.0 - soft_max).clamp(min=0.0).mean()


def pinball_loss(pred: torch.Tensor, target: torch.Tensor,
                 quantiles: tuple[float, ...]) -> torch.Tensor:
    """Quantile (pinball) regression loss. pred: (B, n_q), target: (B,)."""
    t = target.unsqueeze(-1)
    q = torch.tensor(quantiles, device=pred.device, dtype=pred.dtype).view(1, -1)
    e = t - pred
    return torch.maximum(q * e, (q - 1) * e).mean()


class HRLoss(nn.Module):
    """
    Combined objective.

    Parameters
    ----------
    fs : sampling rate, needed by the rhythm prior to convert lags to bpm.
    """

    def __init__(self, fs: float, weights: LossWeights | None = None,
                 quantiles: tuple[float, ...] = (0.05, 0.50, 0.95)):
        super().__init__()
        self.fs = fs
        self.w = weights or LossWeights()
        self.quantiles = quantiles

    def forward(self, out, batch: dict) -> tuple[torch.Tensor, dict]:
        prob = torch.sigmoid(out.beat_logit)
        y_beat = batch["beat"]
        y_off = batch["offset"]
        y_q = batch["usqi"]
        mask = batch.get("beat_mask", (y_beat > 0.5).float())

        l_beat = focal_bce(out.beat_logit, y_beat) + soft_dice_loss(prob, y_beat)

        # offsets are only defined in the immediate neighbourhood of a beat
        denom = mask.sum().clamp(min=1.0)
        l_off = (F.smooth_l1_loss(out.offset, y_off, reduction="none", beta=0.2)
                 * mask).sum() / denom

        l_qual = F.mse_loss(out.usqi, y_q)
        l_rhythm = rhythm_prior_loss(prob, self.fs)

        if "hr" in batch and batch["hr"] is not None:
            valid = torch.isfinite(batch["hr"])
            l_quant = (pinball_loss(out.hr_quantiles[valid], batch["hr"][valid],
                                    self.quantiles)
                       if valid.any() else prob.new_zeros(()))
        else:
            l_quant = prob.new_zeros(())

        total = (self.w.beat * l_beat + self.w.offset * l_off
                 + self.w.quality * l_qual + self.w.rhythm * l_rhythm
                 + self.w.quantile * l_quant)

        return total, dict(
            total=float(total), beat=float(l_beat), offset=float(l_off),
            quality=float(l_qual), rhythm=float(l_rhythm), quantile=float(l_quant),
        )
