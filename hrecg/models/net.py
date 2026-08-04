"""
HR-from-ECG v2: the estimator.

A U-Net over the raw single-lead ECG whose bottleneck is bidirectional
selective state space rather than attention, with four heads:

    beat      per-sample beat probability
    offset    per-sample sub-sample correction to the beat position
    usqi      per-sample utility-based signal quality
    hr        window-level heart-rate quantiles (0.05 / 0.50 / 0.95)

Design decisions worth stating explicitly, because each is a response to a
measured finding from the infrastructure stage:

*   **An offset head, not just a probability map.** The baseline study showed
    localisation jitter of 3.1 ms at 6 dB SNR, which is a non-trivial share of
    a 25 ms RMSSD. A segmentation map quantised to the sample grid cannot do
    better than +-1 sample; regressing a continuous offset can, and it is the
    single cheapest way to make the HRV numbers defensible.

*   **The quality head shares the encoder with the beat head.** Quality is
    defined by how much a segment degrades the heart-rate estimate, so it is
    not a separate problem -- the features that reveal a beat are the features
    that reveal whether the beat can be trusted. A detached branch would have
    to relearn them.

*   **The HR head predicts quantiles, not a mean.** The conformal layer needs a
    heuristic uncertainty notion to sharpen; a quantile head supplies one that
    already adapts to the input, so the conformal correction stays small and
    the intervals stay narrow.

*   **The quality head conditions the beat head** through a learned gate. This
    is what links contributions C1 and C3: the network can suppress its own
    detections where it judges the signal unusable, instead of emitting
    confident detections that a downstream filter must clean up.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssm import BiSSMBlock

__all__ = ["HRModel", "ModelConfig", "ModelOutput"]


@dataclass
class ModelConfig:
    in_channels: int = 1
    base_width: int = 24
    depth: int = 4                 # number of down/up stages
    d_state: int = 8
    expand: int = 1
    kernel_size: int = 5
    n_ssm_blocks: int = 2
    dropout: float = 0.05
    quantiles: tuple[float, ...] = (0.05, 0.50, 0.95)

    @property
    def widths(self) -> list[int]:
        return [self.base_width * (2**i) for i in range(self.depth + 1)]


@dataclass
class ModelOutput:
    beat_logit: torch.Tensor    # (B, L)
    offset: torch.Tensor        # (B, L)   in samples, range ~(-1, 1)
    usqi: torch.Tensor          # (B, L)   in (0, 1)
    hr_quantiles: torch.Tensor  # (B, n_q) in bpm

    def probs(self) -> torch.Tensor:
        return torch.sigmoid(self.beat_logit)


class ConvBlock(nn.Module):
    """Two dilated convolutions with GroupNorm; the residual keeps depth trainable."""

    def __init__(self, c_in: int, c_out: int, dilation: int = 1, dropout: float = 0.0,
                 k: int = 5):
        super().__init__()
        g = max(1, min(8, c_out // 4))
        pad = (k // 2) * dilation
        self.body = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, padding=pad, dilation=dilation),
            nn.GroupNorm(g, c_out), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(c_out, c_out, k, padding=pad, dilation=dilation),
            nn.GroupNorm(g, c_out), nn.GELU(),
        )
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        return F.gelu(self.body(x) + self.skip(x))


class HRModel(nn.Module):
    """
    Parameters
    ----------
    cfg : ModelConfig

    Input is (batch, in_channels, length) with length divisible by 2**depth.
    """

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        w = self.cfg.widths
        d = self.cfg.depth

        self.stem = ConvBlock(self.cfg.in_channels, w[0], dilation=1,
                              dropout=self.cfg.dropout, k=self.cfg.kernel_size)
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(d):
            # dilation grows with depth: cheap multi-scale receptive field
            self.downs.append(ConvBlock(w[i], w[i + 1], dilation=2**i,
                                        dropout=self.cfg.dropout,
                                        k=self.cfg.kernel_size))
            self.pools.append(nn.MaxPool1d(2))

        self.bottleneck = nn.ModuleList([
            BiSSMBlock(w[d], d_state=self.cfg.d_state, expand=self.cfg.expand,
                       dropout=self.cfg.dropout)
            for _ in range(self.cfg.n_ssm_blocks)
        ])

        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(d, 0, -1):
            self.ups.append(nn.ConvTranspose1d(w[i], w[i - 1], 2, stride=2))
            self.up_convs.append(ConvBlock(2 * w[i - 1], w[i - 1],
                                           dropout=self.cfg.dropout,
                                           k=self.cfg.kernel_size))

        f = w[0]
        self.head_quality = nn.Sequential(
            nn.Conv1d(f, f, 5, padding=2), nn.GELU(), nn.Conv1d(f, 1, 1))
        # gate: quality modulates the features the beat head sees
        self.gate = nn.Conv1d(1, f, 1)
        self.head_beat = nn.Sequential(
            nn.Conv1d(f, f, 5, padding=2), nn.GELU(), nn.Conv1d(f, 1, 1))
        self.head_offset = nn.Sequential(
            nn.Conv1d(f, f, 5, padding=2), nn.GELU(), nn.Conv1d(f, 1, 1), nn.Tanh())
        self.head_hr = nn.Sequential(
            nn.Linear(2 * w[d], 64), nn.GELU(),
            nn.Linear(64, len(self.cfg.quantiles)))

    def forward(self, x: torch.Tensor) -> ModelOutput:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.stem(x)
        skips = [h]
        for conv, pool in zip(self.downs, self.pools):
            h = conv(pool(h))
            skips.append(h)

        z = h.transpose(1, 2)
        for blk in self.bottleneck:
            z = blk(z)
        h = z.transpose(1, 2)

        # window-level heart-rate quantiles from pooled bottleneck statistics
        pooled = torch.cat([h.mean(-1), h.amax(-1)], dim=-1)
        hr_q = self.head_hr(pooled)
        # enforce monotone quantiles by construction: q_k = q_0 + cumsum(softplus)
        hr_q = hr_q[:, :1] + torch.cat(
            [torch.zeros_like(hr_q[:, :1]), F.softplus(hr_q[:, 1:])], dim=1).cumsum(1)

        for i, (up, conv) in enumerate(zip(self.ups, self.up_convs)):
            h = up(h)
            s = skips[-(i + 2)]
            if h.shape[-1] != s.shape[-1]:
                h = F.pad(h, (0, s.shape[-1] - h.shape[-1]))
            h = conv(torch.cat([h, s], dim=1))

        q_logit = self.head_quality(h)
        usqi = torch.sigmoid(q_logit)
        h_gated = h * torch.sigmoid(self.gate(q_logit))

        return ModelOutput(
            beat_logit=self.head_beat(h_gated).squeeze(1),
            offset=self.head_offset(h_gated).squeeze(1),
            usqi=usqi.squeeze(1),
            hr_quantiles=hr_q,
        )

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
