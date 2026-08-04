"""
Selective state-space blocks, implemented in pure PyTorch.

Rationale for not depending on `mamba-ssm`
------------------------------------------
The reference Mamba package requires a CUDA-compiled kernel, pins narrow
PyTorch versions, and will not run on CPU at all. Anyone trying to reproduce
this work on a laptop, or to audit it, would be blocked. The recurrence is
therefore re-implemented here with a vectorised associative scan, which runs on
any backend at roughly half the throughput of the fused kernel -- a price worth
paying for a project whose whole argument is about trustworthiness.

The recurrence being solved is the discretised, input-dependent linear system

    h_t = A_t * h_{t-1} + B_t * x_t ,    y_t = C_t . h_t + D * x_t

where A_t, B_t, C_t depend on the input (this input dependence is the
"selective" part, and is what lets the model ignore an artefact burst instead
of integrating it into its state).

A naive Python loop over t is O(L) sequential steps and unusably slow. Because
the update is a first-order affine recurrence, it is an *associative* operation

    (a1, b1) . (a2, b2) = (a1 a2 ,  a2 b1 + b2)

so it can be evaluated with a Hillis-Steele inclusive scan in log2(L)
fully-parallel steps. That is the implementation below.

References
----------
Gu & Dao (2024), "Mamba: Linear-time sequence modeling with selective state
spaces", COLM.
Blelloch (1990), "Prefix sums and their applications", CMU-CS-90-190.
Smith, Warrington & Linderman (2023), "Simplified state space layers for
sequence modeling", ICLR.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["associative_scan", "SelectiveSSM", "BiSSMBlock"]


def associative_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Inclusive scan of h_t = a_t * h_{t-1} + b_t along dim=1.

    Parameters
    ----------
    a, b : (batch, length, ...) tensors with identical shape.

    Returns
    -------
    h : same shape, with h_0 = b_0.

    Uses log2(L) doubling steps. Each step composes the affine maps of
    neighbours at distance `stride`, so after ceil(log2(L)) steps every
    position has absorbed the whole prefix.
    """
    L = a.shape[1]
    a, b = a.clone(), b.clone()
    stride = 1
    while stride < L:
        a_prev = F.pad(a[:, : L - stride], (0,) * 2 * (a.dim() - 2) + (stride, 0))
        b_prev = F.pad(b[:, : L - stride], (0,) * 2 * (b.dim() - 2) + (stride, 0))
        # positions < stride have no predecessor: leave them untouched
        mask_shape = [1, L] + [1] * (a.dim() - 2)
        keep = (torch.arange(L, device=a.device) >= stride).view(mask_shape)
        b = torch.where(keep, a * b_prev + b, b)
        a = torch.where(keep, a * a_prev, a)
        stride *= 2
    return b


class SelectiveSSM(nn.Module):
    """
    One selective state-space layer (the S6 core of Mamba).

    Parameters
    ----------
    d_model : channel width of the input/output.
    d_state : size of the latent state per channel (N).
    d_conv  : width of the causal depthwise convolution applied before the scan;
              it supplies the short-range inductive bias that a pure SSM lacks
              and that QRS morphology needs.
    expand  : inner width multiplier.
    """

    def __init__(self, d_model: int, d_state: int = 8, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1,
        )
        # projections producing the input-dependent dt, B, C
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # A is parameterised as -exp(A_log) so that it is always stable (|a| < 1)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # initialise dt bias so that the effective timescale starts in a
        # useful range rather than saturating softplus at either end
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001))
                       + math.log(0.001))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, length, d_model) -> same shape."""
        B_, L, _ = x.shape
        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1)

        u = self.conv1d(u.transpose(1, 2))[..., :L].transpose(1, 2)
        u = F.silu(u)

        proj = self.x_proj(u)
        dt_raw, Bm, Cm = torch.split(proj, [1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw))                    # (B, L, d_inner)

        A = -torch.exp(self.A_log)                               # (d_inner, N)
        dA = torch.exp(dt.unsqueeze(-1) * A)                     # (B, L, d_inner, N)
        dB = dt.unsqueeze(-1) * Bm.unsqueeze(2) * u.unsqueeze(-1)

        h = associative_scan(dA, dB)                             # (B, L, d_inner, N)
        y = (h * Cm.unsqueeze(2)).sum(-1) + self.D * u
        return self.out_proj(y * F.silu(z))


class BiSSMBlock(nn.Module):
    """
    Bidirectional selective SSM with a pre-norm residual connection.

    Bidirectionality matters here for a concrete physiological reason: deciding
    whether a candidate deflection is a QRS or an artefact depends on the beats
    that come *after* it as much as those before, and an offline heart-rate
    report has no causality constraint to respect.
    """

    def __init__(self, d_model: int, d_state: int = 8, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fwd = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.bwd = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.merge = nn.Linear(2 * d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        f = self.fwd(h)
        b = self.bwd(torch.flip(h, dims=[1])).flip(dims=[1])
        return x + self.drop(self.merge(torch.cat([f, b], dim=-1)))
