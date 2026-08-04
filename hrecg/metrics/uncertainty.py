"""
Uncertainty-quantification and selective-prediction metrics.

These are the metrics against which the central claim of HR from ECG is judged:
that the reported heart-rate interval covers the true heart rate at the
nominal rate, and that abstaining on a small fraction of the data buys a large
reduction in error on the remainder.

Marginal coverage alone is a weak guarantee -- a predictor can achieve 90%
marginal coverage while systematically failing on every noisy segment. Hence
`conditional_coverage` and `coverage_by_group`, which are what a critical
reviewer will ask for.

References
----------
Vovk, Gammerman & Shafer (2005), Algorithmic Learning in a Random World.
Angelopoulos & Bates (2021), arXiv:2107.07511.
Winkler (1972), JASA 67:187-191.
El-Yaniv & Wiener (2010), JMLR 11:1605-1641 (selective prediction).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "IntervalMetrics", "interval_metrics", "coverage_by_group",
    "conditional_coverage", "risk_coverage_curve", "aurc",
    "expected_calibration_error", "winkler_score",
]


@dataclass
class IntervalMetrics:
    picp: float          # prediction interval coverage probability
    mpiw: float          # mean prediction interval width
    nmpiw: float         # normalised by the range of the target
    winkler: float       # proper scoring rule for intervals
    target_coverage: float
    coverage_gap: float  # picp - target (negative = under-coverage, the bad case)
    n: int

    def as_dict(self) -> dict:
        return dict(
            PICP=self.picp, MPIW=self.mpiw, NMPIW=self.nmpiw,
            Winkler=self.winkler, target=self.target_coverage,
            coverage_gap=self.coverage_gap, n=self.n,
        )


def winkler_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> float:
    """
    Winkler interval score: width plus a penalty proportional to the miss
    distance. Lower is better. Unlike coverage it cannot be gamed by widening
    the interval, so it is the primary scalar for comparing UQ methods.
    """
    y, lo, hi = map(lambda a: np.asarray(a, float), (y, lo, hi))
    width = hi - lo
    below = 2.0 / alpha * (lo - y) * (y < lo)
    above = 2.0 / alpha * (y - hi) * (y > hi)
    return float(np.mean(width + below + above))


def interval_metrics(
    y: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    alpha: float = 0.1,
) -> IntervalMetrics:
    """Core interval diagnostics at nominal coverage 1 - alpha."""
    y, lo, hi = map(lambda a: np.asarray(a, float), (y, lo, hi))
    m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    y, lo, hi = y[m], lo[m], hi[m]
    if len(y) == 0:
        nan = float("nan")
        return IntervalMetrics(nan, nan, nan, nan, 1 - alpha, nan, 0)

    covered = (y >= lo) & (y <= hi)
    picp = float(np.mean(covered))
    mpiw = float(np.mean(hi - lo))
    rng = float(np.ptp(y))
    return IntervalMetrics(
        picp=picp, mpiw=mpiw,
        nmpiw=mpiw / rng if rng > 0 else float("nan"),
        winkler=winkler_score(y, lo, hi, alpha),
        target_coverage=1 - alpha,
        coverage_gap=picp - (1 - alpha),
        n=int(len(y)),
    )


def coverage_by_group(
    y: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    group: np.ndarray,
) -> dict:
    """Empirical coverage within each level of a categorical variable."""
    y, lo, hi = map(lambda a: np.asarray(a, float), (y, lo, hi))
    group = np.asarray(group)
    out = {}
    for g in np.unique(group):
        m = (group == g) & np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
        out[str(g)] = dict(
            coverage=float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m]))) if m.sum() else float("nan"),
            width=float(np.mean(hi[m] - lo[m])) if m.sum() else float("nan"),
            n=int(m.sum()),
        )
    return out


def conditional_coverage(
    y: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    covariate: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Coverage as a function of a continuous covariate (typically SNR or uSQI).

    Also returns worst-slab coverage: the minimum bin coverage, which is the
    quantity that exposes the failure mode split conformal is blind to.
    """
    y, lo, hi, c = map(lambda a: np.asarray(a, float), (y, lo, hi, covariate))
    m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi) & np.isfinite(c)
    y, lo, hi, c = y[m], lo[m], hi[m], c[m]
    if len(y) < n_bins:
        return dict(bins=[], worst_slab_coverage=float("nan"))

    edges = np.quantile(c, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    bins = []
    for i in range(n_bins):
        b = (c >= edges[i]) & (c < edges[i + 1])
        if b.sum() == 0:
            continue
        bins.append(dict(
            lo_edge=float(edges[i]), hi_edge=float(edges[i + 1]),
            coverage=float(np.mean((y[b] >= lo[b]) & (y[b] <= hi[b]))),
            width=float(np.mean(hi[b] - lo[b])), n=int(b.sum()),
        ))
    return dict(
        bins=bins,
        worst_slab_coverage=float(min(b["coverage"] for b in bins)) if bins else float("nan"),
    )


def risk_coverage_curve(
    errors: np.ndarray,
    confidence: np.ndarray,
    n_points: int = 101,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Selective-prediction curve.

    Samples are ranked by `confidence` (higher = more trusted); at each
    coverage level the mean error over the retained fraction is reported.
    A useful quality score must make this curve fall steeply.

    Returns (coverage_grid, selective_risk).
    """
    e = np.asarray(errors, float)
    c = np.asarray(confidence, float)
    m = np.isfinite(e) & np.isfinite(c)
    e, c = e[m], c[m]
    if len(e) == 0:
        return np.array([]), np.array([])

    order = np.argsort(-c)
    e_sorted = e[order]
    cum = np.cumsum(e_sorted) / np.arange(1, len(e_sorted) + 1)
    cov = np.arange(1, len(e_sorted) + 1) / len(e_sorted)

    grid = np.linspace(1.0 / len(e_sorted), 1.0, n_points)
    return grid, np.interp(grid, cov, cum)


def aurc(errors: np.ndarray, confidence: np.ndarray) -> float:
    """Area under the risk-coverage curve. Lower is better."""
    cov, risk = risk_coverage_curve(errors, confidence)
    if len(cov) < 2:
        return float("nan")
    return float(np.trapezoid(risk, cov) / (cov[-1] - cov[0]))


def expected_calibration_error(
    errors: np.ndarray,
    predicted_sd: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Regression ECE: bins by predicted uncertainty and compares the predicted
    standard deviation with the realised RMS error in each bin.
    """
    e = np.abs(np.asarray(errors, float))
    s = np.asarray(predicted_sd, float)
    m = np.isfinite(e) & np.isfinite(s)
    e, s = e[m], s[m]
    if len(e) < n_bins:
        return float("nan")

    edges = np.quantile(s, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    total, ece = 0, 0.0
    for i in range(n_bins):
        b = (s >= edges[i]) & (s < edges[i + 1])
        if b.sum() == 0:
            continue
        realised = np.sqrt(np.mean(e[b] ** 2))
        ece += b.sum() * abs(realised - np.mean(s[b]))
        total += b.sum()
    return float(ece / total) if total else float("nan")
