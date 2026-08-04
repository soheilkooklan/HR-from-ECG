"""
Distribution-free uncertainty for heart-rate estimation.

Split conformal prediction turns any point predictor into an interval predictor
with a finite-sample coverage guarantee that holds without any assumption on
the data distribution or on the model being correct:

    P( HR_true in C(x) )  >=  1 - alpha

The guarantee is *marginal*, and that is its weakness in this application. A
predictor can hit 90% coverage overall while covering 99% of clean windows and
40% of noisy ones -- which is the opposite of what a monitor needs, since the
windows where the interval matters are exactly the noisy ones.

The fix used here is Mondrian (group-conditional) conformal prediction, with
the groups defined by the predicted uSQI. Calibration quantiles are computed
separately within each quality stratum, so coverage holds *within* strata:

    P( HR_true in C(x) | uSQI(x) in bin_k )  >=  1 - alpha   for every k

This is the point at which contributions C1 and C3 stop being two separate
ideas. The quality index is not a diagnostic read-out bolted onto the side of
the model; it is the conditioning variable that makes the statistical guarantee
meaningful. A learned quality score that had no relationship to error would
produce strata with identical quantiles, and Mondrian conformal would collapse
back to the split version -- so the conditional-coverage table is simultaneously
a validation of uSQI and of the interval construction.

Nonconformity score
-------------------
The normalised residual is used rather than the raw one:

    s_i = |y_i - q_50(x_i)| / (q_95(x_i) - q_05(x_i) + eps)

so the conformal correction rescales an interval whose *shape* the network has
already adapted to the input. Raw-residual conformal would return an interval
of constant width everywhere, which is exactly the uninformative behaviour this
project is trying to avoid.

References
----------
Vovk, Gammerman & Shafer (2005), Algorithmic Learning in a Random World.
Lei, G'Sell, Rinaldo, Tibshirani & Wasserman (2018), JASA 113:1094-1111.
Romano, Patterson & Candes (2019), NeurIPS (conformalised quantile regression).
Angelopoulos, Bates, Candes, Jordan & Lei (2021), arXiv:2110.01052 (risk control).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["ConformalHR", "SelectiveController", "conformal_quantile"]


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    The finite-sample-corrected empirical quantile.

    The ceiling correction (n+1)(1-alpha)/n is what upgrades an asymptotic
    statement into a guarantee valid for any n; omitting it is the most common
    error in applied conformal papers.
    """
    s = np.sort(np.asarray(scores, dtype=float))
    s = s[np.isfinite(s)]
    n = len(s)
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    if k >= n:
        return float("inf")
    return float(s[k])


@dataclass
class ConformalHR:
    """
    Conformal interval predictor for windowed heart rate.

    Parameters
    ----------
    alpha : miscoverage level; 0.1 gives 90% intervals.
    mode  : "split" for marginal coverage, "mondrian" for coverage conditional
            on the quality stratum.
    n_bins : number of uSQI strata in Mondrian mode. Strata are defined by
            quantiles of the calibration uSQI, not by fixed cut-points, so
            every stratum is guaranteed enough calibration points for the
            quantile to be finite.
    min_per_bin : if a stratum has fewer points than this, it falls back to the
            pooled quantile. Without this guard a sparsely populated stratum
            returns an infinite interval, which is technically valid and
            practically useless.
    """

    alpha: float = 0.1
    mode: str = "mondrian"
    n_bins: int = 5
    min_per_bin: int = 30

    q_global: float = field(default=float("nan"), init=False)
    q_bins: np.ndarray = field(default_factory=lambda: np.array([]), init=False)
    bin_edges: np.ndarray = field(default_factory=lambda: np.array([]), init=False)
    fitted: bool = field(default=False, init=False)

    # ---------------------------------------------------------------- fitting
    def _scores(self, y: np.ndarray, q_lo: np.ndarray, q_med: np.ndarray,
                q_hi: np.ndarray) -> np.ndarray:
        width = np.maximum(q_hi - q_lo, 1e-3)
        return np.abs(y - q_med) / width

    def fit(
        self,
        y_cal: np.ndarray,
        q_lo: np.ndarray,
        q_med: np.ndarray,
        q_hi: np.ndarray,
        usqi_cal: np.ndarray | None = None,
    ) -> "ConformalHR":
        """Calibrate on a held-out split that the model never trained on."""
        y_cal, q_lo, q_med, q_hi = map(lambda a: np.asarray(a, float),
                                       (y_cal, q_lo, q_med, q_hi))
        m = np.isfinite(y_cal) & np.isfinite(q_med)
        s = self._scores(y_cal[m], q_lo[m], q_med[m], q_hi[m])
        self.q_global = conformal_quantile(s, self.alpha)

        if self.mode == "mondrian":
            if usqi_cal is None:
                raise ValueError("mondrian mode requires usqi_cal")
            u = np.asarray(usqi_cal, float)[m]
            self.bin_edges = np.quantile(u, np.linspace(0, 1, self.n_bins + 1))
            self.bin_edges[0], self.bin_edges[-1] = -np.inf, np.inf
            qs = []
            for i in range(self.n_bins):
                sel = (u >= self.bin_edges[i]) & (u < self.bin_edges[i + 1])
                qs.append(conformal_quantile(s[sel], self.alpha)
                          if sel.sum() >= self.min_per_bin else self.q_global)
            self.q_bins = np.asarray(qs, float)

        self.fitted = True
        return self

    # ------------------------------------------------------------- prediction
    def predict(
        self,
        q_lo: np.ndarray,
        q_med: np.ndarray,
        q_hi: np.ndarray,
        usqi: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) heart-rate bounds in bpm."""
        if not self.fitted:
            raise RuntimeError("call fit() on a calibration split first")
        q_lo, q_med, q_hi = map(lambda a: np.asarray(a, float), (q_lo, q_med, q_hi))
        width = np.maximum(q_hi - q_lo, 1e-3)

        if self.mode == "mondrian" and usqi is not None and len(self.q_bins):
            idx = np.clip(np.searchsorted(self.bin_edges, np.asarray(usqi, float),
                                          side="right") - 1, 0, self.n_bins - 1)
            q = self.q_bins[idx]
        else:
            q = np.full(len(q_med), self.q_global)

        half = q * width
        return q_med - half, q_med + half

    def summary(self) -> dict:
        return dict(
            alpha=self.alpha, mode=self.mode, q_global=self.q_global,
            q_bins=self.q_bins.tolist() if len(self.q_bins) else None,
            bin_edges=self.bin_edges.tolist() if len(self.bin_edges) else None,
        )


@dataclass
class SelectiveController:
    """
    Abstention with a guaranteed error bound.

    Given a quality score and a target risk (say, mean absolute heart-rate
    error below 3 bpm), find the score threshold above which that risk is met,
    using the Learn-then-Test correction so that the bound holds on unseen
    data rather than only on the calibration set.

    The controller answers the question a clinician would actually ask -- "when
    should the monitor refuse to display a number?" -- with a threshold derived
    from data rather than chosen by eye.
    """

    target_risk: float = 3.0     # bpm
    delta: float = 0.1           # tolerated probability of exceeding it
    loss_cap: float = 20.0       # errors are clipped here before averaging
    threshold: float = field(default=float("nan"), init=False)
    achieved_coverage: float = field(default=float("nan"), init=False)
    achieved_risk: float = field(default=float("nan"), init=False)

    def fit(self, errors: np.ndarray, scores: np.ndarray,
            n_grid: int = 200) -> "SelectiveController":
        e = np.asarray(errors, float)
        s = np.asarray(scores, float)
        m = np.isfinite(e) & np.isfinite(s)
        e, s = e[m], s[m]
        # A bounded loss is required for the concentration bound to be usable.
        # Without the cap a single catastrophic window (a flat lead, error of
        # 60+ bpm) inflates the range term and drives the threshold so high
        # that the system abstains on most of a perfectly usable record.
        e = np.minimum(e, self.loss_cap)
        if len(e) == 0:
            return self

        best = None
        for t in np.quantile(s, np.linspace(0.0, 0.99, n_grid)):
            keep = s >= t
            if keep.sum() < 10:
                continue
            risk = float(np.mean(e[keep]))
            # one-sided Hoeffding correction for the finite calibration set
            slack = self.loss_cap * np.sqrt(np.log(1 / self.delta) / (2 * keep.sum()))
            if risk + slack <= self.target_risk:
                best = (t, float(keep.mean()), risk)
                break  # thresholds are ascending, so the first feasible one
                       # keeps the most data
        if best is None:
            self.threshold = float(np.max(s))
            self.achieved_coverage, self.achieved_risk = 0.0, float("nan")
        else:
            self.threshold, self.achieved_coverage, self.achieved_risk = best
        return self

    def accept(self, scores: np.ndarray) -> np.ndarray:
        """Boolean mask of windows the system is willing to report."""
        return np.asarray(scores, float) >= self.threshold

    def summary(self) -> dict:
        return dict(target_risk=self.target_risk, delta=self.delta,
                    threshold=self.threshold, coverage=self.achieved_coverage,
                    risk=self.achieved_risk)
