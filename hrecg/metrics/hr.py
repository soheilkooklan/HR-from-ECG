"""
Heart-rate estimation and agreement metrics.

Two conventions for "heart rate" coexist in the literature and are routinely
conflated, which is one reason published HR errors are hard to compare:

    instantaneous HR  = 60 / RR_k                 (beat-indexed)
    windowed HR       = 60 * n_beats / window     (time-indexed)

Both are provided. All benchmarking in this project uses windowed HR on a fixed
grid, because that is what a monitor displays, what the AAMI EC13 standard
regulates, and the only definition for which a conformal interval has an
unambiguous target.

References
----------
ANSI/AAMI EC13:2002 -- Cardiac monitors, heart rate meters, and alarms.
Bland & Altman (1986), Lancet 327:307-310.
Shrout & Fleiss (1979), Psychological Bulletin 86:420-428.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "instantaneous_hr", "windowed_hr", "bland_altman", "icc21",
    "hr_agreement", "AgreementResult",
]


def instantaneous_hr(peaks: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Beat-indexed HR. Returns (hr_bpm, time_s) with time at the second beat."""
    p = np.sort(np.asarray(peaks, dtype=float)) / fs
    if len(p) < 2:
        return np.array([]), np.array([])
    rr = np.diff(p)
    return 60.0 / rr, p[1:]


def windowed_hr(
    peaks: np.ndarray,
    fs: float,
    n_samples: int,
    window_s: float = 10.0,
    step_s: float = 1.0,
    min_beats: int = 3,
    method: str = "mean_rr",
) -> tuple[np.ndarray, np.ndarray]:
    """
    HR on a sliding time grid.

    method="mean_rr" averages the RR intervals fully contained in the window
    (robust, standard). method="count" uses beat count / duration, which is
    what simple monitors do and is biased at window edges -- provided for
    comparison against legacy pipelines.

    Windows with fewer than `min_beats` beats yield NaN rather than a
    fabricated value; propagating NaN instead of guessing is deliberate, since
    the abstention mechanism downstream must be able to see genuine gaps.
    """
    p = np.sort(np.asarray(peaks, dtype=float)) / fs
    dur = n_samples / fs
    starts = np.arange(0.0, max(dur - window_s, 0.0) + 1e-9, step_s)
    centres = starts + window_s / 2.0
    hr = np.full(len(starts), np.nan)

    for i, s in enumerate(starts):
        sel = p[(p >= s) & (p < s + window_s)]
        if len(sel) < min_beats:
            continue
        if method == "count":
            hr[i] = 60.0 * (len(sel) - 1) / (sel[-1] - sel[0]) if sel[-1] > sel[0] else np.nan
        else:
            rr = np.diff(sel)
            hr[i] = 60.0 / np.mean(rr) if len(rr) else np.nan
    return hr, centres


@dataclass
class AgreementResult:
    mae: float
    rmse: float
    bias: float
    loa_lower: float
    loa_upper: float
    pearson_r: float
    icc: float
    pct_within_5bpm: float
    pct_within_10pct: float
    n: int

    def as_dict(self) -> dict:
        return dict(
            MAE_bpm=self.mae, RMSE_bpm=self.rmse, bias_bpm=self.bias,
            LoA_lower=self.loa_lower, LoA_upper=self.loa_upper,
            pearson_r=self.pearson_r, ICC21=self.icc,
            pct_within_5bpm=self.pct_within_5bpm,
            pct_within_10pct=self.pct_within_10pct, n=self.n,
        )


def bland_altman(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Return (bias, lower limit of agreement, upper limit of agreement)."""
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    bias, sd = float(np.mean(d)), float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
    return bias, bias - 1.96 * sd, bias + 1.96 * sd


def icc21(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    ICC(2,1): two-way random effects, absolute agreement, single measurement.

    Preferred over Pearson r for method comparison because it penalises
    systematic bias, whereas r is invariant to it.
    """
    x = np.column_stack([np.asarray(y_true, float), np.asarray(y_pred, float)])
    n, k = x.shape
    if n < 2:
        return float("nan")
    grand = x.mean()
    ms_r = k * np.sum((x.mean(axis=1) - grand) ** 2) / (n - 1)
    ms_c = n * np.sum((x.mean(axis=0) - grand) ** 2) / (k - 1)
    ss_e = np.sum((x - x.mean(axis=1, keepdims=True)
                   - x.mean(axis=0, keepdims=True) + grand) ** 2)
    ms_e = ss_e / ((n - 1) * (k - 1))
    denom = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    return float((ms_r - ms_e) / denom) if denom != 0 else float("nan")


def hr_agreement(y_true: np.ndarray, y_pred: np.ndarray) -> AgreementResult:
    """
    Full agreement analysis on paired HR estimates, ignoring NaN pairs.

    `pct_within_10pct` implements the ANSI/AAMI EC13 accuracy requirement
    (within 10% of the reference, or 5 bpm, whichever is greater).
    """
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    m = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[m], yp[m]
    if len(yt) == 0:
        nan = float("nan")
        return AgreementResult(nan, nan, nan, nan, nan, nan, nan, nan, nan, 0)

    err = yp - yt
    bias, lo, hi = bland_altman(yt, yp)
    r = float(np.corrcoef(yt, yp)[0, 1]) if len(yt) > 1 and np.std(yt) > 0 else float("nan")
    tol = np.maximum(0.10 * yt, 5.0)
    return AgreementResult(
        mae=float(np.mean(np.abs(err))),
        rmse=float(np.sqrt(np.mean(err**2))),
        bias=bias, loa_lower=lo, loa_upper=hi,
        pearson_r=r, icc=icc21(yt, yp),
        pct_within_5bpm=float(np.mean(np.abs(err) <= 5.0) * 100),
        pct_within_10pct=float(np.mean(np.abs(err) <= tol) * 100),
        n=int(len(yt)),
    )
