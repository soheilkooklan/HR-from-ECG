"""
Heart-rate variability features.

Two implementation choices differ from most open-source HRV toolboxes and are
deliberate:

1.  Spectral analysis uses the **Lomb-Scargle periodogram** on the raw,
    unevenly sampled RR series rather than cubic-spline interpolation followed
    by Welch. Interpolation injects power into the HF band and biases LF/HF,
    and the bias grows exactly where this project operates -- when beats are
    missing. Lomb-Scargle handles gaps natively.

2.  Ectopic handling is explicit and pluggable rather than hidden, because the
    choice of correction changes RMSSD by tens of percent and is a documented
    source of irreproducibility in the HRV literature.

These features are the target of the conformal intervals: the framework
propagates beat-level detection uncertainty through this module by Monte-Carlo
resampling, producing an interval on each feature instead of a point value.

References
----------
Task Force of ESC/NASPE (1996), Circulation 93:1043-1065.
Lomb (1976), Astrophys. Space Sci. 39:447-462; Scargle (1982), ApJ 263:835-853.
Brennan, Palaniswami & Kamen (2001), IEEE TBME 48(11):1342-1347 (Poincare).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.signal import lombscargle

__all__ = [
    "rr_from_peaks", "clean_rr", "time_domain", "poincare",
    "frequency_domain", "hrv_features",
]

LF_BAND = (0.04, 0.15)
HF_BAND = (0.15, 0.40)
VLF_BAND = (0.0033, 0.04)


def rr_from_peaks(peaks: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (rr_seconds, t_seconds) where t marks the end of each interval."""
    p = np.sort(np.asarray(peaks, dtype=float)) / fs
    if len(p) < 2:
        return np.array([]), np.array([])
    return np.diff(p), p[1:]


def clean_rr(
    rr: np.ndarray,
    t: np.ndarray | None = None,
    method: Literal["none", "remove", "interpolate", "malik"] = "malik",
    threshold: float = 0.20,
    rr_range: tuple[float, float] = (0.30, 2.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect and handle ectopic / artefactual intervals.

    "malik"       -- flag RR_k if it deviates from RR_{k-1} by more than
                     `threshold` (the standard 20% rule), then interpolate.
    "remove"      -- drop flagged intervals. Correct for time-domain indices,
                     but it silently destroys the time base, so it must not be
                     combined with spectral analysis.
    "interpolate" -- replace flagged values by linear interpolation of
                     neighbouring accepted intervals.

    Returns (rr_clean, t_clean, flagged_mask_on_original).
    """
    rr = np.asarray(rr, dtype=float)
    t = np.arange(len(rr), dtype=float) if t is None else np.asarray(t, dtype=float)
    if len(rr) == 0:
        return rr, t, np.zeros(0, dtype=bool)

    bad = (rr < rr_range[0]) | (rr > rr_range[1])
    if method in ("malik", "interpolate"):
        prev = np.concatenate([[rr[0]], rr[:-1]])
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(rr - prev) / np.where(prev > 0, prev, np.nan)
        bad |= np.nan_to_num(rel, nan=0.0) > threshold

    if method == "none":
        return rr, t, np.zeros(len(rr), dtype=bool)
    if method == "remove":
        return rr[~bad], t[~bad], bad

    good = ~bad
    if good.sum() < 2:
        return rr, t, bad
    rr_out = rr.copy()
    rr_out[bad] = np.interp(t[bad], t[good], rr[good])
    return rr_out, t, bad


def time_domain(rr: np.ndarray) -> dict:
    """Standard time-domain indices. RR in seconds; outputs in ms."""
    rr = np.asarray(rr, dtype=float)
    if len(rr) < 2:
        return {k: float("nan") for k in
                ("mean_rr_ms", "mean_hr_bpm", "sdnn_ms", "rmssd_ms",
                 "pnn50_pct", "cvnn_pct", "median_rr_ms", "hr_max_min")}
    d = np.diff(rr)
    return dict(
        mean_rr_ms=float(np.mean(rr) * 1000),
        mean_hr_bpm=float(60.0 / np.mean(rr)),
        median_rr_ms=float(np.median(rr) * 1000),
        sdnn_ms=float(np.std(rr, ddof=1) * 1000),
        rmssd_ms=float(np.sqrt(np.mean(d**2)) * 1000),
        pnn50_pct=float(np.mean(np.abs(d) > 0.050) * 100),
        cvnn_pct=float(np.std(rr, ddof=1) / np.mean(rr) * 100),
        hr_max_min=float(60.0 / np.min(rr) - 60.0 / np.max(rr)),
    )


def poincare(rr: np.ndarray) -> dict:
    """
    Poincare plot descriptors.

    SD1/SD2 is the discriminative quantity for atrial fibrillation: it
    approaches 1 when successive intervals become uncorrelated.
    """
    rr = np.asarray(rr, dtype=float)
    if len(rr) < 3:
        return {k: float("nan") for k in ("sd1_ms", "sd2_ms", "sd1_sd2", "ellipse_area_ms2")}
    d = np.diff(rr)
    sd1 = np.sqrt(0.5) * np.std(d, ddof=1)
    var = np.var(rr, ddof=1)
    sd2 = np.sqrt(max(2 * var - sd1**2, 0.0))
    sd1_ms, sd2_ms = sd1 * 1000, sd2 * 1000
    return dict(
        sd1_ms=float(sd1_ms), sd2_ms=float(sd2_ms),
        sd1_sd2=float(sd1_ms / sd2_ms) if sd2_ms > 0 else float("nan"),
        ellipse_area_ms2=float(np.pi * sd1_ms * sd2_ms),
    )


def frequency_domain(
    rr: np.ndarray,
    t: np.ndarray,
    n_freq: int = 512,
    f_max: float = 0.5,
) -> dict:
    """
    Lomb-Scargle spectral analysis of the RR tachogram.

    Powers are returned in ms^2. Normalised units follow the Task Force
    definition, LF_nu = LF / (LF + HF) * 100, i.e. excluding VLF.
    """
    rr = np.asarray(rr, dtype=float)
    t = np.asarray(t, dtype=float)
    keys = ("vlf_ms2", "lf_ms2", "hf_ms2", "total_ms2",
            "lf_nu", "hf_nu", "lf_hf_ratio")
    if len(rr) < 8 or np.ptp(t) <= 0:
        return {k: float("nan") for k in keys}

    x = (rr - np.mean(rr)) * 1000.0  # ms, mean-removed
    f = np.linspace(1.0 / np.ptp(t), f_max, n_freq)
    w = 2 * np.pi * f
    # scipy's lombscargle returns the normalised periodogram; the 2/N factor
    # converts it to a one-sided PSD in units of x^2 per Hz.
    pgram = lombscargle(t, x, w)  # x is already mean-removed above
    psd = 2.0 * pgram / len(x)

    def band(lo, hi):
        m = (f >= lo) & (f < hi)
        return float(np.trapezoid(psd[m], f[m])) if m.sum() > 1 else 0.0

    vlf, lf, hf = band(*VLF_BAND), band(*LF_BAND), band(*HF_BAND)
    lfhf = lf + hf
    return dict(
        vlf_ms2=vlf, lf_ms2=lf, hf_ms2=hf, total_ms2=vlf + lf + hf,
        lf_nu=float(lf / lfhf * 100) if lfhf > 0 else float("nan"),
        hf_nu=float(hf / lfhf * 100) if lfhf > 0 else float("nan"),
        lf_hf_ratio=float(lf / hf) if hf > 0 else float("nan"),
    )


def hrv_features(
    peaks: np.ndarray,
    fs: float,
    correction: Literal["none", "remove", "interpolate", "malik"] = "malik",
    include_frequency: bool = True,
) -> dict:
    """Full HRV feature vector from a set of R-peak sample indices."""
    rr, t = rr_from_peaks(peaks, fs)
    if len(rr) < 2:
        out = time_domain(rr) | poincare(rr)
        if include_frequency:
            out |= frequency_domain(rr, t)
        return out | dict(n_intervals=len(rr), ectopic_pct=float("nan"))

    rr_c, t_c, flagged = clean_rr(rr, t, method=correction)
    out = time_domain(rr_c) | poincare(rr_c)
    if include_frequency:
        out |= frequency_domain(rr_c, t_c)
    out["n_intervals"] = int(len(rr_c))
    out["ectopic_pct"] = float(np.mean(flagged) * 100)
    return out
