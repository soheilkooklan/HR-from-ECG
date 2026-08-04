"""
Signal-quality indices: classical baselines and the utility-based target.

The classical indices below are the ones this project must beat. Every one of
them defines quality by a *proxy* -- how impulsive the waveform is, how much
power sits in the QRS band, how periodic the signal looks, or whether two
detectors agree. None of them is defined by the quantity a heart-rate monitor
actually cares about, namely how wrong the reported heart rate will be.

`utility_sqi` supplies that missing definition:

    uSQI(w) = exp( -|HR_hat(w) - HR_true(w)| / tau )

so uSQI = 1 when a window costs nothing and decays towards 0 as the window
degrades the estimate. tau sets the error scale at which quality is considered
half-lost; tau = 5 bpm is used throughout, matching the ANSI/AAMI EC13
tolerance. Because the label is computed from ground truth, it requires either
simulation or an annotated database -- which is precisely why the simulator in
`hrecg.simulation` is a core component rather than a convenience.

References for the baselines
----------------------------
Li, Mark & Clifford (2008), Physiol Meas 29(1):15-32 (bSQI, kSQI, pSQI).
Behar et al. (2013), Physiol Meas 34(9):1113-1138 (SQI ensembles).
Zaman et al. (2022), Front Physiol 13 (cepstral quality index).
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps
from scipy.stats import kurtosis, skew

__all__ = [
    "ksqi", "ssqi", "psqi", "basqi", "cepstral_qi", "bsqi",
    "classical_sqi_vector", "utility_sqi",
]


def ksqi(x: np.ndarray, normalise: bool = True) -> float:
    """
    Kurtosis SQI. A clean ECG is highly impulsive (kurtosis >> 3); noise
    flattens the distribution. Fails badly on electrode motion, which is
    *more* impulsive than the ECG itself.
    """
    k = float(kurtosis(x, fisher=False))
    return float(np.clip(k / 15.0, 0, 1)) if normalise else k


def ssqi(x: np.ndarray) -> float:
    """Skewness SQI: clean single-lead ECG is positively skewed by the R wave."""
    return float(skew(x))


def psqi(x: np.ndarray, fs: float, qrs_band=(5.0, 15.0), full_band=(0.5, 40.0)) -> float:
    """Relative power of the QRS band within the diagnostic band."""
    f, p = sps.welch(x, fs=fs, nperseg=min(len(x), int(2 * fs)))
    num = np.trapezoid(p[(f >= qrs_band[0]) & (f <= qrs_band[1])],
                       f[(f >= qrs_band[0]) & (f <= qrs_band[1])])
    den = np.trapezoid(p[(f >= full_band[0]) & (f <= full_band[1])],
                       f[(f >= full_band[0]) & (f <= full_band[1])])
    return float(num / den) if den > 0 else float("nan")


def basqi(x: np.ndarray, fs: float, base_band=(0.0, 1.0), full_band=(0.0, 40.0)) -> float:
    """Baseline-wander SQI: 1 minus the fraction of power below 1 Hz."""
    f, p = sps.welch(x, fs=fs, nperseg=min(len(x), int(2 * fs)))
    num = np.trapezoid(p[(f >= base_band[0]) & (f <= base_band[1])],
                       f[(f >= base_band[0]) & (f <= base_band[1])])
    den = np.trapezoid(p[(f >= full_band[0]) & (f <= full_band[1])],
                       f[(f >= full_band[0]) & (f <= full_band[1])])
    return float(1.0 - num / den) if den > 0 else float("nan")


def cepstral_qi(x: np.ndarray, fs: float, quefrency_range=(0.3, 2.0)) -> float:
    """
    Cepstral Quality Index: the share of cepstral power concentrated in the
    peak at the quefrency of the mean cardiac cycle. Rewards pseudo-periodic
    structure, so it flags both noise and irregular rhythms -- meaning it
    penalises atrial fibrillation even when every beat is perfectly detectable.
    """
    x = np.asarray(x, float) - np.mean(x)
    if len(x) < 32 or np.allclose(x, 0):
        return float("nan")
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) + 1e-12
    ceps = np.abs(np.fft.irfft(np.log(spec)))
    q = np.arange(len(ceps)) / fs
    total = float(np.sum(ceps**2)) + 1e-12
    m = (q >= quefrency_range[0]) & (q <= quefrency_range[1])
    if not np.any(m):
        return float("nan")
    return float(np.max(ceps[m]) ** 2 / total)


def bsqi(peaks_a: np.ndarray, peaks_b: np.ndarray, fs: float, tol_s: float = 0.15) -> float:
    """
    Agreement between two independent detectors (the classical bSQI).

    Strong when the two detectors fail differently, but degenerate when they
    fail the same way -- which is exactly what happens under electrode motion,
    since both are threshold-based on the same passband.
    """
    from .detection import match_peaks

    n = max(len(peaks_a), len(peaks_b))
    if n == 0:
        return float("nan")
    r = match_peaks(peaks_a, peaks_b, fs, tolerance_s=tol_s)
    return float(r.tp / (r.tp + r.fp + r.fn)) if (r.tp + r.fp + r.fn) else float("nan")


def classical_sqi_vector(x: np.ndarray, fs: float) -> dict:
    """All classical indices for one window, as a comparison baseline for uSQI."""
    return dict(
        kSQI=ksqi(x), sSQI=ssqi(x), pSQI=psqi(x, fs),
        baSQI=basqi(x, fs), CQI=cepstral_qi(x, fs),
    )


def utility_sqi(
    hr_estimated: np.ndarray,
    hr_true: np.ndarray,
    tau_bpm: float = 5.0,
) -> np.ndarray:
    """
    The proposed utility-based signal quality index.

    Defined directly on downstream cost rather than on waveform appearance:
    a window is "good" exactly to the extent that it does not corrupt the
    heart-rate estimate. Windows where no estimate can be produced at all
    (NaN) receive uSQI = 0, which lets a single scalar cover both the
    "degraded" and the "unusable" regimes.
    """
    hr_e = np.asarray(hr_estimated, float)
    hr_t = np.asarray(hr_true, float)
    err = np.abs(hr_e - hr_t)
    u = np.exp(-err / float(tau_bpm))
    return np.where(np.isfinite(err), u, 0.0)
