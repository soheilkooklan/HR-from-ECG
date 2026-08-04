"""
Pan-Tompkins QRS detector -- the classical reference baseline.

A faithful implementation is included rather than a dependency, because the
several open-source variants in circulation differ in their threshold update
rules and search-back logic, and those differences are large enough to change
the ranking of methods under noise. Reproducibility of the *baseline* matters
as much as reproducibility of the proposed method.

Reference
---------
Pan & Tompkins (1985), "A real-time QRS detection algorithm",
IEEE TBME 32(3):230-236.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

__all__ = ["pan_tompkins", "bandpass_ecg", "refine_r_peaks"]


def bandpass_ecg(x: np.ndarray, fs: float, lo: float = 5.0, hi: float = 15.0) -> np.ndarray:
    """Zero-phase Butterworth band-pass; the Pan-Tompkins passband."""
    hi = min(hi, 0.98 * fs / 2)
    b, a = sps.butter(3, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return sps.filtfilt(b, a, x)


def refine_r_peaks(
    x: np.ndarray,
    peaks: np.ndarray,
    fs: float,
    search_ms: float = 50.0,
    polarity: str = "auto",
) -> np.ndarray:
    """
    Snap detections onto the true extremum of the *unfiltered* signal.

    Integration-based detectors systematically lag the R peak by roughly half
    the integration window. Left uncorrected this appears as a bias of 20-40 ms
    in localisation error, and it inflates RMSSD, so refinement is not
    cosmetic.
    """
    w = int(search_ms * fs / 1000)
    out = []
    for p in np.asarray(peaks, dtype=int):
        s, e = max(p - w, 0), min(p + w + 1, len(x))
        if e <= s:
            out.append(p)
            continue
        seg = x[s:e]
        if polarity == "neg":
            out.append(s + int(np.argmin(seg)))
        elif polarity == "pos":
            out.append(s + int(np.argmax(seg)))
        else:
            out.append(s + int(np.argmax(np.abs(seg - np.median(seg)))))
    return np.unique(np.asarray(out, dtype=int))


def pan_tompkins(
    x: np.ndarray,
    fs: float,
    integration_ms: float = 150.0,
    refractory_ms: float = 200.0,
    t_wave_ms: float = 360.0,
    refine: bool = True,
) -> np.ndarray:
    """
    Detect R peaks and return their sample indices.

    Pipeline: band-pass -> differentiate -> square -> moving-window integrate
    -> adaptive dual thresholds with search-back and T-wave rejection.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < int(fs):
        return np.array([], dtype=int)

    filt = bandpass_ecg(x, fs)
    deriv = np.convolve(filt, np.array([1, 2, 0, -2, -1]) * fs / 8.0, mode="same")
    sq = deriv**2
    w = max(int(integration_ms * fs / 1000), 1)
    integ = np.convolve(sq, np.ones(w) / w, mode="same")

    refr = int(refractory_ms * fs / 1000)
    twave = int(t_wave_ms * fs / 1000)
    cand, _ = sps.find_peaks(integ, distance=max(refr, 1))
    if len(cand) == 0:
        return np.array([], dtype=int)

    # Initialise from the first two seconds
    init = integ[: int(2 * fs)]
    spki = float(np.max(init)) * 0.25 if len(init) else float(np.max(integ)) * 0.25
    npki = float(np.mean(init)) * 0.5 if len(init) else float(np.mean(integ)) * 0.5

    peaks: list[int] = []
    rr_hist: list[int] = []
    rr_avg2 = None
    i = 0
    while i < len(cand):
        p = int(cand[i])
        thr1 = npki + 0.25 * (spki - npki)
        val = integ[p]
        is_qrs = False

        if val > thr1:
            if peaks and (p - peaks[-1]) < twave:
                # T-wave discrimination: a T wave rises more slowly than a QRS
                def _slope(idx: int) -> float:
                    seg = filt[max(idx - w // 2, 0): idx + 1]
                    return float(np.max(np.abs(np.diff(seg)))) if len(seg) > 1 else 0.0

                s_new, s_old = _slope(p), _slope(peaks[-1])
                is_qrs = s_new > 0.5 * s_old
            else:
                is_qrs = True

        if is_qrs:
            spki = 0.125 * val + 0.875 * spki
            if peaks:
                rr_hist.append(p - peaks[-1])
                rr_hist = rr_hist[-8:]
                rr_avg2 = float(np.mean(rr_hist))
            peaks.append(p)
        else:
            npki = 0.125 * val + 0.875 * npki

        # Search-back: if no beat for 1.66x the running RR, rescan at half threshold
        if rr_avg2 and peaks and i + 1 < len(cand):
            gap = cand[i + 1] - peaks[-1]
            if gap > 1.66 * rr_avg2:
                lo, hi = peaks[-1] + refr, int(cand[i + 1])
                if hi > lo:
                    seg = integ[lo:hi]
                    j = int(np.argmax(seg))
                    if seg[j] > 0.5 * thr1:
                        peaks.append(lo + j)
                        spki = 0.25 * seg[j] + 0.75 * spki
        i += 1

    out = np.unique(np.asarray(peaks, dtype=int))
    return refine_r_peaks(x, out, fs) if refine and len(out) else out
