"""
Window extraction and label construction.

The label set is what distinguishes this project from a segmentation baseline,
so it is worth being explicit about how each target is built.

beat    Gaussian bumps centred on each annotated R peak, sigma ~ 12 ms. Soft
        targets rather than one-hot spikes, because a one-hot target makes the
        loss landscape nearly flat everywhere and forces the network to be
        certain about a boundary that the annotators themselves place with
        several milliseconds of scatter.

offset  For every sample within one sample of a beat, the signed distance to
        the true (possibly fractional) beat instant. This is what buys
        sub-sample localisation.

usqi    The utility-based signal quality index, computed by running a
        *reference* estimator on the corrupted window, comparing its heart rate
        with the known truth, and mapping the error through exp(-|e|/tau).
        Note carefully that the target is a property of the *signal*, not of
        the network -- it measures how much this segment would degrade any
        reasonable estimator, which is why the resulting score transfers.

hr      The true windowed heart rate, target of the quantile head.

Because uSQI labels require knowing the true heart rate, training uses either
simulated recordings (unlimited, exact) or annotated databases with synthetic
contamination applied on top (real morphology, known truth). Both paths are
supported here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["make_labels", "WindowSampler", "SyntheticDataset", "collate"]


def make_labels(
    n: int,
    fs: float,
    r_peaks: np.ndarray,
    sigma_ms: float = 12.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (beat, offset, mask) targets for a window of length n.

    `r_peaks` may be fractional. Returns float32 arrays.
    """
    beat = np.zeros(n, dtype=np.float32)
    offset = np.zeros(n, dtype=np.float32)
    mask = np.zeros(n, dtype=np.float32)
    sigma = max(sigma_ms * fs / 1000.0, 0.8)
    half = int(np.ceil(4 * sigma))

    for p in np.asarray(r_peaks, dtype=float):
        c = int(round(p))
        lo, hi = max(c - half, 0), min(c + half + 1, n)
        if hi <= lo:
            continue
        idx = np.arange(lo, hi)
        beat[lo:hi] = np.maximum(beat[lo:hi],
                                 np.exp(-((idx - p) ** 2) / (2 * sigma**2)))
        # offset supervision only on the one or two samples straddling the peak
        near = np.abs(idx - p) <= 1.0
        offset[idx[near]] = (p - idx[near]).astype(np.float32)
        mask[idx[near]] = 1.0

    return beat, offset, mask


@dataclass
class WindowSampler:
    """Slice a record into fixed windows with optional random jitter."""

    window: int
    stride: int | None = None
    random: bool = True

    def starts(self, n: int, rng: np.random.Generator, n_windows: int | None = None):
        if self.random:
            k = n_windows or max(n // self.window, 1)
            return rng.integers(0, max(n - self.window, 1), size=k)
        stride = self.stride or self.window
        return np.arange(0, max(n - self.window, 0) + 1, stride)


class SyntheticDataset:
    """
    On-the-fly generator of contaminated windows with full ground truth.

    Every sample is freshly synthesised, so the model never sees the same noise
    realisation twice and the effective training-set size is unbounded. This
    matters because the quality head would otherwise memorise a finite set of
    artefact instances rather than learning what artefact looks like.

    Yields dictionaries with keys: signal, beat, offset, beat_mask, usqi, hr.
    """

    RHYTHMS = ("sinus", "sinus", "sinus", "af", "pvc", "apb", "bigeminy")

    def __init__(
        self,
        n_items: int,
        fs: float = 250.0,
        window: int = 2048,
        tau_bpm: float = 5.0,
        snr_range: tuple[float, float] = (-8.0, 28.0),
        seed: int = 0,
        reference_detector=None,
    ):
        self.n_items = n_items
        self.fs = fs
        self.window = window
        self.tau = tau_bpm
        self.snr_range = snr_range
        self.seed = seed
        if reference_detector is None:
            from ..baselines import pan_tompkins
            reference_detector = pan_tompkins
        self.ref = reference_detector

    def __len__(self) -> int:
        return self.n_items

    def __getitem__(self, i: int) -> dict:
        from ..simulation import corrupt, make_synthetic_ecg

        rng = np.random.default_rng(self.seed * 1_000_003 + i)
        dur = self.window / self.fs + 4.0
        rhythm = str(rng.choice(self.RHYTHMS))
        hr = float(rng.uniform(45, 140))
        sdnn = float(rng.uniform(15, 90))

        e = make_synthetic_ecg(dur, self.fs, rhythm, mean_hr=hr, sdnn_ms=sdnn,
                               seed=int(rng.integers(0, 2**31)))
        c = corrupt(e.signal, self.fs, snr_kind=str(rng.choice(["constant", "piecewise", "ramp"])),
                    snr_range=self.snr_range, p_lead_off=0.2,
                    powerline_hz=float(rng.choice([50.0, 60.0])), rng=rng)

        s0 = int(rng.integers(0, max(len(c.signal) - self.window, 1)))
        sl = slice(s0, s0 + self.window)
        x = c.signal[sl].astype(np.float32)
        if len(x) < self.window:
            x = np.pad(x, (0, self.window - len(x)))

        # fractional ground-truth positions: without them the offset head has
        # nothing to learn, since rounded peaks give an offset of exactly zero
        pex = e.r_peaks_exact
        peaks = pex[(pex >= s0) & (pex < s0 + self.window)] - s0
        beat, offset, mask = make_labels(self.window, self.fs, peaks)

        # ---- utility-based quality target -----------------------------------
        det = self.ref(x, self.fs)
        hr_true = self._hr(peaks, self.window)
        hr_ref = self._hr(det, self.window)
        # per-sample uSQI: local agreement between reference and truth, so that
        # quality can vary within the window rather than being a single scalar
        usqi = self._local_usqi(peaks, det, self.window, mask_len=int(2.0 * self.fs))

        # robust normalisation, identical to inference
        med = float(np.median(x))
        iqr = float(np.subtract(*np.percentile(x, [75, 25])))
        x = (x - med) / max(iqr, 1e-3)

        return dict(
            signal=x.astype(np.float32),
            beat=beat, offset=offset, beat_mask=mask,
            usqi=usqi.astype(np.float32),
            hr=np.float32(hr_true if np.isfinite(hr_true) else np.nan),
            meta=dict(rhythm=rhythm, hr_ref=hr_ref,
                      snr=float(np.mean(c.snr_db[sl]))),
        )

    def _hr(self, peaks: np.ndarray, n: int) -> float:
        p = np.sort(np.asarray(peaks, dtype=float))
        if len(p) < 3:
            return float("nan")
        return float(60.0 / np.mean(np.diff(p) / self.fs))

    def _local_usqi(self, truth: np.ndarray, det: np.ndarray, n: int,
                    mask_len: int) -> np.ndarray:
        """
        Sliding-window uSQI at 1 s resolution, then upsampled.

        Local heart rate is compared inside a 2 s neighbourhood, so a burst of
        artefact lowers quality only where it occurs instead of condemning the
        whole window.
        """
        step = int(self.fs)
        centres = np.arange(0, n, step)
        vals = np.ones(len(centres), dtype=float)
        t = np.asarray(truth, float)
        d = np.asarray(det, float)
        for k, c0 in enumerate(centres):
            lo, hi = c0 - mask_len // 2, c0 + mask_len // 2
            tt = t[(t >= lo) & (t < hi)]
            dd = d[(d >= lo) & (d < hi)]
            if len(tt) < 2:
                vals[k] = 1.0 if len(dd) < 2 else 0.5
                continue
            hr_t = 60.0 / np.mean(np.diff(tt) / self.fs)
            if len(dd) < 2:
                vals[k] = 0.0
                continue
            hr_d = 60.0 / np.mean(np.diff(dd) / self.fs)
            vals[k] = float(np.exp(-abs(hr_d - hr_t) / self.tau))
        return np.interp(np.arange(n), centres, vals).astype(np.float32)


def collate(items: list[dict]) -> dict:
    """Stack a list of dataset items into torch tensors."""
    import torch

    out = {}
    for k in ("signal", "beat", "offset", "beat_mask", "usqi"):
        out[k] = torch.from_numpy(np.stack([it[k] for it in items]))
    out["hr"] = torch.tensor([it["hr"] for it in items], dtype=torch.float32)
    out["meta"] = [it["meta"] for it in items]
    return out
