"""
Noise and artifact models for HR from ECG.

Three canonical ECG contaminants are reproduced (matching the composition of
the MIT-BIH Noise Stress Test Database), plus several failure modes that occur
in wearable recordings but are absent from NSTDB:

    baseline_wander   -- respiration + electrode half-cell drift, 0.05-0.6 Hz
    powerline         -- 50/60 Hz mains and harmonics, with slow amplitude drift
    muscle_artifact   -- surface EMG, 20-200 Hz, bursty (contraction) envelope
    electrode_motion  -- the hard one: sparse high-amplitude transients whose
                         spectrum overlaps the QRS complex, so no linear filter
                         can remove them
    lead_off          -- saturation / flatline / rail-to-rail excursions
    quantization      -- finite ADC resolution

The module's real purpose is `corrupt()`, which mixes noise at a *prescribed,
time-varying* SNR and returns the per-sample SNR trace. That trace is the
supervisory signal for the utility-based signal quality index (uSQI): quality
is defined by how much a segment degrades downstream heart-rate error, and
that requires knowing exactly how contaminated each sample is.

References
----------
Moody, Muldrow & Mark (1984), "A noise stress test for arrhythmia detectors",
Computers in Cardiology 11:381-384.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sps

__all__ = [
    "baseline_wander", "powerline", "muscle_artifact", "electrode_motion",
    "lead_off_segments", "quantize", "mix_at_snr", "snr_profile",
    "corrupt", "CorruptedECG", "load_nstdb_noise",
]


# --------------------------------------------------------------------------- #
#  Individual noise sources (all returned with unit RMS)
# --------------------------------------------------------------------------- #
def _unit_rms(x: np.ndarray) -> np.ndarray:
    r = np.sqrt(np.mean(x**2))
    return x / r if r > 1e-12 else x


def baseline_wander(
    n: int,
    fs: float,
    resp_rate_hz: float = 0.25,
    n_components: int = 4,
    walk_weight: float = 0.4,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Low-frequency drift: respiratory component + electrode half-cell random walk."""
    rng = np.random.default_rng() if rng is None else rng
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * resp_rate_hz * t + rng.uniform(0, 2 * np.pi))
    for _ in range(n_components - 1):
        f = rng.uniform(0.05, 0.6)
        x += rng.uniform(0.2, 0.8) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))

    walk = np.cumsum(rng.standard_normal(n))
    b, a = sps.butter(2, min(0.5 / (fs / 2), 0.99), btype="low")
    walk = sps.filtfilt(b, a, walk)
    x = (1 - walk_weight) * _unit_rms(x) + walk_weight * _unit_rms(walk)
    return _unit_rms(x)


def powerline(
    n: int,
    fs: float,
    f0: float = 50.0,
    harmonics: int = 3,
    am_depth: float = 0.25,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Mains interference with harmonics and slow amplitude modulation."""
    rng = np.random.default_rng() if rng is None else rng
    t = np.arange(n) / fs
    x = np.zeros(n)
    for h in range(1, harmonics + 1):
        f = f0 * h
        if f >= fs / 2:
            break
        x += (1.0 / h) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    am = 1.0 + am_depth * np.sin(2 * np.pi * rng.uniform(0.01, 0.1) * t)
    return _unit_rms(x * am)


def muscle_artifact(
    n: int,
    fs: float,
    band: tuple[float, float] = (20.0, 200.0),
    burst_rate_hz: float = 0.3,
    burst_len_s: float = 1.2,
    stationary_frac: float = 0.35,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Surface EMG: band-limited noise gated by a bursty contraction envelope.

    `stationary_frac` keeps a continuous low-level component so that the
    artifact is never fully absent between bursts, as in real recordings.
    """
    rng = np.random.default_rng() if rng is None else rng
    hi = min(band[1], 0.98 * fs / 2)
    lo = min(band[0], hi * 0.5)
    b, a = sps.butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    x = sps.filtfilt(b, a, rng.standard_normal(n))

    env = np.full(n, stationary_frac)
    n_bursts = max(int(burst_rate_hz * n / fs), 0)
    L = max(int(burst_len_s * fs), 8)
    win = sps.windows.tukey(L, alpha=0.5)
    for _ in range(n_bursts):
        s = rng.integers(0, max(n - L, 1))
        env[s:s + L] += rng.uniform(0.6, 1.4) * win[: min(L, n - s)]
    return _unit_rms(x * env)


def electrode_motion(
    n: int,
    fs: float,
    event_rate_hz: float = 0.35,
    decay_s: float = 0.45,
    osc_band: tuple[float, float] = (1.0, 10.0),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Electrode motion artifact -- the dominant cause of false QRS detections.

    Modelled as a sparse marked point process: each event is a damped
    oscillation in the 1-10 Hz band, i.e. spectrally *overlapping* the QRS
    complex. This is why electrode motion cannot be removed by band-pass
    filtering and why a quality-aware, abstaining estimator is needed rather
    than a better filter.
    """
    rng = np.random.default_rng() if rng is None else rng
    x = np.zeros(n)
    n_events = max(int(event_rate_hz * n / fs), 1)
    for _ in range(n_events):
        s = int(rng.integers(0, max(n - 1, 1)))
        L = min(int(rng.uniform(2.0, 6.0) * decay_s * fs), n - s)
        if L < 8:
            continue
        tt = np.arange(L) / fs
        f = rng.uniform(*osc_band)
        amp = rng.uniform(0.5, 3.0) * rng.choice([-1.0, 1.0])
        x[s:s + L] += amp * np.exp(-tt / (decay_s * rng.uniform(0.6, 1.6))) * \
            np.sin(2 * np.pi * f * tt + rng.uniform(0, 2 * np.pi))

    b, a = sps.butter(2, min(15.0 / (fs / 2), 0.99), btype="low")
    return _unit_rms(sps.filtfilt(b, a, x))


def lead_off_segments(
    x: np.ndarray,
    fs: float,
    n_events: int = 1,
    dur_s: tuple[float, float] = (0.8, 3.0),
    mode: str = "flat",
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Insert electrode-disconnection events.

    Returns the corrupted signal and a boolean mask of affected samples. These
    regions are unrecoverable by construction and therefore form the natural
    positive class for the abstention head.
    """
    rng = np.random.default_rng() if rng is None else rng
    y = x.copy()
    mask = np.zeros(len(x), dtype=bool)
    for _ in range(n_events):
        L = int(rng.uniform(*dur_s) * fs)
        if L >= len(x):
            continue
        s = int(rng.integers(0, len(x) - L))
        if mode == "flat":
            y[s:s + L] = y[s] + 0.01 * rng.standard_normal(L)
        else:  # rail-to-rail saturation
            y[s:s + L] = np.sign(rng.standard_normal()) * np.max(np.abs(x)) * 3.0
        mask[s:s + L] = True
    return y, mask


def quantize(x: np.ndarray, n_bits: int = 12, full_scale_mv: float = 5.0) -> np.ndarray:
    """Uniform mid-tread ADC quantization."""
    lsb = 2 * full_scale_mv / (2**n_bits)
    return np.round(np.clip(x, -full_scale_mv, full_scale_mv) / lsb) * lsb


# --------------------------------------------------------------------------- #
#  SNR-controlled mixing
# --------------------------------------------------------------------------- #
def _local_power(x: np.ndarray, fs: float, win_s: float = 2.0) -> np.ndarray:
    L = max(int(win_s * fs) | 1, 3)
    k = np.ones(L) / L
    return np.convolve(x**2, k, mode="same") + 1e-12


def snr_profile(
    n: int,
    fs: float,
    kind: str = "piecewise",
    snr_range: tuple[float, float] = (-6.0, 24.0),
    n_segments: int = 6,
    smooth_s: float = 1.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Build a per-sample SNR trajectory in dB.

    `kind`:
      "constant"  -- fixed SNR (classic NSTDB-style stress test)
      "piecewise" -- random piecewise-constant segments, smoothed (wearables)
      "ramp"      -- monotone degradation (electrode drying out over time)
    """
    rng = np.random.default_rng() if rng is None else rng
    lo, hi = snr_range
    if kind == "constant":
        return np.full(n, rng.uniform(lo, hi))
    if kind == "ramp":
        return np.linspace(hi, lo, n)

    edges = np.sort(rng.choice(np.arange(1, n), size=max(n_segments - 1, 1), replace=False))
    edges = np.concatenate([[0], edges, [n]])
    prof = np.empty(n)
    for i in range(len(edges) - 1):
        prof[edges[i]:edges[i + 1]] = rng.uniform(lo, hi)
    L = max(int(smooth_s * fs) | 1, 3)
    return np.convolve(prof, np.ones(L) / L, mode="same")


def mix_at_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float | np.ndarray,
    fs: float,
    local: bool = True,
    win_s: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Add `noise` to `clean` at the requested SNR (scalar or per-sample array).

    With `local=True` the SNR is enforced against a running estimate of signal
    power, so a prescribed SNR means the same thing at every point in time even
    when the ECG amplitude drifts. Returns (noisy, gain_trace).
    """
    noise = _unit_rms(noise)
    snr_db = np.asarray(snr_db, dtype=float)
    if snr_db.ndim == 0:
        snr_db = np.full(len(clean), float(snr_db))

    if local:
        p_sig = _local_power(clean, fs, win_s)
    else:
        p_sig = np.full(len(clean), np.mean(clean**2) + 1e-12)

    gain = np.sqrt(p_sig / (10.0 ** (snr_db / 10.0)))
    return clean + gain * noise, gain


# --------------------------------------------------------------------------- #
#  Composite corruption
# --------------------------------------------------------------------------- #
@dataclass
class CorruptedECG:
    signal: np.ndarray            # contaminated ECG
    clean: np.ndarray             # the underlying clean ECG
    fs: float
    snr_db: np.ndarray            # per-sample achieved SNR in dB
    unrecoverable: np.ndarray     # bool mask of lead-off / saturated samples
    components: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def windowed_snr(self, win_s: float = 10.0) -> np.ndarray:
        L = int(win_s * self.fs)
        k = len(self.snr_db) // L
        return self.snr_db[: k * L].reshape(k, L).mean(axis=1)


def corrupt(
    clean: np.ndarray,
    fs: float,
    weights: dict[str, float] | None = None,
    snr_kind: str = "piecewise",
    snr_range: tuple[float, float] = (-6.0, 24.0),
    powerline_hz: float = 50.0,
    p_lead_off: float = 0.15,
    n_bits: int | None = 12,
    rng: np.random.Generator | None = None,
) -> CorruptedECG:
    """
    Apply a realistic composite contamination to a clean ECG.

    `weights` gives the relative RMS contribution of each source; the default
    mirrors ambulatory single-lead recordings, where electrode motion and EMG
    dominate. The mixture is normalised to unit RMS *before* SNR scaling, so
    the requested SNR is respected regardless of the weights.
    """
    rng = np.random.default_rng() if rng is None else rng
    n = len(clean)
    weights = weights or {"bw": 1.0, "ma": 0.8, "em": 0.9, "pl": 0.35}

    comps = {
        "bw": baseline_wander(n, fs, rng=rng),
        "ma": muscle_artifact(n, fs, rng=rng),
        "em": electrode_motion(n, fs, rng=rng),
        "pl": powerline(n, fs, f0=powerline_hz, rng=rng),
    }
    mixture = sum(weights.get(k, 0.0) * v for k, v in comps.items())
    mixture = _unit_rms(np.asarray(mixture))

    snr = snr_profile(n, fs, kind=snr_kind, snr_range=snr_range, rng=rng)
    noisy, gain = mix_at_snr(clean, mixture, snr, fs, local=True)

    unrec = np.zeros(n, dtype=bool)
    if rng.random() < p_lead_off:
        noisy, unrec = lead_off_segments(
            noisy, fs, n_events=int(rng.integers(1, 3)),
            mode=str(rng.choice(["flat", "sat"])), rng=rng,
        )
        snr = np.where(unrec, -40.0, snr)

    if n_bits is not None:
        noisy = quantize(noisy, n_bits=n_bits)

    return CorruptedECG(
        signal=noisy, clean=clean, fs=fs, snr_db=snr, unrecoverable=unrec,
        components=comps,
        meta=dict(weights=weights, snr_kind=snr_kind, snr_range=snr_range,
                  powerline_hz=powerline_hz, n_bits=n_bits),
    )


# --------------------------------------------------------------------------- #
#  Real noise from the MIT-BIH Noise Stress Test Database
# --------------------------------------------------------------------------- #
def load_nstdb_noise(
    record: str = "em",
    fs_target: float = 500.0,
    n_samples: int | None = None,
    pn_dir: str = "nstdb",
    rng: np.random.Generator | None = None,
    local_dir=None,
) -> np.ndarray:
    """
    Fetch a real noise record ('bw', 'em' or 'ma') from PhysioNet.

    Requires `wfdb` and network access to physionet.org. Use this for the
    headline noise-stress results in the paper, and the synthetic generators
    above for large-scale training augmentation, where unlimited independent
    realisations matter more than exact spectral fidelity.
    """
    import wfdb  # imported lazily: not needed for synthetic-only workflows

    if local_dir is not None:
        from pathlib import Path as _P
        rec = wfdb.rdrecord(str(_P(local_dir) / record))
    else:
        rec = wfdb.rdrecord(record, pn_dir=pn_dir)
    x = np.asarray(rec.p_signal[:, 0], dtype=float)
    x = x[np.isfinite(x)]

    if abs(rec.fs - fs_target) > 1e-6:
        n_out = int(round(len(x) * fs_target / rec.fs))
        x = sps.resample(x, n_out)

    if n_samples is not None:
        rng = np.random.default_rng() if rng is None else rng
        if len(x) < n_samples:
            reps = int(np.ceil(n_samples / len(x)))
            x = np.tile(x, reps)
        s = int(rng.integers(0, len(x) - n_samples + 1))
        x = x[s:s + n_samples]

    return _unit_rms(x)
