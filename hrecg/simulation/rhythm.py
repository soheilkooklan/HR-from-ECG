"""
Rhythm generation for HR from ECG.

Generates ground-truth RR-interval sequences with realistic autonomic
structure, plus pathological rhythms (AF, ectopy, bigeminy/trigeminy).

The RR series is the *ground truth* of the whole framework: every downstream
label (R-peak positions, HR, HRV, uSQI) is derived analytically from it, so it
must be physiologically faithful.

References
----------
McSharry, Clifford, Tarassenko & Smith (2003), "A dynamical model for
generating synthetic electrocardiogram signals", IEEE TBME 50(3):289-294.
Task Force of ESC/NASPE (1996), Circulation 93:1043-1065.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

BeatType = Literal["N", "V", "A", "F"]  # Normal, Ventricular(PVC), Atrial(APB), Fusion


@dataclass
class RhythmSeries:
    """Ground-truth beat sequence."""

    rr: np.ndarray                 # RR intervals in seconds, len = n_beats - 1
    beat_times: np.ndarray         # R-peak times in seconds, len = n_beats
    beat_types: list[BeatType]     # per-beat annotation, len = n_beats
    rhythm_label: str = "sinus"
    meta: dict = field(default_factory=dict)

    @property
    def n_beats(self) -> int:
        return len(self.beat_times)

    @property
    def mean_hr(self) -> float:
        return float(60.0 / np.mean(self.rr))

    def normal_mask(self) -> np.ndarray:
        """Boolean mask of sinus (non-ectopic) beats."""
        return np.array([b == "N" for b in self.beat_types], dtype=bool)


# --------------------------------------------------------------------------- #
#  Sinus rhythm with realistic HRV spectrum
# --------------------------------------------------------------------------- #
def rr_spectrum_series(
    n_beats: int,
    mean_hr: float = 70.0,
    sdnn_ms: float = 50.0,
    lf_hf_ratio: float = 1.5,
    f_lf: float = 0.10,
    f_hf: float = 0.25,
    c_lf: float = 0.02,
    c_hf: float = 0.02,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate an RR series whose power spectrum is the classic bimodal
    (Mayer-wave + respiratory sinus arrhythmia) shape of McSharry et al.

    S(f) = (sigma_lf^2 / sqrt(2*pi*c_lf^2)) * exp(-(f-f_lf)^2 / (2 c_lf^2))
         + (sigma_hf^2 / sqrt(2*pi*c_hf^2)) * exp(-(f-f_hf)^2 / (2 c_hf^2))

    with sigma_lf^2 / sigma_hf^2 = lf_hf_ratio. Random phases are drawn
    uniformly, an inverse FFT gives a real-valued series, which is then scaled
    to the requested SDNN and offset to the requested mean HR.

    Parameters
    ----------
    sdnn_ms : target standard deviation of NN intervals in milliseconds.
    lf_hf_ratio : target LF/HF power ratio (sympatho-vagal balance proxy).
    """
    rng = np.random.default_rng() if rng is None else rng
    mean_rr = 60.0 / mean_hr

    # Beat-domain sampling frequency is ~1/mean_rr Hz
    fs_beat = 1.0 / mean_rr
    n = int(2 ** np.ceil(np.log2(max(n_beats, 64))))  # power of two for FFT
    f = np.fft.rfftfreq(n, d=1.0 / fs_beat)

    sigma_hf2 = 1.0
    sigma_lf2 = lf_hf_ratio * sigma_hf2
    s = (
        sigma_lf2 / np.sqrt(2 * np.pi * c_lf**2) * np.exp(-((f - f_lf) ** 2) / (2 * c_lf**2))
        + sigma_hf2 / np.sqrt(2 * np.pi * c_hf**2) * np.exp(-((f - f_hf) ** 2) / (2 * c_hf**2))
    )
    s[0] = 0.0  # remove DC; mean is imposed explicitly

    amp = np.sqrt(s)
    phase = rng.uniform(0, 2 * np.pi, size=amp.shape)
    spec = amp * np.exp(1j * phase)
    x = np.fft.irfft(spec, n=n)[:n_beats]

    sd = np.std(x)
    if sd < 1e-12:
        x = np.zeros(n_beats)
    else:
        x = (x - np.mean(x)) / sd * (sdnn_ms / 1000.0)

    rr = mean_rr + x
    return np.clip(rr, 0.25, 3.0)  # physiological guard: 20-240 bpm


def rr_atrial_fibrillation(
    n_beats: int,
    mean_hr: float = 100.0,
    sd_ms: float = 180.0,
    rng: np.random.Generator | None = None,
    refractory: float = 0.24,
) -> np.ndarray:
    """
    AF RR-intervals: a shifted-gamma renewal model.

    In AF the AV node is bombarded by disorganised atrial activity, so
    conducted beats behave approximately as a renewal process with an absolute
    refractory floor. A shifted gamma (rather than a pure exponential) is used
    because the exponential is far too dispersed relative to real AF Holter
    data; the gamma shape parameter is solved analytically from the requested
    mean and standard deviation.

    This reproduces the hallmark of AF -- near-zero short-term RR correlation
    and a Poincare cloud with SD1/SD2 close to 1 -- which is precisely the
    regime where conventional HRV pipelines break down.

    Parameters
    ----------
    sd_ms : target SD of RR intervals (typical AF Holter range 120-250 ms).
    """
    rng = np.random.default_rng() if rng is None else rng
    mean_rr = 60.0 / mean_hr
    sd = sd_ms / 1000.0
    mu_g = max(mean_rr - refractory, 0.05)          # mean of the gamma part
    sd = min(sd, 0.95 * mu_g)                        # keep the shape sensible
    shape = (mu_g / sd) ** 2
    scale = sd**2 / mu_g
    rr = refractory + rng.gamma(shape=shape, scale=scale, size=n_beats)
    return np.clip(rr, 0.2, 3.0)


# --------------------------------------------------------------------------- #
#  Ectopy
# --------------------------------------------------------------------------- #
def insert_ectopic_beats(
    rr: np.ndarray,
    beat_types: list[BeatType],
    kind: Literal["pvc", "apb"] = "pvc",
    burden: float = 0.05,
    pattern: Literal["random", "bigeminy", "trigeminy"] = "random",
    coupling: float = 0.62,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[BeatType]]:
    """
    Insert premature beats into an RR sequence.

    PVC  : short coupling interval followed by a *fully compensatory* pause
           (RR_pre + RR_post ~= 2 * RR_sinus), because the sinus node is not
           reset by the ventricular ectopic.
    APB  : short coupling interval followed by an *incomplete* pause, because
           the atrial ectopic resets the sinus node.

    Parameters
    ----------
    burden : fraction of beats that are ectopic (ignored for bigeminy /
             trigeminy, which impose 1/2 and 1/3 respectively).
    coupling : coupling interval as a fraction of the preceding sinus RR.
    """
    rng = np.random.default_rng() if rng is None else rng
    rr = rr.astype(float).copy()
    beat_types = list(beat_types)
    n = len(rr)

    if pattern == "bigeminy":
        idx = np.arange(1, n - 1, 2)
    elif pattern == "trigeminy":
        idx = np.arange(1, n - 1, 3)
    else:
        k = max(int(round(burden * n)), 0)
        if k == 0:
            return rr, beat_types
        idx = rng.choice(np.arange(1, n - 1), size=min(k, n - 2), replace=False)
        idx = np.sort(idx)

    label: BeatType = "V" if kind == "pvc" else "A"
    for i in idx:
        if beat_types[i] != "N" or beat_types[i + 1] != "N":
            continue
        base = rr[i]
        pre = coupling * base
        if kind == "pvc":
            post = 2.0 * base - pre                    # full compensatory pause
        else:
            post = (2.0 * base - pre) * rng.uniform(0.82, 0.92)  # incomplete
        rr[i] = pre
        rr[i + 1] = post
        beat_types[i + 1] = label  # the (i+1)-th beat is the premature one

    return rr, beat_types


# --------------------------------------------------------------------------- #
#  Public factory
# --------------------------------------------------------------------------- #
def make_rhythm(
    duration_s: float,
    fs: float = 500.0,
    rhythm: Literal["sinus", "af", "pvc", "apb", "bigeminy", "trigeminy"] = "sinus",
    mean_hr: float = 70.0,
    sdnn_ms: float = 50.0,
    lf_hf_ratio: float = 1.5,
    ectopic_burden: float = 0.06,
    hr_trend: Sequence[float] | None = None,
    rng: np.random.Generator | None = None,
) -> RhythmSeries:
    """
    Build a ground-truth rhythm of the requested duration.

    Parameters
    ----------
    hr_trend : optional (t_frac, hr) control points, e.g. [(0, 60), (1, 120)]
               flattened as [0, 60, 1, 120]; used to impose a non-stationary
               heart-rate ramp (exercise / recovery). Applied multiplicatively
               to the stationary RR series so that HRV structure is preserved.
    """
    rng = np.random.default_rng() if rng is None else rng
    est_beats = int(duration_s * mean_hr / 60.0 * 1.6) + 32

    if rhythm == "af":
        rr = rr_atrial_fibrillation(
            est_beats, mean_hr=mean_hr, sd_ms=max(sdnn_ms, 120.0), rng=rng
        )
        beat_types: list[BeatType] = ["N"] * (est_beats + 1)
        rhythm_label = "atrial_fibrillation"
    else:
        rr = rr_spectrum_series(
            est_beats, mean_hr=mean_hr, sdnn_ms=sdnn_ms,
            lf_hf_ratio=lf_hf_ratio, rng=rng,
        )
        beat_types = ["N"] * (est_beats + 1)
        rhythm_label = "sinus"

        if rhythm in ("pvc", "apb", "bigeminy", "trigeminy"):
            kind = "apb" if rhythm == "apb" else "pvc"
            pattern = rhythm if rhythm in ("bigeminy", "trigeminy") else "random"
            rr, beat_types = insert_ectopic_beats(
                rr, beat_types, kind=kind, burden=ectopic_burden,
                pattern=pattern, rng=rng,
            )
            rhythm_label = f"sinus_with_{rhythm}"

    # Optional non-stationary HR trend
    if hr_trend is not None:
        ctrl = np.asarray(hr_trend, dtype=float).reshape(-1, 2)
        u = np.linspace(0.0, 1.0, len(rr))
        hr_t = np.interp(u, ctrl[:, 0], ctrl[:, 1])
        rr = rr * (mean_hr / hr_t)

    beat_times = np.concatenate([[0.0], np.cumsum(rr)])
    keep = beat_times <= duration_s
    beat_times = beat_times[keep]
    beat_types = beat_types[: len(beat_times)]
    rr = np.diff(beat_times)

    # Offset the first beat so recordings do not always start exactly on an R
    offset = rng.uniform(0.05, 0.45) * float(np.mean(rr))
    beat_times = beat_times + offset
    keep = beat_times <= duration_s - 0.05
    beat_times, beat_types = beat_times[keep], beat_types[: int(keep.sum())]
    rr = np.diff(beat_times)

    return RhythmSeries(
        rr=rr,
        beat_times=beat_times,
        beat_types=beat_types,
        rhythm_label=rhythm_label,
        meta=dict(fs=fs, duration_s=duration_s, target_hr=mean_hr,
                  target_sdnn_ms=sdnn_ms, lf_hf_ratio=lf_hf_ratio),
    )
