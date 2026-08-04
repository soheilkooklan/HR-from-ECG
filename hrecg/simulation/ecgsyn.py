"""
Waveform synthesis for HR from ECG.

Phase-domain implementation of the McSharry dynamical ECG model. Each beat is
rendered as a sum of Gaussians in cardiac phase theta, with R fixed at theta=0:

    z(theta) = sum_i a_i * exp( -(theta - theta_i)^2 / (2 b_i^2) )

Compared with numerically integrating the original 3-D ODE, the phase-domain
form gives *exact* R-peak times (theta = 0 by construction), which is what makes
the ground truth of this framework analytically clean.

Two rate-dependent corrections are applied that plain ECGSYN omits and that
matter a great deal for detector benchmarking:

1.  QRS *duration* is rate-independent in reality, so its angular width must
    shrink as RR grows:      b_QRS(RR) = b_QRS_ref * (RR_ref / RR)
2.  QT interval follows Bazett, QT ~ QT_ref * sqrt(RR / RR_ref), so the T-wave
    phase position scales as: theta_T(RR) = theta_T_ref * sqrt(RR_ref / RR)

References
----------
McSharry et al. (2003), IEEE TBME 50(3):289-294.
Bazett (1920), Heart 7:353-370.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .rhythm import RhythmSeries

RR_REF = 60.0 / 70.0  # reference RR at which the default parameters hold


@dataclass
class WaveParams:
    """Gaussian parameters for one PQRST complex (angles in radians)."""

    theta: np.ndarray  # centres
    a: np.ndarray      # amplitudes
    b: np.ndarray      # angular widths
    names: tuple[str, ...] = ("P", "Q", "R", "S", "T")

    @staticmethod
    def normal() -> "WaveParams":
        return WaveParams(
            theta=np.array([-np.pi / 3, -np.pi / 12, 0.0, np.pi / 12, np.pi / 2]),
            a=np.array([0.14, -0.14, 1.00, -0.28, 0.32]),
            b=np.array([0.25, 0.10, 0.10, 0.10, 0.42]),
        )

    @staticmethod
    def pvc(rng: np.random.Generator) -> "WaveParams":
        """
        Ventricular ectopic: no P wave, wide bizarre QRS, T wave discordant
        (opposite polarity to the main QRS deflection).
        """
        pol = rng.choice([-1.0, 1.0])
        return WaveParams(
            theta=np.array([-np.pi / 3, -np.pi / 6, 0.0, np.pi / 6, np.pi / 2]),
            a=np.array([0.0, -0.30 * pol, 1.45 * pol, -0.55 * pol, -0.45 * pol]),
            b=np.array([0.25, 0.22, 0.24, 0.26, 0.50]),
        )

    @staticmethod
    def apb() -> "WaveParams":
        """Atrial ectopic: narrow QRS (supraventricular), abnormal P morphology."""
        p = WaveParams.normal()
        theta = p.theta.copy()
        a = p.a.copy()
        b = p.b.copy()
        a[0] *= -0.6          # inverted / low-amplitude P
        theta[0] = -np.pi / 4  # shorter PR
        return WaveParams(theta=theta, a=a, b=b)


@dataclass
class SyntheticECG:
    """A synthetic recording together with its exact ground truth."""

    signal: np.ndarray          # clean ECG, millivolts
    fs: float
    r_peaks: np.ndarray         # R-peak sample indices, rounded to the grid
    r_peaks_exact: np.ndarray   # fractional R-peak positions in samples
    rr: np.ndarray              # true RR intervals, seconds
    beat_types: list[str]
    rhythm_label: str
    respiration: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return len(self.signal) / self.fs

    @property
    def t(self) -> np.ndarray:
        return np.arange(len(self.signal)) / self.fs


def _render_beat(
    t_local: np.ndarray,
    rr_prev: float,
    rr_next: float,
    p: WaveParams,
    amp_scale: float,
) -> np.ndarray:
    """
    Render one PQRST complex on a local time axis centred at the R peak.

    Phase is defined piecewise so that the pre-R part (P, Q) is normalised by
    the preceding RR and the post-R part (S, T) by the following RR -- this is
    what makes the model behave correctly across an ectopic beat, where the two
    neighbouring intervals differ by a factor of two.
    """
    out = np.zeros_like(t_local)
    for side, rr_side in ((-1, rr_prev), (1, rr_next)):
        m = (t_local < 0) if side < 0 else (t_local >= 0)
        if not np.any(m):
            continue
        theta = 2.0 * np.pi * t_local[m] / rr_side

        # rate-dependent corrections
        qrs_scale = RR_REF / rr_side       # keeps QRS duration constant in time
        qt_scale = np.sqrt(RR_REF / rr_side)  # Bazett

        acc = np.zeros(theta.shape)
        for j, name in enumerate(p.names):
            if abs(p.a[j]) < 1e-9:
                continue
            th_j, b_j = p.theta[j], p.b[j]
            if name in ("Q", "R", "S"):
                th_j *= qrs_scale
                b_j *= qrs_scale
            elif name == "T":
                th_j *= qt_scale
                b_j *= qt_scale
            elif name == "P":
                th_j *= qrs_scale ** 0.5
                b_j *= qrs_scale ** 0.5
            acc += p.a[j] * np.exp(-((theta - th_j) ** 2) / (2.0 * b_j**2))
        out[m] = acc
    return amp_scale * out


def synthesize(
    rhythm: RhythmSeries,
    fs: float = 500.0,
    duration_s: float | None = None,
    resp_rate_hz: float = 0.25,
    resp_am_depth: float = 0.06,
    beat_amp_jitter: float = 0.03,
    rng: np.random.Generator | None = None,
) -> SyntheticECG:
    """
    Turn a ground-truth rhythm into a clean ECG waveform.

    Parameters
    ----------
    resp_am_depth : depth of respiration-induced amplitude modulation. This is
        the physiological basis of ECG-derived respiration and also a realistic
        nuisance for fixed-threshold detectors.
    beat_amp_jitter : per-beat multiplicative amplitude jitter (std), modelling
        small changes in the cardiac electrical axis.
    """
    rng = np.random.default_rng() if rng is None else rng
    duration_s = duration_s or float(rhythm.meta.get("duration_s", rhythm.beat_times[-1] + 1.0))
    n = int(round(duration_s * fs))
    sig = np.zeros(n, dtype=float)
    t = np.arange(n) / fs

    resp_phase = rng.uniform(0, 2 * np.pi)
    resp = np.sin(2 * np.pi * resp_rate_hz * t + resp_phase)

    bt = rhythm.beat_times
    rr = rhythm.rr
    n_beats = len(bt)

    for k in range(n_beats):
        rr_prev = rr[k - 1] if k > 0 else (rr[0] if len(rr) else 0.85)
        rr_next = rr[k] if k < len(rr) else (rr[-1] if len(rr) else 0.85)

        btype = rhythm.beat_types[k]
        if btype == "V":
            p = WaveParams.pvc(rng)
        elif btype == "A":
            p = WaveParams.apb()
        else:
            p = WaveParams.normal()

        # local support: +-60% of the neighbouring intervals
        lo = bt[k] - 0.6 * rr_prev
        hi = bt[k] + 0.6 * rr_next
        i0, i1 = int(np.floor(lo * fs)), int(np.ceil(hi * fs))
        i0, i1 = max(i0, 0), min(i1, n)
        if i1 <= i0:
            continue

        t_local = t[i0:i1] - bt[k]
        amp = 1.0 + beat_amp_jitter * rng.standard_normal()
        amp *= 1.0 + resp_am_depth * resp[min(int(bt[k] * fs), n - 1)]
        sig[i0:i1] += _render_beat(t_local, rr_prev, rr_next, p, amp)

    r_exact = bt * fs
    r_idx = np.round(r_exact).astype(int)
    valid = (r_idx >= 0) & (r_idx < n)
    r_idx = r_idx[valid]
    r_exact = r_exact[valid]
    beat_types = [rhythm.beat_types[i] for i in np.flatnonzero(valid)]

    return SyntheticECG(
        signal=sig,
        fs=fs,
        r_peaks=r_idx,
        r_peaks_exact=r_exact,
        rr=np.diff(bt[valid]),
        beat_types=beat_types,
        rhythm_label=rhythm.rhythm_label,
        respiration=resp,
        meta=dict(rhythm.meta, fs=fs, resp_rate_hz=resp_rate_hz),
    )


def make_synthetic_ecg(
    duration_s: float = 60.0,
    fs: float = 500.0,
    rhythm: str = "sinus",
    mean_hr: float = 70.0,
    sdnn_ms: float = 50.0,
    seed: int | None = None,
    **kwargs,
) -> SyntheticECG:
    """One-call convenience wrapper: rhythm generation + waveform synthesis."""
    from .rhythm import make_rhythm

    rng = np.random.default_rng(seed)
    rhythm_kwargs = {k: kwargs.pop(k) for k in
                     ("lf_hf_ratio", "ectopic_burden", "hr_trend") if k in kwargs}
    rs = make_rhythm(
        duration_s=duration_s, fs=fs, rhythm=rhythm, mean_hr=mean_hr,
        sdnn_ms=sdnn_ms, rng=rng, **rhythm_kwargs,
    )
    return synthesize(rs, fs=fs, duration_s=duration_s, rng=rng, **kwargs)
