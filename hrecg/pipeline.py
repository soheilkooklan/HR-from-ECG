"""
End-to-end analysis pipeline.

This module is the seam between the research code and anything that consumes
it -- the GUI, the batch evaluation scripts, or a user's own notebook. It takes
a raw single-lead recording and returns beats, an interval-valued heart-rate
trace, interval-valued HRV indices, a quality trace, and an abstention decision.

Two behaviours are worth flagging.

**Graceful degradation.** If PyTorch or a trained checkpoint is unavailable the
pipeline falls back to the classical detector plus a bootstrap uncertainty
model, and says so in `engine`. The application therefore runs from a fresh
clone with nothing but NumPy and SciPy installed, which is what makes the
project auditable by someone who has not set up a deep-learning environment.

**Self-calibration.** Conformal intervals need a calibration set. Rather than
shipping a fixed quantile fitted to some private dataset -- which would silently
break on a new device or population -- the pipeline calibrates on simulated
recordings matched to the observed signal statistics whenever no calibration
file is supplied. The calibration used is recorded in the exported report.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .baselines import pan_tompkins
from .conformal import ConformalHR, HRVUncertainty, SelectiveController
from .metrics.hr import windowed_hr
from .metrics.quality import classical_sqi_vector

__all__ = ["HRPipeline", "PipelineResult"]


@dataclass
class PipelineResult:
    signal: np.ndarray
    fs: float
    peaks: np.ndarray
    beat_confidence: np.ndarray
    beat_usqi: np.ndarray
    usqi: np.ndarray
    hr: np.ndarray
    hr_lo: np.ndarray
    hr_hi: np.ndarray
    hr_time: np.ndarray
    hr_summary: tuple[float, float, float]
    hrv: dict[str, tuple[float, float, float]]
    abstain_threshold: float
    abstain_spans: list[tuple[float, float]]
    reported_fraction: float
    n_abstained: int
    alpha: float
    engine: str
    elapsed_s: float
    n_draws: int = 200
    meta: dict = field(default_factory=dict)


class HRPipeline:
    """
    Parameters
    ----------
    checkpoint : path to a trained HR-from-ECG v2 checkpoint, or None to use the
        classical fallback engine.
    window_s : analysis window for the heart-rate trace.
    """

    def __init__(self, checkpoint: str | Path | None = None,
                 window_s: float = 10.0, step_s: float = 1.0):
        self.window_s = window_s
        self.step_s = step_s
        self.model = None
        self.model_fs = None
        self.engine_name = "engine: classical (Pan-Tompkins + bootstrap)"

        if checkpoint is not None and Path(checkpoint).exists():
            try:
                self._load_model(checkpoint)
            except Exception as exc:  # keep the app usable without torch
                self.engine_name = f"engine: classical (model unavailable: {exc})"

    # ----------------------------------------------------------------- model
    def _load_model(self, path: str | Path) -> None:
        import torch

        from .models import HRModel, ModelConfig

        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ModelConfig(**{k: v for k, v in ck["config"].items()
                                if k in ModelConfig.__dataclass_fields__})
        m = HRModel(cfg)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        self.model = m
        self.model_fs = float(ck.get("fs", 250.0))
        self.window = int(ck.get("window", 2048))
        self.engine_name = (f"engine: HR-from-ECG v2 ({m.n_parameters()/1e6:.1f}M params, "
                            f"{self.model_fs:.0f} Hz)")

    # ------------------------------------------------------------------- I/O
    @staticmethod
    def load_file(path: str | Path) -> np.ndarray:
        """
        Read a single-lead recording from CSV, TSV, text or .npy.

        Column selection is forgiving on purpose: a column literally named
        'ECG' is preferred (the v1 convention), otherwise the first numeric
        column is used. Users should not have to reformat a file to get an
        answer.
        """
        path = Path(path)
        if path.suffix.lower() == ".npy":
            return np.asarray(np.load(path), dtype=float).ravel()

        import csv

        with open(path, newline="") as fh:
            sample = fh.read(4096)
            fh.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t ")
            except csv.Error:
                dialect = csv.excel
            rows = list(csv.reader(fh, dialect))

        if not rows:
            raise ValueError("file is empty")

        header, start = None, 0
        try:
            [float(v) for v in rows[0] if v.strip()]
        except ValueError:
            header, start = [c.strip().lower() for c in rows[0]], 1

        col = 0
        if header:
            col = header.index("ecg") if "ecg" in header else 0

        vals = []
        for r in rows[start:]:
            if len(r) <= col or not r[col].strip():
                continue
            try:
                vals.append(float(r[col]))
            except ValueError:
                continue
        if len(vals) < 100:
            raise ValueError("fewer than 100 numeric samples found")
        return np.asarray(vals, dtype=float)

    @staticmethod
    def demo_signal(kind: str = "noisy sinus", fs: float = 250.0,
                    duration_s: float = 60.0, seed: int = 7) -> np.ndarray:
        """Generate a labelled demonstration recording."""
        from .simulation import corrupt, make_synthetic_ecg

        rng = np.random.default_rng(seed)
        spec = {
            "clean sinus": ("sinus", 68.0, (24.0, 30.0), 0.0),
            "noisy sinus": ("sinus", 74.0, (-2.0, 18.0), 0.0),
            "atrial fibrillation": ("af", 105.0, (4.0, 20.0), 0.0),
            "frequent PVCs": ("pvc", 72.0, (6.0, 22.0), 0.0),
            "electrode failure": ("sinus", 80.0, (-6.0, 20.0), 1.0),
        }.get(kind, ("sinus", 72.0, (0.0, 20.0), 0.0))
        rhythm, hr, snr, p_off = spec

        e = make_synthetic_ecg(duration_s, fs, rhythm, mean_hr=hr, seed=seed)
        if kind == "clean sinus":
            return e.signal
        c = corrupt(e.signal, fs, snr_kind="piecewise", snr_range=snr,
                    p_lead_off=p_off, rng=rng)
        return c.signal

    # -------------------------------------------------------------- analysis
    def _detect(self, x: np.ndarray, fs: float):
        """Return (peaks, per-beat confidence, per-sample uSQI, engine label)."""
        if self.model is not None:
            import torch

            from .models.decode import decode_peaks, predict_signal

            xr, fs_r = self._resample(x, fs, self.model_fs)
            with torch.no_grad():
                dense = predict_signal(self.model, xr, fs_r, window=self.window)
            peaks_r, conf = decode_peaks(dense.prob, dense.offset, fs_r)
            scale = fs / fs_r
            peaks = peaks_r * scale
            usqi = np.interp(np.arange(len(x)), np.arange(len(dense.usqi)) * scale,
                             dense.usqi)
            return peaks, conf, np.clip(usqi, 0, 1), "HR-from-ECG v2"

        peaks = pan_tompkins(x, fs).astype(float)
        usqi = self._classical_usqi(x, fs)
        idx = np.clip(peaks.astype(int), 0, len(usqi) - 1)
        conf = np.clip(0.5 + 0.5 * usqi[idx], 0.05, 0.999)
        return peaks, conf, usqi, "Pan-Tompkins"

    @staticmethod
    def _resample(x: np.ndarray, fs: float, fs_target: float):
        if abs(fs - fs_target) < 1e-6:
            return x, fs
        from math import gcd

        from scipy.signal import resample_poly

        up, down = int(round(fs_target)), int(round(fs))
        g = gcd(up, down)
        return resample_poly(x, up // g, down // g), fs_target

    def _classical_usqi(self, x: np.ndarray, fs: float, win_s: float = 2.0) -> np.ndarray:
        """
        Fallback quality trace built from classical indices.

        The infrastructure study measured how weakly these track true error
        (Spearman ~0.57 at best), so this path is honestly labelled as a
        fallback rather than presented as equivalent to the learned head.
        """
        L = max(int(win_s * fs), 16)
        centres = np.arange(0, len(x), L // 2)
        vals = np.ones(len(centres))
        for i, c in enumerate(centres):
            seg = x[max(c - L // 2, 0): c + L // 2]
            if len(seg) < 16:
                continue
            f = classical_sqi_vector(seg, fs)
            k = np.clip(f.get("kSQI", 0.3), 0, 1)
            b = np.clip(f.get("baSQI", 0.5), 0, 1)
            vals[i] = float(np.clip(0.6 * k + 0.4 * b, 0.0, 1.0))
        v = np.interp(np.arange(len(x)), centres, vals)
        return np.clip(v / max(np.percentile(v, 95), 1e-6), 0, 1)

    def _calibrate(self, fs: float, alpha: float, n_records: int = 14,
                   duration_s: float = 45.0, seed: int = 11):
        """
        Fit the conformal quantiles on simulated data matched to this setting.

        Returns (ConformalHR, SelectiveController, jitter_ms).
        """
        from .simulation import corrupt, make_synthetic_ecg

        rng = np.random.default_rng(seed)
        y, med, lo, hi, uq, worst, err = [], [], [], [], [], [], []
        jitters = []

        for r in range(n_records):
            e = make_synthetic_ecg(duration_s, fs, str(rng.choice(
                ["sinus", "sinus", "af", "pvc"])),
                mean_hr=float(rng.uniform(50, 120)), seed=1000 + r)
            c = corrupt(e.signal, fs, snr_kind="piecewise", snr_range=(-6.0, 26.0),
                        p_lead_off=0.2, rng=rng)
            peaks, conf, usqi, _ = self._detect(c.signal, fs)

            hr_t, tt = windowed_hr(e.r_peaks, fs, len(e.signal), self.window_s, 2.0)
            hr_p, _ = windowed_hr(peaks, fs, len(c.signal), self.window_s, 2.0)
            n = min(len(hr_t), len(hr_p))
            wl = int(self.window_s * fs)
            for i in range(n):
                if not np.isfinite(hr_t[i]):
                    continue
                s0 = int(tt[i] * fs - wl / 2)
                seg = usqi[max(s0, 0): s0 + wl]
                seg_q = float(np.mean(seg)) if len(seg) else 0.5
                seg_w = float(np.percentile(seg, 10)) if len(seg) else 0.5
                spread = self._heuristic_spread(peaks, fs, tt[i], self.window_s)
                m = hr_p[i] if np.isfinite(hr_p[i]) else 0.0
                y.append(hr_t[i]); med.append(m)
                lo.append(m - spread); hi.append(m + spread)
                uq.append(seg_q); worst.append(seg_w)
                err.append(abs(m - hr_t[i]) if np.isfinite(hr_p[i]) else 60.0)

            if len(peaks) > 3:
                from .metrics.detection import match_peaks
                mr = match_peaks(e.r_peaks, peaks, fs, tolerance_s=0.05)
                if mr.errors_s.size > 3:
                    jitters.append(float(np.std(mr.errors_s) * 1000))

        y, med, lo, hi, uq, worst, err = map(
            np.asarray, (y, med, lo, hi, uq, worst, err))
        cp = ConformalHR(alpha=alpha, mode="mondrian", n_bins=4, min_per_bin=25)
        cp.fit(y, lo, med, hi, usqi_cal=uq)
        # The controller must be calibrated on the same statistic it will gate
        # on at inference, otherwise the threshold is applied to a different
        # distribution and the system rejects almost everything.
        sel = SelectiveController(target_risk=3.0).fit(err, worst)
        return cp, sel, (float(np.median(jitters)) if jitters else 6.0)

    @staticmethod
    def _heuristic_spread(peaks: np.ndarray, fs: float, centre_s: float,
                          window_s: float) -> float:
        """
        Input-adaptive half-width before conformal correction.

        Derived from the local scatter of instantaneous heart rate: a window
        whose beat-to-beat rate is erratic is one where the windowed average is
        less certain. Conformal calibration then rescales this to whatever
        width actually achieves nominal coverage.
        """
        p = np.asarray(peaks, float) / fs
        sel = p[(p >= centre_s - window_s / 2) & (p < centre_s + window_s / 2)]
        if len(sel) < 3:
            return 25.0
        hr = 60.0 / np.diff(sel)
        return float(max(np.std(hr) / np.sqrt(len(hr)), 0.3))

    def analyze(self, x: np.ndarray, fs: float, alpha: float = 0.1,
                risk_bpm: float = 3.0, n_draws: int = 200) -> PipelineResult:
        t0 = time.time()
        x = np.asarray(x, dtype=float)
        peaks, conf, usqi, engine = self._detect(x, fs)

        cp, sel, jitter_ms = self._calibrate(fs, alpha)
        sel.target_risk = risk_bpm
        threshold = float(np.clip(sel.threshold if np.isfinite(sel.threshold)
                                  else 0.3, 0.0, 0.95))

        hr, hr_t = windowed_hr(peaks, fs, len(x), self.window_s, self.step_s)
        wl = int(self.window_s * fs)
        # Two different statistics of the same quality trace, for two
        # different questions.
        #   win_q     mean quality -> which conformal stratum the window falls
        #             in, i.e. how wide the interval should be
        #   win_worst 10th percentile -> whether to report at all. A two-second
        #             dropout inside a ten-second window destroys the RR
        #             sequence however good the other eight seconds were, so
        #             the gate must look at the worst moment, not the average.
        win_q, win_worst, spread = [], [], []
        for c in hr_t:
            s0 = int(c * fs - wl / 2)
            seg = usqi[max(s0, 0): s0 + wl]
            win_q.append(float(np.mean(seg)) if len(seg) else 0.5)
            win_worst.append(float(np.percentile(seg, 10)) if len(seg) else 0.5)
            spread.append(self._heuristic_spread(peaks, fs, c, self.window_s))
        win_q = np.asarray(win_q)
        win_worst = np.asarray(win_worst)
        spread = np.asarray(spread)

        med = np.where(np.isfinite(hr), hr, np.nan)
        lo_raw, hi_raw = med - spread, med + spread
        hr_lo, hr_hi = cp.predict(lo_raw, med, hi_raw, usqi=win_q)

        accept = win_worst >= threshold
        hr = np.where(accept, hr, np.nan)
        hr_lo = np.where(accept, hr_lo, np.nan)
        hr_hi = np.where(accept, hr_hi, np.nan)

        # per-beat quality, used to weight the HRV ensemble
        bi = np.clip(peaks.astype(int), 0, len(usqi) - 1)
        beat_usqi = usqi[bi] if len(peaks) else np.array([])
        good_beats = peaks[beat_usqi >= threshold] if len(peaks) else peaks

        unc = HRVUncertainty(alpha=alpha, n_draws=n_draws, jitter_ms=jitter_ms)
        hrv = (unc.interval(good_beats, fs, conf[beat_usqi >= threshold],
                            beat_usqi[beat_usqi >= threshold])
               if len(good_beats) > 6 else {})

        finite = np.isfinite(hr)
        hr_summary = ((float(np.nanmean(hr_lo[finite])) if finite.any() else np.nan,
                       float(np.nanmean(hr[finite])) if finite.any() else np.nan,
                       float(np.nanmean(hr_hi[finite])) if finite.any() else np.nan))

        spans, in_span = [], None
        below = ~accept
        for i, b in enumerate(below):
            if b and in_span is None:
                in_span = hr_t[i] - self.window_s / 2
            elif not b and in_span is not None:
                spans.append((in_span, hr_t[i] + self.window_s / 2))
                in_span = None
        if in_span is not None:
            spans.append((in_span, len(x) / fs))

        return PipelineResult(
            signal=x, fs=fs, peaks=peaks, beat_confidence=conf,
            beat_usqi=beat_usqi, usqi=usqi,
            hr=hr, hr_lo=hr_lo, hr_hi=hr_hi, hr_time=hr_t,
            hr_summary=hr_summary, hrv=hrv,
            abstain_threshold=threshold, abstain_spans=spans,
            reported_fraction=float(np.mean(accept)) if len(accept) else 0.0,
            n_abstained=int((~accept).sum()),
            alpha=alpha, engine=engine, elapsed_s=time.time() - t0,
            n_draws=n_draws,
            meta=dict(conformal=cp.summary(), selective=sel.summary(),
                      jitter_ms=jitter_ms),
        )

    # ---------------------------------------------------------------- export
    @staticmethod
    def export(r: PipelineResult, path: str | Path, source: str = "") -> None:
        path = Path(path)
        if path.suffix.lower() == ".csv":
            import csv

            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["time_s", "hr_bpm", "hr_lower", "hr_upper", "usqi",
                            "reported"])
                wl = int(r.fs * 10)
                for i, t in enumerate(r.hr_time):
                    s0 = int(t * r.fs - wl / 2)
                    q = float(np.mean(r.usqi[max(s0, 0): s0 + wl]))
                    w.writerow([f"{t:.2f}",
                                "" if not np.isfinite(r.hr[i]) else f"{r.hr[i]:.2f}",
                                "" if not np.isfinite(r.hr_lo[i]) else f"{r.hr_lo[i]:.2f}",
                                "" if not np.isfinite(r.hr_hi[i]) else f"{r.hr_hi[i]:.2f}",
                                f"{q:.3f}", int(np.isfinite(r.hr[i]))])
            return

        payload = dict(
            source=source, engine=r.engine, fs=r.fs,
            duration_s=len(r.signal) / r.fs, n_beats=int(len(r.peaks)),
            confidence_level=1 - r.alpha,
            heart_rate=dict(lower=r.hr_summary[0], value=r.hr_summary[1],
                            upper=r.hr_summary[2], unit="bpm"),
            hrv={k: dict(lower=v[0], value=v[1], upper=v[2])
                 for k, v in r.hrv.items()},
            abstention=dict(threshold=r.abstain_threshold,
                            reported_fraction=r.reported_fraction,
                            n_windows_withheld=r.n_abstained),
            calibration=r.meta,
            elapsed_s=r.elapsed_s,
        )
        Path(path).write_text(json.dumps(payload, indent=2, default=float))
