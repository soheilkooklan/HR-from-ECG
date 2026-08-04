"""
Uncertainty propagation from beats to HRV indices.

HRV is reported everywhere as a point value. That convention is hard to defend:
RMSSD is a function of *successive differences*, so a single spurious or missed
beat splits one interval into two, or merges two into one, and the resulting
squared difference enters the mean at full weight. A 5-minute record at 70 bpm
contains roughly 350 intervals; one bad beat can move RMSSD by several
milliseconds, which is the same order as the effect sizes routinely reported as
significant in the autonomic literature.

This module replaces the point value with an interval, by propagating what the
detector actually knows about its own beats:

    1. Each detected beat carries a probability p_i and a local quality u_i.
    2. Draw an ensemble of plausible beat sets. In each draw, a beat is kept
       with probability p_i, and its position is jittered by a Gaussian whose
       width is inflated where quality is low.
    3. Recompute every HRV index on each draw.
    4. Report the empirical percentile interval across draws.

Step 2 is where the modelling assumption sits, and it is deliberately simple:
the ensemble reflects detector uncertainty, not physiological uncertainty, and
the resulting intervals are therefore statements about measurement error. They
are then *conformalised* against ground truth on a calibration set, so that the
final intervals carry an empirical coverage guarantee rather than resting on
the Bernoulli/Gaussian assumption being literally true.

That last step matters: a naive Monte-Carlo interval is only as good as its
noise model, whereas a conformalised one is corrected by however much the noise
model was wrong on held-out data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..metrics.hrv import hrv_features
from .hr_interval import conformal_quantile

__all__ = ["HRVUncertainty", "hrv_ensemble"]

DEFAULT_FEATURES = ("mean_hr_bpm", "sdnn_ms", "rmssd_ms", "pnn50_pct",
                    "sd1_ms", "sd2_ms", "lf_hf_ratio")

# Physiological support of each index. Conformal widening is symmetric, so an
# index that is non-negative by definition can otherwise be handed a negative
# lower bound -- valid as a coverage statement, nonsense as a measurement.
FEATURE_BOUNDS = {
    "mean_hr_bpm": (10.0, 300.0), "sdnn_ms": (0.0, None), "rmssd_ms": (0.0, None),
    "pnn50_pct": (0.0, 100.0), "sd1_ms": (0.0, None), "sd2_ms": (0.0, None),
    "lf_hf_ratio": (0.0, None),
}


def hrv_ensemble(
    peaks: np.ndarray,
    fs: float,
    confidence: np.ndarray,
    usqi: np.ndarray | None = None,
    n_draws: int = 200,
    jitter_ms: float = 4.0,
    features: tuple[str, ...] = DEFAULT_FEATURES,
    correction: str = "malik",
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """
    Draw an ensemble of HRV feature vectors consistent with detector uncertainty.

    Parameters
    ----------
    confidence : per-beat detection probability, in (0, 1].
    usqi : per-beat quality; positions in low-quality regions are jittered more.
    jitter_ms : localisation standard deviation at unit quality. Set this from
        the measured localisation error of the detector, not by guesswork.

    Returns a dict mapping feature name to an array of length n_draws.
    """
    rng = np.random.default_rng() if rng is None else rng
    peaks = np.asarray(peaks, float)
    conf = np.clip(np.asarray(confidence, float), 1e-3, 1.0)
    if usqi is None:
        usqi = np.ones_like(peaks)
    usqi = np.clip(np.asarray(usqi, float), 1e-3, 1.0)

    sigma = jitter_ms / 1000.0 * fs / usqi   # inflate jitter where quality is low
    out = {f: np.full(n_draws, np.nan) for f in features}

    for d in range(n_draws):
        keep = rng.random(len(peaks)) < conf
        if keep.sum() < 5:
            continue
        p = peaks[keep] + rng.normal(0.0, sigma[keep])
        p = np.sort(p)
        f = hrv_features(p, fs, correction=correction, include_frequency=True)
        for k in features:
            out[k][d] = f.get(k, np.nan)
    return out


@dataclass
class HRVUncertainty:
    """
    Conformalised Monte-Carlo intervals for HRV indices.

    Usage
    -----
        unc = HRVUncertainty(alpha=0.1).fit(cal_records)
        lo, hi = unc.interval(peaks, fs, conf, usqi)["rmssd_ms"]

    `fit` expects a list of calibration items, each a dict with keys
    ``peaks``, ``confidence``, ``usqi``, ``fs`` and ``truth`` (the reference
    HRV feature dict computed from expert annotations). It learns, per feature,
    a multiplicative widening factor that brings empirical coverage up to the
    nominal level -- the conformal correction applied to an ensemble interval.
    """

    alpha: float = 0.1
    n_draws: int = 200
    jitter_ms: float = 4.0
    features: tuple[str, ...] = DEFAULT_FEATURES
    scale: dict = field(default_factory=dict, init=False)
    fitted: bool = field(default=False, init=False)

    def _raw_interval(self, ens: dict[str, np.ndarray]) -> dict[str, tuple[float, float, float]]:
        out = {}
        for k, v in ens.items():
            v = v[np.isfinite(v)]
            if len(v) < 5:
                out[k] = (np.nan, np.nan, np.nan)
                continue
            lo, med, hi = np.percentile(v, [100 * self.alpha / 2, 50,
                                            100 * (1 - self.alpha / 2)])
            out[k] = (float(lo), float(med), float(hi))
        return out

    def fit(self, cal_items: list[dict], rng: np.random.Generator | None = None
            ) -> "HRVUncertainty":
        rng = np.random.default_rng(0) if rng is None else rng
        scores = {f: [] for f in self.features}

        for it in cal_items:
            ens = hrv_ensemble(it["peaks"], it["fs"], it["confidence"],
                               it.get("usqi"), n_draws=self.n_draws,
                               jitter_ms=self.jitter_ms, features=self.features,
                               rng=rng)
            raw = self._raw_interval(ens)
            for f in self.features:
                lo, med, hi = raw[f]
                y = it["truth"].get(f, np.nan)
                half = max((hi - lo) / 2.0, 1e-9)
                if np.isfinite(y) and np.isfinite(med):
                    # nonconformity: how many raw half-widths away the truth is
                    scores[f].append(abs(y - med) / half)

        for f in self.features:
            s = np.asarray(scores[f], float)
            q = conformal_quantile(s, self.alpha) if len(s) else 1.0
            # never shrink below the ensemble interval; widening only
            self.scale[f] = float(max(q, 1.0)) if np.isfinite(q) else 3.0
        self.fitted = True
        return self

    def interval(
        self,
        peaks: np.ndarray,
        fs: float,
        confidence: np.ndarray,
        usqi: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> dict[str, tuple[float, float, float]]:
        """
        Return {feature: (lower, point, upper)}.

        If `fit` has not been called the raw ensemble percentiles are returned,
        which are honest but carry no coverage guarantee.
        """
        ens = hrv_ensemble(peaks, fs, confidence, usqi, n_draws=self.n_draws,
                           jitter_ms=self.jitter_ms, features=self.features, rng=rng)
        raw = self._raw_interval(ens)
        out = {}
        for f, (lo, med, hi) in raw.items():
            if not np.isfinite(med):
                out[f] = (np.nan, np.nan, np.nan)
                continue
            k = self.scale.get(f, 1.0) if self.fitted else 1.0
            half = (hi - lo) / 2.0 * k
            lo_b, hi_b = FEATURE_BOUNDS.get(f, (None, None))
            l, u = med - half, med + half
            if lo_b is not None:
                l = max(l, lo_b)
            if hi_b is not None:
                u = min(u, hi_b)
            out[f] = (l, med, u)
        return out

    def summary(self) -> dict:
        return dict(alpha=self.alpha, n_draws=self.n_draws,
                    jitter_ms=self.jitter_ms, scale=dict(self.scale),
                    fitted=self.fitted)
