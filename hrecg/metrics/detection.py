"""
Beat-detection metrics.

Implements one-to-one matching between detected and reference R-peaks within a
tolerance window, following the ANSI/AAMI EC57 convention. Two tolerances are
reported throughout the paper:

    150 ms  -- the AAMI standard window (very permissive)
     50 ms  -- the window used by most recent deep-learning papers
     25 ms  -- a strict window that actually discriminates between methods
               once sensitivity has saturated above 99.5%

Localisation error is reported separately, because a detector can reach an
F1 of 0.998 while still being useless for HRV if its jitter is 20 ms: an RMSSD
of 25 ms would then be dominated by detector noise rather than physiology.
That decoupling is central to the argument of this work.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["match_peaks", "DetectionResult", "detection_metrics"]


@dataclass
class DetectionResult:
    tp: int
    fp: int
    fn: int
    matched_ref: np.ndarray      # indices into the reference array
    matched_det: np.ndarray      # indices into the detection array
    errors_s: np.ndarray         # signed localisation error (det - ref), seconds
    tolerance_s: float

    @property
    def sensitivity(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else float("nan")

    @property
    def ppv(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else float("nan")

    @property
    def f1(self) -> float:
        se, pp = self.sensitivity, self.ppv
        return 2 * se * pp / (se + pp) if (se + pp) > 0 else float("nan")

    @property
    def der(self) -> float:
        """Detection Error Rate = (FP + FN) / n_reference."""
        denom = self.tp + self.fn
        return (self.fp + self.fn) / denom if denom else float("nan")

    def as_dict(self) -> dict:
        e = self.errors_s * 1000.0
        return dict(
            TP=self.tp, FP=self.fp, FN=self.fn,
            Se=self.sensitivity, PPV=self.ppv, F1=self.f1, DER=self.der,
            loc_bias_ms=float(np.mean(e)) if e.size else float("nan"),
            loc_sd_ms=float(np.std(e)) if e.size else float("nan"),
            loc_mae_ms=float(np.mean(np.abs(e))) if e.size else float("nan"),
            tolerance_ms=self.tolerance_s * 1000.0,
        )


def match_peaks(
    ref: np.ndarray,
    det: np.ndarray,
    fs: float,
    tolerance_s: float = 0.05,
    exclude_edges_s: float = 0.0,
    n_samples: int | None = None,
) -> DetectionResult:
    """
    Greedy globally-ordered matching of detections to references.

    Candidate pairs within the tolerance are sorted by absolute error and
    accepted greedily, each reference and detection being used at most once.
    For monotone peak trains with a tolerance smaller than half the shortest
    RR interval -- always true here -- this is provably equivalent to the
    optimal assignment, while running in O(n log n) instead of O(n^3).

    Parameters
    ----------
    ref, det : R-peak locations in *samples*.
    exclude_edges_s : drop beats within this margin of the record boundaries,
        where windowed detectors have no valid context.
    """
    ref = np.sort(np.asarray(ref, dtype=float))
    det = np.sort(np.asarray(det, dtype=float))

    if exclude_edges_s > 0 and n_samples is not None:
        m = exclude_edges_s * fs
        ref = ref[(ref >= m) & (ref <= n_samples - m)]
        det = det[(det >= m) & (det <= n_samples - m)]

    tol = tolerance_s * fs
    pairs: list[tuple[float, int, int]] = []
    if len(ref) and len(det):
        j0 = 0
        for i, r in enumerate(ref):
            while j0 < len(det) and det[j0] < r - tol:
                j0 += 1
            j = j0
            while j < len(det) and det[j] <= r + tol:
                pairs.append((abs(det[j] - r), i, j))
                j += 1

    pairs.sort()
    used_ref = np.zeros(len(ref), dtype=bool)
    used_det = np.zeros(len(det), dtype=bool)
    mi, mj, errs = [], [], []
    for _, i, j in pairs:
        if used_ref[i] or used_det[j]:
            continue
        used_ref[i] = used_det[j] = True
        mi.append(i)
        mj.append(j)
        errs.append((det[j] - ref[i]) / fs)

    order = np.argsort(mi)
    return DetectionResult(
        tp=int(used_ref.sum()),
        fp=int((~used_det).sum()),
        fn=int((~used_ref).sum()),
        matched_ref=np.asarray(mi, dtype=int)[order],
        matched_det=np.asarray(mj, dtype=int)[order],
        errors_s=np.asarray(errs, dtype=float)[order],
        tolerance_s=tolerance_s,
    )


def detection_metrics(
    ref: np.ndarray,
    det: np.ndarray,
    fs: float,
    tolerances_s: tuple[float, ...] = (0.15, 0.05, 0.025),
    **kwargs,
) -> dict:
    """Evaluate at several tolerance windows at once; keys are suffixed by ms."""
    out: dict = {}
    for tol in tolerances_s:
        d = match_peaks(ref, det, fs, tolerance_s=tol, **kwargs).as_dict()
        suffix = f"@{int(round(tol * 1000))}ms"
        out.update({f"{k}{suffix}": v for k, v in d.items() if k != "tolerance_ms"})
    return out
