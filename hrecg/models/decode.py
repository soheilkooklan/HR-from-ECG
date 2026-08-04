"""
Decoding: turning the network's dense outputs into beats, heart rate and quality.

Two details do most of the work here.

**Sub-sample refinement.** The offset head predicts, for every sample, how far
the true beat instant lies from that sample's centre. Applying it turns a
detector whose resolution is bounded by the sampling grid into one whose
resolution is bounded only by the network's accuracy. At 250 Hz the grid alone
costs +-2 ms of quantisation jitter, which is comparable to the entire
localisation error budget of a good detector.

**Overlap-add inference with a Hann taper.** Long records are processed in
overlapping windows; samples near a window edge have seen only half their
context, so they are down-weighted rather than trusted equally. Without this,
detection quality oscillates with a period equal to the window stride, which is
easy to miss and produces a strange comb pattern in the RR series.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .net import HRModel, ModelOutput

__all__ = ["decode_peaks", "predict_signal", "DenseOutput"]


@dataclass
class DenseOutput:
    """Per-sample model outputs for a whole record."""

    prob: np.ndarray
    offset: np.ndarray
    usqi: np.ndarray
    fs: float


def decode_peaks(
    prob: np.ndarray,
    offset: np.ndarray | None,
    fs: float,
    threshold: float = 0.5,
    refractory_s: float = 0.20,
    min_prominence: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract beat positions from a probability map.

    Non-maximum suppression uses a physiological refractory period rather than
    a tuned constant: no two ventricular depolarisations can occur within
    ~200 ms, so any second candidate inside that interval is by definition
    spurious.

    Returns
    -------
    peaks_samples : float array of sub-sample beat positions.
    confidences   : the probability value at each accepted peak.
    """
    from scipy.signal import find_peaks

    dist = max(int(refractory_s * fs), 1)
    idx, props = find_peaks(prob, height=threshold, distance=dist,
                            prominence=min_prominence)
    if len(idx) == 0:
        return np.array([]), np.array([])

    pos = idx.astype(float)
    if offset is not None:
        pos = pos + np.clip(offset[idx], -1.0, 1.0)
    return pos, props["peak_heights"]


@torch.no_grad()
def predict_signal(
    model: HRModel,
    x: np.ndarray,
    fs: float,
    window: int = 2048,
    overlap: float = 0.5,
    device: str = "cpu",
    batch_size: int = 8,
    normalise: bool = True,
) -> DenseOutput:
    """
    Run the model over an arbitrarily long single-lead record.

    Normalisation is per window and robust (median / IQR), so the model is
    invariant to lead gain and to the slow amplitude drift that respiration
    and electrode impedance impose.
    """
    model.eval().to(device)
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    step = max(int(window * (1 - overlap)), 1)

    if n < window:
        x = np.pad(x, (0, window - n))
    starts = list(range(0, max(len(x) - window, 0) + 1, step))
    if starts[-1] + window < len(x):
        starts.append(len(x) - window)

    taper = np.hanning(window).astype(np.float32) + 1e-3
    acc_p = np.zeros(len(x), dtype=np.float64)
    acc_o = np.zeros(len(x), dtype=np.float64)
    acc_q = np.zeros(len(x), dtype=np.float64)
    acc_w = np.zeros(len(x), dtype=np.float64)

    for i in range(0, len(starts), batch_size):
        chunk = starts[i:i + batch_size]
        seg = np.stack([x[s:s + window] for s in chunk])
        if normalise:
            med = np.median(seg, axis=1, keepdims=True)
            iqr = np.subtract(*np.percentile(seg, [75, 25], axis=1)).reshape(-1, 1)
            seg = (seg - med) / np.maximum(iqr, 1e-3)
        t = torch.from_numpy(seg.astype(np.float32)).unsqueeze(1).to(device)
        out: ModelOutput = model(t)
        p = torch.sigmoid(out.beat_logit).cpu().numpy()
        o = out.offset.cpu().numpy()
        q = out.usqi.cpu().numpy()
        for j, s in enumerate(chunk):
            acc_p[s:s + window] += p[j] * taper
            acc_o[s:s + window] += o[j] * taper
            acc_q[s:s + window] += q[j] * taper
            acc_w[s:s + window] += taper

    acc_w = np.maximum(acc_w, 1e-6)
    return DenseOutput(
        prob=(acc_p / acc_w)[:n].astype(np.float32),
        offset=(acc_o / acc_w)[:n].astype(np.float32),
        usqi=(acc_q / acc_w)[:n].astype(np.float32),
        fs=fs,
    )
