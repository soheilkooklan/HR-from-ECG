"""
Tests for the model, the conformal layer and the pipeline.

The conformal tests are the important ones. A coverage guarantee that is not
tested is a coverage claim, and the failure mode is silent: an interval that
under-covers still looks like an interval.
"""

from __future__ import annotations

import numpy as np
import pytest

from hrecg.conformal import (ConformalHR, HRVUncertainty, SelectiveController,
                             conformal_quantile, hrv_ensemble)
from hrecg.metrics.uncertainty import conditional_coverage, interval_metrics
from hrecg.simulation import make_synthetic_ecg

torch = pytest.importorskip("torch")

from hrecg.models import HRModel, ModelConfig, HRLoss  # noqa: E402
from hrecg.models.decode import decode_peaks, predict_signal  # noqa: E402
from hrecg.models.ssm import associative_scan  # noqa: E402


# --- selective state space -------------------------------------------------
def test_associative_scan_matches_sequential_recurrence():
    torch.manual_seed(0)
    a = torch.rand(3, 128, 5)
    b = torch.randn(3, 128, 5)
    got = associative_scan(a, b)
    ref = torch.zeros_like(b)
    h = torch.zeros(3, 5)
    for t in range(128):
        h = a[:, t] * h + b[:, t]
        ref[:, t] = h
    assert torch.allclose(got, ref, atol=1e-5)


def test_associative_scan_is_differentiable():
    a = torch.rand(2, 32, 3, requires_grad=True)
    b = torch.randn(2, 32, 3, requires_grad=True)
    associative_scan(a, b).sum().backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()


def test_scan_handles_non_power_of_two_lengths():
    for L in (1, 7, 33, 100):
        a = torch.rand(1, L, 2)
        b = torch.randn(1, L, 2)
        got = associative_scan(a, b)
        h = torch.zeros(1, 2)
        for t in range(L):
            h = a[:, t] * h + b[:, t]
        assert torch.allclose(got[:, -1], h, atol=1e-5)


# --- network ---------------------------------------------------------------
def _small_net():
    return HRModel(ModelConfig(base_width=8, depth=3, expand=1, n_ssm_blocks=1))


def test_forward_shapes_and_ranges():
    m = _small_net()
    o = m(torch.randn(2, 1, 512))
    assert o.beat_logit.shape == (2, 512)
    assert o.usqi.shape == (2, 512)
    assert (o.usqi > 0).all() and (o.usqi < 1).all()
    assert (o.offset.abs() <= 1).all()


def test_hr_quantiles_are_monotone_by_construction():
    m = _small_net()
    o = m(torch.randn(4, 1, 512))
    assert (o.hr_quantiles.diff(dim=1) >= 0).all()


def test_gradients_reach_every_head():
    m = _small_net()
    o = m(torch.randn(2, 1, 512))
    (o.beat_logit.mean() + o.offset.mean() + o.usqi.mean()
     + o.hr_quantiles.mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in m.parameters() if p.requires_grad)


def test_rhythm_prior_prefers_periodic_maps():
    from hrecg.models.losses import rhythm_prior_loss

    fs, L = 250.0, 2048
    periodic = torch.zeros(1, L)
    for i in range(0, L, 200):
        periodic[0, i] = 1.0
    random = torch.zeros(1, L)
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(L, generator=g)[: L // 200]
    random[0, idx] = 1.0
    assert rhythm_prior_loss(periodic, fs) < rhythm_prior_loss(random, fs)


def test_loss_returns_finite_components():
    m = _small_net()
    x = torch.randn(2, 1, 512)
    batch = dict(beat=torch.rand(2, 512), offset=torch.rand(2, 512) * 2 - 1,
                 beat_mask=(torch.rand(2, 512) > 0.9).float(),
                 usqi=torch.rand(2, 512), hr=torch.tensor([70.0, 85.0]))
    total, parts = HRLoss(fs=250.0)(m(x), batch)
    assert torch.isfinite(total)
    assert all(np.isfinite(v) for v in parts.values())


# --- decoding --------------------------------------------------------------
def test_decode_respects_the_refractory_period():
    prob = np.zeros(1000)
    prob[[100, 105, 400]] = 1.0   # two candidates 20 ms apart at 250 Hz
    peaks, _ = decode_peaks(prob, None, fs=250.0, refractory_s=0.2)
    assert len(peaks) == 2


def test_subsample_offset_is_applied():
    prob = np.zeros(500)
    prob[200] = 1.0
    off = np.zeros(500)
    off[200] = 0.4
    peaks, _ = decode_peaks(prob, off, fs=250.0)
    assert abs(peaks[0] - 200.4) < 1e-6


def test_predict_signal_covers_the_whole_record():
    m = _small_net()
    x = np.random.randn(3000).astype(np.float32)
    d = predict_signal(m, x, 250.0, window=512)
    assert len(d.prob) == len(x) == len(d.usqi) == len(d.offset)
    assert np.isfinite(d.prob).all()


# --- conformal -------------------------------------------------------------
def test_conformal_quantile_uses_the_finite_sample_correction():
    s = np.arange(1, 11, dtype=float)      # n = 10
    # ceil(11 * 0.9) = 10 -> index 9 -> the largest score
    assert conformal_quantile(s, 0.1) == 10.0
    # too few points for the requested level: honest infinity, not a guess
    assert conformal_quantile(np.array([1.0, 2.0]), 0.1) == float("inf")


def test_split_conformal_achieves_nominal_marginal_coverage():
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.normal(70, 10, n)
    med = y + rng.normal(0, 3, n)
    lo, hi = med - 3, med + 3
    cp = ConformalHR(alpha=0.1, mode="split").fit(y[:2000], lo[:2000],
                                                  med[:2000], hi[:2000])
    l, h = cp.predict(lo[2000:], med[2000:], hi[2000:])
    picp = interval_metrics(y[2000:], l, h, alpha=0.1).picp
    assert 0.86 < picp < 0.94


def test_mondrian_improves_worst_stratum_coverage():
    """
    Heteroscedastic setup: error scale depends on quality. Split conformal is
    marginally valid but must under-cover the low-quality stratum; Mondrian
    should not.
    """
    rng = np.random.default_rng(1)
    n = 8000
    q = rng.uniform(0, 1, n)
    scale = 1.0 + 12.0 * (1 - q) ** 2
    y = rng.normal(70, 8, n)
    med = y + rng.normal(0, 1, n) * scale
    lo, hi = med - 1, med + 1
    cal, tst = slice(0, n // 2), slice(n // 2, n)

    worst = {}
    for mode in ("split", "mondrian"):
        cp = ConformalHR(alpha=0.1, mode=mode, n_bins=5, min_per_bin=30)
        cp.fit(y[cal], lo[cal], med[cal], hi[cal], usqi_cal=q[cal])
        l, h = cp.predict(lo[tst], med[tst], hi[tst], usqi=q[tst])
        worst[mode] = conditional_coverage(y[tst], l, h, q[tst],
                                           n_bins=5)["worst_slab_coverage"]
    assert worst["mondrian"] > worst["split"]


def test_selective_controller_reduces_risk():
    rng = np.random.default_rng(2)
    q = rng.uniform(0, 1, 3000)
    err = (1 - q) * 20 + rng.exponential(0.5, 3000)
    sc = SelectiveController(target_risk=3.0).fit(err, q)
    keep = sc.accept(q)
    assert keep.sum() > 0
    assert err[keep].mean() < err.mean()


def test_hrv_ensemble_widens_when_confidence_drops():
    e = make_synthetic_ecg(120, 250.0, "sinus", 72, seed=3)
    p = e.r_peaks.astype(float)
    rng = np.random.default_rng(0)
    sure = hrv_ensemble(p, 250.0, np.full(len(p), 0.99), n_draws=60, rng=rng)
    unsure = hrv_ensemble(p, 250.0, np.full(len(p), 0.75), n_draws=60, rng=rng)
    assert np.nanstd(unsure["rmssd_ms"]) > np.nanstd(sure["rmssd_ms"])


def test_hrv_intervals_respect_physiological_bounds():
    e = make_synthetic_ecg(120, 250.0, "sinus", 72, seed=4)
    p = e.r_peaks.astype(float)
    unc = HRVUncertainty(alpha=0.1, n_draws=60)
    iv = unc.interval(p, 250.0, np.full(len(p), 0.85))
    for key in ("sdnn_ms", "rmssd_ms", "sd1_ms", "pnn50_pct"):
        assert iv[key][0] >= 0.0
    assert iv["pnn50_pct"][2] <= 100.0


# --- pipeline --------------------------------------------------------------
def test_pipeline_runs_without_a_checkpoint():
    from hrecg.pipeline import HRPipeline

    p = HRPipeline(checkpoint=None)
    x = p.demo_signal("clean sinus", 250.0, 30.0)
    r = p.analyze(x, 250.0, alpha=0.1)
    assert len(r.peaks) > 20
    assert np.isfinite(r.hr_summary[1])
    assert r.hr_summary[0] <= r.hr_summary[1] <= r.hr_summary[2]
    assert 0.0 <= r.reported_fraction <= 1.0
