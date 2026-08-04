"""Unit tests for the HR from ECG infrastructure layer. Run with: pytest -q"""

from __future__ import annotations

import numpy as np

from hrecg.baselines import pan_tompkins
from hrecg.metrics.detection import match_peaks
from hrecg.metrics.hr import hr_agreement, icc21, windowed_hr
from hrecg.metrics.hrv import clean_rr, hrv_features, poincare
from hrecg.metrics.quality import utility_sqi
from hrecg.metrics.uncertainty import aurc, interval_metrics, risk_coverage_curve
from hrecg.simulation import corrupt, make_synthetic_ecg, mix_at_snr
from hrecg.simulation.noise import muscle_artifact

FS = 500.0


# --- simulator -------------------------------------------------------------
def test_r_peaks_are_local_maxima():
    e = make_synthetic_ecg(30, FS, "sinus", 72, seed=0)
    for p in e.r_peaks[2:-2]:
        w = e.signal[p - 40:p + 40]
        assert abs(int(np.argmax(w)) - 40) <= 1


def test_target_heart_rate_is_recovered():
    for hr in (50.0, 72.0, 110.0):
        e = make_synthetic_ecg(180, FS, "sinus", hr, seed=1)
        assert abs(60.0 / np.mean(e.rr) - hr) < 1.0


def test_af_has_poincare_ratio_near_one():
    e_af = make_synthetic_ecg(300, FS, "af", 90, seed=2)
    e_sr = make_synthetic_ecg(300, FS, "sinus", 90, seed=2)
    assert poincare(e_af.rr)["sd1_sd2"] > 0.6
    assert poincare(e_sr.rr)["sd1_sd2"] < 0.6


def test_pvc_pause_is_compensatory():
    e = make_synthetic_ecg(300, FS, "pvc", 72, seed=3)
    v = [i for i, b in enumerate(e.beat_types) if b == "V"]
    ratios = [(e.rr[i - 1] + e.rr[i]) / (2 * np.median(e.rr))
              for i in v if 0 < i < len(e.rr)]
    assert ratios and abs(float(np.median(ratios)) - 1.0) < 0.15


# --- noise -----------------------------------------------------------------
def test_snr_is_calibrated():
    e = make_synthetic_ecg(60, FS, "sinus", 72, seed=4)
    for target in (-6.0, 0.0, 12.0, 24.0):
        n = muscle_artifact(len(e.signal), FS, rng=np.random.default_rng(0))
        y, _ = mix_at_snr(e.signal, n, target, FS, local=False)
        achieved = 10 * np.log10(np.mean(e.signal**2) / np.mean((y - e.signal) ** 2))
        assert abs(achieved - target) < 0.1


def test_corrupt_preserves_length_and_reports_snr():
    e = make_synthetic_ecg(30, FS, "sinus", 72, seed=5)
    c = corrupt(e.signal, FS, rng=np.random.default_rng(5))
    assert len(c.signal) == len(e.signal) == len(c.snr_db)


# --- detection metrics -----------------------------------------------------
def test_perfect_match():
    ref = np.arange(0, 100_000, 500)
    r = match_peaks(ref, ref.copy(), FS, tolerance_s=0.05)
    assert r.tp == len(ref) and r.fp == 0 and r.fn == 0 and r.f1 == 1.0


def test_matching_is_one_to_one():
    ref = np.array([1000, 1500, 2000], float)
    det = np.array([1000, 1005, 1010, 1500, 2000], float)  # three near the first
    r = match_peaks(ref, det, FS, tolerance_s=0.05)
    assert r.tp == 3 and r.fp == 2 and r.fn == 0


def test_shifted_detections_fail_strict_tolerance():
    ref = np.arange(0, 50_000, 500).astype(float)
    det = ref + 0.04 * FS  # 40 ms late
    assert match_peaks(ref, det, FS, tolerance_s=0.05).f1 == 1.0
    assert match_peaks(ref, det, FS, tolerance_s=0.025).tp == 0


# --- baseline detector -----------------------------------------------------
def test_pan_tompkins_is_exact_on_clean_signal():
    e = make_synthetic_ecg(120, FS, "sinus", 72, seed=6)
    r = match_peaks(e.r_peaks, pan_tompkins(e.signal, FS), FS, tolerance_s=0.025)
    assert r.f1 > 0.99 and abs(float(np.mean(r.errors_s))) < 0.005


def test_pan_tompkins_degrades_monotonically_with_noise():
    e = make_synthetic_ecg(120, FS, "sinus", 72, seed=7)
    f1 = []
    for snr in (-6, 0, 6, 18):
        c = corrupt(e.signal, FS, snr_kind="constant", snr_range=(snr, snr),
                    p_lead_off=0.0, rng=np.random.default_rng(7))
        f1.append(match_peaks(e.r_peaks, pan_tompkins(c.signal, FS), FS).f1)
    assert f1[0] < f1[-1] and f1[-1] > 0.99


# --- HR / HRV --------------------------------------------------------------
def test_windowed_hr_matches_truth_on_clean_signal():
    e = make_synthetic_ecg(120, FS, "sinus", 72, seed=8)
    hr_t, _ = windowed_hr(e.r_peaks, FS, len(e.signal), 10.0, 1.0)
    hr_p, _ = windowed_hr(pan_tompkins(e.signal, FS), FS, len(e.signal), 10.0, 1.0)
    assert hr_agreement(hr_t, hr_p).mae < 0.5


def test_icc_is_one_for_identical_series():
    x = np.random.default_rng(0).normal(70, 8, 200)
    assert icc21(x, x) > 0.999


def test_lf_hf_ratio_is_recovered():
    e = make_synthetic_ecg(600, FS, "sinus", 70, sdnn_ms=50,
                           lf_hf_ratio=2.5, seed=9)
    f = hrv_features(e.r_peaks, FS, correction="none")
    assert 1.6 < f["lf_hf_ratio"] < 3.6


def test_ectopic_correction_reduces_rmssd():
    e = make_synthetic_ecg(300, FS, "pvc", 72, seed=10)
    raw = hrv_features(e.r_peaks, FS, correction="none")["rmssd_ms"]
    fixed = hrv_features(e.r_peaks, FS, correction="malik")["rmssd_ms"]
    assert fixed < raw * 0.6


def test_clean_rr_flags_outliers():
    rr = np.full(50, 0.85)
    rr[20] = 0.40
    _, _, flagged = clean_rr(rr, method="malik")
    assert flagged[20]


# --- uncertainty -----------------------------------------------------------
def test_coverage_of_a_correct_interval():
    rng = np.random.default_rng(0)
    y = rng.normal(70, 10, 5000)
    m = interval_metrics(y, y - 1.645 * 10 / 10, y + 1.645 * 10 / 10, alpha=0.1)
    assert m.picp == 1.0  # intervals centred on truth always cover


def test_undercoverage_is_detected():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 20_000)
    m = interval_metrics(y, np.full_like(y, -1.0), np.full_like(y, 1.0), alpha=0.1)
    assert m.coverage_gap < -0.10  # true coverage ~68% against a 90% target


def test_risk_coverage_curve_is_increasing_for_a_good_score():
    rng = np.random.default_rng(2)
    err = rng.exponential(1.0, 2000)
    conf = -err + 0.05 * rng.standard_normal(2000)  # informative confidence
    cov, risk = risk_coverage_curve(err, conf)
    assert risk[0] < risk[-1]
    assert aurc(err, conf) < aurc(err, rng.standard_normal(2000))


def test_usqi_is_one_when_error_is_zero():
    u = utility_sqi(np.array([70.0, 70.0]), np.array([70.0, 80.0]), tau_bpm=5.0)
    assert abs(u[0] - 1.0) < 1e-9 and u[1] < 0.2
