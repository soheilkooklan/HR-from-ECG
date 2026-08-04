"""
Benchmark on the MIT-BIH Arrhythmia Database with real NSTDB noise.

Experiments
-----------
R1  Beat detection on the records as recorded, at three tolerance windows.
R2  Noise stress using the *real* electrode-motion, muscle-artifact and
    baseline-wander records from NSTDB, added at controlled SNR.
R3  Conformal heart-rate intervals: marginal coverage, interval width, Winkler
    score, and -- the result that matters -- coverage *conditional* on the
    predicted quality stratum, comparing split against Mondrian conformal.
R4  Selective prediction: how far does abstaining on the worst windows push
    down the error on the rest, and does the learned quality score rank windows
    better than the classical indices.
R5  HRV interval coverage on five-minute segments.

Protocol notes
--------------
*   The four paced records (102, 104, 107, 217) are excluded, following the
    AAMI EC57 recommendation; paced beats are a different detection problem and
    including them inflates or deflates results depending on the detector's
    handling of pacing spikes.
*   Records are split into disjoint calibration and test halves **by record**,
    never by window. Splitting by window would leak the same patient's
    morphology into both sides and inflate the coverage result, which is the
    single most common way conformal evaluations are quietly broken.
*   Lead MLII is used where present, otherwise the first available lead.

Usage
-----
    python scripts/evaluate_mitdb.py --data <dir> --nstdb <dir> --out results/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hrecg.baselines import pan_tompkins
from hrecg.conformal import ConformalHR, HRVUncertainty, SelectiveController
from hrecg.data.physionet import list_records, load_record
from hrecg.metrics.detection import match_peaks
from hrecg.metrics.hr import hr_agreement, windowed_hr
from hrecg.metrics.hrv import hrv_features
from hrecg.metrics.quality import classical_sqi_vector
from hrecg.metrics.uncertainty import (aurc, conditional_coverage,
                                       interval_metrics, risk_coverage_curve)
from hrecg.simulation.noise import mix_at_snr

PACED = {"102", "104", "107", "217"}
FS = 250.0
WINDOW_S = 10.0
STEP_S = 2.0


# --------------------------------------------------------------------------- #
def load_engine(checkpoint: str | None):
    """Return (detect_fn, name). detect_fn(x, fs) -> (peaks, conf, usqi)."""
    if checkpoint and Path(checkpoint).exists():
        import torch

        from hrecg.models import HRModel, ModelConfig
        from hrecg.models.decode import decode_peaks, predict_signal

        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = ModelConfig(**{k: v for k, v in ck["config"].items()
                                if k in ModelConfig.__dataclass_fields__})
        model = HRModel(cfg)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        win = int(ck.get("window", 2048))

        def detect(x, fs):
            with torch.no_grad():
                d = predict_signal(model, x, fs, window=win)
            peaks, conf = decode_peaks(d.prob, d.offset, fs)
            return peaks, conf, d.usqi

        return detect, f"HR-from-ECG v2 ({model.n_parameters()/1e6:.2f}M)"

    def detect(x, fs):
        peaks = pan_tompkins(x, fs).astype(float)
        u = classical_quality_trace(x, fs)
        idx = np.clip(peaks.astype(int), 0, len(u) - 1)
        return peaks, np.clip(0.5 + 0.5 * u[idx], 0.05, 0.999), u

    return detect, "Pan-Tompkins"


def classical_quality_trace(x: np.ndarray, fs: float, win_s: float = 2.0) -> np.ndarray:
    L = max(int(win_s * fs), 16)
    centres = np.arange(0, len(x), L // 2)
    vals = np.ones(len(centres))
    for i, c in enumerate(centres):
        seg = x[max(c - L // 2, 0): c + L // 2]
        if len(seg) < 16:
            continue
        f = classical_sqi_vector(seg, fs)
        vals[i] = float(np.clip(0.6 * np.clip(f["kSQI"], 0, 1)
                                + 0.4 * np.clip(f["baSQI"], 0, 1), 0, 1))
    return np.clip(np.interp(np.arange(len(x)), centres, vals), 0, 1)


def window_features(x, fs, peaks, usqi, ref_peaks, n):
    """Per-window (hr_true, hr_pred, usqi, ksqi, spread) for the conformal stage."""
    hr_t, tt = windowed_hr(ref_peaks, fs, n, WINDOW_S, STEP_S)
    hr_p, _ = windowed_hr(peaks, fs, n, WINDOW_S, STEP_S)
    wl = int(WINDOW_S * fs)
    rows = []
    for i, c in enumerate(tt):
        if not np.isfinite(hr_t[i]):
            continue
        s0 = max(int(c * fs - wl / 2), 0)
        seg = x[s0:s0 + wl]
        if len(seg) < wl // 2:
            continue
        q = float(np.mean(usqi[s0:s0 + wl]))
        k = float(np.clip(classical_sqi_vector(seg, fs)["kSQI"], 0, 1))
        p = np.asarray(peaks, float) / fs
        sel = p[(p >= c - WINDOW_S / 2) & (p < c + WINDOW_S / 2)]
        spread = (float(max(np.std(60.0 / np.diff(sel)) / np.sqrt(len(sel) - 1), 0.3))
                  if len(sel) >= 3 else 25.0)
        pred = hr_p[i] if np.isfinite(hr_p[i]) else 0.0
        rows.append((hr_t[i], pred, q, k, spread))
    return rows


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="directory with MIT-BIH records")
    ap.add_argument("--nstdb", required=True, help="directory with bw/em/ma records")
    ap.add_argument("--checkpoint", default="checkpoints/hr_model.pt")
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--n-records", type=int, default=0, help="0 = all non-paced")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--noise-records", type=int, default=8)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    detect, engine = load_engine(args.checkpoint)
    base_detect, base_name = load_engine(None)
    print(f"engine: {engine}   baseline: {base_name}\n")

    recs = [r for r in list_records(local_dir=args.data)
            if r.isdigit() and r not in PACED]
    if args.n_records:
        recs = recs[: args.n_records]
    n_keep = int(args.minutes * 60 * FS)

    noise = {}
    for nm in ("bw", "em", "ma"):
        try:
            from hrecg.simulation.noise import load_nstdb_noise
            noise[nm] = load_nstdb_noise(nm, fs_target=FS, local_dir=args.nstdb)
        except Exception as exc:
            print(f"  ! could not load NSTDB '{nm}': {exc}")

    signals, results = {}, {}
    out_json = out / "mitdb_benchmark.json"

    def checkpoint(tag: str) -> None:
        """Persist after every experiment: long CPU runs do get interrupted."""
        out_json.write_text(json.dumps(results, indent=2, default=float))
        print(f"  [saved {tag} -> {out_json.name}]", flush=True)

    # ---------------------------------------------------------------- R1 ---
    print("R1  detection on MIT-BIH as recorded")
    r1 = []
    for rid in recs:
        rec = load_record(rid, local_dir=args.data, fs_target=FS,
                          lead="MLII" if rid not in ("114",) else 0)
        x = rec.signal[:n_keep]
        ref = rec.r_peaks[rec.r_peaks < n_keep].astype(float)
        if len(ref) < 20:
            continue
        signals[rid] = (x, ref)

        row = dict(record=rid, n_beats=len(ref))
        for name, fn in ((engine, detect), (base_name, base_detect)):
            peaks, conf, usqi = fn(x, FS)
            for tol in (0.15, 0.05, 0.025):
                m = match_peaks(ref, peaks, FS, tolerance_s=tol)
                row[f"{name}_F1@{int(tol*1000)}"] = m.f1
                if tol == 0.05:
                    row[f"{name}_Se"] = m.sensitivity
                    row[f"{name}_PPV"] = m.ppv
                    row[f"{name}_jit"] = (float(np.std(m.errors_s) * 1000)
                                          if m.errors_s.size else np.nan)
        r1.append(row)
        print(f"  {rid}  {engine} F1@50={row[f'{engine}_F1@50']:.4f}  "
              f"{base_name} F1@50={row[f'{base_name}_F1@50']:.4f}", flush=True)

    def agg(rows, key):
        v = np.array([r[key] for r in rows if np.isfinite(r.get(key, np.nan))])
        return float(np.mean(v)) if len(v) else float("nan")

    results["R1"] = {
        "n_records": len(r1),
        **{f"{n}_{k}": agg(r1, f"{n}_{k}")
           for n in (engine, base_name)
           for k in ("F1@150", "F1@50", "F1@25", "Se", "PPV", "jit")},
    }
    print("  mean:", {k: round(v, 4) for k, v in results["R1"].items()
                      if isinstance(v, float)})
    checkpoint("R1")

    # ---------------------------------------------------------------- R2 ---
    print("\nR2  noise stress with real NSTDB records")
    r2 = []
    subset = [r["record"] for r in r1][:args.noise_records]
    for nm, nz in noise.items():
        for snr in (-6, 0, 6, 12):
            f1s, maes, f1b, maeb = [], [], [], []
            for rid in subset:
                x, ref = signals[rid]
                seg = nz[:len(x)] if len(nz) >= len(x) else np.tile(
                    nz, int(np.ceil(len(x) / len(nz))))[:len(x)]
                y, _ = mix_at_snr(x, seg, snr, FS, local=True)
                hr_t, _ = windowed_hr(ref, FS, len(x), WINDOW_S, STEP_S)
                for fn, F, M in ((detect, f1s, maes), (base_detect, f1b, maeb)):
                    p, _, _ = fn(y, FS)
                    F.append(match_peaks(ref, p, FS, tolerance_s=0.05).f1)
                    hr_p, _ = windowed_hr(p, FS, len(y), WINDOW_S, STEP_S)
                    M.append(hr_agreement(hr_t, hr_p).mae)
            r2.append(dict(noise=nm, snr_db=snr,
                           F1=float(np.nanmean(f1s)), HR_MAE=float(np.nanmean(maes)),
                           F1_base=float(np.nanmean(f1b)),
                           HR_MAE_base=float(np.nanmean(maeb))))
            print(f"  {nm} {snr:+3d} dB   {engine}: F1={r2[-1]['F1']:.4f} "
                  f"MAE={r2[-1]['HR_MAE']:.2f}   {base_name}: "
                  f"F1={r2[-1]['F1_base']:.4f} MAE={r2[-1]['HR_MAE_base']:.2f}",
                  flush=True)
    results["R2"] = r2
    checkpoint("R2")

    # ------------------------------------------------------------ R3 / R4 ---
    print("\nR3  conformal heart-rate intervals (record-disjoint split)")
    ids = [r["record"] for r in r1]
    cal_ids, test_ids = ids[0::2], ids[1::2]
    em = noise.get("em")

    def collect(rec_ids):
        rows = []
        for rid in rec_ids:
            x, ref = signals[rid]
            for snr in (None, 4):
                if snr is None or em is None:
                    y = x
                else:
                    seg = np.tile(em, int(np.ceil(len(x) / len(em))))[:len(x)]
                    y, _ = mix_at_snr(x, seg, snr, FS, local=True)
                p, c, u = detect(y, FS)
                rows += window_features(y, FS, p, u, ref, len(y))
        return np.asarray(rows, dtype=float)

    cal = collect(cal_ids)
    test = collect(test_ids)
    print(f"  calibration windows {len(cal)}, test windows {len(test)}")

    y_c, m_c, q_c, k_c, s_c = cal.T
    y_t, m_t, q_t, k_t, s_t = test.T
    r3 = {}
    for mode in ("split", "mondrian"):
        cp = ConformalHR(alpha=args.alpha, mode=mode, n_bins=5, min_per_bin=40)
        cp.fit(y_c, m_c - s_c, m_c, m_c + s_c, usqi_cal=q_c)
        lo, hi = cp.predict(m_t - s_t, m_t, m_t + s_t, usqi=q_t)
        im = interval_metrics(y_t, lo, hi, alpha=args.alpha)
        cc = conditional_coverage(y_t, lo, hi, q_t, n_bins=5)
        r3[mode] = dict(**im.as_dict(), worst_slab=cc["worst_slab_coverage"],
                        bins=cc["bins"])
        print(f"  {mode:9s} PICP={im.picp:.3f}  MPIW={im.mpiw:.2f} bpm  "
              f"Winkler={im.winkler:.2f}  worst-stratum={cc['worst_slab_coverage']:.3f}")
    results["R3"] = r3
    checkpoint("R3")

    print("\nR4  selective prediction")
    err = np.abs(m_t - y_t)
    r4 = {}
    for nm, score in (("uSQI", q_t), ("kSQI", k_t)):
        cov, risk = risk_coverage_curve(err, score)
        r4[nm] = dict(aurc=aurc(err, score),
                      risk_at_100=float(risk[-1]),
                      risk_at_90=float(np.interp(0.90, cov, risk)),
                      risk_at_75=float(np.interp(0.75, cov, risk)))
        print(f"  {nm:5s} AURC={r4[nm]['aurc']:.3f}  MAE all={r4[nm]['risk_at_100']:.2f} "
              f"-> 90% kept={r4[nm]['risk_at_90']:.2f} -> 75% kept={r4[nm]['risk_at_75']:.2f} bpm")
    sc = SelectiveController(target_risk=3.0).fit(err, q_t)
    r4["controller"] = sc.summary()
    print(f"  controller: threshold={sc.threshold:.3f} keeps "
          f"{sc.achieved_coverage*100:.0f}% at risk {sc.achieved_risk:.2f} bpm")
    results["R4"] = r4
    checkpoint("R4")

    # ---------------------------------------------------------------- R5 ---
    print("\nR5  HRV interval coverage")
    cal_items, test_items = [], []
    for group, ids_ in (("cal", cal_ids), ("test", test_ids)):
        for rid in ids_:
            x, ref = signals[rid]
            p, c, u = detect(x, FS)
            if len(p) < 30:
                continue
            bi = np.clip(p.astype(int), 0, len(u) - 1)
            item = dict(peaks=p, fs=FS, confidence=c, usqi=u[bi],
                        truth=hrv_features(ref, FS, correction="malik"))
            (cal_items if group == "cal" else test_items).append(item)

    unc = HRVUncertainty(alpha=args.alpha, n_draws=120, jitter_ms=6.0)
    unc.fit(cal_items)
    feats = ("mean_hr_bpm", "sdnn_ms", "rmssd_ms", "pnn50_pct")
    cover = {f: [] for f in feats}
    width = {f: [] for f in feats}
    for it in test_items:
        iv = unc.interval(it["peaks"], FS, it["confidence"], it["usqi"])
        for f in feats:
            lo, med, hi = iv.get(f, (np.nan,) * 3)
            y = it["truth"].get(f, np.nan)
            if np.isfinite(y) and np.isfinite(lo):
                cover[f].append(lo <= y <= hi)
                width[f].append(hi - lo)
    results["R5"] = {f: dict(coverage=float(np.mean(cover[f])) if cover[f] else np.nan,
                             mean_width=float(np.mean(width[f])) if width[f] else np.nan,
                             n=len(cover[f])) for f in feats}
    for f in feats:
        v = results["R5"][f]
        print(f"  {f:14s} coverage={v['coverage']:.3f}  width={v['mean_width']:.2f}  n={v['n']}")

    results["meta"] = dict(engine=engine, baseline=base_name, fs=FS,
                           minutes=args.minutes, alpha=args.alpha,
                           n_records=len(r1), excluded_paced=sorted(PACED))
    checkpoint("R5")
    np.save(out / "conformal_windows.npy",
            np.column_stack([y_t, m_t, q_t, k_t, s_t]))
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
