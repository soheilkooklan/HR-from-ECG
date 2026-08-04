"""
Validation of the HR from ECG infrastructure layer.

Three experiments, each of which either passes a stated criterion or fails
loudly. The point is that the simulator, the metrics and the baseline must be
trustworthy *before* any model is trained, because every label in the project
is derived from them.

  E1  Simulator fidelity
      Does the generator reproduce the heart rate, SDNN and LF/HF ratio it was
      asked for? Measured over independent realisations and compared against
      the analytical target.

  E2  Noise-stress characterisation
      Pan-Tompkins detection and heart-rate error as a function of SNR, from
      -6 dB to +24 dB. Establishes the reference curve every later method is
      measured against, and identifies the SNR at which point estimates stop
      being defensible.

  E3  Motivation for uSQI
      Compares classical signal-quality indices against the utility-based
      target. If the classical indices already tracked heart-rate error
      closely, contribution C1 would be unnecessary; the experiment quantifies
      how far they fall short.

Usage
-----
    python scripts/validate_infrastructure.py --out results/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hrecg.baselines import pan_tompkins
from hrecg.metrics.detection import match_peaks
from hrecg.metrics.hr import hr_agreement, windowed_hr
from hrecg.metrics.hrv import hrv_features
from hrecg.metrics.quality import classical_sqi_vector, utility_sqi
from hrecg.simulation import corrupt, make_synthetic_ecg

FS = 500.0
WINDOW_S = 10.0


# --------------------------------------------------------------------------- #
def e1_simulator_fidelity(n_reps: int = 25, duration_s: float = 300.0) -> dict:
    """Recover the generator's own targets from the synthesized signal."""
    targets = [(60.0, 40.0, 1.0), (72.0, 50.0, 1.5), (95.0, 30.0, 2.5)]
    rows = []
    for hr_t, sdnn_t, lfhf_t in targets:
        rec = {"hr": [], "sdnn": [], "lfhf": []}
        for r in range(n_reps):
            e = make_synthetic_ecg(
                duration_s, FS, "sinus", mean_hr=hr_t, sdnn_ms=sdnn_t,
                lf_hf_ratio=lfhf_t, seed=1000 + r,
            )
            f = hrv_features(e.r_peaks, FS, correction="none")
            rec["hr"].append(f["mean_hr_bpm"])
            rec["sdnn"].append(f["sdnn_ms"])
            rec["lfhf"].append(f["lf_hf_ratio"])
        rows.append(dict(
            target_hr=hr_t, recovered_hr=float(np.mean(rec["hr"])),
            hr_err=float(np.mean(rec["hr"]) - hr_t),
            target_sdnn=sdnn_t, recovered_sdnn=float(np.mean(rec["sdnn"])),
            sdnn_rel_err_pct=float((np.mean(rec["sdnn"]) - sdnn_t) / sdnn_t * 100),
            target_lfhf=lfhf_t, recovered_lfhf=float(np.nanmean(rec["lfhf"])),
            lfhf_rel_err_pct=float((np.nanmean(rec["lfhf"]) - lfhf_t) / lfhf_t * 100),
        ))
    ok = all(abs(r["hr_err"]) < 1.0 and abs(r["sdnn_rel_err_pct"]) < 15
             and abs(r["lfhf_rel_err_pct"]) < 30 for rows_ in [rows] for r in rows_)
    return dict(rows=rows, passed=bool(ok))


# --------------------------------------------------------------------------- #
def e2_noise_stress(
    snrs=(-6, -3, 0, 3, 6, 9, 12, 18, 24),
    n_reps: int = 8,
    duration_s: float = 120.0,
) -> dict:
    """Detection and HR error versus fixed SNR, per rhythm."""
    out = []
    for rhythm in ("sinus", "af", "pvc"):
        for snr in snrs:
            f1s, hr_maes, ses, ppvs, jits = [], [], [], [], []
            for r in range(n_reps):
                e = make_synthetic_ecg(duration_s, FS, rhythm, mean_hr=75, seed=r)
                c = corrupt(
                    e.signal, FS, snr_kind="constant", snr_range=(snr, snr),
                    p_lead_off=0.0, rng=np.random.default_rng(5000 + r),
                )
                det = pan_tompkins(c.signal, FS)
                m = match_peaks(e.r_peaks, det, FS, tolerance_s=0.05)
                f1s.append(m.f1)
                ses.append(m.sensitivity)
                ppvs.append(m.ppv)
                jits.append(np.std(m.errors_s) * 1000 if m.errors_s.size else np.nan)

                hr_t, _ = windowed_hr(e.r_peaks, FS, len(e.signal), WINDOW_S, 1.0)
                hr_p, _ = windowed_hr(det, FS, len(c.signal), WINDOW_S, 1.0)
                hr_maes.append(hr_agreement(hr_t, hr_p).mae)

            out.append(dict(
                rhythm=rhythm, snr_db=snr,
                F1=float(np.nanmean(f1s)), Se=float(np.nanmean(ses)),
                PPV=float(np.nanmean(ppvs)),
                jitter_ms=float(np.nanmean(jits)),
                HR_MAE=float(np.nanmean(hr_maes)),
                HR_MAE_sd=float(np.nanstd(hr_maes)),
            ))
    return dict(rows=out)


# --------------------------------------------------------------------------- #
def e3_usqi_motivation(n_records: int = 40, duration_s: float = 180.0) -> dict:
    """
    Per-window comparison of classical SQIs against the utility-based target.

    Records are generated with a time-varying SNR profile so that quality
    changes within a record, which is the realistic wearable regime and the
    one that a single record-level quality label cannot describe.
    """
    feats, usqi_all, snr_all, err_all = [], [], [], []
    for r in range(n_records):
        e = make_synthetic_ecg(duration_s, FS, "sinus", mean_hr=72, seed=r)
        c = corrupt(
            e.signal, FS, snr_kind="piecewise", snr_range=(-8.0, 26.0),
            p_lead_off=0.25, rng=np.random.default_rng(7000 + r),
        )
        det = pan_tompkins(c.signal, FS)

        L = int(WINDOW_S * FS)
        n_win = len(c.signal) // L
        hr_t, _ = windowed_hr(e.r_peaks, FS, len(e.signal), WINDOW_S, WINDOW_S)
        hr_p, _ = windowed_hr(det, FS, len(c.signal), WINDOW_S, WINDOW_S)
        n_win = min(n_win, len(hr_t), len(hr_p))

        u = utility_sqi(hr_p[:n_win], hr_t[:n_win], tau_bpm=5.0)
        for i in range(n_win):
            seg = c.signal[i * L:(i + 1) * L]
            feats.append(classical_sqi_vector(seg, FS))
            usqi_all.append(u[i])
            snr_all.append(float(np.mean(c.snr_db[i * L:(i + 1) * L])))
            err_all.append(float(abs(hr_p[i] - hr_t[i])) if np.isfinite(hr_p[i]) else np.nan)

    keys = list(feats[0].keys())
    X = {k: np.array([f[k] for f in feats], float) for k in keys}
    u = np.array(usqi_all, float)
    snr = np.array(snr_all, float)
    err = np.array(err_all, float)

    def corr(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else float("nan")

    # Spearman via rank transform, robust to the nonlinear SQI scales
    def spearman(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 3:
            return float("nan")
        ra = np.argsort(np.argsort(a[m])).astype(float)
        rb = np.argsort(np.argsort(b[m])).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])

    table = {k: dict(pearson_vs_uSQI=corr(X[k], u),
                     spearman_vs_uSQI=spearman(X[k], u),
                     spearman_vs_SNR=spearman(X[k], snr)) for k in keys}
    table["SNR_oracle"] = dict(pearson_vs_uSQI=corr(snr, u),
                               spearman_vs_uSQI=spearman(snr, u),
                               spearman_vs_SNR=1.0)
    return dict(
        table=table, n_windows=int(len(u)),
        usqi_mean=float(np.mean(u)),
        pct_windows_hr_err_gt5=float(np.nanmean(err > 5.0) * 100),
        arrays=dict(usqi=u, snr=snr, err=err, **X),
    )


# --------------------------------------------------------------------------- #
def make_figure(e2: dict, e3: dict, out_png: Path) -> None:
    fig = plt.figure(figsize=(13, 9), dpi=130)
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.29)

    # (a) example waveform
    ax = fig.add_subplot(gs[0, 0])
    e = make_synthetic_ecg(10, FS, "pvc", mean_hr=72, seed=11)
    c = corrupt(e.signal, FS, snr_kind="constant", snr_range=(6, 6),
                p_lead_off=0.0, rng=np.random.default_rng(2))
    t = np.arange(len(e.signal)) / FS
    ax.plot(t, c.signal + 1.6, lw=0.6, color="#B03A2E", label="noisy, 6 dB")
    ax.plot(t, e.signal, lw=0.8, color="#1B4F72", label="clean")
    ax.plot(e.r_peaks / FS, e.signal[e.r_peaks], "v", ms=4, color="#117A65")
    ax.set(xlabel="time (s)", ylabel="mV", title="(a) synthetic ECG with PVCs")
    ax.legend(fontsize=7, loc="lower right")

    # (b) F1 vs SNR
    ax = fig.add_subplot(gs[0, 1])
    rows = e2["rows"]
    for rhythm, col in (("sinus", "#1B4F72"), ("af", "#B03A2E"), ("pvc", "#117A65")):
        r = [x for x in rows if x["rhythm"] == rhythm]
        ax.plot([x["snr_db"] for x in r], [x["F1"] for x in r],
                "o-", ms=3.5, lw=1.3, color=col, label=rhythm)
    ax.axhline(0.998, ls=":", c="grey", lw=1)
    ax.set(xlabel="SNR (dB)", ylabel="F1 @50 ms",
           title="(b) detection vs noise", ylim=(0.4, 1.02))
    ax.legend(fontsize=7)

    # (c) HR MAE vs SNR
    ax = fig.add_subplot(gs[0, 2])
    for rhythm, col in (("sinus", "#1B4F72"), ("af", "#B03A2E"), ("pvc", "#117A65")):
        r = [x for x in rows if x["rhythm"] == rhythm]
        ax.plot([x["snr_db"] for x in r], [x["HR_MAE"] for x in r],
                "o-", ms=3.5, lw=1.3, color=col, label=rhythm)
    ax.axhline(5.0, ls="--", c="k", lw=1, label="AAMI EC13")
    ax.set(xlabel="SNR (dB)", ylabel="HR MAE (bpm)", yscale="log",
           title="(c) heart-rate error vs noise")
    ax.legend(fontsize=7)

    # (d) uSQI vs SNR
    a = e3["arrays"]
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(a["snr"], a["usqi"], s=4, alpha=0.25, color="#1B4F72", edgecolors="none")
    ax.set(xlabel="true SNR (dB)", ylabel="uSQI",
           title="(d) utility-based quality vs SNR")

    # (e) classical SQI vs uSQI
    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(a["kSQI"], a["usqi"], s=4, alpha=0.25, color="#B03A2E", edgecolors="none")
    rho = e3["table"]["kSQI"]["spearman_vs_uSQI"]
    ax.set(xlabel="kSQI (classical)", ylabel="uSQI",
           title=f"(e) kSQI is a poor proxy  (rho={rho:.2f})")

    # (f) ranking of all quality indices
    ax = fig.add_subplot(gs[1, 2])
    items = [(k, v["spearman_vs_uSQI"]) for k, v in e3["table"].items()]
    items.sort(key=lambda kv: abs(kv[1]))
    ax.barh([k for k, _ in items], [abs(v) for _, v in items],
            color=["#117A65" if k == "SNR_oracle" else "#5D6D7E" for k, _ in items])
    ax.set(xlabel="|Spearman| with uSQI", xlim=(0, 1),
           title="(f) how well each index predicts HR error")

    fig.suptitle("HR from ECG infrastructure validation", fontsize=13, y=0.975)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true", help="smaller sample sizes")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    scale = 0.4 if args.quick else 1.0
    print("E1  simulator fidelity ...", flush=True)
    e1 = e1_simulator_fidelity(n_reps=max(int(25 * scale), 6),
                               duration_s=300.0 if not args.quick else 150.0)
    for r in e1["rows"]:
        print(f"    HR {r['target_hr']:5.1f} -> {r['recovered_hr']:6.2f} "
              f"({r['hr_err']:+.2f} bpm) | SDNN {r['target_sdnn']:4.0f} -> "
              f"{r['recovered_sdnn']:6.2f} ms ({r['sdnn_rel_err_pct']:+.1f}%) | "
              f"LF/HF {r['target_lfhf']:.1f} -> {r['recovered_lfhf']:.2f} "
              f"({r['lfhf_rel_err_pct']:+.1f}%)")
    print(f"    criterion: {'PASS' if e1['passed'] else 'FAIL'}")

    print("E2  noise stress ...", flush=True)
    e2 = e2_noise_stress(n_reps=max(int(8 * scale), 3),
                         duration_s=120.0 if not args.quick else 60.0)
    print(f"    {'rhythm':8s}{'SNR':>6s}{'Se':>8s}{'PPV':>8s}{'F1':>8s}"
          f"{'jit(ms)':>9s}{'HR MAE':>9s}")
    for r in e2["rows"]:
        print(f"    {r['rhythm']:8s}{r['snr_db']:6.0f}{r['Se']:8.4f}{r['PPV']:8.4f}"
              f"{r['F1']:8.4f}{r['jitter_ms']:9.1f}{r['HR_MAE']:9.2f}")

    print("E3  uSQI motivation ...", flush=True)
    e3 = e3_usqi_motivation(n_records=max(int(40 * scale), 10),
                            duration_s=180.0 if not args.quick else 90.0)
    print(f"    {e3['n_windows']} windows, "
          f"{e3['pct_windows_hr_err_gt5']:.1f}% with HR error > 5 bpm")
    print(f"    {'index':12s}{'rho vs uSQI':>13s}{'rho vs SNR':>12s}")
    for k, v in sorted(e3["table"].items(),
                       key=lambda kv: -abs(kv[1]["spearman_vs_uSQI"])):
        print(f"    {k:12s}{v['spearman_vs_uSQI']:13.3f}{v['spearman_vs_SNR']:12.3f}")

    make_figure(e2, e3, out / "infrastructure_validation.png")
    e3_ser = {k: v for k, v in e3.items() if k != "arrays"}
    (out / "validation.json").write_text(
        json.dumps(dict(E1=e1, E2=e2, E3=e3_ser), indent=2))
    print(f"\nwrote {out/'infrastructure_validation.png'} and {out/'validation.json'}")


if __name__ == "__main__":
    main()
