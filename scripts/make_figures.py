"""
Produce the result figures from `results/mitdb_benchmark.json`.

Four panels, each answering one question:

  fig1  Does the model detect beats better than the classical baseline, and
        does the answer depend on how strict the tolerance is?
  fig2  How does that hold up under the three real NSTDB artifact types?
  fig3  Does the conformal interval actually cover, and does conditioning on
        predicted quality fix the stratum where split conformal fails?
  fig4  Does abstaining on low-quality windows buy a real error reduction, and
        does the learned quality score rank windows better than kSQI?

Usage
-----
    python scripts/make_figures.py --results results --out docs/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, RED, GREEN, GREY, ORANGE = "#1B4F72", "#B03A2E", "#117A65", "#7F8C8D", "#CA6F1E"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.tick_params(labelsize=8)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    ax.title.set_size(10)


def fig1(res, out):
    r1 = res["R1"]
    eng = res["meta"]["engine"]
    base = res["meta"]["baseline"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), dpi=150)

    ax = axes[0]
    tol = ["F1@150", "F1@50", "F1@25"]
    x = np.arange(3)
    ax.bar(x - 0.19, [r1[f"{eng}_{t}"] for t in tol], 0.36, color=BLUE, label="HR-from-ECG v2")
    ax.bar(x + 0.19, [r1[f"{base}_{t}"] for t in tol], 0.36, color=GREY, label=base)
    ax.set_xticks(x, ["±150 ms", "±50 ms", "±25 ms"])
    ax.set_ylim(0.80, 1.005)
    ax.set_ylabel("F1")
    ax.set_title("(a) detection vs matching tolerance")
    ax.legend(fontsize=8, frameon=False)
    for i, t in enumerate(tol):
        d = r1[f"{eng}_{t}"] - r1[f"{base}_{t}"]
        ax.annotate(f"+{d:.3f}", (i, max(r1[f'{eng}_{t}'], r1[f'{base}_{t}']) + 0.004),
                    ha="center", fontsize=7.5, color=BLUE)
    _style(ax)

    ax = axes[1]
    labels = ["sensitivity", "PPV", "jitter (ms)"]
    q = [r1[f"{eng}_Se"], r1[f"{eng}_PPV"], r1[f"{eng}_jit"]]
    b = [r1[f"{base}_Se"], r1[f"{base}_PPV"], r1[f"{base}_jit"]]
    x = np.arange(2)
    ax.bar(x - 0.19, q[:2], 0.36, color=BLUE)
    ax.bar(x + 0.19, b[:2], 0.36, color=GREY)
    ax.set_xticks(x, labels[:2])
    ax.set_ylim(0.85, 1.005)
    ax.set_ylabel("rate")
    ax2 = ax.twinx()
    ax2.bar([2 - 0.19], [q[2]], 0.36, color=BLUE)
    ax2.bar([2 + 0.19], [b[2]], 0.36, color=GREY)
    ax2.set_ylabel("localisation jitter (ms)", fontsize=9)
    ax2.tick_params(labelsize=8)
    ax.set_xlim(-0.5, 2.5)
    ax.set_xticks([0, 1, 2], labels)
    ax.set_title("(b) sensitivity, precision and timing jitter")
    _style(ax)

    fig.suptitle(f"MIT-BIH Arrhythmia Database, {r1['n_records']} records "
                 f"(paced excluded) — model trained on simulation only",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "fig1_detection.png", bbox_inches="tight")
    plt.close(fig)


def fig2(res, out):
    rows = res["R2"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), dpi=150)
    styles = {"bw": (BLUE, "o", "baseline wander"),
              "em": (RED, "s", "electrode motion"),
              "ma": (GREEN, "^", "muscle artifact")}

    for key, ylab, title, log in (("F1", "F1 @50 ms", "(a) detection under real artifact", False),
                                  ("HR_MAE", "heart-rate MAE (bpm)",
                                   "(b) heart-rate error under real artifact", True)):
        ax = axes[0] if key == "F1" else axes[1]
        for nm, (c, mk, lab) in styles.items():
            r = [x for x in rows if x["noise"] == nm]
            if not r:
                continue
            snr = [x["snr_db"] for x in r]
            ax.plot(snr, [x[key] for x in r], marker=mk, ms=4, lw=1.5, color=c, label=lab)
            ax.plot(snr, [x[key + "_base"] for x in r], marker=mk, ms=3.5, lw=1.1,
                    color=c, ls="--", alpha=0.55)
        if log:
            ax.set_yscale("log")
            ax.axhline(5.0, color="k", lw=1, ls=":", label="AAMI EC13 (5 bpm)")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        _style(ax)
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower right")
    axes[1].legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.text(0.5, -0.04, "solid: HR-from-ECG v2    dashed: Pan-Tompkins",
             ha="center", fontsize=8, color=GREY)
    fig.tight_layout()
    fig.savefig(out / "fig2_noise_stress.png", bbox_inches="tight")
    plt.close(fig)


def fig3(res, out):
    r3 = res.get("R3")
    if not r3:
        return
    target = 1 - res["meta"]["alpha"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), dpi=150)

    ax = axes[0]
    for mode, c in (("split", GREY), ("mondrian", BLUE)):
        b = r3[mode]["bins"]
        if not b:
            continue
        centres = [(x["lo_edge"] + x["hi_edge"]) / 2 for x in b]
        centres = [c_ if np.isfinite(c_) else (b[i]["hi_edge"] if i == 0 else b[i]["lo_edge"])
                   for i, c_ in enumerate(centres)]
        ax.plot(range(len(b)), [x["coverage"] for x in b], "o-", ms=4, color=c,
                label=f"{mode} conformal")
    ax.axhline(target, color=RED, ls="--", lw=1.2, label=f"nominal {target:.0%}")
    ax.set_xticks(range(len(r3["mondrian"]["bins"])),
                  [f"Q{i+1}" for i in range(len(r3["mondrian"]["bins"]))])
    ax.set_xlabel("predicted-quality stratum  (Q1 = worst signal)")
    ax.set_ylabel("empirical coverage")
    ax.set_title("(a) coverage within quality strata")
    ax.legend(fontsize=7.5, frameon=False)
    _style(ax)

    ax = axes[1]
    metrics = ["PICP", "worst_slab", "MPIW"]
    labels = ["marginal\ncoverage", "worst-stratum\ncoverage", "mean width\n(bpm)"]
    x = np.arange(3)
    sp = [r3["split"]["PICP"], r3["split"]["worst_slab"], r3["split"]["MPIW"]]
    mo = [r3["mondrian"]["PICP"], r3["mondrian"]["worst_slab"], r3["mondrian"]["MPIW"]]
    ax.bar(x[:2] - 0.19, sp[:2], 0.36, color=GREY, label="split")
    ax.bar(x[:2] + 0.19, mo[:2], 0.36, color=BLUE, label="Mondrian")
    ax.axhline(target, color=RED, ls="--", lw=1.2)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("coverage")
    ax2 = ax.twinx()
    ax2.bar([2 - 0.19], [sp[2]], 0.36, color=GREY)
    ax2.bar([2 + 0.19], [mo[2]], 0.36, color=BLUE)
    ax2.set_ylabel("interval width (bpm)", fontsize=9)
    ax2.tick_params(labelsize=8)
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.5, 2.5)
    ax.set_title("(b) marginal validity hides stratum failure")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left",
              bbox_to_anchor=(0.01, 0.62))
    _style(ax)

    fig.tight_layout()
    fig.savefig(out / "fig3_conformal.png", bbox_inches="tight")
    plt.close(fig)


def fig4(res, out, results_dir):
    r4 = res.get("R4")
    if not r4:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), dpi=150)

    ax = axes[0]
    npy = results_dir / "conformal_windows.npy"
    if npy.exists():
        from hrecg.metrics.uncertainty import risk_coverage_curve

        y, m, q, k, s = np.load(npy).T
        err = np.abs(m - y)
        for score, c, lab in ((q, BLUE, "uSQI (learned)"), (k, ORANGE, "kSQI (classical)")):
            cov, risk = risk_coverage_curve(err, score)
            ax.plot(cov * 100, risk, lw=1.6, color=c, label=lab)
        ax.set_xlabel("coverage — % of windows reported")
        ax.set_ylabel("selective risk — MAE on reported (bpm)")
        ax.set_title("(a) risk–coverage")
        ax.legend(fontsize=8, frameon=False)
        _style(ax)

    ax = axes[1]
    names = [n for n in ("uSQI", "kSQI") if n in r4]
    x = np.arange(3)
    w = 0.36
    for i, n in enumerate(names):
        vals = [r4[n]["risk_at_100"], r4[n]["risk_at_90"], r4[n]["risk_at_75"]]
        ax.bar(x + (i - 0.5) * w, vals, w,
               color=BLUE if n == "uSQI" else ORANGE, label=n)
    ax.set_xticks(x, ["100 %\n(no abstention)", "90 % kept", "75 % kept"])
    ax.set_ylabel("MAE on reported windows (bpm)")
    ax.set_title("(b) what abstention buys")
    ax.legend(fontsize=8, frameon=False)
    _style(ax)

    fig.tight_layout()
    fig.savefig(out / "fig4_selective.png", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    rd = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = json.loads((rd / "mitdb_benchmark.json").read_text())

    fig1(res, out)
    print("wrote fig1_detection.png")
    fig2(res, out)
    print("wrote fig2_noise_stress.png")
    fig3(res, out)
    print("wrote fig3_conformal.png")
    fig4(res, out, rd)
    print("wrote fig4_selective.png")


if __name__ == "__main__":
    main()
