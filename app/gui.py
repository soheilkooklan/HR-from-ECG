"""
HR from ECG Analyzer -- desktop application.

The interface is built around one design rule: **no number is displayed without
its uncertainty**. Version 1 of this project printed "Average Heart Rate: 72.43
bpm" for any input, including a flat line. Here every reported quantity carries
an interval, every window carries a quality score, and windows the system
cannot support are drawn as gaps rather than filled with a plausible-looking
guess.

Tkinter is used deliberately rather than Qt: it ships with Python, so the
application runs from a clone with no binary dependencies, which matters for a
tool intended to be reproducible by reviewers and students.
"""

from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from hrecg.pipeline import HRPipeline, PipelineResult

# --------------------------------------------------------------------------- #
#  Palette
# --------------------------------------------------------------------------- #
BG = "#0E1621"
PANEL = "#16212E"
CARD = "#1B2836"
FG = "#E6EDF3"
MUTED = "#8B9BB0"
ACCENT = "#37D6A0"
ACCENT2 = "#4C9AFF"
WARN = "#F2B24C"
BAD = "#FF6B6B"
GRID = "#243447"


class HRApp(tk.Tk):
    def __init__(self, checkpoint: str | None = None):
        super().__init__()
        self.title("HR from ECG Analyzer  v2.0")
        self.geometry("1440x900")
        self.configure(bg=BG)
        self.minsize(1180, 760)

        self.pipeline = HRPipeline(checkpoint=checkpoint)
        self.result: PipelineResult | None = None
        self.signal: np.ndarray | None = None
        self.fs: float = 250.0
        self.source_name = "no signal loaded"
        self._queue: queue.Queue = queue.Queue()

        self._style()
        self._build()
        self.after(120, self._poll)

    # ---------------------------------------------------------------- styling
    def _style(self) -> None:
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=BG, foreground=FG,
                    fieldbackground=CARD, borderwidth=0)
        s.configure("TFrame", background=BG)
        s.configure("Panel.TFrame", background=PANEL)
        s.configure("Card.TFrame", background=CARD)
        s.configure("TLabel", background=BG, foreground=FG,
                    font=("DejaVu Sans", 10))
        s.configure("Panel.TLabel", background=PANEL, foreground=FG)
        s.configure("Card.TLabel", background=CARD, foreground=FG)
        s.configure("Muted.TLabel", background=PANEL, foreground=MUTED,
                    font=("DejaVu Sans", 9))
        s.configure("CardMuted.TLabel", background=CARD, foreground=MUTED,
                    font=("DejaVu Sans", 9))
        s.configure("Title.TLabel", background=BG, foreground=FG,
                    font=("DejaVu Sans", 17, "bold"))
        s.configure("Sub.TLabel", background=BG, foreground=MUTED,
                    font=("DejaVu Sans", 9))
        s.configure("Big.TLabel", background=CARD, foreground=ACCENT,
                    font=("DejaVu Sans Mono", 30, "bold"))
        s.configure("Unit.TLabel", background=CARD, foreground=MUTED,
                    font=("DejaVu Sans", 11))
        s.configure("Section.TLabel", background=PANEL, foreground=ACCENT2,
                    font=("DejaVu Sans", 9, "bold"))
        s.configure("TButton", background=CARD, foreground=FG, padding=(10, 7),
                    font=("DejaVu Sans", 10))
        s.map("TButton", background=[("active", GRID)])
        s.configure("Accent.TButton", background=ACCENT, foreground="#06231A",
                    font=("DejaVu Sans", 10, "bold"))
        s.map("Accent.TButton", background=[("active", "#2FBF8D")])
        s.configure("TScale", background=PANEL, troughcolor=GRID)
        s.configure("Treeview", background=CARD, fieldbackground=CARD,
                    foreground=FG, rowheight=25, borderwidth=0,
                    font=("DejaVu Sans Mono", 9))
        s.configure("Treeview.Heading", background=GRID, foreground=MUTED,
                    font=("DejaVu Sans", 9, "bold"), relief="flat")
        s.map("Treeview", background=[("selected", GRID)])
        s.configure("TCombobox", fieldbackground=CARD, background=CARD)

    # ---------------------------------------------------------------- layout
    def _build(self) -> None:
        head = ttk.Frame(self, padding=(18, 12, 18, 8))
        head.pack(fill="x")
        ttk.Label(head, text="HR from ECG  Analyzer", style="Title.TLabel").pack(anchor="w")
        ttk.Label(head,
                  text="quality-aware heart rate and HRV with distribution-free "
                       "uncertainty  ·  every value reported as an interval",
                  style="Sub.TLabel").pack(anchor="w")

        body = ttk.Frame(self, padding=(14, 0, 14, 12))
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self._build_sidebar(body)
        self._build_main(body)
        self._build_status()

    def _build_sidebar(self, parent) -> None:
        side = ttk.Frame(parent, style="Panel.TFrame", padding=14, width=270)
        side.grid(row=0, column=0, sticky="ns", padx=(0, 12))
        side.grid_propagate(False)

        def section(text):
            ttk.Label(side, text=text.upper(), style="Section.TLabel").pack(
                anchor="w", pady=(14, 6))

        section("signal")
        ttk.Button(side, text="Open ECG file…", command=self.on_open).pack(fill="x", pady=3)
        ttk.Button(side, text="Load demo recording", command=self.on_demo).pack(fill="x", pady=3)

        self.var_demo = tk.StringVar(value="noisy sinus")
        cb = ttk.Combobox(side, textvariable=self.var_demo, state="readonly",
                          values=["clean sinus", "noisy sinus", "atrial fibrillation",
                                  "frequent PVCs", "electrode failure"])
        cb.pack(fill="x", pady=(6, 0))

        ttk.Label(side, text="sampling rate (Hz)", style="Muted.TLabel").pack(
            anchor="w", pady=(12, 2))
        self.var_fs = tk.StringVar(value="250")
        ttk.Entry(side, textvariable=self.var_fs, width=10).pack(fill="x")

        section("uncertainty")
        self.var_alpha = tk.DoubleVar(value=0.10)
        self.lbl_alpha = ttk.Label(side, text="confidence level  90 %", style="Muted.TLabel")
        self.lbl_alpha.pack(anchor="w", pady=(4, 2))
        ttk.Scale(side, from_=0.01, to=0.30, variable=self.var_alpha,
                  command=self._on_alpha).pack(fill="x")

        self.var_risk = tk.DoubleVar(value=3.0)
        self.lbl_risk = ttk.Label(side, text="abstain above  3.0 bpm error",
                                  style="Muted.TLabel")
        self.lbl_risk.pack(anchor="w", pady=(12, 2))
        ttk.Scale(side, from_=1.0, to=10.0, variable=self.var_risk,
                  command=self._on_risk).pack(fill="x")

        section("analysis")
        self.var_engine = tk.StringVar(value=self.pipeline.engine_name)
        ttk.Label(side, textvariable=self.var_engine, style="Muted.TLabel").pack(anchor="w")
        ttk.Button(side, text="Analyze", style="Accent.TButton",
                   command=self.on_analyze).pack(fill="x", pady=(10, 3))
        ttk.Button(side, text="Export report…", command=self.on_export).pack(fill="x", pady=3)
        ttk.Button(side, text="About", command=self.on_about).pack(fill="x", pady=3)

    def _build_main(self, parent) -> None:
        main = ttk.Frame(parent)
        main.grid(row=0, column=1, sticky="nsew")
        main.rowconfigure(1, weight=1)
        main.columnconfigure(0, weight=1)

        cards = ttk.Frame(main)
        cards.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        for i in range(4):
            cards.columnconfigure(i, weight=1)
        self.cards = {}
        for i, (key, title, unit) in enumerate([
            ("hr", "heart rate", "bpm"),
            ("rmssd", "RMSSD", "ms"),
            ("sqi", "mean quality", "uSQI"),
            ("cov", "reported", "% of record"),
        ]):
            c = ttk.Frame(cards, style="Card.TFrame", padding=(14, 11))
            c.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 8, 0))
            ttk.Label(c, text=title.upper(), style="CardMuted.TLabel").pack(anchor="w")
            v = ttk.Label(c, text="—", style="Big.TLabel")
            v.pack(anchor="w")
            ci = ttk.Label(c, text=unit, style="Unit.TLabel")
            ci.pack(anchor="w")
            self.cards[key] = (v, ci)

        lower = ttk.Frame(main)
        lower.grid(row=1, column=0, sticky="nsew")
        lower.columnconfigure(0, weight=3)
        lower.columnconfigure(1, weight=1)
        lower.rowconfigure(0, weight=1)

        self.fig = Figure(figsize=(9.2, 6.2), dpi=100, facecolor=PANEL)
        self.axes = self.fig.subplots(3, 1, sharex=True,
                                      gridspec_kw=dict(height_ratios=[3, 2, 1.2],
                                                       hspace=0.14))
        self.fig.subplots_adjust(left=0.075, right=0.985, top=0.965, bottom=0.075)
        for ax in self.axes:
            self._style_axis(ax)
        self.canvas = FigureCanvasTkAgg(self.fig, master=lower)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self._placeholder()

        right = ttk.Frame(lower, style="Panel.TFrame", padding=12)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        ttk.Label(right, text="HRV WITH 90% INTERVALS", style="Section.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8))
        self.tree = ttk.Treeview(right, columns=("v", "ci"), show="tree headings",
                                 height=14)
        self.tree.heading("#0", text="index")
        self.tree.heading("v", text="value")
        self.tree.heading("ci", text="interval")
        self.tree.column("#0", width=110, anchor="w")
        self.tree.column("v", width=78, anchor="e")
        self.tree.column("ci", width=132, anchor="e")
        self.tree.grid(row=1, column=0, sticky="nsew")
        self.note = ttk.Label(right, text="", style="Muted.TLabel", wraplength=300,
                              justify="left")
        self.note.grid(row=2, column=0, sticky="w", pady=(10, 0))

    def _build_status(self) -> None:
        bar = ttk.Frame(self, style="Panel.TFrame", padding=(16, 7))
        bar.pack(fill="x", side="bottom")
        self.var_status = tk.StringVar(value="ready — load a recording or try a demo")
        ttk.Label(bar, textvariable=self.var_status, style="Muted.TLabel").pack(side="left")
        ttk.Label(bar, text="HR from ECG v2.0", style="Muted.TLabel").pack(side="right")

    # ------------------------------------------------------------- plotting
    def _style_axis(self, ax) -> None:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.grid(True, color=GRID, lw=0.5, alpha=0.6)
        ax.yaxis.label.set_color(MUTED)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_size(9)
        ax.xaxis.label.set_size(9)

    def _placeholder(self) -> None:
        for ax in self.axes:
            ax.clear()
            self._style_axis(ax)
        self.axes[0].text(0.5, 0.5, "load a recording to begin",
                          ha="center", va="center", color=MUTED,
                          transform=self.axes[0].transAxes, fontsize=12)
        self.axes[0].set_ylabel("ECG (mV)")
        self.axes[1].set_ylabel("heart rate (bpm)")
        self.axes[2].set_ylabel("uSQI")
        self.axes[2].set_xlabel("time (s)")
        self.canvas.draw()

    def _draw(self, r: PipelineResult) -> None:
        for ax in self.axes:
            ax.clear()
            self._style_axis(ax)

        t = np.arange(len(r.signal)) / r.fs
        ax = self.axes[0]
        ax.plot(t, r.signal, lw=0.7, color="#9FB3C8")
        if len(r.peaks):
            p = np.clip(r.peaks.astype(int), 0, len(r.signal) - 1)
            good = r.beat_usqi >= r.abstain_threshold
            ax.plot(r.peaks[good] / r.fs, r.signal[p[good]], "v", ms=5,
                    color=ACCENT, label="accepted beats")
            if (~good).any():
                ax.plot(r.peaks[~good] / r.fs, r.signal[p[~good]], "x", ms=5,
                        color=BAD, label="low-quality beats")
        for a, b in r.abstain_spans:
            for axx in self.axes:
                axx.axvspan(a, b, color=BAD, alpha=0.10, lw=0)
        ax.set_ylabel("ECG (mV)")
        ax.legend(loc="upper right", fontsize=7, facecolor=CARD,
                  edgecolor=GRID, labelcolor=FG)

        ax = self.axes[1]
        m = np.isfinite(r.hr)
        if m.any():
            ax.fill_between(r.hr_time[m], r.hr_lo[m], r.hr_hi[m],
                            color=ACCENT2, alpha=0.22,
                            label=f"{int(round((1-r.alpha)*100))}% conformal interval")
            ax.plot(r.hr_time[m], r.hr[m], lw=1.6, color=ACCENT2, label="heart rate")
        ax.set_ylabel("heart rate (bpm)")
        ax.legend(loc="upper right", fontsize=7, facecolor=CARD,
                  edgecolor=GRID, labelcolor=FG)

        ax = self.axes[2]
        ax.plot(t, r.usqi, lw=0.9, color=WARN)
        ax.axhline(r.abstain_threshold, color=BAD, ls="--", lw=1)
        ax.set_ylim(-0.02, 1.05)
        ax.set_ylabel("uSQI")
        ax.set_xlabel("time (s)")
        ax.set_xlim(t[0], t[-1])
        self.canvas.draw()

    # ------------------------------------------------------------- callbacks
    def _on_alpha(self, _=None):
        self.lbl_alpha.config(
            text=f"confidence level  {int(round((1-self.var_alpha.get())*100))} %")

    def _on_risk(self, _=None):
        self.lbl_risk.config(text=f"abstain above  {self.var_risk.get():.1f} bpm error")

    def on_open(self):
        path = filedialog.askopenfilename(
            title="Open ECG",
            filetypes=[("CSV / text", "*.csv *.txt *.tsv"), ("NumPy", "*.npy"),
                       ("All files", "*.*")])
        if not path:
            return
        try:
            x = self.pipeline.load_file(path)
        except Exception as exc:
            messagebox.showerror("Could not read file", str(exc))
            return
        self.signal = x
        self.fs = float(self.var_fs.get())
        self.source_name = Path(path).name
        self.var_status.set(f"loaded {self.source_name} — "
                            f"{len(x)/self.fs:.1f} s at {self.fs:.0f} Hz")
        self.on_analyze()

    def on_demo(self):
        kind = self.var_demo.get()
        self.fs = 250.0
        self.var_fs.set("250")
        self.signal = self.pipeline.demo_signal(kind, fs=self.fs, duration_s=60.0)
        self.source_name = f"demo: {kind}"
        self.var_status.set(f"generated demo recording — {kind}, 60 s at 250 Hz")
        self.on_analyze()

    def on_analyze(self):
        if self.signal is None:
            messagebox.showinfo("No signal", "Load a recording or generate a demo first.")
            return
        self.var_status.set("analysing…")
        self.update_idletasks()
        args = (self.signal.copy(), self.fs, float(self.var_alpha.get()),
                float(self.var_risk.get()))
        threading.Thread(target=self._worker, args=args, daemon=True).start()

    def _worker(self, x, fs, alpha, risk):
        try:
            self._queue.put(("ok", self.pipeline.analyze(x, fs, alpha=alpha,
                                                         risk_bpm=risk)))
        except Exception as exc:  # surfaced in the UI rather than the console
            self._queue.put(("err", exc))

    def _poll(self):
        try:
            tag, payload = self._queue.get_nowait()
        except queue.Empty:
            self.after(120, self._poll)
            return
        if tag == "err":
            messagebox.showerror("Analysis failed", str(payload))
            self.var_status.set("analysis failed")
        else:
            self.result = payload
            self._render(payload)
        self.after(120, self._poll)

    def _render(self, r: PipelineResult):
        self._draw(r)

        def fmt(v, lo, hi, dec=1):
            if not np.isfinite(v):
                return "—", "—"
            return f"{v:.{dec}f}", f"[{lo:.{dec}f}, {hi:.{dec}f}]"

        v, ci = fmt(r.hr_summary[1], r.hr_summary[0], r.hr_summary[2])
        self.cards["hr"][0].config(text=v)
        self.cards["hr"][1].config(text=f"bpm   {ci}")

        rm = r.hrv.get("rmssd_ms", (np.nan,) * 3)
        v, ci = fmt(rm[1], rm[0], rm[2])
        self.cards["rmssd"][0].config(text=v)
        self.cards["rmssd"][1].config(text=f"ms   {ci}")

        self.cards["sqi"][0].config(text=f"{np.nanmean(r.usqi):.2f}")
        self.cards["sqi"][1].config(text="0 = unusable, 1 = ideal")

        self.cards["cov"][0].config(text=f"{r.reported_fraction*100:.0f}")
        self.cards["cov"][1].config(
            text=f"% reported, {100-r.reported_fraction*100:.0f}% abstained")

        for i in self.tree.get_children():
            self.tree.delete(i)
        labels = {"mean_hr_bpm": ("mean HR", "bpm", 1), "sdnn_ms": ("SDNN", "ms", 1),
                  "rmssd_ms": ("RMSSD", "ms", 1), "pnn50_pct": ("pNN50", "%", 1),
                  "sd1_ms": ("SD1", "ms", 1), "sd2_ms": ("SD2", "ms", 1),
                  "lf_hf_ratio": ("LF/HF", "", 2)}
        for key, (name, unit, dec) in labels.items():
            lo, med, hi = r.hrv.get(key, (np.nan,) * 3)
            v, ci = fmt(med, lo, hi, dec)
            self.tree.insert("", "end", text=f"{name} {unit}".strip(), values=(v, ci))

        self.note.config(
            text=f"Intervals are conformal at the {int(round((1-r.alpha)*100))}% level, "
                 f"conditioned on predicted quality. HRV bounds come from "
                 f"{r.n_draws} Monte-Carlo beat ensembles. "
                 f"{len(r.peaks)} beats detected, {r.n_abstained} windows withheld.")
        self.var_status.set(
            f"{self.source_name} — {len(r.signal)/r.fs:.0f} s, {len(r.peaks)} beats, "
            f"engine: {r.engine} — done in {r.elapsed_s:.2f} s")

    def on_export(self):
        if self.result is None:
            messagebox.showinfo("Nothing to export", "Run an analysis first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON report", "*.json"),
                                                 ("CSV table", "*.csv")])
        if not path:
            return
        self.pipeline.export(self.result, path, source=self.source_name)
        self.var_status.set(f"report written to {Path(path).name}")

    def on_about(self):
        messagebox.showinfo(
            "About HR from ECG",
            "HR from ECG Analyzer v2.0\n\n"
            "Quality-aware, uncertainty-calibrated heart rate and HRV\n"
            "estimation from single-lead ECG.\n\n"
            "Every reported value carries a distribution-free interval, "
            "and the analyser abstains where the signal cannot support "
            "an answer.\n\n"
            "MIT licence · HR-from-ECG v2")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/hr_model.pt")
    args = ap.parse_args()
    ck = args.checkpoint if Path(args.checkpoint).exists() else None
    HRApp(checkpoint=ck).mainloop()


if __name__ == "__main__":
    main()
