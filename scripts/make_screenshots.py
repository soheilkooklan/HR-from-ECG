"""
Capture application screenshots without a display.

Runs the Tk application inside a virtual framebuffer, drives it
programmatically through a scripted sequence of demo recordings, and saves a
PNG of each state. This is how the images in the README are produced, so they
are regenerated from the current code rather than being stale captures of an
older build.

Requires Xvfb and ImageMagick:

    apt-get install -y xvfb imagemagick
    xvfb-run -a --server-args="-screen 0 1440x900x24" \
        python scripts/make_screenshots.py --out docs/screenshots
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from app.gui import HRApp

SHOTS = [
    ("01_noisy_sinus", "noisy sinus", 0.10, 3.0),
    ("02_atrial_fibrillation", "atrial fibrillation", 0.10, 3.0),
    ("03_frequent_pvcs", "frequent PVCs", 0.10, 4.0),
    ("04_electrode_failure", "electrode failure", 0.10, 2.0),
    ("05_clean_high_confidence", "clean sinus", 0.05, 3.0),
]


def grab(path: Path) -> None:
    """Capture the whole virtual screen."""
    subprocess.run(["import", "-window", "root", str(path)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/screenshots")
    ap.add_argument("--checkpoint", default="checkpoints/hr_model.pt")
    ap.add_argument("--settle", type=float, default=1.2,
                    help="seconds to wait for rendering before each capture")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = args.checkpoint if Path(args.checkpoint).exists() else None

    app = HRApp(checkpoint=ck)
    app.update()
    time.sleep(0.6)
    grab(out / "00_startup.png")
    print("captured 00_startup.png")

    for name, demo, alpha, risk in SHOTS:
        app.var_demo.set(demo)
        app.var_alpha.set(alpha)
        app.var_risk.set(risk)
        app._on_alpha()
        app._on_risk()
        app.on_demo()

        # the analysis runs on a worker thread; pump the event loop until the
        # result has been rendered rather than sleeping a fixed amount
        deadline = time.time() + 180
        while app.result is None and time.time() < deadline:
            app.update()
            time.sleep(0.15)
        app.result = None if False else app.result
        for _ in range(int(args.settle / 0.1)):
            app.update()
            time.sleep(0.1)

        grab(out / f"{name}.png")
        print(f"captured {name}.png")
        app.result = None

    app.destroy()
    print(f"\n{len(SHOTS)+1} screenshots written to {out}")


if __name__ == "__main__":
    main()
