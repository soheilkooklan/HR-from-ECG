"""
Train HR-from-ECG v2.

Training uses freshly generated recordings rather than a fixed corpus, so no
noise realisation is ever seen twice. This is not a convenience: the quality
head is being asked to recognise *artefact*, and with a finite corpus it would
instead memorise a finite set of artefact instances. The simulator makes the
effective training-set size unbounded at negligible cost.

Fine-tuning on annotated recordings (MIT-BIH and friends) is supported through
the same interface -- see `--records`, which mixes real waveforms with
synthetic contamination so that morphology comes from patients while the
quality labels remain exactly computable.

Usage
-----
    python scripts/train.py --steps 800 --out checkpoints/hr_model.pt
    python scripts/train.py --steps 5000 --device cuda --batch 32
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from hrecg.data.windows import SyntheticDataset, collate
from hrecg.models import HRLoss, HRModel, ModelConfig
from hrecg.models.losses import LossWeights


def build_batch(ds: SyntheticDataset, idx: np.ndarray) -> dict:
    return collate([ds[int(i)] for i in idx])


def evaluate(model, loss_fn, ds, idx, device) -> dict:
    model.eval()
    tot = {}
    n = 0
    with torch.no_grad():
        for i in range(0, len(idx), 8):
            b = build_batch(ds, idx[i:i + 8])
            x = b["signal"].unsqueeze(1).to(device)
            tgt = {k: v.to(device) for k, v in b.items() if k != "meta"}
            _, parts = loss_fn(model(x), tgt)
            for k, v in parts.items():
                tot[k] = tot.get(k, 0.0) + v
            n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in tot.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--fs", type=float, default=250.0)
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--base-width", type=int, default=16)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--expand", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/hr_model.pt")
    ap.add_argument("--log-every", type=int, default=25)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    cfg = ModelConfig(base_width=args.base_width, depth=args.depth,
                         expand=args.expand)
    model = HRModel(cfg).to(device)
    loss_fn = HRLoss(fs=args.fs, weights=LossWeights(), quantiles=cfg.quantiles)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.15)

    train_ds = SyntheticDataset(10**7, fs=args.fs, window=args.window, seed=args.seed + 1)
    val_ds = SyntheticDataset(10**7, fs=args.fs, window=args.window, seed=args.seed + 999)
    val_idx = np.arange(64)

    print(f"HR-from-ECG v2  |  {model.n_parameters():,} parameters  |  "
          f"fs={args.fs:.0f} Hz  window={args.window} ({args.window/args.fs:.1f} s)")
    print(f"device={device}  steps={args.steps}  batch={args.batch}\n")

    history = []
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    model.train()

    for step in range(1, args.steps + 1):
        idx = rng.integers(0, 10**7, size=args.batch)
        b = build_batch(train_ds, idx)
        x = b["signal"].unsqueeze(1).to(device)
        tgt = {k: v.to(device) for k, v in b.items() if k != "meta"}

        loss, parts = loss_fn(model(x), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % args.log_every == 0 or step == 1:
            el = time.time() - t0
            print(f"step {step:5d}/{args.steps}  loss {parts['total']:.4f}  "
                  f"beat {parts['beat']:.4f}  off {parts['offset']:.4f}  "
                  f"qual {parts['quality']:.4f}  rhy {parts['rhythm']:.4f}  "
                  f"quant {parts['quantile']:.3f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  {el:.0f}s", flush=True)
            history.append(dict(step=step, elapsed_s=el, **parts))

        # Periodic checkpointing: long CPU runs are vulnerable to being killed,
        # and losing an hour of training to an OOM event is avoidable.
        if step % 100 == 0:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(state_dict=model.state_dict(), config=cfg.__dict__,
                            fs=args.fs, window=args.window,
                            train_args=vars(args), history=history,
                            validation={}, step=step), args.out)

    val = evaluate(model, loss_fn, val_ds, val_idx, device)
    print("\nvalidation:", {k: round(v, 4) for k, v in val.items()})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(
        state_dict=model.state_dict(),
        config=cfg.__dict__,
        fs=args.fs, window=args.window,
        train_args=vars(args), history=history, validation=val,
    ), out)
    (out.with_suffix(".json")).write_text(json.dumps(
        dict(history=history, validation=val, config=cfg.__dict__,
             n_parameters=model.n_parameters()), indent=2))
    print(f"saved {out}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
