"""
06_lr_sweep.py
--------------
Learning rate sweep on the Tiny model (standard parameterization).

Trains Tiny at 7 LR values for a fixed step budget (default: 2000 steps).
Records the final validation loss for each LR.
Saves: outputs/logs/lr_sweep_sp.json
       outputs/plots/lr_sweep_sp.png

Usage:
    python scripts/06_lr_sweep.py [--max_steps 2000] [--batch_size 64]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model import build_model
from src.dataset import make_datasets
from src.training_utils import build_optimizer, evaluate, get_lr

from torch.utils.data import DataLoader


def run_one_lr(
    lr: float,
    max_steps: int,
    batch_size: int,
    seq_len: int,
    binary_dir: Path,
    device: torch.device,
    use_bf16: bool,
    warmup_steps: int,
) -> dict:
    model = build_model("tiny").to(device)
    if hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = build_optimizer(model, lr=lr)

    train_ds, val_ds = make_datasets(
        binary_dir,
        seq_len=seq_len,
        train_samples=max_steps * batch_size,
        val_samples=50 * batch_size,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=(device.type == "cuda"))

    model.train()
    optimizer.zero_grad(set_to_none=True)
    train_iter = iter(train_loader)
    t0         = time.time()

    for step in range(max_steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        current_lr = get_lr(step, lr, warmup_steps, max_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        if use_bf16:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    val_loss = evaluate(model, val_loader, device, use_bf16=use_bf16, max_batches=50)
    wall_sec = time.time() - t0
    diverged = val_loss > 10.0 or not np.isfinite(val_loss)

    return {
        "lr":        lr,
        "val_loss":  float(val_loss),
        "diverged":  diverged,
        "wall_sec":  round(wall_sec, 1),
        "max_steps": max_steps,
    }


def plot_lr_sweep(results: list[dict], save_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt

        lrs       = [r["lr"]       for r in results]
        val_losses = [r["val_loss"] for r in results]
        diverged   = [r["diverged"] for r in results]

        colors = ["red" if d else "steelblue" for d in diverged]
        best_idx = min(range(len(results)), key=lambda i: val_losses[i])

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(lrs, val_losses, c=colors, s=80, zorder=5)
        ax.scatter([lrs[best_idx]], [val_losses[best_idx]],
                   c="gold", s=150, zorder=6, marker="*",
                   label=f"Best: lr={lrs[best_idx]:.1e}, val={val_losses[best_idx]:.4f}")

        ax.plot(lrs, val_losses, color="steelblue", linewidth=1, linestyle="--", alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("Learning rate (log scale)")
        ax.set_ylabel("Final validation loss")
        ax.set_title("LR Sweep — Tiny model (SP)\n(red = diverged)")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"LR sweep plot saved to {save_path}")
    except Exception as e:
        print(f"WARNING: Could not generate LR sweep plot: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps",  type=int,   default=None,
                        help="Steps per LR run (overrides training_config.yaml)")
    parser.add_argument("--batch_size", type=int,   default=None)
    parser.add_argument("--training_config", default="configs/training_config.yaml")
    parser.add_argument("--data_config",     default="configs/data_config.yaml")
    args = parser.parse_args()

    with open(REPO_ROOT / args.training_config) as f:
        tcfg = yaml.safe_load(f)
    with open(REPO_ROOT / args.data_config) as f:
        dcfg = yaml.safe_load(f)

    binary_dir   = REPO_ROOT / dcfg["paths"]["binary_dir"]
    log_dir      = REPO_ROOT / "outputs" / "logs"
    plots_dir    = REPO_ROOT / "outputs" / "plots"
    log_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    lr_values    = tcfg["lr_sweep"]["lr_values"]
    max_steps    = args.max_steps or tcfg["lr_sweep"]["max_steps"]
    batch_size   = args.batch_size or tcfg["batch_size"]
    seq_len      = 1024
    warmup_steps = tcfg["lr_schedule"]["warmup_steps"]
    use_bf16     = tcfg.get("use_bf16", True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"LR sweep: {len(lr_values)} values × {max_steps} steps each")
    print(f"LRs: {lr_values}\n")

    results = []
    for i, lr in enumerate(lr_values):
        print(f"[{i+1}/{len(lr_values)}] lr={lr:.1e} ...", end=" ", flush=True)
        result = run_one_lr(
            lr=lr,
            max_steps=max_steps,
            batch_size=batch_size,
            seq_len=seq_len,
            binary_dir=binary_dir,
            device=device,
            use_bf16=use_bf16,
            warmup_steps=warmup_steps,
        )
        results.append(result)
        status = "DIVERGED" if result["diverged"] else f"val={result['val_loss']:.4f}"
        print(f"{status}  ({result['wall_sec']:.0f}s)")

    results.sort(key=lambda r: r["lr"])

    best = min(results, key=lambda r: r["val_loss"])
    print(f"\nBest LR: {best['lr']:.1e}  →  val_loss={best['val_loss']:.4f}")
    print(f"Set `learning_rate: {best['lr']:.1e}` in configs/training_config.yaml before running script 05.\n")

    sweep_result = {
        "parameterization": "SP",
        "model":            "tiny",
        "max_steps":        max_steps,
        "best_lr":          best["lr"],
        "best_val_loss":    best["val_loss"],
        "runs":             results,
    }
    out_path = log_dir / "lr_sweep_sp.json"
    with open(out_path, "w") as f:
        json.dump(sweep_result, f, indent=2)
    print(f"Results saved to {out_path}")

    plot_lr_sweep(results, plots_dir / "lr_sweep_sp.png")

    print(f"\n{'LR':>12} {'Val loss':>12} {'Diverged':>10}")
    print("-" * 38)
    for r in results:
        marker = "*" if r["lr"] == best["lr"] else " "
        print(f"{r['lr']:>12.1e} {r['val_loss']:>12.4f} {str(r['diverged']):>10} {marker}")


if __name__ == "__main__":
    main()
