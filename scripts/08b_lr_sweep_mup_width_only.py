"""
08b_lr_sweep_mup_width_only.py
-------------------------------
µP learning rate sweep on the w_xs model (width-only family, d_model=128).

w_xs IS the base model (width_mult=1, base_d_model=128=4*32). The µP sweep
on the base is expected to yield the same optimal LR as the SP sweep — this
is the µP property: the LR found at the base transfers to all larger widths.

Saves:
  outputs/logs/lr_sweep_width_only_mup.json
  outputs/plots/lr_sweep_width_only.png    ← SP vs µP comparison

Usage:
    python scripts/08b_lr_sweep_mup_width_only.py [--max_steps 2000]
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

from src.model_mup import build_mup_model, build_mup_optimizer
from src.dataset import make_datasets
from src.training_utils import evaluate, get_lr_factor, capture_base_lrs, apply_lr

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
    model = build_mup_model("w_xs", config_family="width_only").to(device)
    if hasattr(torch, "compile"):
        model = torch.compile(model)

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    optimizer = build_mup_optimizer(raw_model, lr=lr)

    train_ds, val_ds = make_datasets(
        binary_dir,
        seq_len=seq_len,
        train_samples=max_steps * batch_size,
        val_samples=50 * batch_size,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    train_iter = iter(train_loader)
    t0 = time.time()

    base_lrs = capture_base_lrs(optimizer)

    for step in range(max_steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        factor = get_lr_factor(step, warmup_steps, max_steps)
        apply_lr(optimizer, base_lrs, factor)

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
        "lr": lr,
        "val_loss": float(val_loss),
        "diverged": diverged,
        "wall_sec": round(wall_sec, 1),
        "max_steps": max_steps,
    }


def plot_comparison(sp_path: Path, mup_results: list[dict], save_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))

        lrs_mup = [r["lr"] for r in mup_results]
        vls_mup = [r["val_loss"] for r in mup_results]
        best_mup = min(mup_results, key=lambda r: r["val_loss"])
        ax.plot(lrs_mup, vls_mup, "o--", color="darkorange", label="µP", linewidth=1.5)
        ax.scatter(
            [best_mup["lr"]],
            [best_mup["val_loss"]],
            color="darkorange",
            s=150,
            marker="*",
            zorder=6,
            label=f"µP best: lr={best_mup['lr']:.1e}, val={best_mup['val_loss']:.4f}",
        )

        if sp_path.exists():
            with open(sp_path) as f:
                sp = json.load(f)
            lrs_sp = [r["lr"] for r in sp["runs"]]
            vls_sp = [r["val_loss"] for r in sp["runs"]]
            ax.plot(lrs_sp, vls_sp, "o--", color="steelblue", label="SP", linewidth=1.5)
            ax.scatter(
                [sp["best_lr"]],
                [sp["best_val_loss"]],
                color="steelblue",
                s=150,
                marker="*",
                zorder=6,
                label=f"SP best: lr={sp['best_lr']:.1e}, val={sp['best_val_loss']:.4f}",
            )

        ax.set_xscale("log")
        ax.set_xlabel("Learning rate (log scale)")
        ax.set_ylabel("Final validation loss")
        ax.set_title("LR Sweep Comparison — w_xs (width-only family): SP vs µP")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"LR sweep comparison saved to {save_path}")
    except Exception as e:
        print(f"WARNING: Could not generate comparison plot: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--training_config", default="configs/training_config.yaml")
    parser.add_argument("--data_config", default="configs/data_config.yaml")
    args = parser.parse_args()

    with open(REPO_ROOT / args.training_config) as f:
        tcfg = yaml.safe_load(f)
    with open(REPO_ROOT / args.data_config) as f:
        dcfg = yaml.safe_load(f)

    binary_dir = REPO_ROOT / dcfg["paths"]["binary_dir"]
    log_dir = REPO_ROOT / "outputs" / "logs"
    plots_dir = REPO_ROOT / "outputs" / "plots"
    log_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    lr_values = tcfg["lr_sweep"]["lr_values"]
    max_steps = args.max_steps or tcfg["lr_sweep"]["max_steps"]
    batch_size = args.batch_size or tcfg["batch_size"]
    warmup_steps = tcfg["lr_schedule"]["warmup_steps"]
    use_bf16 = tcfg.get("use_bf16", True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(
        f"µP LR sweep (width_only / w_xs): {len(lr_values)} values × {max_steps} steps each"
    )
    print(f"LRs: {lr_values}\n")

    results = []
    for i, lr in enumerate(lr_values):
        print(f"[{i + 1}/{len(lr_values)}] lr={lr:.1e} ...", end=" ", flush=True)
        result = run_one_lr(
            lr=lr,
            max_steps=max_steps,
            batch_size=batch_size,
            seq_len=1024,
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

    print(
        f"\nBest µP LR (width_only): {best['lr']:.1e}  →  val_loss={best['val_loss']:.4f}"
    )

    sweep_result = {
        "parameterization": "mup",
        "model": "w_xs",
        "config_family": "width_only",
        "max_steps": max_steps,
        "best_lr": best["lr"],
        "best_val_loss": best["val_loss"],
        "runs": results,
    }
    out_path = log_dir / "lr_sweep_width_only_mup.json"
    with open(out_path, "w") as f:
        json.dump(sweep_result, f, indent=2)
    print(f"Results saved to {out_path}")

    plot_comparison(
        sp_path=log_dir / "lr_sweep_width_only_sp.json",
        mup_results=results,
        save_path=plots_dir / "lr_sweep_width_only.png",
    )

    print(f"\n{'LR':>12} {'Val loss':>12} {'Diverged':>10}")
    print("-" * 38)
    for r in results:
        marker = "*" if r["lr"] == best["lr"] else " "
        print(
            f"{r['lr']:>12.1e} {r['val_loss']:>12.4f} {str(r['diverged']):>10} {marker}"
        )


if __name__ == "__main__":
    main()
