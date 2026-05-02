"""
14_width_only_scaling.py
------------------------
Width-only µP comparison: trains all 5 SP and all 5 µP models, fits power
laws, and produces comparison plots.

Reads best LRs from:
  outputs/logs/lr_sweep_width_only_sp.json   (from 06b_lr_sweep_width_only.py)
  outputs/logs/lr_sweep_width_only_mup.json  (from 08b_lr_sweep_mup_width_only.py)

Both LR sweep scripts must be run first. This script then:
  1. Trains all 5 width_only models with SP for 1 epoch.
  2. Trains all 5 width_only models with µP for 1 epoch.
  3. Fits power laws L = a·N^(-α)+c to both result sets.
  4. Plots both fits on one shared log-log graph.
  5. Saves all results, fits, and plots.

Outputs (nothing with width_only in the name overwrites existing Phase 2/3 files):
  outputs/logs/result_sp_<name>_width_only.json   (per SP model)
  outputs/logs/result_mup_<name>_width_only.json  (per µP model)
  outputs/logs/scaling_width_only_sp.json
  outputs/logs/scaling_width_only_mup.json
  outputs/plots/scaling_law_width_only.png
  outputs/plots/lr_sweep_width_only.png            (written by 08b, reproduced here if needed)

Usage:
    python scripts/14_width_only_scaling.py [--lr_sp 3e-4] [--lr_mup 3e-4] [--skip_sp] [--skip_mup]
    python scripts/14_width_only_scaling.py --plot_only   # skip training, replot from saved JSONs
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model import build_model, WIDTH_ONLY_CONFIGS
from src.model_mup import build_mup_model, build_mup_optimizer
from src.dataset import make_datasets
from src.training_utils import (
    build_optimizer, train,
    get_lr_factor, capture_base_lrs, apply_lr,
    evaluate,
)
from src.scaling_law import fit_scaling_law, plot_scaling_law, print_fit_summary, predict

from torch.utils.data import DataLoader

WIDTH_ONLY_MODEL_ORDER = ["w_xs", "w_small", "w_medium", "w_large", "w_xl"]


def count_train_tokens(binary_dir: Path) -> int:
    arr = np.memmap(str(binary_dir / "train.bin"), dtype=np.uint16, mode="r")
    return len(arr)


def load_best_lr(sweep_json: Path, parameterization: str) -> float:
    if not sweep_json.exists():
        raise FileNotFoundError(
            f"{sweep_json} not found. "
            f"Run {'06b_lr_sweep_width_only.py' if parameterization == 'SP' else '08b_lr_sweep_mup_width_only.py'} first."
        )
    with open(sweep_json) as f:
        d = json.load(f)
    lr = d["best_lr"]
    print(f"  Best {parameterization} LR from sweep: {lr:.1e}  (val={d['best_val_loss']:.4f})")
    return lr


def train_sp_model(
    model_name: str,
    peak_lr: float,
    tcfg: dict,
    mcfg: dict,
    binary_dir: Path,
    log_dir: Path,
    ckpt_dir: Path,
    device: torch.device,
) -> dict:
    model = build_model(model_name, config_family="width_only").to(device)
    if hasattr(torch, "compile"):
        model = torch.compile(model)

    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    n_params = raw.count_parameters()
    print(f"  SP {model_name}: {n_params:,} non-emb params")

    effective_batch = tcfg["batch_size"]
    grad_accum      = tcfg.get("grad_accum_steps", 1)
    batch_size      = max(1, effective_batch // grad_accum)
    seq_len         = mcfg.get("max_seq_len", 1024)
    tokens_per_step = effective_batch * seq_len
    n_train_tokens  = count_train_tokens(binary_dir)
    total_steps     = max(1, n_train_tokens // tokens_per_step)

    train_ds, val_ds = make_datasets(
        binary_dir,
        seq_len=seq_len,
        train_samples=total_steps * batch_size,
        val_samples=tcfg.get("val_batches", 50) * batch_size,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=(device.type == "cuda"))

    optimizer = build_optimizer(
        model,
        lr=peak_lr,
        betas=(tcfg["optimizer"]["beta1"], tcfg["optimizer"]["beta2"]),
        weight_decay=tcfg["optimizer"]["weight_decay"],
    )

    train_cfg = {
        "model_name":          f"sp_{model_name}",
        "learning_rate":       peak_lr,
        "warmup_steps":        tcfg["lr_schedule"]["warmup_steps"],
        "total_steps":         total_steps,
        "grad_clip":           tcfg["optimizer"]["grad_clip"],
        "eval_interval":       tcfg["eval_interval"],
        "checkpoint_interval": tcfg["checkpoint_interval"],
        "use_bf16":            tcfg.get("use_bf16", True),
        "grad_accum_steps":    grad_accum,
        "batch_size":          batch_size,
        "seq_len":             seq_len,
    }

    local_ckpt = Path("/tmp/checkpoints_width_only_sp")
    drive_ckpt = ckpt_dir / "sp"
    log_path   = log_dir / f"training_sp_{model_name}_width_only.csv"

    t0 = time.time()
    summary = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        train_cfg=train_cfg,
        local_ckpt_dir=local_ckpt,
        drive_ckpt_dir=drive_ckpt,
        log_path=log_path,
        resume_from=None,
    )
    wall_min = (time.time() - t0) / 60

    result = {
        "model_name":      model_name,
        "parameterization": "SP",
        "config_family":   "width_only",
        "n_params":        n_params,
        "best_val_loss":   summary["best_val_loss"],
        "final_val_loss":  summary.get("final_val_loss", summary["best_val_loss"]),
        "tokens_seen":     summary["tokens_seen"],
        "total_steps":     total_steps,
        "peak_lr":         peak_lr,
        "wall_time_min":   round(wall_min, 1),
    }

    result_path = log_dir / f"result_sp_{model_name}_width_only.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  → SP {model_name}: best_val={summary['best_val_loss']:.4f}  ({wall_min:.1f} min)")
    return result


def train_mup_model(
    model_name: str,
    peak_lr: float,
    tcfg: dict,
    mcfg: dict,
    binary_dir: Path,
    log_dir: Path,
    ckpt_dir: Path,
    device: torch.device,
) -> dict:
    model = build_mup_model(model_name, config_family="width_only").to(device)
    if hasattr(torch, "compile"):
        model = torch.compile(model)

    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    n_params = raw.count_parameters()
    print(f"  µP {model_name}: {n_params:,} non-emb params")

    effective_batch = tcfg["batch_size"]
    grad_accum      = tcfg.get("grad_accum_steps", 1)
    batch_size      = max(1, effective_batch // grad_accum)
    seq_len         = mcfg.get("max_seq_len", 1024)
    tokens_per_step = effective_batch * seq_len
    n_train_tokens  = count_train_tokens(binary_dir)
    total_steps     = max(1, n_train_tokens // tokens_per_step)

    train_ds, val_ds = make_datasets(
        binary_dir,
        seq_len=seq_len,
        train_samples=total_steps * batch_size,
        val_samples=tcfg.get("val_batches", 50) * batch_size,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=(device.type == "cuda"))

    optimizer = build_mup_optimizer(
        raw,
        lr=peak_lr,
        betas=(tcfg["optimizer"]["beta1"], tcfg["optimizer"]["beta2"]),
        weight_decay=tcfg["optimizer"]["weight_decay"],
    )

    train_cfg = {
        "model_name":          f"mup_{model_name}",
        "learning_rate":       peak_lr,
        "warmup_steps":        tcfg["lr_schedule"]["warmup_steps"],
        "total_steps":         total_steps,
        "grad_clip":           tcfg["optimizer"]["grad_clip"],
        "eval_interval":       tcfg["eval_interval"],
        "checkpoint_interval": tcfg["checkpoint_interval"],
        "use_bf16":            tcfg.get("use_bf16", True),
        "grad_accum_steps":    grad_accum,
        "batch_size":          batch_size,
        "seq_len":             seq_len,
    }

    local_ckpt = Path("/tmp/checkpoints_width_only_mup")
    drive_ckpt = ckpt_dir / "mup"
    log_path   = log_dir / f"training_mup_{model_name}_width_only.csv"

    t0 = time.time()
    summary = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        train_cfg=train_cfg,
        local_ckpt_dir=local_ckpt,
        drive_ckpt_dir=drive_ckpt,
        log_path=log_path,
        resume_from=None,
    )
    wall_min = (time.time() - t0) / 60

    result = {
        "model_name":       model_name,
        "parameterization": "mup",
        "config_family":    "width_only",
        "n_params":         n_params,
        "best_val_loss":    summary["best_val_loss"],
        "final_val_loss":   summary.get("final_val_loss", summary["best_val_loss"]),
        "tokens_seen":      summary["tokens_seen"],
        "total_steps":      total_steps,
        "peak_lr":          peak_lr,
        "wall_time_min":    round(wall_min, 1),
    }

    result_path = log_dir / f"result_mup_{model_name}_width_only.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  → µP {model_name}: best_val={summary['best_val_loss']:.4f}  ({wall_min:.1f} min)")
    return result


def plot_combined(
    sp_results:  list[dict],
    mup_results: list[dict],
    plots_dir:   Path,
    log_dir:     Path,
) -> None:
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 6))

        def _do_fit_and_plot(results: list[dict], label: str, color: str) -> dict | None:
            names  = [r["model_name"]   for r in results]
            params = [r["n_params"]     for r in results]
            losses = [r["best_val_loss"] for r in results]
            if len(params) < 2:
                print(f"  Not enough points for {label} fit ({len(params)} models).")
                return None
            fit = fit_scaling_law(params, losses)
            print_fit_summary(fit, label=label)
            plot_scaling_law(
                param_counts=params,
                val_losses=losses,
                fit_result=fit,
                model_names=names,
                label=label,
                color=color,
                ax=ax,
            )
            return fit

        sp_fit  = _do_fit_and_plot(sp_results,  "SP",  "steelblue")
        mup_fit = _do_fit_and_plot(mup_results, "µP", "darkorange")

        ax.set_title(
            "Width-Only Scaling Laws: SP vs µP\n"
            "L = a·N^(−α)+c  |  fixed depth=6, heads=4"
        )
        ax.legend(fontsize=9)
        plt.tight_layout()

        save_path = plots_dir / "scaling_law_width_only.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nScaling law plot saved to {save_path}")

        for fit, label, fname in [
            (sp_fit,  "SP",  "scaling_width_only_sp.json"),
            (mup_fit, "mup", "scaling_width_only_mup.json"),
        ]:
            if fit is None:
                continue
            results = sp_results if label == "SP" else mup_results
            fit_out = {
                "parameterization": label,
                "config_family":    "width_only",
                "models":           [r["model_name"]    for r in results],
                "param_counts":     [r["n_params"]      for r in results],
                "val_losses":       [r["best_val_loss"] for r in results],
                "a":       fit["a"],
                "alpha":   fit["alpha"],
                "c":       fit["c"],
                "r_squared": fit["r_squared"],
            }
            out = log_dir / fname
            with open(out, "w") as f:
                json.dump(fit_out, f, indent=2)
            print(f"Fit JSON saved to {out}")

    except Exception as e:
        print(f"WARNING: Could not generate combined scaling plot: {e}")
        raise


def load_results(log_dir: Path, parameterization: str) -> list[dict]:
    prefix = "sp" if parameterization == "SP" else "mup"
    results = []
    for name in WIDTH_ONLY_MODEL_ORDER:
        p = log_dir / f"result_{prefix}_{name}_width_only.json"
        if p.exists():
            with open(p) as f:
                results.append(json.load(f))
        else:
            print(f"  WARNING: {p.name} not found — skipping {name} from {parameterization} fit")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr_sp",      type=float, default=None,
                        help="Peak SP LR. If omitted, reads from lr_sweep_width_only_sp.json.")
    parser.add_argument("--lr_mup",     type=float, default=None,
                        help="Peak µP LR. If omitted, reads from lr_sweep_width_only_mup.json.")
    parser.add_argument("--skip_sp",    action="store_true",
                        help="Skip SP training (use already-saved result JSONs).")
    parser.add_argument("--skip_mup",   action="store_true",
                        help="Skip µP training (use already-saved result JSONs).")
    parser.add_argument("--plot_only",  action="store_true",
                        help="Skip all training; replot from saved JSONs.")
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--training_config", default="configs/training_config.yaml")
    parser.add_argument("--data_config",     default="configs/data_config.yaml")
    parser.add_argument("--model_config",    default="configs/model_configs.yaml")
    args = parser.parse_args()

    with open(REPO_ROOT / args.training_config) as f:
        tcfg = yaml.safe_load(f)
    with open(REPO_ROOT / args.data_config) as f:
        dcfg = yaml.safe_load(f)
    with open(REPO_ROOT / args.model_config) as f:
        mcfg = yaml.safe_load(f)

    if args.grad_accum is not None:
        tcfg["grad_accum_steps"] = args.grad_accum

    binary_dir = REPO_ROOT / dcfg["paths"]["binary_dir"]
    log_dir    = REPO_ROOT / "outputs" / "logs"
    plots_dir  = REPO_ROOT / "outputs" / "plots"
    ckpt_dir   = REPO_ROOT / "outputs" / "checkpoints_width_only"
    log_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.plot_only:
        print("\n[plot_only] Loading saved results and replotting ...")
        sp_results  = load_results(log_dir, "SP")
        mup_results = load_results(log_dir, "mup")
        plot_combined(sp_results, mup_results, plots_dir, log_dir)
        return

    sp_lr  = args.lr_sp
    mup_lr = args.lr_mup

    if not args.skip_sp and sp_lr is None:
        print("\nReading best SP LR from sweep ...")
        sp_lr = load_best_lr(log_dir / "lr_sweep_width_only_sp.json", "SP")

    if not args.skip_mup and mup_lr is None:
        print("Reading best µP LR from sweep ...")
        mup_lr = load_best_lr(log_dir / "lr_sweep_width_only_mup.json", "mup")

    sp_results: list[dict] = []
    mup_results: list[dict] = []

    if not args.skip_sp:
        print(f"\n{'='*60}")
        print(f"SP training — width_only family — peak LR {sp_lr:.1e}")
        print(f"{'='*60}")
        for name in WIDTH_ONLY_MODEL_ORDER:
            sp_results.append(
                train_sp_model(name, sp_lr, tcfg, mcfg, binary_dir, log_dir, ckpt_dir, device)
            )
    else:
        print("\n[skip_sp] Loading existing SP results ...")
        sp_results = load_results(log_dir, "SP")

    if not args.skip_mup:
        print(f"\n{'='*60}")
        print(f"µP training — width_only family — peak LR {mup_lr:.1e}")
        print(f"  base_d_model=128 (4 heads × 32 BASE_HEAD_DIM) — w_xs is the exact base")
        print(f"{'='*60}")
        for name in WIDTH_ONLY_MODEL_ORDER:
            mup_results.append(
                train_mup_model(name, mup_lr, tcfg, mcfg, binary_dir, log_dir, ckpt_dir, device)
            )
    else:
        print("\n[skip_mup] Loading existing µP results ...")
        mup_results = load_results(log_dir, "mup")

    print(f"\n{'='*70}")
    print(f"{'Model':<12} {'N params':>12} {'SP val':>10} {'µP val':>10} {'Δ (mup-sp)':>12}")
    print(f"{'-'*70}")
    sp_by_name  = {r["model_name"]: r for r in sp_results}
    mup_by_name = {r["model_name"]: r for r in mup_results}
    for name in WIDTH_ONLY_MODEL_ORDER:
        sp  = sp_by_name.get(name)
        mup = mup_by_name.get(name)
        if sp and mup:
            delta = mup["best_val_loss"] - sp["best_val_loss"]
            sign  = "+" if delta >= 0 else ""
            print(
                f"{name:<12} {sp['n_params']:>12,} "
                f"{sp['best_val_loss']:>10.4f} {mup['best_val_loss']:>10.4f} "
                f"{sign}{delta:>10.4f}"
            )
    print(f"{'='*70}")

    print("\nFitting power laws and generating plots ...")
    plot_combined(sp_results, mup_results, plots_dir, log_dir)

    print("\nDone. All width_only outputs written to outputs/logs/ and outputs/plots/.")


if __name__ == "__main__":
    main()
