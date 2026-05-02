"""
05_train_model.py
-----------------
Train a single transformer model for 1 epoch on the SVG dataset.
Checkpoints are written to /tmp/ first, then copied to Drive.

Usage:
    python scripts/05_train_model.py --model_name tiny [--lr 1e-3] [--batch_size 64] [--resume]
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

from src.model import build_model, MODEL_CONFIGS, WIDTH_ONLY_CONFIGS, ALL_CONFIGS
from src.dataset import make_datasets
from src.training_utils import build_optimizer, train


def count_train_tokens(binary_dir: Path) -> int:
    arr = np.memmap(str(binary_dir / "train.bin"), dtype=np.uint16, mode="r")
    return len(arr)


def find_latest_checkpoint(ckpt_dir: Path, model_name: str) -> Path | None:
    model_ckpt_dir = ckpt_dir / model_name
    if not model_ckpt_dir.exists():
        return None
    ckpts = sorted(model_ckpt_dir.glob("step_*.pt"))
    return ckpts[-1] if ckpts else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_family", default="default",
                        choices=["default", "width_only"],
                        help="Model config family: 'default' (tiny…xl) or 'width_only' (w_xs…w_xl)")
    parser.add_argument("--model_name",  default="tiny",
                        choices=list(ALL_CONFIGS.keys()))
    parser.add_argument("--lr",          type=float, default=None,
                        help="Peak learning rate (overrides training_config.yaml)")
    parser.add_argument("--batch_size",  type=int,   default=None)
    parser.add_argument("--grad_accum",  type=int,   default=None,
                        help="Gradient accumulation steps. Per-step batch size is "
                             "halved accordingly; effective batch size is unchanged.")
    parser.add_argument("--resume",      action="store_true",
                        help="Resume from last local checkpoint")
    parser.add_argument("--epochs",      type=int, default=1,
                        help="Number of epochs to train (default 1). "
                             "Steps = epochs × (train_tokens // tokens_per_step).")
    parser.add_argument("--ckpt_dir",    default=None,
                        help="Override checkpoint output dir (default: outputs/checkpoints)")
    parser.add_argument("--result_suffix", default="",
                        help="Suffix appended to result_<model><suffix>.json so balanced "
                             "runs don't overwrite Phase 2/3 results.")
    parser.add_argument("--model_config",    default="configs/model_configs.yaml")
    parser.add_argument("--training_config", default="configs/training_config.yaml")
    parser.add_argument("--data_config",     default="configs/data_config.yaml")
    args = parser.parse_args()

    with open(REPO_ROOT / args.model_config) as f:
        mcfg = yaml.safe_load(f)
    with open(REPO_ROOT / args.training_config) as f:
        tcfg = yaml.safe_load(f)
    with open(REPO_ROOT / args.data_config) as f:
        dcfg = yaml.safe_load(f)

    binary_dir  = REPO_ROOT / dcfg["paths"]["binary_dir"]
    log_dir     = REPO_ROOT / "outputs" / "logs"
    local_ckpt_dir = Path("/tmp/checkpoints_local")
    drive_ckpt_dir = (
        REPO_ROOT / args.ckpt_dir if args.ckpt_dir
        else REPO_ROOT / "outputs" / "checkpoints"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_name = args.model_name
    model = build_model(model_name, config_family=args.config_family)
    model = model.to(device)

    if hasattr(torch, "compile"):
        print("Compiling model with torch.compile ...")
        model = torch.compile(model)

    n_params = (model._orig_mod if hasattr(model, "_orig_mod") else model).count_parameters()
    print(f"Model: {model_name}  |  Non-emb params: {n_params:,}")

    n_train_tokens  = count_train_tokens(binary_dir)
    effective_batch = args.batch_size or tcfg["batch_size"]  # total seqs/step
    grad_accum      = args.grad_accum or tcfg.get("grad_accum_steps", 1)
    batch_size      = max(1, effective_batch // grad_accum)
    seq_len         = mcfg.get("max_seq_len", 1024)
    tokens_per_step = effective_batch * seq_len
    steps_per_epoch = max(1, n_train_tokens // tokens_per_step)
    total_steps     = steps_per_epoch * args.epochs

    print(f"Train tokens: {n_train_tokens:,}  |  steps/epoch: {steps_per_epoch:,}  |  epochs: {args.epochs}  |  total steps: {total_steps:,}")

    train_samples = steps_per_epoch * batch_size  # 1 epoch worth; DataLoader loops internally
    train_ds, val_ds = make_datasets(
        binary_dir,
        seq_len=seq_len,
        train_samples=train_samples,
        val_samples=tcfg.get("val_batches", 50) * batch_size,
    )

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=(device.type == "cuda"))

    peak_lr = args.lr or tcfg["learning_rate"]
    optimizer = build_optimizer(
        model,
        lr=peak_lr,
        betas=(tcfg["optimizer"]["beta1"], tcfg["optimizer"]["beta2"]),
        weight_decay=tcfg["optimizer"]["weight_decay"],
    )

    resume_from = None
    if args.resume:
        resume_from = find_latest_checkpoint(local_ckpt_dir, model_name)
        if resume_from is None:
            resume_from = find_latest_checkpoint(drive_ckpt_dir, model_name)
        if resume_from:
            print(f"Will resume from: {resume_from}")
        else:
            print("No checkpoint found — starting from scratch.")

    train_cfg = {
        "model_name":          model_name,
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

    log_path = log_dir / f"training_{model_name}.csv"

    print("\n" + "=" * 60)
    print(f"Training: {model_name.upper()}")
    print("=" * 60)
    print(f"  Non-emb params:  {n_params:,}")
    print(f"  Peak LR:         {peak_lr:.2e}")
    print(f"  Batch size:      {batch_size} seqs/step × {grad_accum} accum = {effective_batch} seqs effective")
    print(f"  Tokens/step:     {tokens_per_step:,}  (seq_len={seq_len})")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Steps/epoch:     {steps_per_epoch:,}")
    print(f"  Total steps:     {total_steps:,}")
    print(f"  bf16:            {train_cfg['use_bf16']}")
    print(f"  Local ckpts:     {local_ckpt_dir}/{model_name}/")
    print(f"  Drive ckpts:     {drive_ckpt_dir}/{model_name}/")
    print(f"  Log:             {log_path}")
    print("=" * 60)

    t_start = time.time()
    summary = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        train_cfg=train_cfg,
        local_ckpt_dir=local_ckpt_dir,
        drive_ckpt_dir=drive_ckpt_dir,
        log_path=log_path,
        resume_from=resume_from,
    )

    wall_min = (time.time() - t_start) / 60

    results_dir = REPO_ROOT / "outputs" / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "model_name":      model_name,
        "n_params":        n_params,
        "epochs":          args.epochs,
        "final_val_loss":  summary.get("final_val_loss", summary["best_val_loss"]),
        "best_val_loss":   summary["best_val_loss"],
        "tokens_seen":     summary["tokens_seen"],
        "total_steps":     total_steps,
        "peak_lr":         peak_lr,
        "wall_time_min":   round(wall_min, 1),
    }

    result_path = results_dir / f"result_{model_name}{args.result_suffix}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved to {result_path}")

    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE — {model_name.upper()} ({args.epochs} epoch(s))")
    print(f"  Val loss (final):  {summary.get('final_val_loss', summary['best_val_loss']):.4f}")
    print(f"  Val loss (best):           {summary['best_val_loss']:.4f}")
    print(f"  Tokens seen:     {summary['tokens_seen']:,}")
    print(f"  Wall time:       {wall_min:.1f} min")
    if device.type == "cuda":
        mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"  Peak GPU mem:    {mem_gb:.2f} GB")
    print("=" * 60)


if __name__ == "__main__":
    main()
