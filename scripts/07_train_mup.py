"""
07_train_mup.py
---------------
Train a single µP transformer for exactly 1 epoch on the SVG dataset.
Mirrors 05_train_model.py but uses MupTransformerLM + MuAdamW.
Base shapes are built in-memory per model — no .bsh file needed.

Local-first I/O: all heavy writes go to /tmp/, copied to Drive at checkpoints.

Usage:
    python scripts/07_train_mup.py --model_name xl --lr 1e-3 --grad_accum 2

Note: --resume is disabled for µP runs after the attention-scaling patch.
Pre-patch checkpoints used a different attention scale and are incompatible.
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

from src.model import MODEL_CONFIGS
from src.model_mup import build_mup_model, build_mup_optimizer
from src.dataset import make_datasets
from src.training_utils import train

from torch.utils.data import DataLoader


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
    parser.add_argument("--model_name",  default="tiny",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--lr",          type=float, default=None)
    parser.add_argument("--batch_size",  type=int,   default=None)
    parser.add_argument("--grad_accum",  type=int,   default=None,
                        help="Gradient accumulation steps. Per-step batch size "
                             "is divided accordingly; effective batch unchanged.")
    parser.add_argument("--resume",      action="store_true")
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

    binary_dir     = REPO_ROOT / dcfg["paths"]["binary_dir"]
    log_dir        = REPO_ROOT / "outputs" / "logs"
    local_ckpt_dir = Path("/tmp/checkpoints_mup_local")
    drive_ckpt_dir = REPO_ROOT / "outputs" / "checkpoints_mup"

    # Fail fast on --resume (disabled for µP after the attention-scaling patch).
    # Pre-patch checkpoints used bare 1/d_head attention scale and are not
    # compatible with the current model. Doing this before model/data loading
    # so the user doesn't wait 30s before the error.
    if args.resume:
        raise RuntimeError(
            "--resume is disabled for µP runs after the attention-scaling patch. "
            "Pre-patch µP checkpoints used bare 1/d_head attention scale, which "
            "is incompatible with the current model. Delete old µP checkpoints "
            "and result JSONs and rerun from scratch."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_name = args.model_name
    model = build_mup_model(model_name)
    model = model.to(device)

    if hasattr(torch, "compile"):
        print("Compiling model with torch.compile ...")
        model = torch.compile(model)

    n_params = (model._orig_mod if hasattr(model, "_orig_mod") else model).count_parameters()
    print(f"µP Model: {model_name}  |  Non-emb params: {n_params:,}")

    n_train_tokens  = count_train_tokens(binary_dir)
    effective_batch = args.batch_size or tcfg["batch_size"]
    grad_accum      = args.grad_accum or tcfg.get("grad_accum_steps", 1)
    batch_size      = max(1, effective_batch // grad_accum)
    seq_len         = mcfg.get("max_seq_len", 1024)
    tokens_per_step = effective_batch * seq_len
    total_steps     = max(1, n_train_tokens // tokens_per_step)

    print(f"Train tokens: {n_train_tokens:,}  |  steps/epoch: {total_steps:,}")

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

    peak_lr   = args.lr or tcfg["learning_rate"]
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    optimizer = build_mup_optimizer(
        raw_model,
        lr=peak_lr,
        betas=(tcfg["optimizer"]["beta1"], tcfg["optimizer"]["beta2"]),
        weight_decay=tcfg["optimizer"]["weight_decay"],
    )

    resume_from = None

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

    log_path = log_dir / f"training_mup_{model_name}.csv"

    print("\n" + "=" * 60)
    print(f"µP Training: {model_name.upper()}")
    print("=" * 60)
    print(f"  Non-emb params:  {n_params:,}")
    print(f"  Peak LR:         {peak_lr:.2e}")
    print(f"  Batch size:      {batch_size} seqs/step × {grad_accum} accum = {effective_batch} seqs effective")
    print(f"  Total steps:     {total_steps:,}")
    print(f"  Base shapes:     in-memory (per-model)")
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

    result = {
        "model_name":       model_name,
        "parameterization": "mup",
        "n_params":         n_params,
        "final_val_loss":   summary["final_val_loss"],
        "best_val_loss":    summary["best_val_loss"],
        "tokens_seen":      summary["tokens_seen"],
        "total_steps":      total_steps,
        "peak_lr":          peak_lr,
        "wall_time_min":    round(wall_min, 1),
    }
    results_dir = REPO_ROOT / "outputs" / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"result_mup_{model_name}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 60)
    print(f"µP TRAINING COMPLETE — {model_name.upper()}")
    print(f"  Val loss (final, 1 epoch): {summary['final_val_loss']:.4f}")
    print(f"  Val loss (best):           {summary['best_val_loss']:.4f}")
    print(f"  Wall time:       {wall_min:.1f} min")
    if device.type == "cuda":
        print(f"  Peak GPU mem:    {torch.cuda.max_memory_allocated(device)/1e9:.2f} GB")
    print(f"  Result saved to: {result_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
