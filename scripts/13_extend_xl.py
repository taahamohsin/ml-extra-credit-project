"""
13_extend_xl.py
---------------
Continue training the SP XL model for 3 additional epochs from best.pt,
with a fresh cosine-warmup schedule over the new step budget.

The original LR (3e-4) and all other hyperparameters are preserved.
Checkpoints are saved under outputs/checkpoints/xl_extended/ so the
original xl/ checkpoints are untouched.

Epoch length is recomputed from train.bin at runtime (same formula as
05_train_model.py: train_tokens // (batch_size * seq_len)).

Usage:
    python scripts/13_extend_xl.py
    python scripts/13_extend_xl.py --extra_epochs 3 --lr 3e-4 --grad_accum 2
"""

from __future__ import annotations

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

from src.model import TransformerLM
from src.dataset import make_datasets
from src.training_utils import build_optimizer, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_checkpoint", default="outputs/checkpoints/xl/best.pt",
                    help="Checkpoint to resume from (weights only — schedule resets)")
    ap.add_argument("--extra_epochs", type=int, default=3,
                    help="Number of additional epochs to train")
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="Peak LR for the extended run")
    ap.add_argument("--grad_accum", type=int, default=2,
                    help="Gradient accumulation steps (use 2 to halve per-step memory)")
    ap.add_argument("--training_config", default="configs/training_config.yaml")
    ap.add_argument("--data_config",     default="configs/data_config.yaml")
    ap.add_argument("--model_config",    default="configs/model_configs.yaml")
    args = ap.parse_args()

    with open(REPO_ROOT / args.training_config) as f:
        tcfg = yaml.safe_load(f)
    with open(REPO_ROOT / args.data_config) as f:
        dcfg = yaml.safe_load(f)
    with open(REPO_ROOT / args.model_config) as f:
        mcfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    src_path = REPO_ROOT / args.source_checkpoint
    if not src_path.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {src_path}")

    ckpt = torch.load(src_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]

    model = TransformerLM(cfg)
    state = ckpt["model"]
    cleaned = {
        (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
        for k, v in state.items()
    }
    model.load_state_dict(cleaned)
    model = model.to(device)

    n_params = model.count_parameters()
    print(f"Loaded: {src_path}")
    print(f"  Non-emb params: {n_params:,}")
    print(f"  Source step:    {ckpt.get('step', '?')}")
    print(f"  Source val:     {ckpt.get('best_val_loss', float('nan')):.4f}")

    binary_dir = REPO_ROOT / dcfg["paths"]["binary_dir"]
    train_tokens = len(np.memmap(str(binary_dir / "train.bin"), dtype=np.uint16, mode="r"))

    effective_batch = tcfg["batch_size"]           # 64 seqs/step
    grad_accum      = args.grad_accum
    batch_size      = max(1, effective_batch // grad_accum)
    seq_len         = mcfg.get("max_seq_len", 1024)
    tokens_per_step = effective_batch * seq_len    # 65,536

    steps_per_epoch = max(1, train_tokens // tokens_per_step)
    total_steps     = steps_per_epoch * args.extra_epochs

    print(f"\nExtended training plan:")
    print(f"  Train tokens:    {train_tokens:,}")
    print(f"  Steps/epoch:     {steps_per_epoch:,}")
    print(f"  Extra epochs:    {args.extra_epochs}")
    print(f"  Total new steps: {total_steps:,}")
    print(f"  Effective batch: {effective_batch} seqs × {seq_len} tokens = {tokens_per_step:,} tok/step")
    print(f"  Grad accum:      {grad_accum}  (per-step batch = {batch_size})")
    print(f"  Peak LR:         {args.lr:.2e}")

    train_samples = total_steps * batch_size
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

    optimizer = build_optimizer(
        model,
        lr=args.lr,
        betas=(tcfg["optimizer"]["beta1"], tcfg["optimizer"]["beta2"]),
        weight_decay=tcfg["optimizer"]["weight_decay"],
    )

    model_name      = "xl_extended"
    local_ckpt_dir  = Path("/tmp/checkpoints_local")
    drive_ckpt_dir  = REPO_ROOT / "outputs" / "checkpoints"
    log_dir         = REPO_ROOT / "outputs" / "logs"
    log_path        = log_dir / f"training_{model_name}.csv"

    train_cfg = {
        "model_name":          model_name,
        "learning_rate":       args.lr,
        "warmup_steps":        min(200, total_steps // 10),
        "total_steps":         total_steps,
        "grad_clip":           tcfg["optimizer"]["grad_clip"],
        "eval_interval":       tcfg["eval_interval"],
        "checkpoint_interval": tcfg["checkpoint_interval"],
        "use_bf16":            tcfg.get("use_bf16", True),
        "grad_accum_steps":    grad_accum,
        "batch_size":          batch_size,
        "seq_len":             seq_len,
    }

    print("\n" + "=" * 60)
    print(f"Training: {model_name.upper()}")
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
        resume_from=None,   # schedule resets — weights already loaded above
    )

    wall_min = (time.time() - t_start) / 60

    log_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "model_name":       model_name,
        "source_checkpoint": args.source_checkpoint,
        "source_val_loss":  float(ckpt.get("best_val_loss", float("nan"))),
        "n_params":         n_params,
        "extra_epochs":     args.extra_epochs,
        "total_new_steps":  total_steps,
        "final_val_loss":   summary.get("final_val_loss", summary["best_val_loss"]),
        "best_val_loss":    summary["best_val_loss"],
        "tokens_seen":      summary["tokens_seen"],
        "peak_lr":          args.lr,
        "wall_time_min":    round(wall_min, 1),
    }
    result_path = log_dir / f"result_{model_name}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved to {result_path}")

    print("\n" + "=" * 60)
    print(f"EXTENDED TRAINING COMPLETE")
    print(f"  Source val loss: {result['source_val_loss']:.4f}")
    print(f"  Best val loss:   {summary['best_val_loss']:.4f}")
    print(f"  Final val loss:  {summary.get('final_val_loss', summary['best_val_loss']):.4f}")
    print(f"  Tokens seen:     {summary['tokens_seen']:,}")
    print(f"  Wall time:       {wall_min:.1f} min")
    print(f"  Best checkpoint: {drive_ckpt_dir}/{model_name}/best.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
