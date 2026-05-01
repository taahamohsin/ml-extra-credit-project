"""
training_utils.py
-----------------
Training loop, LR schedule, CSV logging, and checkpoint management.

LR schedule: linear warmup for `warmup_steps`, then cosine decay to
             min_lr = peak_lr / 10 over the remaining steps.

Checkpoint format (saved with torch.save):
  {
    "step":       int,
    "model":      state_dict,
    "optimizer":  state_dict,
    "config":     ModelConfig,
    "train_cfg":  dict,
    "best_val_loss": float,
    "tokens_seen": int,
  }
"""

from __future__ import annotations

import csv
import math
import os
import shutil
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def get_lr(
    step: int,
    peak_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """Cosine decay with linear warmup. Returns an absolute LR.

    Prefer get_lr_factor + apply_lr in new code: assigning a single scalar to
    every param group erases per-group LR multipliers, which breaks µP
    (MuAdamW stores per-parameter-group multipliers in pg["lr"]).
    """
    return peak_lr * get_lr_factor(step, warmup_steps, total_steps, min_lr_ratio)


def get_lr_factor(
    step: int,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """Cosine-with-warmup as a unitless factor in [min_lr_ratio, 1].

    Multiply each param group's stored base LR by this factor each step —
    this preserves µP's per-group LR multipliers."""
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    if step >= total_steps:
        return min_lr_ratio
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))


def capture_base_lrs(optimizer) -> list:
    """Snapshot each param group's LR — call once after optimizer construction,
    before the training loop starts mutating pg["lr"]."""
    return [pg["lr"] for pg in optimizer.param_groups]


def apply_lr(optimizer, base_lrs: list, factor: float) -> None:
    """Set each pg["lr"] to base_lr * factor. Works for both AdamW (all groups
    share the same base, so behaves identically to the old scalar assignment)
    and MuAdamW (per-group bases preserved)."""
    for pg, blr in zip(optimizer.param_groups, base_lrs):
        pg["lr"] = blr * factor


def build_optimizer(
    model: nn.Module,
    lr: float,
    betas: tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.1,
) -> torch.optim.AdamW:
    """
    AdamW with weight decay applied only to weight tensors (not biases / LN).
    """
    decay_params     = []
    no_decay_params  = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith(".bias"):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    param_groups = [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=lr, betas=betas)


class CSVLogger:
    """Appends one row per step to a CSV file."""

    FIELDS = [
        "step", "train_loss", "val_loss", "learning_rate",
        "tokens_seen", "wall_time_sec", "gpu_memory_mb",
    ]

    def __init__(self, path: Path):
        self.path   = path
        self._fresh = not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **kwargs):
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            if self._fresh:
                writer.writeheader()
                self._fresh = False
            row = {k: kwargs.get(k, "") for k in self.FIELDS}
            writer.writerow(row)


def save_checkpoint(
    path: Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_cfg: dict,
    best_val_loss: float,
    tokens_seen: int,
    base_lrs: Optional[list] = None,
    is_best: bool = False,
    drive_path: Optional[Path] = None,
) -> None:
    """Save checkpoint locally, then optionally copy to Drive.

    base_lrs: per-group LRs as captured at training start, before the schedule
    starts mutating pg["lr"]. Required to resume µP training correctly —
    capturing post-resume would snapshot a scheduled value, not the µP base."""
    path.parent.mkdir(parents=True, exist_ok=True)

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    ckpt = {
        "step":          step,
        "model":         raw_model.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "config":        raw_model.cfg,
        "train_cfg":     train_cfg,
        "best_val_loss": best_val_loss,
        "tokens_seen":   tokens_seen,
        "base_lrs":      base_lrs,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)

    if is_best:
        best_path = path.parent / "best.pt"
        shutil.copy2(path, best_path)

    if drive_path is not None:
        drive_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, drive_path)
        if is_best:
            shutil.copy2(path, drive_path.parent / "best.pt")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> dict:
    """Load checkpoint in-place. Returns the full checkpoint dict."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw_model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    use_bf16: bool = True,
    max_batches: int = 50,
) -> float:
    """Compute mean cross-entropy loss over up to max_batches val batches."""
    model.eval()
    ctx = torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16) \
          if use_bf16 and device.type == "cuda" else torch.no_grad()

    total_loss = 0.0
    n = 0
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with ctx:
            _, loss = model(x, y)
        total_loss += loss.item()
        n += 1

    model.train()
    return total_loss / max(n, 1)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_cfg: dict,
    local_ckpt_dir: Path,
    drive_ckpt_dir: Optional[Path] = None,
    log_path: Optional[Path] = None,
    resume_from: Optional[Path] = None,
) -> dict:
    """
    Full training loop.

    Returns a summary dict:
      {step, train_loss, best_val_loss, tokens_seen, wall_time_sec}
    """
    peak_lr           = train_cfg["learning_rate"]
    warmup_steps      = train_cfg.get("warmup_steps", 200)
    grad_clip         = train_cfg.get("grad_clip", 1.0)
    eval_interval     = train_cfg.get("eval_interval", 200)
    ckpt_interval     = train_cfg.get("checkpoint_interval", 500)
    total_steps       = train_cfg["total_steps"]
    use_bf16          = train_cfg.get("use_bf16", True) and device.type == "cuda"
    grad_accum_steps  = train_cfg.get("grad_accum_steps", 1)
    model_name        = train_cfg.get("model_name", "model")

    logger = CSVLogger(log_path) if log_path is not None else None

    # Snapshot per-group base LRs BEFORE resume — at this point the optimizer
    # holds its construction-time LRs (which include MuAdamW's per-group µP
    # divisions). Capturing after resume would snapshot a post-schedule value
    # because load_checkpoint restores pg["lr"] to whatever step N's schedule
    # had set it to, not the original base.
    base_lrs = capture_base_lrs(optimizer)

    # Resume
    start_step     = 0
    tokens_seen    = 0
    best_val_loss  = float("inf")
    final_val_loss = float("nan")  # val loss at the most recent eval; "final" once training ends

    if resume_from is not None and resume_from.exists():
        print(f"Resuming from {resume_from} ...")
        ckpt = load_checkpoint(resume_from, model, optimizer, device=str(device))
        start_step    = ckpt["step"] + 1
        tokens_seen   = ckpt.get("tokens_seen", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        # Prefer checkpointed base_lrs if present (correct for µP); fall back
        # to the pre-resume capture above for backward-compatibility with
        # checkpoints written before this field was added.
        ckpt_base_lrs = ckpt.get("base_lrs")
        if ckpt_base_lrs is not None:
            base_lrs = ckpt_base_lrs
        print(f"  Resumed at step {start_step}, best_val_loss={best_val_loss:.4f}")

    model.train()

    t0           = time.time()
    last_log_t   = t0
    train_iter   = iter(train_loader)
    step         = start_step
    micro_step   = 0
    accum_loss   = 0.0
    train_loss   = float("nan")  # populated by the first optimizer step; nan if loop is empty (resume past end)

    print(f"\nTraining {model_name} for {total_steps} steps "
          f"(bf16={use_bf16}, grad_accum={grad_accum_steps})")

    while step < total_steps:
        # Fetch next batch
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        factor = get_lr_factor(step, warmup_steps, total_steps)
        apply_lr(optimizer, base_lrs, factor)
        lr = base_lrs[0] * factor  # for logging only

        if use_bf16:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)

        loss = loss / grad_accum_steps
        loss.backward()
        accum_loss += loss.item()
        micro_step += 1
        tokens_seen += x.numel()

        if micro_step < grad_accum_steps:
            continue

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        train_loss = accum_loss  # already divided by grad_accum_steps above
        accum_loss = 0.0
        micro_step = 0

        # Validation
        val_loss = None
        if step % eval_interval == 0 or step == total_steps - 1:
            val_loss = evaluate(model, val_loader, device, use_bf16=use_bf16)
            final_val_loss = val_loss
            is_best  = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            gpu_mem_mb = (
                torch.cuda.memory_allocated(device) / 1e6
                if device.type == "cuda" else 0.0
            )
            wall_time = time.time() - t0

            print(
                f"step {step:>6} | train {train_loss:.4f} | "
                f"val {val_loss:.4f} {'*' if is_best else ' '} | "
                f"lr {lr:.2e} | tok {tokens_seen/1e6:.1f}M | "
                f"t {wall_time/60:.1f}min | mem {gpu_mem_mb:.0f}MB"
            )

            if logger:
                logger.log(
                    step=step,
                    train_loss=round(train_loss, 6),
                    val_loss=round(val_loss, 6),
                    learning_rate=round(lr, 8),
                    tokens_seen=tokens_seen,
                    wall_time_sec=round(wall_time, 1),
                    gpu_memory_mb=round(gpu_mem_mb, 1),
                )

            if step % ckpt_interval == 0 or is_best or step == total_steps - 1:
                ckpt_name = f"step_{step:07d}.pt"
                local_path = local_ckpt_dir / model_name / ckpt_name
                drive_path = (
                    drive_ckpt_dir / model_name / ckpt_name
                    if drive_ckpt_dir else None
                )
                save_checkpoint(
                    path=local_path,
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    train_cfg=train_cfg,
                    best_val_loss=best_val_loss,
                    tokens_seen=tokens_seen,
                    base_lrs=base_lrs,
                    is_best=is_best,
                    drive_path=drive_path,
                )
        else:
            if step % 50 == 0:
                wall_time = time.time() - t0
                print(
                    f"step {step:>6} | train {train_loss:.4f} | "
                    f"lr {lr:.2e} | tok {tokens_seen/1e6:.1f}M | "
                    f"t {wall_time/60:.1f}min"
                )
                if logger:
                    logger.log(
                        step=step,
                        train_loss=round(train_loss, 6),
                        learning_rate=round(lr, 8),
                        tokens_seen=tokens_seen,
                        wall_time_sec=round(time.time() - t0, 1),
                    )

        step += 1

    wall_time = time.time() - t0
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}  "
          f"Total time: {wall_time/60:.1f} min")

    return {
        "step":           step - 1,
        "train_loss":     train_loss,
        "best_val_loss":  best_val_loss,
        "final_val_loss": final_val_loss,
        "tokens_seen":    tokens_seen,
        "wall_time_sec":  wall_time,
    }
