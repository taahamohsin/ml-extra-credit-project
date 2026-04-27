"""
model_mup.py
------------
µP (Maximal Update Parameterization) version of the decoder-only transformer.

Key differences from model.py (standard parameterization):
  1. Attention scaling: √(BASE_HEAD_DIM)/d_head, which matches SP's 1/√d_head
     at the base width and shrinks as 1/d_head for wider targets — the µP rule.
  2. Output head replaced with MuSharedReadout (weight-tied, µP-aware)
  3. set_base_shapes() applied before optimizer creation so MuAdamW
     applies per-layer LR multipliers correctly

Base shapes are built in-memory inside build_mup_model() using a base and delta
model with the same depth/heads as the target but half the width. No .bsh file needed.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mup import MuSharedReadout, set_base_shapes
from mup.init import normal_ as mup_normal_
from mup.optim import MuAdamW

from src.model import ModelConfig, MODEL_CONFIGS, FeedForward, TransformerBlock


# ---------------------------------------------------------------------------
# µP base width — moved above MupCausalSelfAttention because the attention
# scaling formula references BASE_HEAD_DIM.
# ---------------------------------------------------------------------------

BASE_HEAD_DIM = 32   # d_head used in every model's base for set_base_shapes


# ---------------------------------------------------------------------------
# µP attention: 1/d_head behavior (not 1/√d_head), with a constant chosen so
# that the base model has the same attention temperature as standard SP.
# Concretely, µP prescribes scale ∝ 1/d_head; the proportionality constant is
# √(BASE_HEAD_DIM) so that at the base width the scale equals 1/√(d_head_base),
# matching SP. Without this constant, attention is several× colder than SP at
# the base width, which destabilizes training and shifts the LR optimum.
# Reference: microsoft/mup README ("attention_scores = q @ k.T * 8/d", where
# 8 = √64 is chosen for backward-compat with d_head=64). Our base d_head is
# BASE_HEAD_DIM=32, so the corresponding constant is √32.
# ---------------------------------------------------------------------------

class MupCausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads  = cfg.n_heads
        self.d_head   = cfg.d_model // cfg.n_heads
        self.d_model  = cfg.d_model
        self.dropout  = cfg.dropout

        self.qkv_proj  = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=True)
        self.out_proj   = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.attn_drop  = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len)).view(
                1, 1, cfg.max_seq_len, cfg.max_seq_len
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv_proj(x).split(self.d_model, dim=2)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # µP attention scale (see comment above the class). At base width
        # (d_head == BASE_HEAD_DIM) this equals 1/√d_head, matching SP. As
        # d_head grows, scale shrinks as 1/d_head — the µP rule.
        scale = math.sqrt(BASE_HEAD_DIM) / self.d_head
        att = (q @ k.transpose(-2, -1)) * scale
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(y))


class MupTransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.attn = MupCausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.d_model)
        self.ff   = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Full µP model
# ---------------------------------------------------------------------------

class MupTransformerLM(nn.Module):
    """
    Decoder-only transformer with µP parameterization.
    Call set_base_shapes(model, base_shapes_path) before creating the optimizer.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb   = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.emb_drop  = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([MupTransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f   = nn.LayerNorm(cfg.d_model)

        # MuSharedReadout: weight-tied output head, µP-aware
        # output_mult scales the logits; default 1.0 is correct here.
        self.lm_head = MuSharedReadout(self.token_emb.weight, bias=False)

        # NOTE: init is deferred to mup_init() in build_mup_model, called
        # AFTER set_base_shapes attaches infshape attributes to parameters.
        # Calling nn.init here would use SP init and break µP's variance
        # invariants for non-base widths.

    def mup_init(self):
        """µP-aware init. Must be called AFTER set_base_shapes."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # mup_normal_ scales std by 1/sqrt(fan_in_ratio) for hidden weights
                mup_normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # Embedding is a "input" layer in µP — uses standard init (no width scaling)
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # GPT-2 residual-projection scaling (1/sqrt(2*n_layers)) on top of µP
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("net.2.weight"):
                mup_normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.cfg.n_layers))

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len

        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.emb_drop(self.token_emb(idx) + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_id: int = 2,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.max_seq_len else idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[sorted_remove] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)

            if (next_tok == eos_id).all():
                break

        return idx

    def count_parameters(self) -> int:
        emb_params = self.token_emb.weight.numel() + self.pos_emb.weight.numel()
        return sum(p.numel() for p in self.parameters()) - emb_params


# ---------------------------------------------------------------------------
# Base shapes helper
# ---------------------------------------------------------------------------

# µP base width — the "proxy" width that every target is scaled relative to.
# Default base_d_model = n_heads * BASE_HEAD_DIM, so the BASE model has
# d_head = BASE_HEAD_DIM. Wider targets may have larger d_head (e.g. Medium
# with n_heads=6 and d_model=384 has d_head=64). This makes the attention
# head dimension part of the width scaling: the µP attention scale
# √(BASE_HEAD_DIM)/d_head matches SP's 1/√d_head at the base width and
# shrinks as 1/d_head for wider targets — the µP-prescribed deviation.
# (BASE_HEAD_DIM is defined near the top of this file because attention
# scaling needs it.)


def build_mup_model(name: str, base_d_model: Optional[int] = None, **overrides) -> MupTransformerLM:
    """
    Build a µP model and apply base shapes in-memory.

    The base/delta pair shares n_layers and n_heads with the target (mup
    requires identical parameter names across base/target). Default
    base_d_model = n_heads * BASE_HEAD_DIM, so the base has d_head =
    BASE_HEAD_DIM = 32. d_ff scales proportionally with d_model so width
    ratios are non-trivial for the deeper targets.

    width_mult per target with the default base sizing:
      Tiny:   target d=128, base=4*32=128  → 1×  (proxy base — no-op)
      Small:  target d=192, base=6*32=192  → 1×  (also no-op; same d_head=32)
      Medium: target d=384, base=6*32=192  → 2×  (d_head: 32 base → 64 target)
      Large:  target d=512, base=8*32=256  → 2×  (d_head: 32 base → 64 target)
      XL:     target d=768, base=12*32=384 → 2×  (d_head: 32 base → 64 target)

    Callers (e.g. coord check) may override base_d_model to pin a fixed base
    across multiple targets and produce non-trivial width_mult for diagnostics.
    """
    import dataclasses
    from mup import make_base_shapes

    cfg   = dataclasses.replace(MODEL_CONFIGS[name], **overrides)
    model = MupTransformerLM(cfg)

    # Default base_d_model: n_heads * BASE_HEAD_DIM, which keeps d_head fixed
    # at BASE_HEAD_DIM. Callers (e.g. coord check) may override to pin the
    # base across multiple targets so width_mult is non-trivial.
    if base_d_model is None:
        base_d_model = cfg.n_heads * BASE_HEAD_DIM
    # Scale d_ff with d_model so the d_ff/d_model ratio is preserved
    base_d_ff    = cfg.d_ff * base_d_model // cfg.d_model

    base_cfg  = dataclasses.replace(cfg, d_model=base_d_model,     d_ff=base_d_ff)
    delta_cfg = dataclasses.replace(cfg, d_model=base_d_model * 2, d_ff=base_d_ff * 2)

    base  = MupTransformerLM(base_cfg)
    delta = MupTransformerLM(delta_cfg)
    # Capture the shapes dict — without it, set_base_shapes can't tell which
    # dims are infinite (it would otherwise need both base and delta).
    base_shapes = make_base_shapes(base, delta, savefile=None)
    set_base_shapes(model, base_shapes)

    # Init AFTER set_base_shapes so mup_normal_ can read infshape and apply
    # the correct width-dependent variance scaling.
    model.mup_init()
    return model


def build_mup_optimizer(
    model: MupTransformerLM,
    lr: float,
    betas: tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.1,
) -> MuAdamW:
    """MuAdamW with weight decay only on weight matrices (not biases / LN)."""
    decay_params    = []
    no_decay_params = []
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
    return MuAdamW(param_groups, lr=lr, betas=betas)
