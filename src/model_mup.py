"""
model_mup.py
------------
µP (Maximal Update Parameterization) version of the decoder-only transformer.

Key differences from model.py (standard parameterization):
  1. Attention scaling: 1/d_head instead of 1/√d_head
  2. Output head replaced with MuSharedReadout (weight-tied, µP-aware)
  3. set_base_shapes() applied before optimizer creation so MuAdamW
     applies per-layer LR multipliers correctly
  4. _init_weights unchanged — mup.set_base_shapes handles init scaling

Usage pattern:
    base   = MupTransformerLM(ModelConfig(d_model=128, ...))
    delta  = MupTransformerLM(ModelConfig(d_model=256, ...))  # any wider width
    make_base_shapes(base, delta, savefile="outputs/base_shapes.bsh")

    model  = MupTransformerLM(ModelConfig(d_model=768, ...))
    set_base_shapes(model, "outputs/base_shapes.bsh")
    opt    = MuAdamW(model.parameters(), lr=best_lr)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mup import MuSharedReadout, set_base_shapes
from mup.optim import MuAdamW

from src.model import ModelConfig, MODEL_CONFIGS, FeedForward, TransformerBlock


# ---------------------------------------------------------------------------
# µP attention: scale by 1/d_head not 1/√d_head
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

        # µP: scale by 1/d_head (not 1/√d_head)
        scale = 1.0 / self.d_head
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

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("net.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.cfg.n_layers))

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

def make_mup_base_shapes(save_path: str) -> None:
    """
    Generate and save base shapes from Tiny (d_model=128) and a delta model
    (d_model=256). Call once before any training. The .bsh file is reused
    for all model sizes.
    """
    from mup import make_base_shapes

    base  = MupTransformerLM(MODEL_CONFIGS["tiny"])             # d_model=128
    delta = MupTransformerLM(ModelConfig(                       # d_model=256, same depth
        d_model=256, n_layers=4, n_heads=4, d_ff=512,
        vocab_size=MODEL_CONFIGS["tiny"].vocab_size,
        max_seq_len=MODEL_CONFIGS["tiny"].max_seq_len,
    ))
    make_base_shapes(base, delta, savefile=save_path)
    print(f"Base shapes saved to {save_path}")


def build_mup_model(name: str, base_shapes_path: str, **overrides) -> MupTransformerLM:
    """Build a µP model, apply base shapes, ready for MuAdamW."""
    import dataclasses
    cfg   = dataclasses.replace(MODEL_CONFIGS[name], **overrides)
    model = MupTransformerLM(cfg)
    set_base_shapes(model, base_shapes_path)
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
