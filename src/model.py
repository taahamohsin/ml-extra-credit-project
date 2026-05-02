"""
model.py
--------
Decoder-only Transformer (GPT-2 / nanoGPT style) for SVG generation.

Architecture:
  - Pre-norm (LayerNorm before attention and FFN)
  - Causal multi-head self-attention with learned positional embeddings
  - GELU activation in FFN
  - Weight-tied token embedding and output projection
  - 5 predefined configs: tiny / small / medium / large / xl
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size:   int   = 4096
    d_model:      int   = 128
    n_layers:     int   = 4
    n_heads:      int   = 4
    d_ff:         int   = 512
    max_seq_len:  int   = 1024
    dropout:      float = 0.0

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "tiny":   ModelConfig(d_model=128,  n_layers=4,  n_heads=4,  d_ff=512),
    "small":  ModelConfig(d_model=192,  n_layers=6,  n_heads=6,  d_ff=768),
    "medium": ModelConfig(d_model=384,  n_layers=6,  n_heads=6,  d_ff=1536),
    "large":  ModelConfig(d_model=512,  n_layers=10, n_heads=8,  d_ff=2048),
    "xl":     ModelConfig(d_model=768,  n_layers=12, n_heads=12, d_ff=3072),
}

# Width-only family: depth and heads are fixed (n_layers=6, n_heads=4) so that
# only d_model and d_ff scale. This is the clean confound-free setup for µP
# comparison: width_mult = d_model / 128 is unambiguous for every model.
WIDTH_ONLY_CONFIGS: dict[str, ModelConfig] = {
    "w_xs":     ModelConfig(d_model=128,  n_layers=6, n_heads=4, d_ff=512),
    "w_small":  ModelConfig(d_model=192,  n_layers=6, n_heads=4, d_ff=768),
    "w_medium": ModelConfig(d_model=256,  n_layers=6, n_heads=4, d_ff=1024),
    "w_large":  ModelConfig(d_model=384,  n_layers=6, n_heads=4, d_ff=1536),
    "w_xl":     ModelConfig(d_model=512,  n_layers=6, n_heads=4, d_ff=2048),
}

ALL_CONFIGS: dict[str, ModelConfig] = {**MODEL_CONFIGS, **WIDTH_ONLY_CONFIGS}


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads  = cfg.n_heads
        self.d_head   = cfg.d_model // cfg.n_heads
        self.d_model  = cfg.d_model
        self.dropout  = cfg.dropout

        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=True)
        self.out_proj  = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.attn_drop = nn.Dropout(cfg.dropout)
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

        scale = 1.0 / math.sqrt(self.d_head)
        att = (q @ k.transpose(-2, -1)) * scale
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(y))


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff, bias=True),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model, bias=True),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.d_model)
        self.ff   = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """Decoder-only transformer language model."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb   = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.emb_drop  = nn.Dropout(cfg.dropout)

        self.blocks  = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f    = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.lm_head.weight = self.token_emb.weight

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
        """
        idx     : (B, T) long tensor of token ids
        targets : (B, T) long tensor of target ids (optional)
        Returns (logits, loss) where loss is None if targets is None.
        """
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len, \
            f"Sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}"

        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.emb_drop(self.token_emb(idx) + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

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
        """Autoregressive generation. Stops at EOS or max_new_tokens."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.max_seq_len else idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

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
        """Non-embedding parameter count (excludes token_emb and pos_emb)."""
        emb_params = (
            self.token_emb.weight.numel()
            + self.pos_emb.weight.numel()
        )
        total = sum(p.numel() for p in self.parameters())
        return total - emb_params


def get_config(name: str, config_family: str = "default") -> ModelConfig:
    """Resolve a model name from a config family ('default' or 'width_only')."""
    if config_family == "width_only":
        table = WIDTH_ONLY_CONFIGS
    else:
        table = MODEL_CONFIGS
    if name not in table:
        raise ValueError(
            f"Unknown model name '{name}' for family '{config_family}'. "
            f"Choose from: {list(table)}"
        )
    return table[name]


def build_model(name: str, config_family: str = "default", **overrides) -> TransformerLM:
    """Build a model by config name. config_family: 'default' or 'width_only'."""
    import dataclasses
    cfg = dataclasses.replace(get_config(name, config_family), **overrides)
    return TransformerLM(cfg)


def print_model_summary() -> None:
    """Print parameter counts for all model configs."""
    for family_name, table in [("default", MODEL_CONFIGS), ("width_only", WIDTH_ONLY_CONFIGS)]:
        print(f"\n--- {family_name} ---")
        print(f"{'Model':<12} {'d_model':>8} {'n_layers':>9} {'n_heads':>8} {'d_ff':>6} {'Non-emb params':>16}")
        print("-" * 67)
        for name, cfg in table.items():
            m = TransformerLM(cfg)
            n = m.count_parameters()
            print(f"{name:<12} {cfg.d_model:>8} {cfg.n_layers:>9} {cfg.n_heads:>8} {cfg.d_ff:>6} {n:>16,}")
    print()


if __name__ == "__main__":
    print_model_summary()
