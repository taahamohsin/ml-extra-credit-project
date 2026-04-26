"""
09_coord_check_mup.py
---------------------
µP coordinate check (the canonical sanity test for µP implementations).

Builds the same architecture at four widths (d_model = 64, 128, 256, 512), all
sharing depth, runs a handful of training steps at a deliberately large LR, and
records the L1 norm of activations at each layer per step.

Pass criterion: for a correctly wired µP model, the activation L1 curves for
all widths overlap as training proceeds. If they fan out (grow with width) or
collapse (shrink with width), µP is misimplemented somewhere.

Usage:
    python scripts/09_coord_check_mup.py --out outputs/coord_check.png
    python scripts/09_coord_check_mup.py --sp --out outputs/coord_check_sp.png  # SP for contrast

The SP run is intentionally a control: SP curves SHOULD fan out, µP curves
should NOT. Running both side-by-side makes the plot self-explanatory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt

# Allow `python scripts/...` invocation from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mup.coord_check import get_coord_data, plot_coord_data

from src.model import ModelConfig, TransformerLM
from src.model_mup import MupTransformerLM, build_mup_model, BASE_HEAD_DIM


# Widths to scan. d_head is held constant at BASE_HEAD_DIM by varying n_heads
# accordingly: d_model = n_heads * d_head. Depth is fixed at 6 (Small/Medium).
WIDTHS = [64, 128, 256, 512]
DEPTH  = 6
SEQ_LEN = 64       # short — coord check only needs a few forward/backward passes
BATCH   = 4
N_STEPS = 5        # paper recommends 3–5 steps
VOCAB   = 4096
LR_MUP  = 1e-2     # large LR to surface instability; mup README explicitly says use a large LR
LR_SP   = 1e-3     # SP can't tolerate the µP LR — it explodes and the plot is unreadable


def make_mup_lazy_model_fn(width: int):
    """Returns a zero-arg builder that produces a fresh µP model of the given width."""
    n_heads = width // BASE_HEAD_DIM
    assert width % BASE_HEAD_DIM == 0, f"width {width} not divisible by BASE_HEAD_DIM {BASE_HEAD_DIM}"

    def build():
        # d_ff scales with d_model at the standard 4× ratio
        return build_mup_model(
            "tiny",
            d_model=width,
            n_heads=n_heads,
            n_layers=DEPTH,
            d_ff=width * 4,
            vocab_size=VOCAB,
            max_seq_len=SEQ_LEN,
        )
    return build


def make_sp_lazy_model_fn(width: int):
    """Control: same architecture in standard parameterization."""
    n_heads = width // BASE_HEAD_DIM

    def build():
        cfg = ModelConfig(
            vocab_size=VOCAB,
            d_model=width,
            n_layers=DEPTH,
            n_heads=n_heads,
            d_ff=width * 4,
            max_seq_len=SEQ_LEN,
            dropout=0.0,
        )
        return TransformerLM(cfg)
    return build


def random_dataloader(batch: int, seq_len: int, vocab: int, n_batches: int):
    """Returns a list of (input, target) tuples — coord check is data-agnostic.
    Returns a list (not a generator) because mup.coord_check calls iter() on
    this object multiple times (once per width × seed)."""
    g = torch.Generator().manual_seed(0)
    out = []
    for _ in range(n_batches):
        x = torch.randint(0, vocab, (batch, seq_len), generator=g)
        y = torch.randint(0, vocab, (batch, seq_len), generator=g)
        out.append((x, y))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sp", action="store_true",
                    help="Run SP (control) instead of µP. SP curves should fan out with width.")
    ap.add_argument("--out", type=str, default="outputs/coord_check.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mode = "SP (control — should fan out)" if args.sp else "µP (should overlap)"
    print(f"\nCoord check: {mode}")
    print(f"Widths: {WIDTHS}, depth={DEPTH}, batch={BATCH}, seq={SEQ_LEN}, "
          f"steps={N_STEPS}, lr={LR_SP if args.sp else LR_MUP}, device={device}\n")

    if args.sp:
        models_fn = {w: make_sp_lazy_model_fn(w) for w in WIDTHS}
        lr = LR_SP
    else:
        models_fn = {w: make_mup_lazy_model_fn(w) for w in WIDTHS}
        lr = LR_MUP

    # get_coord_data takes a string optimizer name; mup=True routes to mup.optim,
    # mup=False routes to torch.optim.
    dataloader = random_dataloader(BATCH, SEQ_LEN, VOCAB, N_STEPS)

    df = get_coord_data(
        models_fn,
        dataloader,
        optimizer="adamw",
        mup=not args.sp,
        lr=lr,
        nsteps=N_STEPS,
        nseeds=1,
        dict_in_out=False,
        output_name="loss",
        cuda=(device == "cuda"),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_coord_data(
        df,
        legend="full",
        save_to=str(out_path),
        suptitle=f"Coord check — {mode}",
        face_color=None,
    )
    print(f"\nSaved: {out_path}")
    print("Pass criterion: activation L1 curves overlap across widths.")
    print("Fail signal: curves fan out with width (µP misimplemented) "
          "or collapse to zero (init/scaling wrong).")


if __name__ == "__main__":
    main()
