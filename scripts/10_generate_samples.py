"""
10_generate_samples.py
----------------------
Load the best SP checkpoint and generate SVG samples.

Produces two sets of samples:
  - Unconditional: model continues from "<svg" alone (15 samples = 5 per temp)
  - Prefix-conditioned: 5 partial SVGs × 3 temperatures = 15 samples

Sampling: top_k=50, top_p=0.95, temperatures in {0.5, 0.8, 1.0}.

Saves each generated SVG as a .svg file plus a manifest.json describing
which file came from which prompt/temperature/seed.

Usage:
    python scripts/10_generate_samples.py
    python scripts/10_generate_samples.py --checkpoint outputs/checkpoints/xl/best.pt
    python scripts/10_generate_samples.py --n_uncond 10 --max_new_tokens 1024
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model import TransformerLM
from src.tokenizer_utils import load_tokenizer, BOS_ID, EOS_ID


# Canonical SVG header observed in training data (BPE-merged into a small
# number of tokens). Generation prompts must start with this exact string,
# otherwise the tokenization differs from training and the model emits
# off-distribution garbage.
SVG_HEADER = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" height="200px" width="200px">'

# Five fixed prefixes from the blueprint (Phase 4.3), prepended with the
# canonical header so the model sees a training-distribution prefix.
PREFIXES = [
    SVG_HEADER + '<circle cx="12" cy="12" r="10" fill="none" stroke="black"/><circle cx="9" cy="10" r="1" fill="black"/>',
    SVG_HEADER + '<path d="M2 12 L12 2 L22 12',
    SVG_HEADER + '<g fill="red"><rect x="4" y="4" width="6" height="6"/>',
    SVG_HEADER + '<polygon points="12,2 15,9',
    SVG_HEADER + '<g transform="translate(12,12)"><circle r="10" fill="none" stroke="blue"/>',
]

TEMPERATURES = [0.5, 0.8, 1.0]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_from_checkpoint(ckpt_path: Path, device: torch.device) -> TransformerLM:
    """Build a TransformerLM from a checkpoint and load its weights."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = TransformerLM(cfg)

    state = ckpt["model"]
    # Strip any "_orig_mod." prefix added by torch.compile
    cleaned = {
        (k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k): v
        for k, v in state.items()
    }
    model.load_state_dict(cleaned)
    model = model.to(device)
    model.eval()

    n_params = model.count_parameters()
    step = ckpt.get("step", "?")
    best = ckpt.get("best_val_loss", float("nan"))
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"  Non-emb params: {n_params:,}")
    print(f"  Step:           {step}")
    print(f"  Best val loss:  {best:.4f}")
    return model


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_one(
    model: TransformerLM,
    tokenizer,
    prompt: str,
    temperature: float,
    top_k: int,
    top_p: float,
    max_new_tokens: int,
    device: torch.device,
    seed: int,
) -> tuple[str, list[int]]:
    """Generate a single completion. Returns (decoded_text, token_ids)."""
    torch.manual_seed(seed)

    # The tokenizer's post-processor adds <BOS> and <EOS>; we want only <BOS>
    # at the start (no <EOS> in the prompt!), so we encode without the
    # processor and prepend BOS manually.
    enc = tokenizer.encode(prompt)
    ids = enc.ids
    # Strip a trailing EOS the post-processor may have added.
    if ids and ids[-1] == EOS_ID:
        ids = ids[:-1]
    if not ids or ids[0] != BOS_ID:
        ids = [BOS_ID] + ids

    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_id=EOS_ID,
    )
    out_ids = out[0].tolist()
    text = tokenizer.decode(out_ids, skip_special_tokens=True)
    return text, out_ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="outputs/checkpoints/xl/best.pt")
    ap.add_argument("--tokenizer_dir", type=str, default="outputs/tokenizer")
    ap.add_argument("--out_dir", type=str, default="outputs/samples")
    ap.add_argument(
        "--n_uncond",
        type=int,
        default=15,
        help="Total unconditional samples (split across temperatures)",
    )
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed_base", type=int, default=12345)
    ap.add_argument("--uncond_prompt", type=str,
                    default=SVG_HEADER,
                    help="Prompt for unconditional generation. Should match the "
                         "canonical SVG header seen in training so the BPE "
                         "tokenization matches and the model is in-distribution.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = REPO_ROOT / args.checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    tokenizer = load_tokenizer(REPO_ROOT / args.tokenizer_dir)
    model = load_model_from_checkpoint(ckpt_path, device)

    out_root = REPO_ROOT / args.out_dir
    uncond_dir = out_root / "unconditional"
    prefix_dir = out_root / "prefix"
    uncond_dir.mkdir(parents=True, exist_ok=True)
    prefix_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "checkpoint": args.checkpoint,
        "max_new_tokens": args.max_new_tokens,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperatures": TEMPERATURES,
        "unconditional": [],
        "prefix": [],
    }

    # ---------- Unconditional ----------
    # Distribute n_uncond as evenly as possible across temperatures.
    n_per_temp = max(1, args.n_uncond // len(TEMPERATURES))
    print(f"\n=== Unconditional generation ({n_per_temp} per temperature) ===")
    sample_idx = 0
    for temp in TEMPERATURES:
        for k in range(n_per_temp):
            seed = args.seed_base + sample_idx
            text, ids = generate_one(
                model,
                tokenizer,
                prompt=args.uncond_prompt,
                temperature=temp,
                top_k=args.top_k,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                device=device,
                seed=seed,
            )
            fname = f"uncond_t{temp:.1f}_{k:02d}.svg"
            (uncond_dir / fname).write_text(text, encoding="utf-8")
            manifest["unconditional"].append(
                {
                    "file": f"unconditional/{fname}",
                    "temperature": temp,
                    "seed": seed,
                    "n_tokens": len(ids),
                }
            )
            print(
                f"  [{sample_idx + 1}/{n_per_temp * len(TEMPERATURES)}] "
                f"t={temp}  tokens={len(ids):>4}  {fname}"
            )
            sample_idx += 1

    # ---------- Prefix-conditioned ----------
    print(
        f"\n=== Prefix-conditioned generation "
        f"({len(PREFIXES)} prefixes × {len(TEMPERATURES)} temps) ==="
    )
    sample_idx = 0
    for p_idx, prefix in enumerate(PREFIXES):
        for temp in TEMPERATURES:
            seed = args.seed_base + 1000 + sample_idx
            text, ids = generate_one(
                model,
                tokenizer,
                prompt=prefix,
                temperature=temp,
                top_k=args.top_k,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                device=device,
                seed=seed,
            )
            fname = f"prefix{p_idx}_t{temp:.1f}.svg"
            (prefix_dir / fname).write_text(text, encoding="utf-8")
            manifest["prefix"].append(
                {
                    "file": f"prefix/{fname}",
                    "prefix_index": p_idx,
                    "prefix": prefix,
                    "temperature": temp,
                    "seed": seed,
                    "n_tokens": len(ids),
                }
            )
            print(f"  prefix {p_idx} t={temp}  tokens={len(ids):>4}  {fname}")
            sample_idx += 1

    # Save manifest
    manifest_path = out_root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")
    print(f"Unconditional samples: {len(manifest['unconditional'])} in {uncond_dir}")
    print(f"Prefix samples:        {len(manifest['prefix'])} in {prefix_dir}")


if __name__ == "__main__":
    main()
""
