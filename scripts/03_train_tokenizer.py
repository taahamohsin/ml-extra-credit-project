"""
03_train_tokenizer.py
---------------------
Train a BPE tokenizer on the cleaned SVG corpus and save it.

Design choices (see Section 4.3 of the blueprint):
  - BPE with ByteLevel pre-tokenizer (handles all Unicode)
  - Vocab size = 4096
  - Special tokens: <PAD> (0), <BOS> (1), <EOS> (2), <UNK> (3)
  - Post-processor wraps every sequence with BOS/EOS automatically

Outputs:
  - outputs/tokenizer/tokenizer.json
  - outputs/tokenizer/tokenizer_stats.json  (top tokens, frequencies)
  - outputs/plots/token_freq_distribution.png

Usage:
    python scripts/03_train_tokenizer.py [--config configs/data_config.yaml]
"""

import argparse
import json
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.tokenizer_utils import (
    train_tokenizer,
    compute_tokenizer_stats,
    compute_token_frequencies,
    get_top_tokens,
    SPECIAL_TOKENS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_cleaned_svgs(cleaned_path: Path, max_samples: int | None = None, sample_size: int | None = None, seed: int = 42) -> list[str]:
    """Load SVG strings from the cleaned.jsonl file."""
    svgs = []
    with open(cleaned_path, encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading cleaned SVGs"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                svg = obj.get("svg", "")
                if svg:
                    svgs.append(svg)
            except json.JSONDecodeError:
                continue
            if max_samples and len(svgs) >= max_samples:
                break
    print(f"Loaded {len(svgs):,} cleaned SVGs")

    if sample_size is not None and sample_size < len(svgs):
        import random
        rng = random.Random(seed)
        svgs = rng.sample(svgs, sample_size)
        print(f"Sampled {len(svgs):,} SVGs for tokenizer training (seed={seed})")

    return svgs


def plot_token_frequencies(
    freq_dict: dict[int, int],
    tokenizer,
    save_path: Path,
    top_n: int = 200,
) -> None:
    """Plot Zipf-like token frequency distribution."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # Sort by frequency descending
        sorted_freqs = sorted(freq_dict.values(), reverse=True)
        ranks = list(range(1, len(sorted_freqs) + 1))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Zipf plot (log-log)
        ax = axes[0]
        ax.loglog(ranks[:top_n], sorted_freqs[:top_n], "b.", markersize=3)
        ax.set_xlabel("Token rank (log scale)")
        ax.set_ylabel("Frequency (log scale)")
        ax.set_title(f"Token Frequency Distribution (Zipf plot, top {top_n})")
        ax.grid(True, alpha=0.3)

        # Right: Top-30 tokens bar chart
        ax2 = axes[1]
        vocab = tokenizer.get_vocab()
        id_to_tok = {v: k for k, v in vocab.items()}
        top_30 = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:30]
        # Skip special tokens in bar chart
        special_ids = set(range(len(SPECIAL_TOKENS)))
        top_30 = [(id_to_tok.get(tid, f"id={tid}"), cnt)
                  for tid, cnt in top_30 if tid not in special_ids][:30]
        labels = [t[:20] for t, _ in top_30]
        counts = [c for _, c in top_30]
        ax2.barh(range(len(labels)), counts, color="steelblue")
        ax2.set_yticks(range(len(labels)))
        ax2.set_yticklabels(labels, fontsize=8)
        ax2.invert_yaxis()
        ax2.set_xlabel("Frequency")
        ax2.set_title("Top 30 Tokens by Frequency")

        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Token frequency plot saved to {save_path}")
    except Exception as e:
        print(f"WARNING: Could not generate token frequency plot: {e}")


def plot_sequence_length_histogram(
    length_stats: dict,
    lengths: list[int],
    save_path: Path,
) -> None:
    """Plot a histogram of token lengths per SVG."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(lengths, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("Token count per SVG (includes BOS/EOS)")
        ax.set_ylabel("Number of SVGs (log scale)")
        ax.set_title("SVG Sequence Length Distribution")

        # Annotate with stats
        stats_text = (
            f"Mean:   {length_stats['mean']:.0f}\n"
            f"Median: {length_stats['median']:.0f}\n"
            f"P95:    {length_stats['p95']:.0f}\n"
            f"P99:    {length_stats['p99']:.0f}\n"
            f"Max:    {length_stats['max']}"
        )
        ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
                va="top", ha="right", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Sequence length histogram saved to {save_path}")
    except Exception as e:
        print(f"WARNING: Could not generate sequence length histogram: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str = "configs/data_config.yaml") -> None:
    config_file = REPO_ROOT / config_path
    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    tok_cfg = cfg["tokenizer"]
    paths_cfg = cfg["paths"]
    cleaning_cfg = cfg["cleaning"]

    cleaned_path = REPO_ROOT / paths_cfg["cleaned_data_dir"] / "cleaned.jsonl"
    tokenizer_dir = REPO_ROOT / paths_cfg["tokenizer_dir"]   # may be a Drive symlink
    plots_dir = REPO_ROOT / paths_cfg["plots_dir"]
    stats_file = REPO_ROOT / paths_cfg["stats_file"]

    # Train and save to local disk first, then copy to Drive at the end.
    # This avoids Drive write-quota errors during the tokenizer save step.
    local_tokenizer_dir = Path("/tmp/tokenizer_local")
    local_tokenizer_dir.mkdir(parents=True, exist_ok=True)

    if not cleaned_path.exists():
        print(f"ERROR: {cleaned_path} not found. Run 02_clean_normalize.py first.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 1. Load cleaned SVGs (subsample for tokenizer training to save RAM)
    # -----------------------------------------------------------------------
    sample_size = tok_cfg.get("sample_size", None)
    svgs = load_cleaned_svgs(cleaned_path, sample_size=sample_size, seed=cfg.get("seed", 42))

    # -----------------------------------------------------------------------
    # 2. Train tokenizer — save to local /tmp first
    # -----------------------------------------------------------------------
    print(f"\nTraining BPE tokenizer (vocab_size={tok_cfg['vocab_size']}) ...")
    print(f"  Saving locally to {local_tokenizer_dir} (Drive copy happens after stats)")
    tokenizer = train_tokenizer(
        svg_texts=svgs,
        vocab_size=tok_cfg["vocab_size"],
        save_dir=local_tokenizer_dir,
    )

    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"Actual vocabulary size: {actual_vocab_size:,}")

    # -----------------------------------------------------------------------
    # 3. Show example tokenizations
    # -----------------------------------------------------------------------
    print("\n--- Example tokenizations ---")
    examples = svgs[:3] if len(svgs) >= 3 else svgs
    for i, svg in enumerate(examples):
        enc = tokenizer.encode(svg)
        print(f"\nExample {i+1} ({len(svg)} chars → {len(enc.ids)} tokens):")
        print(f"  SVG (first 120 chars): {svg[:120]!r}")
        print(f"  Token ids (first 20):  {enc.ids[:20]}")
        tokens = [tokenizer.id_to_token(tid) for tid in enc.ids[:20]]
        print(f"  Tokens (first 20):     {tokens}")

    # -----------------------------------------------------------------------
    # 4. Compute token-length statistics
    # -----------------------------------------------------------------------
    print("\nComputing token length statistics (this may take a few minutes) ...")
    length_stats = compute_tokenizer_stats(
        tokenizer, svgs, max_token_length=cleaning_cfg["max_token_length"]
    )
    print("\nToken length statistics:")
    for k, v in length_stats.items():
        print(f"  {k:30s}: {v}")

    # Collect individual lengths for histogram
    from src.tokenizer_utils import encode as tok_encode
    lengths = [len(tok_encode(tokenizer, svg)) for svg in tqdm(svgs, desc="  Computing lengths")]

    # -----------------------------------------------------------------------
    # 5. Compute token frequencies
    # -----------------------------------------------------------------------
    print("\nComputing token frequencies ...")
    freq_dict = compute_token_frequencies(tokenizer, svgs)
    print(f"  Unique tokens seen: {len(freq_dict):,} / {actual_vocab_size:,}")

    top_tokens = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:30]
    vocab = tokenizer.get_vocab()
    id_to_tok = {v: k for k, v in vocab.items()}
    print("\nTop 20 tokens by frequency:")
    print(f"  {'Token':<25} {'ID':>6} {'Count':>12}")
    print(f"  {'-'*25} {'-'*6} {'-'*12}")
    for tid, cnt in top_tokens[:20]:
        tok_str = id_to_tok.get(tid, f"<id={tid}>")
        print(f"  {tok_str:<25} {tid:>6} {cnt:>12,}")

    # -----------------------------------------------------------------------
    # 6. Save tokenizer stats — local first
    # -----------------------------------------------------------------------
    tokenizer_stats = {
        "vocab_size": actual_vocab_size,
        "num_svgs_trained_on": len(svgs),
        "length_stats": length_stats,
        "top_30_tokens": [
            {"id": tid, "token": id_to_tok.get(tid, "?"), "count": cnt}
            for tid, cnt in top_tokens
        ],
    }
    tok_stats_path = local_tokenizer_dir / "tokenizer_stats.json"
    with open(tok_stats_path, "w") as f:
        json.dump(tokenizer_stats, f, indent=2)
    print(f"\nTokenizer stats saved locally to {tok_stats_path}")

    # Merge into main stats file if it exists
    if stats_file.exists():
        with open(stats_file) as f:
            main_stats = json.load(f)
        main_stats["tokenizer"] = tokenizer_stats
        with open(stats_file, "w") as f:
            json.dump(main_stats, f, indent=2)

    # -----------------------------------------------------------------------
    # 7. Generate plots — local disk
    # -----------------------------------------------------------------------
    local_plots_dir = Path("/tmp/plots_local")
    local_plots_dir.mkdir(parents=True, exist_ok=True)
    print("\nGenerating plots ...")
    plot_token_frequencies(
        freq_dict, tokenizer,
        save_path=local_plots_dir / "token_freq_distribution.png",
    )
    plot_sequence_length_histogram(
        length_stats, lengths,
        save_path=local_plots_dir / "sequence_length_histogram.png",
    )

    # -----------------------------------------------------------------------
    # 8. Copy all outputs from local disk to Drive
    # -----------------------------------------------------------------------
    import shutil
    print("\nCopying outputs to Drive ...")

    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    for src in local_tokenizer_dir.iterdir():
        dst = tokenizer_dir / src.name
        shutil.copy2(src, dst)
        print(f"  {src.name}  ({src.stat().st_size / 1e3:.1f} KB) → {dst}")

    drive_plots = REPO_ROOT / plots_dir
    drive_plots.mkdir(parents=True, exist_ok=True)
    for src in local_plots_dir.iterdir():
        dst = drive_plots / src.name
        shutil.copy2(src, dst)
        print(f"  {src.name}  ({src.stat().st_size / 1e3:.1f} KB) → {dst}")

    # -----------------------------------------------------------------------
    # 9. Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Phase 1 Step 3 COMPLETE — tokenizer trained.")
    print(f"Vocabulary size:    {actual_vocab_size:,}")
    print(f"Median seq length:  {length_stats['median']:.0f} tokens")
    print(f"P99 seq length:     {length_stats['p99']:.0f} tokens")
    print(f"SVGs > 1024 tokens: {length_stats['exceeds_max_length']:,} "
          f"({length_stats['exceeds_max_length_pct']:.1f}%)")
    print(f"Tokenizer dir:      {tokenizer_dir}")
    print("\nNext: run scripts/04_prepare_dataset.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BPE tokenizer on cleaned SVGs")
    parser.add_argument("--config", default="configs/data_config.yaml")
    args = parser.parse_args()
    main(args.config)
