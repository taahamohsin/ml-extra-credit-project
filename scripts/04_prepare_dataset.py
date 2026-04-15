"""
04_prepare_dataset.py
---------------------
Tokenize all cleaned SVGs, filter by max token length, split into
train/val/test sets, and save as numpy memory-mapped binary files for
efficient training.

Binary format (nanoGPT-style):
  Concatenated token IDs: <BOS> [svg1_tokens] <EOS> <BOS> [svg2_tokens] <EOS> ...
  Stored as numpy uint16 memmap files.

Why uint16? Vocab size = 4096 < 65536 = 2^16. uint16 uses half the memory of int32.

Outputs:
  - outputs/data/binary/train.bin   — memory-mapped train token array
  - outputs/data/binary/val.bin     — memory-mapped val token array
  - outputs/data/binary/test.bin    — memory-mapped test token array
  - outputs/data/binary/split_info.json  — sizes and metadata
  - outputs/data/dataset_stats.json — updated with final token counts

Usage:
    python scripts/04_prepare_dataset.py [--config configs/data_config.yaml]
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.tokenizer_utils import load_tokenizer, encode as tok_encode, BOS_ID, EOS_ID


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_cleaned_svgs_with_idx(cleaned_path: Path) -> list[str]:
    """Load all SVG strings from cleaned.jsonl."""
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
    return svgs


def tokenize_and_filter(
    svgs: list[str],
    tokenizer,
    max_token_length: int,
) -> tuple[list[list[int]], int]:
    """
    Tokenize all SVGs and discard those that exceed max_token_length.

    Returns
    -------
    (list_of_token_id_lists, num_filtered)
        Each inner list includes the BOS and EOS tokens already added by
        the tokenizer's post-processor.
    """
    tokenized = []
    filtered = 0

    for svg in tqdm(svgs, desc="Tokenizing SVGs"):
        ids = tok_encode(tokenizer, svg, add_special_tokens=True)
        if len(ids) > max_token_length:
            filtered += 1
            continue
        tokenized.append(ids)

    return tokenized, filtered


def split_by_file(
    tokenized: list[list[int]],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """
    Split tokenized SVG lists into train/val/test BY FILE (not by position).

    This prevents data leakage: entire SVGs are in one split only.
    """
    rng = random.Random(seed)
    indices = list(range(len(tokenized)))
    rng.shuffle(indices)

    n = len(indices)
    n_val = max(1, int(n * val_frac))
    n_test = max(1, int(n * (1 - train_frac - val_frac)))
    n_train = n - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train = [tokenized[i] for i in train_idx]
    val = [tokenized[i] for i in val_idx]
    test = [tokenized[i] for i in test_idx]

    return train, val, test


def write_binary(token_lists: list[list[int]], output_path: Path) -> int:
    """
    Concatenate token id lists and write as numpy uint16 memmap.
    Returns total number of tokens written.
    """
    # Compute total length first
    total_tokens = sum(len(ids) for ids in token_lists)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.memmap(str(output_path), dtype=np.uint16, mode="w+", shape=(total_tokens,))

    offset = 0
    for ids in token_lists:
        arr[offset:offset + len(ids)] = ids
        offset += len(ids)

    arr.flush()
    return total_tokens


def verify_binary(path: Path, expected_tokens: int) -> bool:
    """Basic sanity check: load the binary and verify token count."""
    arr = np.memmap(str(path), dtype=np.uint16, mode="r")
    return len(arr) == expected_tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str = "configs/data_config.yaml") -> None:
    config_file = REPO_ROOT / config_path
    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    cleaning_cfg = cfg["cleaning"]
    splitting_cfg = cfg["splitting"]
    paths_cfg = cfg["paths"]

    cleaned_path = REPO_ROOT / paths_cfg["cleaned_data_dir"] / "cleaned.jsonl"
    tokenizer_dir = REPO_ROOT / paths_cfg["tokenizer_dir"]
    binary_dir = REPO_ROOT / paths_cfg["binary_dir"]
    stats_file = REPO_ROOT / paths_cfg["stats_file"]

    if not cleaned_path.exists():
        print(f"ERROR: {cleaned_path} not found. Run 02_clean_normalize.py first.")
        sys.exit(1)

    tokenizer_path = tokenizer_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        print(f"ERROR: {tokenizer_path} not found. Run 03_train_tokenizer.py first.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 1. Load data and tokenizer
    # -----------------------------------------------------------------------
    print("Loading tokenizer ...")
    tokenizer = load_tokenizer(tokenizer_dir)
    print(f"  Vocab size: {tokenizer.get_vocab_size():,}")

    svgs = load_cleaned_svgs_with_idx(cleaned_path)
    print(f"\nTotal SVGs loaded: {len(svgs):,}")

    # -----------------------------------------------------------------------
    # 2. Tokenize and filter
    # -----------------------------------------------------------------------
    max_len = cleaning_cfg["max_token_length"]
    print(f"\nTokenizing and filtering (max_token_length={max_len}) ...")
    tokenized, n_filtered = tokenize_and_filter(svgs, tokenizer, max_len)

    print(f"  Tokenized:  {len(tokenized):,}")
    print(f"  Filtered (too long): {n_filtered:,} ({100*n_filtered/max(len(svgs),1):.1f}%)")

    # -----------------------------------------------------------------------
    # 3. Split by file
    # -----------------------------------------------------------------------
    print("\nSplitting train/val/test by file ...")
    train, val, test = split_by_file(
        tokenized,
        train_frac=splitting_cfg["train_fraction"],
        val_frac=splitting_cfg["val_fraction"],
        seed=splitting_cfg["seed"],
    )
    print(f"  Train SVGs: {len(train):,}")
    print(f"  Val SVGs:   {len(val):,}")
    print(f"  Test SVGs:  {len(test):,}")

    # -----------------------------------------------------------------------
    # 4. Write binary files
    # -----------------------------------------------------------------------
    binary_dir.mkdir(parents=True, exist_ok=True)
    print("\nWriting binary files ...")

    train_path = binary_dir / "train.bin"
    val_path = binary_dir / "val.bin"
    test_path = binary_dir / "test.bin"

    n_train_tokens = write_binary(train, train_path)
    n_val_tokens = write_binary(val, val_path)
    n_test_tokens = write_binary(test, test_path)

    print(f"  train.bin: {n_train_tokens:,} tokens  ({train_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  val.bin:   {n_val_tokens:,} tokens  ({val_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  test.bin:  {n_test_tokens:,} tokens  ({test_path.stat().st_size / 1e6:.1f} MB)")

    # -----------------------------------------------------------------------
    # 5. Verify
    # -----------------------------------------------------------------------
    print("\nVerifying binary files ...")
    assert verify_binary(train_path, n_train_tokens), "train.bin verification failed!"
    assert verify_binary(val_path, n_val_tokens), "val.bin verification failed!"
    assert verify_binary(test_path, n_test_tokens), "test.bin verification failed!"
    print("  All files verified OK.")

    # -----------------------------------------------------------------------
    # 6. Save split info and update stats
    # -----------------------------------------------------------------------
    split_info = {
        "train_svgs": len(train),
        "val_svgs": len(val),
        "test_svgs": len(test),
        "train_tokens": n_train_tokens,
        "val_tokens": n_val_tokens,
        "test_tokens": n_test_tokens,
        "total_tokens": n_train_tokens + n_val_tokens + n_test_tokens,
        "max_token_length": max_len,
        "n_filtered_too_long": n_filtered,
        "vocab_size": tokenizer.get_vocab_size(),
        "dtype": "uint16",
        "bos_id": BOS_ID,
        "eos_id": EOS_ID,
        "seed": splitting_cfg["seed"],
    }

    split_info_path = binary_dir / "split_info.json"
    with open(split_info_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"\nSplit info saved to {split_info_path}")

    # Update main stats file
    if stats_file.exists():
        with open(stats_file) as f:
            main_stats = json.load(f)
    else:
        main_stats = {}

    main_stats["splits"] = split_info
    with open(stats_file, "w") as f:
        json.dump(main_stats, f, indent=2)

    # -----------------------------------------------------------------------
    # 7. Token count check (critical milestone)
    # -----------------------------------------------------------------------
    target = cfg["datasets"]["target_train_tokens"]
    check_mark = "✓" if n_train_tokens >= target else "✗ WARNING: BELOW TARGET"
    print("\n" + "=" * 60)
    print("DATASET PREPARATION SUMMARY")
    print("=" * 60)
    print(f"Training tokens:   {n_train_tokens:>15,}  {check_mark}  (target: {target:,})")
    print(f"Validation tokens: {n_val_tokens:>15,}")
    print(f"Test tokens:       {n_test_tokens:>15,}")
    print(f"Total tokens:      {n_train_tokens + n_val_tokens + n_test_tokens:>15,}")
    print()
    print(f"Binary files in:   {binary_dir}")
    print()
    if n_train_tokens < target:
        print("WARNING: Training token count below 100M target.")
        print("Consider adding more datasets (see Section 4.1 of blueprint).")
        print("Re-run 01_download_data.py to download supplementary datasets,")
        print("then re-run steps 02, 03, 04.")
    print("\nPhase 1 COMPLETE. Ready for Phase 2 (model training).")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize and prepare binary dataset")
    parser.add_argument("--config", default="configs/data_config.yaml")
    args = parser.parse_args()
    main(args.config)
