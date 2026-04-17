"""
01_download_data.py
-------------------
Download SVG datasets from HuggingFace and save the raw SVG strings to disk.

Strategy (from Section 4.1 of the blueprint):
  1. Load starvector/svg-icons-simple (~89k files) — primary dataset
  2. If after tokenization we are under 100M tokens, add svg-emoji-simple
  3. If still under, subsample svg-fonts-simple until we hit the target

This script only handles downloading and saving raw SVG text — no cleaning.
Run 02_clean_normalize.py next.

Usage:
    python scripts/01_download_data.py [--config configs/data_config.yaml]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from datasets import load_dataset
from tqdm import tqdm

# Allow importing from src/ regardless of working directory
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def debug_dataset_structure(ds, dataset_name: str) -> None:
    """Print column names and first example so we can see the actual schema."""
    print(f"\n[DEBUG] {dataset_name} — column names: {ds.column_names}")
    ex = ds[0]
    print("[DEBUG] First example:")
    for k, v in ex.items():
        preview = repr(v)[:200] if not isinstance(v, str) else repr(v[:200])
        print(f"  {k!r}: {type(v).__name__} = {preview}")


def get_svg_field(example: dict) -> str | None:
    """
    Extract the SVG string from a dataset example.
    Tries common column names used across StarVector datasets,
    including both lowercase and title-case variants.
    """
    for key in ("Svg", "svg", "SVG", "image", "svg_code", "text"):
        val = example.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def save_raw_svgs(
    svgs: list[str],
    output_path: Path,
    source_name: str,
) -> None:
    """Save a list of SVG strings as a JSON lines file (.jsonl)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for svg in svgs:
            f.write(json.dumps({"svg": svg, "source": source_name}) + "\n")
    print(f"  Saved {len(svgs):,} SVGs → {output_path}")


def load_and_collect(
    dataset_name: str,
    split: str = "train",
    max_samples: int | None = None,
    subsample_fraction: float | None = None,
) -> list[str]:
    """Load a HuggingFace dataset and collect raw SVG strings."""
    print(f"\nLoading {dataset_name} (split={split}) ...")
    try:
        ds = load_dataset(dataset_name, split=split)
    except Exception as e:
        print(f"  WARNING: Could not load {dataset_name}: {e}")
        return []

    print(f"  Raw dataset size: {len(ds):,} examples")
    debug_dataset_structure(ds, dataset_name)

    # Optional subsampling
    if subsample_fraction is not None and subsample_fraction < 1.0:
        n = int(len(ds) * subsample_fraction)
        ds = ds.shuffle(seed=42).select(range(n))
        print(f"  Subsampled to {len(ds):,} examples (fraction={subsample_fraction})")

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
        print(f"  Capped at {len(ds):,} examples")

    svgs = []
    missing = 0
    for example in tqdm(ds, desc=f"  Collecting SVGs from {dataset_name.split('/')[-1]}"):
        svg = get_svg_field(example)
        if svg:
            svgs.append(svg)
        else:
            missing += 1

    if missing > 0:
        print(f"  WARNING: {missing:,} examples had no SVG field")

    print(f"  Collected {len(svgs):,} SVG strings")
    return svgs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str = "configs/data_config.yaml") -> None:
    config_file = REPO_ROOT / config_path
    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    raw_dir = REPO_ROOT / cfg["paths"]["raw_data_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    datasets_cfg = cfg["datasets"]
    target_tokens = datasets_cfg["target_train_tokens"]

    # -----------------------------------------------------------------------
    # 1. Primary dataset: svg-icons-simple
    # -----------------------------------------------------------------------
    primary_name = datasets_cfg["primary"]
    icons_svgs = load_and_collect(primary_name)
    save_raw_svgs(icons_svgs, raw_dir / "icons_simple.jsonl", source_name="icons-simple")

    # Measured token rate from icons-simple run: ~220 tokens/SVG after BPE-4096
    # (17,726,467 tokens / 80,434 SVGs = 220.4). Use SVG count as the estimator
    # rather than character count, which was wildly off (old: 0.4 chars/token).
    TOKENS_PER_SVG = 220
    estimated_tokens = len(icons_svgs) * TOKENS_PER_SVG
    print(f"\nEstimated tokens from icons-simple: {estimated_tokens:,} (~{TOKENS_PER_SVG} tok/SVG)")

    all_files = ["icons_simple.jsonl"]

    # -----------------------------------------------------------------------
    # 2. Supplementary: svg-emoji-simple (if needed)
    # -----------------------------------------------------------------------
    supplementary = datasets_cfg.get("supplementary", [])
    remaining_datasets = list(supplementary)

    if estimated_tokens < target_tokens and remaining_datasets:
        emoji_name = remaining_datasets.pop(0)
        emoji_svgs = load_and_collect(emoji_name)
        if emoji_svgs:
            save_raw_svgs(emoji_svgs, raw_dir / "emoji_simple.jsonl", source_name="emoji-simple")
            all_files.append("emoji_simple.jsonl")
            estimated_tokens += len(emoji_svgs) * TOKENS_PER_SVG
            print(f"Updated estimated tokens (after emoji): {estimated_tokens:,}")

    # -----------------------------------------------------------------------
    # 3. Supplementary: svg-fonts-simple (subsampled, if still needed)
    # -----------------------------------------------------------------------
    if estimated_tokens < target_tokens and remaining_datasets:
        fonts_name = remaining_datasets.pop(0)
        frac = datasets_cfg.get("fonts_subsample_fraction", 0.05)
        fonts_svgs = load_and_collect(fonts_name, subsample_fraction=frac)
        if fonts_svgs:
            save_raw_svgs(fonts_svgs, raw_dir / "fonts_simple.jsonl", source_name="fonts-simple")
            all_files.append("fonts_simple.jsonl")
            estimated_tokens += len(fonts_svgs) * TOKENS_PER_SVG
            print(f"Updated estimated tokens (after fonts): {estimated_tokens:,}")

    # -----------------------------------------------------------------------
    # 4. Save a manifest of downloaded files
    # -----------------------------------------------------------------------
    manifest = {
        "downloaded_files": all_files,
        "total_raw_svgs": len(icons_svgs),
        "estimated_tokens_rough": estimated_tokens,
        "target_tokens": target_tokens,
        "note": "Actual token count computed after cleaning + tokenization in scripts 02-04",
    }
    manifest_path = raw_dir / "download_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")

    print("\n" + "=" * 60)
    print("Phase 1 Step 1 COMPLETE — data downloaded.")
    print(f"Raw data directory: {raw_dir}")
    print(f"Files: {all_files}")
    print(f"Estimated tokens (rough): {estimated_tokens:,}")
    print("Next: run scripts/02_clean_normalize.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SVG datasets from HuggingFace")
    parser.add_argument("--config", default="configs/data_config.yaml",
                        help="Path to data_config.yaml (relative to repo root)")
    args = parser.parse_args()
    main(args.config)
