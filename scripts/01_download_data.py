"""
01_download_data.py
-------------------
Download SVG datasets from HuggingFace and save raw SVG strings to disk.
Loads icons-simple first; adds emoji-simple and a subsampled slice of
fonts-simple until the ~100M token target is met.
No cleaning — run 02_clean_normalize.py next.

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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def debug_dataset_structure(ds, dataset_name: str) -> None:
    """Print column names and first example so we can see the actual schema."""
    print(f"\n[DEBUG] {dataset_name} — column names: {ds.column_names}")
    ex = ds[0]
    print("[DEBUG] First example:")
    for k, v in ex.items():
        preview = repr(v)[:200] if not isinstance(v, str) else repr(v[:200])
        print(f"  {k!r}: {type(v).__name__} = {preview}")


def get_svg_field(example: dict) -> str | None:
    """Try common column name variants used across StarVector datasets."""
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
    print(f"\nLoading {dataset_name} (split={split}) ...")
    try:
        ds = load_dataset(dataset_name, split=split)
    except Exception as e:
        print(f"  WARNING: Could not load {dataset_name}: {e}")
        return []

    print(f"  Raw dataset size: {len(ds):,} examples")
    debug_dataset_structure(ds, dataset_name)

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


def main(config_path: str = "configs/data_config.yaml", fonts_only: bool = False) -> None:
    config_file = REPO_ROOT / config_path
    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    raw_dir = REPO_ROOT / cfg["paths"]["raw_data_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    datasets_cfg = cfg["datasets"]
    target_tokens = datasets_cfg["target_train_tokens"]

    TOKENS_PER_SVG = 99   # measured from mixed corpus (icons+emoji+fonts): 88.3M / 761k = ~116,
                          # but fonts dominate supplements and average ~99 tok/SVG

    if fonts_only:
        # Re-download fonts-simple with updated fraction; preserve icons/emoji from the existing manifest.
        manifest_path = raw_dir / "download_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            all_files = manifest["downloaded_files"]
            estimated_tokens = manifest.get("estimated_tokens_rough", 0)
            all_files = [f for f in all_files if "fonts" not in f]
            estimated_tokens -= manifest.get("fonts_tokens", 0)
        else:
            print("WARNING: No existing manifest found. Run without --fonts-only first.")
            all_files = []
            estimated_tokens = 0

        frac = datasets_cfg.get("fonts_subsample_fraction", 0.40)
        fonts_name = [d for d in datasets_cfg.get("supplementary", []) if "fonts" in d][0]
        print(f"\nRe-downloading {fonts_name} at subsample_fraction={frac} ...")
        fonts_svgs = load_and_collect(fonts_name, subsample_fraction=frac)
        if fonts_svgs:
            save_raw_svgs(fonts_svgs, raw_dir / "fonts_simple.jsonl", source_name="fonts-simple")
            if "fonts_simple.jsonl" not in all_files:
                all_files.append("fonts_simple.jsonl")
            fonts_tokens = len(fonts_svgs) * TOKENS_PER_SVG
            estimated_tokens += fonts_tokens
            print(f"Fonts tokens estimate: {fonts_tokens:,}  Total estimate: {estimated_tokens:,}")

        manifest = {
            "downloaded_files": all_files,
            "fonts_tokens": fonts_tokens if fonts_svgs else 0,
            "estimated_tokens_rough": estimated_tokens,
            "target_tokens": target_tokens,
            "note": "Actual token count computed after cleaning + tokenization in scripts 02-04",
        }
        manifest_path = raw_dir / "download_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Manifest updated: {manifest_path}")
        print(f"Files in manifest: {all_files}")
        return

    primary_name = datasets_cfg["primary"]
    icons_svgs = load_and_collect(primary_name)
    save_raw_svgs(icons_svgs, raw_dir / "icons_simple.jsonl", source_name="icons-simple")

    estimated_tokens = len(icons_svgs) * TOKENS_PER_SVG
    print(f"\nEstimated tokens from icons-simple: {estimated_tokens:,} (~{TOKENS_PER_SVG} tok/SVG)")

    all_files = ["icons_simple.jsonl"]

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

    fonts_tokens = 0
    if estimated_tokens < target_tokens and remaining_datasets:
        fonts_name = remaining_datasets.pop(0)
        frac = datasets_cfg.get("fonts_subsample_fraction", 0.40)
        fonts_svgs = load_and_collect(fonts_name, subsample_fraction=frac)
        if fonts_svgs:
            save_raw_svgs(fonts_svgs, raw_dir / "fonts_simple.jsonl", source_name="fonts-simple")
            all_files.append("fonts_simple.jsonl")
            fonts_tokens = len(fonts_svgs) * TOKENS_PER_SVG
            estimated_tokens += fonts_tokens
            print(f"Updated estimated tokens (after fonts): {estimated_tokens:,}")

    manifest = {
        "downloaded_files": all_files,
        "fonts_tokens": fonts_tokens,
        "estimated_tokens_rough": estimated_tokens,
        "target_tokens": target_tokens,
        "note": "Actual token count computed after cleaning + tokenization in scripts 02-04",
    }
    manifest_path = raw_dir / "download_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")

    print(f"\nRaw data dir: {raw_dir}")
    print(f"Files: {all_files}")
    print(f"Estimated tokens (rough): {estimated_tokens:,}")
    print("Next: run scripts/02_clean_normalize.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SVG datasets from HuggingFace")
    parser.add_argument("--config", default="configs/data_config.yaml",
                        help="Path to data_config.yaml (relative to repo root)")
    parser.add_argument("--fonts-only", action="store_true",
                        help="Re-download only fonts-simple (preserves icons/emoji on disk)")
    args = parser.parse_args()
    main(args.config, fonts_only=args.fonts_only)
