"""
01b_download_balanced.py
------------------------
Download a diversity-balanced SVG corpus for Phase 4b generation quality.

Mix (by file count):
  - svg-icons-simple:  ALL  (~80K)   — diverse multi-element icons
  - svg-emoji-simple:  ALL  (~4K)    — colourful multi-element emoji
  - svg-fonts-simple:  ~200K         — subsampled (fonts_subsample_fraction=0.20)
  - svg-stack-simple:  up to 50K     — community SVGs for structural diversity

Saves to outputs/data/raw_balanced/ (never touches outputs/data/raw/).

Usage:
    python scripts/01b_download_balanced.py
    python scripts/01b_download_balanced.py --config configs/data_config_balanced.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import yaml
from datasets import load_dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def get_svg_field(example: dict) -> str | None:
    for key in ("Svg", "svg", "SVG", "image", "svg_code", "text"):
        val = example.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def load_and_collect(
    dataset_name: str,
    split: str = "train",
    subsample_fraction: float | None = None,
    max_samples: int | None = None,
    seed: int = 42,
) -> list[str]:
    print(f"\nLoading {dataset_name} ...")
    try:
        ds = load_dataset(dataset_name, split=split)
    except Exception as e:
        print(f"  WARNING: Could not load {dataset_name}: {e}")
        return []

    print(f"  Raw size: {len(ds):,} examples")

    if subsample_fraction is not None and subsample_fraction < 1.0:
        n = int(len(ds) * subsample_fraction)
        ds = ds.shuffle(seed=seed).select(range(n))
        print(f"  Subsampled to {len(ds):,} (fraction={subsample_fraction})")

    if max_samples is not None and len(ds) > max_samples:
        ds = ds.shuffle(seed=seed).select(range(max_samples))
        print(f"  Capped at {len(ds):,} (max_samples={max_samples})")

    svgs = []
    missing = 0
    for ex in tqdm(ds, desc=f"  {dataset_name.split('/')[-1]}"):
        svg = get_svg_field(ex)
        if svg:
            svgs.append(svg)
        else:
            missing += 1

    if missing:
        print(f"  WARNING: {missing:,} examples had no SVG field")
    print(f"  Collected {len(svgs):,} SVGs")
    return svgs


def save_raw_svgs(svgs: list[str], output_path: Path, source_name: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for svg in svgs:
            f.write(json.dumps({"svg": svg, "source": source_name}) + "\n")
    print(f"  Saved {len(svgs):,} SVGs → {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_config_balanced.yaml")
    args = ap.parse_args()

    config_file = REPO_ROOT / args.config
    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    raw_dir = REPO_ROOT / cfg["paths"]["raw_data_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    dcfg = cfg["datasets"]
    seed = cfg.get("seed", 42)

    TOKENS_PER_SVG = 99  # conservative estimate for mixed corpus

    all_files = []
    estimated_tokens = 0

    icons_svgs = load_and_collect(dcfg["primary"], seed=seed)
    if icons_svgs:
        save_raw_svgs(icons_svgs, raw_dir / "icons_simple.jsonl", "icons-simple")
        all_files.append("icons_simple.jsonl")
        estimated_tokens += len(icons_svgs) * TOKENS_PER_SVG

    supplementary = dcfg.get("supplementary", [])
    supp_iter = iter(supplementary)

    emoji_svgs: list[str] = []
    emoji_name = next(supp_iter, None)
    if emoji_name:
        emoji_svgs = load_and_collect(emoji_name, seed=seed)
        if emoji_svgs:
            save_raw_svgs(emoji_svgs, raw_dir / "emoji_simple.jsonl", "emoji-simple")
            all_files.append("emoji_simple.jsonl")
            estimated_tokens += len(emoji_svgs) * TOKENS_PER_SVG

    fonts_svgs: list[str] = []
    fonts_name = next(supp_iter, None)
    if fonts_name:
        frac = dcfg.get("fonts_subsample_fraction", 0.20)
        fonts_svgs = load_and_collect(fonts_name, subsample_fraction=frac, seed=seed)
        if fonts_svgs:
            save_raw_svgs(fonts_svgs, raw_dir / "fonts_simple.jsonl", "fonts-simple")
            all_files.append("fonts_simple.jsonl")
            estimated_tokens += len(fonts_svgs) * TOKENS_PER_SVG

    stack_svgs: list[str] = []
    stack_name = next(supp_iter, None)
    if stack_name:
        stack_max = dcfg.get("stack_max_samples", 50000)
        stack_frac = dcfg.get("stack_subsample_fraction", 1.0)
        stack_svgs = load_and_collect(
            stack_name,
            subsample_fraction=stack_frac if stack_frac < 1.0 else None,
            max_samples=stack_max,
            seed=seed,
        )
        if stack_svgs:
            save_raw_svgs(stack_svgs, raw_dir / "stack_simple.jsonl", "stack-simple")
            all_files.append("stack_simple.jsonl")
            estimated_tokens += len(stack_svgs) * TOKENS_PER_SVG

    manifest = {
        "downloaded_files": all_files,
        "estimated_tokens_rough": estimated_tokens,
        "target_tokens": dcfg["target_train_tokens"],
        "note": "Actual token count computed after cleaning + tokenization in scripts 02-04",
    }
    manifest_path = raw_dir / "download_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest saved to {manifest_path}")
    print(f"Files: {all_files}")
    total_svgs = len(icons_svgs) + len(emoji_svgs) + len(fonts_svgs) + len(stack_svgs)
    print(f"\nBreakdown by source:")
    print(f"  icons  : {len(icons_svgs):,}")
    print(f"  emoji  : {len(emoji_svgs):,}")
    print(f"  fonts  : {len(fonts_svgs):,}")
    print(f"  stack  : {len(stack_svgs):,}")
    print(f"  TOTAL  : {total_svgs:,}")
    print(f"\nEstimated tokens (rough): {estimated_tokens:,}")
    print("\nNext: python scripts/02_clean_normalize.py --config configs/data_config_balanced.yaml")


if __name__ == "__main__":
    main()
