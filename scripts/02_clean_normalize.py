"""
02_clean_normalize.py
---------------------
Read raw SVG JSONL files produced by 01_download_data.py, run the full
cleaning pipeline (see svg_utils.py), and save cleaned SVGs to disk.

Cleaning pipeline (applied to every SVG in order):
  1. Strip XML comments
  2. Strip <?xml?> processing instructions
  3. Strip <metadata>, <desc>, <title> blocks
  4. Extract <svg>...</svg> (discard anything outside)
  5. Round floats to 1 decimal place
  6. Collapse whitespace
  7. Validate as valid XML (lxml strict)
  8. Length filter (discard if < 50 characters)
  9. MD5 deduplication

Outputs:
  - outputs/data/cleaned/cleaned.jsonl  — one cleaned SVG per line
  - outputs/data/dataset_stats.json     — cleaning statistics

Usage:
    python scripts/02_clean_normalize.py [--config configs/data_config.yaml]
"""

import argparse
import json
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.svg_utils import clean_svg, md5_hash


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iter_jsonl(path: Path):
    """Yield (svg_string, source_name) from a .jsonl file."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                yield obj.get("svg", ""), obj.get("source", "unknown")
            except json.JSONDecodeError:
                continue


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str = "configs/data_config.yaml") -> None:
    config_file = REPO_ROOT / config_path
    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    cleaning_cfg = cfg["cleaning"]
    paths_cfg = cfg["paths"]

    raw_dir = REPO_ROOT / paths_cfg["raw_data_dir"]
    cleaned_dir = REPO_ROOT / paths_cfg["cleaned_data_dir"]
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    stats_file = REPO_ROOT / paths_cfg["stats_file"]
    stats_file.parent.mkdir(parents=True, exist_ok=True)

    # Load download manifest to know which files to process
    manifest_path = raw_dir / "download_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        raw_files = [raw_dir / fname for fname in manifest["downloaded_files"]]
    else:
        # Fallback: process all .jsonl files in raw_dir
        raw_files = sorted(raw_dir.glob("*.jsonl"))
        print(f"WARNING: No manifest found. Processing all JSONL files in {raw_dir}")

    if not raw_files:
        print("ERROR: No raw files found. Run 01_download_data.py first.")
        sys.exit(1)

    print(f"Processing {len(raw_files)} raw file(s):")
    for f in raw_files:
        print(f"  {f}")

    # -----------------------------------------------------------------------
    # Cleaning loop
    # -----------------------------------------------------------------------
    cleaned_path = cleaned_dir / "cleaned.jsonl"

    aggregate_stats = {
        "files_processed": [str(p) for p in raw_files],
        "total_input": 0,
        "total_output": 0,
        "removed": {
            "no_svg_root": 0,
            "invalid_xml": 0,
            "too_short": 0,
            "duplicate": 0,
        },
    }

    seen_hashes: set[str] = set()

    with open(cleaned_path, "w", encoding="utf-8") as out_f:
        for raw_file in raw_files:
            if not raw_file.exists():
                print(f"  WARNING: {raw_file} not found, skipping.")
                continue

            print(f"\nCleaning {raw_file.name} ...")
            file_in = 0
            file_ok = 0

            for raw_svg, source in tqdm(iter_jsonl(raw_file), desc=f"  {raw_file.stem}"):
                file_in += 1
                aggregate_stats["total_input"] += 1

                cleaned, reason = clean_svg(
                    raw_svg,
                    decimal_places=cleaning_cfg["decimal_places"],
                    min_length_chars=cleaning_cfg["min_length_chars"],
                    seen_hashes=seen_hashes if cleaning_cfg["deduplicate"] else None,
                )

                if cleaned is not None:
                    out_f.write(json.dumps({"svg": cleaned, "source": source}) + "\n")
                    file_ok += 1
                    aggregate_stats["total_output"] += 1
                else:
                    aggregate_stats["removed"][reason] = (
                        aggregate_stats["removed"].get(reason, 0) + 1
                    )

            print(f"  {raw_file.name}: {file_in:,} in → {file_ok:,} kept "
                  f"({file_in - file_ok:,} removed)")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total_in = aggregate_stats["total_input"]
    total_out = aggregate_stats["total_output"]
    removed = aggregate_stats["removed"]

    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)
    print(f"Total input SVGs:       {total_in:>10,}")
    print(f"Total kept SVGs:        {total_out:>10,}")
    print(f"Total removed:          {total_in - total_out:>10,}  ({100*(total_in-total_out)/max(total_in,1):.1f}%)")
    print(f"  - No <svg> root:      {removed.get('no_svg_root', 0):>10,}")
    print(f"  - Invalid XML:        {removed.get('invalid_xml', 0):>10,}")
    print(f"  - Too short (<{cleaning_cfg['min_length_chars']} chars): {removed.get('too_short', 0):>10,}")
    print(f"  - Duplicates:         {removed.get('duplicate', 0):>10,}")
    print(f"\nCleaned output: {cleaned_path}")

    # Save stats JSON
    with open(stats_file, "w") as f:
        json.dump(aggregate_stats, f, indent=2)
    print(f"Stats saved to: {stats_file}")

    print("\nNext: run scripts/03_train_tokenizer.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and normalize SVG files")
    parser.add_argument("--config", default="configs/data_config.yaml")
    args = parser.parse_args()
    main(args.config)
