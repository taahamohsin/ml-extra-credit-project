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
import hashlib
import json
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.svg_utils import clean_svg, md5_hash, is_valid_xml


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


def debug_xml_validation() -> None:
    """
    Verify lxml is doing real XML parsing by running known-good and
    known-bad SVGs through is_valid_xml. Runs unconditionally at startup,
    before any file processing.
    """
    print("\n[DEBUG] XML validation spot-check (runs before any file processing):")
    cases = [
        (True,  "good SVG",           '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/></svg>'),
        (False, "unclosed tag",       '<svg xmlns="http://www.w3.org/2000/svg"><circle cx="12"</svg>'),
        (False, "mismatched tags",    '<svg xmlns="http://www.w3.org/2000/svg"><rect></circle></svg>'),
        (False, "unquoted attribute",  '<svg xmlns="http://www.w3.org/2000/svg"><path d=M0></path></svg>'),
    ]
    all_ok = True
    for expected, label, test_svg in cases:
        result = is_valid_xml(test_svg)
        ok = result == expected
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{status}] {label}: expected={expected}, got={result}")
    if all_ok:
        print("  All checks passed — lxml is correctly parsing XML.")
    else:
        print("  WARNING: Some checks failed — XML validation may not be working correctly!")


def debug_sample(raw_file: Path, min_length_chars: int, sample_size: int = 500) -> None:
    """
    Print diagnostic information about a raw SVG file:
      1. Shortest SVG lengths (to check if any approach the threshold)
      2. Duplicate rate (MD5 on raw strings before cleaning)
      3. Three SVGs nearest to the length threshold
    """
    print(f"\n[DEBUG] Sampling first {sample_size} SVGs from {raw_file.name} ...")

    svgs = []
    for raw_svg, _ in iter_jsonl(raw_file):
        svgs.append(raw_svg)
        if len(svgs) >= sample_size:
            break

    if not svgs:
        print("[DEBUG] No SVGs loaded — file may be empty or unreadable.")
        return

    # 1. Length distribution
    lengths = sorted(len(s) for s in svgs)
    print("\n[DEBUG] Character lengths (raw, before cleaning):")
    print(f"  min={lengths[0]}, p5={lengths[len(lengths)//20]}, "
          f"median={lengths[len(lengths)//2]}, max={lengths[-1]}")
    print(f"  SVGs below {min_length_chars} chars: "
          f"{sum(1 for n in lengths if n < min_length_chars)} / {len(lengths)}")

    # 2. Duplicate check (on raw strings)
    raw_hashes = [hashlib.md5(s.encode()).hexdigest() for s in svgs]
    n_unique = len(set(raw_hashes))
    print("\n[DEBUG] Deduplication (raw strings, before cleaning):")
    print(f"  Unique MD5 hashes: {n_unique} / {len(svgs)} "
          f"({100 * (len(svgs) - n_unique) / len(svgs):.1f}% duplicates)")

    # 3. Three SVGs closest to the length threshold
    by_dist = sorted(svgs, key=lambda s: abs(len(s) - min_length_chars))
    print(f"\n[DEBUG] 3 SVGs nearest to length threshold ({min_length_chars} chars):")
    for i, s in enumerate(by_dist[:3]):
        print(f"  Example {i+1}: raw_len={len(s)}")
        print(f"    {repr(s[:200])}")


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

    # Run XML validation spot-check immediately — before touching any files
    debug_xml_validation()

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
            debug_sample(raw_file, cleaning_cfg["min_length_chars"])
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
