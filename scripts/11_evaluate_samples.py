"""
11_evaluate_samples.py
----------------------
Evaluate generated SVG samples and compute test set perplexity.

Metrics:
  - XML validity rate (lxml.etree.fromstring)
  - SVG render rate (cairosvg.svg2png)
  - Has <svg> root rate
  - Tags-closed rate (proxy: XML parses without unclosed-tag errors)
  - Test-set perplexity from the best model

Renders every valid SVG to PNG under outputs/samples/rendered/ for
downstream plotting. Saves all metrics to outputs/logs/evaluation_metrics.json.

Usage:
    python scripts/11_evaluate_samples.py
    python scripts/11_evaluate_samples.py --skip_perplexity
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from lxml import etree

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model import TransformerLM


def evaluate_one(svg: str) -> dict:
    """Run all per-sample checks on one SVG string. Returns boolean flags."""
    out = {
        "xml_valid":      False,
        "has_svg_root":   False,
        "tags_closed":    False,
        "svg_renderable": False,
    }

    # XML validity (also serves as the tags-closed signal — lxml strict
    # parsing rejects unclosed/mismatched tags with XMLSyntaxError).
    try:
        tree = etree.fromstring(svg.encode("utf-8"))
        out["xml_valid"]   = True
        out["tags_closed"] = True
        tag = tree.tag
        # lxml may return '{namespace}svg' if a default xmlns is declared.
        if isinstance(tag, str) and (tag == "svg" or tag.endswith("}svg")):
            out["has_svg_root"] = True
    except etree.XMLSyntaxError:
        pass

    # Render check (only attempt if XML is valid — saves time)
    if out["xml_valid"]:
        try:
            import cairosvg
            cairosvg.svg2png(bytestring=svg.encode("utf-8"))
            out["svg_renderable"] = True
        except Exception:
            pass

    return out


def render_to_png(svg: str, png_path: Path, output_size: int = 256) -> bool:
    """Render an SVG to a PNG file. Returns True on success."""
    try:
        import cairosvg
        png_path.parent.mkdir(parents=True, exist_ok=True)
        cairosvg.svg2png(
            bytestring=svg.encode("utf-8"),
            write_to=str(png_path),
            output_width=output_size,
        )
        return True
    except Exception:
        return False


def evaluate_directory(
    svg_dir: Path,
    rendered_dir: Path,
) -> dict:
    files = sorted(svg_dir.glob("*.svg"))
    counts = {"total": len(files), "xml_valid": 0, "has_svg_root": 0,
              "tags_closed": 0, "svg_renderable": 0}
    per_file = []

    for f in files:
        svg = f.read_text(encoding="utf-8", errors="replace")
        flags = evaluate_one(svg)
        for k in ("xml_valid", "has_svg_root", "tags_closed", "svg_renderable"):
            counts[k] += int(flags[k])

        png_ok = False
        if flags["svg_renderable"]:
            png_ok = render_to_png(svg, rendered_dir / (f.stem + ".png"))

        per_file.append({
            "file":           f.name,
            "n_chars":        len(svg),
            "xml_valid":      flags["xml_valid"],
            "has_svg_root":   flags["has_svg_root"],
            "tags_closed":    flags["tags_closed"],
            "svg_renderable": flags["svg_renderable"],
            "png_written":    png_ok,
        })

    rates = {
        f"{k}_rate": (counts[k] / counts["total"]) if counts["total"] else 0.0
        for k in ("xml_valid", "has_svg_root", "tags_closed", "svg_renderable")
    }
    return {"counts": counts, "rates": rates, "per_file": per_file}


@torch.no_grad()
def compute_test_perplexity(
    ckpt_path: Path,
    test_bin_path: Path,
    device: torch.device,
    seq_len: int = 1024,
    batch_size: int = 8,
    max_batches: int = 200,
    use_bf16: bool = True,
) -> dict:
    """Non-overlapping windows from test.bin; every token scored at most once."""
    print(f"\n=== Test-set perplexity ===")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Test bin:   {test_bin_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = TransformerLM(cfg)
    state = ckpt["model"]
    cleaned = {
        (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
        for k, v in state.items()
    }
    model.load_state_dict(cleaned)
    model = model.to(device)
    model.eval()

    data = np.memmap(str(test_bin_path), dtype=np.uint16, mode="r")
    n_tokens = len(data)
    window = seq_len + 1  # need seq_len inputs + 1 target
    stride = seq_len

    n_windows_avail = max(0, (n_tokens - window) // stride + 1)
    n_windows = min(max_batches * batch_size, n_windows_avail)
    print(f"  Test tokens: {n_tokens:,}  |  windows: {n_windows} "
          f"× seq_len {seq_len}")

    total_loss = 0.0
    total_count = 0

    ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16 and device.type == "cuda" else torch.amp.autocast(device_type="cpu", enabled=False)
    )

    starts = [i * stride for i in range(n_windows)]
    for b in range(0, len(starts), batch_size):
        chunk_starts = starts[b : b + batch_size]
        batch = np.stack([
            data[s : s + window].astype(np.int64) for s in chunk_starts
        ])
        x = torch.from_numpy(batch[:, :-1]).to(device)
        y = torch.from_numpy(batch[:, 1:]).to(device)

        with ctx:
            logits, _ = model(x)
        # Use float32 for the loss to keep perplexity numerics clean.
        loss = F.cross_entropy(
            logits.float().view(-1, cfg.vocab_size),
            y.view(-1),
            reduction="sum",
        )
        total_loss  += loss.item()
        total_count += y.numel()

    mean_ce = total_loss / max(total_count, 1)
    ppl = float(np.exp(mean_ce))
    print(f"  Mean CE: {mean_ce:.4f}  |  Perplexity: {ppl:.2f}")

    return {
        "mean_cross_entropy": mean_ce,
        "perplexity":         ppl,
        "n_tokens_scored":    total_count,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/checkpoints/xl/best.pt")
    ap.add_argument("--samples_dir", default="outputs/samples")
    ap.add_argument("--rendered_dir", default="outputs/samples/rendered")
    ap.add_argument("--test_bin", default="outputs/data/binary/test.bin")
    ap.add_argument("--metrics_path", default="outputs/logs/evaluation_metrics.json")
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--ppl_batch_size", type=int, default=8)
    ap.add_argument("--ppl_max_batches", type=int, default=200)
    ap.add_argument("--skip_perplexity", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    samples_dir   = REPO_ROOT / args.samples_dir
    rendered_root = REPO_ROOT / args.rendered_dir

    print("\n=== Unconditional samples ===")
    uncond = evaluate_directory(samples_dir / "unconditional",
                                rendered_root / "unconditional")
    print(f"  {uncond['counts']}")
    for k, v in uncond["rates"].items():
        print(f"  {k}: {v*100:.1f}%")

    print("\n=== Prefix-conditioned samples ===")
    prefix = evaluate_directory(samples_dir / "prefix",
                                rendered_root / "prefix")
    print(f"  {prefix['counts']}")
    for k, v in prefix["rates"].items():
        print(f"  {k}: {v*100:.1f}%")

    metrics = {
        "unconditional": uncond,
        "prefix":        prefix,
    }

    combo_counts = {k: uncond["counts"][k] + prefix["counts"][k]
                    for k in uncond["counts"]}
    combo_rates = {
        f"{k}_rate": (combo_counts[k] / combo_counts["total"]) if combo_counts["total"] else 0.0
        for k in ("xml_valid", "has_svg_root", "tags_closed", "svg_renderable")
    }
    metrics["combined"] = {"counts": combo_counts, "rates": combo_rates}
    print("\n=== Combined ===")
    for k, v in combo_rates.items():
        print(f"  {k}: {v*100:.1f}%")

    if not args.skip_perplexity:
        ckpt_path = REPO_ROOT / args.checkpoint
        test_bin  = REPO_ROOT / args.test_bin
        if ckpt_path.exists() and test_bin.exists():
            metrics["perplexity"] = compute_test_perplexity(
                ckpt_path=ckpt_path,
                test_bin_path=test_bin,
                device=device,
                seq_len=args.seq_len,
                batch_size=args.ppl_batch_size,
                max_batches=args.ppl_max_batches,
            )
        else:
            print(f"\nSkipping perplexity — missing checkpoint or test.bin")
            print(f"  ckpt: {ckpt_path}  exists={ckpt_path.exists()}")
            print(f"  test: {test_bin}  exists={test_bin.exists()}")

    metrics_path = REPO_ROOT / args.metrics_path
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
