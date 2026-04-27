"""
12_plot_samples.py
------------------
Generate the Phase 4 figures from rendered samples + manifest.

Plots:
  1. Unconditional sample grid (2×5) — outputs/plots/samples_unconditional_grid.png
  2. Temperature comparison — same prefix at 3 temperatures side-by-side
     outputs/plots/samples_temperature_comparison.png
  3. Prefix completion — rendered prefix vs rendered completion (one row per prefix)
     outputs/plots/samples_prefix_completion.png

Reads:
  outputs/samples/manifest.json
  outputs/samples/{unconditional,prefix}/*.svg
  outputs/samples/rendered/{unconditional,prefix}/*.png

If a sample failed to render, that cell shows the SVG code as text instead so
the figure stays informative even when XML/render rates are low.

Usage:
    python scripts/12_plot_samples.py
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.image import imread

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def render_svg_string_to_array(svg: str, output_size: int = 256):
    """Render an SVG string to a numpy image array. Returns None on failure."""
    try:
        import cairosvg
        png_bytes = cairosvg.svg2png(
            bytestring=svg.encode("utf-8"),
            output_width=output_size,
        )
        return imread(io.BytesIO(png_bytes), format="png")
    except Exception:
        return None


def show_image_or_text(ax, png_path: Path | None, svg_text: str, title: str):
    """Show a PNG if it exists, else fall back to the truncated SVG code."""
    img = None
    if png_path is not None and png_path.exists():
        try:
            img = imread(png_path)
        except Exception:
            img = None

    if img is not None:
        ax.imshow(img)
    else:
        snippet = (svg_text[:120] + "...") if len(svg_text) > 120 else svg_text
        ax.text(0.5, 0.5, snippet, ha="center", va="center",
                fontsize=6, family="monospace", wrap=True,
                transform=ax.transAxes)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def find_rendered_path(rendered_dir: Path, svg_filename: str) -> Path:
    """Map foo.svg → rendered/foo.png (in the corresponding subdir)."""
    return rendered_dir / Path(svg_filename).with_suffix(".png").name


# ---------------------------------------------------------------------------
# Plot 1: Unconditional grid
# ---------------------------------------------------------------------------

def plot_unconditional_grid(
    manifest: dict,
    samples_dir: Path,
    rendered_dir: Path,
    out_path: Path,
    rows: int = 2,
    cols: int = 5,
) -> None:
    items = manifest.get("unconditional", [])
    n = rows * cols
    items = items[:n]
    if not items:
        print("No unconditional samples to plot.")
        return

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.8))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, item in zip(axes, items):
        svg_path = samples_dir / item["file"]
        rel = Path(item["file"])
        png_path = rendered_dir / rel.parent.name / rel.with_suffix(".png").name
        svg_text = svg_path.read_text(encoding="utf-8", errors="replace") \
                   if svg_path.exists() else ""
        title = f"t={item['temperature']:.1f}  ({item['n_tokens']} tok)"
        show_image_or_text(ax, png_path, svg_text, title)

    # Hide unused cells
    for ax in axes[len(items):]:
        ax.axis("off")

    fig.suptitle(f"Unconditional Samples (n={len(items)})", fontsize=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: Temperature comparison
# ---------------------------------------------------------------------------

def plot_temperature_comparison(
    manifest: dict,
    samples_dir: Path,
    rendered_dir: Path,
    out_path: Path,
) -> None:
    """One row per prefix, columns = temperatures."""
    prefix_items = manifest.get("prefix", [])
    if not prefix_items:
        print("No prefix samples to plot.")
        return

    # Group by prefix_index
    by_prefix: dict[int, list[dict]] = {}
    for it in prefix_items:
        by_prefix.setdefault(it["prefix_index"], []).append(it)

    temps = sorted({it["temperature"] for it in prefix_items})
    prefix_indices = sorted(by_prefix.keys())

    n_rows = len(prefix_indices)
    n_cols = len(temps)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.6, n_rows * 2.6),
                             squeeze=False)

    for r, pi in enumerate(prefix_indices):
        items_by_temp = {it["temperature"]: it for it in by_prefix[pi]}
        for c, t in enumerate(temps):
            ax = axes[r][c]
            it = items_by_temp.get(t)
            if it is None:
                ax.axis("off")
                continue
            svg_path = samples_dir / it["file"]
            rel = Path(it["file"])
            png_path = rendered_dir / rel.parent.name / rel.with_suffix(".png").name
            svg_text = svg_path.read_text(encoding="utf-8", errors="replace") \
                       if svg_path.exists() else ""
            title = f"prefix {pi}  t={t}"
            show_image_or_text(ax, png_path, svg_text, title)

    fig.suptitle("Prefix completions across temperatures", fontsize=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: Prefix completion (prefix-only render vs full completion)
# ---------------------------------------------------------------------------

def plot_prefix_completion(
    manifest: dict,
    samples_dir: Path,
    rendered_dir: Path,
    out_path: Path,
    temperature: float = 0.8,
) -> None:
    """Two columns per prefix: rendered prefix-as-is vs rendered completion.

    The rendered "prefix" cell uses the prefix string with </svg> appended so
    cairosvg can parse it; if it still fails to render, the cell shows the
    prefix code instead."""
    prefix_items = [it for it in manifest.get("prefix", [])
                    if abs(it["temperature"] - temperature) < 1e-6]
    if not prefix_items:
        # Fall back to whatever temperature exists
        prefix_items = manifest.get("prefix", [])
        if prefix_items:
            temperature = prefix_items[0]["temperature"]
            prefix_items = [it for it in manifest.get("prefix", [])
                            if abs(it["temperature"] - temperature) < 1e-6]
    if not prefix_items:
        print("No prefix samples to plot.")
        return

    prefix_items = sorted(prefix_items, key=lambda x: x["prefix_index"])

    n_rows = len(prefix_items)
    fig, axes = plt.subplots(n_rows, 2, figsize=(6, n_rows * 2.6),
                             squeeze=False)

    for r, it in enumerate(prefix_items):
        # Left: rendered prefix (close it so cairosvg can parse)
        prefix_str = it["prefix"]
        if "</svg>" not in prefix_str:
            prefix_render = prefix_str + "</svg>"
        else:
            prefix_render = prefix_str
        img = render_svg_string_to_array(prefix_render)
        if img is not None:
            axes[r][0].imshow(img)
        else:
            snippet = prefix_str if len(prefix_str) <= 120 else prefix_str[:120] + "..."
            axes[r][0].text(0.5, 0.5, snippet, ha="center", va="center",
                            fontsize=6, family="monospace", wrap=True,
                            transform=axes[r][0].transAxes)
        axes[r][0].set_title(f"Prefix {it['prefix_index']} (input)", fontsize=9)
        axes[r][0].axis("off")

        # Right: rendered completion
        svg_path = samples_dir / it["file"]
        rel = Path(it["file"])
        png_path = rendered_dir / rel.parent.name / rel.with_suffix(".png").name
        svg_text = svg_path.read_text(encoding="utf-8", errors="replace") \
                   if svg_path.exists() else ""
        show_image_or_text(axes[r][1], png_path, svg_text,
                           f"Completion (t={it['temperature']})")

    fig.suptitle(f"Prefix → Completion (t={temperature})", fontsize=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_dir", default="outputs/samples")
    ap.add_argument("--rendered_dir", default="outputs/samples/rendered")
    ap.add_argument("--plots_dir", default="outputs/plots")
    ap.add_argument("--prefix_temperature", type=float, default=0.8,
                    help="Temperature to use in the prefix-completion figure")
    args = ap.parse_args()

    samples_dir  = REPO_ROOT / args.samples_dir
    rendered_dir = REPO_ROOT / args.rendered_dir
    plots_dir    = REPO_ROOT / args.plots_dir

    manifest_path = samples_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found at {manifest_path}. "
                                "Run scripts/10_generate_samples.py first.")
    with open(manifest_path) as f:
        manifest = json.load(f)

    plot_unconditional_grid(
        manifest=manifest,
        samples_dir=samples_dir,
        rendered_dir=rendered_dir,
        out_path=plots_dir / "samples_unconditional_grid.png",
    )
    plot_temperature_comparison(
        manifest=manifest,
        samples_dir=samples_dir,
        rendered_dir=rendered_dir,
        out_path=plots_dir / "samples_temperature_comparison.png",
    )
    plot_prefix_completion(
        manifest=manifest,
        samples_dir=samples_dir,
        rendered_dir=rendered_dir,
        out_path=plots_dir / "samples_prefix_completion.png",
        temperature=args.prefix_temperature,
    )

    print("\nAll Phase 4 plots generated.")


if __name__ == "__main__":
    main()
