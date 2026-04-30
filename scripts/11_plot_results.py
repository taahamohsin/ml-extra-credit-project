"""
11_plot_results.py
------------------
Generate all Phase 2 result plots:
  1. Scaling law plot (log N vs val_loss + power law fit)
  2. Training curves (step vs train_loss for all 5 models)
  3. LR sweep plot (re-generated from saved JSON)
  4. Throughput / stats table (printed + saved as CSV)

Reads from:
  outputs/logs/result_*.json        — per-model final results
  outputs/logs/training_*.csv       — step-level logs
  outputs/logs/lr_sweep_sp.json     — LR sweep results

Writes to:
  outputs/plots/scaling_law_sp.png
  outputs/plots/training_curves.png
  outputs/plots/lr_sweep_sp.png     (re-generated)
  outputs/logs/throughput_table.csv

Usage:
    python scripts/11_plot_results.py
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.scaling_law import fit_scaling_law, plot_scaling_law, print_fit_summary, predict

MODEL_ORDER = ["tiny", "small", "medium", "large", "xl"]


def load_results(log_dir: Path) -> dict[str, dict]:
    results = {}
    for name in MODEL_ORDER:
        p = log_dir / f"result_{name}.json"
        if p.exists():
            with open(p) as f:
                results[name] = json.load(f)
    if not results:
        print("WARNING: No result_*.json files found in outputs/logs/. "
              "Run scripts/05_train_model.py for each model first.")
    return results


def load_training_logs(log_dir: Path) -> dict[str, list[dict]]:
    import csv
    logs = {}
    for name in MODEL_ORDER:
        p = log_dir / f"training_{name}.csv"
        if p.exists():
            with open(p) as f:
                reader = csv.DictReader(f)
                logs[name] = list(reader)
    return logs


def load_lr_sweep(log_dir: Path) -> dict | None:
    p = log_dir / "lr_sweep_sp.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    print("WARNING: lr_sweep_sp.json not found — skipping LR sweep plot.")
    return None


def plot_scaling(results: dict, plots_dir: Path) -> None:
    if len(results) < 2:
        print(f"Need at least 2 trained models for scaling plot; have {len(results)}. Skipping.")
        return

    names         = [n for n in MODEL_ORDER if n in results]
    param_counts  = [results[n]["n_params"]      for n in names]
    val_losses    = [results[n]["best_val_loss"]  for n in names]

    print(f"\nFitting scaling law on {len(names)} models: {names}")
    for n, p, l in zip(names, param_counts, val_losses):
        print(f"  {n:<8}  N={p:>10,}  L={l:.4f}")

    fit = fit_scaling_law(param_counts, val_losses)
    print_fit_summary(fit, label="SP")

    plot_scaling_law(
        param_counts=param_counts,
        val_losses=val_losses,
        fit_result=fit,
        save_path=plots_dir / "scaling_law_sp.png",
        model_names=names,
        title="SVG Scaling Law (Standard Parameterization)\nL = a · N^(−α) + c",
        label="SP",
    )

    max_n = max(param_counts)
    p_10x = predict(fit, max_n * 10)
    print(f"\nExtrapolation (10× XL ≈ {max_n*10/1e6:.0f}M params):")
    print(f"  Predicted L = {p_10x['L_pred']:.4f}  "
          f"95% CI: [{p_10x['L_lower']:.4f}, {p_10x['L_upper']:.4f}]")

    fit_out = {
        "parameterization": "SP",
        "models": names,
        "param_counts": param_counts,
        "val_losses": val_losses,
        "a":      fit["a"],
        "alpha":  fit["alpha"],
        "c":      fit["c"],
        "r_squared": fit["r_squared"],
        "extrapolation_10x": p_10x,
    }
    with open(plots_dir.parent / "logs" / "scaling_fit_sp.json", "w") as f:
        json.dump(fit_out, f, indent=2)


def plot_training_curves(logs: dict, plots_dir: Path) -> None:
    if not logs:
        print("No training logs found — skipping training curves plot.")
        return

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(MODEL_ORDER)))

        for name, color in zip(MODEL_ORDER, colors):
            if name not in logs:
                continue
            rows = logs[name]
            steps = []
            train_losses = []
            for r in rows:
                if r.get("train_loss"):
                    steps.append(int(r["step"]))
                    train_losses.append(float(r["train_loss"]))

            if steps:
                ax.plot(steps, train_losses, label=name, color=color, linewidth=1.5, alpha=0.9)

        ax.set_xlabel("Training step")
        ax.set_ylabel("Train loss (cross-entropy)")
        ax.set_title("Training Curves — All 5 Models (SP, 1 epoch)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = plots_dir / "training_curves.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Training curves saved to {save_path}")
    except Exception as e:
        print(f"WARNING: Could not generate training curves: {e}")


def plot_lr_sweep_from_json(sweep: dict, plots_dir: Path) -> None:
    if sweep is None:
        return
    try:
        import matplotlib.pyplot as plt

        runs = sweep["runs"]
        lrs       = [r["lr"]       for r in runs]
        val_losses = [r["val_loss"] for r in runs]
        diverged   = [r.get("diverged", False) for r in runs]

        colors = ["red" if d else "steelblue" for d in diverged]
        best_lr = sweep["best_lr"]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(lrs, val_losses, c=colors, s=80, zorder=5)
        ax.scatter([best_lr], [sweep["best_val_loss"]],
                   c="gold", s=150, zorder=6, marker="*",
                   label=f"Best: lr={best_lr:.1e}, val={sweep['best_val_loss']:.4f}")
        ax.plot(lrs, val_losses, color="steelblue", linewidth=1, linestyle="--", alpha=0.6)

        ax.set_xscale("log")
        ax.set_xlabel("Learning rate (log scale)")
        ax.set_ylabel("Final validation loss")
        ax.set_title(f"LR Sweep — Tiny model (SP, {sweep['max_steps']} steps)")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()

        save_path = plots_dir / "lr_sweep_sp.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"LR sweep plot saved to {save_path}")
    except Exception as e:
        print(f"WARNING: Could not generate LR sweep plot: {e}")


def print_throughput_table(results: dict, logs: dict, log_dir: Path) -> None:
    import csv

    print("\n" + "=" * 80)
    print(f"{'Model':<10} {'N params':>12} {'Best val L':>12} {'Tok seen':>12} {'Wall (min)':>12}")
    print("-" * 80)

    rows = []
    for name in MODEL_ORDER:
        if name not in results:
            continue
        r = results[name]
        print(
            f"{name:<10} {r['n_params']:>12,} {r['best_val_loss']:>12.4f} "
            f"{r['tokens_seen']:>12,} {r.get('wall_time_min', '?'):>12}"
        )
        rows.append({
            "model":          name,
            "n_params":       r["n_params"],
            "best_val_loss":  r["best_val_loss"],
            "tokens_seen":    r["tokens_seen"],
            "wall_time_min":  r.get("wall_time_min", ""),
            "peak_lr":        r.get("peak_lr", ""),
        })
    print("=" * 80)

    if rows:
        out = log_dir / "throughput_table.csv"
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Throughput table saved to {out}")


def main():
    log_dir   = REPO_ROOT / "outputs" / "logs"
    plots_dir = REPO_ROOT / "outputs" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(log_dir)
    logs    = load_training_logs(log_dir)
    sweep   = load_lr_sweep(log_dir)

    print(f"\nLoaded results for: {list(results.keys())}")
    print(f"Loaded training logs for: {list(logs.keys())}")

    plot_scaling(results, plots_dir)
    plot_training_curves(logs, plots_dir)
    plot_lr_sweep_from_json(sweep, plots_dir)
    print_throughput_table(results, logs, log_dir)

    print("\nAll Phase 2 plots generated.")


if __name__ == "__main__":
    main()
