"""
scaling_law.py
--------------
Fit and plot the power law  L = a * N^(-alpha) + c  to (param_count, val_loss)
data points from the 5-scale training experiment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit


def power_law(N: np.ndarray, a: float, alpha: float, c: float) -> np.ndarray:
    """L = a * N^(-alpha) + c"""
    return a * np.power(N, -alpha) + c


def fit_scaling_law(
    param_counts: list[int] | np.ndarray,
    val_losses:   list[float] | np.ndarray,
    p0: tuple[float, float, float] = (10.0, 0.5, 1.0),
    bounds: tuple = ((0, 0, 0), (np.inf, 2.0, np.inf)),
    maxfev: int = 10_000,
) -> dict:
    """
    Fit L = a * N^(-alpha) + c via nonlinear least squares.

    Returns
    -------
    dict with keys:
      a, alpha, c      — fitted parameters
      a_err, alpha_err, c_err — 1-sigma uncertainties from covariance
      r_squared        — coefficient of determination
      popt, pcov       — raw scipy outputs (for downstream use)
    """
    N = np.asarray(param_counts, dtype=float)
    L = np.asarray(val_losses,   dtype=float)

    popt, pcov = curve_fit(
        power_law, N, L,
        p0=p0,
        bounds=bounds,
        maxfev=maxfev,
    )
    a, alpha, c = popt
    perr = np.sqrt(np.diag(pcov))

    L_pred   = power_law(N, *popt)
    ss_res   = np.sum((L - L_pred) ** 2)
    ss_tot   = np.sum((L - L.mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "a":         float(a),
        "alpha":     float(alpha),
        "c":         float(c),
        "a_err":     float(perr[0]),
        "alpha_err": float(perr[1]),
        "c_err":     float(perr[2]),
        "r_squared": float(r_squared),
        "popt":      popt,
        "pcov":      pcov,
    }


def predict(
    fit_result: dict,
    N: int | float,
    confidence: float = 0.95,
) -> dict[str, float]:
    """
    Predict L at a new parameter count N with a confidence interval.

    Uses first-order error propagation from the covariance matrix.
    """
    popt = fit_result["popt"]
    pcov = fit_result["pcov"]
    a, alpha, c = popt

    L_pred = power_law(float(N), a, alpha, c)

    # Jacobian of power_law w.r.t. (a, alpha, c)
    dL_da     =  float(N) ** (-alpha)
    dL_dalpha = -a * float(N) ** (-alpha) * np.log(float(N))
    dL_dc     =  1.0
    J = np.array([dL_da, dL_dalpha, dL_dc])

    # Variance of the prediction
    var = float(J @ pcov @ J.T)
    std = np.sqrt(max(var, 0.0))

    from scipy import stats
    z = stats.norm.ppf((1 + confidence) / 2)

    return {
        "N":         float(N),
        "L_pred":    L_pred,
        "L_lower":   L_pred - z * std,
        "L_upper":   L_pred + z * std,
        "std":       std,
        "confidence": confidence,
    }


def plot_scaling_law(
    param_counts:  list[int] | np.ndarray,
    val_losses:    list[float] | np.ndarray,
    fit_result:    dict,
    save_path:     Optional[Path] = None,
    model_names:   Optional[list[str]] = None,
    title:         str = "Scaling Law: L = a · N^(−α) + c",
    label:         str = "SP",
    ax=None,
    color:         str = "steelblue",
    show:          bool = False,
) -> None:
    """
    Log-log scatter of (N, L) data + fitted power law curve.

    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib

        N = np.asarray(param_counts, dtype=float)
        L = np.asarray(val_losses,   dtype=float)
        popt = fit_result["popt"]
        a, alpha, c = popt

        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=(8, 5))

        ax.scatter(N, L, color=color, s=80, zorder=5,
                   label=f"{label} data")

        if model_names:
            for name, n, l in zip(model_names, N, L):
                ax.annotate(name, (n, l), textcoords="offset points",
                            xytext=(6, 4), fontsize=8)

        N_range = np.logspace(np.log10(N.min() * 0.8), np.log10(N.max() * 1.2), 300)
        L_fit   = power_law(N_range, *popt)
        ax.plot(
            N_range, L_fit,
            color=color, linewidth=2, linestyle="--",
            label=(
                f"{label} fit: L={a:.2f}·N^(−{alpha:.3f})+{c:.3f}  "
                f"(R²={fit_result['r_squared']:.3f})"
            ),
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Non-embedding parameters (N)")
        ax.set_ylabel("Validation loss (cross-entropy)")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

        if own_fig:
            plt.tight_layout()
            if save_path is not None:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"Scaling law plot saved to {save_path}")
            if show:
                plt.show()
            plt.close()

    except Exception as e:
        print(f"WARNING: Could not generate scaling law plot: {e}")


def print_fit_summary(fit_result: dict, label: str = "SP") -> None:
    """Print a human-readable summary of the fit."""
    print(f"\n{'='*50}")
    print(f"Scaling law fit ({label})")
    print(f"{'='*50}")
    print(f"  L = {fit_result['a']:.4f} * N^(-{fit_result['alpha']:.4f}) + {fit_result['c']:.4f}")
    print(f"  a     = {fit_result['a']:.4f}  ±  {fit_result['a_err']:.4f}")
    print(f"  alpha = {fit_result['alpha']:.4f}  ±  {fit_result['alpha_err']:.4f}")
    print(f"  c     = {fit_result['c']:.4f}  ±  {fit_result['c_err']:.4f}")
    print(f"  R²    = {fit_result['r_squared']:.4f}")
    print(f"  (Kaplan et al. NL: alpha ≈ 0.076)")
