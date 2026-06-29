#!/usr/bin/env python3
"""Generate paper-quality figures for Experiment 3 (regime transition).

Produces three PDFs in `results/paper_figures/`:
    fig_regime_transition_v7.pdf   FPR & cluster_count vs lambda for V7 across r0,
                                   with predicted lambda* lines.
    fig_regime_v0_vs_v7.pdf        Side-by-side qualitative comparison at r0=0.1.
    fig_regime_phase_diagram.pdf   2D heatmap (lambda x r0) of regime indicator.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_CSV = RESULTS_DIR / "exp03_results.csv"
PAPER_FIG_DIR = RESULTS_DIR / "paper_figures"

D = 17  # match exp03 fixed dimension


def load() -> pd.DataFrame:
    if not RESULTS_CSV.exists():
        print(f"ERROR: {RESULTS_CSV} not found. Run exp03_regime_transition.py --full first.")
        sys.exit(1)
    return pd.read_csv(RESULTS_CSV)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["algorithm", "r0", "lambda"])
        .agg(
            FPR_mean=("FPR", "mean"),
            FPR_sem=("FPR", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
            n_clusters_mean=("n_clusters", "mean"),
            n_clusters_sem=("n_clusters", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
            frac_above_r0_mean=("frac_above_r0", "mean"),
            top1_frac_mean=("top1_frac", "mean"),
            singleton_frac_mean=("singleton_frac", "mean"),
        )
        .reset_index()
    )


# ── Fig 1: Regime transition for V7 ─────────────────────────────

def fig_regime_transition_v7(agg: pd.DataFrame) -> None:
    v7 = agg[agg["algorithm"] == "V7_full_corrected"].copy()
    r0_vals = sorted(v7["r0"].unique())

    # Vertical layout (2 rows, 1 col) for IEEE single-column placement.
    # Width tuned to ~ \columnwidth = 3.5 in; readable axis labels at this size.
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 5.0), sharex=True)
    colors = {1e-3: "#1f77b4", 1e-1: "#ff7f0e", 1.0: "#2ca02c"}

    for r0 in r0_vals:
        sub = v7[v7["r0"] == r0].sort_values("lambda")
        col = colors.get(r0, "gray")
        lam_star = float(np.sqrt(r0 / D))

        # FPR panel (top)
        axes[0].errorbar(
            sub["lambda"], sub["FPR_mean"], yerr=1.96 * sub["FPR_sem"],
            label=f"$r_0={r0:g}$", color=col, marker="o", capsize=2, markersize=4,
        )
        axes[0].axvline(lam_star, color=col, linestyle=":", alpha=0.6)

        # Cluster count panel (bottom)
        axes[1].errorbar(
            sub["lambda"], sub["n_clusters_mean"],
            yerr=1.96 * sub["n_clusters_sem"],
            color=col, marker="o", capsize=2, markersize=4,
        )
        axes[1].axvline(lam_star, color=col, linestyle=":", alpha=0.6)

    for ax in axes:
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=9)

    axes[0].set_ylabel("FPR (mean $\\pm$ 95% CI)", fontsize=10)
    axes[0].legend(loc="best", fontsize=8, ncol=1, framealpha=0.85)

    axes[1].set_ylabel("# micro-clusters", fontsize=10)
    axes[1].set_xlabel(r"Data scale $\lambda$", fontsize=10)
    axes[1].set_yscale("log")

    fig.suptitle(f"V7 phase transition ($d={D}$, 30 seeds)",
                 fontsize=11, y=0.995)
    fig.tight_layout()
    out = PAPER_FIG_DIR / "fig_regime_transition_v7.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Fig 2: V0 vs V7 qualitative comparison ──────────────────────

def fig_regime_v0_vs_v7(agg: pd.DataFrame, r0_target: float = 0.1) -> None:
    sub = agg[np.isclose(agg["r0"], r0_target)]
    v0 = sub[sub["algorithm"] == "V0_original"].sort_values("lambda")
    v7 = sub[sub["algorithm"] == "V7_full_corrected"].sort_values("lambda")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharex=True)

    metrics = [
        ("FPR_mean", "False Positive Rate"),
        ("n_clusters_mean", "# micro-clusters"),
        ("top1_frac_mean", "Top-1 cluster fraction"),
    ]

    lam_star = float(np.sqrt(r0_target / D))

    for ax, (col, title) in zip(axes, metrics):
        ax.plot(v0["lambda"], v0[col], "o-", color="#d62728", label="V0 (original)")
        ax.plot(v7["lambda"], v7[col], "s-", color="#2ca02c", label="V7 (corrected)")
        ax.axvline(lam_star, color="black", linestyle=":", alpha=0.5,
                   label=f"$\\lambda^* = {lam_star:.3f}$")
        ax.set_xscale("log")
        ax.set_xlabel(r"Data scale $\lambda$")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if col == "n_clusters_mean":
            ax.set_yscale("log")

    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle(
        f"V0 vs V7 qualitative regime difference (r0={r0_target}, d={D}, 30 seeds)",
        y=1.02,
    )
    fig.tight_layout()
    out = PAPER_FIG_DIR / "fig_regime_v0_vs_v7.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Fig 3: Phase diagram (lambda x r0 -> regime indicator) ──────

def fig_regime_phase_diagram(agg: pd.DataFrame) -> None:
    v7 = agg[agg["algorithm"] == "V7_full_corrected"].copy()
    if v7.empty:
        print("No V7 data; skipping phase diagram")
        return

    pivot = v7.pivot_table(
        index="r0", columns="lambda", values="frac_above_r0_mean", aggfunc="first"
    ).sort_index()

    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    cmap = LinearSegmentedColormap.from_list(
        "regime", ["#1f77b4", "#cccccc", "#d62728"]
    )
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        extent=[
            np.log10(pivot.columns.min()),
            np.log10(pivot.columns.max()),
            np.log10(pivot.index.min()),
            np.log10(pivot.index.max()),
        ],
    )

    # Overlay predicted boundary: lambda* = sqrt(r0/d)  ->  log10(lambda*) = 0.5*(log10(r0) - log10(d))
    log_r0 = np.linspace(np.log10(pivot.index.min()), np.log10(pivot.index.max()), 100)
    log_lam_star = 0.5 * (log_r0 - np.log10(D))
    ax.plot(log_lam_star, log_r0, "k--", linewidth=2,
            label=r"Predicted: $\lambda^* = \sqrt{r_0 / d}$")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("frac_clusters with $\\sigma^2 > r_0$  (regime indicator)")

    ax.set_xlabel(r"$\log_{10} \lambda$")
    ax.set_ylabel(r"$\log_{10} r_0$")
    ax.set_title(f"Phase diagram: V7 regime vs $(\\lambda, r_0)$ at d={D}, 30 seeds")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    out = PAPER_FIG_DIR / "fig_regime_phase_diagram.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    print(f"Loaded {len(df)} rows from {RESULTS_CSV.name}")
    agg = aggregate(df)
    print(f"Aggregated to {len(agg)} (algorithm, r0, lambda) cells")
    print()

    fig_regime_transition_v7(agg)
    fig_regime_v0_vs_v7(agg, r0_target=0.1)
    fig_regime_phase_diagram(agg)
    print(f"\nAll figures saved to {PAPER_FIG_DIR}")


if __name__ == "__main__":
    main()
