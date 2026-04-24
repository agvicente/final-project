#!/usr/bin/env python3
"""Generate publication-quality figures for the TEDA High-Dim paper.

Reads CSV results from experiments 1 and 2, and produces three PDF figures
suitable for IEEE conference papers.

Usage:
    cd experiments/teda-high-dim
    ../streaming/venv/bin/python experiments/generate_paper_figures.py
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,  # TrueType fonts in PDF (editable text)
    "ps.fonttype": 42,
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "paper_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SWEEP_CSV = RESULTS_DIR / "exp01_dimensional_sweep.csv"
ABLATION_CSV = RESULTS_DIR / "exp02_ablation.csv"


def ci95(data):
    """Compute 95% confidence interval half-width."""
    n = len(data)
    if n < 2:
        return 0.0
    se = stats.sem(data)
    return se * stats.t.ppf(0.975, n - 1)


# ===================================================================
# Figure 1: Dimensional Sweep — FPR vs d, 2x2 grid by r0
# ===================================================================
def fig_dimensional_sweep():
    print("Generating fig_dimensional_sweep.pdf ...")
    df = pd.read_csv(SWEEP_CSV)
    d_values = sorted(df["d"].unique())
    r0_values = [0.001, 0.1, 1.0, 10.0]

    fig, axes = plt.subplots(2, 2, figsize=(7, 5), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, r0 in enumerate(r0_values):
        ax = axes_flat[idx]
        sub = df[df["r0"] == r0]

        for variant, color, marker, ls, label in [
            ("V0_original", "#d62728", "o", "-", "V0 (original)"),
            ("V7_full_corrected", "#2ca02c", "s", "--", "V7 (corrected)"),
        ]:
            vs = sub[sub["variant"] == variant]
            means = []
            cis = []
            for d in d_values:
                fpr_vals = vs[vs["d"] == d]["FPR"].values
                means.append(np.mean(fpr_vals))
                cis.append(ci95(fpr_vals))

            means = np.array(means)
            cis = np.array(cis)

            ax.plot(
                range(len(d_values)), means,
                color=color, marker=marker, markersize=5,
                linestyle=ls, linewidth=1.2, label=label,
            )
            ax.fill_between(
                range(len(d_values)),
                means - cis, means + cis,
                alpha=0.15, color=color,
            )

        ax.set_xticks(range(len(d_values)))
        ax.set_xticklabels([str(d) for d in d_values])
        ax.set_title(f"$r_0 = {r0}$", fontsize=10)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.3, linewidth=0.5)

        # Highlight r0=0.1 panel with bold border
        if r0 == 0.1:
            for spine in ax.spines.values():
                spine.set_linewidth(2.5)
                spine.set_edgecolor("black")

    # Legend in first panel only
    axes_flat[0].legend(loc="upper right", framealpha=0.9, edgecolor="gray")

    # Shared axis labels
    fig.text(0.5, 0.01, "Dimensionality $d$", ha="center", fontsize=10)
    fig.text(0.01, 0.5, "False Positive Rate (FPR)", va="center",
             rotation="vertical", fontsize=10)

    fig.tight_layout(rect=[0.03, 0.03, 1, 1])
    out = OUTPUT_DIR / "fig_dimensional_sweep.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# ===================================================================
# Figure 2: Ablation Boxplot
# ===================================================================
def fig_ablation_boxplot():
    print("Generating fig_ablation_boxplot.pdf ...")
    df = pd.read_csv(ABLATION_CSV)

    # Map variant codes to short labels
    label_map = {
        "V0_original":        "V0\norig.",
        "V1_welford_var":     "V1\nWelf.",
        "V2_consistent_ecc":  "V2\necc.",
        "V3_welford_and_ecc": "V3\nW+E",
        "V4_selective_update": "V4\nsel.",
        "V5_n1_guard":        "V5\nn1",
        "V6_n2_guard":        "V6\nn2",
        "V7_full_corrected":  "V7\nfull",
    }

    # Compute median FPR per variant for ordering
    variant_medians = df.groupby("variant")["FPR"].median().sort_values()
    ordered_variants = variant_medians.index.tolist()

    # Assign colors by FPR group
    def get_color(median_fpr):
        if median_fpr < 0.05:
            return "#2ca02c"  # green
        elif median_fpr < 0.80:
            return "#ff7f0e"  # orange
        else:
            return "#d62728"  # red

    colors = [get_color(variant_medians[v]) for v in ordered_variants]

    # Prepare data in order
    data = [df[df["variant"] == v]["FPR"].values for v in ordered_variants]
    labels = [label_map.get(v, v) for v in ordered_variants]

    fig, ax = plt.subplots(figsize=(6, 3.5))

    # Draw colored background strips behind each box to show group membership
    # even when boxes collapse to lines (zero or near-zero variance)
    strip_width = 0.7
    for i, color in enumerate(colors):
        ax.axvspan(
            i + 1 - strip_width / 2, i + 1 + strip_width / 2,
            color=color, alpha=0.12, zorder=0,
        )

    bp = ax.boxplot(
        data, tick_labels=labels, patch_artist=True, widths=0.55,
        medianprops=dict(color="white", linewidth=1.5),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
        flierprops=dict(marker="o", markersize=4, alpha=0.6,
                        markerfacecolor="gray", markeredgecolor="gray"),
        boxprops=dict(linewidth=0.8),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor("black")

    # Annotate mean FPR above each box
    for i, (v, d_vals) in enumerate(zip(ordered_variants, data)):
        mean_fpr = np.mean(d_vals)
        # Position above the upper whisker or max outlier
        upper = np.max(d_vals)
        ax.text(
            i + 1, upper + 0.03,
            f"{mean_fpr:.1%}",
            ha="center", va="bottom", fontsize=7.5, fontweight="bold",
        )

    # 5% threshold line
    ax.axhline(y=0.05, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(
        len(ordered_variants) + 0.3, 0.05, "5%",
        va="center", ha="left", fontsize=8, color="gray",
    )

    # Group labels with braces at the bottom
    ax.annotate(
        "", xy=(0.6, -0.12), xytext=(5.4, -0.12),
        xycoords=("data", "axes fraction"),
        textcoords=("data", "axes fraction"),
        arrowprops=dict(arrowstyle="-", color="#2ca02c", lw=2.5),
    )
    ax.annotate(
        "", xy=(5.6, -0.12), xytext=(6.4, -0.12),
        xycoords=("data", "axes fraction"),
        textcoords=("data", "axes fraction"),
        arrowprops=dict(arrowstyle="-", color="#ff7f0e", lw=2.5),
    )
    ax.annotate(
        "", xy=(6.6, -0.12), xytext=(8.4, -0.12),
        xycoords=("data", "axes fraction"),
        textcoords=("data", "axes fraction"),
        arrowprops=dict(arrowstyle="-", color="#d62728", lw=2.5),
    )

    ax.set_ylabel("False Positive Rate (FPR)")
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    out = OUTPUT_DIR / "fig_ablation_boxplot.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# ===================================================================
# Figure 3: Synthetic vs Real (C04) Validation
# ===================================================================
def fig_synthetic_vs_real():
    print("Generating fig_synthetic_vs_real.pdf ...")
    df_sweep = pd.read_csv(SWEEP_CSV)

    # Synthetic data: d=17, r0=0.1
    synth = df_sweep[(df_sweep["d"] == 17) & (df_sweep["r0"] == 0.1)]

    synth_v0 = synth[synth["variant"] == "V0_original"]["FPR"].values
    synth_v7 = synth[synth["variant"] == "V7_full_corrected"]["FPR"].values

    synth_v0_mean = np.mean(synth_v0)
    synth_v0_ci = ci95(synth_v0)
    synth_v7_mean = np.mean(synth_v7)
    synth_v7_ci = ci95(synth_v7)

    # Real C04 values (from paper Table V, r0=0.10, 30 runs)
    real_v0 = 0.544  # 54.4%
    real_v7 = 0.039  # 3.9%

    fig, ax = plt.subplots(figsize=(4, 3))

    x = np.array([0, 1])
    width = 0.30

    # Synthetic bars
    synth_means = [synth_v0_mean, synth_v7_mean]
    synth_cis = [synth_v0_ci, synth_v7_ci]
    bars_synth = ax.bar(
        x - width / 2, synth_means, width,
        yerr=synth_cis, capsize=4,
        color="#1f77b4", alpha=0.8, edgecolor="black", linewidth=0.5,
        label="Synthetic ($d{=}17$, $r_0{=}0.1$)",
    )

    # Real bars
    real_means = [real_v0, real_v7]
    bars_real = ax.bar(
        x + width / 2, real_means, width,
        color="#ff7f0e", alpha=0.8, edgecolor="black", linewidth=0.5,
        label="Real CICIoT2023 (C04)",
    )

    # Annotate bars
    for bar, val in zip(bars_synth, synth_means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + synth_cis[0] + 0.015,
            f"{val:.1%}",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )
    for bar, val in zip(bars_real, real_means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{val:.1%}",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["V0 (original)", "V7 (corrected)"])
    ax.set_ylabel("False Positive Rate (FPR)")
    ax.set_ylim(0, 0.65)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9, edgecolor="gray")
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    out = OUTPUT_DIR / "fig_synthetic_vs_real.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print(f"Reading data from: {RESULTS_DIR}")
    print(f"Saving figures to: {OUTPUT_DIR}\n")

    fig_dimensional_sweep()
    fig_ablation_boxplot()
    fig_synthetic_vs_real()

    print("\nAll figures generated successfully.")
