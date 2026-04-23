#!/usr/bin/env python3
"""Experiment 1: Dimensional Sweep -- FPR vs dimensionality.

Shows how FPR degrades progressively with d, comparing V0 (original)
vs V7 (full corrected). Produces the "collapse curve" -- the central
figure of the paper.

Protocol:
    - Dimensions: d in {2, 5, 10, 15, 17, 20, 30, 50}
    - Algorithms: V0 (original) and V7 (full corrected)
    - Anomaly scales: 3 sigma, 5 sigma, 10 sigma
    - Repetitions: 30 per condition (seeds 0-29)
    - Samples: 1000 per run (950 normal + 50 anomalies)
    - r0 = 0.001

Outputs:
    - results/exp01_dimensional_sweep.csv
    - results/plots/exp01_fpr_vs_d.png
    - results/plots/exp01_recall_vs_d.png
    - results/plots/exp01_synthetic_vs_real.png
"""

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from teda_hd.algorithms.variants import create_variant
from teda_hd.generators.gaussian import GaussianStreamGenerator
from teda_hd.metrics.detection import compute_metrics


# ── Configuration ───────────────────────────────────────────────

DIMENSIONS = [2, 5, 10, 15, 17, 20, 30, 50]
VARIANTS = ["V0_original", "V7_full_corrected"]
ANOMALY_SCALE = 5.0
R0_VALUES = [0.001, 0.1, 1.0, 10.0]
N_SEEDS = 30
N_NORMAL = 950
N_ANOMALIES = 50

# C04 real-data reference values at d=17 for synthetic-to-real validation
C04_REAL = {
    "V0_original": {"FPR": 0.544},       # ~54.4%
    "V7_full_corrected": {"FPR": 0.039},  # ~3.9%
}

RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"


# ── Experiment runner ───────────────────────────────────────────

def run_single(d: int, variant_name: str, r0: float, seed: int) -> dict:
    """Run a single experiment condition and return metrics."""
    gen = GaussianStreamGenerator(
        d=d,
        n_normal=N_NORMAL,
        n_anomalies=N_ANOMALIES,
        anomaly_scale=ANOMALY_SCALE,
        seed=seed,
    )
    X, y_true = gen.generate()

    algo = create_variant(variant_name, r0=r0)

    y_pred = np.zeros(len(X), dtype=int)
    for i, x in enumerate(X):
        result = algo.process(x)
        y_pred[i] = int(result.is_anomaly)

    metrics = compute_metrics(y_true, y_pred)
    metrics["d"] = d
    metrics["variant"] = variant_name
    metrics["r0"] = r0
    metrics["seed"] = seed
    metrics["n_clusters"] = result.num_clusters

    return metrics


def run_experiment():
    """Run the full dimensional sweep experiment."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    total_runs = len(DIMENSIONS) * len(VARIANTS) * len(R0_VALUES) * N_SEEDS
    print(f"Experiment 1: Dimensional Sweep (v2 — r0 sweep)")
    print(f"  Dimensions: {DIMENSIONS}")
    print(f"  Variants: {VARIANTS}")
    print(f"  r0 values: {R0_VALUES}")
    print(f"  Anomaly scale: {ANOMALY_SCALE} (fixed)")
    print(f"  Seeds: 0-{N_SEEDS - 1}")
    print(f"  Total runs: {total_runs}")
    print()

    results = []
    run_count = 0
    t_start = time.time()

    for d in DIMENSIONS:
        for variant_name in VARIANTS:
            for r0 in R0_VALUES:
                for seed in range(N_SEEDS):
                    run_count += 1
                    if seed % 10 == 0 or seed == N_SEEDS - 1:
                        print(
                            f"  Running d={d}, {variant_name}, "
                            f"r0={r0}, seed={seed + 1}/{N_SEEDS}... "
                            f"[{run_count}/{total_runs}]"
                        )

                    metrics = run_single(d, variant_name, r0, seed)
                    results.append(metrics)

    elapsed = time.time() - t_start

    # Save CSV
    df = pd.DataFrame(results)
    csv_cols = [
        "d", "variant", "r0", "seed",
        "FPR", "Recall", "F1", "Precision",
        "TP", "FP", "FN", "TN",
        "n_clusters", "anomaly_rate",
    ]
    df = df[csv_cols]
    csv_path = RESULTS_DIR / "exp01_dimensional_sweep.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(f"Total time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    # ── Synthetic-to-real validation at d=17 ────────────────
    print("\n" + "=" * 60)
    print("Synthetic-to-Real Validation (d=17)")
    print("=" * 60)
    print("  C04 real data used r0=0.10. Comparing with r0=0.1 synthetic:")
    df17 = df[(df["d"] == 17) & (df["r0"] == 0.1)]
    for vname in VARIANTS:
        subset = df17[df17["variant"] == vname]
        synth_fpr = subset["FPR"].mean()
        synth_std = subset["FPR"].std()
        real_fpr = C04_REAL[vname]["FPR"]
        diff_pp = abs(synth_fpr - real_fpr) * 100
        status = "PASS" if diff_pp <= 15 else "WARN"
        print(
            f"  {vname}: synthetic={synth_fpr:.3f} +/- {synth_std:.3f}, "
            f"C04 real={real_fpr:.3f}, diff={diff_pp:.1f}pp [{status}]"
        )
    print()
    print("  Note: divergence is expected. Synthetic uses N(0,I_d),")
    print("  real IoT data has heterogeneous feature scales.")
    print("  The key is qualitative agreement: V0 >> V7 in both.")

    # ── Generate plots ──────────────────────────────────────
    generate_plots(df)

    return df


# ── Plotting ────────────────────────────────────────────────────

def _plot_metric_vs_d(df: pd.DataFrame, metric: str, ylabel: str, filename: str):
    """Plot metric vs d for V0 vs V7 with 95% CI. One subplot per r0 value."""
    n_r0 = len(R0_VALUES)
    fig, axes = plt.subplots(1, n_r0, figsize=(5 * n_r0, 5), sharey=True)
    if n_r0 == 1:
        axes = [axes]

    for ax_idx, r0 in enumerate(R0_VALUES):
        ax = axes[ax_idx]
        df_r0 = df[df["r0"] == r0]

        for variant_name, style in [
            ("V0_original", {"color": "#d62728", "marker": "o", "linestyle": "-"}),
            ("V7_full_corrected", {"color": "#2ca02c", "marker": "s", "linestyle": "--"}),
        ]:
            subset = df_r0[df_r0["variant"] == variant_name]
            grouped = subset.groupby("d")[metric]
            means = grouped.mean()
            stds = grouped.std()
            n = grouped.count()
            ci95 = 1.96 * stds / np.sqrt(n)

            label_short = "V0 (original)" if "V0" in variant_name else "V7 (corrected)"
            ax.errorbar(
                means.index, means.values, yerr=ci95.values,
                label=label_short, capsize=3, capthick=1.5,
                linewidth=2, markersize=6, **style,
            )

        ax.set_xlabel("Dimensionality (d)", fontsize=12)
        ax.set_title(f"r\u2080 = {r0}", fontsize=13,
                      fontweight="bold" if r0 == 0.1 else "normal")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(DIMENSIONS)

    axes[0].set_ylabel(ylabel, fontsize=12)
    axes[0].legend(fontsize=10, loc="best")

    fig.suptitle(
        f"{ylabel} vs Dimensionality -- V0 (original) vs V7 (corrected)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {PLOTS_DIR / filename}")


def _plot_synthetic_vs_real(df: pd.DataFrame):
    """Bar plot comparing synthetic d=17 FPR with C04 real data (r0=0.1)."""
    df17 = df[(df["d"] == 17) & (df["r0"] == 0.1)]

    fig, ax = plt.subplots(figsize=(8, 5))

    variants = ["V0_original", "V7_full_corrected"]
    labels_short = ["V0 (original)", "V7 (corrected)"]
    x = np.arange(len(variants))
    width = 0.35

    synth_means = []
    synth_cis = []
    real_vals = []

    for vname in variants:
        subset = df17[df17["variant"] == vname]
        m = subset["FPR"].mean()
        s = subset["FPR"].std()
        ci = 1.96 * s / np.sqrt(len(subset))
        synth_means.append(m)
        synth_cis.append(ci)
        real_vals.append(C04_REAL[vname]["FPR"])

    bars1 = ax.bar(
        x - width / 2, synth_means, width,
        yerr=synth_cis, capsize=5, label="Synthetic (d=17, r\u2080=0.1)",
        color="#1f77b4", alpha=0.85, edgecolor="black",
    )
    bars2 = ax.bar(
        x + width / 2, real_vals, width,
        label="C04 Real (CICIoT2023)",
        color="#ff7f0e", alpha=0.85, edgecolor="black",
    )

    ax.set_ylabel("False Positive Rate (FPR)", fontsize=12)
    ax.set_title("Synthetic vs Real FPR at d=17", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value annotations
    for bar, val in zip(bars1, synth_means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.1%}", ha="center", va="bottom", fontsize=10,
        )
    for bar, val in zip(bars2, real_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.1%}", ha="center", va="bottom", fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "exp01_synthetic_vs_real.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {PLOTS_DIR / 'exp01_synthetic_vs_real.png'}")


def generate_plots(df: pd.DataFrame):
    """Generate all plots for Experiment 1."""
    print("\nGenerating plots...")
    _plot_metric_vs_d(df, "FPR", "False Positive Rate (FPR)", "exp01_fpr_vs_d.png")
    _plot_metric_vs_d(df, "Recall", "Recall (TPR)", "exp01_recall_vs_d.png")
    _plot_synthetic_vs_real(df)


# ── Main ────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_experiment()
