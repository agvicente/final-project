#!/usr/bin/env python3
"""Experiment 2: Ablation Study -- Contribution of each adaptation.

Tests all 8 variants (V0-V7) at d=17 to isolate the individual
contribution of each of the 5 adaptations to FPR reduction.

Statistical tests (both reported for robustness):
    - Parametric: ANOVA one-way + Tukey HSD post-hoc
    - Non-parametric: Friedman + Nemenyi post-hoc (Demsar 2006)

Protocol:
    - Dimension: d=17 (fixed, IoT real case)
    - Algorithms: V0, V1, V2, V3, V4, V5, V6, V7
    - Repetitions: 30 per variant (seeds 0-29)
    - Samples: 1000 per run (950 normal + 50 anomalies)
    - Anomaly scale: 5 sigma
    - r0 = 0.001

Outputs:
    - results/exp02_ablation.csv
    - results/exp02_statistical_tests.txt
    - results/plots/exp02_ablation_fpr.png
    - results/plots/exp02_ablation_heatmap.png
    - results/plots/exp02_cd_diagram.png
"""

import sys
import time
from io import StringIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from teda_hd.algorithms.variants import create_all_variants, VARIANT_CONFIGS
from teda_hd.generators.gaussian import GaussianStreamGenerator
from teda_hd.metrics.detection import compute_metrics


# ── Configuration ───────────────────────────────────────────────

D = 17
ANOMALY_SCALE = 5.0
N_SEEDS = 30
N_NORMAL = 950
N_ANOMALIES = 50
R0 = 1.0  # Calibrated for N(0,I_d): allows clusters to grow past n=2
          # so that ALL adaptations (not just guard n=1) can differentiate.
          # With r0=0.001, guard n<3 dominates everything and masks
          # the effects of Welford, selective update, etc.

VARIANT_NAMES = list(VARIANT_CONFIGS.keys())

RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"


# ── Experiment runner ───────────────────────────────────────────

def run_single(variant_name: str, algo, seed: int) -> dict:
    """Run a single experiment condition and return metrics."""
    gen = GaussianStreamGenerator(
        d=D,
        n_normal=N_NORMAL,
        n_anomalies=N_ANOMALIES,
        anomaly_scale=ANOMALY_SCALE,
        seed=seed,
    )
    X, y_true = gen.generate()

    algo.reset()

    y_pred = np.zeros(len(X), dtype=int)
    for i, x in enumerate(X):
        result = algo.process(x)
        y_pred[i] = int(result.is_anomaly)

    metrics = compute_metrics(y_true, y_pred)
    metrics["variant"] = variant_name
    metrics["seed"] = seed
    metrics["n_clusters"] = result.num_clusters

    return metrics


def run_experiment():
    """Run the full ablation study experiment."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    total_runs = len(VARIANT_NAMES) * N_SEEDS
    print(f"Experiment 2: Ablation Study (d={D})")
    print(f"  Variants: {VARIANT_NAMES}")
    print(f"  Seeds: 0-{N_SEEDS - 1}")
    print(f"  Anomaly scale: {ANOMALY_SCALE}")
    print(f"  Total runs: {total_runs}")
    print()

    all_variants = create_all_variants(r0=R0)
    results = []
    run_count = 0
    t_start = time.time()

    for variant_name in VARIANT_NAMES:
        algo = all_variants[variant_name]
        for seed in range(N_SEEDS):
            run_count += 1
            if seed % 10 == 0 or seed == N_SEEDS - 1:
                print(
                    f"  Running {variant_name}, seed={seed + 1}/{N_SEEDS}... "
                    f"[{run_count}/{total_runs}]"
                )

            metrics = run_single(variant_name, algo, seed)
            results.append(metrics)

    elapsed = time.time() - t_start

    # Save CSV
    df = pd.DataFrame(results)
    csv_cols = [
        "variant", "seed",
        "FPR", "Recall", "F1", "Precision",
        "TP", "FP", "FN", "TN",
        "n_clusters", "anomaly_rate",
    ]
    df = df[csv_cols]
    csv_path = RESULTS_DIR / "exp02_ablation.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(f"Total time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary: FPR by variant")
    print("=" * 60)
    for metric_name in ["FPR", "Recall"]:
        print(f"\n  {metric_name}:")
        summary = df.groupby("variant")[metric_name].agg(["mean", "std", "min", "max"])
        summary = summary.sort_values("mean", ascending=(metric_name == "FPR"))
        for vname, row in summary.iterrows():
            print(f"    {vname:25s}: {row['mean']:.4f} +/- {row['std']:.4f}  "
                  f"[{row['min']:.4f}, {row['max']:.4f}]")

    # Statistical tests
    stat_report = run_statistical_tests(df)

    # Save statistical tests report
    stat_path = RESULTS_DIR / "exp02_statistical_tests.txt"
    with open(stat_path, "w") as f:
        f.write(stat_report)
    print(f"\nStatistical tests saved to {stat_path}")

    # Generate plots
    generate_plots(df)

    return df


# ── Statistical tests ──────────────────────────────────────────

def run_statistical_tests(df: pd.DataFrame) -> str:
    """Run ANOVA + Tukey HSD and Friedman + Nemenyi tests.

    Returns a formatted report string.
    """
    report = StringIO()
    report.write("=" * 70 + "\n")
    report.write("STATISTICAL TESTS -- Experiment 2: Ablation Study\n")
    report.write(f"Dimension: d={D}, Anomaly scale: {ANOMALY_SCALE}sigma\n")
    report.write(f"Seeds: {N_SEEDS}, Variants: {len(VARIANT_NAMES)}\n")
    report.write("=" * 70 + "\n\n")

    # Prepare data: FPR values grouped by variant
    groups = []
    for vname in VARIANT_NAMES:
        fpr_vals = df[df["variant"] == vname]["FPR"].values
        groups.append(fpr_vals)

    # ── 1. Parametric: ANOVA one-way ────────────────────────
    report.write("-" * 70 + "\n")
    report.write("1. PARAMETRIC: ANOVA one-way + Tukey HSD\n")
    report.write("-" * 70 + "\n\n")

    f_stat, anova_p = stats.f_oneway(*groups)
    report.write(f"ANOVA F-statistic: {f_stat:.4f}\n")
    report.write(f"ANOVA p-value: {anova_p:.6e}\n")
    report.write(f"Significant (p < 0.05): {'YES' if anova_p < 0.05 else 'NO'}\n\n")

    # Tukey HSD
    tukey_pvalues = _tukey_hsd(groups, VARIANT_NAMES)
    report.write("Tukey HSD pairwise p-values:\n")
    report.write(f"{'':25s} " + " ".join(f"{v[:6]:>8s}" for v in VARIANT_NAMES) + "\n")
    for i, vi in enumerate(VARIANT_NAMES):
        row = f"{vi:25s} "
        for j, vj in enumerate(VARIANT_NAMES):
            if i == j:
                row += "     --- "
            elif (vi, vj) in tukey_pvalues:
                p = tukey_pvalues[(vi, vj)]
                row += f"  {p:.4f} "
            elif (vj, vi) in tukey_pvalues:
                p = tukey_pvalues[(vj, vi)]
                row += f"  {p:.4f} "
            else:
                row += "     N/A "
        report.write(row + "\n")

    # ── 2. Non-parametric: Friedman + Nemenyi ───────────────
    report.write("\n" + "-" * 70 + "\n")
    report.write("2. NON-PARAMETRIC: Friedman + Nemenyi (Demsar 2006)\n")
    report.write("-" * 70 + "\n\n")

    # For Friedman, we need the data organized as a matrix:
    # rows = seeds (treated as "datasets"), columns = variants
    fpr_matrix = np.column_stack(groups)  # (N_SEEDS, n_variants)

    friedman_stat, friedman_p = stats.friedmanchisquare(*[fpr_matrix[:, i] for i in range(fpr_matrix.shape[1])])
    report.write(f"Friedman chi-square: {friedman_stat:.4f}\n")
    report.write(f"Friedman p-value: {friedman_p:.6e}\n")
    report.write(f"Significant (p < 0.05): {'YES' if friedman_p < 0.05 else 'NO'}\n\n")

    # Nemenyi post-hoc (manual implementation per Demsar 2006)
    k = len(VARIANT_NAMES)  # number of algorithms
    N = N_SEEDS             # number of "datasets" (seeds)

    # Compute average ranks
    ranks = np.zeros_like(fpr_matrix)
    for i in range(N):
        ranks[i, :] = stats.rankdata(fpr_matrix[i, :])
    avg_ranks = ranks.mean(axis=0)

    report.write("Average ranks (lower = better FPR):\n")
    rank_order = np.argsort(avg_ranks)
    for idx in rank_order:
        report.write(f"  {VARIANT_NAMES[idx]:25s}: {avg_ranks[idx]:.3f}\n")

    # Critical Difference: CD = q_alpha * sqrt(k*(k+1)/(6*N))
    # q_alpha values for Nemenyi test at alpha=0.05 (from Demsar 2006 Table 5)
    # For k groups (using Studentized Range / sqrt(2)):
    q_alpha_table = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
    }
    q_alpha = q_alpha_table.get(k, 3.031)  # default to k=8

    CD = q_alpha * np.sqrt(k * (k + 1) / (6 * N))
    report.write(f"\nCritical Difference (alpha=0.05): CD = {CD:.4f}\n")
    report.write(f"  q_alpha({k}) = {q_alpha}, k = {k}, N = {N}\n\n")

    # Identify significant pairs
    report.write("Nemenyi post-hoc: pairs with |rank_i - rank_j| > CD are significant\n")
    sig_pairs = []
    nonsig_pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            diff = abs(avg_ranks[i] - avg_ranks[j])
            is_sig = diff > CD
            pair_info = (VARIANT_NAMES[i], VARIANT_NAMES[j], diff, is_sig)
            if is_sig:
                sig_pairs.append(pair_info)
            else:
                nonsig_pairs.append(pair_info)

    report.write(f"\nSignificant differences ({len(sig_pairs)} pairs):\n")
    for vi, vj, diff, _ in sorted(sig_pairs, key=lambda x: -x[2]):
        report.write(f"  {vi} vs {vj}: |rank_diff| = {diff:.3f} > CD = {CD:.4f}\n")

    report.write(f"\nNot significant ({len(nonsig_pairs)} pairs):\n")
    for vi, vj, diff, _ in sorted(nonsig_pairs, key=lambda x: -x[2]):
        report.write(f"  {vi} vs {vj}: |rank_diff| = {diff:.3f} <= CD = {CD:.4f}\n")

    # ── 3. Agreement ────────────────────────────────────────
    report.write("\n" + "-" * 70 + "\n")
    report.write("3. AGREEMENT\n")
    report.write("-" * 70 + "\n\n")
    both_sig = anova_p < 0.05 and friedman_p < 0.05
    report.write(f"ANOVA significant: {'YES' if anova_p < 0.05 else 'NO'}\n")
    report.write(f"Friedman significant: {'YES' if friedman_p < 0.05 else 'NO'}\n")
    report.write(f"Both agree: {'YES' if both_sig else 'NO'}\n")
    if both_sig:
        report.write("Conclusion: Result is robust (both parametric and non-parametric agree).\n")
    else:
        report.write("Conclusion: Results diverge. Prioritize Friedman (non-parametric, more robust).\n")

    return report.getvalue()


def _tukey_hsd(groups: list, names: list) -> dict:
    """Compute Tukey HSD pairwise p-values.

    Uses scipy.stats.tukey_hsd (available in scipy >= 1.8).
    Falls back to manual computation if not available.
    """
    try:
        result = stats.tukey_hsd(*groups)
        pvalues = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pvalues[(names[i], names[j])] = result.pvalue[i][j]
        return pvalues
    except AttributeError:
        # Fallback: manual Tukey HSD using studentized range approximation
        return _tukey_hsd_manual(groups, names)


def _tukey_hsd_manual(groups: list, names: list) -> dict:
    """Manual Tukey HSD approximation when scipy version lacks tukey_hsd."""
    k = len(groups)
    all_data = np.concatenate(groups)
    N = len(all_data)
    n_per = [len(g) for g in groups]
    means = [g.mean() for g in groups]

    # MSE (within-group)
    ss_within = sum(np.sum((g - g.mean()) ** 2) for g in groups)
    df_within = N - k
    mse = ss_within / df_within

    pvalues = {}
    for i in range(k):
        for j in range(i + 1, k):
            diff = abs(means[i] - means[j])
            se = np.sqrt(mse * (1.0 / n_per[i] + 1.0 / n_per[j]) / 2.0)
            q_stat = diff / se if se > 0 else float("inf")
            # Approximate p-value using normal distribution (conservative)
            # True Tukey uses studentized range, but this is acceptable for reporting
            p = 2 * (1 - stats.norm.cdf(q_stat / np.sqrt(2)))
            p = min(p * k * (k - 1) / 2, 1.0)  # Bonferroni-like correction
            pvalues[(names[i], names[j])] = p

    return pvalues


# ── Plotting ────────────────────────────────────────────────────

def generate_plots(df: pd.DataFrame):
    """Generate all plots for Experiment 2."""
    print("\nGenerating plots...")
    _plot_ablation_boxplot(df)
    _plot_tukey_heatmap(df)
    _plot_cd_diagram(df)


def _plot_ablation_boxplot(df: pd.DataFrame):
    """Boxplot of FPR by variant."""
    fig, ax = plt.subplots(figsize=(12, 6))

    variant_data = []
    labels = []
    means_order = df.groupby("variant")["FPR"].mean().sort_values()

    for vname in means_order.index:
        variant_data.append(df[df["variant"] == vname]["FPR"].values)
        # Shorter labels for readability
        short = vname.replace("_", "\n", 1).replace("_", " ")
        labels.append(short)

    bp = ax.boxplot(
        variant_data, labels=labels, patch_artist=True,
        notch=True, showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="red", markersize=5),
    )

    # Color coding: V0 red, V7 green, others blue shades
    colors = []
    for vname in means_order.index:
        if "V0" in vname:
            colors.append("#d62728")
        elif "V7" in vname:
            colors.append("#2ca02c")
        else:
            colors.append("#1f77b4")

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("False Positive Rate (FPR)", fontsize=12)
    ax.set_title(
        f"Ablation Study: FPR by Variant (d={D}, {ANOMALY_SCALE:.0f}sigma)",
        fontsize=14, fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(fontsize=9)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "exp02_ablation_fpr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {PLOTS_DIR / 'exp02_ablation_fpr.png'}")


def _plot_tukey_heatmap(df: pd.DataFrame):
    """Heatmap of Tukey HSD p-values."""
    groups = []
    for vname in VARIANT_NAMES:
        groups.append(df[df["variant"] == vname]["FPR"].values)

    tukey_pvalues = _tukey_hsd(groups, VARIANT_NAMES)

    k = len(VARIANT_NAMES)
    pval_matrix = np.ones((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            key = (VARIANT_NAMES[i], VARIANT_NAMES[j])
            if key in tukey_pvalues:
                p = tukey_pvalues[key]
            else:
                p = tukey_pvalues.get((VARIANT_NAMES[j], VARIANT_NAMES[i]), 1.0)
            pval_matrix[i, j] = p
            pval_matrix[j, i] = p

    fig, ax = plt.subplots(figsize=(10, 8))

    short_names = [v.replace("_", "\n", 1) for v in VARIANT_NAMES]
    im = ax.imshow(pval_matrix, cmap="RdYlGn", vmin=0, vmax=0.1, aspect="auto")

    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(short_names, fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(short_names, fontsize=8)

    # Annotate cells
    for i in range(k):
        for j in range(k):
            if i != j:
                p = pval_matrix[i, j]
                text = f"{p:.3f}"
                color = "white" if p < 0.025 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, label="p-value", shrink=0.8)
    ax.set_title(
        "Tukey HSD Pairwise p-values\n(green = not significant, red = significant)",
        fontsize=13, fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "exp02_ablation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {PLOTS_DIR / 'exp02_ablation_heatmap.png'}")


def _plot_cd_diagram(df: pd.DataFrame):
    """Simplified Critical Difference diagram (Demsar 2006).

    Ranks variants by average rank, draws the CD bar, and connects
    groups that are NOT significantly different.
    """
    # Compute ranks per seed
    groups = []
    for vname in VARIANT_NAMES:
        groups.append(df[df["variant"] == vname]["FPR"].values)

    fpr_matrix = np.column_stack(groups)  # (N_SEEDS, k)
    k = len(VARIANT_NAMES)
    N = N_SEEDS

    ranks = np.zeros_like(fpr_matrix)
    for i in range(N):
        ranks[i, :] = stats.rankdata(fpr_matrix[i, :])

    avg_ranks = ranks.mean(axis=0)

    # Critical Difference
    q_alpha_table = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
    }
    q_alpha = q_alpha_table.get(k, 3.031)
    CD = q_alpha * np.sqrt(k * (k + 1) / (6 * N))

    # Sort by average rank
    sorted_idx = np.argsort(avg_ranks)
    sorted_names = [VARIANT_NAMES[i] for i in sorted_idx]
    sorted_ranks = avg_ranks[sorted_idx]

    # Find groups of non-significantly-different algorithms
    cliques = []
    for i in range(k):
        clique = [i]
        for j in range(i + 1, k):
            if sorted_ranks[j] - sorted_ranks[i] <= CD:
                clique.append(j)
        if len(clique) > 1:
            # Only add if this clique is not a subset of an existing one
            is_subset = False
            for existing in cliques:
                if set(clique).issubset(set(existing)):
                    is_subset = True
                    break
            if not is_subset:
                # Remove any existing cliques that are subsets of this one
                cliques = [c for c in cliques if not set(c).issubset(set(clique))]
                cliques.append(clique)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))

    # Draw axis at the top
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(0, k + 2)
    ax.invert_yaxis()

    # Draw rank axis
    ax.axhline(y=0.8, xmin=0, xmax=1, color="black", linewidth=1.5)
    for r in range(1, k + 1):
        ax.plot(r, 0.8, "|", color="black", markersize=10)
        ax.text(r, 0.3, str(r), ha="center", va="bottom", fontsize=10)

    ax.text((k + 1) / 2, 0.0, "Average Rank", ha="center", va="bottom",
            fontsize=12, fontweight="bold")

    # Draw CD bar
    cd_x_start = 0.7
    cd_x_end = cd_x_start + CD
    ax.annotate(
        "", xy=(cd_x_end, 1.3), xytext=(cd_x_start, 1.3),
        arrowprops=dict(arrowstyle="<->", color="red", lw=2),
    )
    ax.text(
        (cd_x_start + cd_x_end) / 2, 1.1, f"CD = {CD:.2f}",
        ha="center", va="bottom", fontsize=10, color="red", fontweight="bold",
    )

    # Draw algorithm names and ranks
    left_algos = sorted_idx[:k // 2]
    right_algos = sorted_idx[k // 2:]

    y_pos = 2.0
    y_step = 0.7

    # Left side (best ranks)
    for rank_pos, algo_idx in enumerate(left_algos):
        y = y_pos + rank_pos * y_step
        r = avg_ranks[algo_idx]
        name = VARIANT_NAMES[algo_idx].replace("_", " ", 1)
        ax.plot(r, 0.8, "o", color="#2ca02c" if "V7" in name else "#1f77b4",
                markersize=8, zorder=5)
        ax.plot([r, r], [0.8, y], color="gray", linewidth=0.8, linestyle="--")
        ax.text(0.3, y, f"{name} ({r:.2f})", ha="left", va="center", fontsize=9)

    # Right side (worst ranks)
    for rank_pos, algo_idx in enumerate(right_algos):
        y = y_pos + rank_pos * y_step
        r = avg_ranks[algo_idx]
        name = VARIANT_NAMES[algo_idx].replace("_", " ", 1)
        ax.plot(r, 0.8, "o", color="#d62728" if "V0" in name else "#1f77b4",
                markersize=8, zorder=5)
        ax.plot([r, r], [0.8, y], color="gray", linewidth=0.8, linestyle="--")
        ax.text(k + 0.7, y, f"({r:.2f}) {name}", ha="right", va="center", fontsize=9)

    # Draw clique bars (connecting non-significantly-different groups)
    clique_y_base = y_pos + k * y_step * 0.6
    for ci, clique in enumerate(cliques):
        clique_ranks = [sorted_ranks[i] for i in clique]
        y_bar = clique_y_base + ci * 0.3
        ax.plot(
            [min(clique_ranks), max(clique_ranks)], [y_bar, y_bar],
            color="black", linewidth=3, solid_capstyle="round",
        )

    ax.set_title(
        f"Critical Difference Diagram (d={D}, {ANOMALY_SCALE:.0f}sigma, Nemenyi alpha=0.05)",
        fontsize=13, fontweight="bold",
    )
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "exp02_cd_diagram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {PLOTS_DIR / 'exp02_cd_diagram.png'}")


# ── Main ────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_experiment()
