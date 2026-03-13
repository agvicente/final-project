#!/usr/bin/env python3
"""Generate publication-quality plots for Campaign-01 results."""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

# ── Style ──────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "serif",
})

RESULTS_DIR = Path(__file__).parent
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


# ── Data Loading ───────────────────────────────────────────────────
def load_all_experiments():
    """Load detection_results.json from every experiment directory."""
    experiments = {}
    for d in sorted(RESULTS_DIR.iterdir()):
        results_file = d / "detection_results.json"
        if d.is_dir() and results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
            # Add run_meta if available
            meta_file = d / "run_meta.json"
            if meta_file.exists():
                with open(meta_file) as f:
                    data["run_meta"] = json.load(f)
            experiments[d.name] = data
    return experiments


def build_summary_table(experiments):
    """Build a structured summary from all experiments."""
    rows = []
    for name, data in experiments.items():
        pm = data.get("prequential_metrics", {})
        ds = data.get("detector_stats", {})

        # Parse scenario
        if name.startswith("A1"):
            scenario, attack_type = "A1", "None (Benign)"
        elif name.startswith("A3"):
            scenario, attack_type = "A3", "DDoS-ICMP"
        else:
            scenario = "A2"
            if "ddos" in name:
                attack_type = "DDoS-ICMP"
            elif "syn" in name:
                attack_type = "DDoS-SYN"
            elif "tcp" in name:
                attack_type = "DDoS-TCP"
            elif "mirai" in name:
                attack_type = "Mirai"
            elif "recon" in name:
                attack_type = "Recon-PortScan"
            else:
                attack_type = "Unknown"

        # Parse r0
        if "r0_" in name:
            r0 = float(name.split("r0_")[1])
        else:
            r0 = 0.10  # A3 default

        rows.append({
            "name": name,
            "scenario": scenario,
            "attack_type": attack_type,
            "algorithm": data.get("algorithm", "unknown"),
            "r0": r0,
            "flows": data["flows_processed"],
            "anomalies": data["anomalies_detected"],
            "anomaly_rate": data["anomaly_rate"],
            "precision": pm.get("precision", 0) * 100,
            "recall": pm.get("recall", 0) * 100,
            "f1": pm.get("f1", 0) * 100,
            "fpr": pm.get("fpr", 0) * 100,
            "tp": pm.get("tp", 0),
            "fp": pm.get("fp", 0),
            "tn": pm.get("tn", 0),
            "fn": pm.get("fn", 0),
            "clusters": ds.get("num_clusters", 0),
            "throughput": data.get("flows_per_second", 0),
        })
    return rows


# ── Plot 1: Anomaly Rate Comparison (the key finding) ─────────────
def plot_anomaly_rate(rows):
    """Bar chart: anomaly rate is invariant across attack types."""
    # Use r0=0.10 for consistency (or best available)
    best = {}
    for r in rows:
        if r["scenario"] == "A3":
            continue
        key = (r["scenario"], r["attack_type"])
        if key not in best or r["r0"] == 0.10:
            best[key] = r

    labels = []
    rates = []
    colors = []
    for (sc, at), r in sorted(best.items()):
        label = at if sc == "A2" else "Benign Only"
        labels.append(label)
        rates.append(r["anomaly_rate"])
        colors.append("#4CAF50" if sc == "A1" else "#E53935")

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, rates, color=colors, edgecolor="white", linewidth=1.5, width=0.6)

    # Reference line at 3.5%
    ax.axhline(y=3.53, color="#888", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(len(labels) - 0.5, 3.65, "A1 baseline (3.53%)",
            ha="right", va="bottom", fontsize=10, color="#666", style="italic")

    # Value labels
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{rate:.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)

    ax.set_ylabel("Anomaly Rate (%)")
    ax.set_title("Anomaly Rate is Invariant Across Attack Types\n(MicroTEDAclus, r₀ = 0.10)", fontsize=14)
    ax.set_ylim(0, 5.5)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    sns.despine(left=True)

    fig.savefig(PLOTS_DIR / "01_anomaly_rate_invariant.png")
    plt.close()
    print("  ✓ 01_anomaly_rate_invariant.png")


# ── Plot 2: Recall by Attack Type ─────────────────────────────────
def plot_recall_by_attack(rows):
    """Bar chart: recall is critically low for all attack types."""
    a2_rows = [r for r in rows if r["scenario"] == "A2" and r["r0"] == 0.10]
    if not a2_rows:
        a2_rows = [r for r in rows if r["scenario"] == "A2" and r["r0"] == 0.15]

    labels = [r["attack_type"] for r in a2_rows]
    recalls = [r["recall"] for r in a2_rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, recalls, color="#E53935", edgecolor="white", linewidth=1.5, width=0.6)

    # Target line at 80%
    ax.axhline(y=80, color="#4CAF50", linestyle="--", linewidth=2, alpha=0.8)
    ax.text(0.02, 82, "Target: 80%", transform=ax.get_yaxis_transform(),
            fontsize=11, color="#4CAF50", fontweight="bold")

    # Value labels
    for bar, recall in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{recall:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=12)

    ax.set_ylabel("Recall (%)")
    ax.set_title("Recall vs. Target by Attack Type\n(MicroTEDAclus, r₀ = 0.10)", fontsize=14)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    sns.despine(left=True)

    fig.savefig(PLOTS_DIR / "02_recall_by_attack.png")
    plt.close()
    print("  ✓ 02_recall_by_attack.png")


# ── Plot 3: Precision vs Recall scatter ────────────────────────────
def plot_precision_recall(rows):
    """Scatter: precision is moderate but recall is catastrophic."""
    a2_rows = [r for r in rows if r["scenario"] == "A2"]

    fig, ax = plt.subplots(figsize=(8, 6))

    attack_colors = {
        "DDoS-ICMP": "#1976D2",
        "DDoS-SYN": "#7B1FA2",
        "DDoS-TCP": "#F57C00",
        "Mirai": "#D32F2F",
        "Recon-PortScan": "#388E3C",
    }

    for r in a2_rows:
        c = attack_colors.get(r["attack_type"], "#999")
        ax.scatter(r["recall"], r["precision"], color=c, s=120, edgecolor="white",
                   linewidth=1.5, zorder=3, label=r["attack_type"])

    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", framealpha=0.9)

    # Reference lines
    ax.axvline(x=80, color="#4CAF50", linestyle="--", linewidth=1.5, alpha=0.6)
    ax.text(81, 70, "Recall\ntarget", fontsize=9, color="#4CAF50")

    ax.set_xlabel("Recall (%)")
    ax.set_ylabel("Precision (%)")
    ax.set_title("Precision vs. Recall — All A2 Experiments\n(each point = one r₀ configuration)", fontsize=13)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    sns.despine()

    fig.savefig(PLOTS_DIR / "03_precision_recall_scatter.png")
    plt.close()
    print("  ✓ 03_precision_recall_scatter.png")


# ── Plot 4: TEDA vs MicroTEDAclus (A3) ────────────────────────────
def plot_teda_vs_micro(rows):
    """Grouped bar: MicroTEDAclus >> TEDA on all metrics."""
    micro = next(r for r in rows if r["name"] == "A2-ddos-r0_0.10")
    teda = next(r for r in rows if r["name"] == "A3-teda-baseline")

    metrics = ["Anomalies\nDetected", "Precision\n(%)", "Recall\n(%)", "F1\n(%)"]
    micro_vals = [micro["anomalies"], micro["precision"], micro["recall"], micro["f1"]]
    teda_vals = [teda["anomalies"], teda["precision"], teda["recall"], teda["f1"]]

    x = np.arange(len(metrics))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 3]})

    # Left: anomalies count (different scale)
    ax1 = axes[0]
    b1 = ax1.bar(0 - width / 2, micro["anomalies"], width, label="MicroTEDAclus", color="#1976D2")
    b2 = ax1.bar(0 + width / 2, teda["anomalies"], width, label="TEDA", color="#FF7043")
    ax1.set_xticks([0])
    ax1.set_xticklabels(["Anomalies\nDetected"])
    ax1.set_ylabel("Count")
    ax1.bar_label(b1, fontweight="bold")
    ax1.bar_label(b2, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)
    sns.despine(ax=ax1)

    # Right: percentages
    ax2 = axes[1]
    metric_names = ["Precision", "Recall", "F1"]
    micro_pct = [micro["precision"], micro["recall"], micro["f1"]]
    teda_pct = [teda["precision"], teda["recall"], teda["f1"]]
    x2 = np.arange(len(metric_names))

    b3 = ax2.bar(x2 - width / 2, micro_pct, width, label="MicroTEDAclus", color="#1976D2")
    b4 = ax2.bar(x2 + width / 2, teda_pct, width, label="TEDA", color="#FF7043")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metric_names)
    ax2.set_ylabel("(%)")
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax2.bar_label(b3, fmt="%.1f%%", fontweight="bold", fontsize=9)
    ax2.bar_label(b4, fmt="%.1f%%", fontweight="bold", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)
    sns.despine(ax=ax2)

    fig.suptitle("A3: TEDA vs. MicroTEDAclus — Same Scenario (DDoS-ICMP, r₀ = 0.10)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    fig.savefig(PLOTS_DIR / "04_teda_vs_microteda.png")
    plt.close()
    print("  ✓ 04_teda_vs_microteda.png")


# ── Plot 5: FPR across r0 values (A1) ─────────────────────────────
def plot_fpr_vs_r0(rows):
    """Line chart: FPR is stable across r0 values."""
    a1_rows = sorted([r for r in rows if r["scenario"] == "A1"], key=lambda r: r["r0"])

    r0_vals = [r["r0"] for r in a1_rows]
    fpr_vals = [r["anomaly_rate"] for r in a1_rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(r0_vals, fpr_vals, "o-", color="#1976D2", linewidth=2.5, markersize=10,
            markerfacecolor="white", markeredgewidth=2.5)

    # Target line
    ax.axhline(y=5.0, color="#E53935", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(0.205, 5.1, "Target ≤ 5%", fontsize=10, color="#E53935", fontweight="bold")

    # Value labels
    for r0, fpr in zip(r0_vals, fpr_vals):
        ax.annotate(f"{fpr:.2f}%", (r0, fpr), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontweight="bold", fontsize=11)

    ax.set_xlabel("r₀ (initial variance)")
    ax.set_ylabel("False Positive Rate (%)")
    ax.set_title("A1: FPR Stability Across r₀ Values\n(Benign traffic only — MicroTEDAclus)", fontsize=13)
    ax.set_ylim(2.5, 6.0)
    ax.set_xlim(0.03, 0.22)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    sns.despine()

    fig.savefig(PLOTS_DIR / "05_fpr_vs_r0.png")
    plt.close()
    print("  ✓ 05_fpr_vs_r0.png")


# ── Plot 6: Confusion matrix heatmap (summary) ────────────────────
def plot_confusion_summary(rows):
    """Heatmap of TP/FP/TN/FN for each attack type."""
    a2_r010 = [r for r in rows if r["scenario"] == "A2" and r["r0"] == 0.10]
    if not a2_r010:
        a2_r010 = [r for r in rows if r["scenario"] == "A2" and r["r0"] == 0.15]

    attacks = [r["attack_type"] for r in a2_r010]
    tp = [r["tp"] for r in a2_r010]
    fp = [r["fp"] for r in a2_r010]
    fn = [r["fn"] for r in a2_r010]
    tn = [r["tn"] for r in a2_r010]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(attacks))
    width = 0.2

    ax.bar(x - 1.5 * width, tp, width, label="TP (attack detected)", color="#4CAF50")
    ax.bar(x - 0.5 * width, fp, width, label="FP (benign flagged)", color="#FFC107")
    ax.bar(x + 0.5 * width, fn, width, label="FN (attack missed)", color="#E53935")
    ax.bar(x + 1.5 * width, tn, width, label="TN (benign correct)", color="#1976D2")

    ax.set_xticks(x)
    ax.set_xticklabels(attacks)
    ax.set_ylabel("Number of Flows")
    ax.set_title("Classification Breakdown by Attack Type\n(MicroTEDAclus, r₀ = 0.10)", fontsize=13)
    ax.legend(loc="upper right", fontsize=9)
    sns.despine(left=True)

    fig.savefig(PLOTS_DIR / "06_confusion_breakdown.png")
    plt.close()
    print("  ✓ 06_confusion_breakdown.png")


# ── Plot 7: Campaign summary dashboard ────────────────────────────
def plot_dashboard(rows):
    """4-panel dashboard summarizing the campaign."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Anomaly rate comparison
    ax = axes[0, 0]
    best = {}
    for r in rows:
        if r["scenario"] == "A3":
            continue
        key = r["attack_type"]
        if key not in best or r["r0"] == 0.10:
            best[key] = r
    labels = list(best.keys())
    rates = [best[k]["anomaly_rate"] for k in labels]
    colors = ["#4CAF50" if "Benign" in k else "#E53935" for k in labels]
    bars = ax.barh(labels, rates, color=colors, edgecolor="white")
    ax.axvline(x=3.53, color="#888", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Anomaly Rate (%)")
    ax.set_title("(a) Anomaly Rate ≈ Constant", fontweight="bold")
    for bar, rate in zip(bars, rates):
        ax.text(rate + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{rate:.2f}%", va="center", fontsize=10)
    ax.set_xlim(0, 5.5)

    # Panel 2: Recall vs target
    ax = axes[0, 1]
    a2_best = {r["attack_type"]: r for r in rows
               if r["scenario"] == "A2" and r["r0"] == 0.10}
    if not a2_best:
        a2_best = {r["attack_type"]: r for r in rows
                   if r["scenario"] == "A2" and r["r0"] == 0.15}
    attacks = list(a2_best.keys())
    recalls = [a2_best[a]["recall"] for a in attacks]
    bars = ax.barh(attacks, recalls, color="#E53935", edgecolor="white")
    ax.axvline(x=80, color="#4CAF50", linestyle="--", linewidth=2, alpha=0.7)
    ax.set_xlabel("Recall (%)")
    ax.set_title("(b) Recall << 80% Target", fontweight="bold")
    ax.set_xlim(0, 100)
    for bar, recall in zip(bars, recalls):
        ax.text(recall + 1, bar.get_y() + bar.get_height() / 2,
                f"{recall:.1f}%", va="center", fontsize=10, fontweight="bold")

    # Panel 3: TEDA vs MicroTEDAclus
    ax = axes[1, 0]
    micro = next(r for r in rows if r["name"] == "A2-ddos-r0_0.10")
    teda = next(r for r in rows if r["name"] == "A3-teda-baseline")
    metrics_names = ["Precision", "Recall", "F1"]
    micro_vals = [micro["precision"], micro["recall"], micro["f1"]]
    teda_vals = [teda["precision"], teda["recall"], teda["f1"]]
    x = np.arange(len(metrics_names))
    w = 0.35
    ax.bar(x - w / 2, micro_vals, w, label="MicroTEDAclus", color="#1976D2")
    ax.bar(x + w / 2, teda_vals, w, label="TEDA", color="#FF7043")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_ylabel("(%)")
    ax.set_title("(c) MicroTEDAclus >> TEDA", fontweight="bold")
    ax.legend(fontsize=9)

    # Panel 4: FPR vs r0
    ax = axes[1, 1]
    a1_rows = sorted([r for r in rows if r["scenario"] == "A1"], key=lambda r: r["r0"])
    r0_vals = [r["r0"] for r in a1_rows]
    fpr_vals = [r["anomaly_rate"] for r in a1_rows]
    ax.plot(r0_vals, fpr_vals, "o-", color="#1976D2", linewidth=2.5, markersize=10,
            markerfacecolor="white", markeredgewidth=2.5)
    ax.axhline(y=5.0, color="#E53935", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("r₀")
    ax.set_ylabel("FPR (%)")
    ax.set_title("(d) FPR Stable Across r₀", fontweight="bold")
    ax.set_ylim(2.5, 6.0)
    for r0, fpr in zip(r0_vals, fpr_vals):
        ax.annotate(f"{fpr:.2f}%", (r0, fpr), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=10)

    for ax in axes.flat:
        sns.despine(ax=ax)

    fig.suptitle("Campaign-01 Results Summary — MicroTEDAclus on CICIoT2023",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "07_campaign_dashboard.png")
    plt.close()
    print("  ✓ 07_campaign_dashboard.png")


# ── Main ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading experiments...")
    experiments = load_all_experiments()
    rows = build_summary_table(experiments)
    print(f"  {len(rows)} experiments loaded\n")

    print("Generating plots:")
    plot_anomaly_rate(rows)
    plot_recall_by_attack(rows)
    plot_precision_recall(rows)
    plot_teda_vs_micro(rows)
    plot_fpr_vs_r0(rows)
    plot_confusion_summary(rows)
    plot_dashboard(rows)

    print(f"\nAll plots saved to: {PLOTS_DIR}/")
