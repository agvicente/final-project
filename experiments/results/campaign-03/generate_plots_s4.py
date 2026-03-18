#!/usr/bin/env python3
"""Generate publication-quality plots for Campaign-03 S4 results.

Campaign-03 S4 tests behavioral window features (v2 = 19 features)
against the baseline v1 (12 features) across 5 attack types.

Plots:
  1. Recall v1 vs v2 per attack (grouped bar, w=10s and w=30s)
  2. FPR v1 vs v2 (benign only)
  3. F1 v1 vs v2 per attack
  4. FPR vs r0 for v2 (line chart, w=10s and w=30s)
  5. Dashboard comparativo S3 -> S4
"""

import json
import re
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

CAMPAIGN_02_DIR = RESULTS_DIR.parent / "campaign-02"

# Attack display names
ATTACK_LABELS = {
    "ddos": "DDoS-ICMP",
    "syn": "DDoS-SYN",
    "tcp": "DDoS-TCP",
    "mirai": "Mirai",
    "recon": "Recon",
}

ATTACK_ORDER = ["ddos", "syn", "tcp", "mirai", "recon"]

# Colors
WF_COLORS = {"v1": "#1976D2", "v2": "#FF9800"}
WINDOW_COLORS = {10: "#1976D2", 30: "#FF9800"}


# ── Data Loading ───────────────────────────────────────────────────

def parse_dirname(dirname: str) -> dict:
    """Parse S4 experiment directory name.

    Examples:
        S4-A1-benign-wfv1-w10s-r0_0.10
        S4-A2-ddos-wfv2-w30s-r0_0.05
    """
    info = {"name": dirname, "step": None, "scenario": None, "attack": None,
            "wf": None, "window": None, "r0": None}

    m = re.match(r"(S\d+)-", dirname)
    if m:
        info["step"] = m.group(1)

    if "-A1-" in dirname:
        info["scenario"] = "A1"
    elif "-A2-" in dirname:
        info["scenario"] = "A2"
        for atk in ATTACK_ORDER:
            if f"-{atk}-" in dirname:
                info["attack"] = atk
                break

    m = re.search(r"-wf(v[12])-", dirname)
    if m:
        info["wf"] = m.group(1)

    m = re.search(r"-w(\d+)s-", dirname)
    if m:
        info["window"] = int(m.group(1))

    m = re.search(r"r0_([\d.]+)", dirname)
    if m:
        info["r0"] = float(m.group(1))

    return info


def load_experiments(results_dir: Path) -> list:
    """Load all experiments from a campaign directory."""
    rows = []
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        det_file = d / "detection_results.json"
        meta_file = d / "run_meta.json"
        if not det_file.exists():
            continue

        with open(det_file) as f:
            data = json.load(f)
        meta = {}
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)

        pm = data.get("prequential_metrics", {})
        info = parse_dirname(d.name)

        rows.append({
            **info,
            "precision": pm.get("precision", 0) * 100,
            "recall": pm.get("recall", 0) * 100,
            "f1": pm.get("f1", 0) * 100,
            "fpr": pm.get("fpr", 0) * 100,
            "tp": pm.get("tp", 0),
            "fp": pm.get("fp", 0),
            "tn": pm.get("tn", 0),
            "fn": pm.get("fn", 0),
            "flows": data.get("flows_processed", 0),
            "anomalies": data.get("anomalies_detected", 0),
            "anomaly_rate": data.get("anomaly_rate", 0),
            "meta": meta,
        })
    return rows


def load_campaign02_s3() -> list:
    """Load S3 results from campaign-02 for comparison."""
    if not CAMPAIGN_02_DIR.exists():
        print(f"  WARNING: campaign-02 dir not found: {CAMPAIGN_02_DIR}")
        return []

    rows = []
    for d in sorted(CAMPAIGN_02_DIR.iterdir()):
        if not d.is_dir() or not d.name.startswith("S3-"):
            continue
        det_file = d / "detection_results.json"
        if not det_file.exists():
            continue

        with open(det_file) as f:
            data = json.load(f)
        pm = data.get("prequential_metrics", {})

        name = d.name
        scenario = "A1" if "-A1-" in name else "A2"
        attack = None
        if scenario == "A2":
            for atk in ATTACK_ORDER:
                if f"-{atk}-" in name:
                    attack = atk
                    break

        window = None
        m = re.search(r"-w(\d+)s-", name)
        if m:
            window = int(m.group(1))

        r0 = 0.10
        m = re.search(r"r0_([\d.]+)", name)
        if m:
            r0 = float(m.group(1))

        rows.append({
            "name": name, "step": "S3", "scenario": scenario, "attack": attack,
            "wf": "v1", "window": window, "r0": r0,
            "precision": pm.get("precision", 0) * 100,
            "recall": pm.get("recall", 0) * 100,
            "f1": pm.get("f1", 0) * 100,
            "fpr": pm.get("fpr", 0) * 100,
            "flows": data.get("flows_processed", 0),
        })
    return rows


# ── Plot helpers ───────────────────────────────────────────────────

def _save(fig, name):
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / name, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {name}")


def _get_attack_label(atk):
    return ATTACK_LABELS.get(atk, atk or "Benign")


# ── Plot 1: Recall v1 vs v2 per attack ────────────────────────────

def plot_recall_v1_vs_v2(rows):
    """Grouped bar: Recall per attack, v1 vs v2 at r0=0.10, for each window."""
    for wsec in [10, 30]:
        lookup = {}  # (wf, attack) -> row
        for r in rows:
            if r["scenario"] != "A2" or r["r0"] != 0.10 or r["window"] != wsec:
                continue
            lookup[(r["wf"], r["attack"])] = r

        attacks = [a for a in ATTACK_ORDER
                   if (("v1", a) in lookup or ("v2", a) in lookup)]
        if not attacks:
            print(f"  SKIP plot_recall_v1v2_w{wsec}s (no data)")
            continue

        x = np.arange(len(attacks))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        v1_vals = [lookup.get(("v1", a), {}).get("recall", 0) for a in attacks]
        v2_vals = [lookup.get(("v2", a), {}).get("recall", 0) for a in attacks]

        b1 = ax.bar(x - width / 2, v1_vals, width, label="v1 (12 basic)", color=WF_COLORS["v1"])
        b2 = ax.bar(x + width / 2, v2_vals, width, label="v2 (19 behavioral)", color=WF_COLORS["v2"])

        ax.bar_label(b1, fmt="%.1f%%", fontsize=9, fontweight="bold")
        ax.bar_label(b2, fmt="%.1f%%", fontsize=9, fontweight="bold")

        ax.axhline(y=80, color="#4CAF50", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.text(0.02, 82, "Target: 80%", transform=ax.get_yaxis_transform(),
                fontsize=10, color="#4CAF50", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([_get_attack_label(a) for a in attacks])
        ax.set_ylabel("Recall (%)")
        ax.set_title(f"S4: Recall — Window Features v1 vs v2\n(window={wsec}s, r₀=0.10)", fontsize=13)
        ax.set_ylim(0, 105)
        ax.legend(loc="upper right")
        sns.despine(left=True)

        _save(fig, f"01_s4_recall_v1_vs_v2_w{wsec}s.png")


# ── Plot 2: FPR v1 vs v2 (benign) ────────────────────────────────

def plot_fpr_v1_vs_v2(rows):
    """Bar chart: FPR for benign traffic, v1 vs v2."""
    benign = [r for r in rows if r["scenario"] == "A1" and r["r0"] == 0.10]
    if not benign:
        print("  SKIP plot_fpr_v1v2 (no data)")
        return

    labels = []
    fprs = []
    colors = []
    for r in sorted(benign, key=lambda r: (r.get("wf", ""), r.get("window", 0))):
        labels.append(f"{r['wf']}/w{r['window']}s")
        fprs.append(r["fpr"])
        colors.append(WF_COLORS.get(r["wf"], "#999"))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, fprs, color=colors, edgecolor="white", linewidth=1.5)
    ax.bar_label(bars, fmt="%.2f%%", fontsize=10, fontweight="bold")

    ax.axhline(y=5.0, color="#E53935", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(len(labels) - 0.5, 5.15, "Target ≤ 5%", ha="right",
            fontsize=10, color="#E53935", fontweight="bold")

    ax.set_xlabel("Config (features / window)")
    ax.set_ylabel("FPR (%)")
    ax.set_title("S4: FPR — Window Features v1 vs v2\n(Benign traffic, r₀=0.10)", fontsize=13)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    sns.despine(left=True)

    _save(fig, "02_s4_fpr_v1_vs_v2.png")


# ── Plot 3: F1 v1 vs v2 per attack ────────────────────────────────

def plot_f1_v1_vs_v2(rows):
    """Grouped bar: F1 per attack, v1 vs v2 at r0=0.10."""
    for wsec in [10, 30]:
        lookup = {}
        for r in rows:
            if r["scenario"] != "A2" or r["r0"] != 0.10 or r["window"] != wsec:
                continue
            lookup[(r["wf"], r["attack"])] = r

        attacks = [a for a in ATTACK_ORDER
                   if (("v1", a) in lookup or ("v2", a) in lookup)]
        if not attacks:
            continue

        x = np.arange(len(attacks))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        v1_vals = [lookup.get(("v1", a), {}).get("f1", 0) for a in attacks]
        v2_vals = [lookup.get(("v2", a), {}).get("f1", 0) for a in attacks]

        b1 = ax.bar(x - width / 2, v1_vals, width, label="v1 (12 basic)", color=WF_COLORS["v1"])
        b2 = ax.bar(x + width / 2, v2_vals, width, label="v2 (19 behavioral)", color=WF_COLORS["v2"])

        ax.bar_label(b1, fmt="%.1f%%", fontsize=9, fontweight="bold")
        ax.bar_label(b2, fmt="%.1f%%", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([_get_attack_label(a) for a in attacks])
        ax.set_ylabel("F1 Score (%)")
        ax.set_title(f"S4: F1 Score — Window Features v1 vs v2\n(window={wsec}s, r₀=0.10)", fontsize=13)
        ax.set_ylim(0, 105)
        ax.legend(loc="upper right")
        sns.despine(left=True)

        _save(fig, f"03_s4_f1_v1_vs_v2_w{wsec}s.png")


# ── Plot 4: FPR vs r0 for v2 ──────────────────────────────────────

def plot_fpr_vs_r0_v2(rows):
    """Line chart: FPR vs r0 for v2 (benign), w=10s and w=30s."""
    data = {}  # (window, r0) -> fpr
    for r in rows:
        if r["scenario"] != "A1" or r["wf"] != "v2":
            continue
        data[(r["window"], r["r0"])] = r["fpr"]

    if not data:
        print("  SKIP plot_fpr_vs_r0 (no data)")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for wsec in [10, 30]:
        points = sorted([(r0, fpr) for (w, r0), fpr in data.items() if w == wsec])
        if points:
            r0s, fprs = zip(*points)
            ax.plot(r0s, fprs, "o-", color=WINDOW_COLORS[wsec], linewidth=2.5,
                    markersize=8, markerfacecolor="white", markeredgewidth=2,
                    label=f"w={wsec}s")
            for r0, fpr in zip(r0s, fprs):
                ax.annotate(f"{fpr:.1f}%", (r0, fpr), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=9)

    ax.axhline(y=5.0, color="#E53935", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(0.02, 0.95, "Target ≤ 5%", transform=ax.transAxes,
            fontsize=10, color="#E53935", fontweight="bold", va="top")

    ax.set_xlabel("r₀ (initial variance)")
    ax.set_ylabel("FPR (%)")
    ax.set_title("S4: FPR vs r₀ — Window Features v2\n(Benign traffic)", fontsize=13)
    ax.legend(loc="upper right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    sns.despine()

    _save(fig, "04_s4_fpr_vs_r0_v2.png")


# ── Plot 5: Dashboard S3 → S4 ─────────────────────────────────────

def plot_dashboard_s3_vs_s4(s4_rows, s3_rows):
    """4-panel dashboard comparing S3 (C02) to S4 (C03)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── (a) Recall: S3 best vs S4-v2 best (at r0=0.10) ──
    ax = axes[0, 0]
    # S3 best per attack at r0=0.10
    s3_best = {}
    for r in s3_rows:
        if r["scenario"] != "A2" or r["r0"] != 0.10:
            continue
        if r["attack"] not in s3_best or r["recall"] > s3_best[r["attack"]]["recall"]:
            s3_best[r["attack"]] = r

    # S4-v2 best per attack at r0=0.10
    s4_best = {}
    for r in s4_rows:
        if r["scenario"] != "A2" or r["r0"] != 0.10 or r["wf"] != "v2":
            continue
        if r["attack"] not in s4_best or r["recall"] > s4_best[r["attack"]]["recall"]:
            s4_best[r["attack"]] = r

    attacks = [a for a in ATTACK_ORDER if a in s3_best or a in s4_best]
    if attacks:
        x = np.arange(len(attacks))
        width = 0.35
        s3_vals = [s3_best.get(a, {}).get("recall", 0) for a in attacks]
        s4_vals = [s4_best.get(a, {}).get("recall", 0) for a in attacks]
        ax.bar(x - width / 2, s3_vals, width, label="S3 (basic)", color="#E53935")
        ax.bar(x + width / 2, s4_vals, width, label="S4-v2 (behavioral)", color="#4CAF50")
        ax.set_xticks(x)
        ax.set_xticklabels([_get_attack_label(a) for a in attacks], fontsize=9)
        ax.axhline(y=80, color="#4CAF50", linestyle="--", linewidth=1, alpha=0.4)
        ax.legend(fontsize=9)
    ax.set_ylabel("Recall (%)")
    ax.set_title("(a) Recall: S3 vs S4-v2", fontweight="bold")
    ax.set_ylim(0, 105)

    # ── (b) FPR comparison ──
    ax = axes[0, 1]
    fpr_data = {}
    for r in s3_rows:
        if r["scenario"] == "A1" and r["r0"] == 0.10:
            fpr_data[f"S3/w{r['window']}s"] = r["fpr"]
    for r in s4_rows:
        if r["scenario"] == "A1" and r["r0"] == 0.10:
            fpr_data[f"S4-{r['wf']}/w{r['window']}s"] = r["fpr"]

    if fpr_data:
        labels = sorted(fpr_data.keys())
        fprs = [fpr_data[k] for k in labels]
        bars = ax.bar(labels, fprs, color="#1976D2", edgecolor="white")
        ax.bar_label(bars, fmt="%.2f%%", fontsize=8)
        ax.axhline(y=5.0, color="#E53935", linestyle="--", linewidth=1, alpha=0.6)
        ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("FPR (%)")
    ax.set_title("(b) FPR: S3 vs S4", fontweight="bold")

    # ── (c) F1: S3 best vs S4-v2 best ──
    ax = axes[1, 0]
    s3_f1 = {}
    for r in s3_rows:
        if r["scenario"] != "A2" or r["r0"] != 0.10:
            continue
        if r["attack"] not in s3_f1 or r["f1"] > s3_f1[r["attack"]]["f1"]:
            s3_f1[r["attack"]] = r
    s4_f1 = {}
    for r in s4_rows:
        if r["scenario"] != "A2" or r["r0"] != 0.10 or r["wf"] != "v2":
            continue
        if r["attack"] not in s4_f1 or r["f1"] > s4_f1[r["attack"]]["f1"]:
            s4_f1[r["attack"]] = r

    if attacks:
        x = np.arange(len(attacks))
        width = 0.35
        s3_vals = [s3_f1.get(a, {}).get("f1", 0) for a in attacks]
        s4_vals = [s4_f1.get(a, {}).get("f1", 0) for a in attacks]
        ax.bar(x - width / 2, s3_vals, width, label="S3 (basic)", color="#E53935")
        ax.bar(x + width / 2, s4_vals, width, label="S4-v2 (behavioral)", color="#4CAF50")
        ax.set_xticks(x)
        ax.set_xticklabels([_get_attack_label(a) for a in attacks], fontsize=9)
        ax.legend(fontsize=9)
    ax.set_ylabel("F1 Score (%)")
    ax.set_title("(c) F1: S3 vs S4-v2", fontweight="bold")
    ax.set_ylim(0, 105)

    # ── (d) Best config summary ──
    ax = axes[1, 1]
    ax.axis("off")

    table_data = []
    for atk in attacks:
        best = s4_best.get(atk)
        if not best:
            continue
        config = f"wf={best.get('wf')}/w{best.get('window')}s"
        table_data.append([
            _get_attack_label(atk),
            config,
            f"{best['recall']:.1f}%",
            f"{best['precision']:.1f}%",
            f"{best['f1']:.1f}%",
        ])

    if table_data:
        table = ax.table(
            cellText=table_data,
            colLabels=["Attack", "Best S4 Config", "Recall", "Precision", "F1"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.5)
        for j in range(5):
            table[0, j].set_facecolor("#1976D2")
            table[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("(d) Best S4 Config per Attack", fontweight="bold")

    for ax in axes.flat:
        sns.despine(ax=ax)

    fig.suptitle("Campaign-03 S4: Behavioral Features — S3 vs S4 Comparison",
                 fontsize=15, fontweight="bold", y=1.01)

    _save(fig, "05_s4_dashboard_s3_vs_s4.png")


# ── Main ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading Campaign-03 S4 experiments...")
    s4_rows = load_experiments(RESULTS_DIR)
    print(f"  {len(s4_rows)} experiments loaded\n")

    print("Loading Campaign-02 S3 for comparison...")
    s3_rows = load_campaign02_s3()
    print(f"  {len(s3_rows)} experiments loaded\n")

    if not s4_rows:
        print("ERROR: No Campaign-03 results found. Run campaign first.")
        exit(1)

    print("Generating plots:")
    plot_recall_v1_vs_v2(s4_rows)
    plot_fpr_v1_vs_v2(s4_rows)
    plot_f1_v1_vs_v2(s4_rows)
    plot_fpr_vs_r0_v2(s4_rows)
    plot_dashboard_s3_vs_s4(s4_rows, s3_rows)

    print(f"\nAll plots saved to: {PLOTS_DIR}/")
