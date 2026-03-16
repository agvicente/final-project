#!/usr/bin/env python3
"""Generate publication-quality plots for Campaign-02 results.

Campaign-02 tests 3 incremental improvements:
  S1: IP-based ground truth (was phase-based in campaign-01)
  S2: Feature expansion (v1 → v2 → v3)
  S3: Window aggregation (5s, 10s, 30s, 60s)
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

CAMPAIGN_01_DIR = RESULTS_DIR.parent / "campaign-01"

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
STEP_COLORS = {"S1": "#1976D2", "S2": "#FF9800", "S3": "#4CAF50"}
FEAT_COLORS = {"v1": "#1976D2", "v2": "#FF9800", "v3": "#4CAF50"}
WINDOW_COLORS = {5: "#1976D2", 10: "#FF9800", 30: "#4CAF50", 60: "#E53935"}


# ── Data Loading ───────────────────────────────────────────────────

def parse_dirname(dirname: str) -> dict:
    """Parse experiment directory name into components.

    Examples:
        S1-A1-benign-r0_0.10       → step=S1, scenario=A1, attack=None, features=v1, window=None, r0=0.10
        S2-A2-ddos-v2-r0_0.10      → step=S2, scenario=A2, attack=ddos, features=v2, window=None, r0=0.10
        S3-A2-mirai-w10s-r0_0.10   → step=S3, scenario=A2, attack=mirai, features=None, window=10, r0=0.10
    """
    info = {"name": dirname, "step": None, "scenario": None, "attack": None,
            "features": None, "window": None, "r0": None}

    # Step
    m = re.match(r"(S[123])-", dirname)
    if m:
        info["step"] = m.group(1)

    # Scenario + attack
    if "-A1-" in dirname:
        info["scenario"] = "A1"
        info["attack"] = None
    elif "-A2-" in dirname:
        info["scenario"] = "A2"
        for atk in ATTACK_ORDER:
            if f"-{atk}-" in dirname or dirname.endswith(f"-{atk}"):
                info["attack"] = atk
                break

    # Features version
    m = re.search(r"-(v[123])-", dirname)
    if m:
        info["features"] = m.group(1)
    elif info["step"] == "S1":
        info["features"] = "v1"  # S1 uses default v1
    elif info["step"] == "S3":
        info["features"] = "v2"  # S3 uses v2

    # Window size
    m = re.search(r"-w(\d+)s-", dirname)
    if m:
        info["window"] = int(m.group(1))

    # r0
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


def load_campaign01() -> list:
    """Load campaign-01 results for comparison."""
    if not CAMPAIGN_01_DIR.exists():
        print(f"  WARNING: campaign-01 dir not found: {CAMPAIGN_01_DIR}")
        return []

    rows = []
    for d in sorted(CAMPAIGN_01_DIR.iterdir()):
        if not d.is_dir():
            continue
        det_file = d / "detection_results.json"
        if not det_file.exists():
            continue

        with open(det_file) as f:
            data = json.load(f)
        pm = data.get("prequential_metrics", {})

        # Parse campaign-01 name (e.g., A2-ddos-r0_0.10)
        name = d.name
        scenario = "A1" if name.startswith("A1") else ("A3" if name.startswith("A3") else "A2")
        attack = None
        if scenario == "A2":
            for atk in ATTACK_ORDER:
                if atk in name:
                    attack = atk
                    break

        r0 = 0.10
        m = re.search(r"r0_([\d.]+)", name)
        if m:
            r0 = float(m.group(1))

        rows.append({
            "name": name, "step": "C1", "scenario": scenario, "attack": attack,
            "features": "v1", "window": None, "r0": r0,
            "precision": pm.get("precision", 0) * 100,
            "recall": pm.get("recall", 0) * 100,
            "f1": pm.get("f1", 0) * 100,
            "fpr": pm.get("fpr", 0) * 100,
            "tp": pm.get("tp", 0), "fp": pm.get("fp", 0),
            "tn": pm.get("tn", 0), "fn": pm.get("fn", 0),
            "flows": data.get("flows_processed", 0),
            "anomalies": data.get("anomalies_detected", 0),
            "anomaly_rate": data.get("anomaly_rate", 0),
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


# ── Plot 1: S1 — Recall comparison (IP vs phase ground truth) ─────

def plot_s1_recall_vs_campaign01(c02_rows, c01_rows):
    """Grouped bar: Recall per attack — campaign-01 (phase GT) vs S1 (IP GT)."""
    # Get S1 rows at r0=0.10 (or 0.15 for attacks that only have that)
    s1 = {}
    for r in c02_rows:
        if r["step"] != "S1" or r["scenario"] != "A2":
            continue
        if r["attack"] not in s1 or r["r0"] == 0.10:
            s1[r["attack"]] = r

    # Get campaign-01 A2 at r0=0.10
    c1 = {}
    for r in c01_rows:
        if r["scenario"] != "A2":
            continue
        if r["attack"] not in c1 or r["r0"] == 0.10:
            c1[r["attack"]] = r

    attacks = [a for a in ATTACK_ORDER if a in s1 and a in c1]
    if not attacks:
        print("  SKIP plot_s1_recall (no matching data)")
        return

    labels = [_get_attack_label(a) for a in attacks]
    c1_recalls = [c1[a]["recall"] for a in attacks]
    s1_recalls = [s1[a]["recall"] for a in attacks]

    x = np.arange(len(attacks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width / 2, c1_recalls, width, label="C-01 (phase GT)", color="#E53935")
    b2 = ax.bar(x + width / 2, s1_recalls, width, label="S1 (IP GT)", color="#1976D2")

    ax.axhline(y=80, color="#4CAF50", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(0.02, 82, "Target: 80%", transform=ax.get_yaxis_transform(),
            fontsize=10, color="#4CAF50", fontweight="bold")

    ax.bar_label(b1, fmt="%.1f%%", fontsize=9, fontweight="bold")
    ax.bar_label(b2, fmt="%.1f%%", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Recall (%)")
    ax.set_title("S1: Impact of IP-Based Ground Truth on Recall\n(Campaign-01 phase GT vs S1 IP GT, r₀ = 0.10)",
                 fontsize=13)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    sns.despine(left=True)

    _save(fig, "01_s1_recall_ip_vs_phase.png")


# ── Plot 2: S2 — Recall by attack × feature set ──────────────────

def plot_s2_recall_by_features(c02_rows):
    """Grouped bar: Recall per attack for v1/v2/v3 at r0=0.10."""
    # Collect: for each (attack, features), get r0=0.10 row
    data = {}  # (attack, features) → row
    for r in c02_rows:
        if r["scenario"] != "A2" or r["r0"] != 0.10:
            continue
        key = (r["attack"], r.get("features"))
        if r["step"] == "S1" and r.get("features") == "v1":
            data[key] = r
        elif r["step"] == "S2":
            data[key] = r

    attacks = [a for a in ATTACK_ORDER
               if any((a, fv) in data for fv in ["v1", "v2", "v3"])]
    if not attacks:
        print("  SKIP plot_s2_recall (no data)")
        return

    feat_versions = ["v1", "v2", "v3"]
    x = np.arange(len(attacks))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, fv in enumerate(feat_versions):
        vals = [data.get((a, fv), {}).get("recall", 0) for a in attacks]
        bars = ax.bar(x + (i - 1) * width, vals, width,
                      label=fv, color=FEAT_COLORS[fv])
        ax.bar_label(bars, fmt="%.1f%%", fontsize=8, fontweight="bold")

    ax.axhline(y=80, color="#4CAF50", linestyle="--", linewidth=1.5, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([_get_attack_label(a) for a in attacks])
    ax.set_ylabel("Recall (%)")
    ax.set_title("S2: Recall by Feature Set\n(r₀ = 0.10, IP ground truth)", fontsize=13)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    sns.despine(left=True)

    _save(fig, "02_s2_recall_by_features.png")


# ── Plot 3: S2 — FPR vs r0 by feature set ────────────────────────

def plot_s2_fpr_vs_r0(c02_rows):
    """Line chart: FPR × r0 for v1/v2/v3 using A1-benign data."""
    # Collect S1 (v1) and S2 (v2/v3) benign rows
    data = {}  # (features, r0) → fpr
    for r in c02_rows:
        if r["scenario"] != "A1":
            continue
        fv = r.get("features", "v1")
        if r["step"] == "S1":
            fv = "v1"
        elif r["step"] == "S2":
            pass  # already set
        else:
            continue
        data[(fv, r["r0"])] = r["fpr"]

    if not data:
        print("  SKIP plot_s2_fpr (no data)")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for fv in ["v1", "v2", "v3"]:
        points = sorted([(r0, fpr) for (f, r0), fpr in data.items() if f == fv])
        if points:
            r0s, fprs = zip(*points)
            ax.plot(r0s, fprs, "o-", color=FEAT_COLORS[fv], linewidth=2.5,
                    markersize=8, markerfacecolor="white", markeredgewidth=2,
                    label=fv)
            for r0, fpr in zip(r0s, fprs):
                ax.annotate(f"{fpr:.1f}%", (r0, fpr), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=9)

    ax.axhline(y=5.0, color="#E53935", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(0.02, 0.95, "Target ≤ 5%", transform=ax.transAxes,
            fontsize=10, color="#E53935", fontweight="bold", va="top")

    ax.set_xlabel("r₀ (initial variance)")
    ax.set_ylabel("FPR (%)")
    ax.set_title("S2: FPR vs r₀ by Feature Set\n(Benign traffic only)", fontsize=13)
    ax.legend(loc="upper right")
    ax.set_xlim(0.03, 0.22)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    sns.despine()

    _save(fig, "03_s2_fpr_vs_r0.png")


# ── Plot 4: S3 — Recall by window size ────────────────────────────

def plot_s3_recall_by_window(c02_rows):
    """Grouped bar: Recall per attack for each window size."""
    s3 = [r for r in c02_rows if r["step"] == "S3" and r["scenario"] == "A2"]
    if not s3:
        print("  SKIP plot_s3_recall (no data)")
        return

    attacks = [a for a in ATTACK_ORDER if any(r["attack"] == a for r in s3)]
    windows = sorted(set(r["window"] for r in s3 if r["window"] is not None))

    # Build lookup
    lookup = {}
    for r in s3:
        lookup[(r["attack"], r["window"])] = r

    x = np.arange(len(attacks))
    n_win = len(windows)
    width = 0.8 / max(n_win, 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, w in enumerate(windows):
        vals = [lookup.get((a, w), {}).get("recall", 0) for a in attacks]
        bars = ax.bar(x + (i - (n_win - 1) / 2) * width, vals, width,
                      label=f"{w}s", color=WINDOW_COLORS.get(w, "#999"))
        ax.bar_label(bars, fmt="%.1f%%", fontsize=8, fontweight="bold")

    ax.axhline(y=80, color="#4CAF50", linestyle="--", linewidth=1.5, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([_get_attack_label(a) for a in attacks])
    ax.set_ylabel("Recall (%)")
    ax.set_title("S3: Recall by Window Size\n(features v2, r₀ = 0.10)", fontsize=13)
    ax.set_ylim(0, 100)
    ax.legend(title="Window", loc="upper right")
    sns.despine(left=True)

    _save(fig, "04_s3_recall_by_window.png")


# ── Plot 5: S3 — FPR by window size ──────────────────────────────

def plot_s3_fpr_by_window(c02_rows):
    """Bar chart: FPR for A1-benign at each window size."""
    s3_benign = [r for r in c02_rows
                 if r["step"] == "S3" and r["scenario"] == "A1"]
    if not s3_benign:
        print("  SKIP plot_s3_fpr (no data)")
        return

    s3_benign.sort(key=lambda r: r.get("window", 0) or 0)
    labels = [f"{r['window']}s" for r in s3_benign]
    fprs = [r["fpr"] for r in s3_benign]
    colors = [WINDOW_COLORS.get(r["window"], "#999") for r in s3_benign]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, fprs, color=colors, edgecolor="white", linewidth=1.5)
    ax.bar_label(bars, fmt="%.2f%%", fontsize=10, fontweight="bold")

    ax.axhline(y=5.0, color="#E53935", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(len(labels) - 0.5, 5.15, "Target ≤ 5%", ha="right",
            fontsize=10, color="#E53935", fontweight="bold")

    ax.set_xlabel("Window Size (seconds)")
    ax.set_ylabel("FPR (%)")
    ax.set_title("S3: FPR by Window Size\n(Benign traffic, features v2, r₀ = 0.10)", fontsize=13)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    sns.despine(left=True)

    _save(fig, "05_s3_fpr_by_window.png")


# ── Plot 6: Recall evolution S1 → S2 → S3 ────────────────────────

def plot_recall_evolution(c02_rows):
    """Grouped bar: Recall per attack — best of S1, S2, S3."""
    # For each step, pick best config per attack at r0=0.10
    best = {}  # (step, attack) → row
    for r in c02_rows:
        if r["scenario"] != "A2" or r["r0"] != 0.10:
            continue
        key = (r["step"], r["attack"])
        if key not in best or r["recall"] > best[key]["recall"]:
            best[key] = r

    attacks = [a for a in ATTACK_ORDER
               if any((s, a) in best for s in ["S1", "S2", "S3"])]
    if not attacks:
        print("  SKIP plot_recall_evolution (no data)")
        return

    steps = ["S1", "S2", "S3"]
    x = np.arange(len(attacks))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, step in enumerate(steps):
        vals = [best.get((step, a), {}).get("recall", 0) for a in attacks]
        label = step
        if step == "S1":
            label = "S1 (IP GT)"
        elif step == "S2":
            label = "S2 (best features)"
        elif step == "S3":
            label = "S3 (best window)"
        bars = ax.bar(x + (i - 1) * width, vals, width,
                      label=label, color=STEP_COLORS[step])
        ax.bar_label(bars, fmt="%.1f%%", fontsize=8, fontweight="bold")

    ax.axhline(y=80, color="#4CAF50", linestyle="--", linewidth=1.5, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([_get_attack_label(a) for a in attacks])
    ax.set_ylabel("Recall (%)")
    ax.set_title("Campaign-02: Recall Evolution Across Steps\n(r₀ = 0.10)", fontsize=14)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    sns.despine(left=True)

    _save(fig, "06_recall_evolution.png")


# ── Plot 7: F1 evolution S1 → S2 → S3 ────────────────────────────

def plot_f1_evolution(c02_rows):
    """Grouped bar: F1 per attack — best of S1, S2, S3."""
    best = {}
    for r in c02_rows:
        if r["scenario"] != "A2" or r["r0"] != 0.10:
            continue
        key = (r["step"], r["attack"])
        if key not in best or r["f1"] > best[key]["f1"]:
            best[key] = r

    attacks = [a for a in ATTACK_ORDER
               if any((s, a) in best for s in ["S1", "S2", "S3"])]
    if not attacks:
        print("  SKIP plot_f1_evolution (no data)")
        return

    steps = ["S1", "S2", "S3"]
    x = np.arange(len(attacks))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, step in enumerate(steps):
        vals = [best.get((step, a), {}).get("f1", 0) for a in attacks]
        bars = ax.bar(x + (i - 1) * width, vals, width,
                      label=step, color=STEP_COLORS[step])
        ax.bar_label(bars, fmt="%.1f%%", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([_get_attack_label(a) for a in attacks])
    ax.set_ylabel("F1 Score (%)")
    ax.set_title("Campaign-02: F1 Score Evolution Across Steps\n(r₀ = 0.10)", fontsize=14)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    sns.despine(left=True)

    _save(fig, "07_f1_evolution.png")


# ── Plot 8: Dashboard ─────────────────────────────────────────────

def plot_dashboard(c02_rows, c01_rows):
    """4-panel summary dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── (a) Recall evolution ──
    ax = axes[0, 0]
    best = {}
    for r in c02_rows:
        if r["scenario"] != "A2" or r["r0"] != 0.10:
            continue
        key = (r["step"], r["attack"])
        if key not in best or r["recall"] > best[key]["recall"]:
            best[key] = r

    attacks = [a for a in ATTACK_ORDER
               if any((s, a) in best for s in ["S1", "S2", "S3"])]
    if attacks:
        steps = ["S1", "S2", "S3"]
        x = np.arange(len(attacks))
        width = 0.25
        for i, step in enumerate(steps):
            vals = [best.get((step, a), {}).get("recall", 0) for a in attacks]
            ax.bar(x + (i - 1) * width, vals, width,
                   label=step, color=STEP_COLORS[step])
        ax.set_xticks(x)
        ax.set_xticklabels([_get_attack_label(a) for a in attacks], fontsize=9)
        ax.axhline(y=80, color="#4CAF50", linestyle="--", linewidth=1, alpha=0.6)
        ax.legend(fontsize=9)
    ax.set_ylabel("Recall (%)")
    ax.set_title("(a) Recall Evolution", fontweight="bold")
    ax.set_ylim(0, 100)

    # ── (b) FPR across steps ──
    ax = axes[0, 1]
    # Collect FPR for A1-benign at r0=0.10 for each step/config
    fpr_data = {}
    for r in c02_rows:
        if r["scenario"] != "A1" or r["r0"] != 0.10:
            continue
        label = r["step"]
        if r["step"] == "S2":
            label = f"S2-{r.get('features', '?')}"
        elif r["step"] == "S3":
            label = f"S3-w{r.get('window', '?')}s"
        fpr_data[label] = r["fpr"]

    if fpr_data:
        labels = sorted(fpr_data.keys())
        fprs = [fpr_data[k] for k in labels]
        bars = ax.bar(labels, fprs, color="#1976D2", edgecolor="white")
        ax.bar_label(bars, fmt="%.2f%%", fontsize=8)
        ax.axhline(y=5.0, color="#E53935", linestyle="--", linewidth=1, alpha=0.6)
        ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("FPR (%)")
    ax.set_title("(b) FPR Across Configurations", fontweight="bold")

    # ── (c) F1 by attack ──
    ax = axes[1, 0]
    best_f1 = {}
    for r in c02_rows:
        if r["scenario"] != "A2" or r["r0"] != 0.10:
            continue
        key = (r["step"], r["attack"])
        if key not in best_f1 or r["f1"] > best_f1[key]["f1"]:
            best_f1[key] = r

    if attacks:
        steps = ["S1", "S2", "S3"]
        x = np.arange(len(attacks))
        width = 0.25
        for i, step in enumerate(steps):
            vals = [best_f1.get((step, a), {}).get("f1", 0) for a in attacks]
            ax.bar(x + (i - 1) * width, vals, width,
                   label=step, color=STEP_COLORS[step])
        ax.set_xticks(x)
        ax.set_xticklabels([_get_attack_label(a) for a in attacks], fontsize=9)
        ax.legend(fontsize=9)
    ax.set_ylabel("F1 Score (%)")
    ax.set_title("(c) F1 Score by Attack", fontweight="bold")
    ax.set_ylim(0, 100)

    # ── (d) Best config summary table ──
    ax = axes[1, 1]
    ax.axis("off")

    # Find overall best config per attack
    table_data = []
    for atk in attacks:
        best_row = None
        for r in c02_rows:
            if r["attack"] != atk or r["r0"] != 0.10:
                continue
            if best_row is None or r["f1"] > best_row["f1"]:
                best_row = r
        if best_row:
            config = best_row["step"]
            if best_row.get("features"):
                config += f"/{best_row['features']}"
            if best_row.get("window"):
                config += f"/w{best_row['window']}s"
            table_data.append([
                _get_attack_label(atk),
                config,
                f"{best_row['recall']:.1f}%",
                f"{best_row['precision']:.1f}%",
                f"{best_row['f1']:.1f}%",
            ])

    if table_data:
        table = ax.table(
            cellText=table_data,
            colLabels=["Attack", "Best Config", "Recall", "Precision", "F1"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.5)
        # Header styling
        for j in range(5):
            table[0, j].set_facecolor("#1976D2")
            table[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("(d) Best Configuration per Attack", fontweight="bold")

    for ax in axes.flat:
        sns.despine(ax=ax)

    fig.suptitle("Campaign-02 Results Summary — Incremental Improvements",
                 fontsize=15, fontweight="bold", y=1.01)

    _save(fig, "08_campaign_dashboard.png")


# ── Main ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading Campaign-02 experiments...")
    c02_rows = load_experiments(RESULTS_DIR)
    print(f"  {len(c02_rows)} experiments loaded\n")

    print("Loading Campaign-01 for comparison...")
    c01_rows = load_campaign01()
    print(f"  {len(c01_rows)} experiments loaded\n")

    if not c02_rows:
        print("ERROR: No Campaign-02 results found. Run campaign first.")
        exit(1)

    print("Generating plots:")
    plot_s1_recall_vs_campaign01(c02_rows, c01_rows)
    plot_s2_recall_by_features(c02_rows)
    plot_s2_fpr_vs_r0(c02_rows)
    plot_s3_recall_by_window(c02_rows)
    plot_s3_fpr_by_window(c02_rows)
    plot_recall_evolution(c02_rows)
    plot_f1_evolution(c02_rows)
    plot_dashboard(c02_rows, c01_rows)

    print(f"\nAll plots saved to: {PLOTS_DIR}/")
