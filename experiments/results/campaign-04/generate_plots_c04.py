#!/usr/bin/env python3
"""
Campaign-04 — Graficos comparativos: Original vs Proprio MicroTEDAclus.
Uso: python experiments/results/campaign-04/generate_plots_c04.py
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Color scheme ──────────────────────────────────────────
C_OURS = "#2E75B6"       # Blue - our implementation
C_ORIGINAL = "#E74C3C"   # Red - original implementation
C_OURS_V2 = "#27AE60"    # Green - our v2
C_ORIG_V2 = "#F39C12"    # Orange - original v2

ATTACKS = ["DDoS-ICMP", "DDoS-SYN", "DDoS-TCP", "Mirai", "Recon"]

# ── Data ──────────────────────────────────────────────────

# Flow-level (r0=0.10) - C02-S1 vs C04-B1
flow_ours_recall = [27.2, 3.5, 0.0, 1.7, 4.5]
flow_orig_recall = [69.4, 37.5, 92.3, 26.8, 46.2]
flow_ours_f1 =     [21.4, 6.3, 0.0, 3.0, 8.4]
flow_orig_f1 =     [6.2, 31.6, 1.3, 17.4, 46.8]
flow_ours_fpr =    [3.6, 4.2, 3.2, 3.5, 3.1]
flow_orig_fpr =    [54.7, 53.3, 54.5, 55.1, 55.3]
flow_ours_fpr_benign = 3.9
flow_orig_fpr_benign = 54.4

# Window 10s v1 (r0=0.10) - C03-S4 vs C04-B2
w10_v1_ours_recall = [0.0, 38.5, 0.0, 46.2, 39.2]
w10_v1_orig_recall = [100, 76.9, 0.0, 84.6, 88.9]
w10_v1_ours_f1 =     [0.0, 20.0, 0.0, 23.1, 35.7]
w10_v1_orig_f1 =     [4.0, 15.5, 0.0, 18.2, 36.2]
w10_v1_ours_fpr =    [13.6, 14.8, 11.3, 15.5, 13.1]
w10_v1_orig_fpr =    [44.7, 47.7, 45.0, 45.3, 52.2]

# Window 10s v2 (r0=0.10) - C03-S4 vs C04-B3
w10_v2_ours_recall = [50.0, 30.8, 0.0, 38.5, 45.5]
w10_v2_orig_recall = [100, 69.2, 0.0, 76.9, 82.7]
w10_v2_ours_f1 =     [5.6, 17.8, 0.0, 21.7, 39.1]
w10_v2_orig_f1 =     [5.7, 14.2, 0.0, 16.7, 33.3]
w10_v2_ours_fpr =    [15.7, 12.5, 2.4, 13.2, 15.4]
w10_v2_orig_fpr =    [46.0, 47.1, 43.3, 45.1, 52.6]

# Benign FPR across granularities
benign_configs = ["Flow", "Win v1\nw=10s", "Win v1\nw=30s", "Win v2\nw=10s", "Win v2\nw=30s"]
benign_ours =    [3.9, 2.9, 5.0, 14.3, 6.7]
benign_orig =    [54.4, 41.9, 74.5, 45.5, 73.6]


def setup_style():
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
    })

setup_style()


# ═══════════════════════════════════════════════════════════
# PLOT 1: FPR Benigno — Todas as Granularidades
# ═══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(benign_configs))
w = 0.35

bars1 = ax.bar(x - w/2, benign_ours, w, label="Proprio (micro_teda)", color=C_OURS, edgecolor='white')
bars2 = ax.bar(x + w/2, benign_orig, w, label="Original (Maia 2020)", color=C_ORIGINAL, edgecolor='white')

ax.set_ylabel("FPR Benigno (%)")
ax.set_title("FPR em Trafego Benigno — Proprio vs Original")
ax.set_xticks(x)
ax.set_xticklabels(benign_configs)
ax.axhline(y=5, color='green', linestyle='--', alpha=0.7, label="Alvo <= 5%")
ax.legend()
ax.set_ylim(0, 85)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9, color=C_OURS)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9, color=C_ORIGINAL)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_fpr_benign_comparison.png", dpi=150)
plt.close()
print("Saved: 01_fpr_benign_comparison.png")


# ═══════════════════════════════════════════════════════════
# PLOT 2: Flow-Level — Recall Comparison
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Recall
ax = axes[0]
x = np.arange(len(ATTACKS))
w = 0.35
ax.bar(x - w/2, flow_ours_recall, w, label="Proprio", color=C_OURS)
ax.bar(x + w/2, flow_orig_recall, w, label="Original", color=C_ORIGINAL)
ax.set_ylabel("Recall (%)")
ax.set_title("Flow-Level: Recall")
ax.set_xticks(x)
ax.set_xticklabels(ATTACKS, rotation=15)
ax.legend()
ax.set_ylim(0, 105)

# FPR
ax = axes[1]
ax.bar(x - w/2, flow_ours_fpr, w, label="Proprio", color=C_OURS)
ax.bar(x + w/2, flow_orig_fpr, w, label="Original", color=C_ORIGINAL)
ax.axhline(y=flow_ours_fpr_benign, color=C_OURS, linestyle='--', alpha=0.5, label=f"FPR benigno proprio ({flow_ours_fpr_benign}%)")
ax.axhline(y=flow_orig_fpr_benign, color=C_ORIGINAL, linestyle='--', alpha=0.5, label=f"FPR benigno original ({flow_orig_fpr_benign}%)")
ax.set_ylabel("FPR (%)")
ax.set_title("Flow-Level: FPR")
ax.set_xticks(x)
ax.set_xticklabels(ATTACKS, rotation=15)
ax.legend(fontsize=9)
ax.set_ylim(0, 65)

plt.suptitle("Campaign-04 vs C02-S1 — Flow-Level (r0=0.10)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_flow_recall_fpr_comparison.png", dpi=150)
plt.close()
print("Saved: 02_flow_recall_fpr_comparison.png")


# ═══════════════════════════════════════════════════════════
# PLOT 3: Window 10s — Recall & FPR (v1 features)
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

x = np.arange(len(ATTACKS))
w = 0.35

ax = axes[0]
ax.bar(x - w/2, w10_v1_ours_recall, w, label="Proprio v1", color=C_OURS)
ax.bar(x + w/2, w10_v1_orig_recall, w, label="Original v1", color=C_ORIGINAL)
ax.set_ylabel("Recall (%)")
ax.set_title("Window 10s v1: Recall")
ax.set_xticks(x)
ax.set_xticklabels(ATTACKS, rotation=15)
ax.legend()
ax.set_ylim(0, 115)

ax = axes[1]
ax.bar(x - w/2, w10_v1_ours_fpr, w, label="Proprio v1", color=C_OURS)
ax.bar(x + w/2, w10_v1_orig_fpr, w, label="Original v1", color=C_ORIGINAL)
ax.axhline(y=5, color='green', linestyle='--', alpha=0.7, label="Alvo FPR <= 5%")
ax.set_ylabel("FPR (%)")
ax.set_title("Window 10s v1: FPR")
ax.set_xticks(x)
ax.set_xticklabels(ATTACKS, rotation=15)
ax.legend()
ax.set_ylim(0, 65)

plt.suptitle("Campaign-04 vs C03-S4 — Window 10s, Features v1 (r0=0.10)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_window10s_v1_comparison.png", dpi=150)
plt.close()
print("Saved: 03_window10s_v1_comparison.png")


# ═══════════════════════════════════════════════════════════
# PLOT 4: Window 10s — v2 features comparison
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.bar(x - w/2, w10_v2_ours_recall, w, label="Proprio v2", color=C_OURS_V2)
ax.bar(x + w/2, w10_v2_orig_recall, w, label="Original v2", color=C_ORIG_V2)
ax.set_ylabel("Recall (%)")
ax.set_title("Window 10s v2: Recall")
ax.set_xticks(x)
ax.set_xticklabels(ATTACKS, rotation=15)
ax.legend()
ax.set_ylim(0, 115)

ax = axes[1]
ax.bar(x - w/2, w10_v2_ours_fpr, w, label="Proprio v2", color=C_OURS_V2)
ax.bar(x + w/2, w10_v2_orig_fpr, w, label="Original v2", color=C_ORIG_V2)
ax.axhline(y=5, color='green', linestyle='--', alpha=0.7, label="Alvo FPR <= 5%")
ax.set_ylabel("FPR (%)")
ax.set_title("Window 10s v2: FPR")
ax.set_xticks(x)
ax.set_xticklabels(ATTACKS, rotation=15)
ax.legend()
ax.set_ylim(0, 65)

plt.suptitle("Campaign-04 vs C03-S4 — Window 10s, Features v2 (r0=0.10)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_window10s_v2_comparison.png", dpi=150)
plt.close()
print("Saved: 04_window10s_v2_comparison.png")


# ═══════════════════════════════════════════════════════════
# PLOT 5: F1 Comparison — All Granularities (Recall vs FPR trade-off)
# ═══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 8))

# Our implementation (various configs)
configs_ours = [
    ("Flow", flow_ours_recall, flow_ours_fpr, 'o', C_OURS),
    ("Win v1 w10s", w10_v1_ours_recall, w10_v1_ours_fpr, 's', C_OURS),
    ("Win v2 w10s", w10_v2_ours_recall, w10_v2_ours_fpr, '^', C_OURS_V2),
]
configs_orig = [
    ("Flow", flow_orig_recall, flow_orig_fpr, 'o', C_ORIGINAL),
    ("Win v1 w10s", w10_v1_orig_recall, w10_v1_orig_fpr, 's', C_ORIGINAL),
    ("Win v2 w10s", w10_v2_orig_recall, w10_v2_orig_fpr, '^', C_ORIG_V2),
]

for label, recalls, fprs, marker, color in configs_ours:
    for i, atk in enumerate(ATTACKS):
        if recalls[i] > 0:
            ax.scatter(fprs[i], recalls[i], marker=marker, color=color, s=100, zorder=5,
                      edgecolors='black', linewidths=0.5)
            ax.annotate(atk, (fprs[i]+0.5, recalls[i]+1), fontsize=7, color=color)

for label, recalls, fprs, marker, color in configs_orig:
    for i, atk in enumerate(ATTACKS):
        if recalls[i] > 0:
            ax.scatter(fprs[i], recalls[i], marker=marker, color=color, s=100, zorder=5,
                      edgecolors='black', linewidths=0.5)

# Legend manually
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=C_OURS, markersize=10, label='Proprio - Flow'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=C_OURS, markersize=10, label='Proprio - Win v1'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor=C_OURS_V2, markersize=10, label='Proprio - Win v2'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=C_ORIGINAL, markersize=10, label='Original - Flow'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=C_ORIGINAL, markersize=10, label='Original - Win v1'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor=C_ORIG_V2, markersize=10, label='Original - Win v2'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

# Acceptable region
ax.axvspan(0, 15, alpha=0.08, color='green', label='FPR aceitavel (<15%)')
ax.axvline(x=15, color='green', linestyle='--', alpha=0.4)
ax.text(7.5, 2, "FPR aceitavel", ha='center', fontsize=9, color='green', alpha=0.7)

ax.set_xlabel("FPR (%)")
ax.set_ylabel("Recall (%)")
ax.set_title("Recall vs FPR — Proprio vs Original (todos os ataques e configs)", fontsize=13, fontweight='bold')
ax.set_xlim(0, 60)
ax.set_ylim(0, 105)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_recall_vs_fpr_scatter.png", dpi=150)
plt.close()
print("Saved: 05_recall_vs_fpr_scatter.png")


# ═══════════════════════════════════════════════════════════
# PLOT 6: Dashboard — Summary
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (0,0) FPR Benigno
ax = axes[0, 0]
x = np.arange(len(benign_configs))
ax.bar(x - 0.175, benign_ours, 0.35, label="Proprio", color=C_OURS)
ax.bar(x + 0.175, benign_orig, 0.35, label="Original", color=C_ORIGINAL)
ax.axhline(y=5, color='green', linestyle='--', alpha=0.7)
ax.set_ylabel("FPR (%)")
ax.set_title("FPR Benigno")
ax.set_xticks(x)
ax.set_xticklabels(benign_configs, fontsize=9)
ax.legend(fontsize=9)
ax.set_ylim(0, 85)

# (0,1) Recall Flow-level
ax = axes[0, 1]
x = np.arange(len(ATTACKS))
ax.bar(x - 0.175, flow_ours_recall, 0.35, label="Proprio", color=C_OURS)
ax.bar(x + 0.175, flow_orig_recall, 0.35, label="Original", color=C_ORIGINAL)
ax.set_ylabel("Recall (%)")
ax.set_title("Recall Flow-Level")
ax.set_xticks(x)
ax.set_xticklabels(ATTACKS, rotation=15, fontsize=9)
ax.legend(fontsize=9)

# (1,0) Recall Window 10s v1
ax = axes[1, 0]
ax.bar(x - 0.175, w10_v1_ours_recall, 0.35, label="Proprio v1", color=C_OURS)
ax.bar(x + 0.175, w10_v1_orig_recall, 0.35, label="Original v1", color=C_ORIGINAL)
ax.set_ylabel("Recall (%)")
ax.set_title("Recall Window 10s (Features v1)")
ax.set_xticks(x)
ax.set_xticklabels(ATTACKS, rotation=15, fontsize=9)
ax.legend(fontsize=9)

# (1,1) F1 best per attack
ax = axes[1, 1]
# Best F1 with FPR < 15%
best_f1_ours = [21.4, 20.0, 0.0, 23.1, 43.7]  # C02-S1/C03-S4 best
best_f1_orig = [6.2, 31.6, 1.3, 17.4, 46.8]   # C04 best (all FPR > 50%)
ax.bar(x - 0.175, best_f1_ours, 0.35, label="Proprio (FPR<15%)", color=C_OURS)
ax.bar(x + 0.175, best_f1_orig, 0.35, label="Original (FPR~55%)", color=C_ORIGINAL, alpha=0.5, hatch='//')
ax.set_ylabel("F1 (%)")
ax.set_title("Melhor F1 por Ataque")
ax.set_xticks(x)
ax.set_xticklabels(ATTACKS, rotation=15, fontsize=9)
ax.legend(fontsize=9)

plt.suptitle("Campaign-04 Dashboard — Original vs Proprio MicroTEDAclus",
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_campaign04_dashboard.png", dpi=150)
plt.close()
print("Saved: 06_campaign04_dashboard.png")

print(f"\nAll plots saved to: {OUTPUT_DIR}")
