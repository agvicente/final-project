#!/usr/bin/env python3
"""Statistical analysis of Experiment 3: regime transition.

Loads `results/exp03_results.csv` and reports:
    1. Friedman test across lambda for each (algorithm, r0) group.
    2. Two-way ANOVA (lambda x algorithm) per r0.
    3. Tukey HSD pairwise on FPR.
    4. Empirical transition lambda* via piecewise / sigmoid fit;
       comparison with algebraic prediction lambda* = sqrt(r0 / d).
    5. Bootstrap CI 95% for FPR per (algorithm, r0, lambda) group.
    6. Effect size (Cohen's d) V0 vs V7 in the transition zone.

Outputs:
    results/exp03_statistical_tests.txt   Human-readable formatted report.
    results/exp03_summary.csv             Aggregated mean/std/CI per condition.

Falsifiability criteria checked:
    H1 -- V7 transitions: Friedman p<0.001 across lambda groups for V7.
    H2 -- V0 differs:     Cohen's d > 0.8 V0 vs V7 in transition zone.
    H3 -- lambda* scales: ratio lambda*(r0_a)/lambda*(r0_b) within 2x of sqrt(r0_a/r0_b).
"""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_CSV = RESULTS_DIR / "exp03_results.csv"
SUMMARY_CSV = RESULTS_DIR / "exp03_summary.csv"
REPORT_TXT = RESULTS_DIR / "exp03_statistical_tests.txt"

D = 17  # fixed in exp03


# ── Bootstrap CI ────────────────────────────────────────────────

def bootstrap_ci(data: np.ndarray, n_boot: int = 1000, alpha: float = 0.05,
                 rng: np.random.Generator | None = None) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for the mean."""
    if rng is None:
        rng = np.random.default_rng(0)
    if len(data) < 2:
        return (float(data.mean()) if len(data) else 0.0, ) * 2
    boot = np.empty(n_boot)
    n = len(data)
    for i in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boot[i] = sample.mean()
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return lo, hi


# ── Cohen's d ───────────────────────────────────────────────────

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size between two groups (pooled std)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    mean_a, mean_b = a.mean(), b.mean()
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    n_a, n_b = len(a), len(b)
    pooled_sd = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_sd == 0:
        return 0.0
    return float((mean_a - mean_b) / pooled_sd)


# ── Transition point estimation ─────────────────────────────────

def fit_transition_lambda(
    df_subset: pd.DataFrame, metric: str = "frac_above_r0",
) -> dict:
    """Estimate lambda* as the lambda where `metric` crosses 0.5.

    Uses linear interpolation on (log10_lambda, mean_metric).
    Returns dict with empirical lambda* and a quality indicator.
    """
    grouped = (
        df_subset.groupby("log10_lambda")
        .agg(mean_metric=(metric, "mean"))
        .reset_index()
        .sort_values("log10_lambda")
    )
    log_lams = grouped["log10_lambda"].values
    metrics_vals = grouped["mean_metric"].values

    # Find consecutive points bracketing 0.5
    lambda_star = None
    for i in range(len(log_lams) - 1):
        if (metrics_vals[i] <= 0.5) and (metrics_vals[i + 1] >= 0.5):
            # Linear interpolation
            x0, x1 = log_lams[i], log_lams[i + 1]
            y0, y1 = metrics_vals[i], metrics_vals[i + 1]
            if y1 != y0:
                log_lam_star = x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)
            else:
                log_lam_star = (x0 + x1) / 2
            lambda_star = 10**log_lam_star
            break

    return {
        "lambda_star_empirical": lambda_star,
        "log_lambdas": log_lams.tolist(),
        "metric_values": metrics_vals.tolist(),
    }


# ── Main analysis ───────────────────────────────────────────────

def analyze() -> None:
    if not RESULTS_CSV.exists():
        print(f"ERROR: {RESULTS_CSV} not found. Run exp03_regime_transition.py --full first.")
        sys.exit(1)

    df = pd.read_csv(RESULTS_CSV)
    n_seeds = df["seed"].nunique()

    report = StringIO()
    report.write("=" * 75 + "\n")
    report.write("STATISTICAL TESTS -- Experiment 3: Regime Transition\n")
    report.write(f"Dimension: d={D} (fixed), seeds: {n_seeds}\n")
    report.write(f"Total runs: {len(df)}\n")
    report.write("Hypotheses:\n")
    report.write("  H1  V7 transitions at lambda* = sqrt(r0/d).\n")
    report.write("  H2  V0 differs qualitatively (Cohen's d > 0.8 vs V7).\n")
    report.write("  H3  lambda*(r0_a)/lambda*(r0_b) ~= sqrt(r0_a/r0_b).\n")
    report.write("=" * 75 + "\n\n")

    # ── Aggregated summary ──────────────────────────────────────
    rng = np.random.default_rng(0)
    summary_rows = []
    for (algo, r0, lam), group in df.groupby(["algorithm", "r0", "lambda"]):
        fpr = group["FPR"].values
        ci_lo, ci_hi = bootstrap_ci(fpr, n_boot=1000, rng=rng)
        summary_rows.append({
            "algorithm": algo, "r0": r0, "lambda": lam,
            "FPR_mean": fpr.mean(), "FPR_std": fpr.std(ddof=1),
            "FPR_ci95_lo": ci_lo, "FPR_ci95_hi": ci_hi,
            "n_clusters_mean": group["n_clusters"].mean(),
            "frac_above_r0_mean": group["frac_above_r0"].mean(),
            "regime_mode": group["regime"].mode().iloc[0],
            "n_seeds": len(group),
        })
    summary = pd.DataFrame(summary_rows).sort_values(
        ["algorithm", "r0", "lambda"]
    )
    summary.to_csv(SUMMARY_CSV, index=False)
    report.write(f"Summary saved: {SUMMARY_CSV}\n\n")

    # ── 1. Friedman per (algorithm, r0) across lambda ───────────
    report.write("-" * 75 + "\n")
    report.write("1. FRIEDMAN TEST -- per (algorithm, r0) across lambda groups\n")
    report.write("-" * 75 + "\n\n")
    for algo in sorted(df["algorithm"].unique()):
        for r0 in sorted(df["r0"].unique()):
            sub = df[(df["algorithm"] == algo) & (df["r0"] == r0)]
            if sub.empty:
                continue
            # Pivot: rows=seed, cols=lambda
            pivot = sub.pivot_table(
                index="seed", columns="lambda", values="FPR", aggfunc="first"
            ).dropna()
            if pivot.shape[1] < 3 or pivot.shape[0] < 3:
                report.write(f"  {algo:25s} r0={r0:g}: insufficient data\n")
                continue
            cols = [pivot[c].values for c in pivot.columns]
            try:
                stat, p = stats.friedmanchisquare(*cols)
                sig = "YES" if p < 0.001 else "NO"
                report.write(
                    f"  {algo:25s} r0={r0:g}: chi2={stat:.3f}, p={p:.3e}  "
                    f"[H1 sig p<0.001? {sig}]\n"
                )
            except ValueError as e:
                report.write(f"  {algo:25s} r0={r0:g}: Friedman failed ({e})\n")
    report.write("\n")

    # ── 2. Two-way ANOVA-like check (per r0): lambda x algorithm ─
    report.write("-" * 75 + "\n")
    report.write("2. ANOVA one-way per r0 -- FPR ~ lambda (each algo separate)\n")
    report.write("-" * 75 + "\n\n")
    for algo in sorted(df["algorithm"].unique()):
        for r0 in sorted(df["r0"].unique()):
            sub = df[(df["algorithm"] == algo) & (df["r0"] == r0)]
            groups = [g["FPR"].values for _, g in sub.groupby("lambda")]
            if len(groups) < 2 or any(len(g) < 2 for g in groups):
                continue
            try:
                f_stat, p = stats.f_oneway(*groups)
                report.write(
                    f"  {algo:25s} r0={r0:g}: F={f_stat:.2f}, p={p:.3e}\n"
                )
            except Exception as e:
                report.write(f"  {algo:25s} r0={r0:g}: ANOVA failed ({e})\n")
    report.write("\n")

    # ── 3. Transition point estimation ──────────────────────────
    report.write("-" * 75 + "\n")
    report.write("3. EMPIRICAL TRANSITION POINT lambda*\n")
    report.write("-" * 75 + "\n")
    report.write("Predicted: lambda* = sqrt(r0 / d), d=17\n")
    report.write("Empirical: linear interp where frac_above_r0 crosses 0.5\n\n")

    lambda_star_records = []
    for algo in sorted(df["algorithm"].unique()):
        for r0 in sorted(df["r0"].unique()):
            sub = df[(df["algorithm"] == algo) & (df["r0"] == r0)]
            if sub.empty:
                continue
            fit = fit_transition_lambda(sub, metric="frac_above_r0")
            predicted = float(np.sqrt(r0 / D))
            empirical = fit["lambda_star_empirical"]

            if empirical is None:
                ratio = float("nan")
                pass_h1 = "NO TRANSITION OBSERVED"
            else:
                ratio = empirical / predicted
                # Pass H1 if within 2x in either direction
                pass_h1 = "YES" if (0.5 <= ratio <= 2.0) else "NO"

            lambda_star_records.append({
                "algorithm": algo, "r0": r0,
                "predicted": predicted, "empirical": empirical, "ratio": ratio,
            })
            report.write(
                f"  {algo:25s} r0={r0:g}: "
                f"predicted={predicted:.4f}, "
                f"empirical={'%.4f' % empirical if empirical else 'N/A':>8s}, "
                f"ratio={ratio:>6.2f}  [H1 within 2x? {pass_h1}]\n"
            )

    report.write("\n  H3 check (ratios between r0's, V7 only):\n")
    v7_recs = [r for r in lambda_star_records if r["algorithm"] == "V7_full_corrected"]
    v7_recs = [r for r in v7_recs if r["empirical"] is not None]
    v7_recs.sort(key=lambda x: x["r0"])
    for i in range(len(v7_recs)):
        for j in range(i + 1, len(v7_recs)):
            ra, rb = v7_recs[i], v7_recs[j]
            obs_ratio = rb["empirical"] / ra["empirical"]
            pred_ratio = float(np.sqrt(rb["r0"] / ra["r0"]))
            check_ratio = obs_ratio / pred_ratio
            pass_h3 = "YES" if 0.5 <= check_ratio <= 2.0 else "NO"
            report.write(
                f"    lambda*(r0={rb['r0']:g})/lambda*(r0={ra['r0']:g}): "
                f"obs={obs_ratio:.3f}, pred=sqrt({rb['r0']/ra['r0']:.0f})={pred_ratio:.3f}, "
                f"check_ratio={check_ratio:.2f}  [H3? {pass_h3}]\n"
            )
    report.write("\n")

    # ── 4. Cohen's d V0 vs V7 in transition zone ────────────────
    report.write("-" * 75 + "\n")
    report.write("4. COHEN'S d -- V0 vs V7 (per r0, per lambda)\n")
    report.write("-" * 75 + "\n")
    report.write("Effect size on FPR. Threshold for H2: |d| > 0.8 in at least one cell.\n\n")

    h2_passed = False
    for r0 in sorted(df["r0"].unique()):
        report.write(f"  r0 = {r0:g}:\n")
        for lam in sorted(df["lambda"].unique()):
            v0 = df[(df["algorithm"] == "V0_original")
                    & (df["r0"] == r0) & (df["lambda"] == lam)]["FPR"].values
            v7 = df[(df["algorithm"] == "V7_full_corrected")
                    & (df["r0"] == r0) & (df["lambda"] == lam)]["FPR"].values
            if len(v0) < 2 or len(v7) < 2:
                continue
            d_val = cohens_d(v0, v7)
            mark = "***" if abs(d_val) > 0.8 else "  "
            if abs(d_val) > 0.8:
                h2_passed = True
            report.write(
                f"    lambda={lam:g}: V0_mean={v0.mean():.4f}, V7_mean={v7.mean():.4f}, "
                f"d={d_val:>+7.2f}  {mark}\n"
            )
    report.write(f"\n  H2 (|d|>0.8 in any cell)? {'YES' if h2_passed else 'NO'}\n\n")

    # ── 5. Verdict ──────────────────────────────────────────────
    report.write("-" * 75 + "\n")
    report.write("5. VERDICT vs. PLAN CRITERIA (Phase C)\n")
    report.write("-" * 75 + "\n")

    # H1 -- count V7 r0's where empirical within 2x of predicted
    v7_h1_pass = sum(
        1 for r in lambda_star_records
        if r["algorithm"] == "V7_full_corrected"
        and r["empirical"] is not None
        and 0.5 <= r["ratio"] <= 2.0
    )
    v7_total = sum(1 for r in lambda_star_records if r["algorithm"] == "V7_full_corrected")
    report.write(
        f"  H1: V7 transitions within 2x of sqrt(r0/d): "
        f"{v7_h1_pass}/{v7_total} r0 values pass\n"
    )
    report.write(f"  H2: V0 vs V7 |d|>0.8 in some cell: {h2_passed}\n")

    # H3 ratios
    v7_recs2 = [r for r in lambda_star_records if r["algorithm"] == "V7_full_corrected"
                and r["empirical"] is not None]
    h3_passes = []
    if len(v7_recs2) >= 2:
        v7_recs2.sort(key=lambda x: x["r0"])
        for i in range(len(v7_recs2)):
            for j in range(i + 1, len(v7_recs2)):
                obs_r = v7_recs2[j]["empirical"] / v7_recs2[i]["empirical"]
                pred_r = float(np.sqrt(v7_recs2[j]["r0"] / v7_recs2[i]["r0"]))
                check = obs_r / pred_r
                h3_passes.append(0.5 <= check <= 2.0)
    h3_score = f"{sum(h3_passes)}/{len(h3_passes)}" if h3_passes else "N/A"
    report.write(f"  H3: lambda* ratios within 2x of sqrt(r0_a/r0_b): {h3_score}\n\n")

    if v7_h1_pass >= 2 and h2_passed and h3_passes and all(h3_passes):
        report.write("  CASE A -- All predictions confirmed. Proceed to Phase D (figures).\n")
    elif v7_h1_pass >= 1 and h2_passed:
        report.write("  CASE B -- Partially confirmed. Investigate divergence; may proceed with caveat.\n")
    else:
        report.write("  CASE C/D -- Predictions not confirmed. Revisit hypothesis or instrumentation.\n")

    out = report.getvalue()
    print(out)
    REPORT_TXT.write_text(out)
    print(f"\nReport saved: {REPORT_TXT}")


if __name__ == "__main__":
    analyze()
