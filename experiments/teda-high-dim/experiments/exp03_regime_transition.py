#!/usr/bin/env python3
"""Experiment 3: Regime Transition Characterization.

Tests whether MicroTEDAclus undergoes a phase transition between
two operational regimes (r0-bounded and data-bounded) governed by
the comparison sigma^2 vs r0 in the eccentricity denominator.

Hypotheses:
    H1 -- V7 shows transition at lambda* such that sigma^2_eff(lambda*) ~ r0.
    H2 -- V0 lacks the explicit max(var, r0) guard and behaves
          qualitatively differently across lambda.
    H3 -- For multiple r0 values, lambda*_observed scales as
          predicted by sigma^2 vs r0 algebraic relation.

Protocol:
    Dimension: d = 17 (matches IoT v1 features).
    Data: N(0, lambda^2 * I_d) with anomalies at anomaly_scale=5 sigma.
    Algorithms: V0 (original), V7 (full corrected).
    Sweep:
        lambda in {10^-3, 10^-2.5, ..., 10^3} (13 log-spaced values).
        r0 in {1e-3, 1e-1, 1.0} (3 values, ~3 orders of magnitude).
    Repetitions: 30 seeds per condition.
    Total runs: 13 * 3 * 2 * 30 = 2340.

Modes:
    --smoke: 4 lambda values, 1 r0, V7 only, 3 seeds. Pipeline check.
    --full:  full sweep above.

Outputs:
    results/exp03_results.csv      Long format (one row per run).
    Console summary on completion.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project src on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from teda_hd.algorithms.variants import create_variant
from teda_hd.generators.gaussian import GaussianStreamGenerator
from teda_hd.metrics.detection import compute_metrics
from teda_hd.metrics.regime import compute_full_regime_metrics


# ── Configuration ───────────────────────────────────────────────

D = 17
ANOMALY_SCALE = 5.0
N_NORMAL = 950
N_ANOMALIES = 50

# Sweep grids
# Range chosen to cover 4 orders of magnitude centered on predicted lambda*'s
# at d=17 for r0 in {1e-3, 1e-1, 1}: lambda* in {0.008, 0.077, 0.243}.
# Upper bound capped at 10 to avoid V0 cluster-explosion (O(N^2) per stream).
FULL_LAMBDAS = np.logspace(-3, 1, 9)             # 10^-3 .. 10^1 step 10^0.5
FULL_R0S = np.array([1e-3, 1e-1, 1.0])
FULL_ALGOS = ["V0_original", "V7_full_corrected"]
FULL_SEEDS = 30

SMOKE_LAMBDAS = np.array([1e-2, 1e-1, 1.0, 10.0])
SMOKE_R0S = np.array([1e-1])
SMOKE_ALGOS = ["V7_full_corrected"]
SMOKE_SEEDS = 3

RESULTS_DIR = PROJECT_ROOT / "results"


# ── Single run ──────────────────────────────────────────────────

def run_single(
    algo_name: str,
    lam: float,
    r0: float,
    seed: int,
) -> dict:
    """Run a single experiment condition and return metrics row."""
    gen = GaussianStreamGenerator(
        d=D,
        n_normal=N_NORMAL,
        n_anomalies=N_ANOMALIES,
        anomaly_scale=ANOMALY_SCALE,
        normal_scale=lam,
        seed=seed,
    )
    X, y_true = gen.generate()

    algo = create_variant(algo_name, r0=r0)

    y_pred = np.zeros(len(X), dtype=int)
    for i, x in enumerate(X):
        result = algo.process(x)
        y_pred[i] = int(result.is_anomaly)

    detection = compute_metrics(y_true, y_pred)
    regime = compute_full_regime_metrics(algo, r0=r0)

    return {
        "algorithm": algo_name,
        "lambda": float(lam),
        "log10_lambda": float(np.log10(lam)),
        "r0": float(r0),
        "log10_r0": float(np.log10(r0)),
        "seed": int(seed),
        # Detection
        "FPR": detection["FPR"],
        "Recall": detection["Recall"],
        "Precision": detection["Precision"],
        "F1": detection["F1"],
        "anomaly_rate": detection["anomaly_rate"],
        # Regime + topology
        "n_clusters": regime["n_clusters"],
        "mean_var": regime["mean_var"],
        "median_var": regime["median_var"],
        "mean_eff_var": regime["mean_eff_var"],
        "frac_above_r0": regime["frac_above_r0"],
        "regime": regime["regime"],
        "singletons": regime["singletons"],
        "singleton_frac": regime["singleton_frac"],
        "top1_n": regime["top1_n"],
        "top1_frac": regime["top1_frac"],
        "shannon_entropy": regime["shannon_entropy"],
    }


# ── Experiment driver ───────────────────────────────────────────

def run_experiment(mode: str) -> pd.DataFrame:
    if mode == "smoke":
        lambdas, r0s, algos, n_seeds = (
            SMOKE_LAMBDAS, SMOKE_R0S, SMOKE_ALGOS, SMOKE_SEEDS
        )
    elif mode == "full":
        lambdas, r0s, algos, n_seeds = (
            FULL_LAMBDAS, FULL_R0S, FULL_ALGOS, FULL_SEEDS
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    total = len(lambdas) * len(r0s) * len(algos) * n_seeds
    print(f"Experiment 3: Regime Transition (mode={mode})")
    print(f"  d = {D}, anomaly_scale = {ANOMALY_SCALE}sigma")
    print(f"  lambdas ({len(lambdas)}): {lambdas}")
    print(f"  r0s ({len(r0s)}): {r0s}")
    print(f"  algos ({len(algos)}): {algos}")
    print(f"  seeds: {n_seeds}")
    print(f"  total runs: {total}")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    t_start = time.time()
    run_count = 0

    for r0 in r0s:
        for algo_name in algos:
            for lam in lambdas:
                cond_t0 = time.time()
                for seed in range(n_seeds):
                    rows.append(run_single(algo_name, lam, r0, seed))
                    run_count += 1

                cond_dt = time.time() - cond_t0
                cond_total = time.time() - t_start
                rate_overall = run_count / cond_total if cond_total > 0 else 0
                eta_s = (total - run_count) / rate_overall if rate_overall > 0 else 0
                # Last batch n_clusters mean: peek into accumulated rows
                last_n = sum(r["n_clusters"] for r in rows[-n_seeds:]) / n_seeds
                print(
                    f"  [{run_count:5d}/{total}] "
                    f"r0={r0:g}, algo={algo_name:24s}, lambda={lam:9g}, "
                    f"dt={cond_dt:6.1f}s ({n_seeds/cond_dt:5.1f} runs/s), "
                    f"<n_clusters>={last_n:6.1f}, ETA {eta_s:5.0f}s",
                    flush=True,
                )

    elapsed = time.time() - t_start
    df = pd.DataFrame(rows)

    suffix = "_smoke" if mode == "smoke" else ""
    out_path = RESULTS_DIR / f"exp03_results{suffix}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved: {out_path}")
    print(f"Total runs: {len(df)} in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Rate: {len(df) / elapsed:.1f} runs/s")

    # Summary
    print("\nSummary by (algorithm, r0, lambda) -- mean FPR (95% CI t-based):")
    summary = (
        df.groupby(["algorithm", "r0", "lambda"])
        .agg(
            FPR_mean=("FPR", "mean"),
            FPR_std=("FPR", "std"),
            n_clusters_mean=("n_clusters", "mean"),
            frac_above_r0_mean=("frac_above_r0", "mean"),
            regime_mode=("regime", lambda s: s.mode().iloc[0]),
        )
        .reset_index()
    )
    summary["FPR_CI95"] = 1.96 * summary["FPR_std"] / np.sqrt(n_seeds)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))

    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true",
                       help="Reduced sweep for pipeline validation")
    group.add_argument("--full", action="store_true",
                       help="Full sweep (2340 runs)")
    args = parser.parse_args()

    mode = "smoke" if args.smoke else "full"
    run_experiment(mode)


if __name__ == "__main__":
    main()
