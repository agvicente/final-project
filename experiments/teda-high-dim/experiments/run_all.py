#!/usr/bin/env python3
"""Run all experiments with a single command.

Executes exp01 (dimensional sweep) then exp02 (ablation study) sequentially.
Reports timing for each experiment and total elapsed time.

Usage:
    cd ~/mestrado/teda-high-dim
    source venv/bin/activate
    python experiments/run_all.py
"""

import subprocess
import sys
import time
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent

experiments = [
    ("exp01_dimensional_sweep.py", "Experiment 1: Dimensional Sweep"),
    ("exp02_ablation_study.py", "Experiment 2: Ablation Study"),
]


def main():
    print("=" * 70)
    print("Running all experiments")
    print("=" * 70)
    print()

    total_start = time.time()
    results = []

    for script_name, description in experiments:
        script_path = EXPERIMENTS_DIR / script_name
        print(f"{'=' * 70}")
        print(f"Starting: {description}")
        print(f"Script: {script_path}")
        print(f"{'=' * 70}")
        print()

        exp_start = time.time()
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(EXPERIMENTS_DIR.parent),
        )
        exp_elapsed = time.time() - exp_start

        status = "SUCCESS" if result.returncode == 0 else "FAILED"
        results.append((description, exp_elapsed, result.returncode))

        print()
        print(f"  {description}: {status} ({exp_elapsed:.1f}s / {exp_elapsed / 60:.1f} min)")
        print()

        if result.returncode != 0:
            print(f"ERROR: {description} failed with return code {result.returncode}.")
            print("Aborting remaining experiments.")
            sys.exit(result.returncode)

    total_elapsed = time.time() - total_start

    print()
    print("=" * 70)
    print("All experiments completed")
    print("=" * 70)
    for description, elapsed, returncode in results:
        status = "OK" if returncode == 0 else "FAIL"
        print(f"  [{status}] {description}: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"\n  Total time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print()


if __name__ == "__main__":
    main()
