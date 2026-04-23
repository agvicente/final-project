# CLAUDE.md

## Project

Research project analyzing the dimensional sensitivity of the MicroTEDAclus
evolving clustering algorithm (Maia 2020). The original variance formula
`(norm*2/dim)^2` causes catastrophic false positive rates in high-dimensional
data (d > 10). This project provides mathematical proofs, synthetic experiments,
and IoT validation.

**Target:** Full conference paper (8-10 pages, IEEE format).

## Structure

```
src/teda_hd/algorithms/   <- Original, corrected, and ablation variants
src/teda_hd/metrics/      <- Variance ratio, FPR, eccentricity analysis
src/teda_hd/generators/   <- Synthetic data and IoT loader
experiments/              <- One script per experiment (E1-E7)
paper/                    <- LaTeX manuscript
tests/                    <- Unit tests
```

## Commands

```bash
# Setup
pip install -e ".[dev]"

# Tests
pytest tests/ -v

# Run single experiment
python experiments/exp01_variance_scaling.py

# Run all experiments
python experiments/run_all.py

# Generate paper figures
python experiments/run_all.py --figures-only
```

## Key Design Decisions

- **No dependency on `evolclustering` package** — algorithms reimplemented from scratch
  for isolation and control (no numba/JIT)
- **8 ablation variants** toggle individual modifications to isolate effects
- **30 repetitions per condition** with fixed seeds for reproducibility
- **Statistical tests** (t-test, ANOVA, Tukey HSD) for all comparisons

## Integration

This module is part of `final-project/experiments/teda-high-dim/`.
Install as editable package in the streaming venv:
```bash
cd experiments/teda-high-dim && pip install -e .
```
This makes `teda_hd` importable from the streaming pipeline (e.g., `variant_micro_teda.py`).
