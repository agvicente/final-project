# TEDA in High Dimensions

Analysis of the dimensional sensitivity of variance estimation in MicroTEDAclus
(Maia et al., 2020) evolving clustering algorithm.

## The Problem

The original MicroTEDAclus implementation uses a variance formula
`(norm(delta) * 2 / dim)^2` that systematically underestimates variance in
high-dimensional data. At d=17 (typical for network flow features), variance
is underestimated by ~70x, causing false positive rates of 42-75%.

## Key Finding

The `(2/d)^2` scaling factor **self-cancels** in the eccentricity ratio for
n >= 3 (the Chebyshev test), meaning the primary anomaly detection mechanism
is actually self-consistent. The failures occur in:

1. **Macro-cluster intersection** — real Euclidean distance vs scaled-down radius
2. **Young cluster test (n < 3)** — scaled variance vs fixed threshold r0
3. **Life decay** — mixed scaled/real distance calculations

## Experiments

| # | Experiment | Purpose |
|---|-----------|---------|
| E1 | Variance Scaling | Variance ratio vs dimensionality |
| E2 | Eccentricity Inflation | Eccentricity distribution vs d |
| E3 | FPR Degradation | End-to-end FPR vs d |
| E4 | Ablation Study | Isolate each of 5 modifications |
| E5 | d=2 Equivalence | Confirm bug dormancy at low d |
| E6 | IoT Validation | CICIoT2023 real-world data |
| E7 | Theory vs Empirical | Overlay proof predictions on data |

## Quick Start

```bash
pip install -e ".[dev]"
pytest tests/ -v
python experiments/exp01_variance_scaling.py
```

## References

- Angelov, P. (2014). "Outside the box: an alternative data analytics framework."
- Maia, J. et al. (2020). "Evolving clustering algorithm based on mixture of typicalities."
- Original code: https://github.com/cseveriano/evolving_clustering
