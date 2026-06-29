# Results

Raw experiment results are stored here. CSVs and statistical reports are
versioned (`exp0*_*.csv`, `exp0*_statistical_tests.txt`); plots and large
intermediate files are not (gitignored).

## Layout

```
results/
├── exp01_dimensional_sweep.csv          # Exp 1: 1440 runs (V0 vs V7 across d, r0)
├── exp02_ablation.csv                   # Exp 2: 240 runs (V0..V7 ablation, d=17)
├── exp02_statistical_tests.txt          # Exp 2: ANOVA, Friedman, Nemenyi report
├── exp03_results.csv                    # Exp 3: 1620 runs (V0/V7 x r0 x lambda x seed)
├── exp03_summary.csv                    # Exp 3: aggregated mean/std/CI per condition
├── exp03_statistical_tests.txt          # Exp 3: Friedman, Cohen's d, transition fit
├── plots/                               # Diagnostic plots (PNG)
└── paper_figures/                       # Paper-quality PDFs
    ├── fig_dimensional_sweep.pdf        # Exp 1
    ├── fig_ablation_boxplot.pdf         # Exp 2
    ├── fig_synthetic_vs_real.pdf        # Synthetic-to-IoT comparison
    ├── fig_regime_transition_v7.pdf     # Exp 3: V7 transition across r0
    ├── fig_regime_v0_vs_v7.pdf          # Exp 3: V0 vs V7 qualitative comparison
    └── fig_regime_phase_diagram.pdf     # Exp 3: 2D phase diagram (lambda x r0)
```

## Regeneration

```bash
# All experiments + figures
python experiments/run_all.py

# Single experiments
python experiments/exp01_dimensional_sweep.py
python experiments/exp02_ablation_study.py
python experiments/exp03_regime_transition.py --full

# Analyses (after experiment run)
python experiments/exp03_statistical_analysis.py
python experiments/exp03_generate_figures.py
```

## Conventions

- **30 seeds** per condition (project standard).
- **CSV long format** (one row per run) for raw results.
- **Statistical tests:** Friedman + Nemenyi (non-parametric) + ANOVA + Tukey HSD (parametric).
- **CI 95%** via bootstrap (1000 samples) for FPR; via t-distribution as fallback.
- **Reproducibility:** seeds 0..29 fixed; results bit-exact across runs of same code.
