---
name: experiment-design-validator
description: Validates scientific rigor of experiment designs. Ensures reproducibility, proper baselines, statistical significance, and alignment with Phase 1 methodology.
version: 1.0.0
activate_when:
  - "experiment"
  - "test"
  - "validate"
  - "design experiment"
---

# Experiment Design Validator

## Purpose
Ensure experiments meet scientific standards and are comparable with Phase 1 baseline.

## Validation Checklist

### 1. Hypothesis
- [ ] Clear, testable hypothesis stated
- [ ] Null hypothesis defined
- [ ] Success criteria quantified

### 2. Dataset
- [ ] Uses CICIoT2023 (consistency with Phase 1)
- [ ] Train/test split matches Phase 1 (or justified if different)
- [ ] No data leakage
- [ ] Stratified sampling if using subset

### 3. Baseline Comparisons
- [ ] Compares with Phase 1 results (at minimum)
- [ ] Includes relevant algorithms from literature
- [ ] Fair comparison (same data, same metrics)

### 4. Metrics
- [ ] Standard metrics: Accuracy, Precision, Recall, F1, AUC-ROC
- [ ] Balanced Accuracy (for imbalanced data)
- [ ] Computational metrics: time, memory, throughput
- [ ] Domain-specific: detection latency, false positive rate

### 5. Statistical Rigor
- [ ] Multiple runs (minimum 5, like Phase 1)
- [ ] Report mean ± std deviation
- [ ] Statistical significance tests (t-test, Wilcoxon)
- [ ] Coefficient of variation for stability

### 6. Reproducibility
- [ ] Random seeds set
- [ ] Parameters documented
- [ ] Code in git with commit hash
- [ ] DVC pipeline stage defined
- [ ] MLflow tracking enabled

### 7. Visualization
- [ ] Confusion matrix
- [ ] ROC curve
- [ ] Performance over time (for streaming)
- [ ] Resource usage plots

## Phase 1 Standards (MUST MATCH)

From baseline experiments:
- 5 runs per configuration
- Grid search for hyperparameters
- MLflow tracking
- Binary classification (Normal vs Attack)
- Metrics: Accuracy, F1, Balanced Accuracy, AUC
- Resource monitoring (CPU, memory, time)

## Example Validation

**Augusto:** "I want to test evolutionary clustering vs K-means"

**You validate:**
```markdown
✅ Good hypothesis: "Evolutionary clustering adapts faster to concept drift than K-means"

Experiment design:
1. Dataset: CICIoT2023 temporal windows (10% sample, same as Phase 1)
2. Baselines: K-means (retrain each window), DBSCAN, Phase 1 GradientBoosting
3. Metrics: Silhouette score, detection F1, adaptation time, CPU usage
4. Setup: 5 runs, vary learning rate α ∈ {0.1, 0.3, 0.5}, 10 temporal windows
5. Statistical test: Paired t-test between evolutionary and K-means F1 scores

Checks:
✅ Clear hypothesis
✅ Uses Phase 1 dataset
✅ Compares with Phase 1 baseline
✅ Standard metrics
✅ 5 runs for statistics
✅ Reproducible (DVC + MLflow)

Suggested additions:
- Add adaptation speed metric (how fast F1 recovers after drift)
- Plot F1 over time windows
- Report p-value for significance
```

## Red Flags

❌ Only 1 run (not reproducible)
❌ Different dataset than Phase 1 (not comparable)
❌ No baseline comparison
❌ Metrics not defined
❌ Parameters not documented
❌ "Looks good" without numbers

## Integration with DVC

Every experiment should be a DVC stage:
```yaml
exp_evolutionary_clustering:
  cmd: python experiments/run_evolutionary.py
  deps:
    - experiments/run_evolutionary.py
    - data/processed/binary/
  params:
    - experiments/params.yaml:evolutionary_clustering
  metrics:
    - experiments/results/evolutionary/metrics.json
```

---
**Use this before running any experiment to ensure scientific quality.**
