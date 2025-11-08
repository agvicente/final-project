# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Repository Scope**: This guide is specifically for the `iot-ids-research` research repository within the `final-project` directory. For workspace-level organization, see the parent directory's README.md.

## Project Overview

This is a Master's dissertation research project at UFMG (Electrical Engineering) developing an **anomaly-based intrusion detection system (IDS) for IoT networks using evolutionary clustering algorithms**. The project follows a 4-phase incremental research plan with baseline experiments completed (Phase 1).

**Current Status**: Phase 1 Complete - 705 experiments executed, 10 ML algorithms evaluated on CICIoT2023 dataset with excellent results (F1 > 0.99).

## Essential Commands

### DVC Pipeline Execution

The project uses DVC for reproducible ML pipelines. All stages are defined in `iot-ids-research/dvc.yaml`.

**Full pipeline execution** (run all stages sequentially):
```bash
cd iot-ids-research
dvc repro
```

**Run specific stage**:
```bash
cd iot-ids-research
dvc repro <stage_name>
```

**Individual experiment stages** (ordered by computational complexity):
- `exp_logistic_regression` - Fastest, O(n)
- `exp_random_forest` - Fast, O(n log n)
- `exp_gradient_boosting` - Moderate, O(n log n)
- `exp_isolation_forest` - Fast anomaly detection, O(n log n)
- `exp_elliptic_envelope` - Moderate anomaly detection, O(n²)
- `exp_local_outlier_factor` - Heavy anomaly detection, Ball Tree optimized
- `exp_linear_svc` - Heavy, optimized for large datasets
- `exp_sgd_classifier` - Fast, stochastic gradient descent SVM
- `exp_sgd_one_class_svm` - Optimized OneClassSVM via SGD
- `exp_mlp` - Neural network, CPU-optimized

**Pipeline stages**:
- `check_quality` - Dataset quality validation
- `sampling` - Stratified sampling (10% of CICIoT2023)
- `eda` - Exploratory data analysis
- `preprocess` - Data normalization and train/test splitting
- `consolidate_results` - Aggregate results from all experiments

### Running Experiments

**Single algorithm experiment** (direct execution):
```bash
cd iot-ids-research
python3 experiments/run_single_algorithm.py <algorithm_name>
```

Algorithm names: `logistic_regression`, `random_forest`, `gradient_boosting`, `isolation_forest`, `elliptic_envelope`, `local_outlier_factor`, `linear_svc`, `sgd_classifier`, `sgd_one_class_svm`, `mlp`

**Consolidate results after experiments**:
```bash
cd iot-ids-research
python3 experiments/consolidate_results.py
```

### Docker Environment

**Start research environment** (Jupyter Lab + MLflow):
```bash
docker-compose up -d
```

**Stop environment**:
```bash
docker-compose down
```

**Access services**:
- Jupyter Lab: http://localhost:8888
- MLflow: http://localhost:5000

**Rebuild container after dependency changes**:
```bash
docker-compose build --no-cache
```

### Local Python Environment Setup

**Create virtual environment** (alternative to Docker):
```bash
cd iot-ids-research
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Note**: The project has 250+ dependencies including scikit-learn, pandas, numpy, MLflow, DVC, and visualization libraries. Docker is recommended for consistency, but local setup works for development.

### MLflow Experiment Tracking

**View tracked experiments** (Docker environment):
1. Start services: `docker-compose up -d`
2. Access MLflow UI: http://localhost:5000
3. Browse experiments by algorithm name, timestamp, and parameters

**MLflow features**:
- Automatic logging of all experiment parameters and metrics
- Experiment comparison across different runs
- Model artifact storage
- Parameter search visualization
- Metric plots and analysis

**Experiment naming convention**: `<algorithm_name>_<timestamp>`

To view experiments from command line:
```bash
cd iot-ids-research
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Test Mode vs. Full Mode

Experiments support two execution modes controlled by `TEST_MODE` in `iot-ids-research/experiments/algorithm_comparison.py`:

- `TEST_MODE = True`: Uses 1000 samples, 1 run per config (fast prototyping)
- `TEST_MODE = False`: Full dataset, 5 runs per config (production experiments)

Results are stored in separate directories: `experiments/results/test/` or `experiments/results/full/`

## Code Architecture

### Experiment Pipeline Flow

```
1. Sampling (dvc_sampling.py)
   ↓ Creates data/processed/sampled.csv (10% stratified sample)

2. Preprocessing (dvc_preprocessing.py)
   ↓ Normalizes, splits train/test, handles missing values
   ↓ Outputs: X_train/test.npy, y_train/test.npy, scaler.pkl
   ↓ Creates both multi-class and binary variants

3. Individual Experiments (run_single_algorithm.py → algorithm_comparison.py)
   ↓ Runs grid search with multiple parameter configurations
   ↓ Executes N_RUNS repetitions per config for statistical rigor
   ↓ Collects: metrics, confusion matrices, resource usage, timing
   ↓ Saves: experiments/results/<mode>/<timestamp>_<algorithm>/results.json

4. Individual Analysis (individual_analysis.py)
   ↓ Generates algorithm-specific visualizations and reports
   ↓ Creates: plots/, tables/, report/ subdirectories

5. Consolidation (consolidate_results.py)
   ↓ Aggregates all algorithms results
   ↓ Bayesian comparison analysis
   ↓ Cross-algorithm performance rankings
   ↓ Final report generation
```

### Key Scripts and Their Roles

**Data Processing**:
- `dvc_sampling.py` - Stratified sampling preserving class distributions
- `dvc_preprocessing.py` - Feature normalization, missing value handling, train/test split
- `check_dataset_quality.py` - Dataset validation and quality metrics

**Experimentation**:
- `algorithm_comparison.py` - Core experiment orchestrator (63KB, 1500+ lines)
  - Contains all algorithm configurations and grid search parameters
  - Implements adaptive sampling strategy based on computational complexity
  - Enhanced metrics collection (CPU, memory, disk I/O)
  - MLflow integration for experiment tracking
- `run_single_algorithm.py` - DVC stage wrapper for individual algorithms
  - Manages shared timestamp across experiment runs
  - Algorithm-specific logging setup
- `individual_analysis.py` - Per-algorithm detailed analysis and visualization
- `consolidate_results.py` - Cross-algorithm comparison and final reporting

**Advanced Metrics**:
- `bayesian_metrics.py` - Brodersen et al. Bayesian accuracy evaluation
- `bayesian_plots.py` - Bayesian comparison visualizations
- `enhanced_metrics_collector.py` - System resource monitoring (CPU/memory/disk)
- `iot_advanced_plots.py` - IoT-specific performance visualizations

### Results Directory Structure

```
experiments/results/
├── test/                           # TEST_MODE = True results
│   └── <timestamp>_<algorithm>/
│       ├── results.json            # Raw experiment data
│       └── individual_analysis/
│           ├── plots/              # Performance visualizations
│           ├── tables/             # Statistical summaries
│           └── report/             # Markdown reports
└── full/                           # TEST_MODE = False results
    ├── <timestamp>_<algorithm>/    # Same structure
    └── consolidation/              # Cross-algorithm analysis
        └── <timestamp>_consolidation/
```

### Shared Timestamp Mechanism

Experiments use a shared timestamp file (`.current_run_timestamp`) to group related runs:
- Created by first algorithm execution
- Shared across all algorithms in same batch
- Cleaned up by consolidation script
- Ensures results can be aggregated correctly

### Data Pipeline Outputs

**Binary classification** (Normal vs. Attack):
- `data/processed/binary/X_train_binary.npy`
- `data/processed/binary/y_train_binary.npy`
- `data/processed/binary/scaler.pkl`
- `data/processed/binary/binary_metadata.json`

**Multi-class classification** (attack types):
- `data/processed/X_train.npy`
- `data/processed/y_train.npy`
- `data/processed/scaler.pkl`
- `data/processed/preprocessing_metadata.json`

Current experiments use binary classification for Phase 1 baseline.

## Important Configuration Files

**DVC Configuration**:
- `dvc.yaml` - Pipeline stage definitions with dependencies
- `dvc.lock` - Locked versions of data and outputs
- `.dvc/config` - DVC remote storage configuration

**Docker Configuration**:
- `docker-compose.yml` - Multi-container orchestration (Jupyter + MLflow + PostgreSQL)
- `Dockerfile` - Research environment with all dependencies
- `docker-entrypoint.sh` - Container initialization script

**Experiment Configuration**:
- `configs/preprocessing.yaml` - Feature engineering parameters
- `algorithm_comparison.py` - Algorithm hyperparameters and grid search configs

## Research Context

**Dataset**: CICIoT2023 (Canadian Institute for Cybersecurity)
- Real IoT network traffic with labeled attacks
- Currently using 10% stratified sample for experiments
- Binary labels: Benign (0) vs. Attack (1)

**Research Objectives**:
1. Phase 1 (Complete): Establish baseline with classical algorithms
2. Phase 2 (Months 4-6): Implement evolutionary clustering, Kafka streaming
3. Phase 3 (Months 7-9): Device-specific models, two-phase architecture
4. Phase 4 (Months 10-12): Optimization, scalability testing, dissertation

**Key Innovation Areas**:
- Evolutionary clustering for concept drift adaptation
- Streaming architecture for real-time detection
- Device-specific models for heterogeneous IoT
- Two-phase detection (anomaly + classification)

## Viewing Results

**Individual algorithm reports**:
```
experiments/results/<mode>/<timestamp>_<algorithm>/individual_analysis/report/individual_report.md
```

**Consolidated analysis**:
```
experiments/results/<mode>/consolidation/<timestamp>_consolidation/final_report.md
```

**Visualizations**:
- Individual: `experiments/results/<mode>/<timestamp>_<algorithm>/individual_analysis/plots/`
- Consolidated: `experiments/results/<mode>/consolidation/<timestamp>_consolidation/plots/`

**MLflow UI** (interactive):
- Access http://localhost:5000 (when Docker is running)
- Compare experiments, view metrics, and explore parameters interactively
- See the MLflow Experiment Tracking section for details

## Research Acceleration System (NEW!)

**Sistema configurado em 2025-11-08 para acelerar desenvolvimento do mestrado.**

### Quick Commands
- `/resume` - Carrega contexto atual e próximos passos
- `/start-sprint` - Inicia nova semana de trabalho
- `/finalize-week` - Gera relatório para orientador
- `/paper-summary [nome]` - Resume paper do Zotero

### Skills Automáticas (9 total)
- `iot-ids-research-context` - Contexto sempre ativo
- `evolutionary-clustering-guide` - Ensina clustering evolutivo
- `kafka-streaming-iot` - Guia de streaming
- `paper-reading-accelerator` - Resume papers rapidamente
- `experiment-design-validator` - Valida rigor científico
- `scientific-paper-writer` - Escreve papers incrementalmente
- `dissertation-writer` - Escreve dissertação (PT/EN)
- `overleaf-formatter-artigo` - Mantém formatação artigo1
- `overleaf-formatter-dissertation` - Mantém formatação dissertação

### Hooks de Automação
- **Auto-save a cada 10min** - Proteção contra travamentos
- **Session start/end** - Carrega e salva contexto automaticamente

### Documentação Evolutiva
- **docs/SESSION_CONTEXT.md** - "Cérebro" do projeto (SEMPRE LEIA ESTE ARQUIVO)
- **docs/weekly-reports/current-week.md** - Relatório semanal vivo
- **docs/progress/** - Logs de cada sessão

### Guias
- **QUICK_START_GUIDE.md** - Como usar o sistema
- **ZOTERO_SETUP.md** - Configurar integração Zotero

---

## Notes for Development

- Always check `TEST_MODE` setting before running full experiments (can take 60+ hours)
- DVC stages are idempotent - rerunning only processes changed dependencies
- Experiments generate ~172MB of results per full run (10 algorithms)
- MLflow tracks all experiments automatically when running via `algorithm_comparison.py`
- Resource usage is monitored and logged for every experiment run
- Binary classification is used for Phase 1; multi-class support exists for future phases
- **NEW:** Use `/resume` para carregar contexto completo em qualquer sessão
