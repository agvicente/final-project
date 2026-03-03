# COMPREHENSIVE REPOSITORY ANALYSIS
# IoT Anomaly-Based Intrusion Detection System with Evolutionary Clustering
## Master's Dissertation - UFMG PPGEE

**Analysis Date**: November 6, 2025  
**Repository Location**: `/Users/augusto/mestrado/final-project`  
**Git Status**: Clean (20+ commits, latest: "first results")  

---

## EXECUTIVE SUMMARY

This is a **well-structured research project** for a Master's dissertation in Electrical Engineering (UFMG) focused on **anomaly-based intrusion detection systems (IDS) for IoT networks using evolutionary clustering**. 

**Current Status**: Phase 1 (Baseline) completed with:
- ✅ 705 ML experiments across 10 algorithms
- ✅ Comprehensive baseline results (F1 > 0.99)
- ✅ Reproducible pipeline with Docker + DVC
- ✅ 4-phase incremental research plan (32,642-byte schedule)
- ✅ 220+ catalogued research references
- ✅ Production-ready codebase (5,552 lines)

---

## 1. PROJECT OVERVIEW

### What This Project Is About
Developing a **real-time anomaly detection system** that:
1. Detects IoT intrusions using evolutionary clustering algorithms
2. Adapts to concept drift in network traffic patterns
3. Operates on high-velocity data streams (Kafka-based)
4. Supports heterogeneous IoT device types
5. Balances accuracy, speed, and computational efficiency

### Research Problem
- IoT networks are vulnerable to attacks
- Signature-based IDS fail against novel attacks
- Real-time detection in data streams is challenging
- Devices have heterogeneous capabilities (edge to cloud)
- Class imbalance and concept drift complicate detection

### Key Innovation
**Evolutionary Clustering** (Mixture of Typicalities) that:
- Automatically adapts to changing traffic patterns
- Reduces false positives in anomaly detection
- Handles streaming data without retraining
- Maintains low computational overhead

---

## 2. PROJECT STRUCTURE

### Main Directories

```
final-project/
├── iot-ids-research/              # Main research project (core)
│   ├── data/                       # Dataset management
│   ├── src/eda/                    # Exploratory data analysis
│   ├── experiments/                # ML experiment orchestration
│   ├── configs/                    # YAML configuration files
│   ├── docs/                       # Documentation, papers, diagrams
│   ├── dvc.yaml                    # DVC pipeline (6 stages)
│   ├── requirements.txt            # 250+ Python packages
│   ├── Dockerfile                  # Container setup
│   └── docker-compose.yml          # Multi-service orchestration
│
├── labs/                           # Teaching/methodology labs
│   ├── lab01/                      # Python workspace setup (reproducible science)
│   ├── lab02/                      # MLflow tracking (planned)
│   └── lab03/                      # EDA methodology (planned)
│
├── proposta/                       # Master's proposal documents
│   ├── Thesis_Proposal_v2.pdf      # ~16,000 words
│   └── Pre-research_Analysis.pdf   # Undermind project
│
└── schedule.md                     # 4-phase research plan (32,642 bytes)
```

### Key Result Artifacts
- **experiments/.results/** (172 MB):
  - 11 algorithm-specific analyses
  - 705 experiment results with parameters/metrics
  - Consolidation report with recommendations
  - Bayesian statistical framework

---

## 3. TECHNOLOGY STACK

### Core Technologies
- **Language**: Python 3.12
- **ML Framework**: scikit-learn 1.7.1
- **Data Processing**: pandas 2.3.1, numpy 2.3.2
- **Pipeline Orchestration**: DVC 3.61.0, MLflow 3.1.4
- **Containerization**: Docker 7.1.0
- **Version Control**: Git

### ML Algorithms Tested (10)
**Classification** (6): LogisticRegression, RandomForest, GradientBoostingClassifier, MLPClassifier, LinearSVC, SGDClassifier

**Anomaly Detection** (4): IsolationForest, EllipticEnvelope, LocalOutlierFactor, SGDOneClassSVM

### Visualization Stack
- matplotlib, seaborn (static)
- plotly, bokeh (interactive)
- Custom IoT-specific plots

---

## 4. MAIN COMPONENTS

### Core Experiment Scripts

1. **algorithm_comparison.py** (63 KB)
   - Main orchestrator for all 10 algorithms
   - 705 total experiments (adaptive configuration strategy)
   - Resource monitoring, Bayesian evaluation
   - Detailed logging with execution IDs

2. **run_single_algorithm.py** (12 KB)
   - Individual algorithm execution
   - Parameter grid exploration
   - Generates per-algorithm analysis reports

3. **consolidate_results.py** (44 KB)
   - Aggregates all algorithm results
   - Generates comprehensive reports
   - Creates summary tables and visualizations

### Data Processing Pipeline

1. **dvc_preprocessing.py**: Missing value imputation, normalization, train/test split
2. **dvc_sampling.py**: 10% stratified sampling maintaining attack distribution
3. **dvc_eda.py**: Exploratory statistics
4. **check_dataset_quality.py**: Data quality metrics (12 MB JSON report)

### Advanced Analysis Modules

1. **bayesian_metrics.py**: Brodersen et al. methodology for accuracy evaluation
2. **bayesian_plots.py**: Posterior distribution visualizations
3. **enhanced_metrics_collector.py**: CPU/memory/disk monitoring
4. **iot_advanced_plots.py**: IoT-specific performance visualizations

---

## 5. DATA AND MODELS

### Dataset: CICIoT2023
- **Source**: Canadian Institute for Cybersecurity
- **Current Use**: 10% stratified sample
- **Original Size**: Large-scale IoT network traffic
- **Labels**: Normal vs. Attack (binary) + Attack types (multi-class)

### Best Performing Models

| Model | F1-Score | Accuracy | Speed | Notes |
|-------|----------|----------|-------|-------|
| GradientBoostingClassifier | **0.9964** | **0.9930** | 88.2h | Best overall |
| RandomForest | 0.9964 | 0.9929 | 14.8h | Strong alternative |
| MLPClassifier | 0.9956 | 0.9913 | 13.4h | Neural network |
| SGDOneClassSVM | 0.9911 | 0.9827 | **128.3s** | ⭐ Fastest |
| IsolationForest | 0.9900 | 0.9805 | 20 min | Anomaly specialist |

### Experimental Results
- **Total Experiments**: 705
- **Mean F1-Score**: 0.9933 ± 0.0023
- **Mean Accuracy**: 0.9868 ± 0.0045
- **Stability**: CV = 0.002-0.005 (excellent)
- **Execution Time**: 224,177 seconds (62.3 hours)

---

## 6. DOCUMENTATION

### Research Documentation
- **Master's Proposal** (16,948 words): Problem statement, literature review, methodology
- **Schedule.md** (32,642 bytes): 4-phase research plan with bibliographic references

### Technical Documentation
- **System Diagrams**: Architecture, data flow, experimental workflow, attack taxonomy
- **Papers**: Brodersen et al. (Balanced Accuracy), Maia et al. (Evolutionary Clustering)
- **Meeting Notes**: 2025-10-17, 2025-10-24 progress tracking

### Lab Documentation
- **lab01**: 280+ lines on Python workspace setup (reproducible science focus)
- **lab01/practice_exercises.py**: 10 automated validation checks

### Results Documentation
- 11 individual algorithm reports (plots, tables, statistics)
- Consolidated final report with recommendations
- Summary table, Bayesian comparison matrix, detailed statistics

---

## 7. RESEARCH CONTEXT

### Research Problem
IoT systems face critical security challenges:
- Heterogeneous devices with different capabilities
- High-velocity data streams (impossible for batch processing)
- Concept drift (traffic patterns change over time)
- Resource constraints on edge devices
- False positive critical in security (nuisance + missed attacks)

### Research Gap
Existing work lacks:
- Integration of evolutionary clustering with streaming architectures
- Device-specific models for heterogeneous IoT
- Comprehensive handling of concept drift in real-time detection
- Balance between accuracy and computational efficiency

### Proposed Solution
1. **Evolutionary Clustering** (Mixture of Typicalities - Maia et al. 2020)
2. **Two-Phase Architecture** (anomaly detection + classification)
3. **Device-Specific Models** (per device/type segments)
4. **High-Throughput Streaming** (Kafka + parallel processing)

### Key References
- **Maia et al. (2020)**: Core evolutionary clustering methodology
- **Park (2018)**: Anomaly pattern detection on streams
- **Surianarayanan et al. (2024)**: High-throughput architecture
- **Golestani & Makaroff (2024)**: Device-specific models
- **220+ additional references** catalogued by research phase

---

## 8. RESULTS AND OUTPUTS

### Phase 1 Deliverables (COMPLETED)

✅ **Baseline Established**
- 10 algorithms comprehensively tested
- GradientBoostingClassifier best performer (F1: 0.9964)
- SGDOneClassSVM fastest (128.3s for full dataset)

✅ **Reproducible Pipeline**
- DVC pipeline with 6 stages
- Docker containerization
- Git version control with 20+ commits

✅ **Comprehensive Analysis**
- 705 experiments with detailed metrics
- Bayesian statistical framework
- Individual and consolidated reports
- 50+ visualizations (plots, heatmaps, boxplots)

✅ **Documentation**
- Master's proposal with methodology
- 4-phase research plan (32,642 bytes)
- 220+ research references
- Experiment logs and meeting notes

### Output Artifacts (172 MB)

```
experiments/.results/full/1760998223_*/
├── [Algorithm Name]/
│   ├── individual_analysis/plots/        # 5-6 visualizations
│   ├── individual_analysis/tables/       # Statistics CSVs
│   ├── individual_analysis/report/       # Algorithm-specific report
│   └── individual_analysis/data/         # Raw results
└── consolidation/
    ├── report/                            # Final comprehensive report
    ├── tables/                            # Summary tables, Bayesian matrix
    └── plots/                             # Algorithm comparisons
```

---

## 9. CURRENT STATUS

### Phase 1: ✅ BASELINE EXPERIMENTS (COMPLETE)
- Baseline established on CICIoT2023
- 10 algorithms tested with 705 experiments
- Concept drift analysis initiated
- Results documented with recommendations

### Phase 2: 🔄 PLANNED (Months 4-6)
- Evolutionary clustering implementation
- Streaming architecture with Kafka
- Real-time latency evaluation

### Phase 3: 📋 PLANNED (Months 7-9)
- Device-specific model development
- Two-phase architecture integration
- Multi-dataset evaluation

### Phase 4: 📋 PLANNED (Months 10-12)
- Production optimization
- Scalability testing (1K-100K devices)
- Final dissertation

### Publications Target
- Month 3: Workshop paper (baseline comparison)
- Month 6: Conference paper (evolutionary clustering)
- Month 9: Journal paper (two-phase architecture)
- Month 12: Main dissertation

---

## 10. KEY STRENGTHS

1. **Reproducible Science**
   - Full Docker containerization
   - DVC pipeline for data versioning
   - Detailed logging and artifact tracking

2. **Comprehensive Evaluation**
   - 705 experiments with statistical rigor (5 runs per config)
   - Multiple metrics (accuracy, F1, balanced accuracy, Bayesian)
   - Resource monitoring (CPU, memory)
   - Stability analysis (CV < 0.01)

3. **IoT-Focused Design**
   - Adaptive configuration strategy by complexity
   - Light/sweet-spot/medium/heavy ranges
   - Edge deployment consideration
   - Real-time performance metrics

4. **Advanced Analytics**
   - Bayesian accuracy evaluation
   - Enhanced metrics collection
   - Automatic report generation
   - Multiple visualization styles

5. **Excellent Planning**
   - 4-phase incremental research structure
   - 220+ organized research references
   - Clear hypothesis testing
   - Publication milestones

---

## 11. RECOMMENDATIONS

### For Phase 2 Continuation
1. Implement evolutionary clustering from Maia et al. (2020)
2. Test concept drift detection on temporal windows
3. Set up Kafka-based streaming pipeline
4. Segment dataset by device type for device-specific models

### For Production Use (Current Results)
1. **Best Accuracy**: GradientBoostingClassifier (F1: 0.9964)
2. **Best Speed**: SGDOneClassSVM (128.3s) for edge
3. **Balanced**: RandomForest for trade-off
4. **Anomaly Focus**: IsolationForest for specialized scoring

### For Further Research
1. Cross-validate on CICIDS2017/2018
2. Test on real IoT network traces
3. Develop interpretability (SHAP, LIME)
4. Optimize for specific IoT scenarios

---

## CONCLUSION

This is an **excellent research project** demonstrating:

- ✅ Strong scientific rigor and reproducibility
- ✅ Comprehensive ML model evaluation (10 algorithms, 705 experiments)
- ✅ Clear research trajectory (4 phases over 12 months)
- ✅ Production-ready technical infrastructure
- ✅ Excellent documentation and organization
- ✅ Well-established baseline for innovation

**The foundation is solid for pursuing novel evolutionary clustering approaches in Phase 2-4.**

---

**Repository Analysis Complete**  
For detailed file-by-file analysis, consult the directory structure and source code comments.

