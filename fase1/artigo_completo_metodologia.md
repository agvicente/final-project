# Comparative Analysis of Machine Learning Algorithms for Anomaly Detection in IoT Networks Using CICIoT2023 Dataset

## Abstract

Internet of Things (IoT) networks face increasing security threats due to their heterogeneous nature and resource constraints. This study presents a comprehensive comparison of machine learning algorithms for anomaly detection in IoT environments using the CICIoT2023 dataset. We implemented a reproducible pipeline using Data Version Control (DVC) to evaluate seven different algorithms across multiple configurations. Our methodology employs stratified sampling, standardized preprocessing, and binary classification to distinguish between benign and malicious network traffic. The experimental framework includes rigorous statistical validation through multiple runs and comprehensive performance metrics including accuracy, precision, recall, F1-score, and ROC AUC. This research contributes to the understanding of algorithm performance in IoT intrusion detection and provides a baseline for future comparative studies.

**Keywords:** IoT Security, Anomaly Detection, Machine Learning, Intrusion Detection, Binary Classification

---

## 1. Introduction

The proliferation of Internet of Things (IoT) devices has created new attack vectors and security challenges in network environments. Traditional signature-based intrusion detection systems are inadequate for IoT networks due to the diverse and evolving nature of attacks. Machine learning-based anomaly detection offers a promising approach to identify malicious activities in IoT networks by learning patterns from normal behavior and detecting deviations.

This study addresses the research question: **"Which machine learning algorithms are most effective for binary anomaly detection in IoT network traffic?"** We conduct a systematic comparison of seven algorithms using standardized experimental conditions and comprehensive evaluation metrics.

### 1.1 Research Objectives

**Primary Objectives:**
1. Establish a comprehensive baseline for ML algorithm performance in IoT anomaly detection
2. Validate the effectiveness of stratified sampling for large-scale IoT datasets
3. Provide reproducible experimental framework for future research
4. Quantify computational and memory requirements for each algorithm

**Secondary Objectives:**
1. Identify optimal hyperparameter configurations for each algorithm
2. Analyze trade-offs between accuracy and computational efficiency
3. Establish statistical significance of performance differences
4. Document best practices for IoT anomaly detection experimentation

---

## 2. Related Work

### 2.1 IoT Security Challenges

IoT networks present unique security challenges including:
- **Resource Constraints**: Limited computational and memory resources
- **Heterogeneity**: Diverse devices with different protocols and behaviors
- **Scale**: Large number of devices generating high-volume traffic
- **Dynamic Topology**: Devices frequently joining and leaving the network

### 2.2 Machine Learning for IoT Security

Previous studies have shown the effectiveness of machine learning approaches for IoT intrusion detection. However, most studies focus on specific algorithms or limited datasets, lacking comprehensive comparative analysis under standardized conditions.

### 2.3 Research Gap

This study addresses the gap by providing:
- Systematic comparison of multiple algorithm families
- Standardized experimental conditions
- Statistical rigor through multiple runs
- Comprehensive resource usage analysis
- Reproducible experimental pipeline

---

## 3. Materials and Methods

### 3.1 Dataset

#### 3.1.1 CICIoT2023 Dataset Description

**Source**: Canadian Institute for Cybersecurity  
**Original Size**: ~23 million network traffic records  
**Features**: 46 network flow features  
**Attack Types**: DDoS, Mirai, Recon, Spoofing, Web-based, Brute Force, Man-in-the-Middle  
**Capture Environment**: Realistic IoT testbed with 105 devices

#### 3.1.2 Sampling Strategy

Due to computational constraints, we implemented a **stratified sampling approach**:

```
Total Sample: 4,501,906 records (19.5% of original dataset)
├── Training Set: 3,601,524 records (80%)
└── Test Set: 900,382 records (20%)

Binary Distribution:
├── Benign Traffic: 105,137 records (2.3%)
└── Malicious Traffic: 4,396,769 records (97.7%)
    ├── DDoS: ~65%
    ├── Mirai: ~15%
    ├── Reconnaissance: ~10%
    ├── Web-based: ~5%
    ├── Spoofing: ~3%
    └── Others: ~2%
```

**Sampling Methodology**:
1. **Proportional Stratification**: Maintained original class proportions
2. **Random State**: Fixed seed (42) for reproducibility
3. **Quality Assurance**: Automated validation of sample representativeness
4. **Temporal Coverage**: Preserved temporal patterns in the sample

**Statistical Validation**:
- Kolmogorov-Smirnov test for distribution comparison

### 3.2 Experimental Design

#### 3.2.1 Research Design Framework

**Study Type**: Quantitative experimental design  
**Comparison Approach**: Seven machine learning algorithms  
**Classification Task**: Binary (Benign vs. Malicious)  
**Validation Method**: Stratified train-test split  
**Statistical Rigor**: Multiple runs per configuration (n=10)  
**Reproducibility**: DVC-based pipeline with version control

#### 3.2.2 Algorithm Selection

We selected seven algorithms representing different learning paradigms:

**Supervised Learning Algorithms**:
1. **Random Forest**: Ensemble method with bagging
2. **Gradient Boosting**: Ensemble method with boosting  
3. **Logistic Regression**: Linear probabilistic classifier
4. **Support Vector Classifier (SVC)**: Kernel-based classifier
5. **Multi-Layer Perceptron (MLP)**: Neural network approach

**Unsupervised/Semi-Supervised Algorithms**:
6. **Isolation Forest**: Tree-based anomaly detection
7. **One-Class SVM**: Novelty detection approach

### 3.3 Data Preprocessing Pipeline

#### 3.3.1 Pipeline Architecture (DVC)

```yaml
stages:
  1. check_quality    → Validação e métricas de qualidade dos dados
  2. sampling         → Amostragem estratificada do dataset
  3. eda              → Análise exploratória dos dados
  4. preprocess       → Normalização e preparação dos dados
  5. baseline_experiment → Execução dos experimentos de ML
```

#### 3.3.2 Quality Control Stage

**Input**: 63 CSV files (raw CICIoT2023 data)  
**Process**:
- Missing value detection and quantification
- Data type consistency validation
- Outlier identification using statistical methods
- Integrity verification across files

**Output**: Quality metrics and validation reports

#### 3.3.3 Sampling Stage

**Stratified Sampling Implementation**:
```python
def stratified_sampling():
    # 1. Análise de distribuição em todos os 63 arquivos
    # 2. Cálculo de quotas proporcionais por tipo de ataque
    # 3. Seleção aleatória estratificada por arquivo
    # 4. Validação estatística da representatividade
    # 5. Geração de arquivo consolidado
```

**Parameters**:
- Sample proportion: 19.5% of original dataset
- Stratification variable: Attack type
- Random state: 42 (reproducibility)
- Validation: Statistical tests for representativeness

#### 3.3.4 Feature Engineering

**Original Features**: 46 columns from CICIoT2023  
**Final Features**: 39 columns after preprocessing

**Feature Categories**:

**Network Layer Features**:
- Header_Length, Protocol Type, Time_To_Live, Rate

**Transport Layer Features**:
- TCP Flags: fin_flag_number, syn_flag_number, rst_flag_number, psh_flag_number, ack_flag_number, ece_flag_number, cwr_flag_number
- Flag Counts: ack_count, syn_count, fin_count, rst_count

**Application Layer Features**:
- Protocols: HTTP, HTTPS, DNS, Telnet, SMTP, SSH, IRC
- Network Types: TCP, UDP, DHCP, ARP, ICMP, IGMP, IPv, LLC

**Statistical Features**:
- Tot sum, Min, Max, AVG, Std, Tot size, IAT, Number, Variance

**Preprocessing Steps**:
1. **Missing Value Treatment**: 
   - Method: Mode imputation (more conservative than mean/median)
   - Scope: All features except target variable
   - Validation: Pre/post statistics comparison

2. **Label Binarization**:
   - BENIGN → 0 (Normal traffic)
   - All attack types → 1 (Malicious traffic)

3. **Feature Selection**:
   - Removed constant/near-constant columns
   - Eliminated highly correlated features (r > 0.95)
   - Retained 39 most informative features

4. **Data Normalization**:
   - Method: StandardScaler (μ=0, σ=1)
   - Split-aware: Fitted only on training data (prevents data leakage)
   - Scope: All numerical features

#### 3.3.5 Split Strategy

```python
# Train-test split BEFORE normalization (prevents data leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y_binary, 
    random_state=42
)

# Normalization fitted only on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit_transform
X_test_scaled = scaler.transform(X_test)        # transform only
```

### 3.4 Experimental Configuration

#### 3.4.1 Algorithm Parameters

Each algorithm was tested with multiple parameter configurations to ensure comprehensive evaluation:

**Random Forest**:
```python
params = [
    {'n_estimators': 50, 'max_depth': 10, 'random_state': 42},
    {'n_estimators': 100, 'max_depth': 15, 'random_state': 42},
    {'n_estimators': 200, 'max_depth': 20, 'random_state': 42}
]
```

**Gradient Boosting**:
```python
params = [
    {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 5, 'random_state': 42},
    {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 7, 'random_state': 42}
]
```

**Logistic Regression**:
```python
params = [
    {'C': 0.1, 'max_iter': 1000, 'random_state': 42},
    {'C': 1.0, 'max_iter': 1000, 'random_state': 42},
    {'C': 10.0, 'max_iter': 1000, 'random_state': 42}
]
```

**Support Vector Classifier**:
```python
params = [
    {'C': 1.0, 'kernel': 'rbf', 'random_state': 42, 'probability': True},
    {'C': 10.0, 'kernel': 'rbf', 'random_state': 42, 'probability': True}
]
```

**Multi-Layer Perceptron**:
```python
params = [
    {'hidden_layer_sizes': (50,), 'max_iter': 500, 'random_state': 42},
    {'hidden_layer_sizes': (100, 50), 'max_iter': 500, 'random_state': 42}
]
```

**Isolation Forest**:
```python
params = [
    {'contamination': 0.1, 'n_estimators': 100, 'random_state': 42},
    {'contamination': 0.05, 'n_estimators': 100, 'random_state': 42}
]
```

**One-Class SVM**:
```python
params = [
    {'nu': 0.1, 'kernel': 'rbf'},
    {'nu': 0.05, 'kernel': 'rbf'}
]
```

#### 3.4.2 Experimental Execution

**Statistical Rigor**:
- **Multiple Runs**: 10 independent executions per parameter configuration
- **Random State Control**: Fixed seeds for all randomized components
- **Memory Management**: Aggressive cleanup between experiments
- **Progress Monitoring**: Real-time tracking of execution and resource usage

**Execution Framework**:
```python
for algorithm in algorithms:
    for param_config in parameter_grid:
        for run in range(10):  # Statistical rigor
            # Execute single experiment
            # Monitor memory and time
            # Record all metrics
            # Clean memory
```

### 3.5 Evaluation Metrics

#### 3.5.1 Primary Performance Metrics

1. **Accuracy**: Overall classification correctness
   ```
   Accuracy = (TP + TN) / (TP + TN + FP + FN)
   ```

2. **Precision**: True Positive Rate among predicted positives
   ```
   Precision = TP / (TP + FP)
   ```

3. **Recall (Sensitivity)**: True Positive Rate among actual positives
   ```
   Recall = TP / (TP + FN)
   ```

4. **F1-Score**: Harmonic mean of precision and recall
   ```
   F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
   ```

5. **ROC AUC**: Area Under the Receiver Operating Characteristic curve
   ```
   AUC = ∫₀¹ TPR(FPR⁻¹(x))dx
   ```

#### 3.5.2 Secondary Metrics

1. **Training Time**: Computational efficiency during model fitting
2. **Prediction Time**: Inference speed on test set
3. **Memory Usage**: Peak memory consumption during training
4. **Confusion Matrix**: Detailed breakdown (TP, TN, FP, FN)

#### 3.5.3 Statistical Analysis

**Descriptive Statistics**:
- Mean and standard deviation across runs
- Confidence intervals (95% level)
- Min/max values per configuration

**Inferential Statistics**:
- ANOVA for comparing algorithm performance
- Post-hoc tests with Bonferroni correction
- Effect size calculation (Cohen's d)

### 3.6 Technical Infrastructure

#### 3.6.1 Hardware Specifications

**Computational Resources**:
- **CPU**: [To be specified based on execution environment]
- **RAM**: 32 GB (sufficient for full dataset processing)
- **Storage**: SSD (fast I/O for large datasets)
- **GPU**: [If applicable for neural network acceleration]

#### 3.6.2 Software Environment

**Operating System**: Linux Ubuntu 20.04+  
**Python Version**: 3.9+

**Key Dependencies**:
```
scikit-learn==1.3.0     # Machine learning algorithms
pandas==2.0.0           # Data manipulation
numpy==1.24.0           # Numerical computing
mlflow==2.5.0           # Experiment tracking
dvc==3.0.0              # Data version control
psutil==5.9.0           # System monitoring
seaborn==0.11.0         # Statistical visualization
matplotlib==3.7.0       # Plotting
```

#### 3.6.3 Reproducibility Framework

**Version Control**:
- **Code**: Git repository with detailed commit history
- **Data**: DVC for large file management
- **Experiments**: MLflow for run tracking
- **Environment**: requirements.txt with pinned versions

**Random State Control**:
```python
# Fixed seeds throughout pipeline
RANDOM_STATE = 42

# Applied to:
train_test_split(random_state=42)
all_algorithms(random_state=42)
numpy.random.seed(42)
pandas.sample(random_state=42)
```

#### 3.6.4 Monitoring and Logging System

**Comprehensive Logging**:
- **Execution ID**: Unique timestamp-based identifier
- **Log File**: `experiments/logs/algorithm_comparison_{timestamp}.log`
- **Format**: `timestamp - level - [function:line] - message`

**Real-time Monitoring**:
```python
def monitor_resources():
    return {
        'memory_rss_mb': process.memory_info().rss / 1024 / 1024,
        'memory_percent': process.memory_percent(),
        'available_mb': psutil.virtual_memory().available / 1024 / 1024,
        'cpu_percent': process.cpu_percent()
    }
```

**Memory Management**:
```python
def cleanup_memory():
    del model, predictions, probabilities
    gc.collect()  # Force garbage collection
    
    # Verify cleanup effectiveness
    memory_freed = memory_before - memory_after
    logger.info(f"Memory freed: {memory_freed:.1f} MB")
```

### 3.7 File Organization and Data Management

#### 3.7.1 Project Structure

```
iot-ids-research/
├── data/
│   ├── raw/CSV/MERGED_CSV/          # 63 original CSV files
│   ├── processed/
│   │   ├── sampled.csv              # 4.5M stratified sample
│   │   └── binary/                  # Preprocessed binary classification data
│   │       ├── X_train_binary.npy   # Training features (3.6M×39)
│   │       ├── X_test_binary.npy    # Test features (900K×39)
│   │       ├── y_train_binary.npy   # Training labels
│   │       ├── y_test_binary.npy    # Test labels
│   │       ├── scaler.pkl           # Fitted StandardScaler
│   │       └── binary_metadata.json # Preprocessing metadata
│   └── metrics/
│       └── quality_check.json       # Data quality metrics
├── experiments/
│   ├── algorithm_comparison.py      # Main experimental script
│   ├── results/                     # Experimental outputs
│   │   ├── experiment_results_full.csv      # All individual results
│   │   ├── aggregated_stats_full.csv        # Statistical summaries
│   │   ├── best_results_summary_full.csv    # Top performers
│   │   └── visualizations/                  # Performance plots
│   ├── logs/                        # Detailed execution logs
│   │   └── algorithm_comparison_{timestamp}.log
│   └── artifacts/                   # MLflow artifacts
├── configs/
│   ├── preprocessing.yaml           # Preprocessing parameters
│   └── experiment_config.yaml       # Experiment configuration
├── src/
│   └── eda/
│       ├── dvc_eda.py              # Exploratory data analysis
│       └── results/                 # EDA visualizations
├── dvc.yaml                         # DVC pipeline definition
├── dvc_sampling.py                  # Sampling implementation
├── dvc_preprocessing.py             # Preprocessing implementation
├── dvc_baseline_experiment.py       # DVC experiment wrapper
└── requirements.txt                 # Python dependencies
```

#### 3.7.2 Data Versioning Strategy

**DVC Pipeline Stages**:
1. **check_quality**: Validate raw data integrity
2. **sampling**: Generate stratified sample
3. **eda**: Exploratory data analysis
4. **preprocess**: Feature engineering and normalization
5. **baseline_experiment**: Algorithm comparison

**Artifact Tracking**:
- All intermediate datasets versioned with DVC
- Experiment results tracked with MLflow
- Code changes managed with Git
- Complete provenance from raw data to final results

---

## 4. Expected Results Framework

### 4.1 Performance Comparison

**Primary Analysis**:
- Ranking algorithms by F1-Score (primary metric)
- Statistical significance testing between algorithms
- Performance vs. computational cost trade-offs
- Hyperparameter sensitivity analysis

**Secondary Analysis**:
- Algorithm-specific insights (e.g., ensemble vs. linear methods)
- Error analysis through confusion matrices
- Resource consumption patterns
- Scalability considerations

### 4.2 Visualization Framework

**Performance Visualizations**:
1. **Box plots**: F1-score distribution by algorithm
2. **Bar charts**: Mean performance metrics comparison
3. **Scatter plots**: Performance vs. computational time
4. **Heatmaps**: Correlation between metrics
5. **ROC curves**: Threshold-independent performance

**Resource Analysis**:
1. **Memory usage patterns** over time
2. **Training time scaling** with dataset size
3. **Efficiency frontiers** (accuracy vs. speed)

### 4.3 Statistical Validation

**Hypothesis Testing**:
- H₀: No significant difference between algorithm performances
- H₁: Significant performance differences exist
- α = 0.05 significance level
- Multiple comparison correction (Bonferroni)

**Effect Size Analysis**:
- Cohen's d for practical significance
- Confidence intervals for performance metrics
- Bootstrap resampling for robust estimates

---

## 5. Contributions and Significance

### 5.1 Scientific Contributions

1. **Comprehensive Baseline**: First systematic comparison of ML algorithms for IoT anomaly detection using CICIoT2023
2. **Methodological Framework**: Standardized evaluation pipeline for IoT security research
3. **Statistical Rigor**: Multi-run experiments with proper statistical validation
4. **Reproducible Research**: Complete DVC pipeline for result replication

### 5.2 Practical Contributions

1. **Algorithm Guidance**: Evidence-based recommendations for IoT security practitioners
2. **Performance Benchmarks**: Quantitative baselines for future research
3. **Resource Planning**: Computational requirements analysis for deployment
4. **Implementation Framework**: Ready-to-use experimental infrastructure

### 5.3 Dataset Contributions

1. **Preprocessed CICIoT2023**: Clean, normalized version with documented preprocessing
2. **Sampling Methodology**: Validated stratified sampling approach for large IoT datasets
3. **Feature Engineering**: Optimized 39-feature set for binary anomaly detection
4. **Evaluation Standards**: Standardized metrics and procedures for IoT security research

---

## 6. Limitations and Scope

### 6.1 Dataset Limitations

**Sampling Constraints**:
- **Reduced Scale**: 19.5% sample due to computational limitations
- **Temporal Scope**: Limited time window may miss long-term patterns
- **Attack Diversity**: Constrained to CICIoT2023 attack types
- **Environmental Specificity**: Testbed data may not capture all real-world variations

**Methodological Limitations**:
- **Binary Classification**: Simplified problem vs. multi-class attack detection
- **Static Analysis**: No temporal/sequential pattern analysis
- **Parameter Space**: Limited hyperparameter exploration due to computational cost
- **Feature Engineering**: No advanced feature selection or creation techniques

### 6.2 Generalizability Considerations

**Context Dependencies**:
- **Dataset Specificity**: Results tied to CICIoT2023 characteristics
- **IoT Environment**: Findings may not generalize across all IoT ecosystems
- **Attack Evolution**: Performance may degrade with emerging attack patterns
- **Deployment Context**: Laboratory vs. production environment differences

### 6.3 Technical Limitations

**Computational Constraints**:
- **Hardware Dependency**: Results influenced by available computational resources
- **Memory Limitations**: Large algorithms may require different approaches at scale
- **Time Constraints**: Limited exploration of computationally expensive algorithms
- **Scalability**: Performance patterns may change with larger datasets

---

## 7. Future Research Directions

### 7.1 Immediate Extensions

**Methodological Enhancements**:
1. **Complete Dataset Analysis**: Evaluation on full 23M record dataset
2. **Multi-class Classification**: Specific attack type identification
3. **Advanced Preprocessing**: Feature selection, dimensionality reduction, ensemble features
4. **Deep Learning**: CNN, LSTM, and Transformer-based approaches

**Analytical Extensions**:
1. **Temporal Analysis**: Time-series and sequential pattern recognition
2. **Concept Drift Detection**: Adaptation to evolving attack patterns
3. **Ensemble Methods**: Combining best-performing individual algorithms
4. **Explainable AI**: Model interpretability for security analysts

### 7.2 Long-term Research Vision

**Advanced Methodologies**:
1. **Real-time Detection**: Streaming ML for online anomaly detection
2. **Transfer Learning**: Cross-dataset and cross-domain generalization
3. **Federated Learning**: Distributed IoT security without centralized data
4. **Adversarial Robustness**: Security against adversarial attacks

**Application Domains**:
1. **Industry-specific IoT**: Healthcare, manufacturing, smart city applications
2. **Edge Computing**: Resource-constrained deployment scenarios
3. **5G/6G Networks**: Next-generation network security
4. **Autonomous Systems**: Self-defending IoT ecosystems

---

## 8. Execution Instructions

### 8.1 Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd iot-ids-research

# Create virtual environment
python -m venv env
source env/bin/activate  # Linux/Mac
# or env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Setup DVC (if using data version control)
dvc init
```

### 8.2 Data Preparation

```bash
# Place CICIoT2023 dataset in data/raw/CSV/MERGED_CSV/
# Should contain 63 CSV files

# Verify data integrity
python dvc_run_quality_check.py

# Execute sampling
python dvc_sampling.py
```

### 8.3 Experiment Execution

**Option 1: Complete DVC Pipeline**
```bash
# Execute entire pipeline
dvc repro baseline_experiment

# Monitor progress
tail -f experiments/logs/algorithm_comparison_*.log
```

**Option 2: Direct Execution**
```bash
# Configure for full experiment
vim configs/experiment_config.yaml  # Set test_mode: false

# Run experiments directly
python experiments/algorithm_comparison.py
```

### 8.4 Results Analysis

```bash
# View results
ls experiments/results/

# Key output files:
# - experiment_results_full.csv: All individual results
# - aggregated_stats_full.csv: Statistical summaries  
# - best_results_summary_full.csv: Top performers
# - *.png: Performance visualizations

# Access MLflow UI
mlflow server --host 127.0.0.1 --port 5000
# Open http://127.0.0.1:5000 in browser
```

---

## 9. Expected Timeline and Resources

### 9.1 Computational Requirements

**Time Estimates** (full experiment):
- Data loading and preprocessing: ~5 minutes
- Random Forest experiments: ~45-90 minutes
- SVC experiments: ~2-4 hours  
- Neural network experiments: ~1-2 hours
- Other algorithms: ~30-60 minutes each
- **Total estimated time**: 6-10 hours

**Resource Requirements**:
- **Memory**: 4-6 GB peak usage
- **Storage**: ~10 GB for data and results
- **CPU**: Multi-core recommended for parallel operations

### 9.2 Execution Phases

**Phase 1: Setup and Validation** (30 minutes)
- Environment configuration
- Data integrity verification
- Pipeline testing with small sample

**Phase 2: Full Experiment Execution** (6-10 hours)
- Automated algorithm comparison
- Real-time monitoring and logging
- Result generation and visualization

**Phase 3: Analysis and Documentation** (2-4 hours)
- Result interpretation
- Statistical analysis
- Performance comparison
- Report generation

---

## 10. Quality Assurance

### 10.1 Validation Procedures

**Data Validation**:
- Statistical tests for sample representativeness
- Cross-validation of preprocessing steps
- Integrity checks throughout pipeline

**Experimental Validation**:
- Multiple independent runs for statistical significance
- Memory and resource monitoring
- Automated error detection and recovery

**Result Validation**:
- Sanity checks for metric ranges
- Consistency verification across runs
- Outlier detection in results

### 10.2 Reproducibility Checklist

- [ ] Fixed random seeds throughout pipeline
- [ ] Documented software versions
- [ ] Version-controlled code and configurations
- [ ] Automated pipeline execution
- [ ] Complete logging and monitoring
- [ ] Standardized evaluation metrics
- [ ] Detailed methodology documentation

---

## Conclusion

This methodology provides a comprehensive framework for comparing machine learning algorithms for IoT anomaly detection. The combination of the CICIoT2023 dataset, standardized preprocessing pipeline, rigorous experimental design, and statistical validation ensures reliable and reproducible results.

The DVC-based pipeline enables complete reproducibility while the comprehensive monitoring system provides insights into both performance and resource requirements. This work establishes a foundation for future research in IoT security and provides practical guidance for security practitioners.

The expected results will contribute to the growing body of knowledge in IoT security by providing evidence-based algorithm recommendations, performance benchmarks, and a reusable experimental framework for the research community.

*[Results section to be populated after experiment completion]*

---

## Appendices

### Appendix A: Complete Algorithm Configuration

```python
# Complete hyperparameter grid for all algorithms
ALGORITHM_CONFIGS = {
    'RandomForest': {
        'class': RandomForestClassifier,
        'params': [
            {'n_estimators': 50, 'max_depth': 10, 'random_state': 42},
            {'n_estimators': 100, 'max_depth': 15, 'random_state': 42},
            {'n_estimators': 200, 'max_depth': 20, 'random_state': 42}
        ]
    },
    # ... [Complete specifications for all 7 algorithms]
}
```

### Appendix B: Statistical Analysis Framework

```python
# Statistical significance testing
from scipy import stats

def compare_algorithms(results_df):
    # ANOVA for overall comparison
    f_stat, p_value = stats.f_oneway(*groups)
    
    # Post-hoc pairwise comparisons
    # Bonferroni correction for multiple comparisons
    # Effect size calculation
```

### Appendix C: Resource Monitoring Implementation

```python
# Memory and performance monitoring
def monitor_experiment():
    # Real-time resource tracking
    # Memory leak detection
    # Performance bottleneck identification
    # Automated cleanup procedures
```

---

**Document Version**: 1.0  
**Last Updated**: September 2025  
**Status**: Ready for Experiment Execution  
**Next Steps**: Execute experiments and populate results section

**Authors**: [To be specified]  
**Institution**: [To be specified]  
**Contact**: [To be specified]
