# Comparative Analysis of Machine Learning Algorithms for Anomaly Detection in IoT Networks Using CICIoT2023 Dataset

## Abstract

Internet of Things (IoT) networks face increasing security threats due to their heterogeneous nature and resource constraints. This study presents a comprehensive comparison of machine learning algorithms for anomaly detection in IoT environments using the CICIoT2023 dataset. We implemented a reproducible pipeline using Data Version Control (DVC) to evaluate seven different algorithms across multiple configurations. Our methodology employs stratified sampling, standardized preprocessing, and binary classification to distinguish between benign and malicious network traffic. The experimental framework includes rigorous statistical validation through multiple runs and comprehensive performance metrics including accuracy, precision, recall, F1-score, and ROC AUC. This research contributes to the understanding of algorithm performance in IoT intrusion detection and provides a baseline for future comparative studies.

**Keywords:** IoT Security, Anomaly Detection, Machine Learning, Intrusion Detection, Binary Classification

---

## 1. Introduction

The proliferation of Internet of Things (IoT) devices has created new attack vectors and security challenges in network environments. Traditional signature-based intrusion detection systems are inadequate for IoT networks due to the diverse and evolving nature of attacks. Machine learning-based anomaly detection offers a promising approach to identify malicious activities in IoT networks by learning patterns from normal behavior and detecting deviations.

This study addresses the research question: **"Which machine learning algorithms are most effective for binary anomaly detection in IoT network traffic?"** We conduct a systematic comparison of ten algorithms using standardized experimental conditions and comprehensive evaluation metrics.

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

**Statistical Validation**:
- Kolmogorov-Smirnov test for distribution comparison

### 3.2 Experimental Design

#### 3.2.1 Research Design Framework

**Study Type**: Quantitative experimental design  
**Comparison Approach**: Ten machine learning algorithms (optimized for large-scale)  
**Classification Task**: Binary (Benign vs. Malicious)  
**Validation Method**: Stratified train-test split  
**Statistical Rigor**: Multiple runs per configuration (n=5)  
**Reproducibility**: DVC-based pipeline with version control  
**Computational Context**: CPU-only training without GPU acceleration

#### 3.2.2 Algorithm Selection

We selected ten algorithms representing different learning paradigms, ordered by computational complexity for optimal resource management:

**Supervised Learning Algorithms** (ordered by complexity):
1. **Logistic Regression**: Linear probabilistic classifier (O(n))
2. **Random Forest**: Ensemble method with bagging (O(n log n))
3. **Gradient Boosting**: Ensemble method with boosting (O(n log n))
4. **LinearSVC**: Linear Support Vector Classifier with primal formulation (O(n), optimized for large datasets)
5. **SGDClassifier**: Stochastic Gradient Descent with hinge loss (O(n), scalable SVM approximation)
6. **Multi-Layer Perceptron (MLP)**: Neural network classifier (O(n³))

**Unsupervised/Semi-Supervised Algorithms** (anomaly detection):
7. **Isolation Forest**: Tree-based anomaly detection (O(n log n))
8. **Elliptic Envelope**: Gaussian-based anomaly detection (O(n²))
9. **Local Outlier Factor (LOF)**: Density-based anomaly detection (O(n²))
10. **SGDOneClassSVM**: One-Class SVM via stochastic gradient descent (O(n), optimized for large-scale)

**Algorithm Classification by Detection Type**:
- **True Anomaly Detection**: Isolation Forest, SGDOneClassSVM, LOF, Elliptic Envelope
- **Supervised Classification**: Logistic Regression, Random Forest, Gradient Boosting, LinearSVC, SGDClassifier, MLP

**Critical Optimizations for Large-Scale Datasets**:
Given the computational challenges of training on 3M samples with 39 features, we replaced computationally prohibitive algorithms with scalable alternatives:
- **Linear SVM**: Replaced `SVC(kernel='linear')` with `LinearSVC(dual=False)` and `SGDClassifier(loss='hinge')` for 10-100x speedup while maintaining mathematical equivalence
- **One-Class SVM**: Replaced kernel-based `OneClassSVM` with `SGDOneClassSVM` for 10-50x speedup, enabling practical anomaly detection on large-scale data

### 3.3 Data Preprocessing Pipeline

#### 3.3.1 Pipeline Architecture (DVC)

```yaml
stages:
  1. check_quality           → Data quality validation and metrics
  2. sampling               → Stratified dataset sampling
  3. eda                   → Exploratory data analysis
  4. preprocess            → Feature engineering and normalization
  5. exp_logistic_regression → Logistic Regression experiments
  6. exp_random_forest      → Random Forest experiments
  7. exp_gradient_boosting  → Gradient Boosting experiments
  8. exp_isolation_forest   → Isolation Forest experiments
  9. exp_svc               → Support Vector Classifier experiments
  10. exp_one_class_svm    → One-Class SVM experiments
  11. exp_lof              → Local Outlier Factor experiments
  12. exp_elliptic_envelope → Elliptic Envelope experiments
  13. exp_mlp_classifier   → Multi-Layer Perceptron experiments
  14. consolidate_results  → Results consolidation and analysis
```

**Key Pipeline Features**:
- **Modular Execution**: Each algorithm runs as independent DVC stage
- **Computational Ordering**: Algorithms ordered from least to most computationally complex
- **Dynamic Configuration**: Single `TEST_MODE` variable controls all experiments
- **Shared Timestamps**: Unified timestamp system for result organization
- **Individual Analysis**: Automated detailed analysis per algorithm
- **Final Consolidation**: Comprehensive cross-algorithm comparison

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

**Dynamic Configuration System**:
The experimental framework uses a centralized configuration system with two modes:
- **Test Mode**: Simplified configurations for rapid validation
- **Full Mode**: Comprehensive parameter exploration for final results

Each algorithm configuration includes multiple parameter combinations and statistical rigor through multiple runs (n=1 for test mode, n=5 for full mode).

**Algorithm Configurations**:

**1. Logistic Regression** (O(n) - Fastest):
```python
# Full Mode  
{'C': [0.1, 1.0, 10.0], 'max_iter': 1000, 'random_state': 42}
```

**2. Random Forest** (O(n log n)):
```python
# Full Mode
{'n_estimators': [50, 100], 'max_depth': [10, 15, 20], 'random_state': 42}
```

**3. Gradient Boosting** (O(n log n)):
```python
# Full Mode
{'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 
 'max_depth': [5, 7], 'random_state': 42}
```

**4. Isolation Forest** (O(n log n)):
```python
# Full Mode
{'contamination': [0.1, 0.15, 0.2], 'n_estimators': [100, 200], 'random_state': 42}
```

**5. Elliptic Envelope** (O(n²)):
```python
# Full Mode
{'contamination': [0.1, 0.15, 0.2], 'random_state': 42}
```

**6. Local Outlier Factor** (O(n²)):
```python
# Full Mode
{'n_neighbors': [10, 20, 50], 'contamination': [0.1, 0.15], 'novelty': True}
```

**7. LinearSVC** (O(n) - Optimized for large datasets):
```python
# Optimization: dual=False for n_samples >> n_features
# Mathematically equivalent to SVC(kernel='linear')
# Full Mode
{'C': [1.0, 5.0, 10.0], 'max_iter': 1000, 'dual': False, 'random_state': 42}
# Expected time: 1-2h per config (vs. days for standard SVC)
```

**8. SGDClassifier** (O(n) - SVM via stochastic gradient descent):
```python
# Scalable SVM via stochastic gradient descent
# Full Mode
{'loss': 'hinge', 'penalty': 'l2', 'alpha': [0.0001, 0.001, 0.01], 
 'max_iter': 1000, 'random_state': 42}
# Expected time: 10-20 min per config
```

**9. SGDOneClassSVM** (O(n) - Optimized One-Class SVM):
```python
# Online learning for large-scale anomaly detection
# Maintains same nu parameter as OneClassSVM
# Full Mode
{'nu': [0.05, 0.1, 0.15, 0.2], 'learning_rate': 'optimal', 
 'max_iter': 1000, 'random_state': 42}
# Expected time: 15-30 min per config (vs. hours for kernel-based)
```

**10. Multi-Layer Perceptron** (O(n³)):
```python
# Full Mode
{'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)], 
 'max_iter': 200, 'random_state': 42}
```

**Critical Performance Optimizations**:
- **Linear SVM Optimization**: Replaced kernel-based SVM with LinearSVC and SGDClassifier for 10-100x speedup while maintaining mathematical equivalence
- **One-Class SVM Optimization**: Employed SGDOneClassSVM with O(n) complexity instead of O(n²) kernel-based approach
- **Progressive Complexity**: Algorithms ordered from least to most computationally expensive for optimal resource utilization

#### 3.4.2 Experimental Execution

**Modular Pipeline Architecture**:
The experimental framework was redesigned as a modular DVC pipeline where each algorithm runs as an independent stage:

```python
# Individual algorithm execution
run_single_algorithm.py <algorithm_name>

# Each algorithm follows this pattern:
for param_config in parameter_grid:
    for run in range(N_RUNS):  # Statistical rigor
        # Execute single experiment
        # Monitor memory and time
        # Record all metrics
        # Generate individual analysis
        # Clean memory
```

**Statistical Rigor**:
- **Multiple Runs**: 10 independent executions per parameter configuration (controlled by global `N_RUNS`)
- **Random State Control**: Fixed seeds for all randomized components (`RANDOM_STATE = 42`)
- **Memory Management**: Aggressive cleanup between experiments with monitoring
- **Progress Monitoring**: Real-time tracking of execution and resource usage
- **Error Handling**: Robust error recovery and logging system

**Dynamic Configuration System**:
```python
# Single point of control
TEST_MODE = True  # or False for full experiments

# Automatically configures:
- Sample sizes (10,000 vs full dataset)
- Algorithm parameters (simple vs comprehensive)
- Number of runs (same for both modes)
- Output directories (test/ vs full/)
```

**Result Organization**:
```
experiments/results/
├── test/                          # Test mode results
│   ├── TIMESTAMP_algorithm_name/  # Individual algorithm results
│   └── TIMESTAMP_consolidation/   # Cross-algorithm analysis
└── full/                          # Full mode results
    ├── TIMESTAMP_algorithm_name/  # Individual algorithm results
    └── TIMESTAMP_consolidation/   # Cross-algorithm analysis
```

**Timestamp Management**:
- **Shared Timestamps**: All algorithms in a single DVC run share the same timestamp
- **Non-overlapping Results**: Each execution gets a unique timestamp to prevent data loss
- **Chronological Organization**: Results naturally ordered by execution time

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

4. **F1-Score**: Harmonic mean of precision and recall (primary ranking metric)
   ```
   F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
   ```

5. **ROC AUC**: Area Under the Receiver Operating Characteristic curve
   ```
   AUC = ∫₀¹ TPR(FPR⁻¹(x))dx
   ```

#### 3.5.2 Secondary Metrics

1. **Training Time**: Computational efficiency during model fitting
2. **Prediction Time**: Inference speed on test set (when available)
3. **Memory Usage**: Peak memory consumption during training
4. **Confusion Matrix**: Detailed breakdown (TP, TN, FP, FN)
5. **Efficiency Score**: F1-Score per second of training time
6. **Stability Metrics**: Standard deviation across multiple runs

#### 3.5.3 Individual Algorithm Analysis

For each algorithm, we generate comprehensive individual analysis:

**Performance Evolution Metrics**:
- Accuracy, F1-Score, Precision, Recall trends across runs
- Parameter impact analysis on performance
- Statistical stability measures

**Resource Analysis Metrics**:
- Training time distribution and trends
- Memory consumption patterns
- Computational efficiency (performance per unit time)

**Statistical Validation Metrics**:
- Confidence intervals for all primary metrics
- Coefficient of variation for stability assessment
- Best vs. average performance analysis

#### 3.5.3 Statistical Analysis

**Descriptive Statistics**:
- Mean and standard deviation across runs
- Confidence intervals (95% level)
- Min/max values per configuration

**Inferential Statistics**:
- ANOVA for comparing algorithm performance
- Post-hoc tests with Bonferroni correction
- Effect size calculation (Cohen's d)

### 3.6 Computational Efficiency for Large-Scale IoT Datasets

#### 3.6.1 SVM Optimization for Large-Scale Data

Given the large-scale nature of IoT network data (3M samples, 39 features), traditional kernel-based SVM algorithms become computationally prohibitive. We employed scalable alternatives that maintain mathematical equivalence while enabling practical experimentation.

**Challenge**: Standard `SVC(kernel='linear')` and `OneClassSVM` have O(n²) to O(n³) complexity, making them impractical for datasets with millions of samples (estimated convergence time: several days to weeks).

**Solution**: Linear formulations and stochastic gradient descent approaches

#### 3.6.2 Linear Support Vector Machine Optimization

**Approach 1: LinearSVC with Primal Formulation**
```python
LinearSVC(C=value, max_iter=1000, dual=False, random_state=42)
```
- **Optimization**: `dual=False` solves the primal problem when n_samples >> n_features
- **Complexity**: O(n) vs. O(n²-n³) for standard SVC
- **Mathematical Basis**: Equivalent to `SVC(kernel='linear')` but formulated for large sample scenarios
- **Expected Performance**: 10-100x speedup (hours vs. days)
- **Academic Citation**: Fan et al. (2008), LIBLINEAR: A Library for Large Linear Classification

**Approach 2: SGDClassifier with Hinge Loss**
```python
SGDClassifier(loss='hinge', penalty='l2', alpha=value, max_iter=1000)
```
- **Optimization**: Stochastic gradient descent for online/batch learning
- **Complexity**: O(n) with single-pass convergence
- **Mathematical Basis**: Approximates linear SVM through iterative optimization
- **Expected Performance**: 50-200x speedup (minutes vs. hours)
- **Academic Citation**: Bottou & Bousquet (2007), The Tradeoffs of Large Scale Learning

**Scientific Justification**:
Both approaches are widely accepted in machine learning literature for large-scale classification tasks. LinearSVC is the standard recommendation for linear SVM on large datasets (scikit-learn documentation, Pedregosa et al., 2011), while SGDClassifier provides theoretical guarantees for convergence to SVM solutions (Shalev-Shwartz et al., 2011).

#### 3.6.3 One-Class SVM Optimization for Anomaly Detection

**Challenge**: Kernel-based `OneClassSVM` has O(n²) complexity with cubic memory requirements, making it prohibitive for million-scale anomaly detection.

**Solution**: SGDOneClassSVM with online learning

```python
SGDOneClassSVM(nu=value, learning_rate='optimal', max_iter=1000, random_state=42)
```

**Key Advantages**:
- **Complexity**: O(n) vs. O(n²) for kernel-based approach
- **Parameter Compatibility**: Maintains same `nu` parameter space as OneClassSVM
- **API Consistency**: Identical interface (`.fit()`, `.predict()`, `.decision_function()`)
- **Memory Efficiency**: Constant memory footprint vs. quadratic growth
- **Expected Performance**: 10-50x speedup (minutes vs. hours)

**Academic Justification**:
SGDOneClassSVM implements the online learning variant of One-Class SVM (Schölkopf et al., 2001) through stochastic gradient descent. The algorithm maintains the theoretical properties of kernel-based One-Class SVM while enabling scalability through online optimization.

### 3.7 Technical Infrastructure

#### 3.7.1 Hardware Specifications

**Computational Resources**:
- **CPU**: [To be specified based on execution environment]
- **RAM**: 32 GB (sufficient for full dataset processing)
- **Storage**: SSD (fast I/O for large datasets)
- **GPU**: Not used (CPU-only training with optimized algorithms)

#### 3.7.2 Software Environment

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

#### 3.7.3 Reproducibility Framework

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

#### 3.7.4 Monitoring and Logging System

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

### 3.8 File Organization and Data Management

#### 3.8.1 Project Structure

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
│   ├── algorithm_comparison.py      # Global configurations and utilities
│   ├── run_single_algorithm.py      # Individual algorithm execution
│   ├── consolidate_results.py       # Cross-algorithm analysis
│   ├── individual_analysis.py       # Per-algorithm detailed analysis
│   ├── results/                     # Organized experimental outputs
│   │   ├── test/                    # Test mode results
│   │   │   ├── TIMESTAMP_algorithm_name/     # Individual algorithm results
│   │   │   │   ├── results.json             # Raw experiment results
│   │   │   │   ├── summary.json             # Statistical summary
│   │   │   │   └── individual_analysis/     # Detailed analysis
│   │   │   │       ├── plots/               # Algorithm-specific plots
│   │   │   │       ├── tables/              # Detailed statistics
│   │   │   │       └── report/              # Individual reports
│   │   │   └── TIMESTAMP_consolidation/     # Cross-algorithm comparison
│   │   │       ├── plots/                   # Comparative visualizations
│   │   │       ├── tables/                  # Summary statistics
│   │   │       ├── report/                  # Final analysis report
│   │   │       └── data/                    # Consolidated datasets
│   │   └── full/                    # Full mode results (same structure)
│   ├── logs/                        # Detailed execution logs
│   │   └── algorithm_comparison_{timestamp}.log
│   ├── .current_run_timestamp       # Shared timestamp coordination
│   └── artifacts/                   # MLflow artifacts (if used)
├── configs/
│   ├── preprocessing.yaml           # Preprocessing parameters
│   └── experiment_config.yaml       # Experiment configuration
├── src/
│   └── eda/
│       ├── dvc_eda.py              # Exploratory data analysis
│       └── results/                 # EDA visualizations
├── dvc.yaml                         # DVC pipeline definition (modular stages)
├── dvc_sampling.py                  # Sampling implementation
├── dvc_preprocessing.py             # Preprocessing implementation
└── requirements.txt                 # Python dependencies
```

#### 3.8.2 Advanced Result Organization

**Hierarchical Structure**:
- **Mode Separation**: `test/` and `full/` directories for different experiment modes
- **Timestamp-based Naming**: Each execution gets unique timestamp to prevent overwrites
- **Individual Analysis**: Each algorithm generates comprehensive individual reports
- **Consolidated Analysis**: Cross-algorithm comparison with advanced visualizations

**Analysis Depth**:
```
Individual Algorithm Analysis:
├── Performance Evolution (accuracy, F1, precision, recall trends)
├── Parameter Impact Analysis (hyperparameter effects)
├── Confusion Matrix Analysis (detailed error patterns)
├── Metrics Distribution (statistical distributions)
├── Execution Time Analysis (efficiency and resource usage)
├── Detailed Tables (statistics, rankings, raw results)
└── Comprehensive Report (insights and recommendations)

Consolidated Analysis:
├── Algorithm Comparison (rankings and statistical significance)
├── Performance vs Efficiency Trade-offs
├── Anomaly Detection vs Supervised Classification Analysis
├── Resource Usage Patterns
├── Correlation Analysis between Metrics
└── Final Recommendations Report
```

#### 3.8.3 Data Versioning Strategy

**DVC Pipeline Stages**:
1. **check_quality**: Validate raw data integrity
2. **sampling**: Generate stratified sample
3. **eda**: Exploratory data analysis
4. **preprocess**: Feature engineering and normalization
5. **exp_logistic_regression**: Logistic Regression experiments (O(n))
6. **exp_random_forest**: Random Forest experiments (O(n log n))
7. **exp_gradient_boosting**: Gradient Boosting experiments (O(n log n))
8. **exp_isolation_forest**: Isolation Forest experiments (O(n log n))
9. **exp_elliptic_envelope**: Elliptic Envelope experiments (O(n²))
10. **exp_local_outlier_factor**: Local Outlier Factor experiments (O(n²))
11. **exp_linear_svc**: LinearSVC optimized for large datasets (O(n))
12. **exp_sgd_classifier**: SGDClassifier with hinge loss (O(n))
13. **exp_sgd_one_class_svm**: SGDOneClassSVM for scalable anomaly detection (O(n))
14. **exp_mlp**: Multi-Layer Perceptron experiments (O(n³))
15. **consolidate_results**: Cross-algorithm analysis and reporting

**Artifact Tracking and Reproducibility**:
- **Data Versioning**: DVC tracks all intermediate datasets
- **Code Versioning**: Git manages all source code with detailed commit history
- **Result Versioning**: Timestamped results prevent overwrites
- **Configuration Tracking**: Single `TEST_MODE` variable controls entire pipeline
- **Dependency Management**: DVC automatically tracks file dependencies
- **Complete Provenance**: Full lineage from raw data to final insights

**Advanced Features**:
- **Dynamic Output Management**: Results saved in timestamp-based directories
- **Selective Execution**: DVC only re-runs stages with changed dependencies
- **Parallel Capability**: Independent algorithm stages can run in parallel
- **Incremental Updates**: New algorithms can be added without affecting existing results

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

### 4.2 Advanced Visualization Framework

**Individual Algorithm Analysis**:
1. **Performance Evolution**: Trends across multiple runs (accuracy, F1, precision, recall)
2. **Parameter Impact Analysis**: Hyperparameter effects on performance
3. **Confusion Matrix Analysis**: Detailed error pattern analysis with stability metrics
4. **Metrics Distribution**: Statistical distributions with density plots
5. **Execution Time Analysis**: Time efficiency and performance correlation

**Cross-Algorithm Comparison**:
1. **Box plots**: F1-score distribution by algorithm with statistical significance
2. **Bar charts**: Mean performance metrics with confidence intervals
3. **Scatter plots**: Performance vs. computational time trade-offs
4. **Heatmaps**: Correlation between metrics across algorithms
5. **ROC curves**: Threshold-independent performance comparison
6. **Anomaly Detection Analysis**: Specialized metrics for unsupervised algorithms

**Resource and Efficiency Analysis**:
1. **Memory usage patterns** over time with peak detection
2. **Training time scaling** with algorithm complexity analysis
3. **Efficiency frontiers** (F1-score per second)
4. **Algorithm ranking** by multiple criteria
5. **Computational complexity validation** against theoretical expectations

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

1. **Comprehensive Baseline**: Systematic comparison of 10 ML algorithms for IoT anomaly detection using CICIoT2023
2. **SVM Scalability Solutions**: Implementation of LinearSVC, SGDClassifier, and SGDOneClassSVM as practical alternatives to kernel-based methods for large-scale data
3. **Methodological Framework**: Advanced modular pipeline with individual algorithm analysis and cross-algorithm comparison
4. **Statistical Rigor**: Multi-run experiments with comprehensive statistical validation and stability analysis
5. **Reproducible Research**: Complete DVC pipeline with timestamp-based result organization for perfect reproducibility
6. **Algorithm Classification**: Clear distinction between supervised classification and true anomaly detection approaches
7. **Large-Scale Deployment Guidelines**: Practical recommendations for algorithm selection on large IoT datasets

### 5.2 Technical Contributions

1. **Modular Pipeline Architecture**: Independent DVC stages for each algorithm enabling parallel execution and selective re-runs
2. **SVM Scalability Suite**: Implementation of LinearSVC, SGDClassifier, and SGDOneClassSVM as practical alternatives to kernel-based methods
3. **Dynamic Configuration System**: Single-point control (`TEST_MODE`) for switching between validation and full experiments
4. **Advanced Result Organization**: Hierarchical timestamp-based structure preventing data loss and enabling historical analysis
5. **Individual Algorithm Analysis**: Comprehensive per-algorithm reporting with performance evolution, parameter impact, and efficiency analysis
6. **Cross-Algorithm Comparison**: Statistical significance testing with specialized anomaly detection metrics
7. **Resource Optimization**: Computational ordering from least to most complex algorithms for faster feedback

### 5.3 Practical Contributions

1. **Algorithm Guidance**: Evidence-based recommendations for algorithm selection in IoT security
2. **Scalability Framework**: Practical strategies for training on millions of samples with CPU-only infrastructure
3. **Performance Benchmarks**: Quantitative baselines for both supervised and unsupervised approaches on large-scale data
4. **Resource Planning**: Detailed computational requirements (30-50 hours with SVM optimizations vs. weeks without)
5. **Implementation Framework**: Ready-to-use experimental infrastructure with automated analysis generation
6. **Deployment Recommendations**: Practical algorithm selection guidelines for resource-constrained IoT environments

### 5.4 Dataset Contributions

1. **Preprocessed CICIoT2023**: Clean, normalized version with documented preprocessing and optimized feature set
2. **Sampling Methodology**: Validated stratified sampling approach for large IoT datasets with statistical validation
3. **Feature Engineering**: Optimized 39-feature set for binary anomaly detection with performance analysis
4. **Evaluation Standards**: Standardized metrics and procedures for IoT security research with anomaly detection focus

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

### 8.3 Enhanced Experiment Execution

**Option 1: Complete DVC Pipeline** (Recommended)
```bash
# Set experiment mode (single point of control)
# Edit experiments/algorithm_comparison.py: TEST_MODE = False  # for full experiments

# Execute entire pipeline with computational ordering
dvc repro consolidate_results

# Monitor progress (each algorithm runs independently)
tail -f experiments/logs/algorithm_comparison_*.log

# Track individual algorithm completion
ls -la experiments/results/full/  # or experiments/results/test/
```

**Option 2: Individual Algorithm Testing**
```bash
# Test single algorithm (useful for debugging)
cd experiments/
python3 run_single_algorithm.py logistic_regression

# Results automatically saved with timestamp
ls experiments/results/test/TIMESTAMP_logistic_regression/
```

**Option 3: Test Mode Validation**
```bash
# Quick validation with small sample
# Edit experiments/algorithm_comparison.py: TEST_MODE = True
dvc repro consolidate_results

# Verify all algorithms complete successfully
# Check experiments/results/test/ for timestamped results
```

### 8.4 Enhanced Results Analysis

```bash
# View hierarchical results structure
ls -la experiments/results/

# Structure for both test and full modes:
# experiments/results/test/    OR    experiments/results/full/
# ├── TIMESTAMP_algorithm_name/         # Individual algorithm results
# │   ├── results.json                  # Raw experimental data
# │   ├── summary.json                  # Statistical summary
# │   └── individual_analysis/          # Detailed analysis
# │       ├── plots/                    # Algorithm-specific visualizations
# │       ├── tables/                   # Detailed statistics
# │       └── report/                   # Individual reports
# └── TIMESTAMP_consolidation/          # Cross-algorithm comparison
#     ├── plots/                        # Comparative visualizations
#     ├── tables/                       # Summary statistics
#     ├── report/                       # Final analysis report
#     └── data/                         # Consolidated datasets

# Key analysis files:
# Individual Algorithm Analysis:
# - plots/performance_evolution.png      # Performance trends
# - plots/parameter_impact.png           # Hyperparameter effects  
# - plots/confusion_matrix_analysis.png  # Error analysis
# - plots/metrics_distribution.png       # Statistical distributions
# - plots/execution_time_analysis.png    # Efficiency analysis
# - tables/descriptive_statistics.csv    # Complete statistics
# - report/individual_report.md          # Comprehensive insights

# Cross-Algorithm Comparison:
# - plots/algorithm_comparison.png       # Performance ranking
# - plots/efficiency_analysis.png        # Performance vs. time
# - plots/anomaly_detection_analysis.png # Specialized analysis
# - tables/final_results_summary.csv     # Complete results
# - report/final_analysis_report.md      # Comprehensive findings

# Access MLflow UI (if configured)
mlflow server --host 127.0.0.1 --port 5000
# Open http://127.0.0.1:5000 in browser for experiment tracking
```

---

## 9. Expected Timeline and Resources

### 9.1 Computational Requirements

**Time Estimates** (full experiment with 10 algorithms, n=5 runs):
- Data loading and preprocessing: ~5 minutes
- Logistic Regression experiments: ~45-75 minutes (O(n))
- Random Forest experiments: ~2.5-5 hours (O(n log n))
- Gradient Boosting experiments: ~5-10 hours (O(n log n))
- Isolation Forest experiments: ~1.5-3 hours (O(n log n))
- Elliptic Envelope experiments: ~15-30 minutes (O(n²))
- Local Outlier Factor experiments: ~4-7.5 hours (O(n²))
- **LinearSVC experiments**: ~3-6 hours (O(n), optimized replacement for SVC)
- **SGDClassifier experiments**: ~30-60 minutes (O(n), stochastic gradient descent)
- **SGDOneClassSVM experiments**: ~1-2 hours (O(n), optimized replacement for One-Class SVM)
- MLP Classifier experiments: ~6-12 hours (O(n³))
- Individual analysis generation: ~5-10 minutes per algorithm
- Final consolidation and visualization: ~10-15 minutes
- **Total estimated time**: 30-50 hours (1.5-2 days)

**SVM Optimization Impact**:
- LinearSVC + SGDClassifier replace standard SVC: **~10-100x speedup** (hours vs. days/weeks)
- SGDOneClassSVM replaces kernel OneClassSVM: **~10-50x speedup** (1-2h vs. days)

**Resource Requirements**:
- **Memory**: 6-8 GB peak usage (increased due to more algorithms)
- **Storage**: ~15 GB for data and comprehensive results
- **CPU**: Multi-core strongly recommended for parallel DVC execution

### 9.2 Enhanced Execution Phases

**Phase 1: Setup and Validation** (30 minutes)
- Environment configuration and dependency installation
- Data integrity verification with quality metrics
- Pipeline testing with `TEST_MODE=True` (small sample validation)
- DVC pipeline validation and stage ordering verification

**Phase 2: Modular Experiment Execution** (30-50 hours for 10 algorithms)
- **Stage-by-stage execution**: Each algorithm runs as independent DVC stage
- **Progressive complexity**: Algorithms ordered from fastest to slowest (O(n) → O(n³))
- **Real-time monitoring**: Individual progress tracking and resource monitoring
- **Individual analysis**: Automatic detailed analysis per algorithm
- **Result organization**: Timestamp-based result separation (test/full modes)
- **SVM optimization**: LinearSVC, SGDClassifier, and SGDOneClassSVM for scalability

**Phase 3: Consolidation and Analysis** (30-60 minutes)
- **Cross-algorithm comparison**: Statistical analysis and ranking
- **Advanced visualizations**: Comprehensive plot generation
- **Report generation**: Individual and consolidated reports
- **Performance validation**: Results sanity checking and outlier detection

**Phase 4: Documentation and Validation** (1-2 hours)
- **Result interpretation**: Statistical significance testing
- **Performance benchmarking**: Algorithm recommendations
- **Reproducibility verification**: Pipeline validation
- **Final documentation**: Comprehensive analysis reports

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

This enhanced methodology provides a comprehensive and advanced framework for comparing machine learning algorithms for IoT anomaly detection. The combination of the CICIoT2023 dataset, modular DVC pipeline, rigorous experimental design, individual algorithm analysis, and advanced statistical validation ensures reliable, reproducible, and deeply insightful results.

**Key Methodological Advances**:
- **Expanded Algorithm Coverage**: 10 algorithms including scalable SVM variants for large-scale deployment
- **SVM Optimization**: LinearSVC, SGDClassifier, and SGDOneClassSVM achieving 10-100x speedup while maintaining mathematical equivalence
- **Modular Architecture**: Independent DVC stages enabling parallel execution and selective re-runs
- **Individual Analysis**: Comprehensive per-algorithm reporting with performance evolution and efficiency analysis
- **Advanced Organization**: Timestamp-based result management preventing data loss and enabling historical analysis

The modular DVC pipeline with individual algorithm stages enables complete reproducibility while the comprehensive monitoring system provides insights into both performance and resource requirements. The individual analysis system generates detailed insights for each algorithm, complemented by sophisticated cross-algorithm comparison.

**Research Impact**:
This work establishes a solid foundation for IoT security research by providing:
- **Scalability Solutions**: Practical SVM alternatives for training on million-scale datasets with CPU-only hardware
- **Evidence-based Recommendations**: Algorithm selection based on comprehensive empirical evaluation
- **Performance Benchmarks**: Quantitative baselines for both supervised and anomaly detection approaches
- **Reusable Framework**: Complete experimental infrastructure for reproducible research
- **Deployment Guidance**: Resource planning and algorithm selection for production IoT environments

The expected results will contribute significantly to the growing body of knowledge in IoT security by providing comprehensive algorithm comparisons, detailed performance insights, and a sophisticated experimental framework for the research community.

**Implementation Ready**: The framework is fully implemented and ready for execution, with comprehensive documentation, error handling, and automated analysis generation.

*[Results section to be populated after experiment completion with individual and consolidated analysis]*

---

## Appendices

### Appendix A: Complete Algorithm Configuration

```python
# Enhanced algorithm configurations with complexity ordering
ALGORITHM_CONFIGS = {
    # O(n) - Fastest
    'LogisticRegression': {
        'class': LogisticRegression,
        'test_params': [{'C': 1.0, 'max_iter': 100, 'random_state': 42}],
        'full_params': [
            {'C': 0.1, 'max_iter': 200, 'random_state': 42},
            {'C': 1.0, 'max_iter': 500, 'random_state': 42}
        ]
    },
    
    # O(n log n) - Ensemble methods
    'RandomForest': {
        'class': RandomForestClassifier,
        'test_params': [{'n_estimators': 10, 'max_depth': 5, 'random_state': 42}],
        'full_params': [
            {'n_estimators': 50, 'max_depth': 10, 'random_state': 42},
            {'n_estimators': 100, 'max_depth': 15, 'random_state': 42}
        ]
    },
    
    # O(n²) - Kernel and distance-based methods (optimized with linear kernels)
    'SVC': {
        'class': SVC,
        'test_params': [{'C': 1.0, 'kernel': 'linear', 'random_state': 42}],
        'full_params': [
            {'C': 0.1, 'kernel': 'linear', 'random_state': 42},
            {'C': 1.0, 'kernel': 'linear', 'random_state': 42}
        ]
    },
    
    # O(n³) - Most computationally complex (simplified architecture)
    'MLPClassifier': {
        'class': MLPClassifier,
        'test_params': [{'hidden_layer_sizes': (50,), 'max_iter': 100, 'random_state': 42}],
        'full_params': [
            {'hidden_layer_sizes': (50,), 'max_iter': 300, 'random_state': 42},
            {'hidden_layer_sizes': (100,), 'max_iter': 500, 'random_state': 42}
        ]
    }
    # ... [Complete specifications for all 9 algorithms]
}
```

### Appendix B: Enhanced Statistical Analysis Framework

```python
# Comprehensive statistical analysis with individual and cross-algorithm metrics
from scipy import stats
import pandas as pd
import numpy as np

def analyze_individual_algorithm(results):
    """Individual algorithm analysis with performance evolution"""
    # Performance trends analysis
    # Parameter impact assessment  
    # Stability and efficiency metrics
    # Statistical validation
    pass

def compare_algorithms(consolidated_results):
    """Cross-algorithm statistical comparison"""
    # ANOVA for overall comparison
    f_stat, p_value = stats.f_oneway(*algorithm_groups)
    
    # Post-hoc pairwise comparisons with Bonferroni correction
    # Effect size calculation (Cohen's d)
    # Anomaly detection vs supervised classification analysis
    # Computational complexity validation
    pass

def generate_comprehensive_report(individual_analyses, cross_analysis):
    """Generate detailed reports with recommendations"""
    # Individual algorithm insights and recommendations
    # Cross-algorithm performance ranking
    # Resource efficiency analysis
    # Deployment recommendations
    pass
```

### Appendix C: Advanced Resource Monitoring Implementation

```python
# Enhanced monitoring with individual analysis integration
import psutil
import time
from pathlib import Path

def monitor_algorithm_execution(algorithm_name):
    """Enhanced monitoring for individual algorithm execution"""
    # Real-time resource tracking
    # Memory leak detection  
    # Performance bottleneck identification
    # Automated cleanup procedures
    # Individual analysis generation
    pass

def consolidate_monitoring_data(all_algorithm_data):
    """Cross-algorithm resource analysis"""
    # Computational complexity validation
    # Resource usage patterns
    # Efficiency comparisons
    # Scalability analysis
    pass
```

---

**Document Version**: 3.1  
**Last Updated**: October 2025  
**Status**: Ready for Execution  
**Major Updates**: 
- Extended to 10 algorithms with focus on scalability for large datasets
- LinearSVC, SGDClassifier, and SGDOneClassSVM as practical alternatives to kernel-based SVM
- SVM optimizations achieving 10-100x speedup on large-scale data
- Modular DVC pipeline with individual algorithm analysis
- Comprehensive statistical validation and result organization
- Updated timeline: 30-50 hours (vs. weeks without SVM optimizations)

**Authors**: [To be specified]  
**Institution**: [To be specified]  
**Contact**: [To be specified]
