# Artigo1 - Plano Detalhado de Revisão e Correção

**Data:** 2025-11-09
**Paper:** Baseline Comparison Paper (model2011.tex)
**Review Score:** 38/50 (Major Revision)
**Objetivo:** Validação meticulosa de código e literatura + correções prioritizadas

---

## Sumário Executivo

Este plano integra:
1. **Academic Review** (5 passes, Dr. Reviewer): 3 críticos, 5 major, múltiplos minor issues
2. **User Verification Items**: 23+ itens de verificação de código e literatura
3. **Systematic Approach**: 6 fases executáveis com checklists

**Tempo Estimado:** 7-10 dias
**Potencial Re-run:** A ser decidido após Phase 1-3

---

## 🔴 PHASE 1: CRITICAL CODE VERIFICATION (1-2 days)

**Objetivo:** Verificar todas as afirmações numéricas e implementações no código contra o que está escrito no paper.

### 1.1 Dataset Size & Sampling Verification

**Paper Claims to Verify:**
- Line 118: "4,501,906 records (19.5% of original dataset)"
- Line 91-92: "23 million network traffic records"
- Line 119: "maintaining original class distribution (stratified sampling)"

**Code Checks:**

- [ ] **Check 1.1.1:** Find original dataset size
  ```bash
  # Verificar arquivo original CICIoT2023
  cd /Users/augusto/mestrado/final-project/iot-ids-research
  # Verificar se existe data/raw/
  ls -lh data/raw/
  # Ou verificar nos scripts de sampling
  grep -r "dataset.*size\|total.*records" experiments/ configs/
  ```
  **Expected:** Confirm if original is truly 23 million records
  **Document:** Actual size, source file name, date acquired

- [ ] **Check 1.1.2:** Verify sampling percentage calculation
  ```bash
  # Check sampling script
  cat experiments/dvc_sampling.py | grep -A 20 "def sample"
  # Look for sampling ratio
  grep -r "0.195\|19.5\|frac\|sample_size" experiments/dvc_sampling.py
  ```
  **Calculate:** If sampled = 4,501,906, then original = 4,501,906 / 0.195 = 23,087,210 (not 23M)
  **Or:** If original = 23,000,000, then 19.5% = 4,485,000 (not 4,501,906)
  **Document:** Which number is correct? Update paper accordingly

- [ ] **Check 1.1.3:** Verify stratified sampling implementation
  ```bash
  # Check if sklearn.model_selection.train_test_split with stratify used
  grep -A 10 "train_test_split\|StratifiedShuffleSplit" experiments/dvc_sampling.py
  ```
  **Verify:** Stratify parameter correctly set
  **Check:** Class distribution before/after sampling matches
  **Document:** Pre-sample and post-sample class ratios

- [ ] **Check 1.1.4:** Verify class distribution (97.7% attacks, 2.3% benign)
  ```bash
  # Check if these numbers are documented in code
  grep -r "97.7\|2.3\|class.*distribution" experiments/
  # Or check preprocessing metadata
  cat data/processed/binary/binary_metadata.json | grep -i "distribution\|benign\|attack"
  ```
  **Document:** Exact percentages, number of samples per class

**CRITICAL MATH ERROR RESOLUTION:**

- [ ] **Check 1.1.5:** Resolve Line 118 mathematical inconsistency
  ```
  Current: "4,501,906 records (19.5% of original dataset)"
  Problem: 23M × 19.5% = 4,485,000 ≠ 4,501,906

  Possible scenarios:
  A) Sampled size is 4,501,906 → percentage is 4,501,906/23,000,000 = 19.57%
  B) Percentage is exactly 19.5% → sampled size is 4,485,000
  C) Original dataset is not 23M → it's 4,501,906/0.195 = 23,087,210

  Action: Check code to determine true values, update paper
  ```

### 1.2 Train-Test Split Verification

**Paper Claims to Verify:**
- Line 143: "80/20 train-test split"
- Line 144: "3,601,524 training samples, 900,382 testing samples"

**Code Checks:**

- [ ] **Check 1.2.1:** Verify train-test split ratio
  ```bash
  # Check preprocessing script
  cat experiments/dvc_preprocessing.py | grep -A 10 "test_size\|train_size"
  ```
  **Calculate:** 3,601,524 + 900,382 = 4,501,906 ✓ (matches claimed sampled size)
  **Calculate:** 900,382 / 4,501,906 = 0.2000 = 20% ✓
  **Document:** Confirm implementation matches paper

- [ ] **Check 1.2.2:** Verify stratified split maintained
  ```bash
  # Check if stratify parameter used in train_test_split
  grep -B 5 -A 10 "train_test_split" experiments/dvc_preprocessing.py
  ```
  **Verify:** `stratify=y` parameter present
  **Document:** Class distribution preserved in both train and test

- [ ] **Check 1.2.3:** Verify random_state for reproducibility
  ```bash
  # Check for random seed in split
  grep "random_state\|random_seed" experiments/dvc_preprocessing.py
  ```
  **Document:** Random state value used (paper claims seeds 42-46)

### 1.3 Multi-Run Validation Strategy

**Paper Claims to Verify:**
- Line 147: "Each experiment was repeated 5 times using different random seeds (42, 43, 44, 45, 46)"
- Line 151-152: "5 different random seeds ensure statistical robustness"

**Code Checks:**

- [ ] **Check 1.3.1:** Verify N_RUNS = 5 in code
  ```bash
  # Check algorithm_comparison.py
  grep -n "N_RUNS\|n_runs\|num_runs" experiments/algorithm_comparison.py
  ```
  **Document:** Confirm 5 runs per configuration

- [ ] **Check 1.3.2:** Verify seed range 42-46
  ```bash
  # Check random seed generation
  grep -A 5 "range.*42\|seeds.*=.*\[42" experiments/algorithm_comparison.py
  # Or check if seeds calculated as range(42, 47)
  grep -B 5 -A 5 "random.*seed\|RandomState" experiments/algorithm_comparison.py
  ```
  **Verify:** Seeds are exactly [42, 43, 44, 45, 46]
  **Document:** How seeds are used (model initialization? train-test split? both?)

- [ ] **Check 1.3.3:** CRITICAL - Verify train-test split is NOT re-split for each run
  ```bash
  # This is the overfitting concern from academic review
  # Check if split happens inside or outside the N_RUNS loop
  cat experiments/algorithm_comparison.py | grep -B 20 -A 20 "for.*run\|for.*seed"
  ```
  **CRITICAL QUESTION:** Is the same train-test split reused across all 5 runs?
  - ✅ **Good:** Split once, reuse for all runs (seeds only affect model initialization)
  - ❌ **Bad:** Split happens inside loop (different data for each run = data leakage)

  **If Bad:** This invalidates the 705 experiments claim (not statistically independent)
  **Document:** Exact implementation, flag if problematic

### 1.4 Hyperparameter Search Verification

**Paper Claims to Verify:**
- Line 142-146: "Grid search with cross-validation"
- Line 144: "705 experiments (70+ configurations per algorithm)"

**Code Checks:**

- [ ] **Check 1.4.1:** Count actual configurations per algorithm
  ```bash
  # Check algorithm_comparison.py for param_grid definitions
  grep -A 30 "param_grid.*=\|PARAM_GRID" experiments/algorithm_comparison.py | head -100
  ```
  **Calculate:** Product of all parameter combinations per algorithm
  **Verify:** Each algorithm has ~70 configurations as claimed
  **Document:** Exact counts per algorithm

- [ ] **Check 1.4.2:** Verify total experiments = 705
  ```bash
  # 10 algorithms × 70 configs/algo × 5 runs = 3,500 total runs
  # Or is it: 10 algorithms × ~14 configs/algo × 5 runs = 700 ≈ 705?
  ```
  **Calculate:** Total = Σ(configs_per_algo) × 5 runs
  **Document:** How 705 is calculated (inconsistency with 70 configs × 10 algos × 5 runs = 3500?)

- [ ] **Check 1.4.3:** Verify if grid search uses separate validation set or cross-validation
  ```bash
  # Check for GridSearchCV or manual cross-validation
  grep -n "GridSearchCV\|cross_val_score\|KFold" experiments/algorithm_comparison.py
  ```
  **CRITICAL:** If no cross-validation found, this means hyperparameters selected based on test set
  **Document:** Validation strategy used for hyperparameter selection

### 1.5 Missing Value Imputation

**Paper Claims to Verify:**
- Line 134-135: "Missing values were imputed using median strategy"

**Code Checks:**

- [ ] **Check 1.5.1:** Verify imputation strategy
  ```bash
  # Check preprocessing script
  grep -n "imputer\|SimpleImputer\|fillna\|median" experiments/dvc_preprocessing.py
  ```
  **Document:** Exact imputation method used

- [ ] **Check 1.5.2:** Verify imputation happens before train-test split
  ```bash
  # CRITICAL: Imputing after split can cause data leakage
  # Check order of operations in preprocessing
  cat experiments/dvc_preprocessing.py | grep -n "impute\|train_test_split\|fit_transform"
  ```
  **Verify:** Imputer fitted only on training data, then applied to test data
  **Document:** Order of operations

### 1.6 Computational Resources Verification

**Paper Claims to Verify:**
- Line 256-257: "Peak memory: 2.8GB (training), 1.2GB (inference)"
- Line 258-259: "Average training time: 45 seconds (Logistic Regression) to 8 minutes (MLP)"

**Code Checks:**

- [ ] **Check 1.6.1:** Find memory measurement implementation
  ```bash
  # Check enhanced_metrics_collector.py
  grep -A 20 "memory\|mem_info\|resource" experiments/enhanced_metrics_collector.py
  ```
  **Verify:** How peak memory is measured (psutil? tracemalloc?)
  **Document:** Measurement method

- [ ] **Check 1.6.2:** Verify training time measurements
  ```bash
  # Check if time.time() or timeit used
  grep -n "time\(\)\|timeit\|training.*time" experiments/algorithm_comparison.py
  ```
  **Verify:** Times are wall-clock time (not CPU time)
  **Document:** Fastest and slowest algorithms with exact times

- [ ] **Check 1.6.3:** Verify hardware specifications
  ```bash
  # Paper claims experiments ran on "Intel Core i7-8700K, 32GB RAM"
  # Check if documented in code or configs
  grep -r "i7\|8700K\|32GB" experiments/ configs/
  ```
  **Document:** Confirm hardware used

### 1.7 Balanced Accuracy Discrepancy Investigation

**Paper Issue Identified by User:**
- Table 3 shows different balanced accuracy values than Figure 4 graph
- Need to verify which is correct

**Code Checks:**

- [ ] **Check 1.7.1:** Find result consolidation logic
  ```bash
  # Check consolidate_results.py for how balanced accuracy is calculated
  cat experiments/consolidate_results.py | grep -A 30 "balanced.*accuracy\|recall_score"
  ```
  **Verify:** Calculation is (sensitivity + specificity) / 2
  **Document:** Exact formula used

- [ ] **Check 1.7.2:** Check for rounding or aggregation differences
  ```bash
  # Table might show mean across runs, graph might show individual runs
  grep -A 20 "mean\(\)\|std\(\)\|aggregate" experiments/consolidate_results.py
  ```
  **Document:** How results are aggregated for table vs. figure

- [ ] **Check 1.7.3:** Verify results.json files match paper tables
  ```bash
  # Find latest full-mode results
  ls -lt iot-ids-research/experiments/results/full/
  # Check a sample results file
  # cat experiments/results/full/<timestamp>_logistic_regression/results.json
  ```
  **Cross-reference:** JSON values → Table 3 values → Figure 4 values
  **Document:** Any discrepancies found

---

## 🔴 PHASE 2: LITERATURE VALIDATION (2-3 days)

**Objetivo:** Validar ou refutar cada afirmação do paper com fontes primárias da literatura.

### 2.1 False Positive Rate Claims

**Paper Claims to Verify:**
- Line 35-36: "False positive rates of 1-2% are considered acceptable for SOC effectiveness"
- Line 201-203: "Our best models achieve ~98% specificity, translating to ~2% false positive rate"

**Literature Search:**

- [ ] **Search 2.1.1:** Find sources for 1-2% FPR acceptability threshold
  ```
  Search queries:
  - "false positive rate" AND "SOC" AND "acceptable" AND "intrusion detection"
  - "false positive rate" AND "1%" AND "2%" AND "IDS"
  - "alert fatigue" AND "security operations center" AND "threshold"

  Databases: Google Scholar, IEEE Xplore, ACM Digital Library
  Years: 2015-2024 (prefer recent)
  ```
  **Expected:** Industry reports, practitioner surveys, or academic papers citing this threshold
  **Document:** 2-3 authoritative citations supporting this claim
  **If Not Found:** Rephrase claim as "aim for low FPR" without specific percentage

- [ ] **Search 2.1.2:** Compare FPR with other IoT IDS papers
  ```
  Search queries:
  - "IoT intrusion detection" AND "false positive rate" AND "specificity"
  - "CICIoT2023" AND "false positive"
  - "IoT IDS" AND "specificity" AND "99%"
  ```
  **Document:** FPR values reported by other papers, compare with our 2%
  **Context:** Is our 2% FPR competitive or not?

- [ ] **Search 2.1.3:** Clarify FPR vs. Specificity terminology
  ```
  User question: "FPR é o inverso de especificidade?"

  Verify mathematical relationship:
  - Specificity = TN / (TN + FP) = True Negative Rate
  - FPR = FP / (FP + TN) = False Positive Rate
  - Therefore: FPR = 1 - Specificity

  If Specificity = 98%, then FPR = 2% ✓
  ```
  **Document:** Add clear definition in paper if not already present
  **Check:** Ensure paper uses terms consistently

### 2.2 Dataset Characteristics & Related Work

**Paper Claims to Verify:**
- Line 91-93: "CICIoT2023 dataset with 23 million network traffic records from 105 IoT devices"
- Line 94-95: "33 attack types including DDoS, Recon, Web-based, Brute Force, Spoofing, Mirai"

**Literature Search:**

- [ ] **Search 2.2.1:** Find ALL papers using CICIoT2023 dataset
  ```
  Search queries:
  - "CICIoT2023" OR "CIC IoT 2023" OR "CIC-IoT-2023"
  - author:"Neto" AND "CICIoT" (original dataset paper)
  - "Canadian Institute for Cybersecurity" AND "IoT" AND "2023"
  ```
  **Document:** Create table comparing:
  - Our approach vs. others using same dataset
  - Preprocessing differences
  - Metric comparisons (can we claim state-of-art?)
  **Expected:** 5-15 papers (dataset is relatively new, 2023)

- [ ] **Search 2.2.2:** Verify dataset specifications from original paper
  ```
  Find: Neto et al. (2023) original CICIoT2023 paper

  Verify from source:
  - Total records: 23 million? (exact number)
  - Number of devices: 105? (exact count)
  - Attack types: 33? (list all)
  - Class distribution: 97.7% attacks, 2.3% benign?
  ```
  **CRITICAL:** Our Line 118 math error might stem from misunderstanding original size
  **Document:** Exact specifications from authoritative source

- [ ] **Search 2.2.3:** Understand real-world IoT traffic patterns
  ```
  User question: "97.7% ataques e 2.3% tráfego benigno - isso é realista?"

  Search queries:
  - "IoT traffic" AND "benign" AND "malicious" AND "ratio"
  - "IoT network" AND "attack traffic" AND "real-world"
  - "IoT honeypot" AND "traffic distribution"
  ```
  **Question:** Is 97.7% attack traffic realistic or artifact of dataset creation?
  **Document:** Explain in paper whether this reflects:
  - Real-world IoT networks (probably not)
  - Honeypot data (more likely)
  - Synthetic attack simulation (most likely)
  **Action:** Add limitation discussion if unrealistic

### 2.3 IoT Deployment Architecture Claims

**Paper Claims to Verify:**
- Line 272-274: "Fog node with 8-core CPU and 31GB RAM can process detection for 50-100 IoT devices"
- Line 275-276: "Typical IoT gateway handles 1,000 packets/second"

**Literature Search:**

- [ ] **Search 2.3.1:** Find fog computing hardware specifications
  ```
  Search queries:
  - "fog computing" AND "hardware" AND "8 core" AND "IoT"
  - "edge server" AND "specifications" AND "IoT gateway"
  - "fog node" AND "CPU" AND "RAM" AND "capacity"
  ```
  **Document:** Typical fog node specs from vendors or academic studies
  **Verify:** Is 8-core/31GB RAM realistic? (seems specific - maybe Raspberry Pi cluster?)

- [ ] **Search 2.3.2:** Validate IoT gateway throughput claim
  ```
  Search queries:
  - "IoT gateway" AND "throughput" AND "packets per second"
  - "IoT gateway" AND "1000 pps" OR "1,000 packets"
  - "LoRaWAN gateway" OR "MQTT broker" AND "throughput"
  ```
  **Question:** 1,000 pps seems low for modern gateway - verify
  **Document:** Range of typical throughputs, cite source

- [ ] **Search 2.3.3:** Understand IoT deployment layers
  ```
  User question: "Camadas de deployment (fog nodes, gateways, etc.)"

  Search: "IoT architecture" AND "layers" AND "fog" AND "edge" AND "gateway"
  ```
  **Action:** Add clear architecture diagram to paper if not present
  **Clarify:** Device → Gateway → Fog Node → Cloud hierarchy

### 2.4 Attack Type: Lateral Movement

**Paper Claims to Verify:**
- Line 99: "Lateral movement attacks" mentioned in attack taxonomy

**Literature Search:**

- [ ] **Search 2.4.1:** Define lateral movement in IoT context
  ```
  User question: "O que é 'lateral movement' no contexto de IoT?"

  Search queries:
  - "lateral movement" AND "IoT" AND "intrusion detection"
  - "lateral movement" AND "network attack" AND "definition"
  - MITRE ATT&CK + "lateral movement" + IoT
  ```
  **Document:** Clear definition with example in IoT network
  **Example:** Attacker compromises one IoT device (e.g., camera), uses it to attack other devices on same network (e.g., smart locks)

- [ ] **Search 2.4.2:** Verify if CICIoT2023 contains lateral movement attacks
  ```
  Check: Original CICIoT2023 paper attack taxonomy
  Question: Is "lateral movement" explicitly labeled or inferred from other attack types?
  ```
  **Document:** Clarify in paper whether lateral movement is:
  - Distinct attack category in dataset
  - Subcategory of another attack type
  - Analysis/interpretation of attack patterns

### 2.5 Ensemble Methods Clarification

**Paper Claims to Verify:**
- Random Forest and Gradient Boosting are ensemble methods (mentioned in results)

**Literature Search:**

- [ ] **Search 2.5.1:** Define ensemble methods clearly
  ```
  User question: "O que são métodos de ensemble?"

  No search needed - define directly:
  - Ensemble = combining multiple models for better prediction
  - Bagging (Bootstrap Aggregating): Random Forest
  - Boosting: Gradient Boosting, AdaBoost
  - Stacking, Voting, etc.
  ```
  **Action:** Add brief explanation in Related Work or Methodology
  **Reference:** Breiman (2001) Random Forests, Friedman (2001) Gradient Boosting

### 2.6 K-Fold Cross-Validation vs. Single Split

**Paper Claims to Verify:**
- Line 143-144: Uses single 80/20 split
- Line 151-152: Claims 5 random seeds ensure "statistical robustness"

**Literature Search:**

- [ ] **Search 2.6.1:** Compare validation strategies in ML literature
  ```
  Search queries:
  - "train test split" vs "k-fold cross-validation" AND "comparison"
  - "single split" AND "overfitting" AND "evaluation"
  - "repeated random subsampling" vs "cross-validation"
  ```
  **Question:** Is single split + 5 seeds equivalent to 5-fold CV?
  - **Answer:** No, 5 seeds with same split = testing same data 5 times (if split inside loop)
  - **Better:** 5-fold CV or 5 different random splits

  **Document:** Acknowledge limitation or justify approach

- [ ] **Search 2.6.2:** Find precedent in IoT IDS literature
  ```
  Search: How do other CICIoT2023 papers do train-test validation?
  ```
  **Document:** If others use single split, cite them; if others use CV, acknowledge limitation

---

## 🔴 PHASE 3: STATISTICAL ANALYSIS REVIEW (1 day)

**Objetivo:** Verificar rigor estatístico de todas as análises.

### 3.1 Section 4.3 Bayesian Comparison Review

**Paper Section to Verify:**
- Line 183-195: Bayesian comparison methodology (Brodersen et al.)

**Statistical Checks:**

- [ ] **Check 3.1.1:** Verify Bayesian comparison implementation
  ```bash
  # Check bayesian_metrics.py
  cat experiments/bayesian_metrics.py | head -100
  # Look for Brodersen et al. implementation
  grep -A 30 "def.*bayesian\|class.*Bayesian" experiments/bayesian_metrics.py
  ```
  **Verify:** Implementation matches Brodersen et al. (2010) paper
  **Document:** Briefly explain methodology in paper

- [ ] **Check 3.1.2:** Verify posterior probability calculations
  ```bash
  # Check for MCMC sampling or analytical solution
  grep -n "mcmc\|posterior\|probability.*distribution" experiments/bayesian_metrics.py
  ```
  **Verify:** Posterior probabilities reported in results are correctly calculated
  **Check:** p(accuracy_A > accuracy_B) calculation

- [ ] **Check 3.1.3:** Verify statistical significance threshold
  ```bash
  # Check if p > 0.95 threshold used for "significantly better"
  grep -n "0.95\|95%\|significance" experiments/consolidate_results.py
  ```
  **Document:** Clearly state significance threshold in paper

### 3.2 Balanced Accuracy Gap Analysis

**Paper Claims to Verify:**
- Line 207-209: "Difference between balanced and standard accuracy: 12-24 percentage points"

**Statistical Checks:**

- [ ] **Check 3.2.1:** Calculate gap for each algorithm
  ```bash
  # Extract balanced accuracy and standard accuracy from results
  # Compare: accuracy - balanced_accuracy for each algorithm
  ```
  **Expected:** Negative gap (balanced < standard) due to imbalance
  **Verify:** Gap is actually 12-24 percentage points as claimed
  **Document:** Show calculation in paper or supplementary material

- [ ] **Check 3.2.2:** Explain why gap exists
  ```
  Reason: Standard accuracy biased toward majority class (97.7% attacks)
  Example: Always predicting "attack" → 97.7% accuracy, 50% balanced accuracy
  Gap = 97.7% - 50% = 47.7 percentage points (if naive)

  Our models should show smaller gap (12-24 pp) due to learning
  ```
  **Action:** Add clear explanation in Discussion section

### 3.3 Overfitting Analysis (CRITICAL MISSING)

**Academic Review Finding:**
- Missing: No comparison between training and testing performance
- Risk: 705 experiments with single split could overfit

**Statistical Checks:**

- [ ] **Check 3.3.1:** Calculate train-test performance gap
  ```bash
  # Check if training metrics are saved in results
  grep -n "train.*accuracy\|fit.*score" experiments/algorithm_comparison.py
  ```
  **If Saved:** Compare train vs. test balanced accuracy for each algorithm
  **If Not Saved:** Re-run experiments with training metrics collection
  **Document:** Add table showing train vs. test performance

- [ ] **Check 3.3.2:** Check for signs of overfitting
  ```
  Red flags:
  - Train accuracy >> Test accuracy (e.g., 99.9% vs. 95%)
  - High variance across 5 runs (unstable model)
  - Perfect 100% train accuracy (memorization)
  ```
  **Analyze:** Do any algorithms show these patterns?
  **Document:** Discuss overfitting mitigation strategies used

- [ ] **Check 3.3.3:** Verify regularization parameters
  ```bash
  # Check if regularization used for Logistic Regression, SVC, MLP
  grep -A 20 "C=\|alpha=\|l1_ratio\|penalty" experiments/algorithm_comparison.py
  ```
  **Document:** Regularization strategies employed

### 3.4 Confidence Intervals and Standard Deviations

**Paper Claims to Verify:**
- Line 152-153: "Results reported as mean ± standard deviation across 5 runs"

**Statistical Checks:**

- [ ] **Check 3.4.1:** Verify all tables include std dev
  ```
  Review: Table 3 (main results) - does it show mean ± std?
  ```
  **If Missing:** Add standard deviations to results table
  **Calculate:** 95% confidence intervals if N=5 is used

- [ ] **Check 3.4.2:** Check if std dev is reasonable
  ```
  Expected: Low std dev (< 1%) indicates stable model
  High std dev (> 3%) indicates instability
  ```
  **Document:** Comment on model stability based on std dev

---

## 🟡 PHASE 4: TERMINOLOGY CLARIFICATION (1 day)

**Objetivo:** Clarificar todos os termos técnicos identificados pelo usuário.

### 4.1 Clarify Statistical Terms

- [ ] **Term 4.1.1:** False Positive Rate (FPR)
  ```
  Definition: FPR = FP / (FP + TN)
  Context: Of all actual negative cases, what % were incorrectly classified as positive?
  Example: Of 100 benign traffic samples, 2 flagged as attacks → FPR = 2%
  IoT IDS Context: FPR = benign traffic incorrectly flagged as attack (false alarm)
  ```
  **Action:** Add glossary or footnote in paper

- [ ] **Term 4.1.2:** Specificity (True Negative Rate)
  ```
  Definition: Specificity = TN / (TN + FP) = 1 - FPR
  Context: Of all actual negative cases, what % were correctly identified?
  Example: Of 100 benign traffic samples, 98 correctly identified → Specificity = 98%
  Relationship: Specificity = 1 - FPR (if Spec = 98%, then FPR = 2%)
  ```
  **Action:** Clarify relationship with FPR in paper

- [ ] **Term 4.1.3:** Balanced Accuracy
  ```
  Definition: Balanced Accuracy = (Sensitivity + Specificity) / 2
  Why Used: Standard accuracy is biased when classes are imbalanced (97.7% vs 2.3%)
  Example:
    - Standard accuracy: 97% (could be just predicting majority class)
    - Balanced accuracy: 85% (shows true performance on both classes)
  ```
  **Action:** Explain clearly in Methodology section

- [ ] **Term 4.1.4:** Sensitivity (Recall, True Positive Rate)
  ```
  Definition: Sensitivity = TP / (TP + FN)
  Context: Of all actual positive cases (attacks), what % were detected?
  IoT IDS Context: Of all real attacks, what % did the IDS catch?
  Trade-off: High sensitivity (catch all attacks) may increase FPR (more false alarms)
  ```

### 4.2 Clarify ML/Security Terms

- [ ] **Term 4.2.1:** Data Leakage
  ```
  Definition: Information from test set "leaking" into training process
  Examples:
    - Normalizing before splitting (test data influences scaler)
    - Using test data for feature selection
    - Hyperparameter tuning on test set
  In Our Paper: Check if any leakage occurred (Phase 1 Check 1.5.2)
  ```
  **Action:** Explicitly state "no data leakage" in paper if verified

- [ ] **Term 4.2.2:** Ensemble Methods (already covered in 2.5)
  ```
  Reference: Add brief explanation in Related Work
  ```

- [ ] **Term 4.2.3:** Lateral Movement (already covered in 2.4)
  ```
  Reference: Add definition in attack taxonomy section
  ```

### 4.3 Clarify IoT Architecture Terms

- [ ] **Term 4.3.1:** Fog Node
  ```
  Definition: Computational resource between IoT devices and cloud
  Purpose: Local processing, reduced latency, privacy preservation
  Example: Small server in building processing data from local IoT devices
  Paper Context: Where our IDS would be deployed
  ```

- [ ] **Term 4.3.2:** IoT Gateway
  ```
  Definition: Device connecting IoT devices to network/internet
  Function: Protocol translation, data aggregation, basic filtering
  Example: Hub connecting Zigbee sensors to WiFi network
  Paper Context: Traffic passes through gateway before reaching fog node
  ```

- [ ] **Term 4.3.3:** Edge Computing vs. Fog Computing
  ```
  Edge: Processing at device level (on IoT device itself)
  Fog: Processing at intermediate layer (gateway/local server)
  Cloud: Processing at remote datacenter
  Paper Context: Our IDS targets fog layer (not edge, not cloud)
  ```

---

## 🟡 PHASE 5: LATEX CORRECTIONS (2-3 days)

**Objetivo:** Implementar todas as correções identificadas no review.

### 5.1 Critical LaTeX Fixes

- [ ] **Fix 5.1.1:** Resolve mathematical inconsistency (Line 118)
  ```latex
  Current: "4,501,906 records (19.5\% of the original dataset)"

  After Phase 1 verification, replace with correct values:
  Option A: "4,501,906 records (19.57\% of the original 23 million records)"
  Option B: "4,485,000 records (19.5\% of the original 23 million records)"
  Option C: "4,501,906 records (19.5\% of the original 23,087,210 records)"
  ```
  **Decision:** Based on Check 1.1.2 results

- [ ] **Fix 5.1.2:** Add overfitting analysis section
  ```latex
  Location: After Section 4.2 (Results)
  New Subsection: 4.X Overfitting Analysis

  Content:
  - Train vs. test performance comparison table
  - Discussion of generalization gap
  - Regularization strategies employed
  ```
  **Reference:** Phase 3 Check 3.3.1 results

- [ ] **Fix 5.1.3:** Clarify validation strategy (Line 143-147)
  ```latex
  Current: Mentions 80/20 split and 5 seeds but unclear how related

  Revised: Add clear explanation:
  "The dataset was split once into 80\% training (3,601,524 samples)
  and 20\% testing (900,382 samples) using stratified sampling. Each
  model configuration was trained 5 times with different random
  initialization seeds (42-46) on the same data split to assess
  stability. Hyperparameters were selected using..."
  ```
  **Clarify:** Hyperparameter selection method (grid search on train, CV on train, or selected based on test?)

### 5.2 Major LaTeX Fixes

- [ ] **Fix 5.2.1:** Shorten abstract to 250 words
  ```latex
  Current: 287 words (Line 28-56)
  Target: 250 words maximum

  Strategy:
  - Remove redundant phrases
  - Condense methodology description
  - Keep key results (F1, balanced accuracy, inference time)
  ```

- [ ] **Fix 5.2.2:** Add missing citations for claims
  ```latex
  Line 35-36: "1-2\% FPR acceptable" → Add citation (from Phase 2 Search 2.1.1)
  Line 272-274: "Fog node specifications" → Add citation (from Phase 2 Search 2.3.1)
  Line 275-276: "1,000 pps gateway" → Add citation (from Phase 2 Search 2.3.2)
  ```

- [ ] **Fix 5.2.3:** Add code/data availability statement
  ```latex
  Location: Before Acknowledgments
  New Section: "Code and Data Availability"

  Content:
  "Code: [GitHub repository URL]
  Data: CICIoT2023 dataset available at [URL]
  Experiment results: [Zenodo DOI or supplementary material]"
  ```

- [ ] **Fix 5.2.4:** Improve hyperparameter table formatting
  ```latex
  Current: Table 2 is very dense (Line 160-170)

  Improvement:
  - Move detailed hyperparameters to supplementary material
  - Keep only final selected parameters in main text
  - Or split into 2 tables: "Search Space" and "Best Parameters"
  ```

### 5.3 Minor LaTeX Fixes

- [ ] **Fix 5.3.1:** Figure 4 contains Portuguese words
  ```
  User identified: "Palavras em português na figura"

  Action:
  - Identify Portuguese words in figure
  - Regenerate figure with English labels
  - Or edit figure file directly if possible
  ```

- [ ] **Fix 5.3.2:** Fix overly long sentences (>30 words)
  ```
  Academic review identified multiple long sentences

  Target locations:
  - Line 32-34 (Introduction)
  - Line 158-161 (Methodology)
  - Line 225-228 (Discussion)

  Strategy: Split into 2 shorter sentences or use semicolon
  ```

- [ ] **Fix 5.3.3:** Add missing definitions on first use
  ```
  Terms to define:
  - IDS (Line 29 - check if defined)
  - IoT (Line 30 - check if defined)
  - ML (Line 31 - check if defined)
  - DDoS (Line 99 - check if defined)
  ```

- [ ] **Fix 5.3.4:** Improve table captions
  ```
  Current: Captions are short (e.g., "Results comparison")
  Better: "Comparison of balanced accuracy, F1-score, and inference
          time across 10 machine learning algorithms on CICIoT2023
          test set. Values shown as mean ± standard deviation over
          5 runs with different random seeds."
  ```

### 5.4 References & Citations

- [ ] **Fix 5.4.1:** Add new references from literature search (Phase 2)
  ```latex
  New citations to add:
  - FPR acceptability sources (Phase 2 Search 2.1.1)
  - CICIoT2023 related work (Phase 2 Search 2.2.1)
  - Fog computing specs (Phase 2 Search 2.3.1)
  - Lateral movement definition (Phase 2 Search 2.4.1)
  ```

- [ ] **Fix 5.4.2:** Fix reference formatting inconsistencies
  ```
  Check: All references follow same citation style (IEEE, ACM, etc.)
  Fix: Missing DOIs, page numbers, or years
  ```

- [ ] **Fix 5.4.3:** Ensure all citations in text have references
  ```bash
  # Check for orphan citations
  grep -o "\\cite{[^}]*}" model2011.tex | sort -u > cited.txt
  # Compare with references.bib entries
  grep "^@" /Users/augusto/mestrado/references.bib | cut -d'{' -f2 | cut -d',' -f1 > available.txt
  # Find differences
  diff cited.txt available.txt
  ```

---

## 🟢 PHASE 6: RE-RUN DECISION (1 day)

**Objetivo:** Decidir se é necessário re-executar os 705 experimentos baseado nos achados das fases anteriores.

### 6.1 Decision Criteria

**Re-run IS REQUIRED if any of these are TRUE:**

- [ ] **Criterion 6.1.1:** Data preprocessing error found
  ```
  Examples:
  - Incorrect sampling percentage
  - Data leakage in normalization/imputation
  - Wrong train-test split
  - Class imbalance not preserved
  ```
  **Status:** To be determined after Phase 1

- [ ] **Criterion 6.1.2:** Validation strategy flaw found
  ```
  Examples:
  - Train-test split happens inside 5-run loop (not independent runs)
  - Hyperparameters selected based on test set (not validation/CV)
  - Same test data used multiple times without correction
  ```
  **Status:** To be determined after Phase 1 Check 1.3.3

- [ ] **Criterion 6.1.3:** Implementation error found
  ```
  Examples:
  - Balanced accuracy calculated incorrectly
  - Wrong metric reported in paper
  - Algorithm implemented incorrectly
  ```
  **Status:** To be determined after Phase 3

**Re-run is NOT required if:**

- [ ] **Criterion 6.1.4:** Only minor discrepancies
  ```
  Examples:
  - Rounding differences (4,501,906 vs 4,485,000 if both are correct)
  - Hardware specs not exactly matching claims (not critical)
  - Missing documentation (can add without re-run)
  ```

- [ ] **Criterion 6.1.5:** Only presentation issues
  ```
  Examples:
  - LaTeX formatting
  - Missing citations (can add)
  - Unclear writing (can clarify)
  - Figure language (can regenerate plots from existing results)
  ```

### 6.2 Re-run Strategy (if needed)

- [ ] **Strategy 6.2.1:** Identify minimum re-run scope
  ```
  Options:
  A) Full re-run: All 10 algorithms × 705 experiments (~60+ hours)
  B) Partial re-run: Only affected algorithms
  C) Validation re-run: Single best config per algorithm × 5 runs (quick check)
  ```

- [ ] **Strategy 6.2.2:** Fix issues before re-run
  ```
  Update:
  - experiments/dvc_preprocessing.py (if preprocessing issues)
  - experiments/algorithm_comparison.py (if validation issues)
  - dvc.yaml (if pipeline issues)

  Test:
  - Run single algorithm first (e.g., Logistic Regression)
  - Verify outputs before full run
  ```

- [ ] **Strategy 6.2.3:** Document changes
  ```
  Create: CHANGELOG.md documenting:
  - What was wrong
  - What was fixed
  - Comparison of old vs. new results (if significant)
  ```

### 6.3 Results Comparison (if re-run executed)

- [ ] **Comparison 6.3.1:** Compare key metrics
  ```
  Metrics to compare:
  - Balanced accuracy (mean ± std)
  - F1-score
  - Inference time
  - Memory usage

  Acceptable difference: < 1% (within statistical noise)
  Significant difference: > 3% (indicates real change)
  ```

- [ ] **Comparison 6.3.2:** Update paper with new results
  ```
  Update:
  - Table 3 (main results)
  - Figure 4 (comparison plot)
  - All metric mentions in text
  - Discussion section if conclusions change
  ```

- [ ] **Comparison 6.3.3:** Decide if differences require explanation
  ```
  If significant differences found:
  - Add subsection explaining what changed and why
  - Acknowledge limitation of previous approach
  - Justify new methodology
  ```

---

## PHASE DEPENDENCIES & TIMELINE

```
Week 1 (2-3 days):
- Phase 1: Code Verification → MUST complete first
- Phase 2: Literature Search → Can parallelize with Phase 1
- Phase 3: Statistical Review → Depends on Phase 1 results

Week 2 (2-3 days):
- Phase 4: Terminology Clarification → Can parallelize with Phase 2
- Phase 5: LaTeX Corrections → Depends on Phase 1-4 findings
- Phase 6: Re-run Decision → Depends on Phase 1-3 findings

Week 3 (2-4 days if re-run needed):
- Phase 6: Execute re-runs (if required)
- Phase 5: Final LaTeX corrections with new results
- Final review before resubmission
```

---

## EXECUTION CHECKLIST SUMMARY

### ✅ Phase 1: Critical Code Verification (1-2 days)
- [ ] 1.1 Dataset Size & Sampling (5 checks)
- [ ] 1.2 Train-Test Split (3 checks)
- [ ] 1.3 Multi-Run Validation (3 checks) **CRITICAL**
- [ ] 1.4 Hyperparameter Search (3 checks)
- [ ] 1.5 Missing Value Imputation (2 checks)
- [ ] 1.6 Computational Resources (3 checks)
- [ ] 1.7 Balanced Accuracy Discrepancy (3 checks)

**Total: 22 code verification checks**

### 🔍 Phase 2: Literature Validation (2-3 days)
- [ ] 2.1 False Positive Rate Claims (3 searches)
- [ ] 2.2 Dataset Characteristics (3 searches)
- [ ] 2.3 IoT Architecture Claims (3 searches)
- [ ] 2.4 Attack Type: Lateral Movement (2 searches)
- [ ] 2.5 Ensemble Methods (1 definition)
- [ ] 2.6 K-Fold vs Single Split (2 searches)

**Total: 14 literature searches**

### 📊 Phase 3: Statistical Analysis Review (1 day)
- [ ] 3.1 Section 4.3 Bayesian Comparison (3 checks)
- [ ] 3.2 Balanced Accuracy Gap (2 checks)
- [ ] 3.3 Overfitting Analysis **CRITICAL** (3 checks)
- [ ] 3.4 Confidence Intervals (2 checks)

**Total: 10 statistical checks**

### 📝 Phase 4: Terminology Clarification (1 day)
- [ ] 4.1 Statistical Terms (4 terms)
- [ ] 4.2 ML/Security Terms (3 terms)
- [ ] 4.3 IoT Architecture Terms (3 terms)

**Total: 10 terminology clarifications**

### 📄 Phase 5: LaTeX Corrections (2-3 days)
- [ ] 5.1 Critical LaTeX Fixes (3 fixes)
- [ ] 5.2 Major LaTeX Fixes (4 fixes)
- [ ] 5.3 Minor LaTeX Fixes (4 fixes)
- [ ] 5.4 References & Citations (3 fixes)

**Total: 14 LaTeX corrections**

### 🔄 Phase 6: Re-run Decision (1 day + execution time if needed)
- [ ] 6.1 Decision Criteria (5 criteria)
- [ ] 6.2 Re-run Strategy (3 strategies) **IF NEEDED**
- [ ] 6.3 Results Comparison (3 comparisons) **IF NEEDED**

**Total: 11 decision/execution items**

---

## GRAND TOTAL: 81 CHECKLIST ITEMS

**Estimated Time:**
- Best case (no re-run): 7-10 days
- Worst case (full re-run): 10-14 days

**Priority Order:**
1. Phase 1 (CRITICAL - determines if re-run needed)
2. Phase 3 (CRITICAL - statistical validity)
3. Phase 2 (HIGH - claims validation)
4. Phase 6 (Decision point)
5. Phase 5 (After all validation complete)
6. Phase 4 (Can be done in parallel with Phase 5)

---

## DECISION LOG

As each phase completes, document decisions here:

### Phase 1 Decisions:
```
[To be filled after Phase 1 completion]
- Dataset size verified: [value]
- Sampling percentage verified: [value]
- Math error resolution: [chosen option A/B/C]
- Train-test split strategy: [good/problematic]
- Overfitting risk: [low/medium/high]
```

### Phase 2 Decisions:
```
[To be filled after Phase 2 completion]
- FPR claim: [validated/refuted/modified]
- Dataset characteristics: [accurate/needs correction]
- IoT architecture claims: [validated/needs citation]
```

### Phase 3 Decisions:
```
[To be filled after Phase 3 completion]
- Bayesian analysis: [correct/needs correction]
- Overfitting: [present/absent/unclear]
- Statistical rigor: [sufficient/insufficient]
```

### Phase 6 Decision:
```
[To be filled after Phase 1-3 completion]
- Re-run required: [YES/NO]
- If YES, scope: [full/partial/validation-only]
- If NO, reason: [only minor issues/presentation only]
```

---

## NOTES & OBSERVATIONS

Use this section to record findings during execution:

```
[Add notes as you work through phases]

Example:
- 2025-11-09: Phase 1.1.2 - Found sampling is actually 20%, not 19.5%
- 2025-11-09: Phase 1.3.3 - CRITICAL: Split happens inside loop, re-run required
- 2025-11-10: Phase 2.1.1 - Found 3 good citations for FPR threshold
```

---

**END OF REVISION PLAN**

**Next Steps:**
1. Review this plan with Augusto
2. Begin Phase 1: Code Verification
3. Update progress in this document as we go
4. Make go/no-go decision on re-run after Phase 1-3
5. Execute corrections and finalize paper
