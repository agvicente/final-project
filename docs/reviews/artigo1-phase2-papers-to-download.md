# Phase 2: Papers to Download from Zotero
**Data:** 2025-11-09
**Status:** Ready for Download
**Workflow:** Zotero → Find Available PDF → Claude reads and extracts

---

## 📋 INSTRUÇÕES PARA DOWNLOAD

Para cada paper listado abaixo:
1. Abra Zotero
2. Busque pela **chave BibTeX** (ex: `ahmedEnhancingInternetThings2025`)
3. Verifique se já tem PDF anexado
4. Se não tem: Clique com botão direito → **"Find Available PDF"**
5. Zotero baixará via acesso institucional
6. ✅ Marque como baixado na lista

**Após baixar todos (ou a maioria):** Me avise que pode prosseguir com a leitura!

---

## 🔴 CLAIM 1: FPR Threshold 1-2% (Linha 71)
**Objetivo:** Encontrar fonte acadêmica para "require FPR below 1-2%"

### Papers para Download (Prioridade Alta → Baixa):

#### 🔥 **PRIORIDADE 1** - Mais promissor
- [X] **`benkhelifaCriticalReviewPractices2018`**
  - **Título:** "A Critical Review of Practices and Challenges in Intrusion Detection Systems for IoT"
  - **Por quê:** Survey que discute operational constraints e challenges in IoT IDS
  - **Buscar:** Seções sobre operational constraints, false positive tolerance, SOC workload

#### 🔥 **PRIORIDADE 2**
- [X] **`alghushairyEfficientSupportVector2024`**
  - **Título:** "An Efficient Support Vector Machine Algorithm Based Network Outlier Detection System"
  - **Por quê:** Caracterizado por "low false alarms" - pode ter thresholds
  - **Buscar:** False alarm rate metrics, acceptable thresholds, operational requirements

#### 🟡 **PRIORIDADE 3**
- [X] **`chalichalamalaLogisticRegressionEnsemble2023`**
  - **Título:** "Logistic Regression Ensemble Classifier for IDS in IoT"
  - **Por quê:** Avaliado usando false alarm rate (FAR), true negative rate (TNR)
  - **Buscar:** FAR thresholds, operational metrics, acceptable performance

#### 🟡 **PRIORIDADE 4**
- [X] **`ferragDeepLearningCyber2020`**
  - **Título:** "Deep Learning for Cyber Security Intrusion Detection"
  - **Por quê:** Usa false alarm rate como key performance indicator
  - **Buscar:** FAR benchmarks, industry standards, acceptable rates

#### 🟡 **PRIORIDADE 5**
- [X] **`ahmedEnhancingInternetThings2025`**
  - **Título:** "Enhancing IoT Security Using Performance Gradient Boosting"
  - **Por quê:** Discute minimizing false positives como preponderant consideration
  - **Buscar:** False positive minimization targets, operational impacts

---

## 🔴 CLAIM 2: Fog/Edge Hardware (8-core/31GB típico) (Linha 139)
**Objetivo:** Validar se 8-core/31GB representa "typical fog node"

### Papers para Download:

#### 🔥 **PRIORIDADE 1**
- [X] **`spadaccinoIntrusionDetectionSystems2022`**
  - **Título:** "Intrusion Detection Systems for IoT: Opportunities and Challenges Offered by Edge Computing and ML"
  - **Por quê:** Survey dedicado sobre IDS em edge computing
  - **Buscar:** Fog node specifications, hardware capabilities, typical configurations

#### 🔥 **PRIORIDADE 2**
- [X] **`capogrossoMachineLearningOrientedSurvey2024`**
  - **Título:** "A Machine Learning-Oriented Survey on Tiny Machine Learning"
  - **Por quê:** Comprehensive survey on TinyML for resource-constrained IoT hardware
  - **Buscar:** Hardware specifications tables, edge device capabilities, memory/CPU ranges

#### 🟡 **PRIORIDADE 3**
- [X] **`biancoBenchmarkAnalysisRepresentative2018`**
  - **Título:** "Benchmark Analysis of Representative Deep Neural Network Architectures"
  - **Por quê:** Analisa memory usage e computational complexity em embedded systems (NVIDIA Jetson TX1)
  - **Buscar:** Hardware specs for edge devices, performance benchmarks, memory requirements

#### 🟡 **PRIORIDADE 4**
- [X] **`diroDistributedAttackDetection2018`**
  - **Título:** "Distributed Attack Detection Scheme Using Deep Learning for IoT"
  - **Por quê:** Discute fog networks e distributed detection
  - **Buscar:** Fog node specifications, distributed vs centralized hardware

---

## 🔴 CLAIM 3: Gateway Throughput 1000 pps (Linha 312)
**Objetivo:** Validar "1,000 packets/second (typical IoT gateway)"

### Papers para Download:

#### 🟡 **PRIORIDADE 1**
- [X] **`taoParallelAlgorithmNetwork2018`**
  - **Título:** "A Parallel Algorithm for Network Traffic Anomaly Detection Based on Isolation Forest"
  - **Por quê:** Addresses network traffic data processing e scalability
  - **Buscar:** Traffic rates, packets per second, gateway throughput, IoT network traffic

#### 🟡 **PRIORIDADE 2**
- [X] **`paulauskasLocalOutlierFactor2015`**
  - **Título:** "Local Outlier Factor Use for the Network Flow Anomaly Detection"
  - **Por quê:** Focuses on network flow metrics
  - **Buscar:** Network flow rates, aggregated metrics, typical traffic volumes

**NOTA:** Se estes não tiverem dados específicos, pode precisar buscar papers adicionais sobre IoT gateway specifications.

---

## 🔴 CLAIM 4: Edge Device RAM 512MB-2GB (Linhas 310, 344, 350)
**Objetivo:** Validar "512MB–2GB RAM typical in IoT deployments"

### Papers para Download:

#### 🔥 **PRIORIDADE 1**
- [X] **`capogrossoMachineLearningOrientedSurvey2024`** (já listado acima)
  - **Buscar:** Resource-constrained IoT hardware, memory constraints, embedded systems specs

#### 🔥 **PRIORIDADE 2**
- [X] **`sPerformanceBenchmarkingML2025`**
  - **Título:** "Performance Benchmarking of ML Models for Resource Constrained Devices"
  - **Por quê:** Focuses on resource-constrained devices com TinyML
  - **Buscar:** Device specifications, memory constraints, typical RAM ranges

#### 🟡 **PRIORIDADE 3**
- [X] **`canzianiAnalysisDeepNeural2017`**
  - **Título:** "An Analysis of Deep Neural Network Models for Practical Applications"
  - **Por quê:** Analisa memory footprint e power consumption for DNNs
  - **Buscar:** Memory requirements, device specifications, deployment constraints

#### 🟡 **PRIORIDADE 4**
- [X] **`howardMobileNetsEfficientConvolutional2017`**
  - **Título:** "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision"
  - **Por quê:** Mobile and embedded vision applications com latency constraints
  - **Buscar:** Target devices specifications, memory constraints, mobile hardware

#### 🟡 **PRIORIDADE 5**
- [X] **`brownleeExploringAccuracyEnergy2021`**
  - **Título:** "Exploring the Accuracy-Energy Trade-off in Machine Learning"
  - **Por quê:** Memory management e computational complexity in constrained environments
  - **Buscar:** Resource constraints, memory limitations, device capabilities

---

## 🔴 CLAIM 5: "First Comprehensive Baseline" CICIoT2023 (Linha 336)
**Objetivo:** Verificar se existem outros estudos comprehensive no CICIoT2023

### Papers para Download:

#### 🔥 **PRIORIDADE 1** - ESSENCIAL
- [X] **`netoCICIoT2023RealTimeDataset2023`**
  - **Título:** "CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment"
  - **Por quê:** DATASET ORIGINAL PAPER
  - **Buscar:**
    - Seção "Related Work" - outros estudos no dataset?
    - Benchmark results apresentados pelos autores
    - Citing papers (verificar no Google Scholar)
    - Qualquer comparação de algoritmos que eles fizeram

**AÇÃO ADICIONAL NECESSÁRIA:**
Após ler este paper, precisamos:
1. Buscar no Google Scholar: "CICIoT2023" (papers que citam)
2. Verificar página oficial do dataset para lista de papers
3. Buscar em IEEE Xplore e ACM Digital Library

---

## 🟡 CLAIM 6: Attack Dominance >95% (Linha 101)
**Objetivo:** Validar "real IoT traffic exhibits attack dominance (often >95%)"

### Papers para Download:

#### 🔥 **PRIORIDADE 1**
- [X] **`netoCICIoT2023RealTimeDataset2023`** (já listado acima)
  - **Buscar:** Real-world IoT traffic patterns, class distribution, attack statistics

#### 🟡 **PRIORIDADE 2**
- [X] **`ahmedEnhancingInternetThings2025`** (já listado acima)
  - **Buscar:** Class imbalance in IoT datasets, typical distribution patterns

#### 🟡 **PRIORIDADE 3**
- [X] **`bagadiRandomForestGradient2025`**
  - **Título:** "Random Forest and Gradient Boosting for Superior Intrusion Detection"
  - **Por quê:** Addresses class imbalance usando SMOTE
  - **Buscar:** Class imbalance ratios, typical distributions in security datasets

**NOTA:** Este claim pode ser problemático - "real IoT traffic" vs. "datasets" são diferentes. Datasets podem ser intencionalmente desbalanceados para treino.

---

## 🟢 CLAIM 7: Lateral Movement (Linha 71)
**Objetivo:** Adicionar definição/citação para "lateral movement"

### Papers para Download:

**NENHUM PAPER NO ZOTERO COBRE ESTE TÓPICO**

**AÇÃO NECESSÁRIA:**
1. Buscar MITRE ATT&CK framework references
2. Ou buscar papers sobre attack tactics/techniques
3. Ou simplesmente adicionar referência ao MITRE ATT&CK online

**SOLUÇÃO RÁPIDA:**
Posso buscar via WebSearch agora papers sobre MITRE ATT&CK para adicionar ao Zotero.

---

## 📊 RESUMO PARA DOWNLOAD

### Contagem por Prioridade:
- **🔥 PRIORIDADE 1 (Essenciais):** 6 papers
- **🟡 PRIORIDADE 2-5 (Importantes):** 12 papers
- **Total:** 18 papers únicos

### Lista Consolidada (evitando duplicatas):

**GRUPO A - Download Primeiro (6 papers essenciais):**
1. ✅ `benkhelifaCriticalReviewPractices2018` - FPR constraints
2. ✅ `spadaccinoIntrusionDetectionSystems2022` - Edge/fog hardware
3. ✅ `capogrossoMachineLearningOrientedSurvey2024` - TinyML + hardware specs
4. ✅ `sPerformanceBenchmarkingML2025` - Resource-constrained devices
5. ✅ `netoCICIoT2023RealTimeDataset2023` - **DATASET ORIGINAL (CRÍTICO)**
6. ✅ `alghushairyEfficientSupportVector2024` - False alarm rates

**GRUPO B - Download Depois (12 papers complementares):**
7. `ahmedEnhancingInternetThings2025`
8. `chalichalamalaLogisticRegressionEnsemble2023`
9. `ferragDeepLearningCyber2020`
10. `biancoBenchmarkAnalysisRepresentative2018`
11. `diroDistributedAttackDetection2018`
12. `taoParallelAlgorithmNetwork2018`
13. `paulauskasLocalOutlierFactor2015`
14. `canzianiAnalysisDeepNeural2017`
15. `howardMobileNetsEfficientConvolutional2017`
16. `brownleeExploringAccuracyEnergy2021`
17. `bagadiRandomForestGradient2025`
18. (Lateral movement - buscar separadamente)

---

## 🎯 PRÓXIMOS PASSOS

### Para Você (Augusto):
1. ✅ Abra Zotero
2. ✅ Comece pelo **GRUPO A** (6 papers essenciais)
3. ✅ Para cada um: Clique direito → "Find Available PDF"
4. ✅ Aguarde downloads completarem
5. ✅ Depois baixe **GRUPO B** se quiser cobertura completa
6. ✅ Me avise quando terminar (pode avisar após GRUPO A se quiser começar logo)

### Para Mim (Claude):
1. ⏳ Aguardando você confirmar downloads
2. ⏳ Depois: Leio PDFs em `/Users/augusto/Zotero/storage/`
3. ⏳ Extraio trechos relevantes com números/specs
4. ⏳ Crio documento com citações prontas por claim
5. ⏳ Sugiro inserções no paper com contexto expandido

---

## 📝 FORMATO DE RETORNO

Quando terminar, me avise assim:
```
Baixei GRUPO A completo (ou: Baixei X papers do Grupo A: lista)
Baixei GRUPO B completo (ou: Baixei Y papers do Grupo B: lista)
```

Também me diga se algum paper não conseguiu baixar, para buscarmos alternativas!

---

**STATUS:** ⏳ Aguardando download dos papers via Zotero
