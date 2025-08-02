# Planejamento Incremental de Experimentos - Dissertação de Mestrado
## Detecção de Intrusão Baseada em Anomalias em Sistemas IoT com Clustering Evolutivo

### 📋 Visão Geral do Planejamento
Este documento apresenta uma sequência incremental de experimentos que culminarão na solução completa proposta na dissertação, permitindo publicações intermediárias e ajustes de curso quando necessário.

### 📚 Estrutura Bibliográfica
Cada fase do planejamento inclui:
- **Referências Essenciais**: Trabalhos fundamentais e diretamente relacionados aos objetivos específicos
- **Referências Complementares**: Trabalhos de apoio, metodologias relacionadas e contexto ampliado  
- **Origem das Referências**: (P) = Proposta de Mestrado, (PI) = Pré-pesquisa Undermind

---

## 🎯 Fase 1: Fundamentos e MVP (Meses 1-3)

### Experimento 1.1: Baseline de Detecção de Anomalias em IoT
**Objetivo**: Estabelecer uma linha de base para comparação futura
**Duração**: 3-4 semanas

**Hipótese**: Algoritmos clássicos de detecção de anomalias podem identificar ataques em dados IoT com métricas mensuráveis

**Pipeline Mínimo**:
- Dataset: CICIoT2023 (amostra de 10% dos dados)
- Pré-processamento básico: normalização, remoção de NaNs
- Algoritmos baseline: Isolation Forest, One-Class SVM, LOF
- Métricas: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- Ferramentas: Jupyter Notebook, sklearn, pandas

**Entregáveis**:
- [ ] Notebook completo com análise exploratória
- [ ] Resultados comparativos dos 3 algoritmos baseline
- [ ] Documentação das limitações identificadas
- [ ] **Possível publicação**: Workshop ou conferência sobre análise comparativa de algoritmos clássicos em IoT

#### 📚 Referências Bibliográficas - Experimento 1.1

**Referências Essenciais:**
1. **Neto et al. (2023)** - CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment (P)
2. **Liu et al. (2008)** - Isolation Forest ([127] PI)
3. **Liu et al. (2012)** - Isolation-Based Anomaly Detection ([121] PI)
4. **Benkhelifa et al. (2018)** - A Critical Review of Practices and Challenges in Intrusion Detection Systems for IoT (P/[125] PI)
5. **Ahmad et al. (2021)** - Network intrusion detection system: A systematic study of machine learning and deep learning approaches (P)
6. **Cook et al. (2020)** - Anomaly detection for iot time-series data: A survey (P)

**Referências Complementares:**
7. **Laskar et al. (2021)** - Extending Isolation Forest for Anomaly Detection in Big Data via K-Means ([19] PI)
8. **Al-amri et al. (2021)** - A Review of Machine Learning and Deep Learning Techniques for Anomaly Detection in IoT Data ([93] PI)
9. **Rafique et al. (2024)** - Machine Learning and Deep Learning Techniques for Internet of Things Network Anomaly Detection ([115] PI)
10. **Markiewicz & Sgandurra (2020)** - Clust-IT: clustering-based intrusion detection in IoT environments ([32] PI)
11. **Alhakami et al. (2019)** - Network Anomaly Intrusion Detection Using a Nonparametric Bayesian Approach and Feature Selection ([43] PI)
12. **Alani & Awad (2023)** - An Intelligent Two-Layer Intrusion Detection System for the Internet of Things ([65] PI)
13. **CICIDS2018 Dataset** - Canadian Institute for Cybersecurity (P)
14. **CICIDS2017 Dataset** - Canadian Institute for Cybersecurity (P)

---

### Experimento 1.2: Análise de Concept Drift em Dados IoT
**Objetivo**: Identificar e quantificar mudanças temporais nos padrões de dados
**Duração**: 2-3 semanas

**Hipótese**: Dados IoT apresentam concept drift que afeta a performance de algoritmos estáticos

**Pipeline**:
- Dataset: CICIoT2023 (dados temporais completos)
- Análise de drift temporal: ADWIN, DDM
- Visualização de mudanças de distribuição
- Impacto do drift nos modelos baseline

**Entregáveis**:
- [ ] Análise quantitativa do concept drift
- [ ] Métricas de degradação temporal dos modelos
- [ ] **Possível publicação**: Artigo sobre caracterização de concept drift em dados IoT

#### 📚 Referências Bibliográficas - Experimento 1.2

**Referências Essenciais:**
1. **Lu et al. (2019)** - Learning under Concept Drift: A Review ([119] PI)
2. **Wahab (2022)** - Intrusion Detection in the IoT Under Data and Concept Drifts: Online Deep Learning Approach ([11] PI)
3. **Xu et al. (2023)** - ADTCD: An Adaptive Anomaly Detection Approach Toward Concept Drift in IoT ([16] PI)
4. **Bharani et al. (2024)** - Adaptive Real-Time Malware Detection for IoT Traffic Streams: A Comparative Study of Concept Drift Detection Techniques ([3] PI)
5. **Yang & Shami (2021)** - A Lightweight Concept Drift Detection and Adaptation Framework for IoT Data Streams ([67] PI)
6. **Bu et al. (2018)** - A pdf-Free Change Detection Test Based on Density Difference Estimation ([122] PI)

**Referências Complementares:**
7. **Xu et al. (2024)** - Addressing Concept Drift in IoT Anomaly Detection: Drift Detection, Interpretation, and Adaptation ([9] PI)
8. **Chu et al. (2023)** - Intrusion detection in the IoT data streams using concept drift localization ([17] PI)
9. **Mahdi et al. (2023)** - Enhancing IoT Intrusion Detection System Performance with the Diversity Measure as a Novel Drift Detection Method ([21] PI)
10. **Qiao et al. (2021)** - Concept Drift Analysis by Dynamic Residual Projection for Effectively Detecting Botnet Cyber-Attacks in IoT Scenarios ([53] PI)
11. **Yang et al. (2021)** - PWPAE: An Ensemble Framework for Concept Drift Adaptation in IoT Data Streams ([20] PI)
12. **Palli et al. (2024)** - Online Machine Learning from Non-stationary Data Streams in the Presence of Concept Drift and Class Imbalance ([124] PI)
13. **Stojnev & Stojanovi (2023)** - Concept Drift Detection and Adaptation in IoT Data Stream Analytics ([55] PI)
14. **Wang et al. (2020)** - Concept drift detection with False Positive rate for multi-label classification in IoT data stream ([111] PI)
15. **Cai et al. (2025)** - CDDA-MD: An efficient malicious traffic detection method based on concept drift detection and adaptation technique ([184] PI)

---

## 🏗️ Fase 2: Desenvolvimento Incremental do Core (Meses 4-6)

### Experimento 2.1: Clustering Evolutivo - Implementação Básica
**Objetivo**: Implementar versão simplificada do Mixture of Typicalities
**Duração**: 4-5 semanas

**Hipótese**: Clustering evolutivo pode se adaptar melhor ao concept drift que algoritmos estáticos

**Pipeline**:
- Implementação do algoritmo baseado em Maia et al. [2020]
- Teste em janelas temporais fixas
- Comparação com K-means tradicional e DBSCAN
- Métricas: Silhouette Score, Adjusted Rand Index, tempo de adaptação

**Entregáveis**:
- [ ] Código do clustering evolutivo documentado
- [ ] Comparação com métodos tradicionais
- [ ] Análise de tempo de adaptação a mudanças
- [ ] **Possível publicação**: Artigo sobre clustering evolutivo aplicado a segurança IoT

#### 📚 Referências Bibliográficas - Experimento 2.1

**Referências Essenciais:**
1. **Maia et al. (2020)** - Evolving clustering algorithm based on mixture of typicalities for stream data mining (P)
2. **Li et al. (2022)** - Online Intrusion Detection for Internet of Things Systems With Full Bayesian Possibilistic Clustering and Ensembled Fuzzy Classifiers ([1] PI)
3. **Tahir et al. (2024)** - A clustering-based method for outlier detection under concept drift ([4] PI)
4. **Yin et al. (2017)** - Anomaly detection model based on data stream clustering ([141] PI)
5. **Sitaram et al. (2013)** - Intrusion Detection System for High Volume and High Velocity Packet Streams: A Clustering Approach ([113] PI)

**Referências Complementares:**
6. **Cristiani et al. (2020)** - A Fuzzy Intrusion Detection System for Identifying Cyber-Attacks on IoT Networks ([46] PI)
7. **Talpini et al. (2023)** - A Clustering Strategy for Enhanced FL-Based Intrusion Detection in IoT Networks ([126] PI)
8. **An et al. (2021)** - Edge Intelligence (EI)-Enabled HTTP Anomaly Detection Framework for the Internet of Things (IoT) ([25] PI)
9. **Diallo & Patras (2021)** - Adaptive Clustering-based Malicious Traffic Classification at the Network Edge ([26] PI)
10. **Alhaidari & Zohdy (2019)** - Hybrid Learning Approach of Combining Cluster-Based Partitioning and Hidden Markov Model for IoT Intrusion Detection ([71] PI)
11. **Nguyen et al. (2023)** - Deep Clustering Based Latent Representation for IoT Malware Detection ([116] PI)
12. **Kumar et al. (2016)** - Adaptive Cluster Tendency Visualization and Anomaly Detection for Streaming Data ([162] PI)
13. **Almalawi (2025)** - A Lightweight Intrusion Detection System for Internet of Things: Clustering and Monte Carlo Cross-Entropy Approach ([106] PI)

---

### Experimento 2.2: Arquitetura de Streaming - Prova de Conceito
**Objetivo**: Implementar pipeline básico de streaming para processamento em tempo real
**Duração**: 3-4 semanas

**Hipótese**: Arquitetura de streaming pode processar dados IoT em tempo real mantendo baixa latência

**Pipeline**:
- Setup básico: Kafka + Python consumer
- Simulação de streaming com dados CICIoT2023
- Integração com clustering evolutivo
- Métricas: throughput, latência, utilização de recursos

**Entregáveis**:
- [ ] Arquitetura funcional de streaming
- [ ] Benchmarks de performance
- [ ] Código dockerizado e reproduzível
- [ ] **Possível publicação**: Demo paper sobre arquitetura de streaming para IoT security

#### 📚 Referências Bibliográficas - Experimento 2.2

**Referências Essenciais:**
1. **Surianarayanan et al. (2024)** - A high-throughput architecture for anomaly detection in streaming data using machine learning algorithms (P)
2. **Mohan et al. (2025)** - Distributed Intrusion Detection System using Kafka and Spark Streaming ([10] PI)
3. **Patil et al. (2022)** - SSK-DDoS: distributed stream processing framework based classification system for DDoS attacks ([8] PI)
4. **Rivera et al. (2021)** - An ML Based Anomaly Detection System in real-time data streams ([7] PI)
5. **Singh et al. (2025)** - Streamlined Data Pipeline for Real-Time Threat Detection and Model Inference ([13] PI)
6. **Saleh et al. (2025)** - Streaming-Based Intrusion Detection with Big Data and Online Learning Algorithms ([6] PI)

**Referências Complementares:**
7. **Atbib et al. (2023)** - Design of A Distributed Intrusion Detection System for Streaming Data in IoT Environments ([5] PI)
8. **Saravanan et al. (2024)** - Real-Time Visualization and Detection of Malicious Network Flows in IoT Devices using a Scalable Stream Processing Pipeline ([22] PI)
9. **Hung et al. (2023)** - Network attack classification framework based on Autoencoder model and online stream analysis technology ([24] PI)
10. **G et al. (2024)** - A Real-Time Network Intrusion Detection Based on Transformer-LSTM Model ([33] PI)
11. **Ouhssini et al. (2021)** - Distributed intrusion detection system in the cloud environment based on Apache Kafka and Apache Spark ([28] PI)
12. **Panero et al. (2018)** - Building a large scale Intrusion Detection System using Big Data technologies ([80] PI)
13. **Pwint & Shwe (2019)** - Network Traffic Anomaly Detection based on Apache Spark ([73] PI)
14. **Hafsa & Jemili (2018)** - Comparative Study between Big Data Analysis Techniques in Intrusion Detection ([52] PI)
15. **Chourasia (2025)** - RTASM: An AI-Driven Real-Time Adaptive Streaming Model for Zero-Latency Big Data Processing ([91] PI)

---

## 🚀 Fase 3: Integração e Otimização (Meses 7-9)

### Experimento 3.1: Modelos Focados em Dispositivos
**Objetivo**: Implementar estratégia de modelos específicos por tipo de dispositivo
**Duração**: 4-5 semanas

**Hipótese**: Modelos específicos por dispositivo têm melhor performance que um modelo geral

**Pipeline**:
- Segmentação do dataset por tipo de dispositivo
- Treinamento de modelos específicos usando clustering evolutivo
- Comparação com modelo generalista
- Análise de trade-offs: accuracy vs. complexidade

**Entregáveis**:
- [ ] Framework de modelos por dispositivo
- [ ] Análise comparativa de performance
- [ ] Estratégia de seleção automática de modelos
- [ ] **Possível publicação**: Artigo sobre personalização de IDS para dispositivos IoT heterogêneos

#### 📚 Referências Bibliográficas - Experimento 3.1

**Referências Essenciais:**
1. **Golestani & Makaroff (2024)** - Device-specific anomaly detection models for iot systems (P)
2. **Meidan et al. (2023)** - CADeSH: Collaborative Anomaly Detection for Smart Homes ([51] PI)
3. **Hoang et al. (2022)** - A Data Sampling and Two-Stage Convolution Neural Network for IoT Devices Identification ([105] PI)
4. **Yu et al. (2021)** - RADAR: A Robust Behavioral Anomaly Detection for IoT Devices in Enterprise Networks ([157] PI)
5. **Zhou et al. (2024)** - HEDVA: Harnessing HTTP Traffic for Enhanced Detection of Vulnerability Attacks in IoT Networks ([47] PI)

**Referências Complementares:**
6. **Tan et al. (2024)** - FlowSpotter: Intelligent IoT Threat Detection via Imaging Network Flows ([60] PI)
7. **Wu et al. (2024)** - Intrusion Detection for Unmanned Aerial Vehicles Security: A Tiny Machine Learning Model ([114] PI)
8. **Wang & Wu (2024)** - Intrusion detection for internet of things security: a hidden Markov model based on fuzzy rough set ([62] PI)
9. **Do et al. (2021)** - An Efficient Feature Extraction Method for Attack Classification in IoT Networks ([64] PI)
10. **Yu et al. (2024)** - Novel Intrusion Detection Strategies With Optimal Hyper Parameters for Industrial Internet of Things ([61] PI)
11. **Chaganti (2025)** - A Scalable, Lightweight AI-Driven Security Framework for IoT Ecosystems ([108] PI)
12. **Hızal et al. (2024)** - IoT-based Smart Home Security System with Machine Learning Models ([102] PI)
13. **Esmaeili et al. (2024)** - Machine Learning-Assisted Intrusion Detection for Enhancing Internet of Things Security ([109] PI)
14. **Uddin et al. (2024)** - A Dual-Tier Adaptive One-Class Classification IDS for Emerging Cyberthreats ([31] PI)

---

### Experimento 3.2: Arquitetura em Duas Fases
**Objetivo**: Implementar sistema completo com detecção + classificação
**Duração**: 5-6 semanas

**Hipótese**: Arquitetura em duas fases (anomalia + classificação) oferece melhor trade-off latência/precisão

**Pipeline**:
- Fase 1: Clustering evolutivo para detecção de anomalias
- Fase 2: Random Forest/CNN para classificação de tipos de ataque
- Balanceamento de carga entre as fases
- Otimização de thresholds adaptativos

**Entregáveis**:
- [ ] Sistema completo integrado
- [ ] Análise de trade-offs latência/precisão
- [ ] Benchmarks comparativos com sistemas monofásicos
- [ ] **Possível publicação**: Artigo principal sobre arquitetura híbrida

#### 📚 Referências Bibliográficas - Experimento 3.2

**Referências Essenciais:**
1. **Park (2018)** - Anomaly pattern detection on data streams (P)
2. **Alani & Awad (2023)** - An Intelligent Two-Layer Intrusion Detection System for the Internet of Things ([65] PI)
3. **Ogobuchi (2022)** - Multi-phase optimized intrusion detection system based on deep learning algorithms for computer networks (P)
4. **Hung et al. (2023)** - Network attack classification framework based on Autoencoder model and online stream analysis technology ([24] PI)
5. **Zhang et al. (2024)** - AOC-IDS: Autonomous Online Framework with Contrastive Learning for Intrusion Detection ([198] PI)

**Referências Complementares:**
6. **Huang & Zhang (2025)** - An Online Intrusion Detection Method using Adaptive Multi-Level Classifier Network and PCA-Guided Model Reuse Mechanism ([95] PI)
7. **Jemili et al. (2024)** - Intrusion detection based on concept drift detection and online incremental learning ([189] PI)
8. **Shyaa et al. (2023)** - Enhanced Intrusion Detection with Data Stream Classification and Concept Drift Guided by the Incremental Learning Genetic Programming Combiner ([34] PI)
9. **Lu et al. (2025)** - Network Intrusion Detection for Modern Smart Grids Based on Adaptive Online Incremental Learning ([63] PI)
10. **Tian et al. (2024)** - Mix-CL: Semi-Supervised Continual Learning for Network Intrusion Detection ([94] PI)
11. **Simioni et al. (2025)** - An Energy-Efficient Intrusion Detection Offloading Based on DNN for Edge Computing ([70] PI)
12. **Horchulhack et al. (2022)** - A Stream Learning Intrusion Detection System for Concept Drifting Network Traffic ([14] PI)
13. **Chouchen & Jemili (2023)** - Intrusion detection based on Incremental Learning ([44] PI)
14. **Cerasuolo et al. (2025)** - Attack-adaptive network intrusion detection systems for IoT networks through class incremental learning ([36] PI)
15. **Abderrahim & Benosman (2025)** - Adaptive intrusion detection in IoT: combining batch and incremental learning for enhanced security ([37] PI)

---

## 🎯 Fase 4: Validação e Otimização Final (Meses 10-12)

### Experimento 4.1: Avaliação em Múltiplos Datasets
**Objetivo**: Validar generalização do sistema em diferentes ambientes IoT
**Duração**: 3-4 semanas

**Hipótese**: O sistema mantém performance consistente em diferentes datasets IoT

**Pipeline**:
- Teste em CICIoT2023, CICIDS2017, CICIDS2018
- Análise de transferência entre domínios
- Identificação de limitações e bias
- Validação estatística (testes t, ANOVA)

**Entregáveis**:
- [ ] Relatório de generalização
- [ ] Análise estatística robusta
- [ ] Identificação de casos edge
- [ ] **Possível publicação**: Artigo sobre robustez e generalização

#### 📚 Referências Bibliográficas - Experimento 4.1

**Referências Essenciais:**
1. **Ferrag et al. (2022)** - Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications ([112] PI)
2. **Guo et al. (2023)** - An IoT Intrusion Detection System Based on TON IoT Network Dataset ([89] PI)
3. **Akif et al. (2025)** - Hybrid Machine Learning Models for Intrusion Detection in IoT: Leveraging a Real-World IoT Dataset ([132] PI)
4. **Khan et al. (2023)** - A hybrid deep learning-based intrusion detection system for IoT networks ([50] PI)
5. **Sharma & Bairwa (2025)** - Leveraging AI for Intrusion Detection in IoT Ecosystems: A Comprehensive Study ([140] PI)

**Referências Complementares:**
6. **Tahir et al. (2025)** - A systematic review of machine learning and deep learning techniques for anomaly detection in data mining ([165] PI)
7. **Cao et al. (2024)** - Revisiting streaming anomaly detection: benchmark and evaluation ([72] PI)
8. **Shahraki et al. (2022)** - A comparative study on online machine learning techniques for network traffic streams analysis ([45] PI)
9. **Munir et al. (2019)** - A Comparative Analysis of Traditional and Deep Learning-Based Anomaly Detection Methods for Streaming Data ([84] PI)
10. **Ray & Susan (2024)** - Performance Analysis of Online Machine Learning Frameworks for Anomaly Detection in IoT Data Streams ([57] PI)
11. **Alqahtany et al. (2025)** - Enhanced grey wolf optimization (egwo) and random forest based mechanism for intrusion detection in iot networks (P)
12. **Souiden et al. (2022)** - A survey of outlier detection in high dimensional data streams ([133] PI)
13. **Shyaa et al. (2024)** - Evolving cybersecurity frontiers: A comprehensive survey on concept drift and feature dynamics aware machine and deep learning in intrusion detection systems ([54] PI)

---

### Experimento 4.2: Benchmarking e Otimização de Performance
**Objetivo**: Otimizar sistema para deployment em produção
**Duração**: 4-5 semanas

**Hipótese**: Sistema pode atender requisitos de latência de aplicações IoT críticas (<100ms)

**Pipeline**:
- Profiling detalhado de performance
- Otimizações de código (vectorização, paralelização)
- Testes de escalabilidade (1K, 10K, 100K devices)
- Comparação com sistemas comerciais/acadêmicos

**Entregáveis**:
- [ ] Sistema otimizado para produção
- [ ] Benchmarks completos de performance
- [ ] Documentação de deployment
- [ ] **Possível publicação**: Paper sobre performance e escalabilidade

#### 📚 Referências Bibliográficas - Experimento 4.2

**Referências Essenciais:**
1. **Sharma et al. (2024)** - Explainable artificial intelligence for intrusion detection in iot networks: A deep learning based approach (P)
2. **Chen et al. (2025)** - Intrusion detection using synaptic intelligent convolutional neural networks for dynamic internet of things environments (P)
3. **Nguyen et al. (2019)** - DÏot: A federated self-learning anomaly detection system for iot (P)
4. **Olanrewaju-George & Pranggono (2025)** - Federated learning-based intrusion detection system for the internet of things using unsupervised and supervised deep learning models (P)
5. **Nayak (2025)** - Scalable Anomaly Detection with Machine Learning: Techniques for Managing High-Dimensional Data Streams ([166] PI)

**Referências Complementares:**
6. **Illy et al. (2019)** - Securing Fog-to-Things Environment Using Intrusion Detection System Based On Ensemble Learning ([104] PI)
7. **Lalouani & Younis (2021)** - Robust Distributed Intrusion Detection System for Edge of Things ([58] PI)
8. **Nixon et al. (2019)** - Practical Application of Machine Learning based Online Intrusion Detection to Internet of Things Networks ([35] PI)
9. **Agbedanu et al. (2022)** - Towards achieving lightweight intrusion detection systems in Internet of Things, the role of incremental machine learning ([48] PI)
10. **Raj et al. (2023)** - Knox: Lightweight Machine Learning Approaches for Automated Detection of Botnet Attacks ([12] PI)
11. **Zolanvari et al. (2021)** - ADDAI: Anomaly Detection using Distributed AI ([167] PI)
12. **Garg et al. (2019)** - A Hybrid Deep Learning-Based Model for Anomaly Detection in Cloud Datacenter Networks ([168] PI)
13. **Lee et al. (2020)** - ReRe: A Lightweight Real-Time Ready-to-Go Anomaly Detection Approach for Time Series ([169] PI)
14. **Mehnaz & Bertino (2020)** - Privacy-preserving Real-time Anomaly Detection Using Edge Computing ([171] PI)
15. **Rosenberger et al. (2021)** - Perspective on efficiency enhancements in processing streaming data in industrial IoT networks ([173] PI)
16. **Sun et al. (2019)** - Fast Anomaly Detection in Multiple Multi-Dimensional Data Streams ([175] PI)

---

## 📖 Referências Sugeridas para Contexto Geral

### Trabalhos Fundamentais e Reviews Abrangentes

**Referências de Alta Relevância:**
1. **Al-garadi et al. (2018)** - A Survey of Machine and Deep Learning Methods for Internet of Things (IoT) Security ([144] PI)
2. **Hajiheidari et al. (2019)** - Intrusion detection systems in the Internet of things: A comprehensive investigation ([142] PI)
3. **Thamilarasu & Chawla (2019)** - Towards Deep-Learning-Driven Intrusion Detection for the Internet of Things ([143] PI)
4. **Koroniotis et al. (2020)** - A new network forensic framework based on deep learning for Internet of Things networks ([145] PI)
5. **Arisdakessian et al. (2023)** - A Survey on IoT Intrusion Detection: Federated Learning, Game Theory, Social Psychology, and Explainable AI as Future Directions ([134] PI)
6. **Aldhaheri et al. (2023)** - Deep learning for cyber threat detection in IoT networks: A review ([117] PI)
7. **Ahsan et al. (2025)** - A systematic review of metaheuristics-based and machine learning-driven intrusion detection systems in IoT ([123] PI)

**Métodos Emergentes e Inovações:**
8. **Bhatia et al. (2021)** - MemStream: Memory-Based Streaming Anomaly Detection ([38] PI)
9. **Yoon et al. (2022)** - Adaptive Model Pooling for Online Deep Anomaly Detection from a Complex Evolving Data Stream ([79] PI)
10. **Miller et al. (2011)** - Anomalous Network Packet Detection Using Data Stream Mining ([18] PI)
11. **Ma et al. (2020)** - AESMOTE: Adversarial Reinforcement Learning With SMOTE for Anomaly Detection ([97] PI)
12. **Gueriani et al. (2023)** - Deep Reinforcement Learning for Intrusion Detection in IoT: A Survey ([139] PI)
13. **Wang et al. (2024)** - A Few-Shot and Anti-Forgetting Network Intrusion Detection System based on Online Meta Learning ([29] PI)

**Arquiteturas e Deployment:**
14. **Rjoub et al. (2022)** - Trust-Augmented Deep Reinforcement Learning for Federated Learning Client Selection ([74] PI)
15. **Li et al. (2022)** - Federated Anomaly Detection on System Logs for the Internet of Things: A Customizable and Communication-Efficient Approach ([86] PI)
16. **Rathee et al. (2023)** - TrustBlkSys: A Trusted and Blockchained Cybersecure System for IIoT ([68] PI)
17. **Abid et al. (2023)** - Real-time data fusion for intrusion detection in industrial control systems based on cloud computing and big data techniques ([66] PI)

**Datasets e Avaliação:**
18. **Ullah et al. (2023)** - TNN-IDS: Transformer neural network-based intrusion detection system for MQTT-enabled IoT Networks ([42] PI)
19. **Widanage et al. (2019)** - Anomaly Detection over Streaming Data: Indy500 Case Study ([160] PI)
20. **Clémençon et al. (2018)** - A secure IoT architecture for streaming data analysis and anomaly detection ([158] PI)

### Trabalhos Específicos de Interesse

**Deep Learning e Anomaly Detection:**
21. **Ge et al. (2019)** - Deep Learning-Based Intrusion Detection for IoT Networks ([92] PI)
22. **Ahmed et al. (2023)** - DLA-ABIDS:Deep Learning Approach for Anomaly Based Intrusion Detection System ([87] PI)
23. **Anandaraj et al. (2023)** - ROAST-IoT: A Novel Range-Optimized Attention Convolutional Scattered Technique for Intrusion Detection in IoT Networks ([75] PI)
24. **Ullah et al. (2025)** - Hybrid Machine Learning Models for Intrusion Detection in IoT Networks ([50] PI)

**Streaming e Time Series:**
25. **Raeiszadeh et al. (2023)** - A Deep Learning Approach for Real-Time Application-Level Anomaly Detection in IoT Data Streaming ([148] PI)
26. **Le et al. (2023)** - VEAD: Variance profile Exploitation for Anomaly Detection in real-time IoT data streaming ([149] PI)
27. **Raeiszadeh et al. (2024)** - Real-Time Adaptive Anomaly Detection in Industrial IoT Environments ([150] PI)
28. **Shao et al. (2023)** - Low-Latency Dimensional Expansion and Anomaly Detection Empowered Secure IoT Network ([151] PI)
29. **Nizam et al. (2022)** - Real-Time Deep Anomaly Detection Framework for Multivariate Time-Series Data in Industrial IoT ([153] PI)

**Outros Métodos Relevantes:**
30. **Jolliffe (2003)** - Principal Component Analysis ([146] PI)
31. **Elmoutaoukkil et al. (2024)** - Network intrusion detection in big datasets using Spark environment and incremental learning ([15] PI)
32. **Nithish et al. (2021)** - Real-Time Anomaly Detection Using Facebook Prophet ([170] PI)
33. **Coughlin & Perrone (2017)** - Multi-scale Anomaly Detection with Wavelets ([178] PI)
34. **Basheer et al. (2021)** - Detecting Anomaly in IoT Devices using Multi-Threaded Autonomous Anomaly Detection ([156] PI)

---

## 📊 Sistema de Tracking e Documentação

### Ferramentas Obrigatórias
- **MLflow**: Tracking de experimentos, métricas e modelos
- **DVC**: Versionamento de dados e pipelines
- **Git**: Versionamento de código com tags por experimento
- **Docker**: Containerização para reprodutibilidade
- **Jupyter Notebooks**: Documentação interativa

### Estrutura de Documentação por Experimento
```
experimentos/
├── experimento-X.Y/
│   ├── README.md                 # Objetivo, hipótese, metodologia
│   ├── notebook.ipynb           # Análise e resultados
│   ├── src/                     # Código específico
│   ├── data/                    # Dados processados
│   ├── results/                 # Métricas e visualizações
│   ├── docker-compose.yml       # Ambiente reproduzível
│   └── CONCLUSIONS.md           # Conclusões e próximos passos
```

### Métricas de Acompanhamento
- **Performance**: Accuracy, Precision, Recall, F1, AUC-ROC
- **Operational**: Latência, Throughput, Utilização de recursos
- **Robustez**: Performance cross-dataset, stability over time
- **Reprodutibilidade**: Tempo para setup, facilidade de replicação

---

## 📊 Resumo das Referências Bibliográficas

### Estatísticas por Fonte
- **Proposta de Mestrado (P)**: 30 referências diretas
- **Pré-pesquisa Undermind (PI)**: 198 referências indexadas
- **Total de Referências Únicas**: ~220 trabalhos catalogados

### Distribuição por Fase
- **Fase 1 (Fundamentos)**: 29 referências (14 essenciais + 15 complementares)
- **Fase 2 (Core Development)**: 28 referências (11 essenciais + 17 complementares)  
- **Fase 3 (Integração)**: 29 referências (10 essenciais + 19 complementares)
- **Fase 4 (Validação)**: 29 referências (10 essenciais + 19 complementares)
- **Contexto Geral**: 34 referências de apoio

### Cobertura Temática
✅ **Clustering Evolutivo e Mixture of Typicalities**: Bem coberto  
✅ **Concept Drift em IoT**: Extensivamente coberto  
✅ **Arquiteturas de Streaming (Kafka/Spark)**: Bem coberto  
✅ **Datasets IoT (CICIoT2023, CICIDS)**: Adequadamente coberto  
✅ **Detecção de Anomalias Clássica**: Bem coberto  
✅ **Modelos Device-Specific**: Adequadamente coberto  
✅ **Performance e Escalabilidade**: Bem coberto  
✅ **Deep Learning para IoT IDS**: Extensivamente coberto  

### Qualidade das Fontes
- **Journals de Alto Impacto**: IEEE Transactions, Computer Networks, Future Generation Computer Systems
- **Conferências Top-Tier**: IEEE INFOCOM, ICDCS, GLOBECOM, ICDE
- **Workshops Especializados**: IoT Security, Streaming Data, Anomaly Detection
- **Datasets Reconhecidos**: CICIoT2023, Edge-IIoTset, TON_IoT, Bot-IoT

---

## 🔄 Pontos de Decisão e Ajuste de Curso

### Checkpoint 1 (Final do Mês 3)
**Critérios de Sucesso**:
- Baseline estabelecido com métricas >80% F1-Score
- Concept drift identificado e quantificado
- Primeira publicação submetida/aceita

**Possíveis Ajustes**:
- Se performance baseline for muito baixa → focar em feature engineering
- Se concept drift for mínimo → reduzir foco em adaptação evolutiva
- Se datasets forem inadequados → buscar datasets alternativos

### Checkpoint 2 (Final do Mês 6)
**Critérios de Sucesso**:
- Clustering evolutivo funcionando e superando baselines
- Arquitetura de streaming processando >1000 eventos/seg
- Segunda publicação em progresso

**Possíveis Ajustes**:
- Se clustering não superar baseline → investigar híbridos
- Se streaming for complexo demais → focar em batch processing otimizado
- Se recursos computacionais forem limitados → simplificar arquitetura

### Checkpoint 3 (Final do Mês 9)
**Critérios de Sucesso**:
- Sistema integrado funcionando end-to-end
- Performance competitiva com estado da arte
- Terceira publicação submetida

**Possíveis Ajustes**:
- Se integração for complexa → focar em componentes isolados
- Se performance não for competitiva → investigar ensemble methods
- Se tempo for limitado → priorizar análise teórica vs. implementação

---

## 📈 Cronograma de Publicações Alvo

| Mês | Experimento | Tipo de Publicação | Venue Alvo |
|-----|-------------|-------------------|-------------|
| 3 | 1.1, 1.2 | Workshop Paper | SBRC/WebMedia Workshop |
| 6 | 2.1 | Conference Paper | BRACIS/ENIAC |
| 9 | 3.1, 3.2 | Journal Paper | JISA/Computers & Security |
| 12 | 4.1, 4.2 | Main Dissertation + Conference | Thesis + ICML/NeurIPS |

---

## 🛠️ Setup Inicial Recomendado

### Ambiente de Desenvolvimento
```bash
# Estrutura do projeto
mkdir iot-ids-research
cd iot-ids-research
git init
dvc init

# Ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# MLflow tracking
mlflow ui --backend-store-uri ./mlruns

# Jupyter lab
jupyter lab
```

### Ferramentas Essenciais
- Python 3.8+, scikit-learn, pandas, numpy
- TensorFlow/PyTorch para deep learning
- Kafka, Docker, Docker Compose
- MLflow, DVC, Git LFS
- Jupyter, matplotlib, seaborn, plotly

---

## 🎯 Métricas de Sucesso do Planejamento

1. **Publicações**: Mínimo 3 publicações (1 workshop + 1 conference + 1 journal)
2. **Reprodutibilidade**: 100% dos experimentos reproduzíveis via Docker
3. **Performance**: Sistema final com latência <100ms e F1-Score >90%
4. **Impacto**: Código open-source com documentação completa
5. **Transferência**: Resultados aplicáveis a outros domínios IoT

Este planejamento garante progresso incremental, resultados publicáveis e flexibilidade para ajustes, mantendo o foco nos objetivos da dissertação enquanto constrói uma base sólida de conhecimento científico.