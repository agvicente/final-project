# Cronograma Detalhado - Fase 1: Fundamentos e MVP
## Detecção de Intrusão Baseada em Anomalias em Sistemas IoT com Clustering Evolutivo

### 📋 Visão Geral da Fase 1
**Duração Total**: 12 semanas (3 meses)  
**Objetivo Principal**: Estabelecer fundamentos sólidos e criar MVP publicável  
**Meta de Publicação**: 1 workshop paper + 1 conference paper  

---

## 🎯 Objetivos Estratégicos da Fase 1

### Objetivos Primários
1. **Estabelecer Baseline Científico**: Criar linha de base robusta para comparações futuras
2. **Validar Metodologia de Amostragem**: Demonstrar representatividade estatística da amostra
3. **Identificar Concept Drift**: Quantificar mudanças temporais em dados IoT
4. **Preparar Base para Fase 2**: Garantir transição suave para clustering evolutivo

### Objetivos Secundários
1. **Setup Completo do Ambiente**: Infraestrutura reproduzível e escalável
2. **Documentação Científica**: Metodologia transparente e replicável
3. **Validação de Hipóteses**: Confirmação de pressupostos para trabalho futuro
4. **Networking Acadêmico**: Primeira publicação para estabelecer presença

---

## 📊 Metodologia de Amostragem Científica

### Estratégia de Amostragem para CICIoT2023

#### Tamanho da Amostra
- **Amostra Principal**: 10% do dataset completo (~2.3M registros de 23M)
- **Justificativa Estatística**: 
  - Margem de erro: ±0.3% (confiança 95%)
  - Power analysis para detectar diferenças de ≥5% entre classes
  - Baseado em Cochran (1977) para populações finitas

#### Método de Seleção
```python
# Estratificação por:
# 1. Tipo de ataque (proporção mantida)
# 2. Tipo de dispositivo (distribuição preservada)  
# 3. Período temporal (coverage de 24h)
# 4. Volume de tráfego (picos e vales representados)

stratified_sample = {
    'normal': 70%,      # ~1.6M registros
    'ddos': 15%,        # ~345K registros  
    'mirai': 8%,        # ~184K registros
    'recon': 4%,        # ~92K registros
    'spoofing': 2%,     # ~46K registros
    'mitm': 1%          # ~23K registros
}
```

#### Validação da Representatividade
1. **Teste Kolmogorov-Smirnov**: Comparar distribuições amostra vs. população
2. **Teste Chi-quadrado**: Verificar independência temporal
3. **Análise de Componentes Principais**: Preservação da variância
4. **Bootstrap Sampling**: Estabilidade dos resultados (n=1000)

#### Documentação da Limitação
```
LIMITAÇÕES EXPLÍCITAS:
- Resultados podem variar com dataset completo
- Ataques raros podem estar sub-representados
- Padrões sazonais longos podem não aparecer
- Validação futura necessária em escala completa
```

---

## 📅 Cronograma Semanal Detalhado

### **Semanas 1-2: Setup e Preparação**

#### Semana 1: Infraestrutura e Ambiente
**Dias 1-2: Setup do Ambiente**
- [ ] Configuração do workspace Python (venv, requirements.txt)
- [ ] Setup MLflow + DVC para tracking
- [ ] Configuração Docker para reprodutibilidade
- [ ] Setup Jupyter Lab com extensões necessárias

**Dias 3-4: Aquisição e Preparação de Dados**
- [ ] Download do CICIoT2023 dataset
- [ ] Análise exploratória inicial (shape, missing values, distributions)
- [ ] Implementação da estratégia de amostragem estratificada
- [ ] Validação estatística da representatividade

**Dia 5: Documentação e Versionamento**
- [ ] Setup Git LFS para dados grandes
- [ ] Criação da estrutura de pastas do projeto
- [ ] Documentação inicial da metodologia
- [ ] Primeiro commit com ambiente funcionando

**Ferramentas**: Python 3.9+, pandas, numpy, scikit-learn, MLflow, DVC, Docker

#### Semana 2: Análise Exploratória Profunda
**Dias 1-2: EDA Quantitativa**
- [ ] Estatísticas descritivas completas
- [ ] Análise de correlações e feature importance
- [ ] Detecção de outliers e dados anômalos
- [ ] Visualizações de distribuições temporais

**Dias 3-4: EDA Qualitativa**
- [ ] Análise de padrões de tráfego por tipo de dispositivo
- [ ] Identificação de características específicas por ataque
- [ ] Análise de sazonalidade e tendências temporais
- [ ] Mapeamento de features mais discriminativas

**Dia 5: Relatório EDA**
- [ ] Notebook completo com findings
- [ ] Visualizações profissionais para publicação
- [ ] Identificação de challenges específicos
- [ ] Preparação para próxima fase

**Leituras Obrigatórias**:
- Neto et al. (2023) - CICIoT2023 dataset paper
- Cook et al. (2020) - Anomaly detection for IoT time-series: A survey
- Benkhelifa et al. (2018) - Critical review of IDS practices in IoT

---

### **Semanas 3-6: Experimento 1.1 - Baseline de Detecção de Anomalias**

#### Semana 3: Pré-processamento e Feature Engineering
**Dias 1-2: Limpeza e Normalização**
- [ ] Implementação de pipeline de limpeza robusto
- [ ] Tratamento de valores missing (estratégias múltiplas)
- [ ] Normalização e padronização (StandardScaler, MinMaxScaler)
- [ ] Análise de impacto das transformações

**Dias 3-4: Feature Engineering**
- [ ] Criação de features temporais (hora, dia da semana, etc.)
- [ ] Features de agregação (rolling statistics)
- [ ] Encoding de variáveis categóricas
- [ ] Feature selection baseada em métrica F1

**Dia 5: Validação do Pipeline**
- [ ] Testes de robustez do pipeline
- [ ] Validação com dados sintéticos
- [ ] Documentação completa do processo
- [ ] Benchmark de tempo de processamento

**Ferramentas**: sklearn.preprocessing, pandas.tools, feature-engine

#### Semana 4: Implementação dos Algoritmos Baseline
**Dias 1-2: Isolation Forest**
- [ ] Implementação com hyperparameter tuning
- [ ] Grid search para parâmetros ótimos
- [ ] Análise de sensibilidade a outliers
- [ ] Estudo de escalabilidade temporal

**Dias 3-4: One-Class SVM e LOF**
- [ ] Implementação e tuning de One-Class SVM
- [ ] Local Outlier Factor com otimização
- [ ] Comparação de kernels (RBF, linear, polynomial)
- [ ] Análise de complexidade computacional

**Dia 5: Integração e Testes**
- [ ] Pipeline unificado para os 3 algoritmos
- [ ] Testes de performance e memória
- [ ] Validação cruzada temporal
- [ ] Preparação para avaliação sistemática

**Leituras Obrigatórias**:
- Liu et al. (2008) - Isolation Forest
- Liu et al. (2012) - Isolation-Based Anomaly Detection
- Laskar et al. (2021) - Extending Isolation Forest via K-Means

#### Semana 5: Avaliação Sistemática
**Dias 1-2: Métricas Clássicas**
- [ ] Implementação de métricas robustas (Precision, Recall, F1)
- [ ] ROC-AUC e PR-AUC para classes desbalanceadas
- [ ] Confusion matrices detalhadas por classe
- [ ] Análise de erros e casos edge

**Dias 3-4: Métricas Específicas para IoT**
- [ ] Taxa de falsos positivos por tipo de dispositivo
- [ ] Latência de detecção (tempo real simulado)
- [ ] Robustez a variações de tráfego
- [ ] Análise de performance por período temporal

**Dia 5: Análise Comparativa**
- [ ] Ranking estatístico dos algoritmos
- [ ] Testes de significância (t-test, Wilcoxon)
- [ ] Análise de trade-offs (accuracy vs. speed)
- [ ] Identificação do melhor baseline

**Ferramentas**: sklearn.metrics, scipy.stats, seaborn, matplotlib

#### Semana 6: Documentação e Primeira Publicação
**Dias 1-2: Análise de Resultados**
- [ ] Interpretação estatística dos resultados
- [ ] Identificação de limitações e bias
- [ ] Comparação com trabalhos relacionados
- [ ] Insights para melhorias futuras

**Dias 3-4: Redação Científica**
- [ ] Abstract e introdução para workshop paper
- [ ] Metodologia detalhada e reproduzível
- [ ] Seção de resultados com visualizações
- [ ] Discussão crítica e trabalhos futuros

**Dia 5: Submissão e Código**
- [ ] Revisão final do paper
- [ ] Submissão para workshop (SBRC/WebMedia)
- [ ] Release do código no GitHub
- [ ] Tag Git para experimento 1.1

**Meta de Publicação**: Workshop paper sobre "Comparative Analysis of Classical Anomaly Detection in IoT: A Systematic Evaluation on CICIoT2023 Dataset"

---

### **Semanas 7-10: Experimento 1.2 - Análise de Concept Drift**

#### Semana 7: Fundamentação Teórica e Setup
**Dias 1-2: Literatura Específica**
- [ ] Estudo detalhado de Lu et al. (2019) - Learning under Concept Drift
- [ ] Análise de Wahab (2022) - Concept Drift in IoT IDS
- [ ] Review de métodos de detecção de drift
- [ ] Identificação de gaps na literatura IoT

**Dias 3-4: Implementação de Detectores**
- [ ] ADWIN (Adaptive Windowing) implementation
- [ ] DDM (Drift Detection Method) implementation
- [ ] Page-Hinkley test para mudanças abruptas
- [ ] KSWIN para drift gradual

**Dia 5: Validação com Dados Sintéticos**
- [ ] Criação de datasets com drift conhecido
- [ ] Validação da sensibilidade dos detectores
- [ ] Calibração de thresholds
- [ ] Análise de falsos positivos

**Ferramentas**: river (online ML), scipy.stats, numpy

#### Semana 8: Análise Temporal do Dataset
**Dias 1-2: Segmentação Temporal**
- [ ] Divisão do dataset em janelas temporais (1h, 6h, 24h)
- [ ] Análise de stationaridade (ADF test, KPSS test)
- [ ] Identificação de pontos de mudança
- [ ] Caracterização de tipos de drift

**Dias 3-4: Análise Estatística de Drift**
- [ ] Teste de mudança de distribuição (KS test)
- [ ] Análise de componentes principais temporal
- [ ] Medidas de divergência (KL, Wasserstein)
- [ ] Quantificação da magnitude do drift

**Dia 5: Visualização e Interpretação**
- [ ] Heatmaps de drift por feature
- [ ] Time series plots de métricas de drift
- [ ] Correlação entre drift e eventos externos
- [ ] Dashboard interativo para exploração

**Leituras Obrigatórias**:
- Xu et al. (2023) - ADTCD: Adaptive Anomaly Detection for IoT
- Bharani et al. (2024) - Comparative Study of Drift Detection
- Yang & Shami (2021) - Lightweight Concept Drift Detection Framework

#### Semana 9: Impacto do Drift nos Modelos Baseline
**Dias 1-2: Avaliação Temporal**
- [ ] Re-avaliação dos baselines em janelas temporais
- [ ] Análise de degradação de performance
- [ ] Identificação de períodos críticos
- [ ] Correlação entre drift e accuracy

**Dias 3-4: Estratégias de Adaptação**
- [ ] Re-training incremental dos modelos
- [ ] Avaliação de janelas de adaptação
- [ ] Comparação: modelo estático vs. adaptativo
- [ ] Análise de custo-benefício da adaptação

**Dia 5: Síntese e Insights**
- [ ] Caracterização completa do concept drift em IoT
- [ ] Recomendações para sistemas adaptativos
- [ ] Identificação de research gaps
- [ ] Preparação para clustering evolutivo

#### Semana 10: Paper de Concept Drift
**Dias 1-3: Redação Científica**
- [ ] Introdução: problema do concept drift em IoT
- [ ] Metodologia: detectores e métricas utilizadas
- [ ] Resultados: caracterização quantitativa do drift
- [ ] Discussão: implicações para IDS adaptativos

**Dias 4-5: Revisão e Submissão**
- [ ] Revisão técnica e linguística
- [ ] Submissão para conferência (BRACIS/ENIAC)
- [ ] Preparação de slides para apresentação
- [ ] Atualização do repositório GitHub

**Meta de Publicação**: Conference paper sobre "Characterizing Concept Drift in IoT Traffic: Implications for Adaptive Intrusion Detection Systems"

---

### **Semanas 11-12: Consolidação e Transição para Fase 2**

#### Semana 11: Integração e Análise Global
**Dias 1-2: Síntese dos Experimentos**
- [ ] Integração dos resultados dos experimentos 1.1 e 1.2
- [ ] Análise holística: baseline + concept drift
- [ ] Identificação de sinergias e contradições
- [ ] Refinamento das hipóteses para Fase 2

**Dias 3-4: Preparação para Clustering Evolutivo**
- [ ] Análise de requisitos para algoritmos adaptativos
- [ ] Identificação de features mais estáveis ao drift
- [ ] Estudo preliminar de Maia et al. (2020) - Mixture of Typicalities
- [ ] Design inicial da arquitetura evolutiva

**Dia 5: Validação da Base Teórica**
- [ ] Review completo da literatura consultada
- [ ] Identificação de gaps para Fase 2
- [ ] Planejamento de leituras complementares
- [ ] Mapeamento de colaborações potenciais

#### Semana 12: Documentação Final e Setup Fase 2
**Dias 1-2: Relatório Técnico Completo**
- [ ] Documento técnico consolidado da Fase 1
- [ ] Metodologia reproduzível documentada
- [ ] Dataset e código versionados (DVC + Git)
- [ ] Lessons learned e best practices

**Dias 3-4: Transição para Fase 2**
- [ ] Planejamento detalhado do Experimento 2.1
- [ ] Setup inicial para clustering evolutivo
- [ ] Preparação do ambiente de streaming
- [ ] Review do cronograma da Fase 2

**Dia 5: Checkpoint Final**
- [ ] Avaliação dos objetivos alcançados
- [ ] Identificação de desvios e ajustes necessários
- [ ] Comunicação de resultados aos orientadores
- [ ] Kick-off da Fase 2

---

## 🛠️ Ferramentas e Tecnologias por Categoria

### **Ambiente de Desenvolvimento**
```bash
# Core Python Stack
python==3.9.16
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.2.2
matplotlib==3.7.1
seaborn==0.12.2
jupyter==1.0.0

# Machine Learning
isolation-forest==0.5.1
river==0.15.0  # Online ML library
imbalanced-learn==0.10.1

# Experiment Tracking
mlflow==2.3.1
dvc[all]==2.58.2
wandb==0.15.3

# Data Validation
great-expectations==0.16.4
pandas-profiling==3.6.6

# Development Tools
pytest==7.3.1
black==23.3.0
flake8==6.0.0
pre-commit==3.3.2
```

### **Infraestrutura**
- **Containerização**: Docker + Docker Compose
- **Versionamento**: Git + Git LFS para datasets
- **CI/CD**: GitHub Actions para testes automatizados
- **Storage**: Local + Google Drive backup
- **Compute**: Jupyter Lab + Google Colab Pro (para experimentos pesados)

### **Métricas e Avaliação**
```python
# Métricas Primárias
metrics = {
    'classification': ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'auc_pr'],
    'drift_detection': ['drift_magnitude', 'detection_delay', 'false_positive_rate'],
    'computational': ['training_time', 'inference_time', 'memory_usage'],
    'robustness': ['cross_validation_std', 'temporal_stability', 'noise_resistance']
}
```

---

## 📚 Bibliografia Organizada por Experimento

### **Experimento 1.1: Baseline de Detecção de Anomalias**

#### Leituras Essenciais (Semanas 1-3)
1. **Neto et al. (2023)** - CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment
   - *Foco*: Características do dataset, metodologia de coleta, benchmarks existentes
   - *Aplicação*: Fundamentação para escolha do dataset e comparação com trabalhos anteriores

2. **Liu et al. (2008)** - Isolation Forest  
   - *Foco*: Algoritmo base, complexidade computacional, casos de uso
   - *Aplicação*: Implementação correta e otimizada do Isolation Forest

3. **Benkhelifa et al. (2018)** - A Critical Review of Practices and Challenges in Intrusion Detection Systems for IoT
   - *Foco*: Estado da arte em IDS para IoT, challenges específicos
   - *Aplicação*: Contextualização do problema e justificativa da abordagem

#### Leituras Complementares (Semanas 4-6)
4. **Cook et al. (2020)** - Anomaly detection for iot time-series data: A survey
   - *Foco*: Métodos específicos para time series IoT
   - *Aplicação*: Adaptações necessárias para dados temporais

5. **Ahmad et al. (2021)** - Network intrusion detection system: A systematic study of machine learning and deep learning approaches
   - *Foco*: Comparação sistemática de abordagens ML/DL
   - *Aplicação*: Positioning do trabalho no contexto atual

6. **Laskar et al. (2021)** - Extending Isolation Forest for Anomaly Detection in Big Data via K-Means
   - *Foco*: Melhorias do Isolation Forest para big data
   - *Aplicação*: Possíveis otimizações para o algoritmo baseline

### **Experimento 1.2: Análise de Concept Drift**

#### Leituras Essenciais (Semanas 7-8)
1. **Lu et al. (2019)** - Learning under Concept Drift: A Review
   - *Foco*: Taxonomia completa de concept drift, métodos de detecção
   - *Aplicação*: Base teórica fundamental para análise de drift

2. **Wahab (2022)** - Intrusion Detection in the IoT Under Data and Concept Drifts: Online Deep Learning Approach
   - *Foco*: Concept drift específico em contexto IoT IDS
   - *Aplicação*: Benchmark e comparação direta com trabalho relacionado

3. **Xu et al. (2023)** - ADTCD: An Adaptive Anomaly Detection Approach Toward Concept Drift in IoT
   - *Foco*: Método adaptativo para concept drift em IoT
   - *Aplicação*: Estado da arte para comparação e inspiração

#### Leituras Complementares (Semanas 9-10)
4. **Bharani et al. (2024)** - Adaptive Real-Time Malware Detection for IoT Traffic Streams: A Comparative Study of Concept Drift Detection Techniques
   - *Foco*: Comparação de técnicas de detecção de drift
   - *Aplicação*: Metodologia comparativa e validação de resultados

5. **Yang & Shami (2021)** - A Lightweight Concept Drift Detection and Adaptation Framework for IoT Data Streams
   - *Foco*: Framework lightweight para IoT
   - *Aplicação*: Inspiração para implementação eficiente

---

## 📊 Entregáveis Específicos da Fase 1

### **Entregáveis Técnicos**
1. **Código Fonte Completo**
   - [ ] Pipeline de pré-processamento reproduzível
   - [ ] Implementação dos 3 algoritmos baseline
   - [ ] Framework de detecção de concept drift
   - [ ] Scripts de avaliação e visualização
   - [ ] Documentação técnica completa

2. **Datasets e Amostras**
   - [ ] Amostra estratificada do CICIoT2023 (validada estatisticamente)
   - [ ] Metadata completo da amostra
   - [ ] Datasets sintéticos para validação de drift
   - [ ] Splits temporais para avaliação

3. **Resultados e Métricas**
   - [ ] Benchmarks completos dos algoritmos baseline
   - [ ] Caracterização quantitativa do concept drift
   - [ ] Análise de correlação drift-performance
   - [ ] Métricas de robustez e estabilidade

### **Entregáveis Científicos**
1. **Workshop Paper** (Semana 6)
   - *Título*: "Comparative Analysis of Classical Anomaly Detection in IoT: A Systematic Evaluation on CICIoT2023 Dataset"
   - *Venue*: SBRC Workshop ou WebMedia Workshop
   - *Contribuição*: Baseline robusto e metodologia de amostragem

2. **Conference Paper** (Semana 10)
   - *Título*: "Characterizing Concept Drift in IoT Traffic: Implications for Adaptive Intrusion Detection Systems"
   - *Venue*: BRACIS, ENIAC, ou SBSeg
   - *Contribuição*: Primeira caracterização abrangente de concept drift em dados IoT

3. **Relatório Técnico** (Semana 12)
   - Documentação completa da metodologia
   - Lessons learned e best practices
   - Roadmap detalhado para Fase 2
   - Código e dados versionados

### **Entregáveis de Infraestrutura**
1. **Ambiente Reproduzível**
   - [ ] Docker containers configurados
   - [ ] Requirements e environment specifications
   - [ ] Scripts de setup automatizado
   - [ ] CI/CD pipeline básico

2. **Sistema de Tracking**
   - [ ] MLflow setup com experimentos categorizados
   - [ ] DVC pipeline para reprodução
   - [ ] Git hooks para versionamento automático
   - [ ] Dashboard de monitoramento

---

## 🔄 Critérios de Sucesso e Checkpoints

### **Checkpoint Semana 4: Baseline Estabelecido**
**Critérios de Sucesso:**
- [ ] Pipeline de dados funcionando sem erros
- [ ] 3 algoritmos baseline implementados e validados
- [ ] Métricas baseline: F1-Score > 80% para classe Normal
- [ ] Tempo de processamento: < 5min para amostra completa

**Ações se Critérios não Atingidos:**
- Revisão da estratégia de pré-processamento
- Simplificação da amostra ou features
- Consulta com orientadores sobre ajustes

### **Checkpoint Semana 8: Concept Drift Identificado**
**Critérios de Sucesso:**
- [ ] Detectores de drift implementados e calibrados
- [ ] Drift quantificado em pelo menos 3 dimensões
- [ ] Impacto na performance dos baselines mensurado
- [ ] Visualizações claras e interpretáveis

**Ações se Critérios não Atingidos:**
- Revisão dos métodos de detecção de drift
- Análise de janelas temporais alternativas
- Consulta da literatura adicional

### **Checkpoint Semana 12: Transição para Fase 2**
**Critérios de Sucesso:**
- [ ] 2 papers submetidos com feedback positivo
- [ ] Base de código estável e documentada
- [ ] Insights claros para desenvolvimento do clustering evolutivo
- [ ] Ambiente pronto para Fase 2

**Ações se Critérios não Atingidos:**
- Extensão da Fase 1 por 2-4 semanas
- Priorização dos elementos essenciais para Fase 2
- Replanejamento do cronograma geral

---

## 🚀 Ponte para Fase 2: Clustering Evolutivo

### **Preparação Específica para Clustering Evolutivo**
1. **Análise de Features Estáveis**: Identificação de features menos afetadas por concept drift
2. **Padrões Temporais**: Caracterização de janelas ótimas para adaptação
3. **Metrics Benchmark**: Estabelecimento de métricas base para comparação
4. **Infrastructure**: Pipeline de streaming básico preparado

### **Research Questions para Fase 2**
1. Como o clustering evolutivo se compara aos baselines estabelecidos?
2. Qual a frequência ótima de adaptação dos clusters?
3. Como balancear estabilidade vs. adaptabilidade?
4. Quais features são mais importantes para formação de clusters?

### **Hipóteses Refinadas**
1. **H1**: Clustering evolutivo terá melhor recall para ataques novos/variantes
2. **H2**: Adaptação contínua resultará em menor degradação temporal
3. **H3**: Trade-off existirá entre latência e precisão adaptativa
4. **H4**: Modelos device-specific superarão modelo generalista

---

## 📋 Checklist Final de Validação

### **Validação Científica**
- [ ] Metodologia de amostragem justificada e validada estatisticamente
- [ ] Limitações explicitamente documentadas
- [ ] Resultados reproduzíveis com código e dados versionados
- [ ] Comparação justa com trabalhos relacionados
- [ ] Contribuições científicas claramente articuladas

### **Validação Técnica**
- [ ] Código passa em todos os testes automatizados
- [ ] Performance satisfaz requisitos de latência (<100ms inference)
- [ ] Escalabilidade testada até limites da amostra
- [ ] Documentação permite reprodução independente
- [ ] Versionamento permite rollback e comparações

### **Validação de Projeto**
- [ ] Cronograma seguido com desvios <10%
- [ ] Objetivos da Fase 1 completamente atendidos
- [ ] Base sólida estabelecida para Fase 2
- [ ] Publicações submetidas nos prazos
- [ ] Feedback dos orientadores incorporado

---

**Este cronograma garante uma Fase 1 sólida, publicável e que estabelece fundamentos robustos para o desenvolvimento do clustering evolutivo na Fase 2. A atenção especial à metodologia de amostragem e validação estatística assegura rigor científico, enquanto os checkpoints regulares permitem ajustes de curso quando necessário.**