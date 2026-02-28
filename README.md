# Anomaly-Based IDS for IoT Networks using Evolutionary Clustering

**Dissertação de Mestrado** - Programa de Pós-Graduação em Engenharia Elétrica (UFMG)

Sistema de detecção de intrusão para redes IoT usando algoritmos de clustering evolutivo com arquitetura streaming.

**Status Atual:** Semana 5 - Fase 2A (Validação Streaming + Ground Truth)

---

## 📋 Visão Geral

Este repositório contém toda a implementação e experimentos da dissertação, organizada em 4 fases:

| Fase | Período | Objetivo | Status |
|------|---------|----------|--------|
| **Fase 1** | Meses 1-3 | Baseline ML com algoritmos clássicos | ✅ **Completo** (705 experimentos) |
| **Fase 2** | Meses 4-6 | Streaming + Clustering Evolutivo | 🔄 **Em andamento** (Semana 5) |
| **Fase 3** | Meses 7-9 | Modelos device-specific + Two-phase | 📅 Planejado |
| **Fase 4** | Meses 10-12 | Otimização + Dissertação | 📅 Planejado |

**Dataset:** CICIoT2023 (Canadian Institute for Cybersecurity)

---

## 🗂️ Estrutura do Repositório

```
final-project/
├── baseline/                    # Fase 1: Experimentos ML baseline
│   ├── experiments/             # Scripts de experimentos (10 algoritmos)
│   ├── data/                    # Datasets processados
│   ├── dvc.yaml                 # Pipeline DVC reproduzível
│   └── mlflow.db                # Tracking de experimentos
│
├── streaming/                   # Fase 2: Pipeline Kafka + Clustering Evolutivo
│   ├── src/
│   │   ├── producer/            # PCAPProducer (PCAP → Kafka)
│   │   ├── consumer/            # FlowConsumer (packets → flows)
│   │   ├── detector/            # MicroTEDAclus + StreamingDetector
│   │   └── metrics/             # GroundTruth + PrequentialMetrics
│   ├── scripts/
│   │   └── run_experiment.py    # Orquestrador de experimentos
│   ├── tests/                   # 67 testes unitários
│   └── results/                 # Resultados de experimentos (JSON)
│
├── data/                        # Dados compartilhados
│   ├── pcaps/                   # PCAPs do CICIoT2023
│   └── processed/               # Datasets processados
│
├── docs/                        # Documentação
│   ├── methodology/             # Metodologia experimental
│   ├── architecture/            # Diagramas de arquitetura
│   ├── paper-summaries/         # Fichamentos de papers
│   ├── weekly-reports/          # Relatórios semanais
│   └── SESSION_CONTEXT.md       # Contexto atual do projeto
│
├── labs/                        # Laboratórios de aprendizado
└── docker-compose.yml           # Kafka + Zookeeper + Jupyter + MLflow
```

---

## 🚀 Quick Start

### Fase 1: Baseline ML (Completo)

Experimentos com 10 algoritmos clássicos de ML no CICIoT2023.

```bash
cd baseline/

# Executar pipeline completo (DVC)
dvc repro

# Executar algoritmo específico
python experiments/run_single_algorithm.py random_forest

# Ver resultados no MLflow
docker-compose up -d
# Acessar: http://localhost:5000
```

**Resultados:** F1 > 0.99 para maioria dos algoritmos. Veja `baseline/experiments/results/full/`

### Fase 2: Streaming Pipeline (Atual - Semana 5)

Pipeline Kafka com MicroTEDAclus para detecção em tempo real.

#### 1. Iniciar Kafka

```bash
# No diretório raiz
docker-compose up -d kafka zookeeper
```

#### 2. Executar Experimento Completo

```bash
cd streaming/

# Experimento com PCAP benign
python scripts/run_experiment.py \
    --pcap ../data/pcaps/benign/Benign_Final.pcap \
    --output results/exp_benign.json

# Experimento com ataque DDoS
python scripts/run_experiment.py \
    --pcap ../data/pcaps/ddos/DDoS-ICMP_Flood.pcap \
    --output results/exp_ddos.json
```

#### 3. Validação Rápida (poucos pacotes)

```bash
python scripts/run_experiment.py \
    --pcap ../data/pcaps/benign/Benign_Final.pcap \
    --max-packets 1000 \
    --max-flows 100 \
    --verbose
```

**Componentes:** PCAPProducer → Kafka(packets) → FlowConsumer → Kafka(flows) → StreamingDetector → results.json

---

## 🔬 Experimentos

### Baseline (Fase 1)

**Algoritmos avaliados:**
- Logistic Regression
- Random Forest
- Gradient Boosting
- Isolation Forest
- Elliptic Envelope
- Local Outlier Factor
- Linear SVC
- SGD Classifier
- SGD OneClassSVM
- MLP

**Configuração:**
- Dataset: 10% sample do CICIoT2023 (~500k flows)
- Grid Search: 5 runs por configuração
- Métricas: Precision, Recall, F1, ROC-AUC
- Tracking: MLflow

### Streaming (Fase 2 - Semana 5)

**Algoritmo:** MicroTEDAclus (Maia et al., 2020)
- Clustering evolutivo multi-cluster
- Resistente a contaminação de outliers
- Threshold dinâmico m(k) baseado no tamanho do cluster

**Validação:**
- Ground truth: Heurística baseada no filename do PCAP
- Métricas prequential: Precision, Recall, F1, MTTD (Gama et al., 2013)
- Parâmetros: r0=0.1, window_size=1000, alpha=0.01

**Próximos passos (Semanas 6-9):**
- Implementar CluStream, DenStream, StreamKM++
- Comparação entre algoritmos
- Ground truth exato via CSVs do CICIoT2023

---

## 📊 Resultados Principais

### Fase 1 (Baseline)

| Algoritmo | F1-Score | Precision | Recall | Tempo (s) |
|-----------|----------|-----------|--------|-----------|
| Random Forest | 0.9945 | 0.9921 | 0.9969 | 45.2 |
| Gradient Boosting | 0.9938 | 0.9912 | 0.9964 | 78.5 |
| Isolation Forest | 0.9876 | 0.9854 | 0.9898 | 12.3 |

**Conclusão:** Algoritmos clássicos têm excelente performance no CICIoT2023, mas não adaptam a concept drift.

### Fase 2 (Streaming - Preliminar)

**MicroTEDAclus v0.2:**
- 67/67 testes unitários passando
- 2350 LOC implementadas
- Pipeline Kafka completo operacional

**Validação (Semana 5):** Em andamento

---

## 🛠️ Pré-requisitos

### Sistema

- Python 3.8+
- Docker + Docker Compose (para Kafka e MLflow)
- Git + DVC (para reproduzir experimentos baseline)

### Dependências Python

```bash
# Baseline
cd baseline/
pip install -r requirements.txt

# Streaming
cd streaming/
pip install -r requirements.txt
```

**Principais:**
- `scikit-learn`, `pandas`, `numpy` (ML)
- `kafka-python` (streaming)
- `mlflow`, `dvc` (experiment tracking)
- `dpkt` (parsing de PCAPs)

### Dataset CICIoT2023

**Baixar PCAPs:**
1. Acessar [CICIoT2023 no Kaggle](https://www.kaggle.com/datasets/madhavmalhotra/ciciot2023)
2. Baixar arquivos PCAP
3. Organizar em `data/pcaps/` por tipo de ataque

**Estrutura esperada:**
```
data/pcaps/
├── benign/
├── ddos/
├── dos/
├── mirai/
├── recon/
├── spoofing/
├── web/
└── brute_force/
```

---

## 📚 Documentação

### Arquivos Principais

- **`docs/SESSION_CONTEXT.md`** - Estado atual do projeto, próximas tarefas
- **`docs/methodology/experiment-methodology.md`** - Metodologia experimental detalhada (Semanas 1-12)
- **`docs/architecture/CURRENT.md`** - Arquitetura implementada (atualizar a cada mudança)
- **`docs/architecture/TARGET.md`** - Arquitetura alvo (visão de longo prazo)
- **`CLAUDE.md`** - Instruções para Claude Code (contexto do projeto)

### Papers Fundamentais

Fichamentos em `docs/paper-summaries/`:
- **Angelov (2014)** - TEDA Framework
- **Maia et al. (2020)** - MicroTEDAclus
- **Gama et al. (2013)** - Prequential Evaluation
- **Neto et al. (2023)** - CICIoT2023 Dataset

### Comandos Úteis

```bash
# Resumir contexto do projeto
/resume

# Iniciar nova semana de trabalho
/start-sprint

# Gerar relatório semanal
/finalize-week

# Resumir paper do Zotero
/paper-summary <nome>
```

---

## 🧪 Testes

### Baseline

```bash
cd baseline/
pytest experiments/tests/ -v
```

### Streaming

```bash
cd streaming/
pytest tests/ -v

# Testes específicos
pytest tests/test_micro_teda.py -v
pytest tests/test_ground_truth.py -v
pytest tests/test_prequential_metrics.py -v
```

**Status:** 67/67 testes passando (Semana 4)

---

## 🐛 Troubleshooting

### Kafka não conecta

```bash
# Verificar se está rodando
docker ps | grep kafka

# Reiniciar
docker-compose down
docker-compose up -d kafka zookeeper
```

### DVC pipeline falha

```bash
# Verificar status
cd baseline/
dvc status

# Limpar cache
dvc gc --workspace
dvc repro --force
```

### MLflow não abre

```bash
# Verificar porta
lsof -i :5000

# Iniciar manualmente
cd baseline/
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## 📖 Referências

### Papers

- Angelov, P. (2014). "Outside the box: an alternative data analytics framework." *Journal of Automation Mobile Robotics and Intelligent Systems*.
- Maia, J. et al. (2020). "Evolving clustering algorithm based on mixture of typicalities." *Future Generation Computer Systems*.
- Gama, J. et al. (2013). "On evaluating stream learning algorithms." *Machine Learning*.
- Neto, E.C.P. et al. (2023). "CICIoT2023: A Real-Time Dataset and Benchmark." *Sensors*.

### Tecnologias

- [Apache Kafka](https://kafka.apache.org/) - Message broker
- [MLflow](https://mlflow.org/) - Experiment tracking
- [DVC](https://dvc.org/) - Data version control
- [CICIoT2023 Dataset](https://www.unb.ca/cic/datasets/iot-dataset-2023.html)

---

## 📝 Licença

Este é um projeto acadêmico de dissertação de mestrado. Código disponível para fins educacionais e de pesquisa.

---

## 👤 Autor

**Augusto** - Mestrado em Engenharia Elétrica (UFMG)

**Orientador:** [Nome do Orientador]

---

**Última atualização:** 2026-02-25 | **Fase:** 2A | **Semana:** 5