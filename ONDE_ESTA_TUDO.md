# Índice - Onde Está Tudo

Guia rápido para localizar qualquer documento ou código do projeto.

---

## 🚀 Para Começar AGORA (próxima sessão)

**Arquivo principal:** `streaming/QUICK_START_NEXT_SESSION.md`
- Comandos prontos para executar grid 3×2
- Experimento DDoS
- Checklist completo

---

## 📊 Status e Contexto

### Entender "onde estamos"
1. **`docs/SESSION_CONTEXT.md`** - Status atual, próximos passos, histórico completo
2. **`SESSAO_2026-02-25_RESUMO.md`** - Resumo executivo do que foi feito hoje
3. **`docs/weekly-reports/semana5-report.md`** - Relatório detalhado da Semana 5

### Planejamento e Metodologia
1. **`docs/methodology/experiment-methodology.md`** - Metodologia experimental completa (Semanas 1-12)
2. **`CLAUDE.md`** - Instruções para Claude Code (comandos, estrutura)
3. **`README.md`** - Visão geral do projeto

---

## 💻 Código Implementado

### Sistema de Isolamento (novo hoje)
```
streaming/
├── src/
│   └── kafka_utils.py              # Purga automática de tópicos
├── scripts/
│   ├── run_experiment.py           # Orquestrador (com isolamento)
│   └── compare_experiments.py      # Comparação de resultados
├── tests/
│   ├── test_kafka_isolation.py     # 11 testes (purga)
│   ├── test_experiment_orchestration.py  # 9 testes (group IDs)
│   └── e2e_test_isolation.sh       # Script E2E
└── docs/
    └── experiment-isolation.md     # Documentação técnica completa
```

### Componentes Core (já implementados)
```
streaming/
├── src/
│   ├── producer/
│   │   └── pcap_producer.py        # PCAP → Kafka packets
│   ├── consumer/
│   │   └── flow_consumer.py        # Packets → Flows
│   ├── detector/
│   │   ├── teda.py                 # TEDADetector (36 testes)
│   │   ├── micro_teda.py           # MicroTEDAclus (31 testes)
│   │   └── streaming_detector.py   # Orquestrador de detecção
│   └── metrics/
│       ├── ground_truth.py         # Labels heurísticos (48 testes)
│       └── prequential_metrics.py  # Métricas streaming (Gama 2013)
```

---

## 📚 Documentação Técnica

### Arquitetura
```
docs/
└── architecture/
    ├── CURRENT.md                  # Estado atual (SEMPRE atualizar!)
    ├── TARGET.md                   # Visão de longo prazo
    └── KAFKA_REFERENCE.md          # Referência educacional Kafka
```

### Teoria
```
docs/
└── theory/
    ├── teda-framework.md           # Fundamentação TEDA/MicroTEDAclus
    ├── concept-drift.md            # Teoria de concept drift
    └── pcap-justification.md       # Por que processar PCAPs
```

### Papers Resumidos
```
docs/
└── paper-summaries/
    ├── angelov-2014-teda.md                # TEDA Framework
    ├── maia-2020-microtedaclus.md          # MicroTEDAclus
    ├── gama-2013-prequential-evaluation.md # Métricas streaming
    └── neto-2023-ciciot.md                 # Dataset CICIoT2023
```

### Metodologia
```
docs/
└── methodology/
    └── experiment-methodology.md   # Semanas 1-12 detalhadas
```

### Relatórios Semanais
```
docs/
└── weekly-reports/
    ├── semana5-report.md           # Semana 5 (atual)
    └── archive/                    # Semanas anteriores
```

---

## 🧪 Testes

### Executar Todos os Testes
```bash
cd streaming
pytest tests/ -v

# Específicos
pytest tests/test_kafka_isolation.py -v      # Isolamento
pytest tests/test_micro_teda.py -v           # MicroTEDAclus
pytest tests/test_ground_truth.py -v         # Ground truth
pytest tests/test_prequential_metrics.py -v  # Métricas
```

### E2E
```bash
cd streaming
./tests/e2e_test_isolation.sh               # Isolamento completo
```

---

## 📊 Resultados de Experimentos

### Estrutura
```
results/
└── week5/
    ├── sanity_quick/               # Validação rápida (100 flows)
    ├── consolidation_test/         # Validação média (2000 flows)
    ├── grid_teda_r0_0.05/         # (pendente)
    ├── grid_teda_r0_0.10/         # (pendente)
    ├── grid_teda_r0_0.20/         # (pendente)
    ├── grid_micro_teda_r0_0.05/   # (pendente)
    ├── grid_micro_teda_r0_0.10/   # (pendente)
    ├── grid_micro_teda_r0_0.20/   # (pendente)
    └── ddos_detection/             # (pendente)
```

### Cada experimento contém 5 artefatos:
```
<experiment_dir>/
├── run_meta.json              # Git commit, parâmetros, timestamps
├── detection_results.json     # Resultados completos
├── metrics_windowed.csv       # Métricas prequential
├── clusters_state.jsonl       # Snapshots de clusters
└── system_usage.csv           # CPU/memória
```

---

## 🎯 Baseline (Fase 1) - Completo

```
baseline/
├── experiments/                # Scripts de experimentos
│   ├── run_single_algorithm.py
│   ├── algorithm_comparison.py
│   └── results/                # 705 experimentos
├── data/                       # Datasets processados
│   ├── processed/
│   └── sampled/
├── dvc.yaml                    # Pipeline reproduzível
└── mlflow.db                   # Tracking de experimentos
```

---

## 🐳 Docker

### Kafka + Zookeeper
```bash
cd /Users/augusto/mestrado/final-project
docker-compose up -d kafka zookeeper

# Verificar
docker ps | grep kafka
```

### MLflow (Baseline)
```bash
cd baseline
docker-compose up -d
# Acessar: http://localhost:5000
```

---

## 🛠️ Utilitários

### DVC (Baseline)
```bash
cd baseline
dvc repro                       # Pipeline completo
dvc repro <stage_name>          # Stage específico
```

### Git
```bash
# Ver mudanças
git status
git diff

# Commit sugerido em SESSAO_2026-02-25_RESUMO.md
```

---

## 📖 Para Aprender/Revisar

### Entender TEDA
1. `docs/theory/teda-framework.md`
2. `docs/paper-summaries/angelov-2014-teda.md`

### Entender MicroTEDAclus
1. `docs/theory/teda-framework.md` (seção MicroTEDAclus)
2. `docs/paper-summaries/maia-2020-microtedaclus.md`

### Entender Métricas Prequential
1. `docs/paper-summaries/gama-2013-prequential-evaluation.md`
2. `streaming/src/metrics/prequential_metrics.py` (docstrings)

### Entender Isolamento de Experimentos
1. `streaming/docs/experiment-isolation.md`
2. `streaming/src/kafka_utils.py` (código + comentários)

---

## 🔍 Buscar Algo

### Por Funcionalidade
- **Processar PCAP:** `streaming/src/producer/pcap_producer.py`
- **Agregar em flows:** `streaming/src/consumer/flow_consumer.py`
- **Detectar anomalias:** `streaming/src/detector/streaming_detector.py`
- **Calcular métricas:** `streaming/src/metrics/prequential_metrics.py`
- **Comparar experimentos:** `streaming/scripts/compare_experiments.py`

### Por Algoritmo
- **TEDA:** `streaming/src/detector/teda.py`
- **MicroTEDAclus:** `streaming/src/detector/micro_teda.py`

### Por Conceito
- **Concept Drift:** `docs/theory/concept-drift.md`
- **Ground Truth:** `streaming/src/metrics/ground_truth.py`
- **Prequential:** `docs/paper-summaries/gama-2013-prequential-evaluation.md`

---

## 🚨 Problemas Comuns

### Kafka não conecta
```bash
docker-compose down
docker-compose up -d kafka zookeeper
sleep 30
```

### Testes falham
```bash
cd streaming
pytest tests/ -v --tb=short
```

### Experimento trava
- Usar `--max-flows` menor
- Verificar logs do Kafka

---

## 📅 Cronograma

**Atual:** Semana 5 de 24 (20.8% completo)

### Próximas Semanas
- **S5 (atual):** Validação streaming + grid comparativo
- **S6-7:** Outros algoritmos (CluStream, DenStream, StreamKM++)
- **S8-9:** Ground truth exato (CSVs) + experimentos full

Ver: `docs/methodology/experiment-methodology.md`

---

**Última Atualização:** 2026-02-25 19:25
**Para Dúvidas:** Ver `docs/SESSION_CONTEXT.md` seção "Current Status"
