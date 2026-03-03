---
name: iot-ids-research-context
description: Core context skill that maintains complete project state, knows all code structure, research phases, and previous work. Always active to prevent repetitive explanations.
version: 2.0.0
author: Research Acceleration System
---

# IoT IDS Research Context

## Purpose

You are working on a Master's dissertation research project at UFMG (Federal University of Minas Gerais) in Electrical Engineering. This skill maintains complete context so you never need to re-explain the project, restate what's been done, or repeat basic information.

**ALWAYS leia `STATUS.md` na raiz antes de qualquer ação** — é a fonte de verdade do estado atual.

## Project Identity

**Title:** Anomaly-based Intrusion Detection System for IoT Networks using Evolutionary Clustering

**Researcher:** Augusto (Master's student)
**Institution:** UFMG PPGEE
**Advisor:** Frederico Gadelha Guimarães (co-autor do MicroTEDAclus — Maia et al. 2020)
**Timeline:** ~6 meses restantes (24 semanas no total)
**Weekly Dedication:** 10-20 hours
**Meetings:** Weekly with advisor

## Current Phase Status (2026-03-03)

**Phase 1:** ✅ COMPLETE — 705 experiments, 10 ML algorithms, F1 > 0.99
**Phase 2A:** ✅ COMPLETE — Teoria + TEDA + MicroTEDAclus + setup Kafka
**Phase 2B:** 🔄 EM ANDAMENTO — Experimentos streaming (Semana 5/24)
**Phase 3:** 📋 PLANEJADO
**Phase 4:** 📋 PLANEJADO

### Phase 2B — Estado Atual (Semana 5)

**O que está funcionando:**
- PCAPProducer v0.1: lê PCAPs → tópico `packets` (2909 pkt/s)
- FlowConsumer v0.1: agrega flows → tópico `flows` (28 features ML + 8 metadata)
- TEDADetector: eccentricidade + tipicalidade + threshold Chebyshev
- MicroTEDAclus: micro-clusters com estatísticas isoladas, threshold dinâmico m(k)
- StreamingDetector v0.2: seleção de algoritmo em runtime (TEDA ou MicroTEDAclus)
- run_experiment.py: orquestrador completo com 5 artefatos por execução
- compare_experiments.py: comparação de múltiplos experimentos
- Métricas prequential (Gama et al. 2013): sliding window P/R/F1/FPR/MTTD
- Ground truth heurístico: inferência por filename do PCAP
- **Testes:** 98 testes unitários passando

**Pendente na Semana 5:**
- Experimento DDoS (validar Recall >= 80%, MTTD <= 500 flows)

## Code Structure (MEMORIZE THIS)

```
/Users/augusto/mestrado/final-project/
├── STATUS.md                    ← LEIA PRIMEIRO: estado atual + próximos passos
├── CLAUDE.md                    ← Instruções para Claude Code
├── README.md                    ← Visão geral do projeto
│
├── streaming/                   ← CÓDIGO PRINCIPAL (Fase 2)
│   ├── src/
│   │   ├── producer/            ← PCAPProducer (lê PCAP → Kafka)
│   │   ├── consumer/            ← FlowConsumer (pacotes → flows)
│   │   ├── detector/            ← TEDADetector + MicroTEDAclus + StreamingDetector
│   │   ├── metrics/             ← PrequentialMetrics + GroundTruthProvider
│   │   └── kafka_utils.py       ← purge_kafka_topics(), pre-flight check
│   ├── scripts/
│   │   ├── run_experiment.py    ← Orquestrador principal (CLI)
│   │   └── compare_experiments.py ← Comparador de resultados
│   ├── tests/                   ← 98 testes unitários + E2E
│   ├── results/week5/           ← Resultados dos experimentos da semana 5
│   ├── docker/                  ← docker-compose.yml (Kafka + Zookeeper + UI)
│   └── venv/                    ← Python environment (ativar com source venv/bin/activate)
│
├── baseline/                    ← Fase 1 (COMPLETA — não modificar)
│   └── experiments/.results/    ← 705 experimentos executados
│
├── data/
│   └── raw/PCAP/                ← PCAPs do CICIoT2023
│       ├── Benign/BenignTraffic.pcap
│       └── DDoS/DDoS-ICMP_Flood.pcap
│
└── docs/
    ├── SESSION_CONTEXT.md       ← Histórico detalhado (referência)
    ├── architecture/CURRENT.md  ← Arquitetura implementada
    ├── weekly-reports/          ← Relatórios para orientador
    ├── theory/                  ← TEDA, MicroTEDAclus, concept drift
    └── paper-summaries/         ← Fichamentos de papers
```

## Dataset: CICIoT2023

**Source:** Canadian Institute for Cybersecurity
**Type:** Real IoT network traffic with labeled attacks (33 ataques, 7 categorias)
**Important:** CSVs são shuffled — processar PCAPs é obrigatório para preservar ordem temporal
**Local PCAPs:** `data/raw/PCAP/` (~548GB total, subset local disponível)

## Kafka Infrastructure

```
Zookeeper: localhost:2181
Kafka:     localhost:9092
Kafka-UI:  localhost:8080
```

**Tópicos:** `packets` (raw), `flows` (aggregated), `alerts` (anomalies)

**Para iniciar:**
```bash
cd streaming/docker && docker-compose up -d
# Aguardar ~30s
```

## Running Experiments

```bash
cd streaming
source venv/bin/activate

# Experimento básico (apenas benigno)
python3 scripts/run_experiment.py \
  --pcap ../data/raw/PCAP/Benign/BenignTraffic.pcap \
  --max-packets 50000 --max-flows 5000 \
  --algorithm micro_teda --r0 0.10 \
  --output results/week5/test/

# Experimento com ataque
python3 scripts/run_experiment.py \
  --pcap ../data/raw/PCAP/Benign/BenignTraffic.pcap \
  --attack-pcap ../data/raw/PCAP/DDoS/DDoS-ICMP_Flood.pcap \
  --max-packets 50000 --max-flows 10000 \
  --output results/week5/ddos_detection/

# Comparar resultados
python3 scripts/compare_experiments.py results/week5/
```

## Test Suite

```bash
cd streaming
source venv/bin/activate
python3 -m pytest tests/ -q    # 98 testes, ~16s
```

## Research Methodology

**Experiment Standards:**
- Métricas prequential: sliding window (1000 flows), fading factor (alpha=0.01)
- Isolamento entre experimentos: group IDs únicos + purge de tópicos (~2s overhead)
- 5 artefatos por execução: run_meta.json, detection_results.json, metrics_windowed.csv, clusters_state.jsonl, system_usage.csv

**Success Criteria (Fase 2B):**
- FPR em tráfego benigno: ≤ 5%
- Recall em ataques: ≥ 80%
- MTTD: ≤ 500 flows
- Throughput: ≥ 100 flows/s

## Key Papers

- **Angelov (2014)** — TEDA Framework original (fichamento: `docs/paper-summaries/angelov-2014-teda.md`)
- **Maia et al. (2020)** — MicroTEDAclus (fichamento: `docs/paper-summaries/maia-2020-microtedaclus.md`)
- **Gama et al. (2013)** — Prequential evaluation (fichamento: `docs/paper-summaries/gama-2013-prequential-evaluation.md`)

## Critical Guidelines

### NÃO FAÇA:
- ❌ Re-explicar o que a Fase 1 fez (está completo)
- ❌ Sugerir re-executar experimentos baseline
- ❌ Assumir tempo ilimitado (20h/semana é a realidade)
- ❌ Ignorar STATUS.md ao retomar trabalho
- ❌ Deixar STATUS.md desatualizado ao final da sessão

### SEMPRE:
- ✅ Ler `STATUS.md` antes de qualquer ação
- ✅ Atualizar `STATUS.md` ao final de sessão significativa
- ✅ Ativar venv antes de rodar código Python: `source streaming/venv/bin/activate`
- ✅ Verificar Kafka rodando antes de experimentos
- ✅ Rodar `pytest tests/ -q` após mudanças no código

## Communication Style

- Direto e eficiente (respeitar tempo limitado)
- Sem elogios desnecessários
- Foco em próximos passos acionáveis
- Explicar o "porquê" de decisões técnicas (objetivo de aprendizado)
- Português para trabalho relacionado à dissertação

---

**Esta skill garante contexto completo sem explicações repetitivas.**
