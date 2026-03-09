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

## Current Phase Status (2026-03-09)

**Phase 1:** ✅ COMPLETE — 705 experiments, 10 ML algorithms, F1 > 0.99
**Phase 2A:** ✅ COMPLETE — Teoria + TEDA + MicroTEDAclus + setup Kafka
**Phase 2B:** 🔄 EM ANDAMENTO — Experimentos streaming
**Phase 3:** 📋 PLANEJADO (nice-to-have)
**Phase 4:** 📋 PLANEJADO (nice-to-have)
**Prazo defesa:** ~maio 2026 (~8 semanas)

### Phase 2B — Estado Atual

**O que está funcionando:**
- PCAPProducer v0.1: lê PCAPs → tópico `packets` (2909 pkt/s)
- FlowConsumer v0.1: agrega flows → tópico `flows` (28 features ML + 8 metadata)
- TEDADetector: eccentricidade + tipicalidade + threshold Chebyshev
- MicroTEDAclus: micro-clusters com estatísticas isoladas, threshold dinâmico m(k)
- StreamingDetector v0.2: seleção de algoritmo em runtime (TEDA ou MicroTEDAclus)
- run_experiment.py: orquestrador completo com 5 artefatos por execução
- compare_experiments.py: comparação de múltiplos experimentos
- Métricas prequential (Gama et al. 2013): sliding window P/R/F1/FPR/MTTD
- Ground truth por fase (evaluate_by_phase): avaliação ao final de cada fase do experimento
- **Testes:** suite unitária passando

**Próximos passos:** ver `STATUS.md`

## Code Structure (MEMORIZE THIS)

```
# Path relativo à raiz do repositório (varia por máquina)
├── STATUS.md                    ← LEIA PRIMEIRO: estado atual + próximos passos
├── CLAUDE.md                    ← Instruções para Claude Code
├── USAGE.md                     ← Guia de uso do repositório por cenário
│
├── research/                    ← CONHECIMENTO
│   ├── bibliography.bib         ← Referências BibTeX consolidadas
│   ├── reading-log.md           ← Leituras + lacunas de conhecimento
│   ├── summaries/               ← Fichamentos de papers
│   └── foundations/             ← Teoria (TEDA, concept drift)
│
├── experiments/                 ← EVIDÊNCIA
│   ├── methodology.md           ← Metodologia científica (cap. 4)
│   ├── campaign-plan.md         ← Plano experimental (cenários A/B/D/E)
│   ├── streaming/               ← Código streaming (Fase 2)
│   │   ├── src/
│   │   │   ├── producer/        ← PCAPProducer (lê PCAP → Kafka)
│   │   │   ├── consumer/        ← FlowConsumer (pacotes → flows)
│   │   │   ├── detector/        ← TEDADetector + MicroTEDAclus + StreamingDetector
│   │   │   ├── metrics/         ← PrequentialMetrics + GroundTruthProvider
│   │   │   └── kafka_utils.py   ← purge_kafka_topics(), pre-flight check
│   │   ├── scripts/
│   │   │   ├── run_experiment.py    ← Orquestrador principal (CLI)
│   │   │   └── compare_experiments.py ← Comparador de resultados
│   │   ├── tests/               ← Testes unitários + E2E
│   │   ├── docker/              ← docker-compose.yml (Kafka + Zookeeper + UI)
│   │   └── venv/                ← Python environment
│   ├── baseline/                ← Fase 1 (COMPLETA — não modificar)
│   │   └── experiments/.results/ ← 705 experimentos executados
│   └── results/campaign-01/     ← Resultados das campanhas experimentais
│
├── writing/                     ← PRODUÇÃO
│   ├── dissertation/            ← Clone do Overleaf (PENDENTE)
│   ├── figures/                 ← Figuras geradas pelos experimentos
│   └── tables/                  ← Tabelas geradas pelos experimentos
│
├── data/
│   └── pcaps/                   ← PCAPs do CICIoT2023 (benign/, ddos/, dos/, mirai/, recon/, spoofing/)
│
└── docs/                        ← OPERACIONAL
    ├── architecture/CURRENT.md  ← Arquitetura implementada
    ├── progress/                ← Logs automáticos de sessão
    └── plans/                   ← Planos de execução
```

## Dataset: CICIoT2023

**Source:** Canadian Institute for Cybersecurity
**Type:** Real IoT network traffic with labeled attacks (33 ataques, 7 categorias)
**Important:** CSVs são shuffled — processar PCAPs é obrigatório para preservar ordem temporal
**Local PCAPs:** `data/pcaps/` (~548GB total, subset local disponível)

## Kafka Infrastructure

```
Zookeeper: localhost:2181
Kafka:     localhost:9092
Kafka-UI:  localhost:8080
```

**Tópicos:** `packets` (raw), `flows` (aggregated), `alerts` (anomalies)

**Para iniciar:**
```bash
cd experiments/streaming/docker && docker-compose up -d
# Aguardar ~30s
```

## Running Experiments

```bash
cd experiments/streaming
source venv/bin/activate

# Experimento básico (apenas benigno)
python3 scripts/run_experiment.py \
  --pcap ../../data/pcaps/benign/BenignTraffic.pcap \
  --max-packets 50000 --max-flows 5000 \
  --algorithm micro_teda --r0 0.10 \
  --output ../results/campaign-01/test/

# Experimento com ataque
python3 scripts/run_experiment.py \
  --pcap ../../data/pcaps/benign/BenignTraffic.pcap \
  --attack-pcap ../../data/pcaps/ddos/DDoS-ICMP_Flood.pcap \
  --max-packets 50000 --max-flows 10000 \
  --output ../results/campaign-01/ddos_detection/

# Comparar resultados
python3 scripts/compare_experiments.py ../results/campaign-01/
```

## Test Suite

```bash
cd experiments/streaming
source venv/bin/activate
python3 -m pytest tests/ -q
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

- **Angelov (2014)** — TEDA Framework original (fichamento: `research/summaries/angelov-2014-teda.md`)
- **Maia et al. (2020)** — MicroTEDAclus (fichamento: `research/summaries/maia-2020-microtedaclus.md`)
- **Gama et al. (2013)** — Prequential evaluation (fichamento: `research/summaries/gama-2013-prequential-evaluation.md`)

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
- ✅ Ativar venv antes de rodar código Python: `source experiments/streaming/venv/bin/activate`
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
