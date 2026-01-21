# SESSION CONTEXT - IoT IDS Research Project
**Last Updated:** 2026-01-21 (Session: Semana 3 COMPLETA - TEDA v0.1 + testes)

---

## 🎯 CURRENT STATUS

**Phase:** Fase 2A - Teoria + Design + Setup
**Week:** Semana 3 de 24 (12.5% complete) ✅ COMPLETE
**Current Task:** Preparar início da Semana 4 (TEDA v0.2 com micro-clusters)

---

## 📊 PROJECT OVERVIEW

**Master's Dissertation - UFMG PPGEE**
*Detecção de Intrusão Baseada em Anomalias em Sistemas IoT com Clustering Evolutivo e Arquitetura de Alto Desempenho em Fluxos*

**Advisor:** Frederico Gadelha Guimarães (co-autor do paper Maia et al. 2020)
**Timeline:** ~6 meses restantes (24 semanas)
**Weekly Dedication:** 10-20 hours
**Weekly Meetings:** Every week with advisor (flexible day)

---

## ✅ COMPLETED WORK

### Fase 1: Baseline Experiments (100% COMPLETE)
- ✅ 705 experiments across 10 ML algorithms
- ✅ CICIoT2023 dataset (10% sample) preprocessed
- ✅ DVC pipeline established
- ✅ Docker + MLflow infrastructure
- ✅ Excellent baseline results (F1 > 0.99)
- ✅ Paper artigo1 in progress (Overleaf)

**Key Files:**
- `iot-ids-research/experiments/` - All baseline experiments
- `artigo1/` - Paper comparing baseline algorithms
- `REPOSITORY_ANALYSIS.md` - Complete Phase 1 analysis

### Fase 2A, Semana 1: Teoria + Design (100% COMPLETE)
- ✅ K-means: algoritmo, limitações, Silhouette Score, Elbow method
- ✅ DBSCAN: density-based, eps/min_samples, comportamento não-linear
- ✅ TEDA Framework: eccentricidade, tipicalidade, Chebyshev test
- ✅ MicroTEDAclus: micro-clusters, mixture of typicalities
- ✅ Concept drift: 4 tipos (súbito, gradual, incremental, recorrente)
- ✅ Análise PCAP vs CSV: CSV é shuffled, PCAP obrigatório
- ✅ Design arquitetura MVP: Kafka 2 tópicos, TEDA apenas
- ✅ Plano de leituras: 8 principais + 12 auxiliares em 4 áreas
- ✅ Relatório semanal finalizado

### Fase 2A, Semana 2: Leitura Angelov + Setup (100% COMPLETE)
- ✅ Fichamento Angelov (2014) - 100% completo
  - Conceitos: frequentista, belief/possibility theory, first principles
  - Métricas de distância: Euclidean, Manhattan, Mahalanobis, Cosine
  - Normalização e por que ξ = π normalizado
  - Fórmulas: π, ξ, τ com exemplos numéricos
  - Derivação matemática completa (Huygens-Steiner)
  - Seções 4-5: Anomaly Detection e Data Clouds
  - Limitações identificadas (zona de influência)
  - Como tipicalidade forma clusters
- ✅ Fichamento MicroTEDAclus (Maia 2020) - 100% completo
  - Arquitetura micro-clusters + macro-clusters
  - Threshold dinâmico m(k) = 3/(1 + e^{-0.007(k-100)})
  - Mixture of typicalities: T_j = w_l × t_l(x)
  - Comparação com DenStream, CluStream, StreamKM++
  - Pseudocódigo completo dos algoritmos
- ✅ Documento de lacunas de conhecimento criado
- ✅ Setup Kafka local (Docker Compose)
  - Zookeeper, Kafka, Kafka-UI rodando
  - 3 partições por tópico (packets, flows)
- ✅ Producer v0.1 completo
  - Lê PCAPs com Scapy
  - Serializa pacotes para JSON
  - Publica no tópico 'packets' (2909 pkt/s)
- ✅ Consumer v0.1 completo
  - Agrega pacotes em flows pela 5-tuple
  - Extrai 27 features por flow
  - Publica flows no tópico 'flows'
- ✅ Pipeline end-to-end testado e funcionando
- ✅ Documentação arquitetura Kafka (partições, offsets, consumer groups)

**Key Files Created:**

*Arquitetura (docs/architecture/):*
- `TARGET.md` - Visão de alto nível (onde queremos chegar)
- `CURRENT.md` - Estado atual da implementação (ATUALIZAR a cada evolução)
- `KAFKA_REFERENCE.md` - Referência educacional sobre Kafka

*Teoria (docs/theory/):*
- `teda-framework.md` - Fundamentação teórica TEDA/MicroTEDAclus
- `concept-drift.md` - Teoria de concept drift
- `pcap-justification.md` - Por que processar PCAPs

*Fichamentos (docs/paper-summaries/):*
- `angelov-2014-teda.md` - Fichamento TEDA (100%)
- `maia-2020-microtedaclus.md` - Fichamento MicroTEDAclus (100%)

*Outros:*
- `docs/KNOWLEDGE_GAPS.md` - Lacunas de conhecimento
- `streaming/src/producer/` - Producer v0.1
- `streaming/src/consumer/` - Consumer v0.1
- `streaming/src/detector/` - TEDADetector + StreamingDetector
- `streaming/docker/docker-compose.yml` - Kafka infrastructure

---

## 🔄 IN PROGRESS

### Completed: Fase 2A, Semana 3 (TEDA v0.1 Básico) ✅ 100%
**Goal:** Implementar TEDA v0.1 (básico) para detecção de anomalias em streaming
**Period:** 2026-01-19 to 2026-01-21

**Tasks Completed:**
- [x] Criar estrutura `streaming/src/detector/`
- [x] Implementar classe TEDADetector
  - [x] Atualização recursiva de μ (média)
  - [x] Atualização recursiva de σ² (variância)
  - [x] Cálculo de eccentricity: ξ = 1/k + ||x-μ||²/(k×σ²)
  - [x] Cálculo de typicality: τ = 1 - ξ
  - [x] Threshold para anomalia (Chebyshev)
- [x] Testes unitários básicos (33 testes, 100% passando)
- [x] Integração com Consumer (flows → TEDA)
- [x] Teste E2E: PCAP → detecção de anomalias
- [x] Atualizar documentação de arquitetura
- [x] Reorganização completa da documentação

**Deliverables:**
1. ✅ `streaming/src/detector/teda.py` - TEDADetector class (commit 7cdef2b)
2. ✅ `streaming/src/detector/streaming_detector.py` - StreamingDetector (commit 8a132b3)
3. ✅ `streaming/tests/test_teda.py` - 33 unit tests (100% passing)
4. ✅ Testes E2E: 127 flows, 2 anomalias detectadas (1.57%), 36.4 flows/s
5. ✅ `docs/architecture/CURRENT.md` - Diagrama completo do sistema
6. ✅ Documentação reorganizada (architecture/, theory/, paper-summaries/)

---

## 📅 ROADMAP ATUALIZADO (24 semanas)

### Fase 2A: Teoria + Design + Setup (Semanas 1-4)
**Goal:** Fundamentação sólida + ambiente pronto

| Semana | Foco Principal | Leituras | Entregáveis |
|--------|---------------|----------|-------------|
| **S1** ✅ | K-means, DBSCAN, TEDA, Design | - | Resumos, Arquitetura |
| **S2** ✅ | Setup Kafka, Producer+Consumer v0.1 | Angelov (2014), Maia (2020) | Pipeline E2E funcionando |
| **S3** ✅ | TEDA v0.1 (básico) | Survey Drift | TEDADetector + 33 testes |
| **S4** | TEDA v0.2 (micro-clusters) | Kafka Guide | MicroTEDAclus v0.1 |

### Fase 2B: Implementação TEDA + Kafka (Semanas 5-10)
**Goal:** MVP funcional com experimentos básicos

| Semana | Foco Principal | Leituras | Entregáveis |
|--------|---------------|----------|-------------|
| **S5** | TEDA v0.2 (micro-clusters) | Kafka Guide (1-3) | Multi-cluster funcionando |
| **S6** | Métricas de avaliação | Temporal Silhouette | Sistema de métricas |
| **S7** | Experimentos drift sintético | CICIoT2023 releitura | Primeiros resultados |
| **S8** | TEDA v0.3 (merge/split) | Kafka Guide (4-6) | MicroTEDAclus completo |
| **S9** | Experimentos comparativos | Survey IDS IoT | Comparação com Fase 1 |
| **S10** | Otimização, bug fixes | Edge IDS | MVP estável |

### Fase 2C: Experimentos + Validação (Semanas 11-14)
**Goal:** Resultados publicáveis

| Semana | Foco Principal | Leituras | Entregáveis |
|--------|---------------|----------|-------------|
| **S11** | Experimentos full dataset | IoT Security Survey | Resultados completos |
| **S12** | Análise concept drift | Mirai Analysis | Gráficos de adaptação |
| **S13** | Validação estatística | Métricas papers | Tabelas comparativas |
| **S14** | Documentação resultados | - | Capítulo de resultados |

### Fase 3: Otimização + Análise (Semanas 15-18)
**Goal:** Refinamento e análise profunda

| Semana | Foco Principal | Entregáveis |
|--------|---------------|-------------|
| **S15** | Performance tuning | Benchmarks otimizados |
| **S16** | Análise de escalabilidade | Gráficos de throughput |
| **S17** | Casos especiais, edge cases | Robustez documentada |
| **S18** | Preparação para dissertação | Outline completo |

### Fase 4: Dissertação + Defesa (Semanas 19-24)
**Goal:** Completar dissertação e defender

| Semana | Foco Principal | Entregáveis |
|--------|---------------|-------------|
| **S19-20** | Escrita dissertação (PT) | Caps 1-4 |
| **S21-22** | Escrita dissertação (PT) | Caps 5-7, revisão |
| **S23** | Tradução (EN) + revisão | Versão EN |
| **S24** | Preparação defesa | Slides, ensaio |

---

## 🧠 KEY DECISIONS LOG

### Decision 001: Development System Architecture (2025-11-08)
**Context:** Project delayed, need to accelerate development
**Decision:** Automated documentation system with skills/hooks
**Impact:** 2-3x acceleration expected

### Decision 002: PCAP Processing Required (2025-12-17)
**Context:** CSVs do CICIoT2023 são shuffled (paper linha 1839)
**Decision:** Processar PCAPs originais (~548GB) é MANDATÓRIO
**Impact:** Pipeline mais complexo, mas streaming válido
**Details:** `docs/summaries/pcap-processing-requirements.md`

### Decision 003: Integrated MVP Architecture (2025-12-17)
**Context:** Tempo limitado, Kafka era Fase 3 separada
**Decision:** Integrar Kafka desde o MVP, remover RF do escopo inicial
**Impact:** Foco em TEDA + Kafka, RF fica para evolução futura
**Details:** `docs/plans/2025-12-17-architecture-design.md`

### Decision 004: Mandatory Reading Plan (2025-12-17)
**Context:** Rigor acadêmico requer fundamentação nas 4 áreas
**Decision:** Mínimo 1 paper principal/semana, 8 principais + 12 auxiliares
**Impact:** Leituras integradas ao cronograma de desenvolvimento
**Details:** `docs/reading-plan.md`

---

## 📚 READING PLAN SUMMARY

### Four Areas of Knowledge

| Área | Papers Principais | Status |
|------|------------------|--------|
| **ML (Clustering)** | Angelov (2014), Maia (2020) | 2 completos ✅✅ |
| **Cibersegurança** | CICIoT2023, Survey IDS IoT | 1 parcial |
| **IoT** | Survey IoT Security, Edge IDS | 0 |
| **Arquitetura** | Streaming paper, Kafka Guide | 0 |

### Next Readings
1. **S2:** Angelov (2014) - TEDA Framework original ✅ COMPLETO
2. **S2:** Maia (2020) - MicroTEDAclus ✅ COMPLETO
3. **S3:** Survey Concept Drift (prioridade para implementar TEDA)
4. **S4:** Kafka Guide (implementação avançada)

**Full plan:** `docs/reading-plan.md`

---

## 🎓 LEARNING OBJECTIVES (Updated)

**ML Domain:**
- [x] K-means, DBSCAN fundamentals ✅
- [x] TEDA: eccentricidade, tipicalidade ✅
- [x] Concept drift types ✅
- [x] TEDA: fórmula recursiva e derivação matemática ✅
- [x] TEDA: Huygens-Steiner para O(n) ✅
- [x] TEDA: Data Clouds vs clusters tradicionais ✅
- [x] TEDA: critério τ > 1/k para novo protótipo ✅
- [x] TEDA: eficiência de memória (estatísticas suficientes) ✅
- [x] MicroTEDAclus: arquitetura micro + macro clusters ✅
- [x] MicroTEDAclus: threshold dinâmico m(k) ✅
- [x] MicroTEDAclus: mixture of typicalities ✅
- [x] MicroTEDAclus: critério de interseção dist < 2(σ_i + σ_j) ✅
- [ ] Métricas de avaliação para clustering (pesquisar)
- [ ] Validação estatística para streaming

**IoT Security Domain:**
- [x] CICIoT2023 structure (33 attacks, 7 categories) ✅
- [ ] IoT attack patterns in depth
- [ ] Real-time detection challenges

**Streaming/Infrastructure:**
- [x] Kafka 2-topic architecture designed ✅
- [x] Kafka implementation (Docker Compose) ✅
- [x] Kafka partições: paralelismo, distribuição por key ✅
- [x] Kafka offsets: committed, current, latest, lag ✅
- [x] Consumer Groups: identificação, rebalancing, auto_offset_reset ✅
- [x] Hot partitions: problema com DDoS, soluções ✅
- [x] Producer v0.1: PCAP → packets topic (2909 pkt/s) ✅
- [x] Consumer v0.1: packets → flows topic (27 features) ✅
- [ ] Performance benchmarking full dataset

---

## 📁 KEY DOCUMENTS

### Paper Summaries (Fichamentos)
- `docs/paper-summaries/angelov-2014-teda.md` - TEDA Framework original (100%)
- `docs/paper-summaries/maia-2020-microtedaclus.md` - MicroTEDAclus (100%)

### Architecture (Arquitetura)
- `docs/architecture/STREAMING_ARCHITECTURE.md` - Kafka pipeline completo
  - Partições, offsets, consumer groups
  - Hot partitions e soluções
  - Producer e Consumer configs

### Summaries (Fundamentação)
- `docs/summaries/clustering-evolutivo-concepts.md` - TEDA/MicroTEDAclus
- `docs/summaries/concept-drift-fundamentals.md` - 4 tipos de drift
- `docs/summaries/pcap-processing-requirements.md` - Pipeline PCAP + ferramentas

### Plans (Planejamento)
- `docs/plans/2025-12-17-architecture-design.md` - Arquitetura MVP
- `docs/reading-plan.md` - Plano de leituras 4 áreas

### Study Aids (Estudo)
- `docs/KNOWLEDGE_GAPS.md` - Lacunas de conhecimento para reforçar

### Reports (Acompanhamento)
- `docs/weekly-reports/current-week.md` - Relatório semanal atual

---

## 🛠️ DEVELOPMENT SETUP

**Primary Tools:**
- Claude Code with custom skills and hooks
- Python 3.12 + scikit-learn, pandas, numpy
- Scapy (packet parsing from PCAPs)
- Apache Kafka (Docker Compose - Zookeeper + Kafka + Kafka-UI)
- DVC for pipeline orchestration
- MLflow for experiment tracking (a validar)

**Streaming Infrastructure (Local - Docker):**
- Zookeeper: localhost:2181
- Kafka: localhost:9092
- Kafka-UI: localhost:8080
- Tópicos: packets (raw), flows (aggregated)

**Remote Resources:**
- PCAPs CICIoT2023 (~548GB) via SSH
- Processamento full dataset rodará na máquina remota

**Active Repositories:**
- `final-project/iot-ids-research/` - Phase 1: ML baseline experiments
- `final-project/streaming/` - Phase 2: Kafka + TEDA streaming (v0.1 funcionando)
- `final-project/data/` - Shared data (PCAPs, CSVs)
- `artigo1/` - Baseline comparison paper
- `dissertation/` - Master's dissertation (PT + EN)

---

## 🔍 RESEARCH GAPS (To Investigate)

| Gap | Área | Prioridade | Leituras Relacionadas |
|-----|------|------------|----------------------|
| Métricas de avaliação para clustering evolutivo | ML | Alta | ML-A1, ML-A3 |
| Design de experimentos de concept drift | ML/Cyber | Alta | ML-A1, ML-A2 |
| Sistema de tracking para streaming | Arq | Média | Testar MLflow |

---

## 💾 RECOVERY INSTRUCTIONS

**If session crashes or you need to resume:**

1. Open new Claude Code session
2. Type: `/resume` OR "Continue from SESSION_CONTEXT.md"
3. Claude will read this file and present current status

**Key files to read on resume:**
- This file (`SESSION_CONTEXT.md`)
- `docs/weekly-reports/current-week.md`
- `docs/plans/2025-12-17-architecture-design.md`

---

## 📝 WEEKLY REPORT STATUS

**Current Week Report:** `docs/weekly-reports/current-week.md`
**Status:** ~90% complete, needs finalization
**Last Finalized:** None yet (first week)

To finalize weekly report: `/finalize-week`

---

## 🔧 USEFUL COMMANDS

- `/resume` - Show current context and next steps
- `/start-sprint` - Begin new weekly sprint
- `/finalize-week` - Generate weekly report for advisor
- `/paper-summary <name>` - Summarize paper from Zotero

---

**END OF SESSION CONTEXT**

*This file is manually updated at the end of each session.*
*Use `/resume` in any new session to load this context.*
