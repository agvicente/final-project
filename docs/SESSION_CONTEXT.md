# SESSION CONTEXT - IoT IDS Research Project
**Last Updated:** 2026-01-05 (Session: Fichamento Angelov 2014 - 95% completo)

---

## 🎯 CURRENT STATUS

**Phase:** Fase 2A - Teoria + Design + Setup
**Week:** Semana 2 de 24 (~60% complete)
**Current Task:** Leitura MicroTEDAclus (Maia 2020), Setup Kafka

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

### Fase 2A, Semana 2: Leitura Angelov + Setup (60% COMPLETE)
- ✅ Fichamento Angelov (2014) - 95% completo
  - Conceitos: frequentista, belief/possibility theory, first principles
  - Métricas de distância: Euclidean, Manhattan, Mahalanobis, Cosine
  - Normalização e por que ξ = π normalizado
  - Fórmulas: π, ξ, τ com exemplos numéricos
  - Derivação matemática completa (Huygens-Steiner)
  - Seções 4-5: Anomaly Detection e Data Clouds
  - Limitações identificadas (zona de influência)
- ✅ Documento de lacunas de conhecimento criado
- ⏳ Relacionar TEDA com MicroTEDAclus (Maia 2020)
- ⏳ Setup ambiente Kafka remoto
- ⏳ Producer v0.1

**Key Files Created:**
- `docs/paper-summaries/angelov-2014-teda.md` - Fichamento detalhado (~95%)
- `docs/KNOWLEDGE_GAPS.md` - Lacunas de conhecimento para estudo

---

## 🔄 IN PROGRESS

### Current Week: Fase 2A, Semana 2 (Leitura Angelov + Setup)
**Goal:** Fichamento Angelov (2014) + Setup Kafka + Producer v0.1
**Started:** 2025-12-23

**Completed:**
- [x] Ler paper Angelov (2014) completo ✅
- [x] Criar fichamento estruturado ✅
- [x] Extrair fórmulas e pseudocódigo ✅
- [x] Documentar conceitos: frequentista, kernels, normalização ✅
- [x] Documentar métricas de distância ✅
- [x] Derivação matemática completa (Huygens-Steiner) ✅
- [x] Seções 4-5: Anomaly Detection e Data Clouds ✅
- [x] Identificar limitações do paper ✅
- [x] Criar documento de lacunas de conhecimento ✅

**Remaining:**
- [ ] Relacionar TEDA com MicroTEDAclus (Maia 2020)
- [ ] Setup Kafka ambiente remoto
- [ ] Producer v0.1 (PCAP reader)
- [ ] Atualizar relatório semanal

**Deliverables Created:**
1. `docs/paper-summaries/angelov-2014-teda.md` ✅ (95% completo)
2. `docs/KNOWLEDGE_GAPS.md` ✅ (novo)

---

## 📅 ROADMAP ATUALIZADO (24 semanas)

### Fase 2A: Teoria + Design + Setup (Semanas 1-4)
**Goal:** Fundamentação sólida + ambiente pronto

| Semana | Foco Principal | Leituras | Entregáveis |
|--------|---------------|----------|-------------|
| **S1** ✅ | K-means, DBSCAN, TEDA, Design | - | Resumos, Arquitetura |
| **S2** | Setup remoto, Producer v0.1 | Angelov (2014) | Ambiente Kafka rodando |
| **S3** | Consumer 1 (windowing) | Maia (2020) | Features extraídas |
| **S4** | TEDA v0.1 | Survey Drift | Pipeline básico E2E |

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
| **ML (Clustering)** | Angelov (2014), Maia (2020) | 1 completo ✅ |
| **Cibersegurança** | CICIoT2023, Survey IDS IoT | 1 parcial |
| **IoT** | Survey IoT Security, Edge IDS | 0 |
| **Arquitetura** | Streaming paper, Kafka Guide | 0 |

### Next Readings
1. **S2 (atual):** Angelov (2014) - TEDA Framework original ✅ COMPLETO
2. **S3:** Maia (2020) - MicroTEDAclus, releitura completa + Survey Drift

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
- [ ] Métricas de avaliação para clustering (pesquisar)
- [ ] Validação estatística para streaming

**IoT Security Domain:**
- [x] CICIoT2023 structure (33 attacks, 7 categories) ✅
- [ ] IoT attack patterns in depth
- [ ] Real-time detection challenges

**Streaming/Infrastructure:**
- [x] Kafka 2-topic architecture designed ✅
- [ ] Kafka implementation
- [ ] Performance benchmarking

---

## 📁 KEY DOCUMENTS

### Paper Summaries (Fichamentos)
- `docs/paper-summaries/angelov-2014-teda.md` - TEDA Framework original (95% completo)

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
- NFStream (feature extraction from PCAPs)
- Apache Kafka (Docker)
- DVC for pipeline orchestration
- MLflow for experiment tracking (a validar)

**Remote Resources:**
- PCAPs CICIoT2023 (~548GB) via SSH
- Processamento/Kafka rodará na máquina remota

**Active Repositories:**
- `final-project/iot-ids-research/` - Main research code
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
