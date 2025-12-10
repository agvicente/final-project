# SESSION CONTEXT - IoT IDS Research Project
**Last Updated:** 2025-12-09 (Session: K-means + DBSCAN fundamentals complete)

---

## 🎯 CURRENT STATUS

**Phase:** Fase 2 - Evolutionary Clustering
**Week:** Semana 1 de 10-12 (Teoria - Fundamentos)
**Current Task:** Estudar fundamentos de clustering e paper Maia et al. (2020)

---

## 📊 PROJECT OVERVIEW

**Master's Dissertation - UFMG PPGEE**
*Anomaly-based Intrusion Detection System for IoT Networks using Evolutionary Clustering*

**Timeline:** 5-7 months remaining
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

---

## 🔄 IN PROGRESS

### Current Week: Fase 2, Semana 1 (Teoria - Fundamentos)
**Goal:** Estudar fundamentos de clustering e clustering evolutivo
**Started:** 2025-12-04

**Tasks This Week:**
- [x] Revisar K-means: algoritmo, limitações, quando usar ✅
- [x] Revisar DBSCAN: density-based, parâmetros eps/min_samples ✅
- [x] Entender clustering particional vs density-based ✅
- [ ] Ler paper Maia et al. (2020) - Mixture of Typicalities ← PRÓXIMO
- [ ] Extrair pseudocódigo e parâmetros principais
- [ ] Entender como lida com concept drift
- [ ] Criar resumo estruturado dos conceitos
- [ ] Esboçar design inicial da arquitetura
- [ ] Preparar relatório semanal

**Entregáveis:**
1. Resumo de Clustering Fundamentals
2. Resumo do Paper Maia et al. 2020
3. Design Draft - esboço inicial da arquitetura
4. Relatório Semanal para orientador

**Previous Week (Setup Week - COMPLETE):**
- ✅ Sistema de aceleração completo e operacional
- ✅ 9 skills + 4 hooks + 4 comandos configurados
- ✅ Zotero integrado (references.bib: 161KB)
- ✅ Skill academic-paper-reviewer adicionada

---

## 📅 ROADMAP (30% Theory / 60% Practice / 10% Review)

### Fase 2: Evolutionary Clustering (10-12 weeks, ~120-150h)
**Goal:** Implement Mixture of Typicalities (Maia et al. 2020)

- Weeks 1-3 (30% theory): Study papers, K-means/DBSCAN review, design
- Weeks 4-9 (60% practice): Implement clustering evolutivo + experiments
- Weeks 10-12 (10% review): Analysis, validation, write chapter

### Fase 3: Streaming Architecture (10-12 weeks, ~120-150h)
**Goal:** Kafka + Real-time clustering integration

- Weeks 1-3 (30% theory): Streaming architectures, Kafka fundamentals
- Weeks 4-9 (60% practice): Build streaming pipeline + benchmarks
- Weeks 10-12 (10% review): Performance analysis, write chapter

### Fase 4: Finalization (6-8 weeks, ~80-100h)
**Goal:** Complete dissertation and prepare defense

- Weeks 1-2: Final optimizations
- Weeks 3-4: Write complete dissertation (PT)
- Weeks 5-6: Translate to English + review
- Weeks 7-8: Defense preparation

---

## 🧠 KEY DECISIONS LOG

### Decision 001: Development System Architecture (2025-11-08)
**Context:** Project delayed, need to accelerate development while maximizing learning
**Decision:** Implement automated documentation system with:
- 9 specialized skills for research context
- 4 hooks for auto-save, session recovery, weekly reports
- Iterative learning approach (concept → code → experiment → repeat)
- Weekly sprints aligned with advisor meetings

**Rationale:**
- 10-20h/week requires high efficiency
- Terminal crashes require protection against data loss
- Flexible meeting dates need continuous report generation
- Learning from scratch (clustering + streaming) needs guided approach

**Impact:** Expected 2-3x acceleration in development speed

---

## 🎓 LEARNING OBJECTIVES

**ML Domain:**
- Clustering algorithms (K-means, DBSCAN, hierarchical)
- Evolutionary clustering (Mixture of Typicalities)
- Concept drift detection and adaptation
- Statistical validation methods

**IoT Security Domain:**
- IoT attack patterns and taxonomy
- Network intrusion detection approaches
- Real-time anomaly detection
- CICIoT2023 dataset characteristics

**Streaming/Infrastructure:**
- Apache Kafka architecture
- Real-time data processing
- High-throughput system design
- Performance benchmarking

---

## 📚 KEY REFERENCES

**Core Papers:**
1. Maia et al. (2020) - Evolving clustering algorithm (Mixture of Typicalities)
2. Neto et al. (2023) - CICIoT2023 dataset
3. Surianarayanan et al. (2024) - High-throughput streaming architecture
4. Brodersen et al. - Bayesian accuracy evaluation (used in Phase 1)

**Zotero Library:** Auto-exported to `/Users/augusto/mestrado/references.bib`

---

## 🛠️ DEVELOPMENT SETUP

**Primary Tools:**
- Claude Code with custom skills and hooks
- Python 3.12 + scikit-learn, pandas, numpy
- DVC for pipeline orchestration
- MLflow for experiment tracking
- Docker for reproducibility
- Overleaf for papers and dissertation

**Active Repositories:**
- `final-project/iot-ids-research/` - Main research code
- `artigo1/` - Baseline comparison paper
- `dissertation/` - Master's dissertation (PT + EN)

---

## 💾 RECOVERY INSTRUCTIONS

**If session crashes or you need to resume:**

1. Open new Claude Code session
2. Type: `/resume` OR "Continue from SESSION_CONTEXT.md"
3. Claude will read this file and present current status
4. Auto-save hook protects against data loss (commits every 10min to wip/auto-save)

**If auto-save branch exists:**
- Claude will ask if you want to recover from interrupted session
- Say "yes" to continue from last saved state

---

## 📝 WEEKLY REPORT STATUS

**Current Week Report:** `docs/weekly-reports/current-week.md`
**Status:** In progress (continuously updated)
**Last Finalized:** None yet (first week)

To finalize weekly report: `/finalize-week`

---

## 🔧 SYSTEM CONFIGURATION

**Skills Active:**
- iot-ids-research-context (always)
- Additional skills load based on context

**Hooks Active:**
- auto-save-hook (every 10 minutes)
- session-start-hook (on startup)
- session-end-hook (on close)
- overleaf-validation-hook (on LaTeX commits)

**Useful Commands:**
- `/resume` - Show current context and next steps
- `/start-sprint` - Begin new weekly sprint
- `/finalize-week` - Generate weekly report for advisor
- `/paper-summary <name>` - Summarize paper from Zotero
- `/check-overleaf` - Validate LaTeX compilation

---

**END OF SESSION CONTEXT**

*This file is automatically updated by hooks and can also be manually edited.*
*Use `/resume` in any new session to load this context.*
