---
name: iot-ids-research-context
description: Core context skill that maintains complete project state, knows all code structure, research phases, and previous work. Always active to prevent repetitive explanations.
version: 1.0.0
author: Research Acceleration System
always_active: true
---

# IoT IDS Research Context

## Purpose

You are working on a Master's dissertation research project at UFMG (Federal University of Minas Gerais) in Electrical Engineering. This skill maintains complete context so you never need to re-explain the project, restate what's been done, or repeat basic information.

## Project Identity

**Title:** Anomaly-based Intrusion Detection System for IoT Networks using Evolutionary Clustering

**Researcher:** Augusto (Master's student)
**Institution:** UFMG PPGEE (Programa de Pós-Graduação em Engenharia Elétrica)
**Timeline:** 5-7 months remaining for completion
**Weekly Dedication:** 10-20 hours
**Meetings:** Weekly with advisor (flexible scheduling)

## Research Problem

IoT networks face critical security challenges:
- Heterogeneous devices with varying capabilities
- High-velocity data streams requiring real-time detection
- Concept drift (traffic patterns change over time)
- Resource constraints on edge devices
- High false positive rates in traditional IDS

**Core Innovation:** Using evolutionary clustering algorithms (Mixture of Typicalities) that adapt to concept drift in streaming IoT network traffic.

## Current Phase Status

**Phase 1:** ✅ COMPLETE (100%)
**Phase 2:** 🔄 STARTING (Evolutionary Clustering)
**Phase 3:** 📋 PLANNED (Streaming Architecture)
**Phase 4:** 📋 PLANNED (Finalization)

### Phase 1 Achievements (DO NOT RE-DO)
- 705 experiments across 10 ML algorithms
- Excellent baseline results: F1 > 0.99 on CICIoT2023 dataset
- Reproducible DVC pipeline established
- Docker + MLflow infrastructure working
- Best performers identified:
  - GradientBoostingClassifier: F1 0.9964 (best accuracy)
  - SGDOneClassSVM: 128.3s (fastest)
  - RandomForest: Good balance
- Paper in progress (artigo1) documenting baseline comparison

**Files Location:** `iot-ids-research/experiments/.results/full/`

##Code Structure (MEMORIZE THIS)

```
/Users/augusto/mestrado/
├── final-project/              # Main research repository
│   ├── iot-ids-research/       # Core research code
│   │   ├── data/
│   │   │   ├── raw/CSV/MERGED_CSV/  # CICIoT2023 original
│   │   │   └── processed/           # Preprocessed data
│   │   │       ├── sampled.csv      # 10% stratified sample
│   │   │       └── binary/          # Binary classification data
│   │   ├── src/
│   │   │   ├── clustering/     # NEW: Evolutionary clustering (Phase 2)
│   │   │   ├── streaming/      # NEW: Kafka pipeline (Phase 3)
│   │   │   └── eda/           # Exploratory analysis
│   │   ├── experiments/        # Experiment orchestration
│   │   │   ├── algorithm_comparison.py  # Core experiment runner
│   │   │   ├── run_single_algorithm.py  # DVC wrapper
│   │   │   ├── consolidate_results.py   # Results aggregation
│   │   │   └── .results/       # All experiment results
│   │   ├── configs/           # Configuration files
│   │   ├── docs/              # Research documentation
│   │   │   ├── SESSION_CONTEXT.md     # PROJECT BRAIN - READ FIRST
│   │   │   ├── weekly-reports/        # Advisor meeting reports
│   │   │   ├── progress/              # Daily session logs
│   │   │   ├── decisions/             # Technical decisions log
│   │   │   └── plans/                 # Design documents
│   │   ├── dvc.yaml           # DVC pipeline definition
│   │   ├── requirements.txt   # 250+ Python dependencies
│   │   └── docker-compose.yml # Multi-service setup
│   ├── .claude/              # Skills, hooks, commands
│   └── CLAUDE.md             # Claude Code guidance
├── artigo1/                   # Baseline comparison paper (Overleaf)
├── dissertation/              # Master's dissertation (PT + EN)
└── references.bib            # Zotero auto-exported bibliography
```

## Dataset: CICIoT2023

**Source:** Canadian Institute for Cybersecurity
**Type:** Real IoT network traffic with labeled attacks
**Current Usage:** 10% stratified sample (maintains attack distribution)
**Classification:** Binary (Benign=0 vs Attack=1) for Phase 1-2
**Files:**
- Train: `data/processed/binary/X_train_binary.npy`, `y_train_binary.npy`
- Test: `data/processed/binary/X_test_binary.npy`, `y_test_binary.npy`
- Metadata: `data/processed/binary/binary_metadata.json`

## Research Methodology

**Learning Style:** Iterative (30% theory / 60% practice / 10% review)
- Learn basic concept → implement minimal version → experiment → review theory → expand
- Weekly sprints with demo for advisor
- Just-in-time learning (don't study everything upfront)

**Experiment Standards (from Phase 1):**
- 5 runs per configuration for statistical rigor
- Grid search for hyperparameter optimization
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Balanced Accuracy
- Bayesian statistical validation
- Resource monitoring (CPU, memory, time)
- MLflow tracking for all experiments

## Phase 2: Evolutionary Clustering (CURRENT)

**Duration:** 10-12 weeks
**Goal:** Implement Mixture of Typicalities (Maia et al. 2020)

**Key Papers to Study:**
1. Maia et al. (2020) - Core algorithm (Mixture of Typicalities)
2. Lu et al. (2019) - Concept drift theory
3. Wahab (2022) - IoT concept drift detection

**Implementation Plan:**
- Weeks 1-3: Study papers, review clustering fundamentals (K-means, DBSCAN)
- Weeks 4-9: Implement evolutionary clustering + experiments
- Weeks 10-12: Analysis, validation, write dissertation chapter

**Current Task:** Setting up development system (this week)
**Next Task:** Begin clustering fundamentals review (next week)

## Phase 3: Streaming Architecture (FUTURE)

**Duration:** 10-12 weeks
**Goal:** Kafka-based real-time streaming with evolutionary clustering integration

**Key Concepts:**
- Apache Kafka for data streaming
- Real-time anomaly detection
- High-throughput architecture
- Latency and performance benchmarking

## Key References (in Zotero)

**Always available in:** `/Users/augusto/mestrado/references.bib`

- Maia et al. (2020) - Evolutionary clustering (core)
- Neto et al. (2023) - CICIoT2023 dataset
- Surianarayanan et al. (2024) - Streaming architecture
- Brodersen et al. - Bayesian accuracy (used in Phase 1)
- 220+ additional papers catalogued by phase

## Development Tools & Commands

**DVC Pipeline:**
```bash
cd iot-ids-research
dvc repro                    # Run full pipeline
dvc repro exp_<algorithm>   # Run specific experiment
```

**Docker Services:**
```bash
docker-compose up -d        # Start Jupyter + MLflow
# Access: Jupyter (8888), MLflow (5000)
```

**Python Environment:**
- Python 3.12
- Key libraries: scikit-learn 1.7.1, pandas 2.3.1, numpy 2.3.2, MLflow 3.1.4, DVC 3.61.0

**Custom Commands (use these):**
- `/resume` - Show current context and next steps
- `/start-sprint` - Begin new weekly sprint
- `/finalize-week` - Generate weekly report for advisor
- `/paper-summary <name>` - Summarize paper from Zotero

## Researcher's Knowledge Level

**ML/Data Science:** Intermediate
- Knows Python, pandas, scikit-learn basics
- Completed Phase 1 experiments successfully
- Needs to learn: Clustering algorithms (starting from basics), evolutionary approaches

**IoT Security:** Learning
- Understands IoT security challenges conceptually
- Knows CICIoT2023 dataset structure
- Needs to deepen: Attack taxonomy, detection patterns

**Streaming/Infrastructure:** Beginner
- Has Docker/DVC experience from Phase 1
- Needs to learn: Kafka, streaming architectures, real-time processing

## Critical Guidelines

### DO NOT:
- ❌ Re-explain what Phase 1 accomplished (it's done, move forward)
- ❌ Suggest re-running baseline experiments (results are validated)
- ❌ Ask "what is your research about" (read this skill)
- ❌ Propose changing the core approach (evolutionary clustering is the thesis)
- ❌ Assume unlimited time (5-7 months, 10-20h/week is reality)

### ALWAYS:
- ✅ Read `docs/SESSION_CONTEXT.md` first when resuming
- ✅ Update SESSION_CONTEXT.md after significant progress
- ✅ Use iterative learning approach (small steps, not big lectures)
- ✅ Generate weekly reports in `docs/weekly-reports/current-week.md`
- ✅ Protect against data loss (auto-save every 10min)
- ✅ Keep advisor meeting format in mind (code + results + insights)

### Working Style:
- Augusto learns best by doing, not long theoretical explanations
- Prefers concrete code examples over abstract concepts
- Values efficiency (limited time availability)
- Needs protection against terminal crashes (happens on his Mac)
- Weekly meetings need demonstrable progress

## Session Recovery

If session is interrupted or resumed:
1. Check `docs/SESSION_CONTEXT.md` for current state
2. Check `docs/weekly-reports/current-week.md` for week progress
3. Check git branch `wip/auto-save` for latest auto-saved work
4. Summarize: "You were doing X, got to Y, next is Z"

## Communication Style

- Be direct and efficient (respect limited time)
- No unnecessary pleasantries or praise
- Focus on actionable next steps
- Explain "why" for technical decisions (learning objective)
- Use Portuguese for dissertation-related work when appropriate

---

**This skill ensures Claude always has full project context without repetitive explanations.**
