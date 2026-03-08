# Anomaly-Based IDS for IoT Networks using Evolutionary Clustering

**Dissertacao de Mestrado** - Programa de Pos-Graduacao em Engenharia Eletrica (UFMG)

Sistema de deteccao de intrusao para redes IoT usando algoritmos de clustering evolutivo com arquitetura streaming.

**Prazo defesa:** ~maio 2026 | **Dataset:** CICIoT2023

---

## Estrutura do Repositorio

```
final-project/
├── STATUS.md                    # Estado atual (leia primeiro)
├── USAGE.md                     # Guia de uso por cenario
├── CLAUDE.md                    # Instrucoes para Claude Code
│
├── research/                    # CONHECIMENTO
│   ├── bibliography.bib         # Referencias BibTeX consolidadas
│   ├── reading-log.md           # Leituras + lacunas de conhecimento
│   ├── summaries/               # Fichamentos de papers
│   └── foundations/             # Teoria (TEDA, concept drift)
│
├── experiments/                 # EVIDENCIA
│   ├── methodology.md           # Metodologia cientifica (cap. 4)
│   ├── streaming/               # Pipeline Kafka + Clustering Evolutivo
│   │   ├── src/                 # Producer, Consumer, Detector, Metrics
│   │   ├── scripts/             # Orquestrador de experimentos
│   │   ├── tests/               # Testes unitarios
│   │   └── docker/              # Kafka + Zookeeper
│   ├── baseline/                # Fase 1: ML baseline (completo)
│   └── results/                 # Resultados das campanhas experimentais
│
├── writing/                     # PRODUCAO
│   ├── dissertation/            # Clone do Overleaf
│   ├── figures/                 # Figuras geradas
│   └── tables/                  # Tabelas geradas
│
├── data/                        # Dados
│   └── raw/PCAP/                # PCAPs do CICIoT2023
│
└── docs/                        # OPERACIONAL
    ├── architecture/            # Arquitetura implementada
    ├── progress/                # Logs automaticos de sessao
    └── plans/                   # Planos de execucao
```

---

## Quick Start

### Streaming (Fase 2 - Atual)

```bash
# Iniciar Kafka
cd experiments/streaming/docker && docker-compose up -d

# Ativar ambiente Python
cd experiments/streaming && source venv/bin/activate

# Rodar experimento
python3 scripts/run_experiment.py \
  --pcap ../../data/raw/PCAP/Benign/BenignTraffic.pcap \
  --max-packets 50000 --max-flows 5000 \
  --algorithm micro_teda --r0 0.10 \
  --output ../results/campaign-01/test/

# Rodar testes
python3 -m pytest tests/ -q
```

### Baseline (Fase 1 - Completo)

```bash
cd experiments/baseline
dvc repro                                              # Pipeline completo
python experiments/run_single_algorithm.py random_forest  # Algoritmo especifico
```

**Resultados Fase 1:** 705 experimentos, 10 algoritmos, F1 > 0.99.

---

## Fases da Pesquisa

| Fase | Objetivo | Status |
|------|----------|--------|
| **Fase 1** | Baseline ML com algoritmos classicos | Completo |
| **Fase 2** | Streaming + Clustering Evolutivo (TEDA/MicroTEDAclus) | Em andamento |
| **Fase 3** | Modelos device-specific + Two-phase | Nice-to-have |
| **Fase 4** | Otimizacao + Dissertacao | Nice-to-have |

---

## Documentacao

| O que | Onde |
|-------|------|
| Estado atual | `STATUS.md` |
| Como usar o repositorio | `USAGE.md` |
| Arquitetura implementada | `docs/architecture/CURRENT.md` |
| Metodologia cientifica | `experiments/methodology.md` |
| Fichamentos de papers | `research/summaries/` |
| Teoria consolidada | `research/foundations/` |

---

## Referencias

- Angelov, P. (2014). "Outside the box: an alternative data analytics framework." *JAMRIS*.
- Maia, J. et al. (2020). "Evolving clustering algorithm based on mixture of typicalities." *FGCS*.
- Gama, J. et al. (2013). "On evaluating stream learning algorithms." *Machine Learning*.
- Neto, E.C.P. et al. (2023). "CICIoT2023: A Real-Time Dataset and Benchmark." *Sensors*.

---

**Autor:** Augusto - Mestrado em Engenharia Eletrica (UFMG)
**Orientador:** Frederico Gadelha Guimaraes
