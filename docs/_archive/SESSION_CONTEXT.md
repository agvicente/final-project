# SESSION CONTEXT - IoT IDS Research Project
**Last Updated:** 2026-02-25 19:15 (Session: Semana 5 - Isolamento + Validação Completos)

---

## 🎯 CURRENT STATUS

**Phase:** Fase 2B - Experimentos Streaming
**Week:** Semana 5 de 24 (20.8% complete) ✅ ISOLAMENTO + VALIDAÇÃO COMPLETOS
**Current Task:** Sistema 100% funcional, isolamento garantido, validação confirmada
**Next Session:** Executar grid 3×2 (TEDA vs MicroTEDAclus) + experimento DDoS
**Estimated Time:** ~90 minutos (60 min grid + 15 min DDoS + 15 min análise)

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
  - 3 partições por tópico (packets, flows, alerts)
- ✅ Producer v0.1 completo
  - Lê PCAPs com Scapy
  - Serializa pacotes para JSON
  - Publica no tópico 'packets' (2909 pkt/s)
- ✅ Consumer v0.1 completo
  - Agrega pacotes em flows pela 5-tuple
  - Extrai 28 features ML + 8 metadata (36 total)
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

*Desenvolvimento (docs/development/):*
- `microtedaclus-implementation-notes.md` - Lições aprendidas na implementação

*Outros:*
- `docs/KNOWLEDGE_GAPS.md` - Lacunas de conhecimento
- `streaming/src/producer/` - Producer v0.1
- `streaming/src/consumer/` - Consumer v0.1
- `streaming/src/detector/` - TEDADetector + MicroTEDAclus + StreamingDetector v0.2
- `streaming/docker/docker-compose.yml` - Kafka infrastructure

### Fase 2A, Semana 5: Orquestração + Experimentos + Isolamento (✅ FASE B COMPLETA - 2026-02-25)
**Goal:** Implementar infraestrutura completa para experimentos streaming + Isolamento total

**Fase A - Preparação (✅ COMPLETO):**
- ✅ GroundTruthProvider: Heurística filename-based para 33+ ataques CICIoT2023
- ✅ PrequentialMetrics v0.2: Implementação rigorosa (Gama et al. 2013)
  - Três estimadores: cumulative, sliding window, fading factor
  - MTTD corrigido (ignora FP antes do ataque)
  - Parâmetros justificados: window=1000, alpha=0.01
- ✅ StreamingDetector estendido: aceita ground_truth + metrics como parâmetros opcionais
- ✅ run_experiment.py v1.0: Orquestrador básico

**Fase B - Isolamento + Validação (✅ COMPLETO - 2026-02-25):**
- ✅ Feature de isolamento total entre experimentos (TDD rigoroso)
  - purge_kafka_topics(): Purga automática de tópicos (~2s overhead)
  - Group IDs únicos por experimento (timestamp-based)
  - Flag --skip-purge para debugging
  - 11/11 testes unitários passando
- ✅ Bugs críticos corrigidos:
  - `--attack-pcap none` tratado como string literal
  - `metrics.get_metrics()` → `metrics.get_global_metrics()`
  - Logging de métricas usando dicionário correto
- ✅ Validação experimental executada:
  - Teste rápido (100 flows): 6.76s, 58 clusters, 49% anomalias
  - Teste consolidação (2000 flows): 9.73s, 96 clusters, 4.3% anomalias
  - Cluster dominante: 1822 flows (91% do tráfego) - consolidação confirmada!
  - Throughput: 444 flows/s

**Fase B - Execução (✅ COMPLETO - 2026-02-25):**
- ✅ **5 Artefatos Estruturados** salvos por execução:
  1. `run_meta.json` - Git commit, parâmetros, timestamps
  2. `detection_results.json` - Resultados completos de detecção
  3. `metrics_windowed.csv` - Precision, Recall, F1, FPR, erros prequential
  4. `clusters_state.jsonl` - Snapshots de clusters
  5. `system_usage.csv` - CPU/memória com psutil
- ✅ **System monitoring**: CPU e memória RSS coletados
- ✅ **Cluster snapshots**: Estado dos micro-clusters salvo
- ✅ **Múltiplos PCAPs**: Suporte a `--pcap` (benign) + `--attack-pcap` (ataque) processados sequencialmente
- ✅ **compare_experiments.py**: Script básico para comparar múltiplas execuções
  - Tabela comparativa no terminal
  - Relatório Markdown gerado automaticamente
- ✅ README.md principal atualizado com documentação completa

**Deliverables:**
1. ✅ `streaming/scripts/run_experiment.py` - Orquestrador completo com 5 artefatos
2. ✅ `streaming/scripts/compare_experiments.py` - Comparador básico
3. ✅ `streaming/src/metrics/ground_truth.py` - 48/48 testes passando
4. ✅ `streaming/src/metrics/prequential_metrics.py` - Métricas academicamente rigorosas
5. ✅ `docs/paper-summaries/gama-2013-prequential-evaluation.md` - Fichamento completo
6. ✅ `README.md` - Documentação unificada do projeto

**✅ PRÓXIMOS PASSOS (Próxima Sessão - 2026-02-26):**

**1. Grid de Parâmetros - TEDA vs MicroTEDAclus (6 experimentos)**
```bash
cd streaming
for algo in teda micro_teda; do
    for r0 in 0.05 0.10 0.20; do
        python3 scripts/run_experiment.py \
            --pcap ../data/raw/PCAP/Benign/BenignTraffic.pcap \
            --max-packets 50000 \
            --max-flows 5000 \
            --algorithm $algo \
            --r0 $r0 \
            --output ../results/week5/grid_${algo}_r0_${r0}/
    done
done
```
**Estimativa:** ~10 min/experimento × 6 = ~60 minutos

**2. Comparação de Resultados**
```bash
python3 scripts/compare_experiments.py ../results/week5/
# Gera: comparison_report.md com tabela consolidada
```

**3. Experimento com Ataque DDoS (validar detecção)**
```bash
python3 scripts/run_experiment.py \
    --pcap ../data/raw/PCAP/Benign/BenignTraffic.pcap \
    --attack-pcap ../data/raw/PCAP/DDoS/DDoS-ICMP_Flood.pcap \
    --max-packets 50000 \
    --max-flows 10000 \
    --output ../results/week5/ddos_detection/
# Validar: Precision, Recall, F1, MTTD
```

**4. Análise e Documentação**
- Gráficos: consolidação de clusters, taxa de anomalia
- Tabela comparativa: TEDA vs MicroTEDAclus
- Atualizar semana5-report.md com resultados finais

**Critérios de Sucesso:**
- ✅ FPR <= 5% durante benign (já validado: 4.3%)
- [ ] Recall >= 80% durante ataque DDoS
- [ ] MTTD <= 500 flows
- [ ] Diferença clara entre TEDA e MicroTEDAclus

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

### Completed: Fase 2A, Semana 4 (MicroTEDAclus v0.1) ✅ 100%
**Goal:** Implementar MicroTEDAclus para resolver problema de contaminação
**Period:** 2026-01-25

**Tasks Completed:**
- [x] Analisar problema de contaminação no TEDA básico
- [x] Documentar problema (`docs/theory/teda-contamination-problem.md`)
- [x] Implementar classe MicroCluster com estatísticas isoladas
- [x] Implementar threshold dinâmico m(k)
- [x] Implementar Chebyshev rejection + criação de novos clusters
- [x] Implementar MicroTEDAclus (orquestrador)
- [x] Criar 31 testes unitários (100% passando)
- [x] Testar resistência a contaminação
- [x] Documentar lições aprendidas
- [x] Integrar MicroTEDAclus com StreamingDetector v0.2
- [x] Atualizar exports no __init__.py

**Deliverables:**
1. ✅ `streaming/src/detector/micro_teda.py` - MicroCluster + MicroTEDAclus (~400 linhas)
2. ✅ `streaming/tests/test_micro_teda.py` - 31 testes unitários
3. ✅ `docs/theory/teda-contamination-problem.md` - Análise do problema
4. ✅ `docs/development/microtedaclus-implementation-notes.md` - Lições aprendidas
5. ✅ `streaming/src/detector/streaming_detector.py` v0.2 - Suporte dual TEDA/MicroTEDAclus
6. ✅ Total: 67 testes passando (36 TEDA + 31 MicroTEDAclus)

**Key Learnings:**
- r0 (variância mínima) deve ser proporcional à escala dos dados
- Threshold para n=1 precisa ser permissivo (13.0) mas não infinito
- Clusters jovens precisam de tratamento especial
- Comparação direta com TEDA básico valida a solução
- Strategy pattern permite seleção de algoritmo em runtime

---

## 📅 ROADMAP ATUALIZADO (24 semanas)

### Fase 2A: Teoria + Design + Setup (Semanas 1-4)
**Goal:** Fundamentação sólida + ambiente pronto

| Semana | Foco Principal | Leituras | Entregáveis |
|--------|---------------|----------|-------------|
| **S1** ✅ | K-means, DBSCAN, TEDA, Design | - | Resumos, Arquitetura |
| **S2** ✅ | Setup Kafka, Producer+Consumer v0.1 | Angelov (2014), Maia (2020) | Pipeline E2E funcionando |
| **S3** ✅ | TEDA v0.1 (básico) | Survey Drift | TEDADetector + 33 testes |
| **S4** ✅ | MicroTEDAclus v0.1 | - | MicroTEDAclus + StreamingDetector v0.2 |

### Fase 2B: Implementação TEDA + Kafka (Semanas 5-10)
**Goal:** MVP funcional com experimentos básicos

| Semana | Foco Principal | Leituras | Entregáveis |
|--------|---------------|----------|-------------|
| **S5** | **Orquestração + Experimentos E2E** (Preparação + Execução) | Kafka Guide (1-3) | Infraestrutura completa + primeiros resultados |
| **S6** | Métricas de avaliação | Temporal Silhouette | Sistema de métricas |
| **S7** | Experimentos drift sintético | CICIoT2023 releitura | Primeiros resultados |
| **S8** | TEDA v0.3 (merge/split) | Kafka Guide (4-6) | MicroTEDAclus completo |
| **S9** | Experimentos comparativos | Survey IDS IoT | Comparação streaming vs literatura |
| **S10** | Otimização, bug fixes | Edge IDS | MVP estável |

**Nota:** S5 original previa "TEDA v0.2 (micro-clusters)" mas isso foi antecipado e concluído na S4 com MicroTEDAclus.

---

### Detalhes S5: Orquestração + Experimentos E2E (EXPANDIDO)

**Objetivo Geral:** Implementar toda infraestrutura de orquestração E fazer os primeiros experimentos reais.

#### FASE A: Preparação - Infraestrutura (Estimativa: 13-19h)

**Objetivo:** Implementar infraestrutura para que experimentos sejam apenas "executar script".

**O que JÁ ESTÁ PRONTO (não reimplementar):**
- ✅ Kafka infrastructure (Docker Compose)
- ✅ PCAPProducer (~600 LOC, testado)
- ✅ FlowConsumer (~500 LOC, testado)
- ✅ TEDADetector + MicroTEDAclus (~650 LOC, 67 testes passando)
- ✅ StreamingDetector (~600 LOC)
- **Total:** 2350 linhas funcionais

**O que FALTA IMPLEMENTAR:**
1. Script de orquestração `run_experiment.py` (~400 LOC)
2. Sistema de métricas prequential (~200 LOC)
3. Ground truth heurístico (~50 LOC)
4. Script de comparação `compare_experiments.py` (~200 LOC)
5. Verificação/download de PCAPs
6. Testes de integração E2E

**Ordem de Implementação (3 dias):**

**Dia 1: Estrutura Base (4-5h)**
- [x] Verificar PCAPs localmente (benign + DDoS-ICMP)
- [x] Criar estrutura de diretórios (data/pcaps/, results/week5/)
- [x] Implementar `ground_truth.py` (inferência por filename)
- [x] Implementar `prequential_metrics.py` (sliding window)

**Dia 2: Orquestração (6-8h)**
- [ ] Implementar `run_experiment.py` estrutura base (CLI + classes)
- [ ] Implementar fase de execução (Kafka + Producer + Consumer + Detector)
- [ ] Implementar coleta de métricas em tempo real

**Dia 3: Validação e Análise (3-6h)**
- [ ] Sanity check com PCAP pequeno (~1000 flows)
- [ ] Implementar `compare_experiments.py`
- [ ] Documentar e commitar

**Componentes Críticos:**

**1. Ground Truth Heurístico (`src/metrics/ground_truth.py`):**
- Infere label de ataque do nome do arquivo PCAP
- Limitação: Todos flows do mesmo PCAP recebem mesmo label
- Melhoria futura (S6+): Integrar com CSVs do CICIoT2023

**2. Métricas Prequential (`src/metrics/prequential_metrics.py`):**
- Sliding window (default: 1000 flows)
- Métricas: Precision, Recall, F1, FPR, MTTD
- Baseado em Gama et al. (2013) - Test-then-train

**3. Script de Orquestração (`scripts/run_experiment.py`):**
- Verifica conexão Kafka (pre-flight check)
- Inicia Producer → Consumer → Detector em sequência
- Processa PCAP especificado
- Coleta métricas e logs centralizados
- Para tudo automaticamente ao finalizar
- Parâmetros: `--pcap`, `--attack-pcap`, `--algorithm`, `--output`
- Gera 5 artefatos: run_meta.json, detection_results.json, metrics_windowed.csv, clusters_state.jsonl, system_usage.csv

**Critérios de "Pronto" (Fase A):**
- ✅ Script `run_experiment.py` funciona end-to-end com 1 PCAP pequeno
- ✅ Sistema de métricas coletando P/R/F1 em tempo real
- ✅ 5 artefatos sendo gerados automaticamente
- ✅ PCAPs necessários disponíveis localmente
- ✅ Sanity check passando (benign sem alertas excessivos)

---

#### FASE B: Execução - Primeiros Experimentos (Estimativa: 8-12h)

**Objetivo:** Executar os primeiros experimentos reais e validar o sistema.

**Cenário A: Detecção Básica**
- Warm-up: Benign (5-10 min, ~1000-2000 flows)
- Teste: Benign + DDoS-ICMP (30 min, ~5000-10000 flows)
- Métricas: Precision, Recall, F1, MTTD (Mean Time To Detection)
- Expectativa: F1 > 0.90, MTTD < 100 flows

**Cenário Sanity Check:**
- Apenas Benign (10 min)
- Expectativa: FPR < 0.05 (menos de 5% de falsos positivos)

**Entregáveis Fase B:**
1. Resultados de 2 experimentos (Sanity + Cenário A)
2. Gráficos temporais de F1, Precision, Recall
3. Análise de MTTD
4. Benchmark de throughput (flows/segundo)
5. Documentação inicial em `results/week5/report.md`

**Experimentos a Executar:**
```bash
# Pré-requisito: Kafka rodando
cd /Users/augusto/mestrado/final-project
docker-compose up -d kafka zookeeper
# Aguardar ~30s para inicialização

# Sanity Check
cd streaming
python scripts/run_experiment.py \
  --pcap ../data/pcaps/benign/BenignTraffic.pcap \
  --max-packets 2000 \
  --max-flows 500 \
  --algorithm micro_teda \
  --output ../results/week5/sanity/

# Cenário A: Detecção Básica
python scripts/run_experiment.py \
  --pcap ../data/pcaps/benign/BenignTraffic.pcap \
  --attack-pcap ../data/pcaps/ddos/DDoS-ICMP_Flood.pcap \
  --max-packets 5000 \
  --max-flows 10000 \
  --algorithm micro_teda \
  --output ../results/week5/scenarioA/
```

**Análise dos Resultados:**
- Comparar métricas: MicroTEDAclus vs TEDA básico
- Identificar problemas de threshold (muitos FP? muitos FN?)
- Documentar comportamento durante warm-up
- Validar resistência a contaminação

**Critérios de Sucesso (Fase B):**
- ✅ Sanity check: FPR < 0.05 em tráfego 100% benign
- ✅ Cenário A: F1 > 0.90 na detecção de DDoS
- ✅ MTTD < 100 flows (detecção rápida)
- ✅ Sistema roda sem crashes por > 30 min
- ✅ Throughput > 100 flows/s

---

**Resumo da Semana 5:**
- **Fase A (Preparação):** 13-19h → Infraestrutura completa
- **Fase B (Execução):** 8-12h → Primeiros resultados validados
- **Total:** 21-31h de trabalho
- **Entregáveis:** Script orquestrador + 2 experimentos documentados + benchmark

**Referência Detalhada:** `docs/methodology/experiment-methodology.md` seções 8.0.5.x

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

### Metodologia (docs/methodology/)
- `experiment-methodology.md` - Design de experimentos com PCAPs, avaliação prequential, cenários de concept drift

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
- Tópicos: packets (raw), flows (aggregated), alerts (anomalies)

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
