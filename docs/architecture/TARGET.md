# Arquitetura Alvo - IoT IDS com Clustering Evolutivo

**Criado:** 2025-12-17
**Última Atualização:** 2026-01-20
**Status:** Documento de Visão (Alto Nível)

> **Propósito:** Este documento descreve ONDE QUEREMOS CHEGAR. Para o estado atual da implementação, veja [CURRENT.md](./CURRENT.md).

---

## 1. Objetivo

Implementar um **Sistema de Detecção de Intrusão (IDS)** para redes IoT usando:
- **Clustering evolutivo** (TEDA/MicroTEDAclus)
- **Arquitetura de streaming** via Kafka
- **Adaptação a concept drift** em tempo real

### 1.1 Escopo

| Fase | Objetivo | Status |
|------|----------|--------|
| **Fase A** | Pipeline PCAP → Kafka → TEDA funcional | ✅ Completo |
| **Fase B** | MicroTEDAclus com micro/macro clusters | Em andamento |
| **Fase C** | Comparação com Fase 1 (mesmas métricas) | Planejado |
| **Fase D** | Otimização e escalabilidade | Planejado |

---

## 2. Arquitetura Alvo

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ARQUITETURA IoT IDS - VISÃO COMPLETA                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐                                                           │
│  │    PCAPs     │ (CICIoT2023 - ~548GB)                                     │
│  │  ou Tráfego  │                                                           │
│  │    Real      │                                                           │
│  └──────┬───────┘                                                           │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────┐                                                           │
│  │   Producer   │  Lê pacotes, preserva timestamps                          │
│  └──────┬───────┘                                                           │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                          KAFKA CLUSTER                               │   │
│  │   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │   │
│  │   │ topic: packets  │    │  topic: flows   │    │  topic: alerts  │ │   │
│  │   │   (raw pkts)    │    │   (features)    │    │   (anomalias)   │ │   │
│  │   └────────┬────────┘    └────────┬────────┘    └────────▲────────┘ │   │
│  └────────────│─────────────────────│──────────────────────│───────────┘   │
│               │                      │                      │               │
│               ▼                      │                      │               │
│  ┌────────────────────────┐         │                      │               │
│  │  Flow Aggregator       │         │                      │               │
│  │  - Agrupa N pacotes    │─────────┘                      │               │
│  │  - Extrai features     │                                │               │
│  └────────────────────────┘                                │               │
│                                      │                      │               │
│                                      ▼                      │               │
│                         ┌────────────────────────┐         │               │
│                         │   DETECTOR EVOLUTIVO   │         │               │
│                         │  ┌──────────────────┐  │         │               │
│                         │  │  MicroTEDAclus   │  │         │               │
│                         │  │  - Micro-clusters │  │         │               │
│                         │  │  - Macro-clusters │  │─────────┘               │
│                         │  │  - Concept drift  │  │                         │
│                         │  └──────────────────┘  │                         │
│                         └────────────────────────┘                         │
│                                      │                                      │
│                                      ▼                                      │
│                         ┌────────────────────────┐                         │
│                         │   Dashboard / SIEM     │                         │
│                         │   (Visualização)       │                         │
│                         └────────────────────────┘                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Componentes Planejados

### 3.1 Pipeline de Dados

| Componente | Entrada | Saída | Status |
|------------|---------|-------|--------|
| **PCAPProducer** | Arquivos PCAP | `topic:packets` | ✅ Implementado |
| **FlowConsumer** | `topic:packets` | `topic:flows` | ✅ Implementado |
| **StreamingDetector** | `topic:flows` | `topic:alerts` | ✅ Implementado |
| **TrafficMixer** | Múltiplos PCAPs | Stream misturado | ❌ Planejado |

### 3.2 Algoritmos de Detecção

| Versão | Algoritmo | Características | Status |
|--------|-----------|-----------------|--------|
| **v0.1** | TEDA básico | 1 centro global, Chebyshev | ✅ Implementado |
| **v0.2** | MicroTEDAclus | Múltiplos micro-clusters | ❌ Planejado (S4) |
| **v0.3** | + Merge/Split | Clusters dinâmicos | ❌ Planejado |
| **v1.0** | Completo | + Macro-clusters, concept drift | ❌ Planejado |

### 3.3 Infraestrutura

| Componente | Tecnologia | Status |
|------------|------------|--------|
| Message Broker | Apache Kafka | ✅ Docker |
| Monitoramento | Kafka UI | ✅ Docker |
| Experimentos | MLflow | ✅ Baseline |
| Orquestração | Docker Compose | ✅ Local |
| Orquestração | Kubernetes | ❌ Futuro |

---

## 4. Decisões de Design

### 4.1 Decisões Confirmadas

| # | Decisão | Justificativa |
|---|---------|---------------|
| D1 | **Kafka com 3 tópicos** | Separação clara: packets → flows → alerts |
| D2 | **Python** | Ecossistema ML, prototipagem rápida |
| D3 | **TEDA incremental** | v0.1 simples → v1.0 completo |
| D4 | **PCAPs como fonte** | CSVs são shuffled, perdem ordem temporal |
| D5 | **Features do CICIoT2023** | Comparabilidade com literatura |

### 4.2 Decisões Pendentes

| # | Decisão | Opções | Quando Decidir |
|---|---------|--------|----------------|
| P1 | Métricas de clustering | Silhouette vs. Temporal Silhouette | Semana 6 |
| P2 | Merge de clusters | Threshold fixo vs. dinâmico | Semana 5 |
| P3 | Dashboard | Grafana vs. Custom | Fase C |

---

## 5. Roadmap de Evolução

### Fase 2A: Teoria + Setup (Semanas 1-4)
- [x] Setup Kafka local
- [x] Producer v0.1
- [x] Consumer v0.1
- [x] TEDA v0.1 (básico)
- [ ] MicroTEDAclus v0.1

### Fase 2B: Implementação (Semanas 5-10)
- [ ] MicroTEDAclus v0.2 (merge/split)
- [ ] Sistema de métricas
- [ ] Experimentos comparativos
- [ ] MVP estável

### Fase 2C: Validação (Semanas 11-14)
- [ ] Experimentos full dataset
- [ ] Análise de concept drift
- [ ] Documentação resultados

### Fase 3-4: Dissertação (Semanas 15-24)
- [ ] Otimização
- [ ] Escrita
- [ ] Defesa

---

## 6. Métricas de Sucesso

### 6.1 Performance de Detecção

| Métrica | Alvo | Baseline Fase 1 |
|---------|------|-----------------|
| F1-Score | > 0.95 | 0.99 |
| Precision | > 0.95 | 0.99 |
| Recall | > 0.90 | 0.99 |
| Tempo detecção drift | < 100 amostras | N/A |

### 6.2 Performance de Sistema

| Métrica | Alvo |
|---------|------|
| Throughput | > 1000 flows/s |
| Latência | < 100ms |
| Memória | O(clusters), não O(dados) |

---

## 7. Referências

### Documentos Relacionados

| Documento | Propósito |
|-----------|-----------|
| [CURRENT.md](./CURRENT.md) | Estado atual da implementação |
| [KAFKA_REFERENCE.md](./KAFKA_REFERENCE.md) | Detalhes técnicos de Kafka |
| [theory/teda-framework.md](../theory/teda-framework.md) | Teoria TEDA |
| [theory/concept-drift.md](../theory/concept-drift.md) | Teoria concept drift |

### Papers Fundamentais

- **Angelov (2014)** - Framework TEDA original
- **Maia et al. (2020)** - MicroTEDAclus
- **Neto et al. (2023)** - CICIoT2023 Dataset

---

**Nota:** Este documento é atualizado quando há mudanças nos OBJETIVOS do projeto. Para acompanhar o progresso da implementação, veja [CURRENT.md](./CURRENT.md).
