# Design de Arquitetura - IoT IDS com Clustering Evolutivo

**Data:** 2025-12-17
**Status:** Parcial (lacunas identificadas para pesquisa)
**Versão:** 0.1 (MVP)

---

## 1. Resumo Executivo

### 1.1 Objetivo
Implementar um Sistema de Detecção de Intrusão (IDS) para redes IoT usando clustering evolutivo (TEDA/MicroTEDAclus) com arquitetura de streaming via Kafka.

### 1.2 Escopo do MVP
- **Fase A:** Prova de conceito técnica (pipeline PCAP → Kafka → TEDA funcional)
- **Fase B:** Resultados comparáveis com Fase 1 (mesmas métricas, mesmo dataset)

### 1.3 Decisões Principais

| Aspecto | Decisão |
|---------|---------|
| Algoritmo | TEDA simples → MicroTEDAclus (evolutivo) |
| Streaming | Kafka com 2 tópicos |
| Dados | PCAPs do CICIoT2023 (~548GB via SSH) |
| Execução | Tudo remoto na máquina com PCAPs |
| Linguagem | Python |

---

## 2. Arquitetura Geral

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ARQUITETURA IoT IDS - MVP                            │
│                     (Execução: Máquina Remota via SSH)                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐                                                       │
│  │    PCAPs     │ (~548 GB, subset inicial ~50GB)                       │
│  │  CICIoT2023  │                                                       │
│  └──────┬───────┘                                                       │
│         │                                                               │
│         ▼                                                               │
│  ┌──────────────┐                                                       │
│  │   Producer   │                                                       │
│  │ (PCAP Reader)│                                                       │
│  └──────┬───────┘                                                       │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                          KAFKA                                   │   │
│  │                                                                  │   │
│  │   ┌─────────────────┐              ┌─────────────────┐          │   │
│  │   │ topic: packets  │              │  topic: flows   │          │   │
│  │   │   (raw pkts)    │              │ (47 features)   │          │   │
│  │   └────────┬────────┘              └────────┬────────┘          │   │
│  │            │                                │                    │   │
│  └────────────│────────────────────────────────│────────────────────┘   │
│               │                                │                        │
│               ▼                                ▼                        │
│  ┌────────────────────────────┐   ┌────────────────────────────┐       │
│  │  Consumer 1: Janelamento   │   │    Consumer 2: TEDA        │       │
│  │  - Lê de topic:packets     │   │    - Lê de topic:flows     │       │
│  │  - Agrupa N pacotes        │   │    - Eccentricidade        │       │
│  │  - Extrai 47 features      │   │    - Tipicalidade          │       │
│  │  - Publica em topic:flows  │   │    - Micro-clusters        │       │
│  │  - N = configurável        │   │    - Detecção anomalia     │       │
│  └────────────┬───────────────┘   └─────────────┬──────────────┘       │
│               │                                  │                      │
│               ▼                                  ▼                      │
│        ┌────────────┐                  ┌─────────────────┐              │
│        │ topic:flows│                  │ Métricas/Output │              │
│        │  (publica) │                  │  (formato TBD)  │              │
│        └────────────┘                  └─────────────────┘              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Fluxo de Dados

1. **Producer** lê PCAPs preservando timestamps → publica em `topic:packets`
2. **Consumer 1** lê de `topic:packets` → agrupa em janelas → extrai features → publica em `topic:flows`
3. **Consumer 2** lê de `topic:flows` → executa TEDA → detecta anomalias → gera resultados

---

## 3. Componentes Detalhados

### 3.1 Producer (PCAP Reader)

**Responsabilidade:** Ler arquivos PCAP preservando informação temporal e publicar pacotes no Kafka.

```python
class PCAPProducer:
    def __init__(self, kafka_broker, topic="packets"):
        self.producer = KafkaProducer(bootstrap_servers=kafka_broker)
        self.topic = topic

    def process_pcap(self, pcap_path, speed_multiplier=1.0):
        """
        Lê PCAP e publica pacotes respeitando timestamps.
        speed_multiplier: 1.0 = tempo real, 10.0 = 10x mais rápido
        """
        for timestamp, packet_data in read_pcap(pcap_path):
            message = {
                "timestamp": timestamp,
                "raw_packet": packet_data,
                "source_file": pcap_path
            }
            self.producer.send(self.topic, message)

            # Controle de velocidade (simula tempo real)
            wait_time = calculate_delay(timestamp, speed_multiplier)
            time.sleep(wait_time)
```

**Parâmetros configuráveis:**
- `pcap_path`: Arquivo ou diretório de PCAPs
- `speed_multiplier`: Acelerar/desacelerar replay
- `kafka_broker`: Endereço do Kafka
- `topic`: Nome do tópico de saída

**Decisões:**
| Aspecto | Decisão | Justificativa |
|---------|---------|---------------|
| Biblioteca PCAP | `dpkt` ou `scapy` | Usadas no paper original |
| Serialização | JSON (MVP) → Avro (produção) | Simplicidade inicial |
| Controle velocidade | Configurável | Permite acelerar testes |
| Granularidade | 1 pacote = 1 mensagem | Flexibilidade no janelamento |

---

### 3.2 Consumer 1 (Janelamento + Features)

**Responsabilidade:** Ler pacotes brutos, agrupar em janelas, extrair 47 features, publicar flows.

```python
class WindowingConsumer:
    def __init__(self, kafka_broker, window_size=100):
        self.consumer = KafkaConsumer("packets", bootstrap_servers=kafka_broker)
        self.producer = KafkaProducer(bootstrap_servers=kafka_broker)
        self.window_size = window_size  # Configurável
        self.buffer = []

    def process(self):
        for message in self.consumer:
            packet = message.value
            self.buffer.append(packet)

            if len(self.buffer) >= self.window_size:
                flow = self.extract_features(self.buffer)
                self.producer.send("flows", flow)
                self.buffer = []

    def extract_features(self, packets):
        """
        Extrai 47 features da janela de pacotes.
        Baseado no paper CICIoT2023.
        """
        return {
            "window_id": generate_id(),
            "timestamp_start": packets[0]["timestamp"],
            "timestamp_end": packets[-1]["timestamp"],
            "features": {
                "flow_duration": calc_duration(packets),
                "header_length": calc_header_stats(packets),
                "protocol_type": get_protocol(packets),
                # ... 44 features restantes
            }
        }
```

**Parâmetros configuráveis:**
- `window_size`: 10, 100, ou outro valor (configurável para experimentos)
- `overlap`: Janelas sobrepostas (futuro)
- `timeout`: Flush se não receber pacotes em X segundos

**Decisões:**
| Aspecto | Decisão | Justificativa |
|---------|---------|---------------|
| Ferramenta features | `NFStream` ou custom | NFStream tem 48 features prontas |
| Janela | Por contagem (N pacotes) | Consistente com paper original |
| Buffer | Em memória | Simples para MVP |
| Agregação | Média/soma por janela | Como no paper CICIoT2023 |

---

### 3.3 Consumer 2 (TEDA)

**Responsabilidade:** Ler flows, executar algoritmo TEDA, detectar anomalias, gerenciar micro-clusters.

```python
class TEDAConsumer:
    def __init__(self, kafka_broker, m=3):
        self.consumer = KafkaConsumer("flows", bootstrap_servers=kafka_broker)
        self.m = m  # Parâmetro Chebyshev
        self.micro_clusters = []
        self.global_stats = GlobalStats()  # n, mean, variance

    def process(self):
        for message in self.consumer:
            flow = message.value
            x = flow["features"]  # Vetor de 47 features

            result = self.process_point(x)
            self.output_result(flow, result)

    def process_point(self, x):
        """
        Algoritmo TEDA simplificado (evolui para MicroTEDAclus)
        """
        # 1. Atualizar estatísticas globais
        self.global_stats.update(x)

        # 2. Calcular eccentricidade
        eccentricity = self.calculate_eccentricity(x)
        typicality = 1 - eccentricity

        # 3. Teste de Chebyshev
        threshold = (self.m**2 + 1) / (2 * self.global_stats.n)
        is_anomaly = eccentricity > threshold

        return {
            "eccentricity": eccentricity,
            "typicality": typicality,
            "is_anomaly": is_anomaly,
            "threshold": threshold,
            "n_points_seen": self.global_stats.n
        }

    def calculate_eccentricity(self, x):
        n = self.global_stats.n
        mean = self.global_stats.mean
        variance = self.global_stats.variance

        dist_squared = np.sum((x - mean) ** 2)
        return (1/n) + dist_squared / (n * variance)
```

**Evolução planejada:**

| Versão | Funcionalidade | Complexidade |
|--------|---------------|--------------|
| v0.1 | TEDA global (1 centro) | Baixa |
| v0.2 | + Múltiplos micro-clusters | Média |
| v0.3 | + Merge/split de clusters | Média-Alta |
| v1.0 | MicroTEDAclus completo | Alta |

**Parâmetros configuráveis:**
- `m`: Desvios padrão para Chebyshev (default: 3)
- `min_points_stable`: Pontos mínimos antes de confiar nas decisões (cold start)

---

## 4. Decisões de Design

### 4.1 Decisões Confirmadas

| # | Decisão | Opções Consideradas | Escolha | Justificativa |
|---|---------|---------------------|---------|---------------|
| D1 | Escopo MVP | PoC / Comparável / Ambos | A→B | Começar simples, evoluir |
| D2 | Fases | TEDA+RF / TEDA apenas / Paralelo | TEDA apenas | Foco na contribuição principal |
| D3 | Tópicos Kafka | 1 / 2 / 3 | 2 tópicos | Separação de responsabilidades |
| D4 | Janelamento | Fixo / Por tempo / Configurável | Configurável | Flexibilidade para experimentos |
| D5 | Execução | Local / Remoto / Híbrido | Remoto | PCAPs já estão lá (548GB) |
| D6 | Implementação TEDA | Do zero / Adaptar / Simples→Completo | Simples→Completo | Aprendizado + rigor matemático |

### 4.2 Decisões Pendentes (Requerem Pesquisa)

| # | Decisão | Opções | Ação Necessária |
|---|---------|--------|-----------------|
| P1 | Métricas de avaliação | Internas / Externas / Híbrido / Binário | Pesquisar literatura |
| P2 | Tracking de experimentos | JSON / MLflow / TimescaleDB | Testar MLflow primeiro |
| P3 | Design experimentos drift | Temporal / Sintético / Ambos | Pesquisar literatura |

---

## 5. Lacunas e Pesquisa Pendente

### 5.1 Métricas e Avaliação

**Pergunta:** Como avaliar clustering evolutivo em IDS?

**Opções identificadas:**
- Métricas internas: Silhouette, Davies-Bouldin
- Métricas externas: ARI, NMI (usando labels como ground truth)
- Detecção binária: F1/Precision/Recall (normal vs ataque)

**Pesquisar:**
- Como papers de clustering evolutivo avaliam performance?
- Métricas específicas para concept drift?

**Leituras relacionadas:** ML-A1, ML-A3 (ver `reading-plan.md`)

### 5.2 Tracking de Experimentos

**Pergunta:** Qual sistema usar para streaming/drift?

**Evolução planejada:**
1. JSON/Parquet simples (MVP)
2. MLflow (se adequado para streaming)
3. TimescaleDB (se necessário para séries temporais)

**Pesquisar:**
- MLflow funciona bem para experimentos de streaming?
- Alternativas específicas para concept drift?

### 5.3 Design de Experimentos de Drift

**Pergunta:** Como estruturar experimentos para validar adaptação?

**Hipótese inicial:**
- Treinar TEDA apenas com tráfego benigno
- Cada tipo de ataque novo = concept drift a ser detectado
- Avaliar: tempo de detecção, criação de novos clusters, adaptação

**Pesquisar:**
- É a abordagem padrão na literatura?
- Como medir "sucesso" na adaptação?

**Leituras relacionadas:** ML-A1, ML-A2, CS-A1 (ver `reading-plan.md`)

---

## 6. Plano de Leituras Obrigatórias

Este design requer fundamentação teórica. Ver documento completo: `docs/reading-plan.md`

### Resumo por Área

| Área | Principais | Auxiliares | Prioridade |
|------|------------|------------|------------|
| **ML (Clustering)** | Angelov (2014), Maia (2020) | 3 surveys | Alta |
| **Cibersegurança** | CICIoT2023, Survey IDS IoT | 3 papers | Alta |
| **IoT** | Survey IoT Security, Edge IDS | 3 papers | Média |
| **Arquitetura** | Streaming paper, Kafka Guide | 3 refs | Média |

### Próximas Leituras (Semanas 1-2)

1. **Angelov (2014)** - Framework TEDA original
2. **Maia (2020)** - Releitura completa com fichamento
3. **Survey Concept Drift (2024)** - Métricas e métodos

---

## 7. Próximos Passos

### Imediato (Esta Semana)
- [ ] Finalizar documentação de arquitetura
- [ ] Iniciar leitura de Angelov (2014)
- [ ] Commit e push dos documentos

### Curto Prazo (Semanas 1-4)
- [ ] Completar leituras ML-P1, ML-P2, ML-A1
- [ ] Resolver lacuna P1 (métricas de avaliação)
- [ ] Setup ambiente remoto (Kafka Docker)

### Médio Prazo (Semanas 5-8)
- [ ] Implementar Producer (v0.1)
- [ ] Implementar Consumer 1 (v0.1)
- [ ] Implementar TEDA simples (v0.1)
- [ ] Primeiro experimento end-to-end

---

## 8. Referências

### Documentos Relacionados
- `docs/summaries/clustering-evolutivo-concepts.md` - Conceitos TEDA
- `docs/summaries/concept-drift-fundamentals.md` - Tipos de drift
- `docs/summaries/pcap-processing-requirements.md` - Pipeline PCAP
- `docs/reading-plan.md` - Plano de leituras obrigatórias

### Papers Principais
- Angelov (2014) - TEDA Framework
- Maia et al. (2020) - MicroTEDAclus
- Neto et al. (2023) - CICIoT2023 Dataset

---

**Este documento será atualizado conforme as lacunas forem resolvidas.**

*Última atualização: 2025-12-17*
