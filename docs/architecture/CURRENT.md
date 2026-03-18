# Arquitetura Atual - IoT IDS Streaming

**Criado:** 2026-01-20
**Última Atualização:** 2026-03-16
**Versão:** 0.4.0

> **Propósito:** Este documento descreve O QUE ESTÁ IMPLEMENTADO AGORA. Para a visão de alto nível (onde queremos chegar), veja [TARGET.md](./TARGET.md).

> **IMPORTANTE PARA CLAUDE:** Este arquivo DEVE ser atualizado sempre que houver mudanças na arquitetura do sistema (novos componentes, novos tópicos Kafka, novas classes, etc.). Adicione uma entrada no changelog e atualize os diagramas correspondentes.

---

## Changelog

| Versão | Data | Descrição |
|--------|------|-----------|
| 0.1.0 | 2026-01-20 | Arquitetura inicial: PCAPProducer, FlowConsumer, TEDADetector, StreamingDetector |
| 0.2.0 | 2026-01-29 | MicroTEDAclus implementado, StreamingDetector v0.2 com seleção de algoritmo |
| 0.3.0 | 2026-03-10 | Sincronização FlowConsumer-Detector, métricas prequential, orquestrador de experimentos v2 |
| 0.4.0 | 2026-03-16 | Feature sets v1/v2/v3, WindowAggregator (detecção por janela temporal), DetectionGranularity enum |
| 0.5.0 | 2026-03-16 | WindowAggregator v2: 7 behavioral features (entropy, ratios, rates), --window-features CLI arg |

---

## 1. Visão Geral

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           STREAMING IoT IDS - Arquitetura v0.2                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│   PCAP Files     │
│  (CICIoT2023)    │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  src/producer/pcap_producer.py       │
│  ┌────────────────────────────────┐  │
│  │       PCAPProducer             │  │
│  │  ──────────────────────────    │  │
│  │  + connect()                   │  │
│  │  + process_pcap(path)          │  │
│  │  + _extract_packet_info()      │  │
│  │  + _read_pcap()                │  │
│  │  + close()                     │  │
│  └────────────────────────────────┘  │
│          │                           │
│          │  KafkaProducer            │
└──────────┼───────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                    KAFKA CLUSTER                                        │
│  ┌─────────────────┐    ┌──────────────────────────────────────────────────────────┐   │
│  │   Zookeeper     │    │                      Topics                               │   │
│  │   :2181         │    │  ┌──────────┐    ┌──────────┐    ┌──────────┐            │   │
│  └─────────────────┘    │  │ packets  │    │  flows   │    │  alerts  │            │   │
│  ┌─────────────────┐    │  │ (raw)    │    │(features)│    │(anomalies)│           │   │
│  │   Kafka         │    │  └────┬─────┘    └────┬─────┘    └────▲─────┘            │   │
│  │   :9092         │    │       │               │               │                   │   │
│  └─────────────────┘    └───────┼───────────────┼───────────────┼───────────────────┘   │
│  ┌─────────────────┐            │               │               │                       │
│  │   Kafka UI      │            │               │               │                       │
│  │   :8080         │            │               │               │                       │
│  └─────────────────┘            │               │               │                       │
└─────────────────────────────────┼───────────────┼───────────────┼───────────────────────┘
                                  │               │               │
                                  ▼               │               │
┌──────────────────────────────────────┐         │               │
│  src/consumer/flow_consumer.py       │         │               │
│  ┌────────────────────────────────┐  │         │               │
│  │       FlowConsumer             │  │         │               │
│  │  ──────────────────────────    │  │         │               │
│  │  + connect()                   │  │         │               │
│  │  + run()                       │  │         │               │
│  │  + _process_packet()           │  │         │               │
│  │  + _complete_flow()            │  │         │               │
│  │  + _check_flow_timeouts()      │  │         │               │
│  └────────────────────────────────┘  │         │               │
│  ┌────────────────────────────────┐  │         │               │
│  │       FlowData                 │  │         │               │
│  │  ──────────────────────────    │  │         │               │
│  │  + add_packet()                │  │         │               │
│  │  + to_features() ──────────────┼──┼─────────┘               │
│  │    (28 features ML)            │  │  KafkaProducer          │
│  └────────────────────────────────┘  │                         │
└──────────────────────────────────────┘                         │
                                                                 │
                                  ┌──────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  src/detector/streaming_detector.py (v0.2)                                           │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │       StreamingDetector                                                        │  │
│  │  ──────────────────────────────────────────────────────────────────────────    │  │
│  │  + connect()          # KafkaConsumer + KafkaProducer                          │  │
│  │  + run()              # Main loop                                              │  │
│  │  + _extract_features()  # 17 features → numpy array                            │  │
│  │  + _process_flow()    # Detector + alert logic                                 │  │
│  │  + _create_alert()    # JSON com severidade (adapta por algoritmo)             │  │
│  │  + _calculate_severity()  # low/medium/high/critical                           │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
│                          │                                                            │
│            ┌─────────────┴─────────────┐                                              │
│            │   DetectorAlgorithm       │                                              │
│            │   (TEDA | MICRO_TEDA)     │                                              │
│            └─────────────┬─────────────┘                                              │
│                          │                                                            │
│         ┌────────────────┴────────────────┐                                           │
│         ▼                                 ▼                                           │
│  ┌─────────────────────────────┐  ┌─────────────────────────────────────────────┐    │
│  │  src/detector/teda.py       │  │  src/detector/micro_teda.py                 │    │
│  │  ┌───────────────────────┐  │  │  ┌───────────────────────────────────────┐  │    │
│  │  │  TEDADetector         │  │  │  │  MicroTEDAclus                        │  │    │
│  │  │  (Angelov 2014)       │  │  │  │  (Maia 2020)                          │  │    │
│  │  │  ─────────────────    │  │  │  │  ─────────────────────────────────    │  │    │
│  │  │  + update(x)          │  │  │  │  + process(x)       # Processa ponto  │  │    │
│  │  │  + predict(x)         │  │  │  │  + predict(x)       # Só classifica   │  │    │
│  │  │  + get_statistics()   │  │  │  │  + reset()                            │  │    │
│  │  │                       │  │  │  │  + get_statistics()                   │  │    │
│  │  │  Vulnerável a         │  │  │  │  + get_cluster_centers()              │  │    │
│  │  │  contaminação!        │  │  │  │  + get_cluster_sizes()                │  │    │
│  │  └───────────────────────┘  │  │  │                                       │  │    │
│  │  ┌───────────────────────┐  │  │  │  Robusto a contaminação               │  │    │
│  │  │  TEDAResult           │  │  │  │  (estatísticas isoladas)              │  │    │
│  │  │  - eccentricity       │  │  │  └───────────────────────────────────────┘  │    │
│  │  │  - typicality         │  │  │  ┌───────────────────────────────────────┐  │    │
│  │  │  - threshold          │  │  │  │  MicroCluster                         │  │    │
│  │  │  - is_anomaly         │  │  │  │  - cluster_id, n, mean, variance      │  │    │
│  │  └───────────────────────┘  │  │  │  + dynamic_m()      # m(k) threshold  │  │    │
│  └─────────────────────────────┘  │  │  + calculate_eccentricity(x)          │  │    │
│         (algorithm=teda)          │  │  + chebyshev_accepts(x)               │  │    │
│                                   │  │  + update(x)                          │  │    │
│                                   │  └───────────────────────────────────────┘  │    │
│                                   │  ┌───────────────────────────────────────┐  │    │
│                                   │  │  MicroTEDAResult                      │  │    │
│                                   │  │  - eccentricity, typicality           │  │    │
│                                   │  │  - cluster_id, num_clusters           │  │    │
│                                   │  │  - is_anomaly, new_cluster_created    │  │    │
│                                   │  └───────────────────────────────────────┘  │    │
│                                   └─────────────────────────────────────────────┘    │
│                                             (algorithm=micro_teda) [DEFAULT]         │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Fluxo de Dados

```
PCAP ──► PCAPProducer ──► [packets] ──► FlowConsumer ──► [flows] ──► StreamingDetector ──► [alerts]
              │                              │                              │
              │                              │                              │
         Extrai:                       Agrega em flows:              Detecta anomalias:
         - IP src/dst                  - packet_count                - TEDA ou MicroTEDAclus
         - Portas                      - total_bytes                 - Eccentricity > threshold?
         - Protocolo                   - fwd/bwd stats               - Novo cluster criado?
         - Tamanho                     - IAT stats                   - Severity calculation
         - Flags TCP                   - TCP flags                   - Alert JSON
         - Timestamp                   - Duration
```

---

## 3. Componentes

### 3.1 Produtores

| Componente | Arquivo | Entrada | Saída (Topic) | Descrição |
|------------|---------|---------|---------------|-----------|
| **PCAPProducer** | `producer/pcap_producer.py` | Arquivos PCAP | `packets` | Lê PCAPs e publica pacotes individuais |

### 3.2 Processadores de Stream

| Componente | Arquivo | Entrada (Topic) | Saída (Topic) | Descrição |
|------------|---------|-----------------|---------------|-----------|
| **FlowConsumer** | `consumer/flow_consumer.py` | `packets` | `flows` | Agrega pacotes em flows (28 features ML + 8 metadata) |
| **StreamingDetector** | `detector/streaming_detector.py` | `flows` | `alerts` | Detecta anomalias usando TEDA ou MicroTEDAclus (per-flow ou per-window) |
| **WindowAggregator** | `detector/window_aggregator.py` | flows (interno) | feature vectors | Agrega flows por IP em janelas temporais (v1: 12 basic, v2: 19 behavioral) |

### 3.3 Algoritmos de Detecção

| Componente | Arquivo | Algoritmo | Status | Default |
|------------|---------|-----------|--------|---------|
| **TEDADetector** | `detector/teda.py` | TEDA (Angelov 2014) - Single-center | ✅ Implementado | Não |
| **MicroTEDAclus** | `detector/micro_teda.py` | Maia 2020 - Multi-cluster evolutivo | ✅ Implementado | **Sim** |

### 3.4 Comparação dos Algoritmos

| Aspecto | TEDADetector | MicroTEDAclus |
|---------|--------------|---------------|
| **Estatísticas** | Globais (μ, σ²) | Isoladas por cluster |
| **Contaminação** | Vulnerável | Resistente |
| **Threshold** | Fixo (m=3) | Dinâmico m(k) |
| **Outliers** | Atualizam stats | Criam novos clusters |
| **Uso** | Debug, comparação | Produção |

---

## 4. Tópicos Kafka

| Tópico | Produtor | Consumidor | Schema | Descrição |
|--------|----------|------------|--------|-----------|
| `packets` | PCAPProducer | FlowConsumer | [PacketSchema](#41-packet-schema) | Pacotes individuais do PCAP |
| `flows` | FlowConsumer | StreamingDetector | [FlowSchema](#42-flow-schema) | Flows agregados com features |
| `alerts` | StreamingDetector | (Dashboard/SIEM) | [AlertSchema](#43-alert-schema) | Alertas de anomalias detectadas |

### 4.1 Packet Schema

```json
{
  "timestamp": "float",
  "src_ip": "string",
  "dst_ip": "string",
  "src_port": "int",
  "dst_port": "int",
  "protocol": "string",
  "length": "int",
  "tcp_flags": "dict (optional)"
}
```

### 4.2 Flow Schema

> **NOTA:** Este schema está simplificado para legibilidade. O código real em `flow_consumer.py::to_features()` retorna **36 campos totais** (28 features ML + 8 metadata/identificação). Veja código fonte para lista completa.

```json
{
  "flow_id": "string",
  "src_ip": "string",
  "dst_ip": "string",
  "src_port": "int",
  "dst_port": "int",
  "protocol": "string",
  "packet_count": "int",
  "total_bytes": "int",
  "fwd_packet_count": "int",
  "bwd_packet_count": "int",
  "fwd_bytes": "int",
  "bwd_bytes": "int",
  "packets_per_second": "float",
  "bytes_per_second": "float",
  "packet_size_mean": "float",
  "packet_size_std": "float",
  "iat_mean": "float",
  "iat_std": "float",
  "syn_count": "int",
  "ack_count": "int",
  "fin_count": "int",
  "rst_count": "int",
  "fwd_bwd_ratio": "float",
  "flow_duration": "float"
}
```

### 4.3 Alert Schema (v0.2)

```json
{
  "flow_id": "string",
  "src_ip": "string",
  "dst_ip": "string",
  "src_port": "int",
  "dst_port": "int",
  "protocol": "string",
  "packet_count": "int",
  "total_bytes": "int",
  "flow_duration": "float",
  "eccentricity": "float",
  "typicality": "float",
  "is_anomaly": "boolean",
  "sample_number": "int",
  "detected_at": "string (ISO 8601)",
  "severity": "string (low|medium|high|critical)",
  "algorithm": "string (teda|micro_teda)",

  // Campos específicos TEDA
  "threshold": "float (only for teda)",
  "normalized_eccentricity": "float (only for teda)",

  // Campos específicos MicroTEDAclus
  "cluster_id": "int (only for micro_teda)",
  "num_clusters": "int (only for micro_teda)",
  "new_cluster_created": "boolean (only for micro_teda)"
}
```

---

## 5. Infraestrutura Docker

```
┌─────────────────────────────────────────┐
│  experiments/streaming/docker/docker-compose.yml │
├─────────────────────────────────────────┤
│  zookeeper    :2181  (coordination)     │
│  kafka        :9092  (message broker)   │
│  kafka-ui     :8080  (web monitoring)   │
└─────────────────────────────────────────┘
```

### Comandos Úteis

```bash
# Iniciar infraestrutura
cd streaming/docker && docker-compose up -d

# Ver status
docker-compose ps

# Ver logs do Kafka
docker-compose logs -f kafka

# Listar tópicos
docker exec iot-kafka kafka-topics --list --bootstrap-server localhost:9092

# Consumir mensagens de um tópico
docker exec iot-kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic <TOPIC_NAME> \
  --from-beginning
```

---

## 6. CLI do StreamingDetector

```bash
# Usar MicroTEDAclus (default, robusto)
python -m src.detector.streaming_detector

# Usar MicroTEDAclus com parâmetros específicos
python -m src.detector.streaming_detector --algorithm micro_teda --r0 0.1 --min-samples 10

# Usar TEDA básico (para comparação/debug)
python -m src.detector.streaming_detector --algorithm teda --m 3.0

# Modo verboso
python -m src.detector.streaming_detector --verbose

# Limitar número de flows
python -m src.detector.streaming_detector --max-flows 1000
```

### Parâmetros

| Parâmetro | Default | Descrição |
|-----------|---------|-----------|
| `--algorithm` | `micro_teda` | Algoritmo: `teda` ou `micro_teda` |
| `--m` | `3.0` | Parâmetro m do TEDA básico |
| `--r0` | `0.1` | Variância mínima do MicroTEDAclus |
| `--min-samples` | `10` | Amostras antes de detectar anomalias |
| `--verbose` | `False` | Modo verboso |
| `--max-flows` | `None` | Limite de flows a processar |
| `--no-publish` | `False` | Não publicar alertas no Kafka |

---

## 7. Estatísticas de Código

| Arquivo | Classes | LOC | Função Principal |
|---------|---------|-----|------------------|
| `producer/pcap_producer.py` | `PCAPProducer` | ~600 | PCAP → packets |
| `consumer/flow_consumer.py` | `FlowConsumer`, `FlowData` | ~500 | packets → flows |
| `detector/teda.py` | `TEDADetector`, `TEDAResult` | ~250 | Algoritmo TEDA |
| `detector/micro_teda.py` | `MicroCluster`, `MicroTEDAclus`, `MicroTEDAResult` | ~400 | MicroTEDAclus |
| `detector/streaming_detector.py` | `StreamingDetector`, `StreamingDetectorConfig`, `DetectorAlgorithm` | ~600 | flows → alerts |
| **Total** | **10 classes** | **~2350** | |

### Testes

| Arquivo | Testes | Status |
|---------|--------|--------|
| `tests/test_teda.py` | 36 | ✅ Passando |
| `tests/test_micro_teda.py` | 31 | ✅ Passando |
| `tests/test_window_aggregator.py` | 25 | ✅ Passando |
| **Total** | **123** | **~17s** |

---

## 8. Roadmap de Evolução

### Implementado (v0.1.0 - Semana 3)
- [x] PCAPProducer - Leitura de PCAPs
- [x] FlowConsumer - Agregação de pacotes em flows
- [x] TEDADetector - Detecção básica (Angelov 2014)
- [x] StreamingDetector v0.1 - Integração Kafka + TEDA
- [x] 3 tópicos Kafka: packets, flows, alerts

### Implementado (v0.2.0 - Semana 4)
- [x] MicroCluster - Estatísticas isoladas por cluster
- [x] MicroTEDAclus - Multi-cluster evolutivo (Maia 2020)
- [x] Threshold dinâmico m(k)
- [x] StreamingDetector v0.2 - Seleção de algoritmo
- [x] 67 testes unitários

### Implementado (v0.3.0 - Semana 8)
- [x] Sincronização FlowConsumer-Detector (`wait_for_flow_consumer()`)
- [x] Métricas prequential (`src/metrics/prequential_metrics.py`)
- [x] Orquestrador de experimentos com pipeline sincronizado
- [x] Isolamento de experimentos (purga de tópicos Kafka)
- [x] 5 artefatos estruturados por experimento
- [x] 98 testes unitários

### Implementado (v0.4.0 - Semana 10)
- [x] Feature sets v1 (17), v2 (25), v3 (32) com seleção via `--features`
- [x] WindowAggregator: detecção por janela temporal (12 features agregadas por IP)
- [x] DetectionGranularity enum (FLOW | WINDOW)
- [x] FlowConsumer: features IAT direcionais (fwd/bwd_iat_mean/std)
- [x] extract_attack_ips.py: fix double-read, --pcap-files, progress logging
- [x] 110 testes unitários

### Implementado (v0.5.0 - Semana 11) ✅ ATUAL
- [x] WindowAggregator v2: 7 behavioral features (entropy, ratios, rates) — 19 features total
- [x] `_shannon_entropy()` helper for discrete value entropy computation
- [x] `WINDOW_FEATURES_V2` constant (12 base + 7 behavioral)
- [x] `--window-features {v1,v2}` CLI argument in run_experiment.py
- [x] StreamingDetectorConfig.window_feature_version wiring
- [x] Campaign-03 S4 experiment scripts (48 runs)
- [x] 123 testes unitários (13 novos para v2 features)

### Planejado (v0.6.0)
- [ ] Two-Stage Detection (per-flow + IP anomaly concentration)
- [ ] Suporte a multi-fase (`--drift-pcap`) para cenários de drift
- [ ] Suporte a holdout (`--holdout-pcap`) para cenários zero-day

---

## 9. Referências

- **Angelov 2014**: "Outside the box: an alternative data analytics framework" - Framework TEDA original
- **Maia 2020**: "Evolving clustering algorithm based on mixture of typicalities" - MicroTEDAclus
- **CICIoT2023**: Dataset de ataques IoT usado para testes

---

**Nota:** Este documento é atualizado conforme o sistema evolui. Sempre consulte a versão mais recente no repositório.
