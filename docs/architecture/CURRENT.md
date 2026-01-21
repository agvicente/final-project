# Arquitetura Atual - IoT IDS Streaming

**Criado:** 2026-01-20
**Última Atualização:** 2026-01-20
**Versão:** 0.1.0

> **Propósito:** Este documento descreve O QUE ESTÁ IMPLEMENTADO AGORA. Para a visão de alto nível (onde queremos chegar), veja [TARGET.md](./TARGET.md).

> **IMPORTANTE PARA CLAUDE:** Este arquivo DEVE ser atualizado sempre que houver mudanças na arquitetura do sistema (novos componentes, novos tópicos Kafka, novas classes, etc.). Adicione uma entrada no changelog e atualize os diagramas correspondentes.

---

## Changelog

| Versão | Data | Descrição |
|--------|------|-----------|
| 0.1.0 | 2026-01-20 | Arquitetura inicial: PCAPProducer, FlowConsumer, TEDADetector, StreamingDetector |

---

## 1. Visão Geral

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           STREAMING IoT IDS - Arquitetura v0.1                          │
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
│  │    (17 features)               │  │  KafkaProducer          │
│  └────────────────────────────────┘  │                         │
└──────────────────────────────────────┘                         │
                                                                 │
                                  ┌──────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│  src/detector/streaming_detector.py                              │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │       StreamingDetector                                    │  │
│  │  ──────────────────────────────────────────────────────    │  │
│  │  + connect()          # KafkaConsumer + KafkaProducer      │  │
│  │  + run()              # Main loop                          │  │
│  │  + _extract_features()  # 17 features → numpy array        │  │
│  │  + _process_flow()    # TEDA + alert logic                 │  │
│  │  + _create_alert()    # JSON com severidade                │  │
│  │  + _calculate_severity()  # low/medium/high/critical       │  │
│  └────────────────────────────────────────────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  src/detector/teda.py                                      │  │
│  │  ┌──────────────────────────────────────────────────────┐  │  │
│  │  │       TEDADetector (Angelov 2014)                    │  │  │
│  │  │  ────────────────────────────────────────────────    │  │  │
│  │  │  + update(x)           # Processa 1 amostra          │  │  │
│  │  │  + _update_statistics()  # μ, σ² recursivos O(1)     │  │  │
│  │  │  + _calculate_eccentricity()  # ξ = f(x, μ, σ²)      │  │  │
│  │  │  + _calculate_threshold()     # Chebyshev            │  │  │
│  │  │  + predict(x)          # Só classifica               │  │  │
│  │  │  + get_statistics()    # Métricas do detector        │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  │  ┌──────────────────────────────────────────────────────┐  │  │
│  │  │       TEDAResult                                     │  │  │
│  │  │  ────────────────────────────────────────────────    │  │  │
│  │  │  - eccentricity (ξ)                                  │  │  │
│  │  │  - typicality (τ = 1 - ξ)                            │  │  │
│  │  │  - threshold                                         │  │  │
│  │  │  - is_anomaly                                        │  │  │
│  │  │  - sample_count                                      │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. Fluxo de Dados

```
PCAP ──► PCAPProducer ──► [packets] ──► FlowConsumer ──► [flows] ──► StreamingDetector ──► [alerts]
              │                              │                              │
              │                              │                              │
         Extrai:                       Agrega em flows:              Detecta anomalias:
         - IP src/dst                  - packet_count                - TEDADetector.update()
         - Portas                      - total_bytes                 - Eccentricity > threshold?
         - Protocolo                   - fwd/bwd stats               - Severity calculation
         - Tamanho                     - IAT stats                   - Alert JSON
         - Flags TCP                   - TCP flags
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
| **FlowConsumer** | `consumer/flow_consumer.py` | `packets` | `flows` | Agrega pacotes em flows com 17 features |
| **StreamingDetector** | `detector/streaming_detector.py` | `flows` | `alerts` | Detecta anomalias usando TEDA |

### 3.3 Algoritmos de Detecção

| Componente | Arquivo | Algoritmo | Status |
|------------|---------|-----------|--------|
| **TEDADetector** | `detector/teda.py` | TEDA (Angelov 2014) - Eccentricity/Typicality | ✅ Implementado |
| **MicroTEDAclus** | `detector/microtedaclus.py` | Maia 2020 - Micro/Macro clusters | ❌ Planejado (S4) |

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

### 4.3 Alert Schema

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
  "threshold": "float",
  "is_anomaly": "boolean",
  "sample_number": "int",
  "detected_at": "string (ISO 8601)",
  "severity": "string (low|medium|high|critical)"
}
```

---

## 5. Infraestrutura Docker

```
┌─────────────────────────────────────────┐
│     streaming/docker/docker-compose.yml │
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

## 6. Estatísticas de Código

| Arquivo | Classes | LOC | Função Principal |
|---------|---------|-----|------------------|
| `producer/pcap_producer.py` | `PCAPProducer` | ~600 | PCAP → packets |
| `consumer/flow_consumer.py` | `FlowConsumer`, `FlowData` | ~500 | packets → flows |
| `detector/teda.py` | `TEDADetector`, `TEDAResult` | ~430 | Algoritmo TEDA |
| `detector/streaming_detector.py` | `StreamingDetector`, `StreamingDetectorConfig` | ~560 | flows → alerts |
| **Total** | **7 classes** | **~2090** | |

---

## 7. Roadmap de Evolução

### Implementado (v0.1.0)
- [x] PCAPProducer - Leitura de PCAPs
- [x] FlowConsumer - Agregação de pacotes em flows
- [x] TEDADetector - Detecção básica (Angelov 2014)
- [x] StreamingDetector - Integração Kafka + TEDA
- [x] 3 tópicos Kafka: packets, flows, alerts

### Planejado (v0.2.0 - Semana 4)
- [ ] MicroTEDAclus - Micro/macro clusters (Maia 2020)
- [ ] Concept drift detection
- [ ] Métricas de cluster (densidade, raio)

### Planejado (v0.3.0 - Semana 5-6)
- [ ] Device-specific models
- [ ] Two-phase detection (anomaly + classification)
- [ ] Dashboard de monitoramento

---

## 8. Referências

- **Angelov 2014**: "Outside the box: an alternative data analytics framework" - Framework TEDA original
- **Maia 2020**: "Evolving clustering algorithm based on mixture of typicalities" - MicroTEDAclus
- **CICIoT2023**: Dataset de ataques IoT usado para testes

---

**Nota:** Este documento é atualizado automaticamente conforme o sistema evolui. Sempre consulte a versão mais recente no repositório.
