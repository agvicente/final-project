---
name: kafka-streaming-iot
description: Teaches Kafka streaming architecture for real-time IoT anomaly detection. From setup to production-grade pipeline with evolutionary clustering integration.
version: 1.0.0
activate_when:
  - "kafka"
  - "streaming"
  - "real-time"
  - "Phase 3"
---

# Kafka Streaming for IoT IDS

## Purpose
Guide Phase 3 implementation: Kafka-based streaming pipeline for real-time anomaly detection with evolutionary clustering.

## Learning Path

### Week 1: Kafka Fundamentals
**Concepts:** Topics, producers, consumers, partitions, offsets
**Practice:** Local Kafka setup with Docker, simple producer/consumer in Python
**Code:**
```python
from kafka import KafkaProducer, KafkaConsumer
import json

# Producer: simulate IoT traffic
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Send CICIoT2023 samples as stream
for sample in dataset:
    producer.send('iot-traffic', value=sample)

# Consumer: receive and process
consumer = KafkaConsumer(
    'iot-traffic',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

for message in consumer:
    traffic_sample = message.value
    # Process here
```

### Week 2: Streaming Pipeline Architecture
**Components:**
- Producer: Simulates IoT devices sending traffic
- Kafka: Message broker for buffering
- Consumer: Runs evolutionary clustering + detection
- MLflow: Logs detections and performance
- Dashboard: Real-time visualization (optional)

**Files to Create:**
- `src/streaming/producer.py` - Send CICIoT2023 data as stream
- `src/streaming/consumer.py` - Process with evolutionary clustering
- `src/streaming/detector.py` - Anomaly detection logic
- `docker-compose-streaming.yml` - Kafka + Zookeeper + services

### Weeks 3-6: Implementation
**Sprint 1:** Producer simulates realistic IoT traffic (vary speed, batching)
**Sprint 2:** Consumer with evolutionary clustering (from Phase 2)
**Sprint 3:** Performance optimization (throughput, latency)
**Sprint 4:** Benchmarks and validation

### Weeks 7-9: Validation
**Metrics:**
- Throughput (messages/second)
- Latency (detection delay)
- Resource usage (CPU, memory)
- Detection accuracy (compare with Phase 1)

**Success Criteria:**
- Process 1000+ messages/second
- <100ms detection latency
- Maintain F1 > 0.99 from Phase 1

## Integration with Evolutionary Clustering
```python
# consumer.py
from src.clustering.evolutionary_clustering import EvolutionaryClusterer

clusterer = EvolutionaryClusterer()
clusterer.fit_initial(initial_batch)

for message in consumer:
    traffic = preprocess(message.value)
    cluster_id = clusterer.predict(traffic)

    # Anomaly detection
    typicality = clusterer.get_typicality(traffic, cluster_id)
    if typicality < threshold:
        alert("Anomaly detected", traffic)

    # Update clusters periodically
    if buffer.size() >= batch_size:
        clusterer.update(buffer.get_all())
        buffer.clear()
```

## Key Papers
- Surianarayanan et al. (2024) - High-throughput architecture
- Mohan et al. (2025) - Kafka + Spark for IDS
- Rivera et al. (2021) - Real-time anomaly detection

## Docker Setup
```yaml
# docker-compose-streaming.yml
version: '3'
services:
  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181

  kafka:
    image: confluentinc/cp-kafka:latest
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092

  producer:
    build: .
    command: python src/streaming/producer.py
    depends_on:
      - kafka

  consumer:
    build: .
    command: python src/streaming/consumer.py
    depends_on:
      - kafka
```

---
**Use this skill for Phase 3 streaming implementation.**
