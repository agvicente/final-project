# Streaming Architecture - IoT IDS

**Criado em:** 2025-01-16
**Status:** Documento de Planejamento
**Autor:** Augusto (com assistência de Claude)

---

## 1. Visao Geral

Este documento descreve a arquitetura de streaming para o sistema de deteccao de intrusao em IoT, incluindo:
- Evolucao da arquitetura monolitica para distribuida
- Metodologia para simulacao de trafego realista usando PCAPs
- Pesquisa bibliografica para implementacao futura

---

## 2. Arquitetura Atual (v0.1 - Desenvolvimento)

### 2.1 Diagrama

```
┌─────────────────────────────────────────────────────┐
│              Maquina Local (Mac/Ubuntu)             │
│                                                     │
│  ┌─────────────┐    ┌─────────────┐                │
│  │  Producer   │───►│    Kafka    │                │
│  │  (script)   │    │  (Docker)   │                │
│  └─────────────┘    └─────────────┘                │
│        ▲                   │                        │
│        │                   ▼                        │
│   PCAP local         ┌─────────────┐               │
│                      │  Consumer   │  (futuro)     │
│                      │  (flows)    │               │
│                      └─────────────┘               │
└─────────────────────────────────────────────────────┘
```

### 2.2 Componentes Atuais

| Componente | Localizacao | Funcao |
|------------|-------------|--------|
| Kafka + Zookeeper | `streaming/docker/` | Message broker |
| PCAP Producer | `streaming/src/producer/` | Le PCAPs, publica pacotes |
| Topicos | `packets`, `flows` | Canais de comunicacao |

### 2.3 Limitacoes da v0.1

- **Execucao manual**: `python pcap_producer.py arquivo.pcap`
- **Sem persistencia**: Crash = perda de progresso
- **Mono-instancia**: Nao escala horizontalmente
- **Sem monitoramento**: Apenas logs basicos

---

## 3. Arquitetura Distribuida (Visao Futura)

### 3.1 Diagrama Completo

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Cluster Kubernetes / Docker Swarm            │
│                                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │  Producer   │  │  Producer   │  │  Producer   │   ← Replicas     │
│  │  Service 1  │  │  Service 2  │  │  Service 3  │     escalaveis   │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                  │
│         │                │                │                          │
│         ▼                ▼                ▼                          │
│  ┌───────────────────────────────────────────────────┐              │
│  │                 Kafka Cluster                      │              │
│  │   ┌─────────┐  ┌─────────┐  ┌─────────┐          │              │
│  │   │Broker 1 │  │Broker 2 │  │Broker 3 │          │              │
│  │   └─────────┘  └─────────┘  └─────────┘          │              │
│  └───────────────────────────────────────────────────┘              │
│         │                │                │                          │
│         ▼                ▼                ▼                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │  Consumer   │  │  Consumer   │  │  Consumer   │   ← Consumer     │
│  │  (Flows)    │  │  (Flows)    │  │  (Flows)    │     Group        │
│  └─────────────┘  └─────────────┘  └─────────────┘                  │
│         │                │                │                          │
│         ▼                ▼                ▼                          │
│  ┌───────────────────────────────────────────────────┐              │
│  │              ML Inference Layer                    │              │
│  │   ┌─────────────────┐    ┌─────────────────┐     │              │
│  │   │ Anomaly Detect. │    │  Evolutionary   │     │              │
│  │   │ (Phase 1 model) │    │  Clustering     │     │              │
│  │   └─────────────────┘    └─────────────────┘     │              │
│  └───────────────────────────────────────────────────┘              │
│                          │                                           │
│                          ▼                                           │
│                  ┌─────────────┐                                    │
│                  │   Alertas   │                                    │
│                  │  Dashboard  │                                    │
│                  └─────────────┘                                    │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.2 Servicos Independentes

Cada componente sera um servico Docker independente:

```
producer-service/
├── Dockerfile
├── requirements.txt
├── src/
│   └── producer/
│       ├── main.py          # FastAPI/Flask app
│       ├── pcap_producer.py
│       └── config.py
└── kubernetes/
    ├── deployment.yaml
    └── service.yaml
```

### 3.3 Endpoints REST (Producer)

| Endpoint | Metodo | Funcao |
|----------|--------|--------|
| `/ingest/pcap` | POST | Upload de arquivo PCAP |
| `/ingest/stream` | POST | Streaming em tempo real |
| `/health` | GET | Health check (K8s) |
| `/metrics` | GET | Prometheus metrics |
| `/status` | GET | Status do processamento |

### 3.4 Comparacao: Script vs Servico

| Aspecto | Script Atual | Servico Distribuido |
|---------|--------------|---------------------|
| Execucao | `python script.py` | Container always-on |
| Input | Arquivo local | API REST / gRPC / TCP |
| Escalabilidade | 1 instancia | N replicas |
| Monitoramento | Logs manuais | Prometheus + Grafana |
| Resiliencia | Crash = fim | K8s reinicia automaticamente |
| Deploy | Manual | CI/CD automatizado |

### 3.5 Consumer Groups e Particionamento

Kafka distribui particoes entre consumers do mesmo grupo:

```
Topic: packets (3 particoes)
├── Particao 0 → Consumer 1
├── Particao 1 → Consumer 2
└── Particao 2 → Consumer 3
```

#### 3.5.1 Relacao Consumers vs Particoes

**Regra fundamental**: Cada particao e lida por NO MAXIMO 1 consumer do mesmo grupo.

**Mais consumers que particoes → Consumers ociosos**
```
3 particoes, 5 consumers:
┌─────────────────────────────────────────────────┐
│ Particao 0 ──────► Consumer 1  ✓ trabalhando    │
│ Particao 1 ──────► Consumer 2  ✓ trabalhando    │
│ Particao 2 ──────► Consumer 3  ✓ trabalhando    │
│                                                 │
│                    Consumer 4  ✗ OCIOSO         │
│                    Consumer 5  ✗ OCIOSO         │
└─────────────────────────────────────────────────┘
Consumers extras servem como BACKUP (assumem se outro cair)
```

**Mais particoes que consumers → Consumers acumulam particoes**
```
6 particoes, 2 consumers:
┌─────────────────────────────────────────────────┐
│ Particao 0 ──┐                                  │
│ Particao 1 ──┼──► Consumer 1 (3 particoes)      │
│ Particao 2 ──┘                                  │
│                                                 │
│ Particao 3 ──┐                                  │
│ Particao 4 ──┼──► Consumer 2 (3 particoes)      │
│ Particao 5 ──┘                                  │
└─────────────────────────────────────────────────┘
Todas as particoes sao lidas, mas cada consumer trabalha mais.
```

#### 3.5.2 Tabela de Cenarios

| Particoes | Consumers | Resultado | Observacao |
|-----------|-----------|-----------|------------|
| 3 | 1 | 1 consumer le tudo | Sem paralelismo |
| 3 | 2 | 1 le 2, outro le 1 | Desbalanceado |
| 3 | 3 | Cada um le 1 | **Ideal** |
| 3 | 5 | 3 trabalham, 2 ociosos | Desperdicio |
| 6 | 2 | Cada um le 3 | Funciona |
| 6 | 6 | Cada um le 1 | **Ideal** |

#### 3.5.3 Rebalanceamento Automatico

Quando consumers entram ou saem do grupo, Kafka redistribui automaticamente:

```
Estado inicial (3 particoes, 3 consumers):
├── P0 → C1
├── P1 → C2
└── P2 → C3

Consumer 2 cai (falha/desconecta):
├── P0 → C1
├── P1 → C3  ← Kafka redistribui P1 para C3
└── P2 → C3

Novo consumer 4 entra no grupo:
├── P0 → C1
├── P1 → C4  ← Kafka redistribui P1 para C4
└── P2 → C3
```

**Tempo de rebalanceamento**: Alguns segundos (configuravel via `session.timeout.ms`).
Durante o rebalanceamento, o consumo pausa brevemente.

#### 3.5.4 Regra Pratica para Planejamento

```
num_particoes >= num_maximo_consumers_planejado

# Exemplo: Se planeja escalar para 10 consumers no futuro
# Crie pelo menos 10 particoes desde o inicio

# Nosso caso atual:
3 particoes → maximo 3 consumers paralelos
Se quisermos mais paralelismo → precisamos mais particoes
```

**IMPORTANTE**: Aumentar particoes depois e possivel, mas:
- Keys existentes podem mudar de particao (hash muda com mais particoes)
- Mensagens antigas ficam nas particoes originais
- Recomendado: planejar particoes desde o inicio

#### 3.5.5 Offset - Controle de Posicao

**Offset** e o numero sequencial de cada mensagem dentro de uma particao.
Funciona como o "indice" ou "posicao" da mensagem.

```
Particao 0:
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│msg0 │msg1 │msg2 │msg3 │msg4 │msg5 │msg6 │msg7 │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
  ↑     ↑     ↑     ↑     ↑     ↑     ↑     ↑
  0     1     2     3     4     5     6     7   ← OFFSETS
```

**Cada particao tem offsets independentes:**
```
Topic: packets
Particao 0: [0] [1] [2] ... [15792]   ← 15.793 mensagens
Particao 1: [0] [1] [2] ... [18280]   ← 18.281 mensagens
Particao 2: [0] [1] [2] ... [19490]   ← 19.491 mensagens

Offset 5 da particao 0 ≠ Offset 5 da particao 1 (mensagens diferentes!)
```

**Tipos de offset:**
```
┌─────────────────────────────────────────────────────────────┐
│  [0] [1] [2] [3] [4] [5] [6] [7] [8] [9] [10]              │
│                       ↑           ↑        ↑               │
│                       │           │        │               │
│            committed offset    current   latest            │
│            (confirmado)       position  (ultima msg)       │
└─────────────────────────────────────────────────────────────┘

committed offset: Ultimo offset CONFIRMADO como processado (salvo no Kafka)
current position: Onde o consumer esta lendo agora
latest offset:    Ultima mensagem disponivel no topico
```

**Consumer Lag (mensagens pendentes):**
```
lag = latest_offset - committed_offset

Exemplo:
  latest offset:    19490
  committed offset: 19000
  ─────────────────────────
  lag:              490 mensagens pendentes

Se lag cresce continuamente → consumer nao acompanha producer
```

**auto_offset_reset - SOMENTE quando nao ha offset salvo:**

IMPORTANTE: Este parametro e frequentemente mal interpretado!
```
┌─────────────────────────────────────────────────────────────┐
│ auto_offset_reset = "earliest"                              │
│                                                             │
│ Significa:                                                  │
│   "SE nao houver offset salvo para este GRUPO,              │
│    comece do inicio"                                        │
│                                                             │
│ NAO significa:                                              │
│   "Sempre comece do inicio"                                 │
└─────────────────────────────────────────────────────────────┘
```

**Fluxograma de decisao do Kafka:**
```
Consumer conecta com group_id="X"
            │
            ▼
┌─────────────────────────┐
│ Existe offset salvo     │
│ para grupo "X"?         │
└───────────┬─────────────┘
            │
     ┌──────┴──────┐
     │             │
    SIM           NAO
     │             │
     ▼             ▼
┌─────────┐  ┌─────────────────┐
│ Usa o   │  │ Usa              │
│ offset  │  │ auto_offset_reset│
│ salvo   │  │ (earliest/latest)│
└─────────┘  └─────────────────┘
```

**Valores possiveis:**
```python
auto_offset_reset = "earliest"  # Offset 0 (le todas as mensagens)
auto_offset_reset = "latest"    # Ultimo offset (so mensagens novas)
```
```
earliest:                              latest:
[0] [1] [2] ... [999] [1000]          [0] [1] [2] ... [999] [1000]
 ↑                                                            ↑
 comeca aqui (le historico)                    comeca aqui (so novas)
```

**Cenarios praticos:**

| Situacao | auto_offset_reset | Resultado |
|----------|-------------------|-----------|
| Grupo NOVO, sem offset | E usado | Comeca do earliest/latest |
| Grupo EXISTENTE, com offset | **IGNORADO** | Continua do offset salvo |
| Consumer extra entra no grupo | **IGNORADO** | Usa offset do grupo (ou fica ocioso) |

**Exemplo: Consumer 4 entra em grupo existente**
```
Estado do grupo "flow-processor" (ja processou):
├── Particao 0: offset 15000 (committed)
├── Particao 1: offset 18000 (committed)
└── Particao 2: offset 19000 (committed)

Consumer 4 entra com auto_offset_reset="earliest":
1. Kafka verifica: offset salvo para "flow-processor"? SIM
2. auto_offset_reset e IGNORADO
3. Consumer 4 fica OCIOSO (3 particoes, 4 consumers)
4. NAO reprocessa mensagens antigas!
```

**Como forcar reprocessamento (quando necessario):**
```python
# Opcao 1: Mudar o group_id (novo grupo = sem offset salvo)
group_id = "flow-processor-v2"  # Comeca do 0

# Opcao 2: Resetar offset via codigo
consumer.seek_to_beginning(partition)

# Opcao 3: Resetar offset via CLI do Kafka
# kafka-consumer-groups --reset-offsets --to-earliest \
#     --group flow-processor --topic packets --execute
```

**Reprocessamento - Voltar no tempo:**
```python
# Voltar para o inicio da particao
consumer.seek_to_beginning(partition)

# Voltar para offset especifico
consumer.seek(partition, offset=100)

# Casos de uso: corrigir bugs, reprocessar com nova logica
```

**Propriedade fundamental**: Offset e IMUTAVEL. Uma vez atribuido, nunca muda.
Mensagem no offset 100 sera sempre offset 100. Isso permite replay deterministico.

#### 3.5.6 Identificacao do Consumer e Persistencia do Offset

**Pergunta comum**: Se um consumer cai, como o Kafka sabe que o consumer que voltou e o mesmo?

**Resposta**: Kafka NAO identifica o consumer individual. Identifica o **Consumer Group**.

**Offset e salvo por grupo, nao por instancia:**
```
Chave do offset salvo:
┌─────────────────────────────────────────────────┐
│  group_id  +  topic  +  partition  →  offset    │
└─────────────────────────────────────────────────┘

Exemplo no topico interno __consumer_offsets:
("flow-processor", "packets", 0) → offset 15000
("flow-processor", "packets", 1) → offset 18000
("flow-processor", "packets", 2) → offset 19000
("outro-grupo", "packets", 0)    → offset 5000
```

**Cenario: Consumer cai e outro assume**
```
ANTES (3 consumers no grupo "flow-processor"):
┌─────────────────────────────────────────────────┐
│ Consumer A (id: abc-123) → Particao 0, offset 100│
│ Consumer B (id: def-456) → Particao 1, offset 200│
│ Consumer C (id: ghi-789) → Particao 2, offset 300│
└─────────────────────────────────────────────────┘
                    │
                    │ Consumer B cai!
                    ▼
REBALANCEAMENTO AUTOMATICO:
┌─────────────────────────────────────────────────┐
│ Consumer A → Particao 0, offset 100             │
│ Consumer A → Particao 1, offset 200  ← assumiu! │
│ Consumer C → Particao 2, offset 300             │
└─────────────────────────────────────────────────┘
                    │
                    │ Consumer B volta (com NOVO id: xyz-999)
                    ▼
REBALANCEAMENTO AUTOMATICO:
┌─────────────────────────────────────────────────┐
│ Consumer A → Particao 0, offset 100             │
│ Consumer B → Particao 1, offset 200  ← retomou! │
│ Consumer C → Particao 2, offset 300             │
└─────────────────────────────────────────────────┘

Consumer B tem ID interno DIFERENTE, mas mesmo group_id = continua do offset salvo
```

**O que importa e o group_id:**
```python
# config.py
group_id: str = "flow-processor"  # ← ISSO identifica o grupo

# Qualquer consumer com group_id="flow-processor"
# compartilha os mesmos offsets salvos
```

**Implicacoes praticas:**
```python
# Se mudar o group_id, perde o progresso!

# Execucao 1:
group_id = "flow-processor"      # Processa ate offset 10000

# Execucao 2 (mesmo group_id):
group_id = "flow-processor"      # Continua do 10000 ✓

# Execucao 3 (group_id diferente):
group_id = "flow-processor-v2"   # Comeca do 0 (ou latest) ✗
```

**Por que usamos group_id dinamico nos testes:**
```bash
# Cria grupo novo cada vez → reprocessa do inicio
group_id = "test-consumer-$(date +%s)"
```

**Analogia**: O group_id e como uma "conta de usuario" em streaming (Netflix).
Nao importa de qual dispositivo (consumer) voce acessa - o progresso esta
salvo na conta (group_id). Trocar dispositivo nao perde progresso, mas
criar conta nova sim.

### 3.6 Particionamento e Hot Partitions (Problema com DDoS)

#### 3.6.1 Estrategia Atual de Particionamento

O Producer usa a **5-tuple do flow** como chave de particionamento:

```python
# pcap_producer.py
flow_key = f"{src_ip}-{dst_ip}-{src_port}-{dst_port}-{protocol}"
producer.send(topic, key=flow_key, value=packet)
```

**Funcionamento do Kafka:**
```
particao = hash(key) % num_particoes

# Exemplo com 3 particoes:
hash("192.168.1.1-10.0.0.1-80-443-TCP") % 3 = 0  → Particao 0
hash("192.168.1.2-10.0.0.1-80-443-TCP") % 3 = 2  → Particao 2
hash("192.168.1.3-10.0.0.5-80-443-TCP") % 3 = 0  → Particao 0
```

**Garantia**: Mesma key → Mesma particao → Ordem preservada dentro do flow.

#### 3.6.2 Problema: Hot Partition em Ataques DDoS

Em ataques DDoS, um unico flow pode gerar **milhoes de pacotes**:

```
Distribuicao normal (trafego benigno):
┌─────────────────────────────────────────────────┐
│ Particao 0: [~33% dos flows] ████████           │
│ Particao 1: [~33% dos flows] ████████           │
│ Particao 2: [~34% dos flows] ████████           │
└─────────────────────────────────────────────────┘
                    ↓ Balanceado

Distribuicao com DDoS (um flow dominante):
┌─────────────────────────────────────────────────┐
│ Particao 0: [1 flow DDoS]    ████████████████████████████│ ← HOT!
│ Particao 1: [flows normais]  ██                 │
│ Particao 2: [flows normais]  ██                 │
└─────────────────────────────────────────────────┘
                    ↓ Desbalanceado!
```

**Consequencias:**
- Consumer da particao 0 fica sobrecarregado
- Consumers das outras particoes ficam ociosos
- Latencia aumenta para TODOS os flows na particao 0
- Memoria do consumer pode estourar (acumulo de mensagens)

#### 3.6.3 Solucoes Potenciais

**Solucao 1: Aceitar o desbalanceamento (v0.1 - atual)**
```
Pros: Simples, ordem garantida
Cons: Uma particao sobrecarregada durante ataques
Quando usar: Desenvolvimento, baixo volume
```

**Solucao 2: Key por IP origem apenas**
```python
# Menos granular, melhor distribuicao
flow_key = f"{src_ip}"

# DDoS de multiplas origens se distribui melhor
# Mas pacotes do mesmo flow podem ir para particoes diferentes!
```
```
Pros: Melhor distribuicao em DDoS distribuido
Cons: Perde ordenacao por flow (precisa reordenar no consumer)
Quando usar: Ataques de multiplas origens
```

**Solucao 3: Sub-key com timestamp/contador**
```python
# Adiciona componente que varia
packet_count = get_packet_count(flow_key)
sub_key = f"{flow_key}-{packet_count // 10000}"

# A cada 10k pacotes, muda de particao
```
```
Pros: Distribui flows grandes
Cons: Complexidade, precisa reordenar no consumer
Quando usar: Flows muito grandes conhecidos
```

**Solucao 4: Particao round-robin para flows "quentes"**
```python
# Detecta flows com alto volume
if is_hot_flow(flow_key):
    partition = next_partition_round_robin()
else:
    partition = hash(flow_key) % num_partitions
```
```
Pros: Adapta dinamicamente
Cons: Precisa detectar hot flows, complexidade
Quando usar: Producao com monitoramento
```

**Solucao 5: Mais particoes**
```
# Aumentar de 3 para 10+ particoes
# Dilui o impacto de hot flows

Pros: Simples de implementar
Cons: Mais overhead, nao resolve 100%
Quando usar: Sempre que possivel em producao
```

#### 3.6.4 Recomendacao para IoT IDS

| Fase | Estrategia | Motivo |
|------|------------|--------|
| Desenvolvimento | Solucao 1 (aceitar) | Simplicidade |
| Prototipo | Solucao 5 (mais particoes, 6-10) | Baixo custo |
| Producao | Solucao 4 + 5 (deteccao + particoes) | Robustez |

#### 3.6.5 Monitoramento de Hot Partitions

Metricas a observar:
```
# Lag por particao (mensagens nao processadas)
kafka_consumer_lag{partition="0"} > threshold → ALERTA

# Throughput por particao
kafka_partition_messages_per_sec{partition="0"} >> outras → HOT

# Tempo de processamento por consumer
consumer_processing_time_p99 > threshold → SOBRECARGA
```

### 3.7 Padroes para Distribuicao

#### 3.7.1 Stateless Services
```python
# Estado vai para servicos externos
- Kafka (mensagens)
- Redis (cache/sessoes)
- PostgreSQL (persistencia)
```

#### 3.7.2 Configuration via Environment (12-Factor App)
```python
# NAO fazer (hardcoded)
bootstrap_servers = "localhost:9092"

# Fazer (environment variables)
bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
```

#### 3.7.3 Health Checks
```python
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "kafka_connected": producer.is_connected(),
        "uptime_seconds": get_uptime()
    }
```

#### 3.7.4 Graceful Shutdown
```python
@app.on_event("shutdown")
async def shutdown():
    producer.flush()  # Envia mensagens pendentes
    producer.close()  # Fecha conexao limpa
```

---

## 4. Simulacao de Trafego Realista com PCAPs

### 4.1 Contexto do Problema

O dataset CICIoT2023 contem:
- **~548 GB** de dados em PCAPs
- **33 tipos de ataques** em 7 categorias
- **105 dispositivos IoT** reais
- PCAPs separados por tipo de ataque

**Desafio**: Como misturar os PCAPs de diferentes ataques com trafego benigno para simular um fluxo de rede realista?

### 4.2 Como o CICIoT2023 Foi Gerado

Baseado no paper "CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment" (Neto et al., 2023):

#### 4.2.1 Topologia
- 105 dispositivos IoT reais (cameras, speakers, sensores)
- 7 Raspberry Pi como atacantes
- Network tap (Gigamon) para captura
- 2 monitores de rede com Wireshark

#### 4.2.2 Coleta de Dados
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Atacantes  │────►│ Network Tap │────►│  Monitores  │
│  (7 RPi)    │     │  (Gigamon)  │     │ (Wireshark) │
└─────────────┘     └─────────────┘     └─────────────┘
       │                                       │
       ▼                                       ▼
┌─────────────┐                         ┌─────────────┐
│   Vitimas   │                         │    PCAP     │
│ (98 IoT)    │                         │   Files     │
└─────────────┘                         └─────────────┘
```

#### 4.2.3 Dados Benignos
- 16 horas de captura
- Dispositivos em idle + interacoes humanas
- Sem scripts maliciosos ativos

#### 4.2.4 Dados de Ataque
- Cada ataque executado separadamente
- Todo trafego rotulado com o tipo de ataque
- Ferramentas: hping3, nmap, Mirai adaptado, DVWA, etc.

### 4.3 Distribuicao dos Dados

| Categoria | Rows | Percentual |
|-----------|------|------------|
| DDoS | 33,984,560 | 72.8% |
| DoS | 8,090,738 | 17.3% |
| Mirai | 2,634,124 | 5.6% |
| Benign | 1,098,195 | 2.4% |
| Spoofing | 486,504 | 1.0% |
| Recon | 354,565 | 0.8% |
| Web | 24,829 | 0.05% |
| BruteForce | 13,064 | 0.03% |

**Problema**: Em redes reais, trafego benigno >> trafego malicioso. O dataset e desbalanceado no sentido oposto.

---

## 5. Pesquisa Bibliografica: Traffic Replay e Mixing

### 5.1 Fontes Academicas Relevantes

#### 5.1.1 Survey Fundamental
**"Network Traffic Generation: A Survey and Methodology"**
ACM Computing Surveys, 2022
- [Link](https://dl.acm.org/doi/10.1145/3488375)
- Top 10 geradores: iperf2, netperf, httperf, moongen, scapy, pktgen, netcat, **TCPreplay**, iperf3, DPDK
- Metodologia para criacao de datasets

#### 5.1.2 Framework de Sintese
**"Network Traffic Synthesis and Simulation Framework for Cybersecurity Exercise Systems"**
Computers, Materials & Continua, 2024
- [Link](https://www.techscience.com/cmc/v80n3/57885/html)
- Usa CTGAN (Conditional Tabular GAN) para sintetizar trafego
- Gera cenarios realistas para exercicios de cybersecurity

#### 5.1.3 Geracao com LLMs
**"GPT on the Wire: Realistic Network Traffic with Large Language Models"**
Computer Networks, 2025
- [Link](https://www.sciencedirect.com/science/article/pii/S1389128625002762)
- Usa LLMs para gerar conversacoes de rede realistas
- Abordagem inovadora para traffic generation

#### 5.1.4 Dataset Real-Time
**"A Real Network Environment Dataset for Traffic Analysis"**
Nature Scientific Data, 2025
- [Link](https://www.nature.com/articles/s41597-025-04876-2)
- Coleta em tempo real vs datasets estaticos
- Metodologia para captura continua

#### 5.1.5 Compatibilidade de Features
**"Revisiting Network Traffic Analysis: Compatible Network Flows for ML Models"**
arXiv, 2025
- [Link](https://arxiv.org/html/2511.08345)
- Analisa CICIoT2023, Bot-IoT, IoT-23
- Ferramenta HERA para extracao consistente de features

### 5.2 Ferramentas para Traffic Replay

#### 5.2.1 TCPreplay
- [Site oficial](https://tcpreplay.appneta.com/)
- [GitHub](https://github.com/appneta/tcpreplay)
- Replay de PCAPs em tempo real
- Reescrita de headers L2/L3/L4
- Suporte a dual NIC

**Limitacoes**:
- Stateless: nao responde a congestao
- Continua enviando mesmo com link down
- Nao simula TCP flow control realista

#### 5.2.2 Scapy
- Manipulacao de pacotes em Python
- Mais flexivel, mas mais lento
- Permite modificacao programatica

#### 5.2.3 DPDK pktgen
- Alto desempenho (line rate)
- Mais complexo de configurar
- Para testes de stress

### 5.3 Metodologia para Mixing de Trafego

Baseado na literatura, a metodologia recomendada:

#### 5.3.1 Criterios para Merge
1. **Periodo temporal similar**: Dados de epocas proximas
2. **Redes similares**: Topologias comparaveis
3. **Replay em ambiente local**: Unifica dependencias temporais
4. **Balanceamento**: Proporcoes realistas benigno/ataque

#### 5.3.2 Proporcoes Realistas

Em redes reais:
- **~95-99%** trafego benigno
- **~1-5%** trafego malicioso (quando sob ataque)
- **~0%** malicioso (operacao normal)

Para avaliacao de IDS, proporcoes comuns em pesquisa:
- 80/20 (benigno/ataque) - teste balanceado
- 95/5 - mais realista
- 99/1 - cenario de producao

#### 5.3.3 Base-Rate Fallacy

> "Se o numero de flows de ataque e pequeno comparado ao benigno, a taxa de falsos alarmes sera alta a menos que o classificador tenha recall muito alto."

Implicacao: O modelo precisa ser muito preciso para funcionar em cenarios realistas.

---

## 6. Proposta de Implementacao Futura

### 6.1 Traffic Mixer v1.0 (Planejado)

```python
class TrafficMixer:
    """
    Mistura PCAPs de diferentes tipos para simular trafego realista.
    """

    def __init__(self, config: MixerConfig):
        self.benign_ratio = config.benign_ratio  # e.g., 0.95
        self.attack_distribution = config.attack_distribution
        # {
        #     "DDoS": 0.02,
        #     "DoS": 0.01,
        #     "Recon": 0.01,
        #     "Mirai": 0.01
        # }

    def mix(self, benign_pcaps: List[Path],
            attack_pcaps: Dict[str, Path]) -> Iterator[Packet]:
        """
        Gera stream de pacotes misturados seguindo as proporcoes.

        Estrategias possiveis:
        1. Round-robin com pesos
        2. Distribuicao temporal (ataques em rajadas)
        3. Aleatorio com seed reproduzivel
        """
        pass
```

### 6.2 Estrategias de Mixing

#### Estrategia 1: Proporcional Simples
```
Para cada 100 pacotes:
├── 95 pacotes benignos
├── 2 pacotes DDoS
├── 1 pacote DoS
├── 1 pacote Recon
└── 1 pacote Mirai
```

#### Estrategia 2: Temporal Realista (Rajadas)
```
Timeline:
├── 0-10min: 100% benigno (operacao normal)
├── 10-15min: 70% benigno + 30% DDoS (ataque inicia)
├── 15-20min: 40% benigno + 60% DDoS (pico do ataque)
├── 20-25min: 80% benigno + 20% DDoS (ataque diminui)
└── 25-30min: 100% benigno (operacao normal)
```

#### Estrategia 3: Multi-Ataque Concorrente
```
Simula cenario real onde multiplos ataques ocorrem:
├── DDoS em rajadas
├── Recon continuo (baixa intensidade)
├── Mirai tentando infectar
└── Trafego benigno como baseline
```

### 6.3 Roadmap de Implementacao

| Fase | Tarefa | Prioridade |
|------|--------|------------|
| v0.1 | Producer basico (FEITO) | - |
| v0.2 | Consumer com extracao de features | Alta |
| v0.3 | Traffic Mixer simples (proporcional) | Media |
| v0.4 | Traffic Mixer avancado (temporal) | Baixa |
| v1.0 | Servicos distribuidos | Futura |

---

## 7. Referencias

### Papers
1. Neto, E.C.P. et al. "CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment." Sensors 2023, 23, 5941. https://doi.org/10.3390/s23135941

2. "Network Traffic Generation: A Survey and Methodology." ACM Computing Surveys, 2022. https://dl.acm.org/doi/10.1145/3488375

3. "Network Traffic Synthesis and Simulation Framework for Cybersecurity Exercise Systems." CMC, 2024. https://www.techscience.com/cmc/v80n3/57885/html

### Ferramentas
- TCPreplay: https://tcpreplay.appneta.com/
- Scapy: https://scapy.net/
- DPKT: https://dpkt.readthedocs.io/

### Datasets Relacionados
- CICIoT2023: https://www.unb.ca/cic/datasets/iotdataset-2023.html
- IoT-23: https://www.stratosphereips.org/datasets-iot23
- CICIDS2017: https://www.unb.ca/cic/datasets/ids-2017.html

---

## 8. Notas de Implementacao

### 8.1 Ordem de Prioridades
1. **Primeiro**: Terminar v0.1 completa (producer + consumer funcionando)
2. **Segundo**: Validar pipeline end-to-end localmente
3. **Terceiro**: Implementar mixing simples
4. **Quarto**: Distribuir servicos

### 8.2 Decisoes Arquiteturais Pendentes
- [ ] Framework web: FastAPI vs Flask
- [ ] Orquestracao: Kubernetes vs Docker Swarm vs Compose
- [ ] Observabilidade: Stack a definir (Prometheus? ELK?)
- [ ] Storage: Redis vs PostgreSQL para estado

---

*Documento vivo - sera atualizado conforme o projeto evolui.*
