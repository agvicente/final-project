# Isolamento de Experimentos - Sistema IDS Streaming

## 1. Problema: Por que Experimentos Interferem Uns nos Outros

### 1.1 Tópicos Kafka Compartilhados

Quando múltiplos experimentos consecutivos executam no mesmo sistema, eles compartilham os **mesmos tópicos Kafka**:

```
Experimento 1: PCAP benign → [packets] → [flows] → Detector1
                                  ↑
                                 COMPARTILHADO
                                  ↓
Experimento 2: PCAP attack → [packets] → [flows] → Detector2
```

Sem limpeza entre experimentos, o tópico `packets` pode conter:
- Últimos pacotes do experimento anterior
- Dados duplicados se o experimento foi re-executado
- Mistura de dados de múltiplas corridas

### 1.2 Consumer Groups com Offsets Persistentes

Kafka rastreia a "posição de leitura" de cada consumer group usando **offsets**:

```
Tópico 'packets':
┌─────┬─────┬─────┬─────┬─────┬─────┐
│ P0  │ P1  │ P2  │ P3  │ P4  │ P5  │  (Mensagens no tópico)
└─────┴─────┴─────┴─────┴─────┴─────┘
                      ↑
             Offset do "experiment-flow-processor"
             (Consumer group)
```

**Problema**: Se o offset não for resetado:
- Experimento 1 lê P0-P5, offset avança para 6
- Experimento 2 inicia lendo a partir do offset 6 (pulando P0-P5!)
- Com `auto_offset_reset="earliest"`, Kafka procura offset persistido no Zookeeper/broker
- Se tópico não foi truncado, dados antigos ainda existem

### 1.3 Acúmulo de Dados Entre Execuções

**Cenários comuns que causam problemas**:

1. **Tópico cresce indefinidamente**
   - Cada experimento adiciona dados
   - Sem purga, tópico acumula GB de dados históricos
   - Memoria do broker aumenta
   - Busca de dados anteriores fica lenta

2. **Consumer groups não limpam estado**
   - Grupo `experiment-flow-processor` retém offsets
   - Grupo `experiment-detector` retém offsets
   - Dados históricos de métricas/clusters acumulam

3. **Dados acidentalmente reprocessados**
   ```
   Experimento A: Injeta PCAP benign (1000 pacotes)
   Experimento B: Injeta PCAP attack (500 pacotes)
   Detector B lê AMBOS os PCAPs (1500 pacotes) se não houver limpeza!
   → Dados incorretos, ground truth quebrado, métricas inválidas
   ```

### 1.4 Impacto na Validade dos Experimentos

| Problema | Impacto | Severidade |
|----------|--------|-----------|
| Tópicos não truncados | Detector processa dados antigos | CRÍTICO |
| Offsets não resetados | Consumer pula dados ou reprocessa | CRÍTICO |
| Dados acumulados | Anomalias falsas, métricas inválidas | CRÍTICO |
| Memory leak | Degradação de performance | ALTO |

---

## 2. Solução: Estratégia de Isolamento

### 2.1 Princípios de Design

A estratégia de isolamento segue 3 princípios:

1. **Isolamento Completo**: Cada experimento começa com estado limpo
2. **Determinismo**: Múltiplas execuções do mesmo experimento produzem resultados idênticos
3. **Segurança**: Falhas na limpeza não interrompem o experimento (fallback gracioso)

### 2.2 Abordagem de 3 Camadas

```
┌────────────────────────────────────────────────────────┐
│ CAMADA 1: Purga de Tópicos Kafka                       │
│ ├─ Deletar/recriar tópicos 'packets', 'flows', 'alerts'│
│ └─ Zeroar offsets de consumer groups                   │
├────────────────────────────────────────────────────────┤
│ CAMADA 2: IDs Únicos por Experimento                   │
│ ├─ Gerar experiment_id único (UUID + timestamp)        │
│ └─ Usar em group_id para rastrear independentemente    │
├────────────────────────────────────────────────────────┤
│ CAMADA 3: Validação no Início do Experimento           │
│ └─ Verificar tópicos estão vazios antes de iniciar     │
└────────────────────────────────────────────────────────┘
```

### 2.3 Purga Automática de Tópicos

**Procedimento**:

```python
def purge_kafka_topics(
    bootstrap_servers: str,
    topics: List[str] = ["packets", "flows", "alerts"],
    timeout_ms: int = 30000
) -> Dict[str, bool]:
    """
    Deleta e recria tópicos Kafka para isolamento de experimentos.

    Garante:
    - Tópicos vazios e prontos para novos dados
    - Consumer groups resetados
    - Zero possibilidade de dados antigos interferindo

    Returns: Dicionário indicando sucesso/falha por tópico
    """
```

**Operações executadas**:

1. **Deletar tópico** (se existir)
   - Kafka remove todas as partições
   - Dados são perdidos permanentemente
   - Consumer group metadata é removido

2. **Recriar tópico**
   - Partições zeradas
   - Offsets resetados
   - Pronto para novos dados

3. **Validar estado**
   - Conectar como consumer
   - Verificar tópico vazio
   - Confirmar offset em 0

**Timeline**:
```
T0:  Iniciar purga
T1:  Kafka inicia limpeza interna (~500ms)
T2:  Recriar tópicos (~500ms)
T3:  Validação (~200ms)
T4:  Pronto para experimento (~2s total)
```

### 2.4 Group IDs Únicos por Experimento

Cada experimento recebe seu próprio **group ID único**:

```python
# Gerar experiment_id
experiment_id = f"{datetime.now().isoformat()}__{uuid.uuid4().hex[:8]}"
# Exemplo: "2025-02-25T18:30:45.123456__a7f2c9e1"

# Consumer groups específicos do experimento
flow_consumer_group = f"exp-{experiment_id}-flow-processor"
detector_group = f"exp-{experiment_id}-detector"
```

**Benefícios**:

1. **Rastreabilidade**: Cada execução deixa footprint único
2. **Paralelização**: Múltiplos experimentos podem rodar em paralelo
3. **Debugging**: Logs incluem experiment_id, fácil encontrar bugs
4. **Versionamento**: Histórico completo de qual experimento usou qual grupo

### 2.5 Por que Manter `auto_offset_reset="earliest"`

**Configuração recomendada**:
```python
config = StreamingDetectorConfig(
    auto_offset_reset="earliest",  # SEMPRE usar earliest
    ...
)
```

**Razão**:

- **`earliest`**: Começa do início do tópico (desde partition offset 0)
- **`latest`**: Começa do final do tópico (pula dados anteriores)

Para experimentos reproduzíveis:
- Precisamos que TODOS os dados sejam processados
- Com `latest`, se tópico não está vazio, dados são pulados
- Com `earliest`, garantimos determinismo

**Invariante de segurança**:
```
SE (tópico está vazio)
   ENTÃO (earliest == latest)
   ENTÃO (determinismo garantido)
```

Com purga automática garantindo tópico vazio → `earliest` é seguro e determinístico.

---

## 3. Implementação: Componentes Técnicos

### 3.1 Função `purge_kafka_topics()`

**Localização**: `src/utils/kafka_utils.py`

**Signature**:
```python
def purge_kafka_topics(
    bootstrap_servers: str = "localhost:9092",
    topics: Optional[List[str]] = None,
    timeout_ms: int = 30000,
    verbose: bool = False
) -> Dict[str, bool]:
    """
    Purga tópicos Kafka para isolar experimentos.

    Deletar/recriar é preferível a apenas truncar porque:
    1. Zera offsets de consumer groups (mais limpo)
    2. Remove metadata acumulada
    3. Garante estado "novo" para reproduzibilidade

    Args:
        bootstrap_servers: Endereço do Kafka
        topics: Lista de tópicos a purgar. Default: ["packets", "flows", "alerts"]
        timeout_ms: Timeout para operações Kafka
        verbose: Log detalhado de cada passo

    Returns:
        {"packets": True, "flows": True, "alerts": True}
        (True = sucesso, False = erro)

    Raises:
        KafkaConnectionError: Se não conseguir conectar ao Kafka
        KafkaAdminError: Se falhar ao deletar/recriar (opcional, ver fallback)
    """
```

**Implementação interna**:

```python
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError, UnknownTopicOrPartError

def purge_kafka_topics(...):
    admin = KafkaAdminClient(bootstrap_servers=bootstrap_servers)
    results = {}

    for topic in topics:
        try:
            # 1. Deletar
            admin.delete_topics([topic], timeout_ms=timeout_ms)
            time.sleep(0.5)  # Aguardar limpeza interna

            # 2. Recriar com configurações padrão
            new_topic = NewTopic(
                name=topic,
                num_partitions=1,
                replication_factor=1,
                topic_configs={"retention.ms": "-1"}  # Retenção infinita
            )
            admin.create_topics([new_topic], timeout_ms=timeout_ms)
            time.sleep(0.5)

            # 3. Validar
            metadata = admin.describe_topics([topic])
            results[topic] = metadata[topic]["partitions"][0]["leader"] >= 0

        except UnknownTopicOrPartError:
            # Tópico não existe = OK, cria novo
            results[topic] = True
        except Exception as e:
            logger.error(f"Erro ao purgar {topic}: {e}")
            results[topic] = False

    admin.close()
    return results
```

### 3.2 Geração de `experiment_id` Único

**Localização**: `src/utils/experiment_utils.py`

**Função**:
```python
from datetime import datetime
import uuid
import os

def generate_experiment_id(
    prefix: str = "exp"
) -> str:
    """
    Gera ID único para isolamento de experimentos.

    Format: {prefix}-{ISO8601_timestamp}__{uuid_8chars}
    Example: "exp-2025-02-25T18:30:45.123456__a7f2c9e1"

    Inclui:
    - Timestamp: Ordenação temporal
    - UUID: Unicidade global
    - Curto mas legível

    Args:
        prefix: Prefixo do ID (padrão: "exp")

    Returns:
        String de ID único
    """
    timestamp = datetime.now().isoformat()  # "2025-02-25T18:30:45.123456"
    uuid_short = uuid.uuid4().hex[:8]       # "a7f2c9e1"
    return f"{prefix}-{timestamp}__{uuid_short}"


def get_experiment_group_ids(experiment_id: str) -> Dict[str, str]:
    """
    Retorna group IDs Kafka específicos para este experimento.

    Returns:
        {
            "flow_processor": "exp-2025-02-25T18:30:45.123456__a7f2c9e1-flow-processor",
            "detector": "exp-2025-02-25T18:30:45.123456__a7f2c9e1-detector",
            "alerts": "exp-2025-02-25T18:30:45.123456__a7f2c9e1-alerts"
        }
    """
    return {
        "flow_processor": f"{experiment_id}-flow-processor",
        "detector": f"{experiment_id}-detector",
        "alerts": f"{experiment_id}-alerts"
    }
```

### 3.3 Integração em `run_experiment.py`

**Modificações necessárias**:

```python
# No início de main()
def main():
    parser = argparse.ArgumentParser(...)
    # ... argumentos existentes ...

    # NOVO: argumento para controlar isolamento
    parser.add_argument(
        "--skip-purge",
        action="store_true",
        help="Pular purga de tópicos (apenas para debugging)"
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="ID de experimento (gerado automaticamente se não fornecido)"
    )

    args = parser.parse_args()

    # Gerar experiment_id
    experiment_id = args.experiment_id or generate_experiment_id()
    logger.info(f"Experiment ID: {experiment_id}")

    # NOVO: Purgar tópicos no início
    if not args.skip_purge:
        logger.info("=" * 60)
        logger.info("PURGA: Limpando tópicos Kafka")
        logger.info("=" * 60)

        purge_result = purge_kafka_topics(
            bootstrap_servers=args.bootstrap_servers,
            verbose=args.verbose
        )

        if not all(purge_result.values()):
            logger.warning("Alguns tópicos não foram purgados:")
            for topic, success in purge_result.items():
                status = "OK" if success else "ERRO"
                logger.warning(f"  {topic}: {status}")

            if not args.force:  # Se não --force, abortar
                logger.error("Use --force para continuar mesmo com purga incompleta")
                sys.exit(1)

    # Obter group IDs únicos
    group_ids = get_experiment_group_ids(experiment_id)

    # MODIFICADO: Usar group IDs únicos
    flow_consumer_process = start_flow_consumer(
        group_id=group_ids["flow_processor"],  # NOVO
        verbose=args.verbose
    )

    results = run_detector(
        group_id=group_ids["detector"],  # NOVO
        pcap_path=gt_pcap,
        ...
    )

    # Log do experiment_id nos resultados
    config_used["experiment_id"] = experiment_id
    config_used["group_ids"] = group_ids
```

**Modificação em `start_flow_consumer()`**:
```python
def start_flow_consumer(
    group_id: str,  # NOVO parâmetro
    verbose: bool = False
) -> subprocess.Popen:
    cmd = [
        "python", "-m", "src.consumer.flow_consumer",
        "--group-id", group_id,  # MODIFICADO
        "--from-beginning",
    ]
    ...
```

**Modificação em `run_detector()`**:
```python
def run_detector(
    group_id: str,  # NOVO parâmetro
    pcap_path: str,
    ...
) -> Dict[str, Any]:
    config = StreamingDetectorConfig(
        group_id=group_id,  # MODIFICADO
        auto_offset_reset="earliest",
        ...
    )
    ...
```

### 3.4 Estrutura de Arquivos

```
streaming/
├── src/
│   ├── utils/                          (NOVO)
│   │   ├── __init__.py
│   │   ├── kafka_utils.py              (NOVO - purge_kafka_topics)
│   │   └── experiment_utils.py         (NOVO - generate_experiment_id)
│   ├── producer/
│   ├── consumer/
│   └── detector/
├── scripts/
│   └── run_experiment.py               (MODIFICADO - integração)
└── docs/
    └── experiment-isolation.md         (NOVO - este arquivo)
```

---

## 4. Garantias: O que a Feature Assegura

### 4.1 Zero Interferência Entre Experimentos

**Invariante**: Após purga bem-sucedida, tópicos Kafka estão em estado inicial:

```
ANTES:
┌─────┬─────┬─────┬─────┬─────┐
│ P0  │ P1  │ P2  │ P3  │ P4  │  (Dados do experimento anterior)
└─────┴─────┴─────┴─────┴─────┘
 offset=0

DEPOIS:
┌─────┐
│     │  (Tópico vazio, pronto para novos dados)
└─────┘
 offset=0
```

**Verificação automática**:
```python
# Antes de iniciar detector
consumer = KafkaConsumer(
    topic,
    bootstrap_servers=bootstrap_servers,
    auto_offset_reset="earliest",
    group_id=f"verify-{experiment_id}"
)
msg = consumer.poll(timeout_ms=1000)
assert msg is None, "Tópico não está vazio!"
consumer.close()
```

### 4.2 Sem Acúmulo de Memória

**Impacto**:
- Antes: Broker Kafka acumula ~100MB por experimento × 100 experimentos = 10GB
- Depois: Broker mantém ~0MB (tópicos sempre vazios entre experimentos)

**Métrica**:
```bash
# Monitorar tamanho do broker
docker exec kafka du -sh /var/lib/kafka/data
```

**Expected output**:
```
100M /var/lib/kafka/data  (tamanho base, nunca cresce)
```

### 4.3 Idempotência Garantida

**Propriedade**: Múltiplas execuções do mesmo PCAP produzem resultados idênticos:

```python
# Experimento A, execução 1
metrics_1 = run_experiment(pcap="benign.pcap", experiment_id="exp-001")
# Output: {"precision": 0.95, "recall": 0.93, "f1": 0.94, ...}

# Experimento A, execução 2 (mesmo PCAP, novo experiment_id)
metrics_2 = run_experiment(pcap="benign.pcap", experiment_id="exp-002")
# Output: {"precision": 0.95, "recall": 0.93, "f1": 0.94, ...}

# Idempotência: metrics_1 == metrics_2
assert metrics_1["f1"] == metrics_2["f1"]  # ✓ PASSA
```

**Por que funciona**:
1. Tópicos sempre começam vazios (purga)
2. Consumer grupos começam no offset 0 (resetado)
3. PCAP determinístico → flows determinísticos → detecção determinística
4. Única variável: timing do sistema (mitigado com janelas deslizantes)

---

## 5. Trade-offs: Decisões de Design

### 5.1 Por que Deletar/Recriar vs. Outras Abordagens

| Abordagem | Pros | Cons | Adotado? |
|-----------|------|------|----------|
| **Deletar/recriar** | Limpeza total, sem dados fantasma | ~2s overhead | ✅ SIM |
| **Truncar tópico** | Mais rápido (~200ms) | Metadata suja, offsets confusos | ❌ NÃO |
| **Novo tópico por experimento** | Ultra-isolado, debug fácil | Explosão de tópicos (1000+ após 100 exp) | ❌ NÃO |
| **Reset offsets sem truncar** | Rápido | Dados antigos reprocessados | ❌ NÃO |

**Justificativa da escolha**:
```
Confiabilidade >> Velocidade (para pesquisa)
2s por experimento << valor de garantia de isolamento
```

### 5.2 Overhead de ~2 Segundos por Experimento

**Timeline detalhado**:

```
start_experiment()
  │
  ├─ purge_kafka_topics()           ~2000ms
  │  ├─ KafkaAdminClient connect    ~100ms
  │  ├─ Delete tópicos              ~500ms
  │  ├─ Wait Kafka internal cleanup  ~500ms
  │  ├─ Create tópicos              ~500ms
  │  ├─ Validate state              ~200ms
  │  └─ Close admin client          ~100ms
  │
  ├─ inject_pcap()                   Variable (sec)
  ├─ run_detector()                  Variable (sec)
  └─ save_results()                  ~100ms

Total overhead: 2% para experimentos com duração > 100s
               20% para experimentos com duração ~10s
```

**Decisão**:
- Overhead aceitável para a garantia fornecida
- Crítico para pesquisa (reproduzibilidade > velocidade)
- Para produção: avaliar paralelização (múltiplas partições)

### 5.3 Segurança: Continua se Purga Falhar

**Padrão de tratamento**:

```python
try:
    purge_result = purge_kafka_topics(...)

    if not all(purge_result.values()):
        failed_topics = [t for t, success in purge_result.items() if not success]
        logger.warning(f"Purga parcial: {failed_topics}")

        if args.force:
            logger.warning("Continuando mesmo com purga incompleta (--force)")
            # Proceder com cuidado
        else:
            logger.error("Use --force para continuar")
            sys.exit(1)
except Exception as e:
    logger.error(f"Erro crítico na purga: {e}")
    if args.force:
        logger.warning("Pulando purga (--force)")
    else:
        sys.exit(1)
```

**Três niveis de segurança**:

1. **Nível 1: Validação pós-purga**
   - Conecta como consumer, verifica tópico vazio
   - Se vazio → OK, proceder
   - Se não vazio → aviso, perguntar `--force`

2. **Nível 2: Validação pré-detector**
   - Antes de iniciar detecção, log: "Processando X flows"
   - Se X > esperado, aviso potencial reprocessamento
   - Se X muito maior → possível interferência anterior

3. **Nível 3: Análise pós-experimento**
   - Métricas anormais (F1 muito alto/baixo)
   - Indicador de dados contaminados
   - Recomendação: reexecutar com --force-purge

---

## 6. Fluxo de Execução Detalhado

### 6.1 Execução Normal

```
ENTRADA:
  pcap: data/benign.pcap
  experiment_id: (gerado automaticamente)
  --no-skip-purge (default)

┌──────────────────────────────────────────────────┐
│ 1. run_experiment.py iniciado                    │
└──────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────┐
│ 2. Gerar experiment_id                           │
│    exp-2025-02-25T18:30:45.123456__a7f2c9e1    │
└──────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────┐
│ 3. purge_kafka_topics()                          │
│    ├─ Delete [packets, flows, alerts]   ~1.0s   │
│    ├─ Recreate [packets, flows, alerts] ~1.0s   │
│    └─ Verify state                      ~0.2s   │
└──────────────────────────────────────────────────┘
         │ (sucesso)
         ▼
┌──────────────────────────────────────────────────┐
│ 4. inject_pcap(benign.pcap)                      │
│    ├─ Read PCAP file                            │
│    ├─ Extract packets                           │
│    └─ Send to Kafka [packets] topic             │
└──────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────┐
│ 5. start_flow_consumer()                         │
│    group_id: "exp-2025-...-flow-processor"     │
│    ├─ Connect to [packets]                      │
│    ├─ auto_offset_reset: earliest               │
│    └─ Output to [flows]                         │
└──────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────┐
│ 6. run_detector()                                │
│    group_id: "exp-2025-...-detector"           │
│    ├─ Connect to [flows]                        │
│    ├─ MicroTEDAclus detection                   │
│    └─ Produce [alerts]                          │
└──────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────┐
│ 7. Collect metrics & save results                │
│    ├─ run_meta.json (experiment_id)             │
│    ├─ detection_results.json                    │
│    ├─ metrics_windowed.csv                      │
│    ├─ clusters_state.jsonl                      │
│    └─ system_usage.csv                          │
└──────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────┐
│ 8. Log sucesso                                   │
│    ✅ EXPERIMENTO CONCLUÍDO                     │
└──────────────────────────────────────────────────┘
```

### 6.2 Execução com Purga Falhada

```
CENÁRIO: Kafka está respondendo lentamente

ENTRADA:
  pcap: data/attack.pcap
  --skip-purge: False (default)
  --force: False (default)

┌──────────────────────────────────────────────────┐
│ 1-2. Generate experiment_id                      │
└──────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────┐
│ 3. purge_kafka_topics()                          │
│    ├─ Delete packets... [TIMEOUT]               │
│    └─ ❌ ERROR: timeoutError                    │
└──────────────────────────────────────────────────┘
         │ (falha)
         ▼
┌──────────────────────────────────────────────────┐
│ 4. Log erro e perguntar                          │
│    ⚠️ "Purga falhou. Use --force para continuar?"│
└──────────────────────────────────────────────────┘
         │ (usuário não usou --force)
         ▼
┌──────────────────────────────────────────────────┐
│ 5. sys.exit(1)                                   │
│    ❌ EXPERIMENTO CANCELADO                     │
└──────────────────────────────────────────────────┘
```

### 6.3 Paralelização Futura (Multi-Experimento)

```
Experimento 1:
  group_id: "exp-2025-02-25T18:30:00__a7f2c9e1-detector"
  Lê flows → Processa com cluster1 → Alerta

                    [Tópicos Kafka Compartilhados]
                    ┌─────────────────────────────┐
                    │ [packets]  [flows]  [alerts]│
                    └─────────────────────────────┘
                    (dados interpolados, mas
                     groups isolados = sem crash)

Experimento 2 (paralelo):
  group_id: "exp-2025-02-25T18:35:00__f3k8l2m9-detector"
  Lê flows → Processa com cluster2 → Alerta

Vantagem: Dois experimentos usam mesmo Kafka, mas sem interferência
Graças a: group_ids únicos + consumer group isolation natural do Kafka
```

---

## 7. Exemplos de Uso

### 7.1 Execução Padrão (Isolamento Automático)

```bash
cd /Users/augusto/mestrado/final-project/streaming

# Simples: purga automática, experiment_id gerado
python scripts/run_experiment.py \
  --pcap ../data/raw/PCAP/Benign/Benign_Final.pcap \
  --output results/exp_benign.json

# Output:
# Experiment ID: exp-2025-02-25T18:30:45.123456__a7f2c9e1
# PURGA: Limpando tópicos Kafka
#   packets: OK
#   flows: OK
#   alerts: OK
# ✅ Purga concluída em 2.1s
# ETAPA 1: Injetando PCAP no Kafka
# ✅ PCAP injetado: 12345 pacotes em 3.2s
# ETAPA 2: Iniciando FlowConsumer
# ✅ FlowConsumer iniciado
# ETAPA 3: Executando detecção
# ✅ EXPERIMENTO CONCLUÍDO
```

### 7.2 Múltiplos Experimentos Sequenciais

```bash
# Cada experimento: purga automática → isolamento garantido

for pcap in data/raw/PCAP/*/*.pcap; do
  python scripts/run_experiment.py \
    --pcap "$pcap" \
    --output "results/$(basename $pcap).json"
done

# Cada iteração:
#   - Tópicos purgados
#   - experiment_id único gerado
#   - Dados isolados
#   - Resultados salvos com experiment_id
```

### 7.3 Debugging: Pular Purga

```bash
# CUIDADO: Apenas para debugging, não para experimentos válidos

python scripts/run_experiment.py \
  --pcap data/raw/PCAP/Test/Test.pcap \
  --skip-purge \
  --verbose

# Útil para:
# - Testes rápidos (não precisa purgar tópicos)
# - Entender estado do sistema (dados residuais visíveis)
# - Desenvolvimento (agilizar feedback)

# NUNCA usar para resultados de pesquisa!
```

### 7.4 Forcing Continuation (Fallback)

```bash
# Se purga falhar mas você quer continuar (risco!)

python scripts/run_experiment.py \
  --pcap data/raw/PCAP/Attack/DDoS.pcap \
  --force \
  --output results/ddos.json

# Se purga falhar:
# ⚠️ Continuando mesmo com purga incompleta (--force)
# 📝 Possível dados contaminados em tópicos!
# 📝 Métricas podem estar inválidas!
```

---

## 8. Checklist de Implementação

- [ ] Criar `src/utils/kafka_utils.py` com `purge_kafka_topics()`
- [ ] Criar `src/utils/experiment_utils.py` com `generate_experiment_id()`
- [ ] Modificar `scripts/run_experiment.py`:
  - [ ] Importar funções de utils
  - [ ] Adicionar argumento `--skip-purge`
  - [ ] Adicionar argumento `--force`
  - [ ] Chamar `purge_kafka_topics()` no início
  - [ ] Gerar `experiment_id`
  - [ ] Usar `experiment_id` em group_ids
  - [ ] Passar `group_id` para `start_flow_consumer()` e `run_detector()`
  - [ ] Incluir `experiment_id` em `run_meta.json`
- [ ] Testar purga com Kafka rodando
- [ ] Testar com `--skip-purge` para fallback
- [ ] Documentar em QUICK_START.md
- [ ] Adicionar testes unitários para `purge_kafka_topics()`

---

## 9. Leitura Complementar

- **Kafka Documentation**: https://kafka.apache.org/documentation/#consumerconfigs
- **Consumer Groups**: Como Kafka rastreia offsets por grupo
- **Concept Drift**: Por que isolamento importa para detecção
- **Experiment Methodology**: `docs/methodology/experiment-methodology.md` (Semana 5)

---

## 10. Changelog

| Data | Versão | Mudança |
|------|--------|---------|
| 2025-02-25 | 1.0 | Documento inicial - Isolamento de Experimentos |
| TBD | 1.1 | Implementação de `purge_kafka_topics()` |
| TBD | 1.2 | Integração em `run_experiment.py` |
| TBD | 2.0 | Suporte para paralelização (múltiplos Kafka brokers) |

---

**Documento autorizado para**: Dissertação UFMG - Detecção de Anomalias em IoT com Clustering Evolutivo

**Responsável**: Augusto
**Última atualização**: 2025-02-25
