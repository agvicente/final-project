# Sincronização FlowConsumer-Detector

**Criado:** 2026-03-10
**Última Atualização:** 2026-03-10
**Versão:** 1.0

---

## 1. Problema: Race Condition no Pipeline

### 1.1 Arquitetura do Pipeline

O orquestrador (`run_experiment.py`) coordena 3 estágios sequenciais:

```
PCAP → [packets] → FlowConsumer → [flows] → Detector → resultados
```

O problema é que FlowConsumer e Detector operam como processos independentes
sobre tópicos Kafka, sem sincronização explícita.

### 1.2 O Race Condition

**Sequência original (defeituosa):**

```
T0:  Injetar 100k packets no Kafka          (~42s)
T1:  Iniciar FlowConsumer (background)
T2:  sleep(3s)                               ← única "sincronização"
T3:  Iniciar Detector (lê tópico 'flows')
T4:  Detector lê ~500 flows disponíveis
T5:  Detector idle por 10s (IDLE_LIMIT)      ← FlowConsumer ainda processando!
T6:  Detector encerra                        ← perdeu ~7000 flows
```

**Causa raiz:** O Detector tem um timeout de inatividade de 10 segundos
(`IDLE_LIMIT = 10` polls × 1s). Quando o FlowConsumer tem uma pausa na
produção de flows (ex: esperando timeout de evento para fechar flows ativos),
o Detector interpreta como "fim dos dados" e encerra prematuramente.

### 1.3 Dois Mecanismos de Fechamento de Flow

O FlowConsumer fecha flows por dois mecanismos:

1. **Timeout de evento** (`flow_timeout_seconds = 60s`): usa o relógio do PCAP
   (`_pcap_clock`), não wall-clock. Flows são fechados quando o timestamp do
   último pacote excede o timeout.

2. **Flush no shutdown** (`_flush_all_flows()`): quando o processo recebe
   SIGTERM, todos os flows ativos são fechados e publicados.

**Impacto:** Sem o flush, flows com menos de 60s de atividade (em tempo de PCAP)
permanecem "ativos" e nunca são publicados no tópico `flows`.

### 1.4 Resultados Antes da Correção

| Métrica | Valor |
|---------|-------|
| Packets injetados | 100.000 |
| Flows esperados | ~7.500 |
| Flows processados pelo detector | ~500 |
| Flows perdidos | ~7.000 (93%) |

---

## 2. Solução: Sincronização em 3 Fases

### 2.1 Princípio

Monitorar o tópico `flows` e esperar o FlowConsumer terminar antes de iniciar
o Detector. A sincronização tem 3 fases:

```
Fase 1: Esperar flows pararem de crescer (timeout de evento esgotado)
Fase 2: Terminar FlowConsumer (SIGTERM → flush dos flows ativos)
Fase 3: Esperar flows do flush aparecerem no Kafka
```

### 2.2 Sequência Corrigida

```
T0:   Injetar 100k packets no Kafka              (~42s)
T1:   Iniciar FlowConsumer (background)
T2:   sleep(3s) — inicialização do consumer
T3:   SINCRONIZAÇÃO Fase 1:
      │  Monitorar end offset do tópico 'flows'
      │  Esperar estabilizar por 5s
      │  flows: 0 → 977 → 2308 → ... → 7141 (estável)
T4:   SINCRONIZAÇÃO Fase 2:
      │  SIGTERM → FlowConsumer
      │  FlowConsumer chama _flush_all_flows()
      │  ~400 flows adicionais publicados
T5:   SINCRONIZAÇÃO Fase 3:
      │  Esperar tópico estabilizar por 3s
      │  flows: 7141 → 7538 (estável)
T6:   Iniciar Detector (todos os flows disponíveis)
T7:   Detector processa 7538 flows
T8:   Detector idle por 10s → encerra normalmente
```

### 2.3 Diagrama Temporal

```
         ┌─ Injeção PCAP ──┐
         │   100k packets   │
Packets  ████████████████████
         0s               42s

         ┌─ init ─┐
FlowCons ──────────████████████████████──SIGTERM──flush
         42s      45s                 ~63s       ~63s

                  Fase 1: monitor     F2   F3
Sync     ─────────░░░░░░░░░░░░░░░░░░░▓▓▓▓▓░░░
                  45s              ~63s     ~68s

                                              ┌─ Detecção ──┐
Detector ─────────────────────────────────────████████████████
                                              ~68s         ~120s
```

---

## 3. Implementação

### 3.1 Função `wait_for_flow_consumer()`

**Localização:** `src/kafka_utils.py`

Monitora o end offset do tópico `flows`. Quando o offset para de crescer
por `stable_seconds`, considera o FlowConsumer "estável" (sem mais flows
sendo produzidos por timeout de evento).

```python
def wait_for_flow_consumer(
    bootstrap_servers: str = "localhost:9092",
    flows_topic: str = "flows",
    stable_seconds: float = 5.0,    # Tempo de estabilidade
    poll_interval: float = 1.0,      # Intervalo entre checks
    timeout_seconds: float = 300.0,  # Timeout máximo
) -> int:
    """Retorna número final de flows no tópico."""
```

**Dependência auxiliar:**
```python
def get_topic_end_offset(topic: str, bootstrap_servers: str) -> int:
    """Retorna soma dos end offsets de todas as partições."""
```

### 3.2 Modificação no `run_experiment.py`

Entre as ETAPAs 2 (FlowConsumer) e 3 (Detector), adicionada ETAPA 2.5:

```python
# ETAPA 2.5: Sincronização

# Fase 1: Esperar FlowConsumer processar (timeout de evento)
wait_for_flow_consumer(stable_seconds=5.0, timeout_seconds=600)

# Fase 2: Terminar FlowConsumer → flush dos flows ativos
flow_consumer_process.terminate()
flow_consumer_process.wait(timeout=15)

# Fase 3: Esperar flows do flush aparecerem
time.sleep(2)
total_flows = wait_for_flow_consumer(stable_seconds=3.0, timeout_seconds=30)
```

### 3.3 Arquivos Modificados

| Arquivo | Mudança |
|---------|---------|
| `src/kafka_utils.py` | Adicionadas `get_topic_end_offset()` e `wait_for_flow_consumer()` |
| `scripts/run_experiment.py` | ETAPA 2.5 de sincronização + import |

---

## 4. Resultados Após Correção

| Métrica | Antes | Depois |
|---------|-------|--------|
| Packets injetados | 100.000 | 100.000 |
| Flows por timeout de evento | ~500 | ~7.141 |
| Flows por flush (shutdown) | 0 | ~397 |
| **Total flows processados** | **~500** | **~7.538** |
| Flows perdidos | ~93% | 0% |
| Tempo total do experimento | ~95s | ~128s |

**Overhead da sincronização:** ~33s (principalmente espera de estabilização).
Aceitável para garantir integridade dos resultados.

---

## 5. Trade-offs

| Decisão | Alternativa | Justificativa |
|---------|-------------|---------------|
| Monitorar end offset do tópico | Monitorar committed offset do consumer group | End offset não depende de commit do FlowConsumer |
| Terminar FlowConsumer via SIGTERM | Adicionar idle timeout ao FlowConsumer | SIGTERM é simples e garante flush; idle timeout adicionaria complexidade ao componente |
| `stable_seconds=5.0` | Valor menor (2s) ou maior (10s) | 5s é bom equilíbrio: rápido o suficiente, mas evita falsos positivos durante pausas curtas |
| Duas fases de wait | Uma única fase com timeout longo | Duas fases separam "timeout de evento" de "flush no shutdown" — mais previsível |

---

## 6. Changelog

| Data | Versão | Mudança |
|------|--------|---------|
| 2026-03-10 | 1.0 | Documento inicial — sincronização FlowConsumer-Detector |
