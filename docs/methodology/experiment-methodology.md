# Metodologia de Experimentos - IoT IDS com Clustering Evolutivo

**Criado:** 2026-01-29
**Última Atualização:** 2026-02-19
**Capítulo da Dissertação:** 4 - Metodologia

---

## 1. Introdução

Este documento define a metodologia para conduzir experimentos válidos e reprodutíveis com o sistema de detecção de intrusão baseado em MicroTEDAclus, utilizando os arquivos PCAP do dataset CICIoT2023.

### 1.1 Objetivos dos Experimentos

1. **Validar detecção de anomalias:** O sistema detecta ataques conhecidos?
2. **Avaliar resistência a concept drift:** O sistema se adapta a novos padrões?
3. **Medir generalização:** O sistema detecta ataques não vistos durante treinamento?
4. **Comparar com baseline:** Como se compara aos algoritmos da Fase 1?

### 1.2 Fundamentação Acadêmica

A metodologia é baseada em práticas estabelecidas na literatura:

- [Data-Driven Evaluation of Intrusion Detectors: A Methodological Framework](https://link.springer.com/chapter/10.1007/978-3-031-30122-3_9) (Ayoubi et al., 2023)
- [On Evaluating Stream Learning Algorithms](https://link.springer.com/article/10.1007/s10994-012-5320-9) (Gama et al., 2013)
- [Evolving Cybersecurity Frontiers: Concept Drift and Feature Dynamics in IDS](https://www.sciencedirect.com/science/article/pii/S0952197624013010) (2024)
- [CICIoT2023: A Real-Time Dataset and Benchmark](https://www.mdpi.com/1424-8220/23/13/5941) (Neto et al., 2023)

---

## 2. Dataset CICIoT2023

### 2.1 Estrutura do Dataset

| Componente | Descrição | Tamanho |
|------------|-----------|---------|
| **PCAP/** | Tráfego original com timestamps | ~548 GB |
| **CSV/** | Features extraídas (shuffled) | ~13 GB |

**Importante:** Os CSVs são shuffled e perdem ordem temporal. Para experimentos de streaming e concept drift, **usar PCAPs é obrigatório**.

### 2.2 Categorias de Tráfego

| Categoria | Nº Ataques | % Dataset | Característica |
|-----------|------------|-----------|----------------|
| **Benign** | - | 2.4% | Tráfego normal IoT |
| **DDoS** | 12 | 73% | Alta volumetria |
| **DoS** | 5 | 15% | Negação de serviço |
| **Mirai** | 3 | 5.6% | Botnet IoT |
| **Recon** | 5 | 3.3% | Reconhecimento |
| **Spoofing** | 2 | 0.6% | Falsificação |
| **Web-Based** | 6 | 0.05% | Ataques web |
| **Brute Force** | 1 | 0.03% | Força bruta |

### 2.3 Desafios Metodológicos

Baseado em [Survey of IDS: Techniques, Datasets and Challenges](https://link.springer.com/article/10.1186/s42400-019-0038-7):

| Desafio | Impacto | Mitigação |
|---------|---------|-----------|
| **Desbalanceamento** | Benign é apenas 2.4% | Usar métricas balanceadas (F1, AUC) |
| **Ordem temporal** | CSVs são shuffled | Processar PCAPs diretamente |
| **Concept drift** | Ataques evoluem | Avaliação prequencial |
| **Generalização** | Overfitting em ataques conhecidos | Separar ataques para teste |

---

## 3. Abordagens de Avaliação

### 3.1 Avaliação Tradicional (Batch)

Usada para comparação com Fase 1 e baselines.

```
┌─────────────────────────────────────────────────────────────┐
│                    SPLIT ESTRATIFICADO                       │
│                                                              │
│   Dataset ──────┬──────► Train (70-80%)                      │
│                 │                                            │
│                 └──────► Test (20-30%)                       │
│                                                              │
│   Validação: k-fold cross-validation (k=5 ou k=10)          │
└─────────────────────────────────────────────────────────────┘
```

**Quando usar:** Comparação com algoritmos batch da Fase 1.

### 3.2 Avaliação Prequential (Streaming)

**Recomendada para MicroTEDAclus** - baseada em [On Evaluating Stream Learning Algorithms](https://link.springer.com/article/10.1007/s10994-012-5320-9).

```
┌─────────────────────────────────────────────────────────────┐
│              PREQUENTIAL (Test-Then-Train)                   │
│                                                              │
│   Para cada amostra x:                                       │
│     1. Prever: ŷ = modelo.predict(x)                        │
│     2. Avaliar: comparar ŷ com y real                        │
│     3. Atualizar: modelo.update(x, y)                        │
│                                                              │
│   Métricas calculadas com sliding window ou fading factor   │
└─────────────────────────────────────────────────────────────┘
```

**Vantagens:**
- Usa todas as amostras para treino E teste
- Simula cenário real de streaming
- Detecta degradação por concept drift

**Implementação sugerida:**
```python
# Pseudocódigo para avaliação prequential
window_size = 1000  # Sliding window para métricas

for flow in stream:
    # 1. Test
    prediction = detector.predict(flow.features)
    metrics.update(prediction, flow.label)

    # 2. Train
    detector.process(flow.features)

    # 3. Report (a cada window_size)
    if metrics.count % window_size == 0:
        report_metrics(metrics.get_windowed())
```

### 3.3 Avaliação de Generalização (Zero-Day)

Baseada em [usfAD Based Unknown Attack Detection](https://arxiv.org/html/2403.11180v1):

```
┌─────────────────────────────────────────────────────────────┐
│           AVALIAÇÃO DE ATAQUES DESCONHECIDOS                 │
│                                                              │
│   1. Selecionar ataques para "holdout" (ex: Mirai, Recon)   │
│   2. Treinar APENAS com outros ataques + Benign             │
│   3. Testar com ataques holdout                              │
│                                                              │
│   Exemplo:                                                   │
│   - Train: Benign + DDoS + DoS                              │
│   - Test:  Mirai + Recon (nunca vistos)                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Design de Experimentos com PCAPs

### 4.1 Estrutura Temporal dos PCAPs

O CICIoT2023 organiza PCAPs por cenário de ataque. Para simular streaming realista:

```
Tempo ────────────────────────────────────────────────────►

│ Warm-up │    Treino     │        Teste           │
│ (Benign)│ (Benign+Atk1) │ (Benign+Atk1+Atk2)    │
│         │               │                        │
│  t0     │      t1       │         t2             │
```

### 4.2 Cenários de Experimento

#### Cenário A: Detecção Básica
**Objetivo:** Validar que o sistema detecta ataques conhecidos.

| Fase | PCAPs | Duração | Propósito |
|------|-------|---------|-----------|
| Warm-up | Benign | 5-10 min | Estabelecer baseline normal |
| Teste | Benign + DDoS | 30 min | Detectar ataques em tempo real |

**Métricas:** Precision, Recall, F1, Tempo de detecção

#### Cenário B: Concept Drift Súbito
**Objetivo:** Avaliar adaptação a mudança abrupta de padrão.

| Fase | PCAPs | Evento |
|------|-------|--------|
| 1 | Benign + DDoS | Aprende padrão DDoS |
| 2 | Benign + Mirai | **Drift súbito** - novo tipo de ataque |

**Métricas:** Tempo de adaptação, F1 antes/depois do drift

#### Cenário C: Concept Drift Gradual
**Objetivo:** Avaliar adaptação a mudança progressiva.

| Fase | Composição |
|------|------------|
| 1 | 100% DDoS |
| 2 | 75% DDoS + 25% Mirai |
| 3 | 50% DDoS + 50% Mirai |
| 4 | 25% DDoS + 75% Mirai |
| 5 | 100% Mirai |

**Métricas:** Curva de adaptação, F1 por fase

#### Cenário D: Concept Drift Recorrente
**Objetivo:** Avaliar se o sistema "lembra" padrões anteriores.

| Fase | PCAPs |
|------|-------|
| 1 | DDoS (aprende) |
| 2 | Benign (período calmo) |
| 3 | DDoS (deve reconhecer) |
| 4 | Benign |
| 5 | DDoS (deve reconhecer rapidamente) |

**Métricas:** Tempo de reconhecimento em cada retorno

#### Cenário E: Generalização (Zero-Day)
**Objetivo:** Detectar ataques nunca vistos.

| Fase | PCAPs | Propósito |
|------|-------|-----------|
| Train | Benign + DDoS + DoS | Aprender padrões |
| Test | Mirai + Recon + Spoofing | Ataques nunca vistos |

**Métricas:** Taxa de detecção de zero-day, False Positive Rate

### 4.3 Combinação de PCAPs para Streaming

Para simular tráfego realista, os PCAPs devem ser combinados respeitando timestamps:

```python
# Pseudocódigo para merge de PCAPs
def merge_pcaps_by_timestamp(pcap_list):
    """
    Combina múltiplos PCAPs ordenando por timestamp.
    Simula tráfego misto (benign + attacks) em tempo real.
    """
    streams = [read_pcap(p) for p in pcap_list]

    # Merge sort por timestamp
    while any_stream_has_packets(streams):
        # Pega pacote com menor timestamp
        next_packet = min(streams, key=lambda s: s.peek().timestamp)
        yield next_packet.pop()
```

**Abordagem recomendada:** Usar ferramenta `mergecap` do Wireshark:
```bash
mergecap -w combined.pcap benign.pcap ddos.pcap mirai.pcap
```

---

## 5. Métricas de Avaliação

### 5.1 Métricas de Detecção

| Métrica | Fórmula | Interpretação |
|---------|---------|---------------|
| **Precision** | TP / (TP + FP) | Qualidade dos alertas |
| **Recall** | TP / (TP + FN) | Cobertura de ataques |
| **F1-Score** | 2 × (P × R) / (P + R) | Balanço P/R |
| **AUC-ROC** | Área sob curva ROC | Performance geral |

### 5.2 Métricas de Streaming/Drift

| Métrica | Descrição | Uso |
|---------|-----------|-----|
| **Prequential F1** | F1 calculado em sliding window | Evolução temporal |
| **Tempo de Detecção** | Amostras até primeira detecção | Latência |
| **Tempo de Adaptação** | Amostras até F1 recuperar após drift | Resiliência |
| **Taxa de Novos Clusters** | Clusters criados / amostra | Sensibilidade |

### 5.3 Métricas de Clustering (MicroTEDAclus)

| Métrica | Descrição |
|---------|-----------|
| **Número de Clusters** | Total de micro-clusters ativos |
| **Variância Média** | Compacidade dos clusters |
| **Cluster Purity** | Homogeneidade de labels por cluster |

### 5.4 Comparação com Baseline (Fase 1)

Para comparar MicroTEDAclus com algoritmos batch da Fase 1:

| Aspecto | Fase 1 (Batch) | Fase 2 (Streaming) |
|---------|----------------|-------------------|
| **Avaliação** | k-fold CV | Prequential |
| **Métricas** | F1, Accuracy | Prequential F1, Tempo adaptação |
| **Dataset** | CSV (shuffled) | PCAP (temporal) |

**Nota:** Comparação direta é limitada devido a metodologias diferentes. Usar mesmo subset de dados quando possível.

---

## 6. Protocolo Experimental

### 6.1 Preparação

1. **Selecionar PCAPs** para cada cenário
2. **Definir parâmetros** do MicroTEDAclus (r0, min_samples)
3. **Configurar métricas** e logging
4. **Definir seeds** para reprodutibilidade

### 6.2 Execução

```bash
# Exemplo de execução
./run_experiment.sh \
    --scenario concept_drift_sudden \
    --pcaps benign.pcap,ddos.pcap,mirai.pcap \
    --algorithm micro_teda \
    --r0 0.1 \
    --output results/exp001/
```

### 6.3 Coleta de Dados

Para cada experimento, coletar:

| Dado | Arquivo | Frequência |
|------|---------|------------|
| Métricas por janela | `metrics_windowed.csv` | A cada 1000 amostras |
| Estado dos clusters | `clusters_state.json` | A cada 5000 amostras |
| Alertas gerados | `alerts.json` | Cada alerta |
| Logs de execução | `experiment.log` | Contínuo |

### 6.4 Análise

1. **Gráficos temporais:** F1, Precision, Recall ao longo do tempo
2. **Análise de drift:** Momento e impacto de mudanças
3. **Comparação:** Tabelas comparativas entre cenários/algoritmos
4. **Validação estatística:** Intervalos de confiança, testes de significância

---

## 7. Validação Estatística

### 7.1 Repetições

- **Mínimo:** 5 execuções por configuração
- **Recomendado:** 10 execuções para intervalos de confiança estreitos

### 7.2 Análise de Variância

```python
# Reportar média ± desvio padrão
# Exemplo: F1 = 0.95 ± 0.02

from scipy import stats

# Teste de significância entre algoritmos
t_stat, p_value = stats.ttest_ind(results_algo1, results_algo2)
```

### 7.3 Reprodutibilidade

Documentar para cada experimento:
- Versão do código (git commit)
- Seeds utilizados
- Parâmetros completos
- Hardware utilizado
- Tempo de execução

---

## 8. Plano Tático Detalhado (Semanas 5-9)

Este capítulo detalha COMO executar cada semana da Fase 2B, incluindo decisões concretas sobre PCAPs, parâmetros, scripts, e critérios de sucesso.

---

### 8.1 Semana 5: Orquestração + E2E + Benchmark

**Objetivo:** Automatizar execução de experimentos e validar MicroTEDAclus end-to-end.

**Definição de "pronto" para Semana 5:**
Rodar com 1 comando:
- Stream curto e controlado (benign warm-up + ataque)
- Comparar TEDA vs MicroTEDAclus no mesmo stream
- Gerar outputs replicáveis (métricas, alertas, clusters, throughput/memória)
- Sanity check de sucesso (detectar DDoS com baixa latência, FPR aceitável no benign)

Isso é o **MVP do Cenário A (Detecção básica)**.

---

#### 8.1.1 PCAPs Selecionados (Decisão Concreta)

**Escolha para primeiro E2E:**
- **Benign:** `BenignTraffic*.pcap` (tráfego normal do dataset)
- **Ataque:** `DDoS-ICMP_Flood*.pcap` (ataque volumétrico com separação clara)

**Justificativa:**
- DDoS e Benign são classes com muitos exemplos e separáveis
- Paper CICIoT2023 mostra que DDoS é muito detectável
- Começar pelo "ataque mais detectável" valida infraestrutura
- Classes Web/Bruteforce são mais confusas (deixar para depois)

**Duração / Quantidade (Decisão Concreta):**

| Fase | Limite | Justificativa |
|------|--------|---------------|
| **Warm-up benign** | 5.000 flows | Estabelecer baseline sem sobrecarregar |
| **Teste ataque** | 10.000 flows | Suficiente para MTTD + estatísticas |

**Por que em flows, não minutos?**
- Em replay acelerado, minutos são arbitrários
- Em IDS streaming, o que importa é número de decisões (flows) e MTTD em flows

**Paths concretos (exemplo):**
```bash
BENIGN_PCAP=data/pcaps/benign/BenignTraffic*.pcap
ATTACK_PCAP=data/pcaps/ddos/DDoS-ICMP_Flood*.pcap
MAX_FLOWS_WARMUP=5000
MAX_FLOWS_TEST=10000
```

---

#### 8.1.2 Parâmetros Concretos do MicroTEDAclus

**Default (para rodar agora):**
```yaml
algorithm: micro_teda        # Default no sistema
r0: 0.1                      # Variância mínima
min_samples: 10              # Amostras antes de detectar
window_size: 1000            # Sliding window para métricas prequential
```

**Micro-grid Mínimo (Semana 5 - 6 execuções):**

Rodar 3 configs × 2 algoritmos:

**Configs MicroTEDAclus:**
1. `r0=0.05, min_samples=10` (mais sensível)
2. `r0=0.10, min_samples=10` (default)
3. `r0=0.20, min_samples=10` (mais permissivo)

**Algoritmos:**
- TEDA (single-center) para comparação/debug
- MicroTEDAclus (produção)

**Por que esse grid?**
- `r0` controla sensibilidade inicial: maior = mais permissivo (menos alertas)
- Manter `min_samples` fixo evita explodir combinações
- Já produz trade-off FP vs TP

**Saída esperada:** Gráficos de F1/Recall vs r0, FPR vs r0, MTTD vs r0

---

#### 8.1.3 Script de Orquestração (Estrutura Pronta)

**Decisão:** Python como driver (bash só como wrapper)

**Motivo:** Compor cenários, logar metadata, medir throughput/memória, salvar JSON/CSV

**Arquivo:** `streaming/scripts/run_experiment.py`

**Interface CLI:**
```python
# Entradas
--scenario: A_basic                    # Semana 5 só isso
--benign_pcap: path                    # Path do PCAP benign
--attack_pcap: path                    # Path do PCAP ataque
--attack_start: after_warmup           # Padrão
--max_flows_warmup: 5000               # Flows de warm-up
--max_flows_test: 10000                # Flows de teste
--replay_speed: 10                     # 10x acelerado
--algorithm: teda|micro_teda           # Algoritmo
--r0: float                            # MicroTEDAclus: variância mínima
--min_samples: int                     # Amostras antes de detectar
--window_size: 1000                    # Sliding window métricas
--seed: int                            # Reprodutibilidade
--output_dir: path                     # Diretório de saída
```

**Fluxo de Execução:**
1. Subir Kafka + consumers (se ainda não rodando)
2. Processar benign PCAP até warm-up completar (5000 flows)
3. Processar attack PCAP (ou mix benign+attack se tiver mixer)
4. Coletar do tópico `alerts` e `flows` (opcional) e salvar
5. Escrever métricas por janela
6. Parar componentes gracefully

**Artefatos Obrigatórios (5 arquivos):**

| Arquivo | Conteúdo | Formato |
|---------|----------|---------|
| `run_meta.json` | Git commit, parâmetros, paths PCAP, tempos, volumes | JSON |
| `alerts.jsonl` | Um alerta por linha (AlertSchema do CURRENT.md) | JSON Lines |
| `metrics_windowed.csv` | Por janela: TP/FP/FN/TN, P/R/F1, FPR, MTTD, #clusters | CSV |
| `clusters_state.jsonl` | Snapshot a cada 5000 flows: #clusters, sizes, variâncias | JSON Lines |
| `system_usage.csv` | A cada 1s: RSS memory (MB), CPU%, flows/s | CSV |

**Exemplo de `run_meta.json`:**
```json
{
  "git_commit": "abc1234",
  "scenario": "A_basic",
  "algorithm": "micro_teda",
  "params": {"r0": 0.1, "min_samples": 10, "window_size": 1000},
  "pcaps": {
    "benign": "data/pcaps/benign/BenignTraffic.pcap",
    "attack": "data/pcaps/ddos/DDoS-ICMP_Flood.pcap"
  },
  "execution": {
    "start_time": "2026-02-20T10:00:00Z",
    "end_time": "2026-02-20T10:15:00Z",
    "duration_seconds": 900
  },
  "volumes": {
    "total_packets": 125000,
    "total_flows": 15000,
    "warmup_flows": 5000,
    "test_flows": 10000
  }
}
```

---

#### 8.1.4 Métricas Específicas (Mínimo para E2E)

**8.1.4.1 Detecção (mínimo):**

| Métrica | Descrição | Uso |
|---------|-----------|-----|
| **Recall (attack)** | Quantos flows de ataque viraram alertas | TP / (TP + FN) |
| **FPR (benign)** | Quantos flows benign viraram alertas | FP / (FP + TN) |
| **F1 (binário)** | Balanço Precision/Recall | 2PR / (P+R) |
| **AUC-ROC** | Se tiver score (eccentricity/typicality) | Área sob curva |

**8.1.4.2 Streaming (mínimo):**

| Métrica | Descrição | Como Calcular |
|---------|-----------|---------------|
| **MTTD (flows)** | Flows desde 1º flow ataque até 1º alerta | `first_alert_idx - first_attack_idx` |
| **Throughput (flows/s)** | Vazão de processamento | `total_flows / elapsed_time` |
| **Memória RSS (MB)** | Pico de uso de RAM | `psutil.Process().memory_info().rss / 1e6` |

**MTTD refinado:** Usar "k alertas em sequência" (ex.: 3) para reduzir ruído de FP isolados.

**8.1.4.3 Clustering (mínimo):**

| Métrica | Descrição |
|---------|-----------|
| **#microclusters** | Total de clusters ativos ao longo do tempo |
| **Taxa de criação** | Clusters criados por 1000 flows |
| **Variância média** | Compacidade dos clusters |

**Nota:** Cluster purity fica para Semana 6 (precisa mapear label por cluster).

---

#### 8.1.5 Critérios de Sucesso (Objetivos para Pass/Fail)

**Sucesso Mínimo (pass/fail):**

✅ **Pipeline:**
- [ ] Completa o stream sem crashes
- [ ] Gera os 5 artefatos obrigatórios

✅ **Durante warm-up benign:**
- [ ] FPR <= 5% (não é alarme maluco)

✅ **Durante ataque DDoS:**
- [ ] Recall >= 80% (detecta maioria)
- [ ] MTTD <= 500 flows (latência razoável)

**Sucesso "Bom":**
- [ ] FPR <= 1% (poucos falsos positivos)
- [ ] Recall >= 95% (detecta quase todos)
- [ ] MTTD <= 100 flows (rápido)

**Justificativa:** DDoS costuma ser evidente, então alvos agressivos são razoáveis para validar.

---

#### 8.1.6 Como Medir Throughput e Memória (Procedimento Exato)

**No driver Python (`run_experiment.py`):**

```python
import time
import psutil

process = psutil.Process()
start_time = time.time()
flows_processed = 0

# Loop principal
while processing:
    # A cada 1 segundo
    if time.time() - last_measure >= 1.0:
        elapsed = time.time() - start_time
        flows_per_sec = (flows_processed - prev_flows) / 1.0

        # Métricas de sistema
        rss_mb = process.memory_info().rss / 1e6
        cpu_percent = process.cpu_percent()

        # Gravar em system_usage.csv
        writer.writerow({
            'timestamp': time.time(),
            'elapsed_sec': elapsed,
            'flows_processed': flows_processed,
            'flows_per_sec': flows_per_sec,
            'rss_mb': rss_mb,
            'cpu_percent': cpu_percent
        })

        prev_flows = flows_processed
        last_measure = time.time()
```

**Por que isso importa:**
- Arquitetura Kafka-based tem custo de serialization + network + consumers
- Vira argumento de "viabilidade online" (central para streaming IDS)

---

#### 8.1.7 Checklist Executável (Procedimento Passo a Passo)

**Pré-requisitos:**
- [ ] Kafka up (`docker-compose up -d`)
- [ ] Dataset montado localmente (paths válidos)
- [ ] Tópicos criados: `packets`, `flows`, `alerts`

**Passos:**

**1. Sanity check (2 minutos):**
```bash
# Rodar PCAP benign curto sem ataque
python scripts/run_experiment.py \
    --benign_pcap data/pcaps/benign/BenignTraffic.pcap \
    --attack_pcap none \
    --max_flows_warmup 2000 \
    --output results/week5/sanity/
```

Verificar:
- [ ] `flows` tem eventos
- [ ] `alerts` tem poucos ou nenhum alerta (< 5%)

**2. Cenário A completo (warm-up + ataque):**
```bash
# 6 execuções: 3 configs × 2 algoritmos
for algo in teda micro_teda; do
    for r0 in 0.05 0.10 0.20; do
        python scripts/run_experiment.py \
            --scenario A_basic \
            --benign_pcap data/pcaps/benign/BenignTraffic.pcap \
            --attack_pcap data/pcaps/ddos/DDoS-ICMP_Flood.pcap \
            --max_flows_warmup 5000 \
            --max_flows_test 10000 \
            --algorithm $algo \
            --r0 $r0 \
            --min_samples 10 \
            --output results/week5/scenarioA/${algo}_r0${r0}/
    done
done
```

**3. Análise:**
```bash
# Consolidar resultados
python scripts/compare_experiments.py \
    --input results/week5/scenarioA/ \
    --output results/week5/scenarioA/comparison_report.md
```

**Saídas esperadas:**
- [ ] Diretório `results/week5/scenarioA/<run_id>/` contém os 5 artefatos
- [ ] Gráfico simples (notebook): F1 por janela, #clusters por janela, MTTD por execução

**Entregáveis Finais S5:**
- [ ] `scripts/run_experiment.py` - Script de orquestração funcionando
- [ ] `scripts/compare_experiments.py` - Script de comparação
- [ ] `results/week5/sanity/` - Sanity check OK
- [ ] `results/week5/scenarioA/` - 6 execuções completas
- [ ] `results/week5/scenarioA/comparison_report.md` - Relatório consolidado
- [ ] `docs/weekly-reports/semana5-report.md` - Relatório semanal

---

### 8.2 Semana 6: Sistema de Métricas Prequential

**Objetivo:** Implementar sistema de métricas em tempo real para avaliação streaming.

#### 8.2.1 Componente: MetricsCollector

**Arquivo:** `streaming/src/metrics/prequential_metrics.py`

**Funcionalidade:**
```python
class PrequentialMetrics:
    """
    Calcula métricas em sliding window para avaliação prequential.
    Implementa test-then-train: avalia ANTES de atualizar o modelo.
    """

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)
        self.ground_truth = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)

    def update(self, y_pred: bool, y_true: bool, timestamp: float):
        """Adiciona predição e ground truth."""
        self.predictions.append(y_pred)
        self.ground_truth.append(y_true)
        self.timestamps.append(timestamp)

    def get_current_metrics(self) -> Dict[str, float]:
        """Calcula métricas na janela atual."""
        return {
            'precision': precision_score(...),
            'recall': recall_score(...),
            'f1': f1_score(...),
            'window_size': len(self.predictions)
        }
```

#### 8.2.2 Ground Truth: Como Obter?

**Problema:** PCAPs não têm labels por flow individual.

**Solução 1: Labels Heurísticos (S6)**
```python
def infer_label_from_filename(pcap_path: str) -> str:
    """
    Infere label do nome do arquivo.
    Ex: 'DDoS-ICMP_Flood.pcap' -> label='DDoS'
    """
    if 'benign' in pcap_path.lower():
        return 'benign'
    elif 'ddos' in pcap_path.lower():
        return 'ddos'
    # ... outros ataques

def label_flow(flow: Dict, pcap_label: str) -> bool:
    """
    Todos os flows do PCAP recebem o mesmo label.
    Simplificação para S6-S7.
    """
    return pcap_label != 'benign'
```

**Solução 2: CSV do CICIoT2023 (S8+)**
- Processar PCAPs + carregar CSV correspondente
- Fazer matching flow-by-flow (5-tuple + timestamp)
- Mais preciso mas mais complexo

**Decisão S6:** Usar Solução 1 (heurística por arquivo)

#### 8.2.3 Métricas de Adaptação a Drift

**Tempo de Detecção do Drift:**
```python
def calculate_drift_detection_time(metrics_timeline: List[Dict]) -> int:
    """
    Detecta quando F1 cai abruptamente (concept drift).
    Retorna número de amostras até F1 recuperar.
    """
    baseline_f1 = np.mean([m['f1'] for m in metrics_timeline[:500]])

    drift_detected = False
    drift_start_idx = None

    for i, m in enumerate(metrics_timeline):
        if not drift_detected and m['f1'] < baseline_f1 * 0.8:
            # F1 caiu 20% - drift detectado
            drift_detected = True
            drift_start_idx = i

        if drift_detected and m['f1'] >= baseline_f1 * 0.95:
            # F1 recuperou 95% do baseline - adaptação completa
            return i - drift_start_idx

    return None  # Não recuperou
```

**Clustering Metrics:**
```python
def calculate_cluster_purity(clusters: List[MicroCluster],
                             flow_labels: List[bool]) -> float:
    """
    Purity = média de (flows majoritários / total flows) por cluster.
    Mede homogeneidade dos clusters.
    """
    purities = []
    for cluster in clusters:
        labels = [flow_labels[i] for i in cluster.flow_indices]
        majority_label = Counter(labels).most_common(1)[0][1]
        purity = majority_label / len(labels)
        purities.append(purity)
    return np.mean(purities)
```

#### 8.2.4 Sistema de Logging Estruturado

**Formato:** JSON Lines (`.jsonl`) - um JSON por linha

**Arquivo de Métricas:** `metrics_timeline.jsonl`
```jsonl
{"sample": 1000, "f1": 0.95, "precision": 0.94, "recall": 0.96, "num_clusters": 3, "timestamp": 1706558400.123}
{"sample": 2000, "f1": 0.93, "precision": 0.92, "recall": 0.94, "num_clusters": 4, "timestamp": 1706558410.456}
```

**Vantagens:**
- Processável por linha (streaming)
- Fácil de plotar com pandas: `pd.read_json('file.jsonl', lines=True)`
- Append-only (não corrompe arquivo inteiro)

**Integração com StreamingDetector:**
```python
# Adicionar ao StreamingDetector
self.metrics = PrequentialMetrics(window_size=1000)
self.metrics_file = open(output_dir / 'metrics_timeline.jsonl', 'a')

# No loop principal
result = self.detector.predict(features)
y_true = self._get_ground_truth(flow)  # Heurística
self.metrics.update(result.is_anomaly, y_true, time.time())

if self.sample_count % 100 == 0:
    current_metrics = self.metrics.get_current_metrics()
    current_metrics['sample'] = self.sample_count
    self.metrics_file.write(json.dumps(current_metrics) + '\n')
    self.metrics_file.flush()
```

**Entregáveis S6:**
- [ ] `src/metrics/prequential_metrics.py` - Métricas prequential
- [ ] `src/metrics/drift_metrics.py` - Métricas de adaptação
- [ ] `src/metrics/clustering_metrics.py` - Purity, Silhouette
- [ ] Atualizar `StreamingDetector` para coletar métricas em tempo real
- [ ] Testes unitários para métricas
- [ ] Script `scripts/plot_metrics.py` para visualizar timelines

---

### 8.3 Semana 7: Cenário A - Detecção Básica

**Objetivo:** Executar primeiro experimento científico completo (Cenário A).

#### 8.3.1 Configuração do Cenário A

**Fases do Experimento:**

| Fase | PCAP | Duração | Propósito |
|------|------|---------|-----------|
| **Warm-up** | `benign_traffic.pcap` | 5 min | Estabelecer baseline normal |
| **Teste** | `benign_traffic.pcap` + `DDoS-ICMP_Flood.pcap` | 30 min | Detectar ataques em tempo real |

**PCAPs Específicos:**
```bash
# Fase 1: Warm-up (apenas benign)
data/pcaps/benign/IoT_benign_traffic.pcap  # ~5 minutos

# Fase 2: Teste (benign + ataque misturados)
# Usar mergecap para combinar:
mergecap -w scenario_a_test.pcap \
    data/pcaps/benign/IoT_benign_traffic.pcap \
    data/pcaps/ddos/DDoS-ICMP_Flood.pcap
```

**Parâmetros do Experimento:**
```yaml
algorithm: micro_teda
r0: 0.1
min_samples: 50  # Warm-up de 50 flows antes de detectar
max_flows: null  # Processar PCAP completo
ground_truth: heuristic  # Usar heurística por arquivo
```

#### 8.3.2 Repetições e Validação Estatística

**Número de Repetições:** 5 (mínimo para intervalos de confiança)

**Variação entre repetições:**
- Mesmos PCAPs, mesmos parâmetros
- Apenas ordem de processamento pode variar (Kafka partições)

**Script de Execução:**
```bash
# scripts/run_scenario_a.sh
for run in {1..5}; do
    python scripts/run_experiment.py \
        --pcap data/pcaps/scenarios/scenario_a_test.pcap \
        --algorithm micro_teda \
        --output results/scenario_a/run_$run/ \
        --r0 0.1 \
        --min-samples 50
done

# Consolidar resultados
python scripts/consolidate_runs.py \
    --input results/scenario_a/ \
    --output results/scenario_a/consolidated_report.md
```

#### 8.3.3 Critérios de Sucesso

**Métricas Alvo:**

| Métrica | Alvo | Baseline (Fase 1) | Critério de Aceitação |
|---------|------|-------------------|----------------------|
| **F1-Score** | > 0.90 | 0.99 | Aceitável se > 0.85 |
| **Precision** | > 0.90 | 0.99 | Minimizar falsos positivos |
| **Recall** | > 0.85 | 0.99 | Detectar maioria dos ataques |
| **Tempo de Detecção** | < 100 flows | N/A | Latência aceitável |
| **Throughput** | > 500 flows/s | N/A | Performance adequada |

**Validação:**
```python
def validate_scenario_a(results: List[ExperimentResult]) -> bool:
    """Valida se Cenário A passou nos critérios."""
    mean_f1 = np.mean([r.metrics['f1'] for r in results])
    std_f1 = np.std([r.metrics['f1'] for r in results])

    checks = {
        'f1_mean': mean_f1 > 0.85,
        'f1_stability': std_f1 < 0.05,  # Baixa variância entre runs
        'throughput': all(r.throughput > 500 for r in results),
        'no_crashes': all(r.status == 'success' for r in results)
    }

    return all(checks.values()), checks
```

#### 8.3.4 Análise de Resultados

**Gráficos a Gerar:**
1. **Timeline de Métricas:** F1, Precision, Recall ao longo do tempo
2. **Distribuição de Alertas:** Histograma de eccentricities
3. **Clusters Evolution:** Número de clusters ao longo do tempo
4. **Throughput:** Flows/s processados

**Análise Estatística:**
- Média ± desvio padrão para cada métrica
- Intervalos de confiança (95%)
- Comparação com Fase 1 (tabela lado a lado)

**Entregáveis S7:**
- [ ] PCAP combinado `scenario_a_test.pcap`
- [ ] Script `scripts/run_scenario_a.sh`
- [ ] Resultados de 5 repetições em `results/scenario_a/`
- [ ] Relatório consolidado com gráficos
- [ ] Validação estatística aprovada

---

### 8.4 Semana 8: Cenários B e C - Concept Drift

**Objetivo:** Avaliar adaptação do MicroTEDAclus a concept drift súbito e gradual.

#### 8.4.1 Cenário B: Concept Drift Súbito

**Configuração:**

| Fase | PCAP | Duração | Tipo de Tráfego |
|------|------|---------|-----------------|
| 1 | `benign.pcap` + `DDoS-ICMP.pcap` | 15 min | 50% benign, 50% DDoS |
| 2 | `benign.pcap` + `Mirai-*.pcap` | 15 min | 50% benign, 50% Mirai (DRIFT SÚBITO) |

**Como Criar PCAP de Drift Súbito:**
```bash
# Fase 1: DDoS period
mergecap -w phase1.pcap \
    benign_1.pcap \
    ddos_icmp.pcap

# Fase 2: Mirai period
mergecap -w phase2.pcap \
    benign_2.pcap \
    mirai_attack.pcap

# Concatenar com timestamps ajustados (gap de 1 segundo)
editcap -t 900 phase2.pcap phase2_delayed.pcap  # +900s offset
mergecap -w scenario_b.pcap phase1.pcap phase2_delayed.pcap
```

**Métricas Específicas:**
- **Ponto de Drift:** Sample onde F1 cai > 20%
- **Tempo de Adaptação:** Samples até F1 recuperar 95% do baseline
- **F1 antes vs depois:** Comparar estabilidade

**Análise Visual:**
- Plotar F1 timeline com linha vertical marcando drift
- Número de clusters antes/durante/depois do drift
- Distribuição de typicalities por fase

#### 8.4.2 Cenário C: Concept Drift Gradual

**Configuração (5 fases):**

| Fase | Composição | Duração | Script |
|------|------------|---------|--------|
| 1 | 100% DDoS | 5 min | `ddos_only.pcap` |
| 2 | 75% DDoS + 25% Mirai | 5 min | Mix proporcional |
| 3 | 50% DDoS + 50% Mirai | 5 min | Mix proporcional |
| 4 | 25% DDoS + 75% Mirai | 5 min | Mix proporcional |
| 5 | 100% Mirai | 5 min | `mirai_only.pcap` |

**Como Criar Mix Proporcional:**
```python
# scripts/create_gradual_drift.py
def create_proportional_mix(pcap_a: str, pcap_b: str,
                           ratio_a: float, output: str):
    """
    Cria PCAP com proporção específica de dois PCAPs.

    Exemplo: ratio_a=0.75 -> 75% pacotes de pcap_a, 25% de pcap_b
    Intercala pacotes aleatoriamente respeitando timestamps relativos.
    """
    packets_a = rdpcap(pcap_a)
    packets_b = rdpcap(pcap_b)

    # Amostragem proporcional
    n_a = int(len(packets_a) * ratio_a)
    n_b = int(len(packets_b) * (1 - ratio_a))

    sampled_a = random.sample(packets_a, n_a)
    sampled_b = random.sample(packets_b, n_b)

    # Intercalar mantendo ordem temporal
    mixed = sorted(sampled_a + sampled_b, key=lambda p: p.time)

    wrpcap(output, mixed)

# Gerar fases
create_proportional_mix('ddos.pcap', 'mirai.pcap', 0.75, 'phase2_75_25.pcap')
create_proportional_mix('ddos.pcap', 'mirai.pcap', 0.50, 'phase3_50_50.pcap')
create_proportional_mix('ddos.pcap', 'mirai.pcap', 0.25, 'phase4_25_75.pcap')

# Concatenar tudo
mergecap -w scenario_c.pcap \
    ddos_only.pcap \
    phase2_75_25.pcap \
    phase3_50_50.pcap \
    phase4_25_75.pcap \
    mirai_only.pcap
```

**Métricas Específicas:**
- **Curva de Adaptação:** F1 em cada fase (5 pontos)
- **Degradação:** Máxima queda de F1 durante transição
- **Estabilidade:** Variância de F1 dentro de cada fase

**Análise Visual:**
- Plot de F1 com 5 regiões coloridas (fases)
- Heatmap de typicalities ao longo do tempo
- Evolução de clusters por fase

#### 8.4.3 Merge/Split de Clusters (v0.3)

**Implementação Planejada:**

**Merge de Clusters:**
```python
def should_merge(cluster_i, cluster_j) -> bool:
    """
    Critério de merge: centros muito próximos e variâncias similares.

    Condição: dist(μ_i, μ_j) < 2 * (σ_i + σ_j)
    (mesmo critério do paper Maia 2020)
    """
    dist = np.linalg.norm(cluster_i.mean - cluster_j.mean)
    threshold = 2 * (np.sqrt(cluster_i.variance) + np.sqrt(cluster_j.variance))
    return dist < threshold
```

**Split de Clusters:**
```python
def should_split(cluster) -> bool:
    """
    Critério de split: variância cresceu muito.

    Condição: σ² atual > 5 * σ² histórico
    """
    if cluster.n < 100:
        return False  # Não split clusters jovens

    historical_variance = cluster.variance_history.mean()
    return cluster.variance > 5 * historical_variance
```

**Entregáveis S8:**
- [ ] Script `create_gradual_drift.py` para mixing proporcional
- [ ] PCAPs: `scenario_b.pcap`, `scenario_c.pcap`
- [ ] Implementação merge/split em `micro_teda.py`
- [ ] Testes unitários para merge/split
- [ ] Experimentos Cenários B e C (5 repetições cada)
- [ ] Análise de drift com gráficos

---

### 8.5 Semana 9: Comparação com Algoritmos de Clustering Streaming

**Objetivo:** Comparar MicroTEDAclus com algoritmos de clustering streaming estabelecidos (como no paper Maia et al. 2020).

**Motivação:** Validar que MicroTEDAclus compete com estado-da-arte em clustering evolutivo para IDS, usando mesma metodologia do paper original.

---

#### 8.5.1 Algoritmos para Comparação

**Baseado no paper MicroTEDAclus (Maia et al. 2020):**

| Algoritmo | Tipo | Características | Implementação |
|-----------|------|-----------------|---------------|
| **TEDA** | Single-center | Baseline simples, vulnerável a contaminação | ✅ Já implementado |
| **CluStream** | Micro+Macro | Two-phase, k-medoids, requer k | `river` ou implementar |
| **DenStream** | Density-based | Core/outlier micro-clusters, DBSCAN-like | `river` ou implementar |
| **StreamKM++** | K-means based | Coreset + k-means++, requer k | Implementar |
| **MicroTEDAclus** | TEDA-based | Mixture of typicalities, sem k | ✅ Já implementado |

**Sobre incluir TEDA:**
- **Sim, incluir:** TEDA é streaming (online), serve como baseline "simples"
- Mostra evolução: TEDA → MicroTEDAclus (com multi-cluster)
- Evidencia problema de contaminação em cenários drift

---

#### 8.5.2 Onde Encontrar Implementações

**Bibliotecas de Streaming:**

1. **River** (sucessor do scikit-multiflow):
   - CluStream: `river.cluster.CluStream`
   - DenStream: `river.cluster.DenStream`
   - Documentação: https://riverml.xyz/latest/api/cluster/

2. **StreamKM++:**
   - Não disponível em libs Python populares
   - **Decisão S9:** Implementar versão simplificada OU usar apenas CluStream/DenStream/TEDA/MicroTEDAclus (4 algoritmos)

**Recomendação para S9 (pragmática):**
```python
# Algoritmos para comparação (4 total)
algorithms = [
    'teda',              # Baseline simples (já implementado)
    'clustream',         # Estado-da-arte micro+macro (river)
    'denstream',         # Density-based (river)
    'micro_tedaclus'     # Nossa contribuição (já implementado)
]
```

**StreamKM++:** Deixar para S10+ se houver tempo, ou argumentar que 4 algoritmos são suficientes (paper Maia comparou com 3).

---

#### 8.5.3 Integração com River

**Adapter para uniformizar interface:**

```python
# streaming/src/detector/streaming_adapter.py
class StreamingClusterAdapter:
    """Adapta algoritmos River para interface comum."""

    def __init__(self, algorithm: str, **params):
        if algorithm == 'clustream':
            from river.cluster import CluStream
            self.model = CluStream(
                n_macro_clusters=params.get('k', 5),
                max_micro_clusters=params.get('max_micro', 100),
                time_window=params.get('time_window', 1000)
            )
        elif algorithm == 'denstream':
            from river.cluster import DenStream
            self.model = DenStream(
                decaying_factor=params.get('lambda', 0.01),
                epsilon=params.get('eps', 0.1),
                beta=params.get('beta', 0.5),
                mu=params.get('mu', 2)
            )

    def process(self, x: np.ndarray) -> ClusterResult:
        """Processa amostra e retorna resultado."""
        # Converter numpy para dict (interface River)
        x_dict = {f'f{i}': x[i] for i in range(len(x))}

        # Aprender
        self.model.learn_one(x_dict)

        # Prever cluster
        cluster_id = self.model.predict_one(x_dict)

        # Determinar anomalia (heurística: clusters pequenos/novos)
        is_anomaly = self._is_anomaly(cluster_id)

        return ClusterResult(
            cluster_id=cluster_id,
            is_anomaly=is_anomaly,
            num_clusters=len(self.model.centers)
        )
```

---

#### 8.5.4 Metodologia de Comparação

**Dataset:** Usar Cenário B (concept drift súbito) - mais desafiador que Cenário A.

**Por quê Cenário B?**
- Testa adaptação a drift (principal vantagem de clustering evolutivo)
- DDoS → Mirai representa mudança clara de padrão
- Mesmo cenário usado no paper Maia et al. para comparação

**Configuração:**

| Algoritmo | Parâmetros | Justificativa |
|-----------|------------|---------------|
| **TEDA** | `m=3.0` | Default do paper Angelov 2014 |
| **CluStream** | `k=5, max_micro=100, time_window=1000` | Paper CluStream original |
| **DenStream** | `lambda=0.01, eps=0.1, beta=0.5, mu=2` | Paper DenStream original |
| **MicroTEDAclus** | `r0=0.1, min_samples=10` | Default S5 (já validado) |

**Nota:** Algoritmos que requerem k (CluStream) usam `k=5` baseado em análise exploratória dos dados.

---

#### 8.5.5 Experimento Comparativo

**Workflow:**

```bash
# 1. Rodar cada algoritmo no Cenário B (5 repetições cada)
for algo in teda clustream denstream micro_tedaclus; do
    for run in {1..5}; do
        python scripts/run_experiment.py \
            --scenario B_sudden_drift \
            --benign_pcap data/pcaps/benign/BenignTraffic.pcap \
            --attack_pcaps data/pcaps/ddos/DDoS-ICMP.pcap,data/pcaps/mirai/Mirai-greeth_flood.pcap \
            --algorithm $algo \
            --output results/week9/comparison/${algo}/run_${run}/ \
            --seed $run
    done
done

# 2. Consolidar resultados
python scripts/compare_streaming_algorithms.py \
    --input results/week9/comparison/ \
    --output results/week9/comparison_report.md \
    --generate_plots
```

**Tempo estimado:** ~2-3 horas (4 algoritmos × 5 runs × ~10min cada)

---

#### 8.5.6 Métricas de Comparação (Clustering Streaming)

**Baseado no paper MicroTEDAclus:**

**8.5.6.1 Qualidade de Detecção:**

| Métrica | Descrição | Objetivo |
|---------|-----------|----------|
| **Precision** | TP / (TP + FP) | Minimizar falsos positivos |
| **Recall** | TP / (TP + FN) | Detectar maioria dos ataques |
| **F1-Score** | Harmônica P/R | Balanço geral |
| **AUC-ROC** | Área sob curva | Performance geral |

**8.5.6.2 Adaptação a Drift:**

| Métrica | Descrição | Como Calcular |
|---------|-----------|---------------|
| **Tempo de Adaptação** | Flows até F1 recuperar após drift | `recovery_idx - drift_idx` |
| **F1 antes drift** | F1 na fase 1 (DDoS) | Média janelas [0-7500] |
| **F1 depois drift** | F1 na fase 2 (Mirai) | Média janelas [7500-15000] |
| **Degradação máxima** | Queda de F1 no drift | `min(F1) - baseline_F1` |

**8.5.6.3 Eficiência Computacional:**

| Métrica | Descrição | Unidade |
|---------|-----------|---------|
| **Throughput** | Flows processados por segundo | flows/s |
| **Memória média** | Média de RSS durante execução | MB |
| **Memória pico** | Pico de RSS | MB |
| **Tempo por flow** | Latência média | ms/flow |

**8.5.6.4 Características de Clustering:**

| Métrica | Descrição |
|---------|-----------|
| **#Clusters médio** | Número médio de clusters ativos |
| **Taxa de criação** | Novos clusters / 1000 flows |
| **Estabilidade** | Variância de #clusters ao longo do tempo |

---

#### 8.5.7 Tabela Comparativa Esperada

**Baseado em resultados do paper Maia et al. 2020:**

| Algoritmo | F1 | Precision | Recall | Tempo Adapt. | Throughput | Memória | #Clusters |
|-----------|----|-----------| -------|--------------|------------|---------|-----------|
| **TEDA** | ~0.85 | ~0.82 | ~0.88 | >1000 flows | ~1000 | <50MB | 1 |
| **CluStream** | ~0.88 | ~0.86 | ~0.90 | ~800 | ~800 | ~150MB | ~5-10 |
| **DenStream** | ~0.90 | ~0.89 | ~0.91 | ~600 | ~700 | ~120MB | ~8-15 |
| **MicroTEDAclus** | **~0.92** | **~0.91** | **~0.93** | **~400** | ~900 | <100MB | ~5-12 |

**Nota:** Valores aproximados baseados no paper. Experimento real vai preencher com resultados reais.

---

#### 8.5.8 Análise Comparativa (Dissertação)

**Perguntas a Responder:**

1. **MicroTEDAclus é competitivo?**
   - F1 comparável ou superior aos baselines?
   - Tempo de adaptação menor (principal vantagem esperada)?

2. **Trade-offs identificados:**
   - Memória: MicroTEDAclus vs CluStream/DenStream
   - Throughput: Custo computacional por flow
   - Simplicidade: Menos hiperparâmetros que DenStream

3. **Quando usar cada algoritmo?**
   - **TEDA:** Baseline rápido, cenários sem drift
   - **CluStream:** Quando k é conhecido, dados bem separados
   - **DenStream:** Clusters de densidade variável
   - **MicroTEDAclus:** Concept drift frequente, sem conhecimento prévio de k

**Gráficos a Gerar:**

1. **F1 timeline (4 algoritmos sobrepostos)** - mostra adaptação a drift
2. **Box plot de F1** - variância entre runs
3. **Memória vs Throughput** - scatter plot trade-off
4. **Tempo de adaptação** - bar chart comparativo
5. **#Clusters ao longo do tempo** - line plot evolução

---

#### 8.5.9 Entregáveis S9

- [ ] `src/detector/streaming_adapter.py` - Adapter para River
- [ ] Integração CluStream + DenStream testada
- [ ] Script `compare_streaming_algorithms.py`
- [ ] Experimentos: 4 algoritmos × 5 runs = 20 execuções
- [ ] `results/week9/comparison_report.md` com tabelas e gráficos
- [ ] Análise estatística (ANOVA, Tukey HSD para diferenças significativas)
- [ ] Seção 5.3 da dissertação rascunhada: "Comparação com Estado-da-Arte"

---

#### 8.5.10 Fallback (se River não funcionar)

**Plano B:** Comparar apenas TEDA vs MicroTEDAclus (2 algoritmos já implementados)

**Argumentação válida:**
- Mostra evolução: single-center → multi-cluster
- Evidencia resolução do problema de contaminação
- Paper Maia comparou com 3 algoritmos (CluStream, DenStream, StreamKM++)
- Nossa comparação TEDA vs MicroTEDAclus + argumentação teórica é suficiente

**Decisão:** Tentar integrar River na S9, mas ter Plano B documentado.

---

### 8.6 Cronograma de Experimentos (Resumo)

| Semana | Experimentos | Cenários | Entregáveis Chave |
|--------|--------------|----------|-------------------|
| **S5** | Orquestração + E2E + Benchmark | A (básico) | Scripts, 6 execuções validadas |
| **S6** | Sistema de métricas | - | Prequential metrics implementado |
| **S7** | Primeiros experimentos | A (Detecção básica) | Resultados Cenário A (5 runs) |
| **S8** | Experimentos drift + merge/split | B, C (Súbito, Gradual) | Cenários B e C + merge/split |
| **S9** | **Comparação streaming algorithms** | B (drift súbito) | **TEDA/CluStream/DenStream/MicroTEDAclus** |
| **S10** | Otimização, bug fixes | - | MVP estável |
| **S11** | Full dataset | Todos os cenários | Resultados completos |
| **S12** | Análise concept drift | B, C, D completos | Análise aprofundada |

---

## 9. Checklist Pré-Experimento

- [ ] PCAPs selecionados e verificados
- [ ] Parâmetros do detector definidos
- [ ] Métricas configuradas
- [ ] Script de execução testado
- [ ] Diretório de output criado
- [ ] Seeds definidos
- [ ] Hardware disponível
- [ ] Tempo estimado calculado

---

## 10. Referências

### Papers Metodológicos

1. Ayoubi, S., et al. (2023). [Data-Driven Evaluation of Intrusion Detectors: A Methodological Framework](https://link.springer.com/chapter/10.1007/978-3-031-30122-3_9). LNCS.

2. Gama, J., et al. (2013). [On Evaluating Stream Learning Algorithms](https://link.springer.com/article/10.1007/s10994-012-5320-9). Machine Learning.

3. (2024). [Evolving Cybersecurity Frontiers: Concept Drift in IDS](https://www.sciencedirect.com/science/article/pii/S0952197624013010). Engineering Applications of AI.

4. [Expectations Versus Reality: Evaluating IDS in Practice](https://arxiv.org/html/2403.17458v2). arXiv 2024.

### Dataset

5. Neto, E.C.P., et al. (2023). [CICIoT2023: A Real-Time Dataset and Benchmark](https://www.mdpi.com/1424-8220/23/13/5941). Sensors.

### Streaming e Concept Drift

6. [Enhanced Intrusion Detection with Data Stream Classification and Concept Drift](https://www.mdpi.com/1424-8220/23/7/3736). Sensors 2023.

7. [INSOMNIA: Towards Concept-Drift Robustness in Network Intrusion Detection](https://www.researchgate.net/publication/356218930_INSOMNIA_Towards_Concept-Drift_Robustness_in_Network_Intrusion_Detection). AISec 2021.

### Ferramentas

8. [scikit-multiflow: Prequential Evaluation](https://scikit-multiflow.readthedocs.io/en/stable/api/generated/skmultiflow.evaluation.EvaluatePrequential.html)

9. [MOA: Machine Learning for Data Streams](https://book.moa.cms.waikato.ac.nz/chapter_6.html/)

---

**Este documento serve como guia para conduzir experimentos válidos e reprodutíveis com o sistema IoT IDS.**

*Atualizar conforme experimentos forem realizados e novas práticas identificadas.*
