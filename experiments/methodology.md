# Metodologia de Experimentos - IoT IDS com Clustering Evolutivo

> Documento base para o Capítulo 4 da dissertação. Seções táticas (plano semanal) foram removidas — ver campaign-plan.md.

**Criado:** 2026-01-29
**Última Atualização:** 2026-03-07

---

## 1. Introdução

Este documento define a metodologia para conduzir experimentos válidos e reprodutíveis com o sistema de detecção de intrusão baseado em MicroTEDAclus, utilizando os arquivos PCAP do dataset CICIoT2023.

### 1.1 Objetivos dos Experimentos

1. **Validar detecção de anomalias:** O sistema detecta ataques conhecidos?
2. **Avaliar resistência a concept drift:** O sistema se adapta a novos padrões?
3. **Medir generalização:** O sistema detecta ataques não vistos durante treinamento?
4. **Comparar MicroTeda com outros algoritmos:** Como se compara aos algoritmos de clustering?

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

### 3.1 Avaliação Tradicional (Batch) (DESCONTINUADA)

~~Usada para comparação com Fase 1 e baselines.~~

**STATUS:** ❌ **NÃO SERÁ IMPLEMENTADO**

**Motivo:** Comparação direta batch vs streaming é **metodologicamente inválida**:
- Protocolos diferentes: k-fold CV vs prequential
- Datasets diferentes: CSV shuffled vs PCAP temporal
- Objetivos diferentes: accuracy final vs adaptação contínua

**Alternativa:** Ver seção 3.4 "Comparação com Literatura" abaixo.

### 3.2 Avaliação Prequential (Streaming)

**Recomendada para MicroTEDAclus** - baseada em [On Evaluating Stream Learning Algorithms](https://link.springer.com/article/10.1007/s10994-012-5320-9).

```
┌─────────────────────────────────────────────────────────────┐
│              PREQUENTIAL (Test-Then-Train)                   │
│                                                              │
│   Para cada amostra x:                                       │
│     1. Prever: ŷ = modelo.predict(x)  ← is_anomaly?        │
│     2. Avaliar: comparar ŷ com y_true (aplicado externamente)│
│     3. Atualizar: modelo.update(x)    ← sem labels          │
│                                                              │
│   Nota: o detector é não-supervisionado — atualiza apenas   │
│   com x. O y_true é aplicado pelo orquestrador offline.    │
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

---

### 3.4 Comparação com Algoritmos Streaming

**Objetivo:** Validar MicroTEDAclus comparando com algoritmos streaming estabelecidos.

**Protocolo:** Avaliação prequential com mesmo PCAP para todos os algoritmos.

#### Algoritmos para Comparação (Semana 9)

| Algoritmo | Tipo | Status |
|-----------|------|--------|
| **TEDA** | Single-center baseline | ✅ Implementado |
| **MicroTEDAclus** | Multi-cluster evolutivo (proposto) | ✅ Implementado |
| **CluStream** | Micro+Macro clustering | A implementar (biblioteca `river`) |
| **DenStream** | Density-based streaming | A implementar (biblioteca `river`) |
| **StreamKM++** | K-means streaming | A implementar (ou deixar opcional) |

**Plano mínimo:** TEDA vs MicroTEDAclus (4 algoritmos se incluir CluStream e DenStream).

#### Protocolo de Comparação

1. **Mesmo PCAP:** Todos algoritmos processam exatamente o mesmo stream temporal
2. **Mesmo protocolo:** Avaliação prequential (test-then-train)
3. **Mesmas métricas:** Prequential F1, MTTD, tempo de adaptação, throughput, memória
4. **Mesmo hardware:** Executar no mesmo ambiente
5. **Múltiplas execuções:** 5 runs com seeds diferentes para validação estatística

#### Métricas de Comparação

- **Prequential F1:** Antes e depois de drift
- **Tempo de adaptação:** Flows até recuperar 95% do F1 baseline
- **MTTD:** Mean Time To Detection (flows até primeira detecção)
- **Throughput:** Flows processados por segundo
- **Memória:** Uso de RAM ao longo do tempo
- **Número de clusters:** Evolução temporal

**Detalhamento completo:** Ver seção 8.9 "Semana 9: Comparação de Algoritmos Streaming"

---

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

## 4.4 Limitações do Ground Truth por Fase

### Abordagem Atual

O detector MicroTEDAclus é **puramente não-supervisionado** — não recebe labels durante execução. O ground truth é aplicado **externamente pelo orquestrador** (`run_experiment.py`) após a detecção, usando a estratégia de **rotulação por fase de injeção**:

```
Injeção sequencial:
  [BenignTraffic.pcap]  →  50k packets  → Kafka (tópico packets)
  (pausa 2s)
  [Attack.pcap]         → 100k packets  → Kafka (tópico packets)

FlowConsumer agrega packets em flows (timeout 60s event-time)
Detector consome flows do tópico "flows" (auto_offset_reset=earliest)

Ground truth atribuído por posição ordinal:
  benign_ratio = benign_packets / total_packets
  benign_flow_count = total_flows × benign_ratio
  flow[0..N-1]    → y_true = False  (benigno)
  flow[N..fim]    → y_true = True   (ataque)
```

Código relevante (`run_experiment.py:300-316`):
```python
benign_ratio = benign_packets_sent / total_packets
benign_flow_count = int(len(detection_results) * benign_ratio)
for i, result in enumerate(detection_results):
    y_true = i >= benign_flow_count  # False=benign, True=ataque
```

### Limitações Identificadas

#### L1: Não-linearidade packets → flows

A proporção de packets **não corresponde** linearmente à proporção de flows. Exemplo: 50k packets benignos podem gerar 3.000 flows (muitos packets por flow — conexões longas), enquanto 100k packets de ataque podem gerar 5.000 flows (poucos packets por flow — conexões curtas). A divisão `benign_ratio` assume linearidade, introduzindo erro na fronteira.

**Impacto estimado:** Moderado. Desloca a fronteira benign/ataque em até centenas de flows, afetando primariamente flows próximos à transição de fase.

#### L2: Flows de fronteira (boundary flows)

O FlowConsumer usa timeout de **60 segundos em event-time** para finalizar flows. Um flow que inicia durante a fase benigna pode conter packets da fase de ataque (e vice-versa). Esses "boundary flows" recebem label incorreto independente do método de divisão.

```
Tempo de injeção:  ──── benign ────┊──── ataque ────
                                   ↑
Flow com timeout 60s:     [═══════════════]
                          ↑ inicia benigno, mas inclui
                            packets de ataque
```

**Impacto estimado:** Baixo. Afeta apenas flows ativos no momento da transição (~dezenas de flows, <1% do total).

#### L3: Reordenação no Kafka

O FlowConsumer emite flows quando seu timeout expira, **não na ordem de injeção dos packets**. Flows benignos com timeout longo podem ser emitidos após flows de ataque com timeout curto, criando intercalação na ordem de consumo do detector.

**Impacto estimado:** Moderado a alto. Se flows benignos são emitidos tardiamente, eles recebem label de ataque (e vice-versa). A magnitude depende da distribuição de durações de flow no dataset.

#### L4: Granularidade de label

Todos os flows de uma fase recebem o **mesmo label**. Não há distinção entre:
- Flows de ataque que realmente contêm tráfego malicioso
- Flows benignos que por acaso foram criados durante a fase de ataque (ex: tráfego de fundo IoT que continua durante o ataque)

O dataset CICIoT2023 contém **tráfego misto** nos PCAPs de ataque — os dispositivos IoT continuam gerando tráfego benigno durante os ataques. A rotulação por fase classifica **todo** esse tráfego como ataque.

**Impacto estimado:** Alto. Infla artificialmente o número de FN (flows benignos na fase de ataque que o detector corretamente identifica como normais, mas que são contados como "ataques não detectados"). Isso **subestima o Recall real** do detector.

### Quantificação do Impacto nas Métricas

| Limitação | Efeito no Recall | Efeito na Precision | Efeito no FPR |
|-----------|-------------------|---------------------|----------------|
| L1 (não-linearidade) | ± (desloca fronteira) | ± (desloca fronteira) | ± |
| L2 (boundary flows) | Negligível (<1%) | Negligível | Negligível |
| L3 (reordenação Kafka) | ↓ subestima | ↓ subestima | ↑ superestima |
| L4 (granularidade) | ↓ **subestima significativamente** | ↑ superestima | Sem efeito |

**Conclusão:** As limitações L3 e L4 tendem a **subestimar o Recall** e **superestimar o FPR**. Contudo, com Recall observado de ~3-4% e alvo de ≥80%, mesmo corrigindo essas limitações é improvável que o Recall atinja valores aceitáveis. O problema de representação (features indistinguíveis) permanece como causa raiz.

### Alternativa: Ground Truth por IP (Solução S3)

Uma abordagem mais precisa seria rotular flows pelo **IP de origem/destino**, usando os IPs conhecidos dos atacantes no CICIoT2023:

```python
# Pseudocódigo — ground truth por IP
ATTACK_IPS = {"192.168.1.100", "192.168.1.101", ...}  # IPs do CICIoT2023

for flow in detection_results:
    y_true = (flow["src_ip"] in ATTACK_IPS or
              flow["dst_ip"] in ATTACK_IPS)
```

**Vantagens:**
- Elimina L1, L2, L3 e L4 simultaneamente
- Label granular por flow individual
- Independe da ordem de emissão do FlowConsumer

**Pré-requisitos:**
- Mapear IPs de atacantes no dataset CICIoT2023 (documentação do dataset)
- Flows precisam preservar IPs de origem/destino (atualmente disponível)

**Status:** Implementado. O script `extract_attack_ips.py` gera `data/attack_ips.json`
e o `run_experiment.py` aceita `--ground-truth ip` (default) com fallback automático
para `--ground-truth phase` se o arquivo de IPs não existir.

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

### 5.4 Comparação com Baseline (Fase 1) (ISSO NÃO SERÁ MAIS FEITO)

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

---

## 8. Detecção por Janela Temporal (Campaign-02, Step 3)

### 8.1 Motivação

Campaign-01 revelou que flows individuais de ataque são estatisticamente indistinguíveis de flows benignos no espaço de features. Um flow DDoS individual pode ter as mesmas características que um heartbeat IoT. Porém, um IP atacante gera **centenas de flows similares em poucos segundos** — esse padrão agregado é anômalo.

### 8.2 Arquitetura

```
flows → WindowAggregator → features_por_IP_por_janela → MicroTEDAclus → is_anomaly?
```

O `WindowAggregator` acumula flows por `src_ip` em janelas temporais fixas (default: 10 segundos). Quando uma janela fecha, emite um vetor de 12 features agregadas para cada IP que atingiu `min_flows_per_window` (default: 5).

### 8.3 Features Agregadas (12)

| Feature | Sinal de ataque |
|---------|-----------------|
| `flow_count` | DDoS: muito alto |
| `total_packets` | DDoS: muito alto |
| `total_bytes` | DDoS: muito alto |
| `unique_dst_ips` | Recon: alto |
| `unique_dst_ports` | PortScan: alto |
| `mean_flow_duration` | DDoS: muito curto |
| `mean_packets_per_flow` | Característico por tipo |
| `mean_packet_size` | DDoS: uniforme, pequeno |
| `std_packet_size` | DDoS: muito baixo |
| `fwd_bwd_ratio_mean` | DDoS: fortemente unidirecional |
| `syn_ratio` | SYN flood: muito alto |
| `mean_iat` | DDoS: muito baixo |

### 8.4 Ground Truth em Modo Window

Em modo window, a unidade de avaliação muda de "flow" para "IP/janela". O ground truth é: `y_true = src_ip in attack_ips`. Isso é semanticamente correto pois estamos classificando o **comportamento do IP**, não flows individuais.

### 8.5 Behavioral Window Features (v2 = 19 features)

**Motivação (Campaign-02 S3):** As 12 features básicas de janela são agregados simples (somas, médias, contagens únicas). Um IP benigno com 20 flows/janela e um atacante com 20 flows/janela produzem vetores similares nessas features. Features comportamentais capturam **como** os flows se distribuem, não apenas **quanto**.

| Feature | Fórmula | Sinal de ataque |
|---------|---------|-----------------|
| `flows_per_second` | `flow_count / window_size_seconds` | DDoS/scan: 10-100x mais flows/s |
| `dst_port_entropy` | Shannon entropy das dst_port | PortScan: alta, DDoS: ≈0 (porta fixa) |
| `dst_ip_entropy` | Shannon entropy das dst_ip | DDoS flood: ≈0 (alvo único), benigno: variado |
| `unanswered_ratio` | flows com SYN>0, ACK==0 / total | SYN flood: handshake nunca completa |
| `payload_std` | std(total_bytes) dos flows | DDoS: pacotes uniformes (std baixo) |
| `small_flow_ratio` | flows com packet_count≤3 / total | Scan/probe: muitos flows minúsculos |
| `fwd_only_ratio` | flows com bwd_packet_count==0 / total | DDoS/scan: fortemente unidirecionais |

Todos os campos necessários já existem no flow dict do FlowConsumer. Seleção via `--window-features {v1,v2}`.

### 8.6 Parâmetros

| Parâmetro | CLI | Default | Descrição |
|-----------|-----|---------|-----------|
| Tamanho da janela | `--window-seconds` | 10.0 | Segundos por janela |
| Min flows | `--min-flows-per-window` | 5 | Mínimo de flows para emitir vetor |
| Granularidade | `--granularity` | flow | `flow` ou `window` |
| Window features | `--window-features` | v1 | `v1` (12 basic) ou `v2` (19 behavioral) |

---

**Este documento serve como guia para conduzir experimentos válidos e reprodutíveis com o sistema IoT IDS.**

*Atualizar conforme experimentos forem realizados e novas práticas identificadas.*
