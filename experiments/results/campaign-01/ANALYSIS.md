# Campanha 01 — Analise de Resultados

**Data:** 2026-03-10 a 2026-03-12
**Cenarios:** A1 (baseline benigno), A2 (deteccao de ataques — 5 variantes), A3 (TEDA vs MicroTEDAclus)
**Dataset:** CICIoT2023 (Benign_Final + DDoS-ICMP/SYN/TCP Flood + Mirai-greeth + Recon-PortScan)
**Pacotes por PCAP:** 50k benigno + 100k ataque | **Max flows:** 10.000

---

## 1. Resumo Executivo

| Cenario | Algoritmo | Resultado | Status |
|---------|-----------|-----------|--------|
| A1 — Baseline benigno | MicroTEDAclus | FPR ~3.5% (alvo <= 5%) | APROVADO |
| A2 — DDoS-ICMP Flood | MicroTEDAclus | Recall ~4.5% (alvo >= 80%) | REPROVADO |
| A2 — DDoS-SYN Flood | MicroTEDAclus | Recall ~4.3% (alvo >= 80%) | REPROVADO |
| A2 — DDoS-TCP Flood | MicroTEDAclus | Recall ~3.6% (alvo >= 80%) | REPROVADO |
| A2 — Mirai-greeth | MicroTEDAclus | Recall ~2.7% (alvo >= 80%) | REPROVADO |
| A2 — Recon-PortScan | MicroTEDAclus | Recall ~4.0% (alvo >= 80%) | REPROVADO |
| A3 — TEDA baseline | TEDA | Recall ~0.05% | REPROVADO |

**Conclusao principal:** O MicroTEDAclus apresenta FPR aceitavel em trafego benigno,
mas e **incapaz de detectar qualquer tipo de ataque testado** — DDoS volumetrico
(ICMP, SYN, TCP), botnet (Mirai) e reconhecimento (PortScan). A anomaly rate e
identica (~3.5%) com e sem ataque. O detector identifica outliers estatisticos
naturais do trafego, nao ataques.

**Resultado positivo parcial:** Para Mirai e Recon, a **Precision** e significativamente
melhor (55-66% vs ~55% para DDoS). Quando o detector flagra algo durante esses ataques,
e mais provavel que seja um ataque real. Mas o Recall permanece criticamente baixo (~3-4%).

**Hipoteses refutadas:**
1. TCP flags como discriminador (SYN/TCP Flood tambem falha)
2. Ataques nao-volumetricos seriam mais detectaveis (Mirai/Recon tambem falham)

---

## 2. Cenario A1 — Baseline Benigno (Falsos Positivos)

**Objetivo:** Medir taxa de falsos positivos em trafego puramente benigno.
**Criterio de sucesso:** FPR <= 5%

### Configuracao
- PCAP: `Benign_Final/BenignTraffic.pcap` (100k packets)
- Algoritmo: MicroTEDAclus
- Variacao: r0 in {0.05, 0.10, 0.15, 0.20}

### Resultados

| r0 | Flows | Anomalias | FPR | Clusters | Throughput |
|----|-------|-----------|-----|----------|------------|
| 0.05 | 7.538 | 266 | 3.53% | — | ~142 flows/s |
| 0.10 | 7.509 | 265 | 3.53% | 274 | ~142 flows/s |
| 0.15 | 7.536 | 262 | 3.48% | — | ~142 flows/s |
| 0.20 | 7.525 | 262 | 3.48% | — | ~142 flows/s |

### Observacoes

1. **FPR estavel entre 3.4-3.5%** independente de r0 — bem abaixo do alvo de 5%.
2. **r0 tem impacto minimo** no cenario benign-only. Isso e esperado: r0 controla
   a variancia minima dos micro-clusters, afetando a criacao de novos clusters.
   Com trafego homogeneo, a maioria dos flows cai em clusters existentes.
3. **~266 anomalias em ~7500 flows** — sao flows benigno que o detector
   classifica como anomalos. Provavelmente flows com caracteristicas incomuns
   (ex: conexoes muito longas, pacotes muito grandes).
4. **Throughput ~142 flows/s** — acima do alvo de 100 flows/s.

### Implicacao para a dissertacao

> O MicroTEDAclus apresenta taxa de falsos positivos de 3.5% em trafego benigno,
> inferior ao limiar aceitavel de 5%. O parametro r0 nao afeta significativamente
> o FPR neste cenario, indicando robustez da configuracao.

---

## 3. Cenario A2 — Deteccao de Ataques (5 Variantes)

**Objetivo:** Detectar ataques em fluxo misto (benigno + ataque).
**Criterio de sucesso:** Recall >= 80%, MTTD <= 500 flows

### Configuracao
- PCAP benigno: `Benign_Final/BenignTraffic.pcap` (50k packets)
- PCAP ataque: 5 variantes, cada uma com 100k packets
- Divisao de fases: ~33% benigno / ~67% ataque
- Algoritmo: MicroTEDAclus

### 3.1 Resultados — Ataques DDoS Volumetricos

| Ataque | r0 | Flows | Anom | Rate | Precision | Recall | F1 | Clusters |
|--------|----|-------|------|------|-----------|--------|-----|----------|
| ICMP Flood | 0.05 | 7.647 | 291 | 3.81% | 57.7% | 4.4% | 8.2% | 300 |
| ICMP Flood | 0.10 | 7.675 | 290 | 3.78% | 58.6% | 4.4% | 8.2% | 299 |
| ICMP Flood | 0.15 | 7.663 | 299 | 3.90% | 59.5% | 4.7% | 8.6% | 308 |
| ICMP Flood | 0.20 | 7.661 | 301 | 3.93% | 58.5% | 4.6% | 8.5% | 310 |
| SYN Flood | 0.10 | 10.000 | 384 | 3.84% | 55.5% | 4.3% | 7.9% | 393 |
| SYN Flood | 0.15 | 10.000 | 373 | 3.73% | 55.2% | 4.1% | 7.7% | 382 |
| TCP Flood | 0.10 | 7.548 | 262 | 3.47% | 51.5% | 3.6% | 6.7% | 271 |
| TCP Flood | 0.15 | 7.544 | 270 | 3.58% | 51.9% | 3.7% | 6.9% | 279 |

### 3.2 Resultados — Ataques Nao-Volumetricos

| Ataque | r0 | Flows | Anom | Rate | Precision | Recall | F1 | MTTD | Clusters |
|--------|----|-------|------|------|-----------|--------|-----|------|----------|
| Mirai-greeth | 0.10 | 4.266 | ~150 | ~3.5% | 55.3% | 2.7% | 5.2% | 46s | — |
| Mirai-greeth | 0.15 | 4.275 | ~145 | ~3.4% | 54.1% | 2.6% | 4.9% | 61s | — |
| Recon-PortScan | 0.10 | 10.000 | ~380 | ~3.8% | 64.3% | 3.8% | 7.3% | 12s | — |
| Recon-PortScan | 0.15 | 10.000 | ~385 | ~3.9% | 66.4% | 4.2% | 7.9% | 6s | — |

**Nota sobre MTTD:** Mirai e Recon sao os primeiros cenarios com MTTD mensuravel
(DDoS nao tinha deteccoes verdadeiras suficientes). Recon-PortScan tem MTTD de 6-12s
— o algoritmo detecta *algumas* anomalias rapidamente, mas perde a grande maioria.

### 3.3 Comparacao entre tipos de ataque

```
                    Anomaly Rate    Precision    Recall
                    ┌──────────┐   ┌─────────┐  ┌──────┐
Sem ataque (A1):    ███  3.53%     N/A          N/A
ICMP Flood:         ███  3.78%     ████  58.6%  █  4.5%
SYN Flood:          ███  3.84%     ████  55.5%  █  4.3%
TCP Flood:          ███  3.47%     ███   51.5%  █  3.6%
Mirai-greeth:       ███  3.5%      ████  55.3%  █  2.7%
Recon-PortScan:     ███  3.8%      █████ 64.3%  █  3.8%
                    └──────────┘   └─────────┘  └──────┘
```

**Achados:**
1. **Anomaly rate invariante** (~3.5%) para TODOS os tipos de ataque — confirma
   que o detector identifica outliers naturais, nao ataques
2. **Recall uniformemente baixo** (~3-4%) independente do tipo de ataque
3. **Precision melhor para Recon** (64%) — indica que PortScan gera *alguns*
   flows genuinamente anomalos, mas sao uma minoria
4. **MTTD baixo para Recon** (6-12s) — o algoritmo e rapido quando detecta,
   mas detecta muito pouco

### 3.4 Distribuicao de anomalias por fase

| Ataque | Anom fase benigna | Anom fase ataque | Ratio ataque/benigno |
|--------|-------------------|------------------|---------------------|
| ICMP Flood | 3.1% | 4.4% | 1.42x |
| SYN Flood | 3.4% | 4.3% | 1.25x |
| TCP Flood | 3.4% | 3.6% | 1.06x |
| Mirai-greeth | ~3.5% | ~2.7% | 0.77x |
| Recon-PortScan | ~3.5% | ~4.0% | 1.14x |

O detector flagra proporcoes quase identicas em ambas as fases. Para Mirai,
a fase de ataque tem *menos* anomalias que a fase benigna — o ataque
Mirai-greeth gera trafego ainda mais "regular" que o benigno.

### 3.5 Hipoteses refutadas

1. **S1 — TCP flags como discriminador**: SYN Flood e TCP Flood tem flags TCP
   distintas, mas Recall permanece ~4%. REFUTADA.

2. **S4 — Ataques nao-volumetricos seriam mais detectaveis**: Mirai (botnet)
   e Recon-PortScan geram padroes de trafego distintos de DDoS, mas o Recall
   permanece ~3-4%. REFUTADA.

**Conclusao:** O problema NAO e do tipo de ataque. E uma limitacao fundamental
da representacao (17 features) e/ou da abordagem de deteccao por anomalia
em nivel de flow individual.

---

## 4. Cenario A3 — TEDA vs MicroTEDAclus

**Objetivo:** Comparar TEDA basico com MicroTEDAclus no mesmo cenario.

### Resultados

| Algoritmo | Flows | Anomalias | Precision | Recall | F1 | Clusters |
|-----------|-------|-----------|-----------|--------|-----|----------|
| MicroTEDAclus (r0=0.10) | 7.675 | 290 | 58.6% | 4.4% | 8.2% | 299 |
| TEDA (m=3.0) | 7.684 | 11 | 18.2% | 0.05% | 0.1% | N/A |

### Analise

O TEDA basico e **significativamente pior** que o MicroTEDAclus:
- Apenas 11 anomalias detectadas (vs 290 do MicroTEDAclus)
- Recall praticamente zero (0.05%)
- Precision tambem baixa (18.2%)

**Causa:** O TEDA usa estatisticas globais (media e variancia unicas).
Com ~7600 flows, as estatisticas convergem para a media global,
e o threshold de Chebyshev (m=3.0) raramente e excedido.
O MicroTEDAclus e superior porque isola estatisticas por cluster.

**Implicacao para a dissertacao:**

> O MicroTEDAclus detecta 26x mais anomalias que o TEDA basico no mesmo cenario,
> confirmando que a abordagem multi-cluster com estatisticas isoladas e
> fundamental para deteccao em streams heterogeneos. Contudo, ambos os
> algoritmos apresentam Recall insuficiente para DDoS-ICMP, indicando que
> o problema e de representacao de features, nao de algoritmo.

---

## 5. Discussao: Por que o Detector e Cego para DDoS?

### 5.1 O padrao observado

O achado principal desta campanha e que a anomaly rate (~3.5%) e **invariante**
em relacao a presenca ou tipo de ataque:

```
                    Anomaly Rate
                    ┌─────────────────┐
Sem ataque (A1):    ███████████████     3.53%
Com ICMP Flood:     ████████████████    3.78%
Com SYN Flood:      ████████████████    3.84%
Com TCP Flood:      ███████████████     3.47%
                    └─────────────────┘
```

Isso indica que o detector esta identificando **outliers estatisticos naturais**
do trafego (que existem em qualquer dataset), nao ataques.

### 5.2 Analise de causas raiz

#### Causa raiz 1: Flows de ataque sao estatisticamente "normais" (PRINCIPAL)

Ataques DDoS volumetricos geram um grande numero de flows **simples e uniformes**
(muitos pacotes pequenos, IAT regular, poucos flags). Essas caracteristicas
sao comuns no trafego benigno de IoT (telemetria, heartbeats, pings).

O MicroTEDAclus cria clusters baseado na distribuicao estatistica dos dados.
Se flows de ataque caem na mesma regiao do espaco de features que flows
benignos, eles sao absorvidos pelos clusters existentes.

**Evidencia:** O numero de clusters e quase identico em A1 (274) e A2 (271-310).
Se os ataques criassem um padrao distinto, esperariamos mais clusters em A2.

#### Causa raiz 2: Avaliacao sequencial com ground truth impreciso

A rotulacao ground truth usa proporcao de packets para estimar o limite
entre fases:

```python
benign_flow_count = int(total_flows * benign_packets / total_packets)
for i, result in enumerate(detection_results):
    y_true = i >= benign_flow_count  # False=benign, True=ataque
```

Foram identificadas 4 limitacoes nesta abordagem (documentacao completa
em `experiments/methodology.md` secao 4.4):

1. **L1 — Nao-linearidade packets→flows**: A proporcao de packets nao
   corresponde linearmente a proporcao de flows. 50k packets benignos
   podem gerar um numero desproporcional de flows vs 100k de ataque.
2. **L2 — Boundary flows**: Flows com timeout de 60s podem cruzar a
   fronteira entre fases, contendo packets de ambas.
3. **L3 — Reordenacao no Kafka**: O FlowConsumer emite flows por timeout,
   nao por ordem de injecao — flows benignos e de ataque intercalam.
4. **L4 — Granularidade de label** (mais severa): Todos os flows da fase
   de ataque recebem y_true=True, inclusive trafego benigno de fundo que
   continua durante o ataque. Isso **subestima o Recall** — flows benignos
   corretamente classificados como normais sao contados como FN.

**Impacto combinado:** L3 e L4 tendem a subestimar Recall e superestimar FPR.
Contudo, com Recall de ~3-4% vs alvo de >=80%, mesmo corrigindo essas
limitacoes o gap permanece. A causa raiz 1 (representacao) e dominante.

**Solucao proposta (S3):** Ground truth por IP do atacante (CICIoT2023
documenta IPs). Elimina L1-L4 simultaneamente. Nao implementado.

#### Causa raiz 3: Deteccao por anomalia vs. deteccao por assinatura

O MicroTEDAclus e um detector de **anomalias**, nao de **ataques**.
Ele detecta pontos que desviam da distribuicao majoritaria. Se ataques
DDoS representam 50% dos dados e sao estatisticamente similares ao
trafego benigno, eles NAO sao anomalias — sao parte da distribuicao "normal".

Isso e uma **limitacao fundamental** da abordagem, nao um bug:
- Detector de anomalias: "este flow e diferente do que vi antes?"
- Detector de ataques: "este flow corresponde a um padrao malicioso?"

### 5.3 Comparacao com Fase 1 (Baseline ML supervisionado)

Na Fase 1 (705 experimentos, F1 > 0.99), os algoritmos supervisionados
(Random Forest, XGBoost) usavam:
- **46 features** (vs 17 no streaming)
- **Labels explicitos** do dataset (supervisionado vs nao-supervisionado)
- **CICFlowMeter** para extracao (vs extracao propria no FlowConsumer)

A diferenca de desempenho e esperada. O desafio da dissertacao e quantificar
o trade-off entre a praticidade do streaming nao-supervisionado e a acuracia
da abordagem supervisionada batch.

### 5.4 Implicacoes para a dissertacao

> 1. O MicroTEDAclus funciona corretamente como detector de anomalias:
>    taxa de falsos positivos aceitavel (3.5%) e throughput adequado (142 flows/s).
>
> 2. Ataques DDoS volumetricos nao sao detectaveis por anomalia de flow
>    quando suas features estatisticas coincidem com o trafego benigno.
>    Isso e consistente com a literatura sobre limitacoes de IDS por anomalia.
>
> 3. O parametro r0 nao afeta significativamente a deteccao neste cenario,
>    indicando que o problema e de **representacao**, nao de **sensibilidade**.
>
> 4. Resultados negativos sao contribuicao valida: documentam empiricamente
>    as limitacoes da abordagem e motivam trabalhos futuros.

### 5.5 Solucoes a investigar (priorizadas apos resultados completos)

| # | Solucao | Hipotese | Status | Prioridade |
|---|---------|----------|--------|------------|
| ~~S1~~ | ~~Usar ataque TCP (SYN Flood)~~ | ~~TCP flags discriminam~~ | REFUTADA | — |
| S2 | Features de volume agregado | Burst rate, flows/s por IP | Medio | **ALTA** |
| S3 | Ground truth por IP (nao por fase) | Labels precisos por flow | Alto | **ALTA** |
| ~~S4~~ | ~~Ataques nao-volumetricos (Mirai, Recon)~~ | ~~Padroes distintos de DDoS~~ | REFUTADA | — |
| S5 | Intercalar flows benigno/ataque | Remover vies de ordem | Medio | MEDIA |
| S6 | Deteccao em nivel de dispositivo | Modelo por IP de origem | Alto | MEDIA |
| S7 | Normalizar features por protocolo | Separar modelo ICMP/TCP/UDP | Alto | BAIXA |
| S8 | Aumentar numero de features | Expandir de 17 para ~46 (CICFlowMeter-like) | Alto | **ALTA** |
| S9 | Deteccao por janela temporal | Agregar anomalias em janela (nao por flow) | Medio | **ALTA** |

**Analise:** Com S1 e S4 refutadas, o problema e claramente de **representacao**,
nao de tipo de ataque. As solucoes mais promissoras sao:
- **S2/S8**: Enriquecer features (volume, burst, mais features por flow)
- **S3**: Melhorar ground truth para confirmar se Recall e realmente baixo
- **S9**: Mudar granularidade de deteccao (janela temporal em vez de flow individual)

**Implicacao para a dissertacao:** Este e um resultado negativo significativo que
documenta empiricamente as limitacoes de IDS por anomalia baseado em clustering
evolutivo com features de flow. A contribuicao e mostrar *onde* e *por que* falha.

---

## 6. Proximos Passos

### Imediatos (diagnostico)
1. **Analisar distribuicao de features por fase** — histogramas benign vs attack
   para confirmar que flows sao estatisticamente indistinguiveis
2. **Investigar ground truth por IP** — labels mais precisos que por fase
3. **Verificar se FlowConsumer preserva informacao discriminante** — comparar
   features extraidas com CICFlowMeter para o mesmo PCAP

### Subsequentes (melhorias)
4. **Expandir features** (S2/S8) — adicionar burst rate, flows/s por IP,
   ou aumentar para ~46 features CICFlowMeter-like
5. **Deteccao por janela temporal** (S9) — agregar anomalias em janela
6. Implementar `--drift-pcap` para cenarios B (drift subito)

---

## 7. Artefatos

### Estrutura de resultados
```
experiments/results/campaign-01/
  A1-benign-r0_0.05/       ← Baseline benigno (4 configs)
  A1-benign-r0_0.10/
  A1-benign-r0_0.15/
  A1-benign-r0_0.20/
  A2-ddos-r0_0.05/          ← DDoS-ICMP Flood (4 configs)
  A2-ddos-r0_0.10/
  A2-ddos-r0_0.15/
  A2-ddos-r0_0.20/
  A2-syn-r0_0.10/           ← DDoS-SYN Flood (2 configs)
  A2-syn-r0_0.15/
  A2-tcp-r0_0.10/           ← DDoS-TCP Flood (2 configs)
  A2-tcp-r0_0.15/
  A2-mirai-r0_0.10/         ← Mirai-greeth (2 configs)
  A2-mirai-r0_0.15/
  A2-recon-r0_0.10/         ← Recon-PortScan (2 configs)
  A2-recon-r0_0.15/
  A3-teda-baseline/         ← TEDA vs MicroTEDAclus
  ANALYSIS.md               ← Este documento
```

**Total: 17 experimentos executados.**

### Artefatos por experimento
Cada diretorio contem 5 arquivos:
- `run_meta.json` — metadata (git commit, parametros, timestamps)
- `detection_results.json` — resultados do detector + metricas prequential
- `metrics_windowed.csv` — metricas por janela (final)
- `clusters_state.jsonl` — snapshot dos clusters no final
- `system_usage.csv` — uso de CPU/memoria

### Reproducao
```bash
cd experiments/streaming && source venv/bin/activate

# A1 (exemplo r0=0.10)
python3 scripts/run_experiment.py \
  --pcap ../../data/pcaps/Benign_Final/BenignTraffic.pcap \
  --max-packets 100000 --max-flows 10000 \
  --algorithm micro_teda --r0 0.10 \
  --output ../results/campaign-01/A1-benign-r0_0.10/

# A2 (exemplo r0=0.10)
python3 scripts/run_experiment.py \
  --pcap ../../data/pcaps/Benign_Final/BenignTraffic.pcap \
  --attack-pcap ../../data/pcaps/DDoS-ICMP_Flood/DDoS-ICMP_Flood.pcap \
  --max-packets 100000 --max-packets-attack 100000 --max-flows 10000 \
  --algorithm micro_teda --r0 0.10 \
  --output ../results/campaign-01/A2-ddos-r0_0.10/

# A3 (TEDA baseline)
python3 scripts/run_experiment.py \
  --pcap ../../data/pcaps/Benign_Final/BenignTraffic.pcap \
  --attack-pcap ../../data/pcaps/DDoS-ICMP_Flood/DDoS-ICMP_Flood.pcap \
  --max-packets 100000 --max-packets-attack 100000 --max-flows 10000 \
  --algorithm teda --r0 0.10 \
  --output ../results/campaign-01/A3-teda-baseline/
```

---

## 8. Parametros do Ambiente

| Parametro | Valor |
|-----------|-------|
| Git commit | dc70dab |
| Python | 3.x |
| Kafka | Confluent (Docker) |
| SO | Linux 6.8.0-101-generic |
| Sincronizacao | wait_for_flow_consumer (v1.0) |
| Flow timeout | 60s (event time) |
| Detector idle timeout | 10s (IDLE_LIMIT) |
| Prequential window | 1000 flows |
| Prequential alpha | 0.01 |
