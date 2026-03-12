# Campanha 01 — Analise de Resultados

**Data:** 2026-03-10 a 2026-03-12
**Cenarios:** A1 (baseline benigno), A2 (deteccao DDoS — 3 variantes), A3 (TEDA vs MicroTEDAclus)
**Dataset:** CICIoT2023 (Benign_Final + DDoS-ICMP/SYN/TCP Flood)
**Pacotes por PCAP:** 100.000 | **Max flows:** 10.000

---

## 1. Resumo Executivo

| Cenario | Algoritmo | Resultado | Status |
|---------|-----------|-----------|--------|
| A1 — Baseline benigno | MicroTEDAclus | FPR ~3.5% (alvo <= 5%) | APROVADO |
| A2 — DDoS-ICMP Flood | MicroTEDAclus | Recall ~4.5% (alvo >= 80%) | REPROVADO |
| A2 — DDoS-SYN Flood | MicroTEDAclus | Recall ~4.3% (alvo >= 80%) | REPROVADO |
| A2 — DDoS-TCP Flood | MicroTEDAclus | Recall ~3.6% (alvo >= 80%) | REPROVADO |
| A3 — TEDA baseline | TEDA | Recall ~0.05% | REPROVADO |

**Conclusao principal:** O MicroTEDAclus apresenta FPR aceitavel em trafego benigno,
mas e **incapaz de detectar ataques DDoS independente do protocolo** (ICMP, SYN, TCP).
A anomaly rate e identica (~3.5%) com e sem ataque. O detector identifica outliers
estatisticos, mas nao distingue ataque de trafego benigno.

**Hipotese refutada:** A causa NAO e a ausencia de TCP flags (SYN Flood tambem falha).
A causa raiz e mais fundamental — ver Secao 5.

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

## 3. Cenario A2 — Deteccao DDoS (3 Variantes)

**Objetivo:** Detectar ataques DDoS em fluxo misto (benigno + ataque).
**Criterio de sucesso:** Recall >= 80%, MTTD <= 500 flows

### Configuracao
- PCAP benigno: `Benign_Final/BenignTraffic.pcap` (100k packets)
- PCAP ataque: 3 variantes, cada uma com 100k packets
- Divisao de fases: ~50% benigno / ~50% ataque
- Algoritmo: MicroTEDAclus

### Resultados completos

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

### Observacao critica: Anomaly Rate constante independente do ataque

```
A1 (benign-only):  3.53%  ← baseline sem ataque
A2 (ICMP Flood):   3.78%  ← +0.25% vs baseline
A2 (SYN Flood):    3.84%  ← +0.31% vs baseline
A2 (TCP Flood):    3.47%  ← -0.06% vs baseline (!)
```

A anomaly rate e **estatisticamente identica** com e sem ataque, e para
**todos os protocolos**. Isso invalida a hipotese de que TCP flags
seriam suficientes para discriminar.

### Distribuicao de anomalias por fase

| Ataque | Anom fase benigna | Anom fase ataque | Ratio ataque/benigno |
|--------|-------------------|------------------|---------------------|
| ICMP Flood | 3.1% | 4.4% | 1.42x |
| SYN Flood | 3.4% | 4.3% | 1.25x |
| TCP Flood | 3.4% | 3.6% | 1.06x |

O detector flagra proporcoes quase identicas em ambas as fases. Nao ha
separacao significativa entre trafego benigno e ataque.

### Hipotese refutada: TCP flags como discriminador (S1)

A solucao S1 ("usar ataque TCP para que flags discriminem") foi testada
e **refutada**. SYN Flood e TCP Flood tem flags TCP distintas, mas o
Recall permanece ~4%. Conclusao: o detector e cego para ataques DDoS
independente do protocolo.

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
```

Problemas:
1. **Nao-linearidade packets→flows**: 100k packets benignos e 100k de ataque
   podem gerar numeros muito diferentes de flows
2. **Flows de fronteira**: um flow que comeca na fase benigna pode incluir
   packets da fase de ataque (timeout de 60s em event-time)
3. **Mistura no Kafka**: o FlowConsumer emite flows baseado em timeout,
   nao em ordem de injecao — flows benignos e de ataque podem se intercalar

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

### 5.5 Solucoes a investigar (priorizadas apos resultados SYN/TCP)

| # | Solucao | Hipotese | Status | Prioridade |
|---|---------|----------|--------|------------|
| ~~S1~~ | ~~Usar ataque TCP (SYN Flood)~~ | ~~TCP flags discriminam~~ | REFUTADA | — |
| S2 | Features de volume agregado | Burst rate, flows/s por IP | Medio | **ALTA** |
| S3 | Ground truth por IP (nao por fase) | Labels precisos por flow | Alto | **ALTA** |
| S4 | Ataques nao-volumetricos (Mirai, Recon) | Padroes distintos de DDoS | Baixo | **ALTA** |
| S5 | Intercalar flows benigno/ataque | Remover vies de ordem | Medio | MEDIA |
| S6 | Deteccao em nivel de dispositivo | Modelo por IP de origem | Alto | MEDIA |
| S7 | Normalizar features por protocolo | Separar modelo ICMP/TCP/UDP | Alto | BAIXA |

**Recomendacao imediata:** Executar S4 com ataques nao-volumetricos (Mirai,
Recon/PortScan) que tem padroes de trafego fundamentalmente diferentes.

---

## 6. Proximos Passos

### Imediatos
1. **Rodar A2 com Mirai** — ataque nao-volumetrico (botnet com padroes distintos)
2. **Rodar A2 com Recon/PortScan** — ataque de reconhecimento (muitas conexoes curtas)
3. **Investigar ground truth por IP** — labels mais precisos que por fase

### Subsequentes
4. Implementar `--drift-pcap` para cenarios B (drift subito)
5. Considerar features de volume agregado (flows/s por IP, burst rate)
6. Analisar distribuicao de features por fase (histogramas benign vs attack)

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
  A3-teda-baseline/         ← TEDA vs MicroTEDAclus
  ANALYSIS.md               ← Este documento
```

**Total: 13 experimentos executados.**

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
