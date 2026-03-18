# Campaign-02 — Analise de Resultados

**Data:** 2026-03-14 a 2026-03-16
**Steps:** S1 (Ground Truth IP), S2 (Feature Expansion), S3 (Window Aggregation)
**Dataset:** CICIoT2023 (Benign_Final + 5 ataques)
**Pacotes por PCAP:** 50k benigno + 50k ataque | **Max flows:** 10.000
**Total de runs:** 72

---

## 1. Resumo Executivo

Campaign-02 testou 3 hipoteses incrementais para melhorar o Recall ~3-4% da Campaign-01:

| Step | Hipotese | Resultado | Veredicto |
|------|----------|-----------|-----------|
| S1 (IP GT) | Ground truth por fase era artefato | DDoS-ICMP: 4%->27%. Resto inalterado. FPR estavel ~3-4%. | Parcialmente util — corrige labels mas nao resolve deteccao |
| S2 (Features v1->v3) | Mais features per-flow ajudam | Zero impacto (+-1pp). v1=v2=v3 | Descartado — features per-flow saturam |
| S3 (Window 5-60s) | Agregacao temporal melhora | SYN 3%->54%, Recon 4%->45% MAS FPR explodiu (58% @60s) | Direcao certa, implementacao insuficiente |

**Configuracao cumulativa best-so-far:**

| Decisao | Valor | Origem |
|---------|-------|--------|
| Algoritmo | MicroTEDAclus | C01 |
| Ground truth | IP-based | C02-S1 |
| Features per-flow | v1 (17) — Occam | C02-S2 |
| Granularidade | Window (direcao certa) | C02-S3 |
| Window features | **variavel aberta** | C03-S4 |

---

## 2. Step S1 — Ground Truth por IP

**Objetivo:** Substituir ground truth por fase (limitacoes L1-L4) por rotulacao por IP de atacante.
**Variavel:** ground_truth={phase->ip}. Demais parametros congelados de C01.

### 2.1 Resultados A1 — Baseline Benigno

| r0 | Flows | Anomalias | FPR |
|----|-------|-----------|-----|
| 0.05 | 3.228 | 127 | 3.93% |
| 0.10 | 3.229 | 125 | 3.87% |
| 0.15 | 3.231 | 121 | 3.74% |
| 0.20 | 3.226 | 115 | 3.56% |

FPR estavel entre 3.5-3.9%, bem abaixo do alvo de 5%. Consistente com C01.

### 2.2 Resultados A2 — Deteccao de Ataques

| Ataque | r0 | Flows | Precision | Recall | F1 | FPR | TP | FP |
|--------|----|-------|-----------|--------|-----|-----|----|----|
| DDoS-ICMP | 0.05 | 3.339 | 16.5% | 26.9% | 20.4% | 3.91% | 25 | 127 |
| DDoS-ICMP | 0.10 | 3.341 | 17.6% | 27.2% | 21.4% | 3.60% | 25 | 117 |
| DDoS-ICMP | 0.15 | 3.335 | 17.5% | 27.2% | 21.3% | 3.64% | 25 | 118 |
| DDoS-ICMP | 0.20 | 3.332 | 17.0% | 26.9% | 20.8% | 3.77% | 25 | 122 |
| DDoS-SYN | 0.10 | 6.196 | 30.4% | 3.5% | 6.3% | 4.23% | 75 | 172 |
| DDoS-SYN | 0.15 | 6.116 | 29.1% | 3.5% | 6.2% | 4.26% | 71 | 173 |
| DDoS-TCP | 0.10 | 3.247 | 0.0% | 0.0% | 0.0% | 3.16% | 0 | 102 |
| DDoS-TCP | 0.15 | 3.250 | 0.0% | 0.0% | 0.0% | 3.27% | 0 | 106 |
| Mirai | 0.10 | 4.234 | 13.0% | 1.7% | 3.0% | 3.51% | 17 | 114 |
| Mirai | 0.15 | 4.242 | 13.5% | 1.7% | 3.1% | 3.35% | 17 | 109 |
| Recon | 0.10 | 9.917 | 61.6% | 4.5% | 8.4% | 3.06% | 233 | 145 |
| Recon | 0.15 | 10.000 | 64.8% | 5.2% | 9.6% | 3.26% | 278 | 151 |

### 2.3 Analise S1

**DDoS-ICMP: Recall 4% -> 27%.** A mudanca de ground truth revelou que a C01 subestimava o Recall do DDoS-ICMP. Com IP GT, o detector de fato detecta ~27% dos flows de ICMP Flood. A melhoria vem da correcao de L4 (flows benignos na fase de ataque nao sao mais contados como FN).

**Demais ataques: sem melhoria significativa.** SYN (~3.5%), TCP (0%), Mirai (~1.7%), Recon (~4.5%). O IP GT corrige as labels mas nao muda a capacidade de deteccao — confirma que o problema e de representacao, nao de labeling.

**DDoS-TCP: 0 TP.** Apenas 12-16 flows de ataque reais no dataset (de 3.250 totais). Provavelmente os flows TCP Flood sao indistinguiveis de trafego TCP benigno ao nivel de flow individual.

**Recon: Precision alta (65%) mas Recall baixo (5%).** Quando o detector flagra algo durante Recon, e provavelmente real — mas detecta muito pouco.

---

## 3. Step S2 — Expansao de Features

**Objetivo:** Testar se mais features per-flow melhoram a separabilidade.
**Variavel:** features={v1(17), v2(25), v3(32)}. GT=ip (congelado de S1).

### 3.1 Resultados Comparativos (r0=0.10)

**A1 — FPR:**

| Features | Flows | FPR |
|----------|-------|-----|
| v1 (17) | 3.229 | 3.87% |
| v2 (25) | 3.224 | 3.60% |
| v3 (32) | 3.233 | 3.74% |

**A2 — Deteccao (r0=0.10):**

| Ataque | Metrica | v1 | v2 | v3 |
|--------|---------|-----|-----|-----|
| DDoS-ICMP | Recall | 27.2% | 27.2% | 26.9% |
| | F1 | 21.4% | 21.0% | 20.9% |
| DDoS-SYN | Recall | 3.5% | 3.7% | 3.2% |
| | F1 | 6.3% | 6.5% | 5.8% |
| DDoS-TCP | Recall | 0.0% | 0.0% | 0.0% |
| Mirai | Recall | 1.7% | 1.7% | 1.7% |
| | F1 | 3.0% | 3.0% | 3.1% |
| Recon | Recall | 4.5% | 4.5% | 4.7% |
| | F1 | 8.4% | 8.4% | 8.8% |

### 3.2 Analise S2

**Zero impacto.** v1, v2, v3 produzem resultados virtualmente identicos (+-1pp). Adicionar `flow_duration`, `packet_size_min/max`, `fwd/bwd_packet_size_mean`, `iat_min/max`, `psh_count` e features direcionais nao melhora a separabilidade.

**Diagnostico:** Features estatisticas per-flow saturam rapidamente. Flows de ataque e benignos ocupam a mesma regiao do espaco de features independente da dimensionalidade. O problema nao e falta de features — e que a granularidade per-flow e insuficiente para capturar padroes de ataque.

**Decisao:** Manter v1 (17 features) por Occam's razor. Features adicionais adicionam dimensionalidade sem beneficio.

---

## 4. Step S3 — Deteccao por Janela Temporal

**Objetivo:** Mudar granularidade de per-flow para per-IP/janela temporal.
**Variavel:** window_seconds={5, 10, 30, 60}. GT=ip, features=v1 (congelados de S1/S2).
**Implementacao:** WindowAggregator com 12 features agregadas (somas, medias, contagens).

### 4.1 Resultados A1 — FPR por Janela

| Janela | Vetores | Anomalias | FPR |
|--------|---------|-----------|-----|
| 5s | 259 | 26 | 10.04% |
| 10s | 209 | 36 | 17.22% |
| 30s | 137 | 7 | 5.11% |
| 60s | 103 | 60 | **58.25%** |

### 4.2 Resultados A2 — Deteccao por Janela (r0=0.10)

| Ataque | Janela | Vetores | Precision | Recall | F1 | FPR |
|--------|--------|---------|-----------|--------|-----|-----|
| DDoS-ICMP | 5s | 263 | 0.0% | 0.0% | 0.0% | 9.96% |
| DDoS-ICMP | 10s | 216 | 0.0% | 0.0% | 0.0% | 3.74% |
| DDoS-ICMP | 30s | 140 | 0.0% | 0.0% | 0.0% | 5.07% |
| DDoS-ICMP | 60s | 112 | 0.0% | 0.0% | 0.0% | 66.97% |
| DDoS-SYN | 5s | 290 | 12.8% | 35.7% | 18.9% | 12.32% |
| DDoS-SYN | 10s | 232 | 19.1% | 30.8% | 23.5% | 7.76% |
| DDoS-SYN | 30s | 157 | 13.0% | **53.9%** | 20.9% | 32.64% |
| DDoS-SYN | 60s | 118 | 7.8% | 40.0% | 13.1% | 43.52% |
| DDoS-TCP | 5s | 261 | 0.0% | 0.0% | 0.0% | 8.81% |
| DDoS-TCP | 10s | 210 | 0.0% | 0.0% | 0.0% | 3.33% |
| DDoS-TCP | 30s | 139 | 0.0% | 0.0% | 0.0% | 5.04% |
| DDoS-TCP | 60s | 105 | 0.0% | 0.0% | 0.0% | 60.95% |
| Mirai | 5s | 278 | 14.3% | 22.2% | 17.4% | 9.23% |
| Mirai | 10s | 229 | 16.7% | 7.7% | 10.5% | 2.31% |
| Mirai | 30s | 148 | 12.5% | 10.0% | 11.1% | 5.07% |
| Mirai | 60s | 109 | 5.3% | 33.3% | 9.1% | 54.00% |
| Recon | 5s | 486 | 14.3% | 1.9% | 3.4% | 1.38% |
| Recon | 10s | 367 | 38.1% | **45.3%** | **41.4%** | 12.42% |
| Recon | 30s | 253 | 18.8% | 33.3% | 24.0% | 39.20% |
| Recon | 60s | 196 | 21.3% | 31.4% | 25.4% | 40.69% |

### 4.3 Analise S3

**Melhorias dramaticas de Recall:**
- **DDoS-SYN**: 3.5% -> 53.9% (@30s) — melhoria de 15x
- **Recon-PortScan**: 4.5% -> 45.3% (@10s) — melhoria de 10x
- **Mirai**: 1.7% -> 33.3% (@60s) — melhoria de 20x

**Problemas criticos:**
1. **FPR explode com janelas grandes:** 58% @60s em trafego benigno. Inaceitavel.
2. **Trade-off FPR/Recall severo:** Configs com bom Recall (SYN@30s=54%) tem FPR alto (33%).
3. **DDoS-ICMP regride para 0%:** Paradoxalmente, o ataque melhor detectado em S1 (27%) torna-se indetectavel com janelas. Os flows ICMP agregados por IP produzem vetores similares ao benigno.
4. **DDoS-TCP permanece em 0%** em todas as janelas.
5. **Poucos vetores:** Janelas grandes comprimem ~3000-10000 flows para ~100-260 vetores. MicroTEDAclus nao forma clusters estaveis com tao poucos pontos.

**Melhor resultado balanceado:** Recon @10s — F1=41.4%, FPR=12.4%. Unica configuracao com F1 > 20% e FPR < 15%.

### 4.4 Diagnostico: Por que FPR Explode?

As 12 features de janela sao agregados basicos: `flow_count`, `total_packets`, `total_bytes`, `unique_dst_ips`, `unique_dst_ports`, `mean_*`, `std_*`, `syn_ratio`, `mean_iat`.

Um IP benigno com 20 flows/janela e um atacante com 20 flows/janela produzem vetores **similares** nessas features. Faltam features que capturem o **comportamento**:
- Entropia das portas de destino (PortScan = alta, DDoS = 0)
- Entropia dos IPs de destino (DDoS = 0, benigno = variado)
- Taxa de flows sem resposta (SYN flood)
- Proporcao de flows unidirecionais
- Variabilidade do payload

**Proxima acao:** Campaign-03 S4 — 7 features comportamentais (entropy, ratios, rates).

---

## 5. Comparacao Cross-Step (best config por ataque, r0=0.10)

| Ataque | Best Step | Config | Recall | Precision | F1 | FPR |
|--------|-----------|--------|--------|-----------|-----|-----|
| DDoS-ICMP | S1 | v1, flow-level | 27.2% | 17.6% | 21.4% | 3.6% |
| DDoS-SYN | S3 | window 30s | 53.9% | 13.0% | 20.9% | 32.6% |
| DDoS-TCP | — | indetectavel | 0.0% | 0.0% | 0.0% | ~3% |
| Mirai | S3 | window 60s | 33.3% | 5.3% | 9.1% | 54.0% |
| Recon | S3 | window 10s | 45.3% | 38.1% | 41.4% | 12.4% |

---

## 6. Conclusoes e Proximos Passos

### 6.1 Achados Principais

1. **IP GT (S1) corrige medicao mas nao resolve deteccao.** Unica excecao: DDoS-ICMP melhora de 4% para 27% — o modo phase subestimava o Recall.

2. **Features per-flow saturaram (S2).** v1=v2=v3 — adicionar mais features estatisticas na granularidade de flow nao ajuda. O problema e estrutural.

3. **Janelas temporais sao a direcao certa (S3)** mas a implementacao com features basicas e insuficiente. Recall melhora dramaticamente (ate 54%) mas FPR acompanha.

4. **O detector precisa de features comportamentais** que capturem *como* os flows se distribuem, nao apenas *quanto*.

### 6.2 Configuracao Cumulativa para Campaign-03

| Parametro | Valor | Justificativa |
|-----------|-------|---------------|
| Algoritmo | MicroTEDAclus | C01 — superior ao TEDA |
| Ground truth | IP | C02-S1 — elimina limitacoes L1-L4 |
| Features per-flow | v1 (17) | C02-S2 — Occam, v2/v3 sem beneficio |
| Granularidade | Window | C02-S3 — melhora Recall significativamente |
| Window features | v1 (12 basic) -> **testar v2 (19 behavioral)** | C03-S4 |

### 6.3 Proximos Passos (Campaign-03)

- **S4:** 7 features comportamentais (entropia, ratios, taxas) — hipotese: melhoram separabilidade sem explodir FPR
- **S5:** Two-Stage Detection (se S4 insuficiente)
- **S6:** Threshold adaptativo (refinamento final)

---

## 7. Artefatos

### Estrutura de resultados
```
experiments/results/campaign-02/
  S1-A1-benign-r0_{0.05..0.20}/        # 4 runs
  S1-A2-{ddos,syn,tcp,mirai,recon}-*/   # 12 runs
  S2-A1-benign-{v2,v3}-r0_{0.05..0.20}/ # 8 runs
  S2-A2-{ataque}-{v2,v3}-r0_{0.10,0.15}/ # 20 runs
  S3-A1-benign-w{5,10,30,60}s-*/        # 4 runs
  S3-A2-{ataque}-w{5,10,30,60}s-*/      # 20 runs
  generate_plots.py
  plots/
  ANALYSIS.md                            # Este documento
```

**Total: 72 experimentos.**

### Reproducao
```bash
cd experiments/streaming && source venv/bin/activate

# S1 exemplo (DDoS-ICMP, r0=0.10)
python scripts/run_experiment.py \
  --pcap ../../data/pcaps/Benign_Final/BenignTraffic.pcap \
  --attack-pcap ../../data/pcaps/DDoS-ICMP_Flood/DDoS-ICMP_Flood.pcap \
  --max-packets 50000 --max-packets-attack 50000 --max-flows 10000 \
  --algorithm micro_teda --r0 0.10 --ground-truth ip \
  --output ../../experiments/results/campaign-02/S1-A2-ddos-r0_0.10/

# S3 exemplo (Recon, window 10s)
python scripts/run_experiment.py \
  --pcap ../../data/pcaps/Benign_Final/BenignTraffic.pcap \
  --attack-pcap ../../data/pcaps/Recon-PortScan/Recon-PortScan.pcap \
  --max-packets 50000 --max-packets-attack 50000 --max-flows 10000 \
  --algorithm micro_teda --r0 0.10 --ground-truth ip \
  --granularity window --window-seconds 10 \
  --output ../../experiments/results/campaign-02/S3-A2-recon-w10s-r0_0.10/
```

---

## 8. Parametros do Ambiente

| Parametro | Valor |
|-----------|-------|
| Git commit | d823fbf |
| Python | 3.x |
| Kafka | Confluent (Docker) |
| SO | Linux 6.8.0-101-generic |
| Sincronizacao | wait_for_flow_consumer |
| Flow timeout | 60s (event time) |
| Detector idle timeout | 10s |
| Prequential window | 1000 flows |
| Prequential alpha | 0.01 |
