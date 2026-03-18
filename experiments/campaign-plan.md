# Plano de Campanha Experimental

**Criado:** 2026-03-08
**Prazo defesa:** ~maio 2026 (~8 semanas)
**Metodologia:** ver `experiments/methodology.md`

---

## Objetivo

Produzir evidencias experimentais suficientes para 3 contribuicoes da dissertacao:

1. **Deteccao funciona** — MicroTEDAclus detecta ataques IoT em streaming
2. **Adaptacao a drift** — sistema se adapta a mudancas abruptas de padrao
3. **Generalizacao** — sistema detecta ataques nunca vistos (zero-day)

---

## Cenarios Priorizados

| Prioridade | Cenario | O que testa | Semana |
|------------|---------|-------------|--------|
| **Obrigatorio** | A - Deteccao Basica | Sistema detecta ataques? | S2 |
| **Obrigatorio** | B - Drift Subito | Adapta a mudanca abrupta? | S3 |
| **Obrigatorio** | E - Zero-Day | Detecta ataques nunca vistos? | S3-S4 |
| Se der tempo | D - Drift Recorrente | "Lembra" padroes anteriores? | S4 |
| **Cortado** | C - Drift Gradual | Complexo, valor marginal vs B | — |

---

## Semana 2: Validacao de Arquitetura + Cenario A

### Objetivo
Confirmar que o pipeline funciona end-to-end e produzir primeiros resultados publicaveis.

### Pre-requisitos
- [ ] Testes do streaming passando (97 passed — OK)
- [ ] Docker Kafka operacional
- [ ] PCAPs disponiveis: Benign, DDoS-ICMP_Flood

### Experimentos

**A1: Baseline benigno (falsos positivos)**
```bash
cd experiments/streaming && source venv/bin/activate
python3 scripts/run_experiment.py \
  --pcap ../../data/pcaps/Benign_Final/BenignTraffic.pcap \
  --max-packets 100000 --max-flows 10000 \
  --algorithm micro_teda --r0 0.10 \
  --output ../results/campaign-01/A1-benign/
```
- Criterio de sucesso: FPR <= 5%
- Repetir com r0 = {0.05, 0.10, 0.15, 0.20} para calibrar

**A2: Deteccao DDoS**
```bash
python3 scripts/run_experiment.py \
  --pcap ../../data/pcaps/Benign_Final/BenignTraffic.pcap \
  --attack-pcap ../../data/pcaps/DDoS-ICMP_Flood/DDoS-ICMP_Flood.pcap \
  --max-packets 100000 --max-flows 10000 \
  --algorithm micro_teda --r0 0.10 \
  --output ../results/campaign-01/A2-ddos/
```
- Criterio de sucesso: Recall >= 80%, MTTD <= 500 flows
- Repetir com melhor r0 encontrado em A1

**A3: Comparacao TEDA vs MicroTEDAclus**
```bash
# Mesmo PCAP, mesmo protocolo, algoritmo diferente
python3 scripts/run_experiment.py \
  --pcap ../../data/pcaps/Benign_Final/BenignTraffic.pcap \
  --attack-pcap ../../data/pcaps/DDoS-ICMP_Flood/DDoS-ICMP_Flood.pcap \
  --max-packets 100000 --max-flows 10000 \
  --algorithm teda --r0 0.10 \
  --output ../results/campaign-01/A3-teda-baseline/
```
- Comparar: F1, FPR, MTTD, throughput, memoria

### Entregaveis S2
- [ ] Tabela comparativa TEDA vs MicroTEDAclus (Cenario A)
- [ ] Grafico temporal F1 (prequential) para ambos algoritmos
- [ ] r0 otimo calibrado
- [ ] Decisao: prosseguir com MicroTEDAclus ou ajustar

---

## Semana 3: Cenario B (Drift Subito) + Inicio Cenario E

### Pre-requisitos
- [ ] Cenario A completo com resultados satisfatorios
- [ ] PCAPs disponiveis: Benign, DDoS, Mirai (pelo menos 1 variante)

### Experimentos

**B1: Drift DDoS → Mirai**
```bash
# Fase 1: Treinar com DDoS
# Fase 2: Testar com Mirai (drift subito)
python3 scripts/run_experiment.py \
  --pcap ../../data/pcaps/Benign_Final/BenignTraffic.pcap \
  --attack-pcap ../../data/pcaps/DDoS-ICMP_Flood/DDoS-ICMP_Flood.pcap \
  --drift-pcap ../../data/pcaps/Mirai-greeth_flood/Mirai-greeth_flood.pcap \
  --max-flows 20000 \
  --algorithm micro_teda --r0 [otimo de A1] \
  --output ../results/campaign-01/B1-drift-ddos-mirai/
```
- Metricas: F1 antes/depois do drift, tempo de adaptacao (flows ate recuperar 95% F1)
- **NOTA:** pode ser necessario implementar `--drift-pcap` no orquestrador

**B2: Drift Mirai → Recon** (se houver PCAPs)

**E1: Zero-day com holdout**
```bash
# Treinar: Benign + DDoS + DoS
# Testar: Mirai + Recon (nunca vistos)
python3 scripts/run_experiment.py \
  --pcap ../../data/pcaps/Benign_Final/BenignTraffic.pcap \
  --attack-pcap ../../data/pcaps/DDoS-ICMP_Flood/DDoS-ICMP_Flood.pcap \
  --holdout-pcap ../../data/pcaps/Mirai-greeth_flood/Mirai-greeth_flood.pcap \
  --max-flows 20000 \
  --algorithm micro_teda --r0 [otimo] \
  --output ../results/campaign-01/E1-zero-day/
```
- Metricas: taxa de deteccao de zero-day, FPR
- **NOTA:** pode ser necessario implementar `--holdout-pcap` no orquestrador

### Entregaveis S3
- [ ] Grafico de adaptacao ao drift (F1 temporal com marcador de drift)
- [ ] Tempo de adaptacao medido
- [ ] Taxa de deteccao zero-day

---

## Semana 4: Finalizar E + Cenario D (se der tempo)

### Experimentos

**E2: Mais combinacoes de holdout**
- Treinar com {Benign + Mirai}, testar com {DDoS} — inverte
- Treinar com {Benign + DoS}, testar com {Spoofing}

**D1: Drift recorrente** (se der tempo)
```
Fase 1: DDoS (aprende)
Fase 2: Benign (periodo calmo)
Fase 3: DDoS (deve reconhecer mais rapido)
Fase 4: Benign
Fase 5: DDoS (deve reconhecer ainda mais rapido)
```
- Metrica: tempo de reconhecimento decresce a cada retorno?

### Entregaveis S4
- [ ] Tabela consolidada de todos os cenarios
- [ ] Analise estatistica (media +- desvio, 5 runs por config)
- [ ] Dados prontos para gerar figuras da dissertacao

---

## Semanas 5-6: Analise e Figuras

### Tarefas
- [ ] Gerar figuras para dissertacao (`writing/figures/`)
  - Grafico temporal F1 (cenario A)
  - Grafico de adaptacao ao drift (cenario B)
  - Tabela comparativa TEDA vs MicroTEDAclus
  - Grafico de deteccao zero-day (cenario E)
- [ ] Gerar tabelas (`writing/tables/`)
  - Tabela de parametros otimos
  - Tabela de resultados por cenario
  - Tabela comparativa com literatura
- [ ] Comparar resultados com papers da area (seção de trabalhos relacionados)
- [ ] Escrever capitulo de Resultados (cap. 5)

---

## Semanas 7-8: Dissertacao + Defesa

### Tarefas
- [ ] Finalizar capitulos 2-5
- [ ] Revisao geral
- [ ] Preparar apresentacao de defesa
- [ ] Rehearsal com orientador

---

## PCAPs Necessarios

| PCAP | Cenarios | Disponivel? |
|------|----------|-------------|
| Benign_Final/BenignTraffic.pcap | Todos | ✅ |
| DDoS-ICMP_Flood/DDoS-ICMP_Flood.pcap | A, B, D, E | ✅ |
| Mirai-greeth_flood/Mirai-greeth_flood.pcap | B, E | ✅ |
| DoS-TCP_Flood/DoS-TCP_Flood.pcap | E | ✅ |
| Recon-PortScan/Recon-PortScan.pcap | E | ✅ |
| DNS_Spoofing/DNS_Spoofing.pcap | E (opcional) | ✅ |

---

## Funcionalidades a Implementar no Orquestrador

O `run_experiment.py` atual suporta `--pcap` e `--attack-pcap`. Para os cenarios B e E:

1. **Multi-fase:** suporte a `--drift-pcap` (fase 3 do experimento)
2. **Holdout:** suporte a `--holdout-pcap` (ataques zero-day)
3. **Metricas de adaptacao:** tempo ate recuperar 95% do F1 baseline

Avaliar se implementar no orquestrador ou como scripts separados.

---

## Criterios de Sucesso Globais

| Metrica | Alvo | Cenario |
|---------|------|---------|
| FPR (trafego benigno) | <= 5% | A |
| Recall (ataques conhecidos) | >= 80% | A |
| MTTD | <= 500 flows | A |
| Throughput | >= 100 flows/s | Todos |
| Tempo de adaptacao (drift) | <= 2000 flows | B |
| Deteccao zero-day | >= 60% | E |
| Execucoes por config | >= 5 | Todos |

---

---

## Campaign-02: 3 Melhorias Incrementais

**Criado:** 2026-03-16
**Motivação:** Campaign-01 revelou Recall ~3-4% — flows de ataque indistinguíveis no espaço de 17 features. Problema de representação, não de algoritmo.

### Estratégia

3 melhorias incrementais, cada uma mudando **exatamente uma variável**:

| Variável | Campaign-01 | S1 | S2 | S3 |
|----------|-------------|-----|-----|-----|
| Ground truth | Phase | **IP** | IP | IP |
| Features | 17 (v1) | 17 (v1) | **25 (v2)** | 25 (v2) |
| Granularidade | Per-flow | Per-flow | Per-flow | **Per-window** |

### Step 1: Ground Truth por IP

Re-rodar A1+A2 com ground truth por IP para confirmar se Recall ~3-4% é real.

```
campaign-02/S1-A1-benign-r0_0.10/
campaign-02/S1-A2-ddos-{icmp,syn,tcp}-r0_0.10/
campaign-02/S1-A2-mirai-r0_0.10/
campaign-02/S1-A2-recon-r0_0.10/
```

### Step 2: Expansão de Features (v2=25, v3=32)

**v2 (25 features = v1 + 8 curadas):** flow_duration, packet_size_min/max, fwd/bwd_packet_size_mean, iat_min/max, psh_count
**v3 (32 features = v2 + 7):** fwd/bwd_packet_size_std, urg_count, fwd/bwd_iat_mean/std

```
campaign-02/S2-A1-benign-v2-r0_{0.05..0.30}/   # calibração
campaign-02/S2-A2-{attack}-v2-r0_<best>/         # 5 ataques
```

### Step 3: Detecção por Janela Temporal

Mudar unidade de detecção de flow individual para comportamento agregado por IP em janela temporal (WindowAggregator).

```
campaign-02/S3-A1-benign-window-{5s,10s,30s}/
campaign-02/S3-A2-{attack}-window-10s/
```

### Novos argumentos CLI (run_experiment.py)

| Argumento | Default | Descrição |
|-----------|---------|-----------|
| `--features {v1,v2,v3}` | v2 | Conjunto de features |
| `--granularity {flow,window}` | flow | Granularidade de detecção |
| `--window-seconds` | 10.0 | Tamanho da janela (modo window) |
| `--min-flows-per-window` | 5 | Mínimo de flows por IP/janela |

---

---

## Campaign-03: Features Comportamentais + Abordagens Evolutivas

**Criado:** 2026-03-16
**Motivação:** Campaign-02 S3 mostrou que janelas temporais melhoram Recall (SYN 3%→54%, Recon 4%→45%) mas FPR explode (58% @60s). Diagnóstico: 12 features básicas de janela (somas, médias) não capturam **comportamento** (entropia, taxas, assimetria). Um IP benigno e um atacante com 20 flows/janela produzem vetores similares.

### Configuração Cumulativa (frozen from C02)

| Decisão | Valor | Origem |
|---------|-------|--------|
| Algoritmo | MicroTEDAclus | C01 |
| Ground truth | IP-based | C02-S1 |
| Features per-flow | v1 (17) | C02-S2 (Occam — v2/v3 sem impacto) |
| Granularidade | Window | C02-S3 (direção certa) |
| Window features | **variável aberta** | C03-S4 |

### Step S4: Behavioral Window Features (v2 = 19 features)

7 novas features computadas por IP/janela:

| Feature | Discriminação |
|---------|--------------|
| `flows_per_second` | DDoS/scan gera 10-100x mais flows/s |
| `dst_port_entropy` | PortScan = alta entropia, DDoS = 0 |
| `dst_ip_entropy` | DDoS flood → alvo único (≈0) |
| `unanswered_ratio` | SYN flood: handshake nunca completa |
| `payload_std` | DDoS envia pacotes uniformes (std baixo) |
| `small_flow_ratio` | Scan/probe: muitos flows ≤3 pkts |
| `fwd_only_ratio` | DDoS/scan são unidirecionais |

**Variáveis:** `window_features={v1,v2}`, `window_seconds={10,30}`, `r0={0.05,0.10,0.15}`

**Runs:** 48 (12 control v1 + 12 features v2 + 24 r0 sweep v2)

**Script:** `experiments/streaming/scripts/run_campaign03_s4.sh`
**Plots:** `experiments/results/campaign-03/generate_plots_s4.py`

### Step S5: Two-Stage Detection (após S4)

Stage 1 = per-flow MicroTEDAclus. Stage 2 = monitorar concentração de anomalias por IP em janela.

### Step S6: Threshold Adaptativo (após S5)

r0 dinâmico baseado em percentil de eccentricidade ou desvio do cluster.

---

## Notas

- Resultados anteriores (`streaming/results/week5/`) sao descartaveis
- Campanha comeca do zero com metodologia rigorosa
- Cada experimento deve ter: run_meta.json, detection_results.json, metrics_windowed.csv
- Reprodutibilidade: registrar git commit, seeds, parametros completos
