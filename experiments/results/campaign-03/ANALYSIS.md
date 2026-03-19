# Campaign-03 S4 — Analise de Resultados

**Data:** 2026-03-18
**Step:** S4 — Behavioral Window Features (v2 = 19 features)
**Dataset:** CICIoT2023 (Benign_Final + 5 ataques)
**Pacotes por PCAP:** 50k benigno + 50k ataque | **Max flows:** 10.000
**Total de runs:** 48 (12 controle v1 + 12 v2 + 24 r0 sweep)

---

## 1. Resumo Executivo

| Criterio | Alvo | Resultado | Status |
|----------|------|-----------|--------|
| FPR (benign, v2) | <= FPR(v1) no mesmo window | v2 FPR **pior** que v1 em w=10s (14.3% vs 2.9%). Comparavel em w=30s (6.7% vs 5.0%) | REPROVADO (w=10s), MARGINAL (w=30s) |
| Recall SYN (v2 > v1) | Melhoria sobre v1 | v2 **pior** que v1 em w=10s (30.8% vs 38.5%). Ambos 0% em w=30s | REPROVADO |
| Recall Recon (v2 > v1) | Melhoria sobre v1 | v2 **melhor** em w=10s (45.5% vs 39.2%). v2 pior em w=30s | APROVADO (w=10s) |
| F1 v2 > v1 em >=3/5 ataques | Melhoria consistente | v2 melhor em 2/5 (DDoS-ICMP, Recon @w10s), pior em 2/5 (Mirai, SYN), igual em 1/5 (TCP=0) | REPROVADO |

**Conclusao principal:** As features comportamentais v2 **nao produzem melhoria consistente** sobre v1. Desbloqueiam deteccao de DDoS-ICMP (0%->50% Recall) mas degradam Mirai e SYN. O FPR em trafego benigno piora significativamente com w=10s. A hipotese de que features de entropia/ratio resolveriam o problema foi **parcialmente refutada**.

**Resultado positivo:** Recon-PortScan com v2/w10s/r0=0.05 atinge o melhor F1 da dissertacao ate agora: **43.7%** (Recall 49%, Precision 39%). Melhoria de 5% sobre v1.

---

## 2. Comparacao v1 vs v2 (r0=0.10)

### 2.1 Window = 10s

| Ataque | v1 Recall | v2 Recall | v1 F1 | v2 F1 | v1 FPR | v2 FPR | Veredicto |
|--------|-----------|-----------|-------|-------|--------|--------|-----------|
| DDoS-ICMP | 0.0% | **50.0%** | 0.0% | 5.6% | 13.6% | 15.7% | v2 desbloqueia deteccao |
| DDoS-SYN | **38.5%** | 30.8% | **20.0%** | 17.8% | 14.8% | **12.5%** | v1 melhor |
| DDoS-TCP | 0.0% | 0.0% | 0.0% | 0.0% | 11.3% | **2.4%** | Ambos falham (v2 menor FPR) |
| Mirai | **46.2%** | 38.5% | **23.1%** | 21.7% | 15.5% | **13.2%** | v1 melhor Recall, v2 melhor FPR |
| Recon | 39.2% | **45.5%** | 35.7% | **39.1%** | **13.1%** | 15.4% | v2 melhor deteccao |

### 2.2 Window = 30s

| Ataque | v1 Recall | v2 Recall | v1 F1 | v2 F1 | v1 FPR | v2 FPR |
|--------|-----------|-----------|-------|-------|--------|--------|
| DDoS-ICMP | 0.0% | 0.0% | 0.0% | 0.0% | 4.3% | 5.1% |
| DDoS-SYN | 0.0% | 0.0% | 0.0% | 0.0% | 4.1% | 4.2% |
| DDoS-TCP | 0.0% | 0.0% | 0.0% | 0.0% | 5.1% | 5.8% |
| Mirai | 10.0% | 10.0% | 10.5% | 11.8% | 5.8% | **4.3%** |
| Recon | **33.3%** | 19.6% | **29.5%** | 18.3% | 26.8% | 26.9% |

**Window 30s e amplamente inferior** — a maioria dos ataques tem 0% Recall. A agregacao longa suaviza as assinaturas de ataque. O unico resultado util em w=30s e Recon com v1 (33.3%).

### 2.3 FPR em Trafego Benigno (A1)

| Config | FPR |
|--------|-----|
| v1/w10s/r0=0.10 | **2.86%** |
| v2/w10s/r0=0.10 | 14.29% |
| v1/w30s/r0=0.10 | 5.04% |
| v2/w30s/r0=0.10 | 6.67% |

v2 **quintuplica** o FPR em w=10s (2.9% -> 14.3%). Em w=30s a degradacao e menor (5.0% -> 6.7%). As features comportamentais adicionam ruido que gera mais falsos positivos.

---

## 3. r0 Sweep (v2)

### 3.1 Efeito de r0 no FPR Benigno

| Window | r0=0.05 | r0=0.10 | r0=0.15 |
|--------|---------|---------|---------|
| 10s | 13.81% | 14.29% | 15.31% |
| 30s | 5.07% | 6.67% | 4.23% |

r0 tem **impacto minimo** no FPR com v2. Em w=10s o FPR permanece alto (~14-15%) independente de r0. Em w=30s flutua entre 4-7%.

### 3.2 Melhores Resultados por Ataque (v2, todos r0)

| Ataque | Melhor Config | Recall | Precision | F1 | FPR |
|--------|--------------|--------|-----------|-----|-----|
| DDoS-ICMP | w10s/r0=0.05 | 50.0% | 3.3% | 6.3% | 13.7% |
| DDoS-ICMP | w10s/r0=0.10 | 50.0% | 2.9% | 5.6% | 15.7% |
| DDoS-SYN | w10s/r0=0.10 | 30.8% | 12.5% | 17.8% | 12.5% |
| DDoS-SYN | w30s/r0=0.05 | **61.5%** | 13.1% | 21.6% | 36.1% |
| Mirai | w10s/r0=0.10 | 38.5% | 15.2% | 21.7% | 13.2% |
| Recon | w10s/r0=0.05 | **49.1%** | 39.4% | **43.7%** | 12.9% |
| Recon | w10s/r0=0.15 | 33.3% | **56.7%** | 42.0% | **4.2%** |

**Destaque:** Recon com r0=0.15/w10s atinge Precision de 56.7% com FPR de apenas 4.2% — a melhor combinacao FPR/Precision da dissertacao, com F1=42.0%.

---

## 4. Analise: Por que v2 nao Resolve?

### 4.1 O que v2 Faz Bem

1. **Desbloqueia DDoS-ICMP:** De 0% (v1) para 50% Recall. A feature `flows_per_second` ou `dst_ip_entropy` captura o padrao de flood ICMP que v1 nao distinguia.

2. **Melhora Recon marginalmente:** De 39% para 45% Recall (@w10s). Entropia de portas (`dst_port_entropy`) provavelmente ajuda a identificar PortScan.

3. **Reduz FPR para TCP em w=10s:** De 11.3% para 2.4%. As features v2 tornam o detector mais conservador para TCP.

### 4.2 O que v2 Piora

1. **FPR benigno explode em w=10s:** 2.9% -> 14.3%. As 7 features adicionais criam mais dimensoes onde IPs benignos podem parecer anomalos. O MicroTEDAclus com 19 features e poucos vetores (~210) nao forma clusters estaveis.

2. **Mirai e SYN regridem:** Features como `unanswered_ratio` e `fwd_only_ratio` deveriam ajudar com SYN flood mas nao se traduzem em melhoria. Possivel causa: com apenas ~13 vetores de ataque, o sinal se perde.

3. **DDoS-ICMP: alto Recall mas Precision de 3%.** O detector flagra 50% dos ataques mas tambem flagra muitos benignos. F1 e apenas 5.6%.

### 4.3 Problema Fundamental: Poucos Vetores

Com janelas de 10s e min_flows=5, o dataset comprime para ~210-370 vetores. Destes, apenas ~2-55 sao de ataque. O MicroTEDAclus com 19 features em 210 vetores esta em regime de alta dimensionalidade / poucos dados — a maldição da dimensionalidade.

| Config | Vetores Totais | Vetores Ataque (tipico) |
|--------|---------------|------------------------|
| w=10s | 210-370 | 2-55 |
| w=30s | 135-255 | 5-55 |

### 4.4 Comparacao com C02-S3

| Ataque | S3 Best (v1) | S4-v1 Best | S4-v2 Best | Tendencia |
|--------|-------------|------------|------------|-----------|
| DDoS-ICMP | 0% (@10s) | 0% | 50% (@10s) | v2 ajuda |
| DDoS-SYN | 53.9% (@30s) | 38.5% (@10s) | 61.5% (@30s,r0=0.05) | r0 pequeno ajuda |
| DDoS-TCP | 0% | 0% | 0% | Indetectavel |
| Mirai | 33.3% (@60s) | 46.2% (@10s) | 38.5% (@10s) | v1 melhor |
| Recon | 45.3% (@10s) | 39.2% (@10s) | 49.1% (@10s,r0=0.05) | v2+r0 ajuda |

---

## 5. Decisoes e Proximos Passos

### 5.1 Configuracao Cumulativa "Best So Far"

Nao ha uma configuracao unica otima. Resultados dependem do tipo de ataque:

**Para Recon (melhor resultado global):**
- v2/w10s/r0=0.05: F1=43.7%, Recall=49.1%, Precision=39.4%, FPR=12.9%
- v2/w10s/r0=0.15: F1=42.0%, Recall=33.3%, Precision=56.7%, FPR=4.2%

**Para DDoS-SYN:**
- v2/w30s/r0=0.05: Recall=61.5%, F1=21.6%, FPR=36.1% (FPR inaceitavel)
- v1/w10s/r0=0.10: Recall=38.5%, F1=20.0%, FPR=14.8%

**Para DDoS-ICMP:**
- v2/w10s: Recall=50%, F1=5.6% (Precision muito baixa)

### 5.2 Veredicto sobre S5 (Two-Stage Detection)

Os resultados de S4 indicam que:
1. **Features comportamentais ajudam para alguns ataques** (ICMP, Recon) mas nao para outros
2. **O FPR permanece problematico** — mesmo v1 com w=10s tem FPR de ~3-15%
3. **O problema fundamental e poucos vetores** em regime de alta dimensionalidade

**Recomendacao:** S5 (Two-Stage) pode ser util como mecanismo de **reducao de FPR** — Stage 1 per-flow identifica candidatos, Stage 2 concentracao por IP confirma. Isso atacaria diretamente o trade-off FPR/Recall.

### 5.3 Proximos Passos

1. **Documentar resultados na dissertacao** — S4 e resultado misto, nao negativo
2. **Avaliar S5** — Two-Stage Detection focado em reducao de FPR
3. **Consolidar best configs** por tipo de ataque para tabela final

---

## 6. Artefatos

### Estrutura de resultados
```
experiments/results/campaign-03/
  S4-A1-benign-wf{v1,v2}-w{10,30}s-r0_{0.05..0.15}/   # 8 runs
  S4-A2-{ddos,syn,tcp,mirai,recon}-wf{v1,v2}-w{10,30}s-r0_{0.05..0.15}/  # 40 runs
  generate_plots_s4.py
  plots/
    01_s4_recall_v1_vs_v2_w10s.png
    01_s4_recall_v1_vs_v2_w30s.png
    02_s4_fpr_v1_vs_v2.png
    03_s4_f1_v1_vs_v2_w10s.png
    03_s4_f1_v1_vs_v2_w30s.png
    04_s4_fpr_vs_r0_v2.png
    05_s4_dashboard_s3_vs_s4.png
  ANALYSIS.md
```

**Total: 48 experimentos.**

### Reproducao
```bash
cd experiments/streaming && source venv/bin/activate
bash scripts/run_campaign03_s4.sh
python experiments/results/campaign-03/generate_plots_s4.py
```

---

## 7. Parametros do Ambiente

| Parametro | Valor |
|-----------|-------|
| Git commit | 1f2d2cf |
| Python | 3.x |
| Kafka | Confluent (Docker) |
| SO | Linux 6.8.0-101-generic |
| Flow timeout | 60s (event time) |
| Detector idle timeout | 10s |
| Prequential window | 1000 flows |
| Prequential alpha | 0.01 |
| Tempo total | 66m 5s (48 runs) |
