# Reunião com Orientador — Progresso Experimental Fase 2

**Data:** 19 de março de 2026
**Aluno:** Augusto
**Programa:** Mestrado PPGEE — UFMG
**Tema:** Detecção de Intrusão em IoT com Clustering Evolutivo em Streaming

---

## 1. Situação Atual

### Fase 1 — Completa (Referência)

A Fase 1 do trabalho avaliou **705 experimentos** com algoritmos de ML supervisionados (Random Forest, XGBoost, etc.) sobre o dataset CICIoT2023, atingindo **F1 > 0.99**. Os resultados foram publicados em artigo de workshop. Esta fase serve como baseline de referência — o "teto" de desempenho com abordagem supervisionada batch.

### Fase 2 — Pipeline Streaming (Foco Atual)

A Fase 2 investiga detecção de intrusão **não-supervisionada em streaming**, usando:

- **Apache Kafka** como backbone de streaming (ingestão de PCAPs → tópicos Kafka)
- **FlowConsumer** para reconstrução de flows em tempo real (timeout de 60s, extração de 17 features por flow)
- **MicroTEDAclus** como detector de anomalias (clustering evolutivo baseado em tipicalidade — Maia 2020)

O **MicroTEDAclus** é um algoritmo de clustering evolutivo que:
- Mantém micro-clusters com estatísticas suficientes (média, variância, contagem)
- Classifica cada novo ponto como **normal** (absorvido por cluster existente) ou **anômalo** (cria novo cluster ou é outlier)
- Não requer número de clusters *a priori* e adapta-se a concept drift
- Usa a distância de tipicalidade (TEDA — Angelov 2014) como critério de pertencimento

### Dataset e Avaliação

- **Dataset:** CICIoT2023 — PCAPs de tráfego IoT real (33 tipos de ataque)
- **Ataques testados:** DDoS-ICMP Flood, DDoS-SYN Flood, DDoS-TCP Flood, Mirai-greeth, Recon-PortScan
- **Avaliação:** Prequential (test-then-train) — cada flow/janela é primeiro avaliado e depois usado para treino
- **Ground truth:** Por IP de atacante (documentado no CICIoT2023), eliminando vieses de rotulação por fase temporal

---

## 2. Metodologia Experimental

### Ablation Study com Configuração Cumulativa

A investigação segue uma metodologia de **ablation study**: cada step altera **uma única variável**, mantendo as demais congeladas. Ao final de cada step, a melhor decisão é **congelada** e carregada para o próximo:

```
C01 (Baseline)
 └─ Decisão: MicroTEDAclus > TEDA (26x mais detecções)

C02-S1 (Ground Truth)
 ├─ Variável: phase GT → IP GT
 └─ Decisão: Adotar IP GT (corrige vieses L1-L4)

C02-S2 (Features per-flow)
 ├─ Variável: v1(17) vs v2(25) vs v3(32) features
 └─ Decisão: Manter v1 (Occam — zero impacto de mais features)

C02-S3 (Granularidade)
 ├─ Variável: per-flow → window (5s, 10s, 30s, 60s)
 └─ Decisão: Window é direção certa (Recall 15-20x melhor)

C03-S4 (Features de Janela)
 ├─ Variável: v1(12 básicas) vs v2(19 comportamentais)
 └─ Resultado: Misto — v2 ajuda em 2/5 ataques, piora em 2/5
```

### Volume Experimental

| Campanha | Steps | Runs | Período |
|----------|-------|------|---------|
| Campaign-01 | Baseline | 17 | Mar 10-12 |
| Campaign-02 | S1, S2, S3 | 72 | Mar 14-16 |
| Campaign-03 | S4 | 48 | Mar 18 |
| **Total** | **4 steps** | **137** | **9 dias** |

---

## 3. Resultados por Campanha

### 3.1 Campaign-01 — Baseline (17 runs)

**Objetivo:** Avaliar MicroTEDAclus na configuração padrão (per-flow, 17 features, GT por fase).

| Métrica | Resultado | Alvo | Status |
|---------|-----------|------|--------|
| FPR benigno | ~3.5% | ≤ 5% | ✅ Aprovado |
| Recall ataques | ~3-4% (todos) | ≥ 80% | ❌ Reprovado |
| TEDA vs MicroTEDAclus | MicroTEDAclus 26x melhor | — | MicroTEDAclus adotado |

**Achado principal:** A anomaly rate é **invariante** (~3.5%) com ou sem ataque. O detector identifica outliers estatísticos naturais do tráfego, não ataques. Flows de ataque DDoS são estatisticamente indistinguíveis de flows benignos IoT (ambos são curtos, pequenos, regulares).

**Diagnóstico:** O problema é de **representação** (features per-flow insuficientes), não de algoritmo.

### 3.2 Campaign-02 — 3 Hipóteses (72 runs)

| Step | Variável Testada | Resultado | Decisão |
|------|-----------------|-----------|---------|
| **S1** — Ground Truth | phase → IP | DDoS-ICMP: 4% → 27% Recall. Resto inalterado. FPR estável ~3-4%. | Adotar IP GT |
| **S2** — Features per-flow | v1(17) → v2(25) → v3(32) | **Zero impacto** (±1pp em todos os ataques). v1 ≈ v2 ≈ v3. | Manter v1 (Occam) |
| **S3** — Granularidade | flow → window (5-60s) | SYN: 3% → 54%. Recon: 4% → 45%. **MAS FPR explode** (58% @60s). | Direção certa, features insuficientes |

**S1 — Ground Truth por IP:**
A mudança de ground truth revelou que a C01 **subestimava** o Recall do DDoS-ICMP: de 4% para 27%. Para os demais ataques, o IP GT corrige os labels mas não muda a capacidade de detecção — confirma que o problema é de representação.

**S2 — Features per-flow saturaram:**
Adicionar features como `flow_duration`, `packet_size_min/max`, `fwd/bwd_packet_size_mean`, `iat_min/max`, `psh_count` e features direcionais **não melhora a separabilidade**. O problema é estrutural: na granularidade per-flow, flows de ataque e benignos ocupam a mesma região do espaço de features.

**S3 — Janelas temporais mudam o jogo:**
A detecção por janela temporal muda a pergunta fundamental:
- Per-flow: "Este flow individual é anômalo?"
- Per-window: "Este IP tem comportamento anômalo nos últimos N segundos?"

Resultados por janela (r0=0.10):

| Ataque | Per-flow Recall | Best Window Recall | Melhoria |
|--------|----------------|-------------------|----------|
| DDoS-SYN | 3.5% | 53.9% (@30s) | 15x |
| Recon | 4.5% | 45.3% (@10s) | 10x |
| Mirai | 1.7% | 33.3% (@60s) | 20x |
| DDoS-ICMP | 27.2% | 0% (regride) | — |
| DDoS-TCP | 0% | 0% | — |

**Problema:** O FPR acompanha a melhoria do Recall. Em w=60s, o FPR benigno atinge 58% — inaceitável. As 12 features de janela (contagens, médias, somas) não capturam **comportamento** — faltam features como entropia de portas/IPs.

### 3.3 Campaign-03 S4 — Features Comportamentais (48 runs)

**Hipótese:** 7 features comportamentais (entropias, ratios, taxas) melhoram a separabilidade sem explodir o FPR.

**Features v2 (19 total = 12 básicas + 7 comportamentais):**
- `dst_port_entropy` — entropia das portas de destino
- `dst_ip_entropy` — entropia dos IPs de destino
- `flows_per_second` — taxa de flows por segundo
- `unanswered_ratio` — proporção de flows sem resposta
- `fwd_only_ratio` — proporção de flows unidirecionais
- `small_packet_ratio` — proporção de flows com pacotes pequenos
- `syn_only_ratio` — proporção de flows SYN-only

**Comparação v1 vs v2 @ w=10s (r0=0.10):**

| Ataque | v1 Recall | v2 Recall | v1 F1 | v2 F1 | Veredicto |
|--------|-----------|-----------|-------|-------|-----------|
| DDoS-ICMP | 0.0% | **50.0%** | 0.0% | 5.6% | v2 desbloqueia detecção |
| DDoS-SYN | **38.5%** | 30.8% | **20.0%** | 17.8% | v1 melhor |
| DDoS-TCP | 0.0% | 0.0% | 0.0% | 0.0% | Ambos falham |
| Mirai | **46.2%** | 38.5% | **23.1%** | 21.7% | v1 melhor |
| Recon | 39.2% | **45.5%** | 35.7% | **39.1%** | v2 melhor |

**FPR benigno:** v1 = 2.9% → v2 = 14.3% (piora 5x em w=10s)

**Conclusão S4:** As features comportamentais **não produzem melhoria consistente**. Desbloqueiam DDoS-ICMP (0%→50%) e melhoram Recon (39%→45%), mas degradam Mirai e SYN. O FPR piora significativamente.

---

## 4. Tabela Consolidada — Melhor Resultado por Ataque

Considerando todas as campanhas e configurações testadas:

| Ataque | Campanha | Config | Recall | F1 | FPR |
|--------|----------|--------|--------|-----|-----|
| DDoS-ICMP | C03-S4 | v2 / w10s / r0=0.10 | 50.0% | 5.6% | 15.7% |
| DDoS-SYN | C03-S4 | v2 / w30s / r0=0.05 | 61.5% | 21.6% | 36.1% |
| DDoS-TCP | — | Indetectável (todas configs) | 0.0% | 0.0% | — |
| Mirai | C03-S4 | v1 / w10s / r0=0.10 | 46.2% | 23.1% | 15.5% |
| Recon | C03-S4 | v2 / w10s / r0=0.05 | 49.1% | 43.7% | 12.9% |
| Benigno (FPR) | C02-S1 | flow-level / r0=0.10 | — | — | 3.5% |

**Destaques:**
- **Recon F1=43.7%** é o melhor resultado não-supervisionado da dissertação
- **Recon com r0=0.15** atinge Precision de 56.7% com FPR de apenas 4.2% (F1=42.0%)
- **Não existe configuração única ótima** — a melhor config depende do tipo de ataque

---

## 5. Insights e Discussão

### 1. Detecção per-flow é fundamentalmente limitada

Flows individuais de ataque DDoS são estatisticamente indistinguíveis de flows benignos IoT. Ambos são curtos, com poucos pacotes, IAT regular. Adicionar mais features per-flow (de 17 para 32) não ajuda — o problema é estrutural, não de dimensionalidade.

### 2. Janelas temporais são a direção certa

A mudança de granularidade de per-flow para per-IP/janela temporal transforma a pergunta do detector:
- **Per-flow:** "Este flow individual é diferente?" → Não, flows de ataque são "normais"
- **Per-window:** "Este IP tem comportamento anômalo?" → Sim, padrão de scanning/flooding emerge

Essa mudança produziu melhorias de **10-20x no Recall** para SYN, Recon e Mirai.

### 3. Curse of dimensionality

Com janelas de 10s e min_flows=5, o dataset comprime para ~210 vetores. Destes, apenas 2-55 são de ataque. O MicroTEDAclus com 19 features em 210 pontos está em regime de **alta dimensionalidade / poucos dados**. Os clusters não convergem, gerando instabilidade no FPR.

### 4. Não existe configuração única ótima

Cada tipo de ataque responde de forma diferente às variáveis do detector:
- **DDoS-ICMP** precisa de v2 (features comportamentais)
- **Mirai e SYN** funcionam melhor com v1 (features básicas)
- **Recon** é o mais tratável em ambas as configurações
- **DDoS-TCP** é indistinguível em qualquer configuração

### 5. Resultado positivo: Recon F1=43.7%

O melhor resultado da dissertação em detecção não-supervisionada:
- **Config:** v2/w10s/r0=0.05
- **Recall 49.1%**, Precision 39.4%, **F1 43.7%**, FPR 12.9%
- Comparável com resultados publicados de IDS não-supervisionados na literatura
- Demonstra que o pipeline Kafka → MicroTEDAclus **funciona** para certos tipos de ataque

---

## 6. Próximos Passos — 3 Opções para Discussão

### Opção A: Mais um round experimental (S5) + escrita (~2 + 5 semanas)

**S5 — Two-Stage Detection:**
- Stage 1: Detecção per-flow (FPR baixo, ~3.5%)
- Stage 2: Concentração de anomalias por IP em janela temporal
- Um IP é declarado malicioso se a **taxa de anomalias** dos seus flows excede um threshold

**Vantagens:**
- Ataca diretamente o trade-off FPR/Recall
- Stage 1 mantém FPR baixo, Stage 2 amplifica o sinal
- Fundamentação teórica sólida (ensemble de evidências)

**Riscos:**
- Pode não funcionar — flows de ataque são "normais" no Stage 1
- Perde 2 semanas se não der resultado

### Opção B: Consolidar e escrever (~6-7 semanas)

**Foco:** Documentar os resultados atuais como contribuição válida.

**Argumento:**
- 137 experimentos com ablation study rigoroso é contribuição metodológica
- Resultados negativos documentados são valiosos (onde/por que IDS por anomalia falha)
- Recon F1=43.7% é resultado positivo demonstrável
- Mais tempo para escrita + revisão do orientador

**Vantagens:**
- Menos risco de prazo
- Resultados atuais já sustentam uma dissertação

**Desvantagens:**
- Menos ambicioso — "mostramos que não funciona bem" é contribuição mais fraca

### Opção C: S5 + S6 (threshold adaptativo) + escrita (~3 + 4 semanas)

**S5:** Two-Stage Detection (como Opção A)
**S6:** Threshold adaptativo — ajustar r0 ou threshold de janela por tipo de tráfego

**Vantagens:**
- Maximiza profundidade experimental
- Se S5 funcionar, a dissertação tem resultado forte

**Riscos:**
- 3 semanas de experimentos = apenas 4 semanas para escrita
- Defesa em ~maio 2026 — margem apertada

---

## 7. Cronograma até Defesa

```
Semana  Data         Opção A              Opção B            Opção C
──────  ──────────   ──────────────────   ────────────────   ──────────────────
S5      Mar 23-29    S5 experimentos      Cap. Metodologia   S5 experimentos
S6      Mar 30-Abr5  S5 análise           Cap. Resultados    S6 experimentos
S7      Abr 6-12     Cap. Metodologia     Cap. Discussão     S6 análise
S8      Abr 13-19    Cap. Resultados      Revisão orient.    Cap. Metodologia
S9      Abr 20-26    Cap. Discussão       Ajustes finais     Cap. Resultados
S10     Abr 27-Mai3  Revisão orient.      Preparação defesa  Cap. Discussão
S11     Mai 4-10     Ajustes finais       Defesa ←           Revisão orient.
S12     Mai 11-17    Defesa ←                                Defesa ←
```

**Recomendação do aluno:** Opção A — equilíbrio entre profundidade e segurança. S5 (Two-Stage) é uma abordagem promissora e relativamente simples de implementar (~1 semana). Se não funcionar, descarta e segue para escrita (efetivamente vira Opção B com 1 semana a menos).

---

*Documento preparado para reunião com orientador. Dados extraídos dos arquivos ANALYSIS.md das campanhas C01, C02 e C03.*
