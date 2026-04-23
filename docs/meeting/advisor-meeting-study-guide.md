# Guia de Estudo para Reunião com Orientador

> **Propósito:** Documento vivo de preparação para a reunião de orientação. Consolida (1) domínio matemático do TEDA/MicroTEDAclus, (2) citações que justificam os achados experimentais, (3) perguntas prováveis dos orientadores com respostas fundamentadas, e (4) dúvidas que surgem ao longo do estudo.
>
> **Status:** Documento em construção — atualizar conforme dúvidas forem sendo resolvidas.
>
> **Contexto:** 167 experimentos em 4 campanhas, resultados abaixo das metas (Recall máx. 61.5% vs alvo 80%), ~5 semanas para defesa.

---

## Índice

1. [Citações que Justificam os Achados](#1-citações)
2. [Domínio Matemático](#2-domínio-matemático)
3. [Perguntas Prováveis dos Orientadores (Q&A)](#3-perguntas-prováveis)
4. [Dúvidas Resolvidas Durante a Preparação](#4-dúvidas-resolvidas)
5. [Lacunas e Itens de Pesquisa Futura](#5-lacunas)

---

<a id="1-citações"></a>
## 1. Citações que Justificam os Achados

### 1.1 "Detecção per-flow é fundamentalmente limitada para DDoS"

| Citação | Achado Relevante |
|---------|-----------------|
| **Kopmann et al. (2022)** "MIDA: Micro-flow Independent Detection of DDoS Attacks with CNNs" ESOCC 2022 | Propõe agregação temporal justamente porque micro-flow analysis é insuficiente para DDoS volumétrico |
| **Chen et al. (2021)** "Is low-rate DDoS a great threat?" IET InfoSec 15(5) | "The average rate of each attack flow is indistinguishable from legitimate traffic flows" |
| **Zhang et al. (2012)** "Flow level detection of low-rate DDoS" Computer Networks 56(15) | Propõe métrica agregada (CPR) porque features per-flow não distinguem LDDoS |
| **Braga et al. (2010)** "Lightweight DDoS flooding attack detection" IEEE LCN | "Detection is very hard because of similarities between normal traffic and useless packets" |
| **Sperotto et al. (2010)** "An Overview of IP Flow-Based Intrusion Detection" IEEE ComSurveys 12(3) | Survey de IDS flow-based reconhecendo limitações para certos tipos de ataque |
| **Dados próprios C02-S2** | Expansão de 17→32 features: ZERO impacto (±1pp). Evidência empírica direta. |

### 1.2 "Detecção por janela temporal supera per-flow"

| Citação | Achado Relevante |
|---------|-----------------|
| **Li et al. (2023)** "Towards real-time ML-based DDoS detection via cost-efficient window-based feature extraction" Sci China Info Sci 66 | Window-based reduz delay e melhora acurácia vs per-flow |
| **Goldschmidt & Kucera (2024)** "Windower: Feature Extraction for Real-Time DDoS Detection" NOMS 2024 | 99% flood detection com FPR <1% usando window-based |
| **Lakhina et al. (2004)** "Diagnosing Network-Wide Traffic Anomalies" ACM SIGCOMM | Seminal: análise agregada revela anomalias invisíveis per-flow |
| **Lakhina et al. (2005)** "Mining Anomalies Using Traffic Feature Distributions" ACM SIGCOMM | Distribuições de features (entropia) sobre múltiplos flows contêm o sinal |
| **Dados próprios C02-S3** | SYN: 3.5%→53.9% (15x), Recon: 4.5%→45.3% (10x), Mirai: 1.7%→33.3% (20x) |

### 1.3 "Anomaly detection encontra outliers, não ataques"

| Citação | Achado Relevante |
|---------|-----------------|
| **Sommer & Paxson (2010)** "Outside the Closed World: On Using Machine Learning for Network Intrusion Detection" IEEE S&P — **CITAÇÃO MAIS IMPORTANTE** | Gap semântico entre anomalia e ataque; explica por que ML para IDS raramente é usado em produção |
| **Gates & Taylor (2006)** "Challenging the Anomaly Detection Paradigm" NSPW | Questiona se comportamento anômalo = comportamento malicioso |
| **Axelsson (2000)** "The Base-Rate Fallacy and the Difficulty of Intrusion Detection" ACM TISSEC 3(3) | Matematicamente: FPR é o fator limitante devido ao base-rate baixo |
| **Chandola et al. (2009)** "Anomaly Detection: A Survey" ACM Comp Surveys 41(3) | Anomalia ≠ ataque. Mapeamento é imperfeito. |
| **Wagner & Soto (2002)** "Mimicry Attacks on Host-Based IDS" ACM CCS | Prova prática de evasão por mimetização de normalidade |
| **Denning (1987)** "An Intrusion-Detection Model" IEEE TSE SE-13(2) | Hipótese original que os resultados desafiam |

### 1.4 Características dos Ataques que Explicam as Taxas de Detecção

| Ataque | Recall obtido | Por quê (hipótese) | Citações de apoio |
|--------|--------------|--------------------|--------------------|
| **DDoS-TCP** | **0%** | TCP flood usa flags válidos (ACK, PSH+ACK), completa handshakes → indistinguível. **Zero dimensões anômalas.** | Zargar et al. (2013) IEEE ComSurveys 15(4): "TCP/ACK floods appear to be legitimate acknowledgments" |
| **Recon-PortScan** | **49%** | Anomalia em 5+ dimensões: port entropy↑, dst diversity↑, flow size↓, duração↓, direcionalidade↑ | Nychis et al. (2008) IMC: "Destination ports exhibit greatest deviation during scans"; Bhuyan et al. (2017) ICCICCT |
| **DDoS-SYN** | **38-62%** | Parcialmente detectável via syn_ratio↑ e unanswered_ratio↑ (1-2 dims anômalas) | RFC 4987 (Eddy, 2007); Bellaiche & Bhunyan (2012) Sec Comm Networks 5(7): SYN/ACK ratio como signal primário |
| **DDoS-ICMP (v1→v2)** | **0→50%** | Features v1 são TCP-cêntricas (sem portas ICMP); v2 adiciona flows/s, payload_std, dst_ip_entropy | Wang et al. (2022) IEEE TNSM 19(2): entropia de dst_IP; David & Thomas (2015) Procedia CS 50 |
| **Mirai-greeth** | **46%** | Roda EM dispositivos IoT → herda características de tráfego legítimo. Apenas rate é anômalo. | Antonakakis et al. (2017) USENIX Security: análise do Mirai; Meidan et al. (2018) IEEE Pervasive: N-BaIoT |

**Framework unificador: "Dimensões Anômalas"**

Quanto mais dimensões do espaço de features um ataque deforma, mais fácil é detectá-lo por anomaly detection. DDoS-TCP tem zero dimensões anômalas — fundamentalmente indetectável. Recon tem 5+ — é o mais tratável.

### 1.5 TEDA em Alta Dimensionalidade e Adaptações

| Citação | Suporte |
|---------|---------|
| **Beyer et al. (1999)** "When Is Nearest Neighbor Meaningful?" ICDT | Distâncias concentram com >10-15 dimensões — fundamento teórico |
| **Aggarwal et al. (2001)** "On the Surprising Behavior of Distance Metrics in High-Dimensional Space" ICDT | L2 (Euclidiana) perde significado em alta-D; L1 é preferível |
| **Francois et al. (2007)** "The Concentration of Fractional Distances" IEEE TKDE 19(7) | Concentração de distância é propriedade intrínseca, não artefato de amostra |
| **Zimek et al. (2012)** "A Survey on Unsupervised Outlier Detection in High-Dimensional Numerical Data" SADM 5(5) | Survey: concentração reduz poder discriminativo, "hubness" distorce NN |
| **Welford (1962)** Technometrics 4(3) | Algoritmo original de variância online |
| **Chan et al. (1983)** "Algorithms for Computing Sample Variance" Am Stat 37(3) | Welford é numericamente superior — análise definitiva |
| **NS-TEDA (Chen et al., 2024)** CMC 78(2) | Dentro da família TEDA: "new data points interact solely with clusters in close proximity" — valida update seletivo |
| **Kohonen (1990)** "The Self-Organizing Map" Proc IEEE 78(9) | Fundamento teórico: winner-take-all preserva diversidade de clusters |
| **DenStream — Cao et al. (2006)** SDM | Padrão em stream clustering: atualiza apenas cluster mais próximo |
| **Amini et al. (2014)** "Density-Based Data Streams Clustering" J Comp Sci Tech 29(1) | Survey confirma: todos usam nearest-centroid update, não update-all |
| **Reynolds (2009)** "Gaussian Mixture Models" Encyclopedia of Biometrics | Padrão de regularização de variância (análogo ao r0) |

### 1.6 Extensões TEDA Relevantes

| Citação | Relevância |
|---------|-----------|
| **Angelov (2014)** "Outside the Box" JAMRIS 8(2) | Paper fundacional do TEDA |
| **Angelov (2014b)** "Anomaly Detection Based on Eccentricity Analysis" IEEE EALS | TEDA aplicado a anomaly detection |
| **Kangin & Angelov (2015)** "Evolving Classifier TEDAClass for Big Data" Procedia CS 53 | Extensão para classificação + paralelização |
| **Costa et al. (2016)** "An Evolving Approach to Unsupervised and Real-Time Fault Detection" Expert Syst Appl 63 | TEDA para falhas industriais (domínio próximo) |
| **Bezerra et al. (2018)** "Automatic Detection of Network Traffic Anomalies Based on Eccentricity Analysis" FUZZ-IEEE | TEDA em network IDS (mas apenas 4-6 features) |
| **Bezerra et al. (2020)** "AutoCloud" Information Sciences 518 | TEDA para data clouds em stream |
| **Silva et al. (2021)** "Hardware Architecture for TEDA" IEEE Sensors J 21(18) | Implementação em FPGA (viabilidade edge) |
| **Maia et al. (2020)** "Evolving Clustering Algorithm Based on Mixture of Typicalities" FGCS 106 | MicroTEDAclus original — referência direta |

### 1.7 Valor de Resultados Negativos e Metodologia

| Citação | Suporte |
|---------|---------|
| **Arp et al. (2022)** "Dos and Don'ts of ML in Computer Security" USENIX Security | Documentar failure modes é essencial para ML-security |
| **Pendlebury et al. (2019)** "TESSERACT: Eliminating Experimental Bias in Malware Classification" USENIX Security | Avaliação temporal previne resultados inflados |
| **Matosin et al. (2014)** "Negativity Towards Negative Results" Disease Models & Mechanisms 7(2) | Resultados negativos previnem duplicação de esforço |
| **Fanelli (2012)** "Negative Results Are Disappearing" Scientometrics 90(3) | Viés de publicação prejudica o progresso científico |
| **Mehta (2019)** "Highlight Negative Results" Nature 573 | Editorial Nature defendendo valor de resultados negativos |
| **Lipton & Steinhardt (2019)** "Troubling Trends in ML Scholarship" Queue 17(1) | Crítica à ausência de ablation studies — seu trabalho é o oposto |
| **Tavallaee et al. (2010)** "Toward Credible Evaluation of Anomaly-Based IDS" IEEE SMC-C 40(5) | Avaliação realista de IDS não-supervisionado deve esperar taxas modestas |

### 1.8 Avaliação Prequential

| Citação | Suporte |
|---------|---------|
| **Gama et al. (2013)** "On Evaluating Stream Learning Algorithms" Machine Learning 90(3) | Formaliza prequential com fading factor |
| **Dawid (1984)** JRSS-A 147(2) | Origem do termo "prequential" |
| **Bifet et al. (2010)** "MOA: Massive Online Analysis" JMLR 11 | Prequential como padrão de facto para stream learning |
| **Bifet et al. (2015)** "Efficient Online Evaluation of Big Data Stream Classifiers" KDD | Fading factors para big data streams |
| **Gama et al. (2014)** "A Survey on Concept Drift Adaptation" ACM Comp Surveys 46(4) | Seção sobre avaliação argumenta contra CV em streams |

### 1.9 Dataset e Trabalhos Relacionados CICIoT2023

| Citação | Suporte |
|---------|---------|
| **Neto et al. (2023)** "CICIoT2023: A Real-Time Dataset" Sensors 23(13) | Dataset oficial; reconhece que Benign/Recon/Spoofing são similares |
| **Mirsky et al. (2018)** "Kitsune: An Ensemble of Autoencoders for Online NIDS" NDSS | Baseline para IDS não-supervisionado em stream |
| **Shone et al. (2018)** "A Deep Learning Approach to NIDS" IEEE TETCI 2(1) | Deep autoencoder não-supervisionado (comparação) |

---

<a id="2-domínio-matemático"></a>
## 2. Domínio Matemático

### 2.1 Eccentricidade — A Base de Tudo

**Por que é importante:** A eccentricidade é uma medida **não-paramétrica** de quão "diferente" um ponto é. A beleza do TEDA (Angelov, 2014) é que ela NÃO assume distribuição gaussiana, independência, ou amostras infinitas — diferente de métodos tradicionais como o Z-score. Usa apenas a **estrutura espacial mútua** dos dados.

**Fórmula:**

```
ξ(xₖ) = 1/k + ||xₖ - μₖ||² / (k × σ²ₖ)
```

**Decomposição termo a termo:**

| Termo | Significado | Comportamento |
|-------|-------------|---------------|
| `1/k` | "Peso de novidade" — decresce com mais amostras | k=10: 0.10 / k=100: 0.01 / k=1000: 0.001 |
| `‖xₖ - μₖ‖²` | Distância euclidiana ao quadrado do ponto ao centro | Quanto mais longe → maior eccentricidade |
| `k × σ²ₖ` | Normalização pela variância total acumulada | Cluster maduro (σ² grande) → mais tolerante |

**Exemplo numérico (2D):**

Cluster com k=100 pontos, μ = [5.0, 3.0], σ² = 2.0.

Chega x = [5.5, 3.5]:
```
‖x - μ‖² = (5.5-5.0)² + (3.5-3.0)² = 0.25 + 0.25 = 0.50
ξ = 1/100 + 0.50/(100 × 2.0) = 0.01 + 0.0025 = 0.0125
```
→ Eccentricidade baixíssima. Ponto **muito típico**.

Agora x = [15.0, 3.0] (longe):
```
‖x - μ‖² = (15.0-5.0)² + (3.0-3.0)² = 100
ξ = 1/100 + 100/(100 × 2.0) = 0.01 + 0.50 = 0.51
```
→ Eccentricidade alta. Ponto **suspeito/anômalo**.

**Tipicalidade:**
```
τ(xₖ) = 1 - ξ(xₖ)
```
τ ≈ 1: muito típico. τ ≤ 0: outlier.

**Interpretação geométrica:** É como um Z-score generalizado sem assumir gaussianidade — mede quantos "raios" do cluster o ponto está distante, normalizado pela maturidade.

---

### 2.2 Teste de Chebyshev — O Critério de Decisão

**Origem — Desigualdade de Chebyshev clássica:**
```
P(|X - μ| ≥ mσ) ≤ 1/m²
```
Válida para QUALQUER distribuição (não precisa gaussianidade).

Com m=3: P(|X-μ| ≥ 3σ) ≤ 1/9 ≈ 11% → 89% dos pontos ficam dentro de 3σ.

**No TEDA:** usa-se a eccentricidade normalizada ζ = ξ/2, e o threshold de aceitação é:
```
threshold = (m² + 1) / (2n)

Aceita se: ζ(x) ≤ threshold
```

**Threshold dinâmico m(k) (Maia, 2020):**
```
m(k) = 3 / (1 + e^{-0.007(k-100)})
```

| k | m(k) | Threshold (n=k) | Comportamento |
|---|------|-----------------|---------------|
| 1 | ~0.60 | **13.0** (caso especial) | Muito permissivo — cluster precisa crescer |
| 10 | ~0.76 | 0.079 | Ainda tolerante |
| 50 | ~1.10 | 0.022 | Moderado |
| 100 | ~1.50 | 0.016 | Transição (ponto de inflexão) |
| 200 | ~2.22 | 0.014 | Convergindo |
| 500 | ~2.90 | 0.009 | Quase m=3 |
| 1000 | ~3.00 | 0.005 | Estrito — cluster maduro |

**Por que faz sentido em IoT:** Clusters jovens (primeiros flows) precisam ser permissivos para acumular estatísticas. Clusters maduros (padrão estabelecido) podem ser estritos e rejeitar tráfego genuinamente diferente.

**Base teórica:** Amidan et al. (2005) e Shershakov (2020) validaram o uso de Chebyshev para outlier detection sem assumir distribuição.

---

### 2.3 Welford vs Fórmula Original — A Causa Raiz do FPR 54%

**Esta é possivelmente a contribuição técnica mais forte da dissertação.** Nenhum paper TEDA testou com mais de ~6 dimensões antes. Maia (2020) testou com dados 2D sintéticos onde a fórmula funciona perfeitamente. Foi descoberto que ela colapsa em 17D.

**A fórmula original (Maia 2020):**
```
σ² = (‖δ‖ × 2/dim)²
```
onde δ = x - μ, dim = número de dimensões.

**Demonstração numérica do problema:**

Suponha δ = [1.0, 0.5, 0.3, 0.8, 0.4, ..., 0.2] (17 dimensões), com ‖δ‖ = 2.5

**Fórmula original:**
```
σ² = (2.5 × 2/17)²
   = (2.5 × 0.1176)²
   = (0.2941)²
   = 0.0865
```

**Welford (correto):**
```
σ² = Σ(δᵢ × δ'ᵢ) / (n-1)
   ≈ soma dos quadrados das diferenças por dimensão
   ≈ 1.0² + 0.5² + 0.3² + ... ≈ 6.18 (para contribuição inicial)
```

**Razão: 6.18 / 0.0865 ≈ 71x de subestimação!**

**Efeito cascata:**
```
ξ = 1/n + ‖x-μ‖² / (n × σ²)
```
Com σ² 70x menor, ξ é inflada → Chebyshev rejeita quase tudo → **FPR catastrófico de 54%**.

**Verificação dimensional (por que o problema só aparece em alta-D):**

| Dimensão | Fator (2/dim)² | Efeito |
|----------|----------------|--------|
| 2D | (2/2)² = **1.00** | Neutro — fórmula funciona! (contexto do Maia 2020) |
| 5D | (2/5)² = 0.16 | Subestima 6x |
| 10D | (2/10)² = 0.04 | Subestima 25x |
| **17D** | **(2/17)² = 0.014** | **Subestima ~70x (nosso caso)** |
| 32D | (2/32)² = 0.004 | Subestima 250x |

**Por que Welford resolve:** Computa variância de forma incremental via `dot(δ_old, δ_new)`, que soma contribuições de **todas as dimensões** corretamente, independente de quantas dimensões existam. Numericamente estável conforme Chan et al. (1983).

**Algoritmo de Welford (incremental):**
```python
n += 1
delta = x - mean
mean = mean + delta / n
delta2 = x - mean              # mean já atualizada
var_sum += dot(delta, delta2)  # produto escalar
variance = var_sum / (n - 1)
```

**Suporte teórico:** Beyer et al. (1999), Aggarwal et al. (2001), e Zimek et al. (2012) documentam que distâncias euclidianas e estimativas de variância colapsam em alta dimensionalidade. Nenhum paper TEDA havia testado com >6 features, então essa é uma contribuição genuinamente nova.

---

### 2.4 MicroTEDAclus — Fluxo Completo de Decisão

```
Ponto x chega
    │
    ▼
┌─ Para cada micro-cluster MCᵢ: ──────────────────┐
│   1. Calcula ζᵢ = ξᵢ(x) / 2                     │
│   2. Calcula thresholdᵢ = (mᵢ(k)² + 1) / (2nᵢ) │
│   3. Se ζᵢ ≤ thresholdᵢ → MC aceita x           │
│      (caso n=2: rejeita somente se               │
│       ζ > threshold E σ² ≥ r0)                   │
└──────────────────────────────────────────────────┘
    │
    ├── Algum MC aceita? ─── SIM ──→ Atualiza MC com MAIOR tipicalidade
    │                                (SÓ UM — diferença crucial vs original!)
    │                                → Label: NORMAL
    │
    └── Nenhum aceita? ───── NÃO ──→ Cria novo micro-cluster com x como centro
                                     → Label: ANOMALIA (se total_samples ≥ min_samples)
```

**As 5 Adaptações e Por Que Importam:**

| # | O quê | Original (Maia 2020) | Adaptação própria | Impacto | Suporte |
|---|-------|---------------------|-------------------|---------|---------|
| 1 | Variância | `(‖δ‖·2/dim)²` | Welford `dot(δ,δ')` | **CRÍTICO** — causa raiz do FPR 54% | Welford (1962), Chan (1983), Beyer (1999) |
| 2 | Update policy | Atualiza TODOS os clusters aceitantes | Só o melhor (max typicality) | Evita convergência de clusters | Kohonen (1990), DenStream (2006), NS-TEDA (2024) |
| 3 | Threshold n=1 | Sem proteção → cluster morre | threshold=13 (m=5) | Cold start — clusters jovens sobrevivem | Extensão de Maia (2020) |
| 4 | Guard n=2 | Só checa var > limit | Rejeita se ζ > thr **E** σ² ≥ r0 | Menos fragmentação prematura | Adaptação própria |
| 5 | Piso de variância | Implícito via r0 | Explícito `max(σ², min_var)` | Evita divisão por zero | Reynolds (2009) GMM reg_covar |

---

### 2.5 Per-flow vs Window — Por Que Muda Tudo

**A mudança de granularidade muda a pergunta do detector:**

- **Per-flow:** "Este flow individual é diferente?" → Geralmente NÃO, porque um flow DDoS individual é idêntico a um flow IoT benigno (curto, poucos pacotes, tamanho regular)
- **Per-window:** "Este IP está se comportando de forma anômala nos últimos N segundos?" → O padrão de flooding/scanning emerge aqui

**Formalização via "Dimensões Anômalas":**

Um ataque é detectável por anomaly detection proporcionalmente ao número de dimensões do espaço de features em que ele deforma o tráfego.

| Ataque | # Dims anômalas | Quais | Recall obtido |
|--------|-----------------|-------|---------------|
| Recon-PortScan | 5+ | port_entropy↑, dst_diversity↑, flow_size↓, duration↓, fwd_only↑ | 49% |
| DDoS-ICMP (v2) | 3-4 | flows/s↑, payload_std↓, dst_ip_entropy↓, small_flow↑ | 50% |
| Mirai | 2-3 | rate↑, target_concentration↑ | 46% |
| DDoS-SYN | 1-2 | syn_ratio↑, unanswered↑ | 38-62% |
| DDoS-TCP | **0** | Todas as dimensões sobrepõem com benigno | **0%** |

Lakhina et al. (2004, 2005) formalizaram isso: as **distribuições agregadas** contêm o sinal, não os pontos individuais.

---

### 2.6 Avaliação Prequential

```
Para cada amostra xₖ no stream:
  1. PREDIZ: ŷₖ = detector.predict(xₖ)    ← SEM atualizar estatísticas
  2. AVALIA: compara ŷₖ com y_true         ← Acumula métricas
  3. TREINA: detector.process(xₖ)          ← Agora atualiza estatísticas
```

**Por que NÃO usar holdout/cross-validation em streams:**
- **Holdout** desperdiça dados e ignora evolução temporal
- **Cross-validation** embaralha dados → destrói dependências temporais (Gama et al., 2014)
- **Prequential** respeita a ordem temporal e captura degradação com drift (Bifet et al., 2015)

**Fading factor α=0.01** dá mais peso às predições recentes:
```
P_α(n) = α × eₙ + (1-α) × P_α(n-1)
```
Com α=0.01, "half-life" ≈ 69 amostras: erros antigos pesam exponencialmente menos.

---

### 2.7 Métricas no Contexto de Anomaly Detection

Dado que o problema é altamente desbalanceado (benigno >> ataque), métricas clássicas precisam ser interpretadas com cuidado:

| Métrica | Fórmula | Interpretação em IDS |
|---------|---------|----------------------|
| **Precision** | TP / (TP + FP) | Dos alertas gerados, quantos são reais? Baixa precision = muito ruído ao operador |
| **Recall** | TP / (TP + FN) | Dos ataques reais, quantos foram detectados? Baixo recall = ataques passam |
| **F1** | 2·P·R/(P+R) | Média harmônica — penaliza desbalanceamento |
| **FPR** | FP / (FP + TN) | Taxa de falsos positivos — crítico para viabilidade operacional |
| **MTTD** | Samples até primeira detecção | Quão rapidamente o sistema reage |

**Base-rate fallacy (Axelsson, 2000):** Se o base rate de ataques é 1%, mesmo um detector com 99% de accuracy gera mais falsos positivos do que verdadeiros positivos — porque 99% dos eventos são benignos. Isso justifica o foco em FPR baixo como critério primário.

---

<a id="3-perguntas-prováveis"></a>
## 3. Perguntas Prováveis dos Orientadores (Q&A)

### P1: "Por que o Recall é tão baixo? O alvo era 80% e o melhor é 61.5%."

> O Recall é limitado por uma questão **estrutural, não algorítmica**. Flows individuais de ataque DDoS são estatisticamente indistinguíveis de flows benignos IoT — ambos são curtos, com poucos pacotes e tamanho regular. Sommer & Paxson (2010) previram exatamente isso: anomaly detection encontra outliers estatísticos, não ataques. Nossa evidência: expandir features de 17 para 32 teve ZERO impacto (C02-S2). A melhoria veio da mudança de **granularidade** (per-flow → window), que transformou a pergunta do detector de "este flow é diferente?" para "este IP comporta-se de forma anômala?" — produzindo melhorias de 10-20x no Recall.

### P2: "O que diferencia sua implementação da original do Maia?"

> Cinco adaptações necessárias para funcionar em 17 dimensões. A mais crítica é a **fórmula de variância**: a original usa `(‖δ‖·2/dim)²`, que em 2D (onde foi testada) tem fator `(2/2)²=1` — neutro. Em 17D, o fator é `(2/17)²=0.014`, subestimando σ² em ~70x. Isso infla a eccentricidade, o Chebyshev rejeita quase tudo, e o FPR sobe para 54%. A substituição pelo Welford (1962), com estabilidade comprovada por Chan et al. (1983), resolve. As demais adaptações (update seletivo, cold start, guard n=2, piso de variância) são secundárias mas necessárias. Nenhum paper TEDA havia testado com >6 features — isso faz da descoberta uma contribuição original.

### P3: "Por que não usar deep learning? Dá resultados melhores."

> Dá — com dados rotulados. A Fase 1 com Random Forest (supervisionado) atingiu F1>0.99. Mas o cenário da Fase 2 é **streaming não-supervisionado**: não há labels em produção, o modelo precisa processar um ponto por vez, e deve adaptar a concept drift sem retreino. Deep learning para anomaly detection (autoencoders) requer treino offline e é computacionalmente pesado para IoT edge. MicroTEDAclus processa em O(1) por ponto, usa memória constante, e adapta-se criando novos clusters. A comparação justa seria com Kitsune (Mirsky et al., 2018, NDSS), que usa autoencoders não-supervisionados — eles também reportam dificuldades com ataques que mimetizam tráfego normal. Silva et al. (2021) mostram que TEDA pode rodar em FPGA, inviável para DL.

### P4: "Qual a contribuição científica se os resultados são negativos?"

> São quatro contribuições distintas:
>
> 1. **Contribuição técnica** — Identificação e correção de 5 falhas na implementação original do MicroTEDAclus para alta dimensionalidade (validada em C04 com 30 runs, FPR 3.9% vs 54.4%)
> 2. **Contribuição empírica** — 167 experimentos com ablation study rigoroso documentando ONDE e POR QUE anomaly detection falha para IoT IDS, confirmando empiricamente as previsões teóricas de Sommer & Paxson (2010)
> 3. **Contribuição metodológica** — Framework de avaliação prequential adaptado para IDS não-supervisionado em streaming
> 4. **Resultado positivo** — Recon-PortScan com F1=43.7% demonstra que o pipeline funciona para ataques com assinatura estatística distinta
>
> Arp et al. (2022, USENIX Security) argumentam que documentar failure modes é essencial para o progresso de ML em segurança. Matosin et al. (2014) mostram que resultados negativos previnem duplicação de esforço.

### P5: "O DDoS-TCP é indetectável por quê?"

> Porque TCP flood gera flows que completam handshakes TCP válidos usando flags legítimos (ACK, PSH+ACK). Cada flow individual é indistinguível de uma conexão TCP benigna — mesmo tamanho de pacotes, mesma duração, mesmos flags. Zargar et al. (2013) classificam TCP/ACK floods como os mais difíceis de detectar justamente porque "appear to be legitimate acknowledgments." Mesmo com features de janela (entropia, taxas), o tráfego TCP flood se sobrepõe ao benigno em **todas** as dimensões do espaço de features — tem zero dimensões anômalas. Para detectar, seria necessário: (a) deep packet inspection, (b) correlação com o estado do servidor (CPU/memória), ou (c) análise de volume absoluto com contexto global, não relativo.

### P6: "Como o r0 foi escolhido?"

> O r0 é o piso de variância que evita rejeição excessiva em clusters jovens. Testamos r0 ∈ {0.001, 0.05, 0.10, 0.15, 0.20, 0.30}. Na calibração em A1 (benigno-only), r0=0.10 produziu FPR=3.5% (≤5% aceito). R0 menores (0.001, 0.05) são mais restritivos (mais anomalias); r0 maiores (0.15, 0.20) são mais tolerantes. O papel do r0 é análogo ao `reg_covar` em Gaussian Mixture Models — regularização da covariância para evitar singularidade (Reynolds, 2009). Na prática, r0 é o hiperparâmetro principal do MicroTEDAclus, similar a ε no DBSCAN.

### P7: "E o concept drift? Vocês testaram?"

> O MicroTEDAclus lida com concept drift **nativamente**: quando nenhum cluster aceita um ponto (Chebyshev rejeita), um novo cluster é criado. Isso significa que novos padrões de ataque são automaticamente representados sem retreino. No entanto, os cenários de drift planejados (B1: DDoS→Mirai, D1: drift recorrente) do plano original **não foram implementados** — priorizamos o diagnóstico do problema de representação (mais fundamental). Concept drift é um ponto forte do algoritmo em teoria, suportado por Gama et al. (2014), mas não foi validado experimentalmente neste trabalho. É um item claro para trabalhos futuros.

### P8: "167 experimentos é suficiente estatisticamente?"

> Sim, para um ablation study. Cada campanha testa uma variável por vez: C01 (algoritmo), C02-S1 (ground truth), C02-S2 (features), C02-S3 (granularidade), C03-S4 (features de janela), C04 (implementação). Os resultados são consistentes e reproduzíveis — o FPR benigno varia menos de ±1pp entre runs com mesma configuração. O que não fizemos (e seria ideal com mais tempo) é 5+ repetições por configuração com seeds diferentes para reportar intervalo de confiança. Mas os efeitos observados são grandes (10-20x melhoria de Recall com janelas, 14x diferença de FPR entre implementações) — não são artefatos estatísticos. Arp et al. (2022) chamam isso de "large-effect validation" — quando o efeito é dominante, poucas repetições bastam para evidenciar causalidade.

### P9: "Por que usar Kafka? Não é overengineering?"

> Kafka serve dois propósitos: (1) **desacoplamento** — o produtor de PCAPs, o reconstrutor de flows, e o detector operam independentemente, o que permite testar cada componente isoladamente e simular cenários de produção; (2) **reprodutibilidade** — os tópicos Kafka mantêm a ordem de inserção, garantindo que cada experimento processa exatamente os mesmos dados na mesma ordem. Para a dissertação, Kafka demonstra que a arquitetura é compatível com streaming real em produção. Para os experimentos em si, poderíamos ter usado filas in-memory com o mesmo resultado.

### P10: "As features de janela v2 (comportamentais) não ajudaram consistentemente. O que fazer?"

> As features v2 desbloqueiam DDoS-ICMP (0→50% Recall) e melhoram Recon (39→45%), mas degradam Mirai e SYN e quintuplicam o FPR (2.9→14.3%). O diagnóstico é **curse of dimensionality**: com janelas de 10s, temos ~210 vetores com 19 features — poucos dados para alta dimensionalidade. Os clusters não convergem. Zimek et al. (2012) documentam esse efeito para outlier detection em alta-D. Soluções possíveis: (a) seleção de features por tipo de ataque (mas viola o princípio unsupervised), (b) redução de dimensionalidade (PCA antes do TEDA), (c) Two-Stage detection onde Stage 1 opera per-flow com baixo FPR e Stage 2 analisa concentração de anomalias por IP — essa é a proposta do S5.

### P11: "Por que não normalizar os dados?"

> Os flows são normalizados antes de entrar no detector. A questão não é normalização — é que flows de ataque, mesmo normalizados, ocupam a mesma região do espaço que flows benignos. A normalização ajuda a equilibrar a escala entre features, mas não resolve sobreposição de distribuições. É exatamente o ponto de Sommer & Paxson (2010): o problema não é matemático, é semântico.

### P12: "Qual é o melhor resultado da dissertação? Como se compara com a literatura?"

> Recon-PortScan com F1=43.7% (Recall 49.1%, Precision 39.4%, FPR 12.9%) usando v2/w10s/r0=0.05. Para contextualizar:
>
> - Métodos **supervisionados** no CICIoT2023 atingem F1>95% — mas requerem labels
> - Métodos **não-supervisionados** no CICIoT2023 são praticamente inexistentes na literatura — quase todo paper publicado usa supervised ML
> - Kitsune (Mirsky et al., 2018) com autoencoders não-supervisionados em outros datasets reporta F1 de 70-85% para alguns ataques, mas também falha em ataques que mimetizam tráfego normal
>
> Nosso resultado demonstra que MicroTEDAclus **funciona** para ataques com assinatura estatística distinta, e documenta sistematicamente para quais não funciona.

### P13: "Se o MicroTEDAclus não funciona bem para DDoS, por que não mudar de algoritmo?"

> O problema não é o algoritmo — é a **representação**. Provamos isso em C02-S2: adicionar features não ajuda. E em C04: mesmo o algoritmo original, com fórmulas diferentes, tem o mesmo problema fundamental (pior, com FPR 54%). Qualquer detector baseado em distância/densidade teria o mesmo problema com flows DDoS que se sobrepõem ao benigno. A solução está na **granularidade** (janelas temporais), não na troca de algoritmo. Se trocarmos para Isolation Forest ou DBSCAN per-flow, teremos o mesmo resultado. Sommer & Paxson (2010) argumentam que essa limitação é fundamental para anomaly detection, não específica de um algoritmo.

### P14: "A avaliação prequential é adequada? Por que não usar train/test split?"

> Em streaming, train/test split viola premissas fundamentais: (1) assume dados i.i.d., o que não vale para tráfego de rede com dependências temporais; (2) desperdiça dados que poderiam ser usados para adaptação; (3) não captura degradação com concept drift. Gama et al. (2013) formalizaram a avaliação prequential com fading factor como o padrão para stream learning. Bifet et al. (2015) a implementaram no framework MOA como método padrão. O protocolo test-then-train garante que cada ponto é avaliado ANTES de ser usado para treino — eliminando data leakage. Pendlebury et al. (2019) em TESSERACT mostram que avaliação não-temporal infla resultados artificialmente.

### P15: "O que o Frederico (co-autor do Maia 2020) vai pensar das críticas à implementação original?"

> As diferenças não são "bugs" — são **adaptações para um contexto diferente**. O Maia (2020) testou com dados 2D sintéticos onde `(2/2)²=1` é neutro. O paper é correto para o domínio em que foi validado. A contribuição da dissertação é demonstrar que a **transferência** para dados de alta dimensionalidade (17 features de rede IoT) requer adaptações específicas. Isso é uma **extensão** do trabalho original, não uma crítica. O framing deve ser: "As 5 adaptações propostas **estendem** o MicroTEDAclus para o domínio de IoT IDS de alta dimensionalidade, mantendo os princípios fundamentais do framework TEDA."

---

<a id="4-dúvidas-resolvidas"></a>
## 4. Dúvidas Resolvidas Durante a Preparação

> **Esta seção é atualizada incrementalmente** conforme você faz perguntas durante o estudo. Cada entrada registra a dúvida original e a explicação construída em conjunto.

<!-- TEMPLATE para novas entradas:
### D[N]: [Título curto da dúvida]
**Contexto:** [De onde surgiu a dúvida]
**Pergunta:** [Formulação original]
**Resposta:**
[Explicação completa com exemplos se relevante]
**Fontes:** [Citações ou arquivos relacionados]
-->

---

### D1: De onde vieram as informações do slide "Dimensões Anômalas"?

**Contexto:** Durante a revisão do slide do framework de "Dimensões Anômalas" na apresentação, surgiu a dúvida legítima sobre a origem das informações — especialmente da coluna "# Dimensões Anômalas" com valores 5+, 3-4, 2-3, 1-2, 0 por tipo de ataque. Esta é uma pergunta que os orientadores muito provavelmente vão fazer, e é crucial saber responder com honestidade intelectual.

**Pergunta original:** "Não entendi bem este slide. De onde vieram estas informações?"

**Resposta — Desconstrução em 3 camadas de informação:**

O slide original misturava três tipos de informação com graus diferentes de suporte científico, o que é um risco se apresentado como se tudo fosse igualmente sólido. A desconstrução honesta é:

**Camada 1 — O que É experimentalmente medido (dados próprios):**

A coluna "Recall" contém números reais extraídos dos ANALYSIS.md das campanhas:

| Ataque | Recall | Fonte |
|--------|--------|-------|
| Recon-PortScan | 49% | C03-S4, v2/w10s/r0=0.05 |
| DDoS-ICMP | 50% | C03-S4, v2/w10s/r0=0.10 |
| Mirai-greeth | 46% | C03-S4, v1/w10s/r0=0.10 |
| DDoS-SYN | 38-62% | C03-S4, faixa ao variar r0 |
| DDoS-TCP | 0% | todas as configurações testadas |

Rastreáveis em `experiments/results/campaign-02/ANALYSIS.md` e `campaign-03/ANALYSIS.md`.

**Camada 2 — O que tem suporte direto na literatura (citações reais):**

As características por ataque (quais features são anômalas para cada um) têm suporte em papers específicos:

- **Recon usa port_entropy alto + dst_diversity:** Nychis et al. (2008) IMC, "Entropy-based Traffic Anomaly Detection" — demonstram experimentalmente que entropia de portas é o sinal mais forte para scans
- **DDoS-TCP usa flags válidos e é indistinguível:** Zargar et al. (2013) IEEE ComSurveys 15(4) — classificam TCP/ACK floods como os mais difíceis porque "appear to be legitimate acknowledgments"
- **Mirai herda características do tráfego IoT:** Antonakakis et al. (2017) USENIX Security — análise completa do botnet; Meidan et al. (2018) "N-BaIoT" IEEE Pervasive — propõem detecção por dispositivo justamente porque tráfego Mirai se parece com IoT legítimo
- **DDoS-SYN tem syn_ratio e unanswered altos:** Bellaiche & Bhunyan (2012) Sec Comm Networks 5(7); RFC 4987 (Eddy, 2007)
- **DDoS-ICMP detectável por dst_ip entropy:** Wang et al. (2022) IEEE TNSM 19(2)

**Camada 3 — O que é síntese analítica (sem paper direto):**

Duas coisas foram **construídas durante a preparação da apresentação** e NÃO estão em nenhum paper:

1. **A contagem "# Dimensões Anômalas" (5+, 3-4, 2-3, 1-2, 0).** Esta contagem foi construída qualitativamente, raciocinando sobre cada ataque com base nas features que a literatura identifica como discriminativas. Ninguém mediu isso formalmente. É uma organização analítica proposta neste trabalho.

2. **O framework causal em si ("mais dimensões anômalas → maior detectabilidade").** Não existe paper dizendo exatamente isso. É uma hipótese que é consistente com os dados, mas não foi validada estatisticamente. A inspiração conceitual vem de Sommer & Paxson (2010) (gap semântico anomalia-ataque) e Lakhina et al. (2005) (distribuições agregadas contêm o sinal), mas nenhum dos dois propõe esse framework específico.

**Decisão tomada:** Manter como "framework proposto" (Opção 1), mas reestruturar o slide único em **3 slides** deixando o caráter de proposta analítica completamente explícito e propondo um experimento de validação estatística. Isso transforma uma potencial fraqueza em contribuição honesta.

**Como o framework ficou estruturado (3 slides):**

**Slide 11 — Definição e Hipótese Causal**
- Caixa dourada no topo com caveat explícito: "Síntese analítica proposta neste trabalho — NÃO é conceito estabelecido na literatura"
- Definição formal: uma feature f_j é "anômala" para o ataque A quando D_benign(f_j) ≠ D_attack(f_j). A contagem d_A é o número de features com distribuições separáveis.
- Exemplo concreto da feature `dst_port_entropy`: benigno IoT ~1.2 bits (2-3 portas ativas) vs Recon ~6.5 bits (100+ portas probadas) → distribuições separáveis → conta como 1 dimensão anômala para Recon
- Hipótese causal com derivação matemática: mais dimensões anômalas → maior ‖x−μ‖² → maior ξ → maior rejeição pelo Chebyshev → maior Recall. Conexão direta com a fórmula ξ = 1/k + ‖x−μ‖²/(k·σ²): como ‖x−μ‖² = Σ(x_j − μ_j)² é aditivo por dimensão, cada dimensão deformada contribui aditivamente para a eccentricidade.
- Inspiração: Sommer & Paxson (2010), Lakhina et al. (2005)

**Slide 12 — Metodologia de Contagem e Aplicação**
- Metodologia explicitada em 3 passos: (1) inspeção visual das 19 features v2 nos dados das campanhas, (2) cross-reference com literatura, (3) classificação forte/fraca/nula com apenas "fortes" entrando na contagem
- Aviso vermelho explícito: "Classificação é qualitativa. Não há teste estatístico formal."
- Tabela aplicada aos 5 ataques com coluna adicional de **citação da literatura** por linha (Nychis, Wang, Antonakakis, Bellaiche, Zargar)
- Observação: padrão monotônico observado (5+→49%, 0→0%), consistente com hipótese, mas N=5 não permite validação estatística rigorosa

**Slide 13 — Validação Estatística Proposta (não executada)**
- Fase 1: KS-test (Kolmogorov-Smirnov 2-sample, não-paramétrico, distribution-free) por feature e por ataque, com correção de Bonferroni para 19 testes múltiplos, contando como "anômala" se p < 0.01 → gera d_A estatístico reproduzível
- Fase 2: Teste de correlação Spearman entre d_A estatístico e Recall_A (H0: sem correlação / H1: correlação positiva), com limitação explícita de N=5 dar poder baixo mas direcionalidade reportável
- Bottom: o que o experimento resolveria (contagem objetiva, validação/falsificação da hipótese, índice de detectabilidade reproduzível) + tempo estimado de 2-3 dias (scripts já existem)
- Extensão futura: expandir para 10+ ataques do CICIoT2023

**Como defender o framework se os orientadores perguntarem:**

**P: "Isso está em qual paper?"**
> "Nenhum. É uma síntese analítica que construí para organizar os resultados dos 167 experimentos. A contagem de dimensões é qualitativa, baseada em inspeção dos dados com cross-reference na literatura sobre cada ataque. O slide 11 começa com o caveat explícito de que é uma proposta deste trabalho, e o slide 13 propõe o experimento de validação formal."

**P: "Por que não fez o experimento de validação?"**
> "Priorizei o diagnóstico do problema fundamental (representação per-flow insuficiente) e a validação da contribuição técnica de C04 (30 runs comparando implementações). O framework das dimensões anômalas emergiu como ferramenta analítica depois que os dados já estavam consolidados. O teste estatístico via KS-test + Spearman é o próximo passo natural — 2-3 dias de trabalho, os scripts de extração já existem das campanhas anteriores."

**P: "A contagem qualitativa é rigorosa?"**
> "Não é estatisticamente rigorosa. É análise qualitativa baseada em inspeção visual dos dados e cross-reference com literatura específica por ataque (Nychis, Zargar, Wang, etc.). Por isso o slide 12 tem um aviso vermelho explícito sobre essa limitação, e o slide 13 propõe o caminho para a contagem estatística via KS-test."

**P: "Isso vai entrar como contribuição da dissertação?"**
> "Como framework conceitual sim, mas com o rigor que realmente tem. Na dissertação apresentarei como 'ferramenta analítica proposta para organizar os achados experimentais', não como resultado estatisticamente validado. A validação formal fica como trabalho futuro ou, se der tempo antes da defesa, como experimento adicional. O que é contribuição robusta é o padrão empírico observado (Recall cresce monotonicamente com d_A qualitativo), que é consistente com a hipótese e com as previsões teóricas de Sommer & Paxson."

**Lição aprendida (para futuras sessões):** Quando construir frameworks analíticos próprios durante a preparação, **sempre** separar explicitamente as 3 camadas (dado medido / citação direta / síntese própria) antes de apresentar. O caveat deve estar no slide, não só na cabeça do apresentador. Isso é o que distingue "contribuição honesta" de "alegação não suportada" — e os orientadores percebem a diferença imediatamente.

**Fontes:**
- Slides 11-13 da apresentação (`docs/meeting/2026-03-19-advisor-meeting.pptx`)
- Código gerador: `docs/meeting/generate_meeting_pptx.py` linhas ~810-1100
- Dados empíricos: `experiments/results/campaign-02/ANALYSIS.md`, `experiments/results/campaign-03/ANALYSIS.md`
- Citações de suporte ao framework: Sommer & Paxson (2010), Lakhina et al. (2005), Nychis et al. (2008), Zargar et al. (2013), Wang et al. (2022), Antonakakis et al. (2017), Bellaiche & Bhunyan (2012)
- Citações para o método de validação: Massey (1951) KS-test, Spearman (1904) correlação

---

<a id="5-lacunas"></a>
## 5. Lacunas e Itens de Pesquisa Futura

Pontos onde **não temos citação direta** e que devem ser apresentados com honestidade intelectual ao orientador:

| Claim | Status | Ação Sugerida |
|-------|--------|---------------|
| Fórmula original subestima variância em ~70x | Derivação própria — sem paper dizendo isso explicitamente | Apresentar como contribuição original; suportar com Beyer (1999) / Aggarwal (2001) sobre curse of dimensionality |
| TEDA falha em alta dimensão | Nenhum paper TEDA testou >6D | Contribuição genuinamente nova — framing como "first empirical study of TEDA in high-dimensional network data" |
| Prequential para IDS não-supervisionado | Sem paper específico aplicando prequential a anomaly-based IDS não-supervisionado | Argumentar que prequential adapta naturalmente; contribuição metodológica menor |
| CICIoT2023 com métodos não-supervisionados | Quase nenhum paper publicado | GAP na literatura — motivação forte para o trabalho |
| Comparação direta per-flow vs window no mesmo algoritmo/dataset | Não encontramos estudo controlado equivalente | Nossos dados C02-S3 SÃO o estudo controlado — framing como "first controlled comparison" |
| Dimensões anômalas como framework unificador | Conceito construído a partir dos dados | Apresentar como ferramenta de análise proposta neste trabalho, inspirada em Sommer & Paxson (2010) |

**Itens para experimentação adicional (se houver tempo):**

1. **Visualização t-SNE/PCA** dos flows DDoS-TCP vs Benigno — evidência visual direta da sobreposição
2. **Teste de KS/KL-divergence** por feature para quantificar sobreposição de distribuições
3. **Supervised baseline** (Random Forest) em DDoS-TCP — confirmar que mesmo supervisionado falha (isolando problema de representação)
4. **S5 Two-Stage Detection** — hipótese principal para melhorar detecção sem aumentar FPR

---

## Apêndice: Arquivos de Referência

| O quê | Onde |
|-------|------|
| Apresentação (md) | `docs/meeting/2026-03-19-advisor-meeting.md` |
| Apresentação (pptx) | `docs/meeting/2026-03-19-advisor-meeting.pptx` |
| Gerador do PPTX | `docs/meeting/generate_meeting_pptx.py` |
| Detector próprio | `experiments/streaming/src/detector/micro_teda.py` |
| Detector original (adapter) | `experiments/streaming/src/detector/original_micro_teda.py` |
| Teoria TEDA | `research/foundations/teda-framework.md` |
| Metodologia | `experiments/methodology.md` |
| Análise C01 | `experiments/results/campaign-01/ANALYSIS.md` |
| Análise C02 | `experiments/results/campaign-02/ANALYSIS.md` |
| Análise C03 | `experiments/results/campaign-03/ANALYSIS.md` |
| Análise C04 | `experiments/results/campaign-04/ANALYSIS.md` |
| Bibliografia | `research/bibliography.bib` |
| Plano de campanha | `experiments/campaign-plan.md` |
| Status do projeto | `STATUS.md` |
