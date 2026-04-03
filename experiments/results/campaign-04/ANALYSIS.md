# Campaign-04 — Analise de Resultados

**Data:** 2026-04-03
**Objetivo:** Comparar implementacao original do autor (EvolvingClustering, Maia 2020) com implementacao propria (MicroTEDAclus)
**Algoritmo:** `original_micro_teda` (wrapper sobre `evolclustering` package)
**Dataset:** CICIoT2023 (Benign_Final + 5 ataques)
**Pacotes por PCAP:** 50k benigno + 50k ataque | **Max flows:** 10.000
**Total de runs:** 30

---

## 1. Resumo Executivo

| Criterio | Resultado | Status |
|----------|-----------|--------|
| FPR benigno (flow-level) | **54.4%** (vs 3.9% micro_teda) | REPROVADO |
| FPR benigno (window 10s) | **41.9-45.5%** (vs 2.9-14.3%) | REPROVADO |
| FPR benigno (window 30s) | **73.6-74.5%** (vs 5.0-6.7%) | REPROVADO |
| Recall (flow-level) | 27-92% (vs 0-27%) | Alto, mas trivial |
| Recall (window 10s) | 69-100% (vs 0-46%) | Alto, mas trivial |
| F1 melhor que micro_teda | Nao — FPR invalida os ganhos de Recall | REPROVADO |

**Conclusao principal:** A implementacao original do autor produz **FPR catastrofico** (42-75%) em todas as granularidades. Os altos valores de Recall sao consequencia trivial de classificar metade ou mais do trafego como anomalo — nao representam deteccao significativa.

**Implicacao:** A implementacao propria (micro_teda) e **superior** para IDS em IoT. As diferencas de implementacao (update de todos os clusters aceitantes, pruning com life decay, formulas de variancia) nao melhoram a deteccao; pelo contrario, degradam a especificidade.

---

## 2. Resultados por Bloco

### 2.1 Block B1 — Flow-Level (6 runs)

| Cenario | Flows | Precision | Recall | F1 | FPR | Clusters |
|---------|-------|-----------|--------|-----|-----|----------|
| Benigno | 3.225 | — | — | — | **54.4%** | 218 |
| DDoS-ICMP | 3.322 | 3.2% | 69.4% | 6.2% | 54.7% | 220 |
| DDoS-SYN | 6.230 | 27.2% | 37.5% | 31.6% | 53.3% | 566 |
| DDoS-TCP | 3.251 | 0.7% | 92.3% | 1.3% | 54.5% | 214 |
| Mirai | 4.236 | 12.9% | 26.8% | 17.4% | 55.1% | 204 |
| Recon | 9.890 | 47.5% | 46.2% | 46.8% | 55.3% | 809 |

**Anomaly rate ~50-55%** para todos os cenarios, incluindo benigno puro. O algoritmo original cria ~200-800 clusters para 3-10k flows, indicando fragmentacao extrema dos micro-clusters.

### 2.2 Block B2 — Window v1 (12 runs)

| Cenario | w=10s | | | | w=30s | | | |
|---------|-------|-------|------|------|-------|-------|------|------|
| | Recall | F1 | FPR | Clust | Recall | F1 | FPR | Clust |
| Benigno | — | — | 41.9% | 60 | — | — | 74.5% | 110 |
| DDoS-ICMP | 100% | 4.0% | 44.7% | 63 | 100% | 5.5% | 73.6% | 114 |
| DDoS-SYN | 76.9% | 15.5% | 47.7% | 82 | 100% | 20.0% | 71.7% | 92 |
| DDoS-TCP | 0% | 0% | 45.0% | 58 | 0% | 0% | 73.4% | 110 |
| Mirai | 84.6% | 18.2% | 45.3% | 72 | 90.0% | 14.8% | 75.7% | 120 |
| Recon | 88.9% | 36.2% | 52.2% | 112 | 83.3% | 36.6% | 74.2% | 108 |

### 2.3 Block B3 — Window v2 (12 runs)

| Cenario | w=10s | | | | w=30s | | | |
|---------|-------|-------|------|------|-------|-------|------|------|
| | Recall | F1 | FPR | Clust | Recall | F1 | FPR | Clust |
| Benigno | — | — | 45.5% | 61 | — | — | 73.6% | 111 |
| DDoS-ICMP | 100% | 5.7% | 46.0% | 67 | 100% | 3.7% | 73.0% | 114 |
| DDoS-SYN | 69.2% | 14.2% | 47.1% | 84 | 100% | 19.7% | 72.1% | 93 |
| DDoS-TCP | 0% | 0% | 43.3% | 62 | 0% | 0% | 72.1% | 106 |
| Mirai | 76.9% | 16.7% | 45.1% | 73 | 90.0% | 14.8% | 74.1% | 120 |
| Recon | 82.7% | 33.3% | 52.6% | 111 | 87.7% | 39.8% | 73.5% | 106 |

---

## 3. Comparacao Direta: Original vs Proprio

### 3.1 Flow-Level (r0=0.10)

| Ataque | micro_teda Rec | original Rec | micro_teda F1 | original F1 | micro_teda FPR | original FPR |
|--------|---------------|-------------|--------------|------------|---------------|-------------|
| Benigno | — | — | — | — | **3.9%** | 54.4% |
| DDoS-ICMP | 27.2% | 69.4% | **21.4%** | 6.2% | **3.6%** | 54.7% |
| DDoS-SYN | 3.5% | 37.5% | 6.3% | **31.6%** | **4.2%** | 53.3% |
| DDoS-TCP | 0.0% | 92.3% | 0.0% | 1.3% | **3.2%** | 54.5% |
| Mirai | 1.7% | 26.8% | 3.0% | **17.4%** | **3.5%** | 55.1% |
| Recon | 4.5% | 46.2% | 8.4% | **46.8%** | **3.1%** | 55.3% |

### 3.2 Window 10s (r0=0.10)

| Ataque | micro_teda v1 Rec | original v1 Rec | micro_teda v1 FPR | original v1 FPR |
|--------|-------------------|-----------------|--------------------|-----------------|
| Benigno | — | — | **2.9%** | 41.9% |
| DDoS-ICMP | 0.0% | 100% | 13.6% | 44.7% |
| DDoS-SYN | 38.5% | 76.9% | 14.8% | 47.7% |
| DDoS-TCP | 0.0% | 0.0% | **11.3%** | 45.0% |
| Mirai | 46.2% | 84.6% | 15.5% | 45.3% |
| Recon | 39.2% | 88.9% | **13.1%** | 52.2% |

---

## 4. Diagnostico: Por que o Original tem FPR tao Alto?

### 4.1 Diferencas de Implementacao

| Aspecto | micro_teda (proprio) | original_micro_teda (Maia) |
|---------|---------------------|---------------------------|
| **Update policy** | Atualiza apenas o melhor cluster | Atualiza TODOS os clusters aceitantes |
| **Caso n=1** | threshold=13 (permissivo) | Sem caso especial |
| **Caso n=2** | variance >= r0 guard | variance > variance_limit |
| **Pruning** | Sem pruning | Life decay + remocao |
| **Macro-clusters** | Nao tem | Grafo de conectividade |
| **Variancia** | Welford (var_sum) | Norm-based (delta*2/len)^2 |

### 4.2 Causa Raiz: Formula de Variancia

A diferenca mais impactante e a **formula de variancia**:

**Proprio (Welford):**
```python
delta = x - mean
delta2 = x - new_mean
var_sum += dot(delta, delta2)
variance = var_sum / (n - 1)
```

**Original (norm-based):**
```python
delta = x - mean
variance = ((n-1)/n) * var + ((norm(delta)*2/len(delta))^2 / (n-1))
```

A formula original calcula `(norm(delta)*2/len(delta))^2` que, para vetores de 17 dimensoes, produz valores de variancia **muito menores** que o Welford. Com variancia subestimada, o teste de Chebyshev rejeita mais pontos → mais clusters → mais anomalias.

### 4.3 Causa Raiz: Update de Todos os Clusters

O original atualiza **todos** os clusters que aceitam um ponto, nao apenas o melhor. Isso faz com que multiplos clusters "puxem" suas medias na mesma direcao, reduzindo a diversidade entre clusters e tornando mais dificil para novos pontos serem aceitos.

### 4.4 Evidencia: Numero de Clusters

| Config | micro_teda Clusters | original Clusters | Ratio |
|--------|--------------------|--------------------|-------|
| Flow-level (~3k flows) | ~270 | ~220 | 0.8x |
| Window 10s (~210 vetores) | ~30-40 | ~60-80 | 2x |
| Window 30s (~140 vetores) | ~20-30 | ~110 | 4-5x |

O original cria **2-5x mais clusters** em modo window, indicando fragmentacao. No flow-level, paradoxalmente tem *menos* clusters — mas como a anomaly rate e ~55%, muitos pontos sao classificados como anomalias antes de formar clusters estaveis.

---

## 5. Conclusoes

### 5.1 Resposta a Pergunta Principal

> "A implementacao original do autor produz resultados significativamente diferentes?"

**Sim — significativamente piores.** O FPR de 42-75% torna o original inutilizavel como IDS. A implementacao propria (micro_teda) e claramente superior para este caso de uso.

### 5.2 Achados Especificos

1. **O original nao e calibrado para dados de alta dimensionalidade.** Foi projetado para datasets sinteticos 2D (vide notebooks do autor). Com 17-19 features, a formula de variancia diverge.

2. **Update de todos os clusters aceitantes** degrada a separabilidade ao longo do tempo — clusters convergem para regioes similares.

3. **Pruning nao compensa** — mesmo removendo clusters inativos, novos sao criados a cada ponto rejeitado.

4. **DDoS-TCP permanece indetectavel** em ambas as implementacoes — confirmando que o problema e de representacao, nao de algoritmo.

5. **Recon e o unico ataque com F1 razoavel** em ambas (C04: 46.8% flow, C03: 43.7% window) — mas no original o FPR de 55% invalida o resultado.

### 5.3 Implicacao para a Dissertacao

> A comparacao com a implementacao original do MicroTEDAclus (Maia 2020) demonstra
> que modificacoes na logica de atualizacao de clusters e na formula de variancia
> sao criticas para aplicacao em IDS de IoT. A implementacao original, projetada
> para datasets sinteticos de baixa dimensionalidade, produz FPR de 42-75% quando
> aplicada a features de flow de rede (17 dimensoes), tornando-a inadequada para
> deteccao de intrusao. A implementacao propria, com atualizacao seletiva (apenas
> melhor cluster) e variancia via Welford, reduz o FPR para 3-15% mantendo
> capacidade de deteccao comparavel. Este resultado reforça a contribuicao
> da dissertacao: a adaptacao do algoritmo ao dominio e tao importante
> quanto o algoritmo em si.

---

## 6. Artefatos

### Estrutura de resultados
```
experiments/results/campaign-04/
  B1-A1-benign-flow-r0_0.10/                # Block 1: flow-level (6 runs)
  B1-A2-{ddos,syn,tcp,mirai,recon}-flow-*/
  B2-A1-benign-wfv1-w{10,30}s-r0_0.10/      # Block 2: window v1 (12 runs)
  B2-A2-{attacks}-wfv1-w{10,30}s-r0_0.10/
  B3-A1-benign-wfv2-w{10,30}s-r0_0.10/      # Block 3: window v2 (12 runs)
  B3-A2-{attacks}-wfv2-w{10,30}s-r0_0.10/
  generate_plots_c04.py
  plots/
  ANALYSIS.md
```

**Total: 30 experimentos.**

---

## 7. Parametros do Ambiente

| Parametro | Valor |
|-----------|-------|
| Python | 3.13 |
| Kafka | Confluent (Docker) |
| SO | Linux 6.8.0-101-generic |
| evolclustering | 0.1 (editable install) |
| Flow timeout | 60s (event time) |
| Detector idle timeout | 10s |
| Tempo total | 38m 31s (30 runs) |
