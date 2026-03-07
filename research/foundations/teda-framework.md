# TEDA Framework - Fundamentação Teórica

**Criado:** 2025-12-09
**Última Atualização:** 2026-01-20
**Baseado em:** Angelov (2014), Maia et al. (2020)
**Capítulo da Dissertação:** 2 - Fundamentação Teórica

> **Propósito:** Documento de FUNDAMENTAÇÃO TEÓRICA para a dissertação. Explica os conceitos de TEDA, eccentricidade, tipicalidade, micro-clusters e MicroTEDAclus.

---

## 1. Por que Clustering Evolutivo?

### Limitações dos Algoritmos Tradicionais

| Algoritmo | Limitação | Problema para IoT IDS |
|-----------|-----------|----------------------|
| **K-means** | K fixo, clusters esféricos | Não sabe quantos tipos de ataque existem |
| **DBSCAN** | Parâmetros eps/min_samples fixos | Não adapta a mudanças na distribuição |
| **Ambos** | Processamento em batch | Não funciona para streaming contínuo |

### O que Clustering Evolutivo Resolve

- **Streaming:** Processa um ponto por vez (single-pass)
- **K automático:** Cria/remove clusters conforme necessário
- **Concept drift:** Adapta quando padrões mudam
- **Memória constante:** Não armazena todos os dados

---

## 2. Framework TEDA (Typicality and Eccentricity Data Analytics)

O framework TEDA foi proposto por Angelov (2014) e fornece uma técnica não-paramétrica para determinar quão excêntrico/típico uma observação é em relação às outras observações geradas pelo mesmo processo.

### 2.1 Eccentricidade (ξ)

**Definição:** Mede quão "diferente" um ponto é em relação à distribuição.

**Fórmula:**
```
ξ(xₖ) = 1/k + (μₖ - xₖ)ᵀ(μₖ - xₖ) / (k × σ²ₖ)

Onde:
- k = número de pontos vistos
- μₖ = média acumulada
- σ²ₖ = variância acumulada
```

**Interpretação:**
- ξ ≈ 0: Ponto muito similar aos outros (típico)
- ξ ≈ 1: Ponto muito diferente (outlier)
- ξ > 1: Ponto extremamente diferente

**Código Python:**
```python
def calculate_eccentricity(x, mean, variance, n):
    dist_to_mean = np.sum((x - mean) ** 2)
    if variance > 0:
        return (1/n) + dist_to_mean / (n * variance)
    return 1/n
```

### 2.2 Tipicalidade (τ)

**Definição:** Inverso da eccentricidade - quão "típico" é o ponto.

**Fórmula:**
```
τ(xₖ) = 1 - ξ(xₖ)
```

**Interpretação:**
- τ ≈ 1: Ponto muito típico (pertence ao cluster)
- τ ≈ 0: Ponto na fronteira
- τ < 0: Ponto definitivamente não pertence

---

## 3. Atualização Recursiva (Single-Pass)

### O Problema
Em streaming, não podemos armazenar todos os dados para calcular média e variância.

### A Solução: Algoritmo de Welford
Manter apenas 3 valores: `n`, `μ`, `σ²`

**Atualização em O(1) por ponto:**
```python
class RecursiveTEDA:
    def __init__(self, n_features):
        self.n = 0
        self.mean = np.zeros(n_features)
        self.var_sum = 0

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean = self.mean + delta / self.n
        delta2 = x - self.mean
        self.var_sum += np.dot(delta, delta2)

        # Calcular eccentricidade
        if self.n > 1:
            variance = self.var_sum / (self.n - 1)
            dist = np.sum((x - self.mean) ** 2)
            eccentricity = (1/self.n) + dist / (self.n * variance)
        else:
            eccentricity = 1.0

        return eccentricity
```

### Cold Start
- Primeiros ~100-200 pontos: estatísticas instáveis
- Após ~200 pontos: eccentricidade estabiliza
- **Estratégia:** Não tomar decisões críticas no cold start

---

## 4. Micro-clusters e Macro-clusters

### 4.1 Problema do Centro Único

Com um único centro global, grupos distintos parecem igualmente "excêntricos":

```
Grupo A em (0,0)          Grupo B em (10,10)
     ●●●                        ●●●
    ●●●●●                      ●●●●●
     ●●●                        ●●●
        \        Centro        /
         \    Global (5,5)    /
          \       ✖         /
     dist=5              dist=5

Resultado: Ambos têm mesma eccentricidade!
```

### 4.2 Solução: Múltiplos Micro-clusters

Cada micro-cluster tem:
- Seu próprio centro (μ)
- Sua própria variância (σ²)
- Sua própria noção de tipicalidade

```python
class MicroCluster:
    def __init__(self, center, n_features):
        self.n = 1
        self.mean = center.copy()
        self.var_sum = 0.0

    def calculate_typicality(self, x):
        ecc = self.calculate_eccentricity(x)
        return 1 - ecc

    def update(self, x):
        # Atualização recursiva
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.var_sum += np.dot(delta, delta2)
```

### 4.3 Macro-clusters

- Grupos de micro-clusters conectados/sobrepostos
- Representam estruturas de alto nível (ex: tipo de ataque)
- Micro-clusters próximos são "merged" em macro-clusters

---

## 5. Mixture of Typicalities

### Conceito Central

Para cada novo ponto, calcular tipicalidade para TODOS os micro-clusters:

```
Ponto x chega:
├── Tipicalidade para MC_1: 0.95  ← MAIOR
├── Tipicalidade para MC_2: 0.23
├── Tipicalidade para MC_3: -0.15
└── Tipicalidade para MC_4: 0.01

Decisão: x pertence a MC_1 (maior tipicalidade)
```

### Algoritmo
```python
def assign_point(x, micro_clusters):
    best_cluster = None
    best_typicality = -float('inf')

    for mc in micro_clusters:
        typ = mc.calculate_typicality(x)
        if typ > best_typicality:
            best_typicality = typ
            best_cluster = mc

    return best_cluster, best_typicality
```

---

## 6. Teste de Chebyshev

### Propósito
Decidir se um ponto deve ser:
- **Aceito** em um cluster existente, ou
- **Rejeitado** (criar novo cluster)

### Fórmula
```
threshold = (m² + 1) / (2n)

Aceitar se: eccentricidade ≤ threshold

Onde:
- m = número de desvios padrão (tipicamente 3)
- n = número de pontos no cluster
```

### Comportamento Adaptativo

| N pontos | Threshold (m=3) | Comportamento |
|----------|-----------------|---------------|
| 10 | 0.5000 | Tolerante (cluster jovem) |
| 100 | 0.0500 | Moderado |
| 500 | 0.0100 | Estrito (cluster maduro) |

**Implicação:** Clusters maduros são mais "exigentes" - rejeitam pontos diferentes, que formam novos clusters.

### Código
```python
def chebyshev_test(eccentricity, n, m=3):
    threshold = (m**2 + 1) / (2 * n)
    return eccentricity <= threshold
```

---

## 7. Tratamento de Concept Drift

### Como o Algoritmo se Adapta

1. **Novo padrão aparece** (ex: novo tipo de ataque)
2. **Clusters existentes rejeitam** (Chebyshev test falha)
3. **Novo cluster é criado** para o novo padrão
4. **Novo cluster "aprende"** conforme mais pontos chegam
5. **Clusters antigos podem ser desativados** se param de receber pontos

### Diagrama
```
Tempo t=0: Clusters A e B ativos
           [A: Normal] [B: DDoS]

Tempo t=100: Novo ataque (Mirai) aparece
             → Rejeitado por A e B
             → Cria cluster C
           [A: Normal] [B: DDoS] [C: Mirai]

Tempo t=500: DDoS para de ocorrer
             → Cluster B fica inativo
           [A: Normal] [C: Mirai]
```

---

## 8. MicroTEDAclus: Algoritmo Completo

### Pseudocódigo Simplificado
```python
def process_stream(data_stream):
    micro_clusters = []

    for x in data_stream:
        # 1. Encontrar melhor cluster
        best_mc, best_typ = None, -inf
        for mc in micro_clusters:
            typ = mc.calculate_typicality(x)
            if typ > best_typ:
                best_typ = typ
                best_mc = mc

        # 2. Testar aceitação (Chebyshev)
        if best_mc and chebyshev_test(1 - best_typ, best_mc.n):
            # Aceito: atualizar cluster
            best_mc.update(x)
        else:
            # Rejeitado: criar novo cluster
            new_mc = MicroCluster(center=x)
            micro_clusters.append(new_mc)

        # 3. Merge clusters sobrepostos (opcional)
        merge_overlapping_clusters(micro_clusters)

        # 4. Retornar atribuição
        yield assign_to_macro_cluster(x, micro_clusters)
```

### Parâmetros
| Parâmetro | Valor Típico | Descrição |
|-----------|--------------|-----------|
| m | 3 | Desvios padrão para Chebyshev |
| merge_threshold | - | Distância para merge de clusters |

**Vantagem:** Poucos parâmetros, todos com valores default razoáveis!

---

## 9. Aplicação para IoT IDS

### Mapeamento de Conceitos

| Conceito TEDA | Aplicação IoT IDS |
|---------------|-------------------|
| Micro-cluster | Padrão de tráfego específico |
| Macro-cluster | Categoria (Normal, DDoS, Mirai, etc.) |
| Alta tipicalidade | Tráfego reconhecido |
| Baixa tipicalidade | Possível anomalia/ataque |
| Novo cluster | Novo tipo de ataque detectado |
| Concept drift | Evolução dos padrões de ataque |

### Pipeline Proposto
```
Tráfego IoT → Pré-processamento → MicroTEDAclus → Decisão
                                       ↓
                               [Tipicalidade baixa?]
                                   ↓         ↓
                                 SIM        NÃO
                                   ↓         ↓
                               ALERTA    Normal
```

---

## 10. Descobertas Experimentais (CICIoT2023)

### Experimentos Realizados

| Experimento | Descoberta |
|-------------|------------|
| K-means (K=2 a 10) | Dataset tem ~8-10 clusters naturais |
| Silhouette Score | Melhor qualidade com K=10 |
| Classes desbalanceadas | Normal=3.2%, Attack=96.8% |
| Eccentricidade | Normal tem maior eccentricidade (é minoria) |
| DBSCAN | Comportamento não-linear com eps |

### Implicações

1. **Múltiplos tipos de ataque:** Confirmado pelos ~8-10 clusters
2. **Desbalanceamento:** Em produção, a relação inverte (mais normal)
3. **Eccentricidade funciona:** Detecta classe minoritária com 20x amplificação

---

## 11. Referências

### Paper Principal

```bibtex
@article{maiaEvolvingClusteringAlgorithm2020,
  title = {Evolving Clustering Algorithm Based on Mixture of Typicalities for Stream Data Mining},
  author = {Maia, José and Severiano Jr., Carlos Alberto and Guimarães, Frederico Gadelha and de Castro, Cristiano Leite and Lemos, André Paim and Fonseca Galindo, Juan Camilo and Weiss Cohen, Miri},
  year = {2020},
  journal = {Future Generation Computer Systems},
  volume = {106},
  pages = {672--684},
  doi = {10.1016/j.future.2020.01.017}
}
```

### Framework TEDA

- **Angelov, P. (2014).** "Outside the box: an alternative data analytics framework." *Journal of Automation Mobile Robotics and Intelligent Systems*, 8(2), pp.29-35. DOI: [10.14313/JAMRIS_2-2014/16](https://doi.org/10.14313/JAMRIS_2-2014/16)
  - Paper fundacional do framework TEDA
  - Introduz conceitos de tipicalidade e eccentricidade

- **TEDA R Package:** [CRAN - teda](https://cran.r-project.org/web/packages/teda/index.html)
  - Implementação de referência em R
  - Métodos batch e recursivo

### Algoritmos Relacionados

- **NS-TEDA (2024):** "Improved Data Stream Clustering Method: Incorporating KD-Tree for Typicality and Eccentricity-Based Approach" - [TechScience](https://www.techscience.com/cmc/v78n2/55536/html)
  - Versão otimizada com KD-Tree
  - Melhora eficiência computacional

- **AutoCloud:** Clustering evolutivo baseado em TEDA para data clouds sem forma pré-definida

### Concept Drift e Data Streams

- **Benchmark Survey (2024):** "A benchmark and survey of fully unsupervised concept drift detectors on real-world data streams" - [Springer](https://link.springer.com/article/10.1007/s41060-024-00620-y)
  - Revisão de 10 algoritmos de detecção de concept drift
  - Análise de arquiteturas e suposições

- **Systematic Review (2024):** "A systematic review on detection and adaptation of concept drift in streaming data using machine learning techniques" - [Wiley](https://wires.onlinelibrary.wiley.com/doi/10.1002/widm.1536)
  - Revisão abrangente de métodos de concept drift
  - Métricas de avaliação

- **Temporal Silhouette (2023):** "Temporal silhouette: validation of stream clustering robust to concept drift" - [Machine Learning Journal](https://link.springer.com/article/10.1007/s10994-023-06462-2)
  - Validação de clustering em streams
  - Índice robusto a concept drift

### IoT Intrusion Detection

- **ML-based IDS Survey (2024):** "Machine Learning-Based Intrusion Detection Methods in IoT Systems: A Comprehensive Review" - [MDPI Electronics](https://www.mdpi.com/2079-9292/13/18/3601)
  - Revisão de métodos de ML para IDS em IoT
  - Comparação de arquiteturas

- **Two-step Clustering IDS (2024):** "Two-step data clustering for improved intrusion detection system using CICIoT2023 dataset" - [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2772671124002535)
  - Usa o mesmo dataset CICIoT2023
  - Abordagem de clustering em duas etapas

- **Evolutionary ML for IDS (2024):** "Hybrid evolutionary machine learning model for advanced intrusion detection architecture" - [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0308206)
  - Algoritmos genéticos para otimização de IDS
  - Validação em datasets IoT

- **Deep Learning IoT IDS (2024):** "A novel deep learning-based intrusion detection system for IoT DDoS security" - [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S2542660524002774)
  - Foco em ataques DDoS
  - Arquitetura de deep learning

### Dataset

- **CICIoT2023:** Neto et al. (2023) - Dataset do Canadian Institute for Cybersecurity
  - Tráfego IoT real com ataques rotulados
  - Múltiplas categorias de ataque (DDoS, DoS, Mirai, etc.)

### Implementações

- **Python - TEDA Regressor:** [GitHub](https://github.com/pedrohmeiraa/TEDA-Regressor)
  - Código base em Python (foco em regressão)
  - Pode ser adaptado para clustering

- **R - teda package:** [GitHub (CRAN mirror)](https://github.com/cran/teda)
  - Implementação oficial do framework TEDA

---

## 12. Próximos Passos

1. **Implementar MicroTEDAclus** para CICIoT2023
2. **Comparar** com baselines da Fase 1 (Random Forest, Isolation Forest, etc.)
3. **Simular concept drift** para testar adaptação
4. **Integrar com Kafka** (Fase 3) para streaming real-time
5. **Documentar** resultados para dissertação

---

**Este documento serve como referência rápida para os conceitos de clustering evolutivo aplicados ao projeto de mestrado.**

*Última atualização: 2025-12-09*
