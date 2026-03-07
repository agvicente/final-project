# Fichamento: MicroTEDAclus - Evolving Clustering Algorithm

**Referência completa:** Maia, J., Severiano Junior, C.A., Guimarães, F.G., Castro, C.L., Lemos, A.P., Galindo, J.C.F., Cohen, M.W. (2020). "Evolving clustering algorithm based on mixture of typicalities for stream data mining." *Future Generation Computer Systems*, 106, pp.672-684. DOI: 10.1016/j.future.2020.01.017

**Data de leitura:** 2026-01-13
**Área:** Machine Learning (Clustering Evolutivo / Stream Data Mining)
**Orientador como co-autor:** Frederico Gadelha Guimarães ✓

---

## 1. Objetivo do Artigo

Propor um algoritmo de clustering evolutivo chamado **MicroTEDAclus** que:
- Processa dados em streaming one-sample-at-a-time
- Lida com concept drift (mudanças na distribuição)
- Divide o problema em **micro-clusters** e **macro-clusters**
- Usa **mixture of typicalities** para estimar densidade
- Não requer parâmetros de granularidade definidos pelo usuário

---

## 2. Motivação

### 2.1 Problemas dos Algoritmos Existentes

| Algoritmo | Limitações |
|-----------|------------|
| **Incremental k-means** | Requer k conhecido; assume clusters esféricos |
| **CluStream** | Problemas com clusters não-esféricos |
| **DenStream/DBStream** | Muitos parâmetros livres para ajustar |
| **CEDAS** | Só retorna atribuição, não densidades |

### 2.2 Características Desejadas

1. **Single-pass:** Processar um sample por vez
2. **Formas arbitrárias:** Clusters de qualquer shape
3. **Sem parâmetros de granularidade:** Não definir raio/número
4. **Retornar membership:** Grau de pertinência, não só label

---

## 3. Concept Drift (Seção 2)

### 3.1 Tipos de Drift (Figura 1 do paper)

| Tipo | Descrição | Característica |
|------|-----------|----------------|
| **Sudden/Abrupt** | Conceito muda instantaneamente | Mudança significativa |
| **Incremental** | Sequência de conceitos intermediários | Mudança gradual |
| **Gradual** | Alternância entre componentes | Transição suave |

### 3.2 Definição Formal

Drift ocorre quando:
```
P_t(X, Y) ≠ P_{t+d}(X, Y)
```

Pode ser causado por:
- Mudança em P(X) — distribuição das features
- Mudança em P(Y|X) — relação features→labels

### 3.3 Concept Evolution

Caso especial: **surgimento de novo padrão** (nova classe/cluster).

---

## 4. Revisão do TEDA (Seção 3)

### 4.1 Fórmulas Base (do Angelov 2014)

**Cumulative Proximity:**
```
π_k(x) = Σ_{i=1}^{k} d(x_k, x_i)
```

**Eccentricity:**
```
ξ_k(x) = 2π_k(x) / Σ_{i=1}^{k} π_k(x_i)
```

**Forma Recursiva (Euclidiana):**
```
ξ(x_k) = 1/k + (μ_k - x_k)ᵀ(μ_k - x_k) / (k × σ_k²)
```

**Atualização Recursiva de μ e σ²:**
```
μ_k = ((k-1)/k) × μ_{k-1} + x_k/k

σ_k² = ((k-1)/k) × σ_{k-1}² + (1/(k-1)) × ||x_k - μ_k||²
```

**Typicality:**
```
τ(x_k) = 1 - ξ(x_k)
```

**Normalizações:**
```
ζ(x_k) = ξ(x_k) / 2        (soma = 1)
t(x_k) = τ(x_k) / (k-2)    (soma = 1)
```

### 4.2 Threshold de Outlier (Chebyshev)

Condição para outlier:
```
ζ_k(x) > (m² + 1) / (2k)
```

Onde m = número de desvios-padrão (tipicamente m=3).

---

## 5. O Algoritmo MicroTEDAclus (Seção 4)

### 5.1 Visão Geral

```
┌─────────────────────────────────────────────────────────┐
│                    MicroTEDAclus                        │
├─────────────────────────────────────────────────────────┤
│  1. MICRO-CLUSTER UPDATE                                │
│     - Criar/atualizar micro-clusters com TEDA           │
│     - Variância limitada dinamicamente                  │
│                                                         │
│  2. MACRO-CLUSTER UPDATE                                │
│     - Agrupar micro-clusters que se intersectam         │
│     - Filtrar por densidade (ativar apenas os densos)   │
│                                                         │
│  3. CLUSTER ASSIGNMENT                                  │
│     - Atribuir ao macro-cluster com maior T_j(x)        │
│     - T_j = mixture of typicalities                     │
└─────────────────────────────────────────────────────────┘
```

### 5.2 Inovação Principal: Threshold Dinâmico m(k)

**Problema:** Com poucos dados (k pequeno), não é confiável dizer que x está "mσ" da média.

**Solução:** m cresce dinamicamente com k:

```
m(k) = 3 / (1 + e^{-0.007(k-100)})
```

**Comportamento:**
- k ≈ 1: m(k) ≈ 1 (restritivo)
- k → ∞: m(k) → 3 (saturação)
- k ≈ 1000: m(k) ≈ 3

**Por que:**
1. Com k=2, ζ sempre = 0.5, então m ≥ 1 necessário
2. Conforme k cresce, estimativas de μ e σ são mais confiáveis
3. Saturação em 3 é padrão para Chebyshev (99.7% dos dados)

### 5.3 Micro-Clusters (Algorithm 1)

**Estrutura de um micro-cluster mⁱ:**

| Parâmetro | Significado |
|-----------|-------------|
| Sⁱ_k | Número de samples |
| μⁱ_k | Centro (média) |
| (σⁱ_k)² | Variância |
| ξⁱ(x_k), ζⁱ(x_k) | Eccentricity |
| τⁱ(x_k), tⁱ(x_k) | Typicality |
| Dⁱ_k = ζⁱ(x₁)/k | Densidade |
| mⁱ_k(Sⁱ_k) | Threshold dinâmico |

**Inicialização (primeiro ponto):**
```
n = 1; S¹_1 = 1; μ¹_1 = x_1; (σ¹_1)² = 0
```

**Condição de Outlier (k > 2):**
```
ζⁱ(x_k) > (mⁱ_k(Sⁱ_k)² + 1) / (2Sⁱ_k)
```

**Condição Especial para k = 2:**
```
[Condição de outlier] AND (σⁱ_2)² < r₀
```

Onde **r₀ = 0.001** (limite de variância para evitar micro-clusters gigantes).

**Lógica:**
```
SE x_k NÃO é outlier para ≥1 micro-cluster:
    Atualizar TODOS os micro-clusters onde não é outlier
SENÃO:
    Criar NOVO micro-cluster com x_k
```

**Fórmulas de Atualização:**
```
Sⁱ_k = Sⁱ_{k-1} + 1

μⁱ_k = ((Sⁱ_k - 1) / Sⁱ_k) × μⁱ_{Sⁱ_k - 1} + x_k / Sⁱ_k

(σⁱ_k)² = ((Sⁱ_k - 1) / Sⁱ_k) × (σⁱ_{k-1})² + (1/(Sⁱ_k - 1)) × ||x_k - μⁱ_k||² / d

ξⁱ(x_k) = 1/Sⁱ_k + (2||x_k - μⁱ_k||² / d) / ((Sⁱ_k)² × (σⁱ_k)²)
```

### 5.4 Macro-Clusters (Algorithm 2)

**Condição de Interseção:**
```
dist(μⁱ_k, μʲ_k) < 2(σⁱ_k + σʲ_k)    ∀i ≠ j
```

Dois micro-clusters se intersectam se a distância entre seus centros é menor que a soma de seus "raios" (2σ).

**Problema:** Em dados overlapping, todos os micro-clusters podem se conectar → 1 macro-cluster gigante.

**Solução: Filtro de Densidade**

Ativar apenas micro-clusters com densidade ≥ média:
```
active(mˡ) = Dˡ_k ≥ mean(Dˡ_k)    para l = 1, ..., |M_j|
```

Micro-clusters em regiões de baixa densidade são desativados → separa macro-clusters overlapping.

### 5.5 Mixture of Typicalities (Inovação Principal)

**Densidade do macro-cluster M_j:**
```
T_j(x_k) = Σ_{l ∈ M_j} w^l_k × t^l_k(x_k)
```

Onde o peso é a densidade normalizada:
```
w^l_k = D^l_k / Σ_{l ∈ M_j} D^l_k
```

**Atribuição:**
```
cluster(x_k) = argmax_j T_j(x_k)
```

O ponto é atribuído ao macro-cluster com maior "mixture of typicalities score".

---

## 6. Complexidade Computacional (Seção 7)

| Etapa | Complexidade | Variáveis |
|-------|--------------|-----------|
| Micro-cluster update | O(dnk) | d=dimensões, n=samples, k=micro-clusters |
| Macro-cluster update | O(k²_ch × d + k_ch) | k_ch = micro-clusters alterados |
| Cluster assignment | O(n × k_act) | k_act = micro-clusters ativos |

**Nota:** k_ch ≤ k e k_act < k, então é mais eficiente que O(dnk²).

---

## 7. Experimentos (Seções 5-6)

### 7.1 Datasets

**Estáticos:**
- ST-D1: 3031 samples, 9 clusters
- ST-D2: 2551 samples, 8 clusters
- Cassini: 4000 samples, 3 clusters (formas não-convexas)

**Com Concept Drift:**
- STR-B1: 2 clusters se movendo e cruzando
- STR-B2: 2 clusters estáticos + 1 móvel
- RBF: Eventos de criação/split/merge/deleção de clusters

### 7.2 Métrica: Adjusted Rand Index (ARI)

Mede similaridade entre clustering predito e ground truth:
- ARI = 1: Partições idênticas
- ARI ≈ 0: Partições aleatórias

### 7.3 Comparação com Estado da Arte

| Dataset | DenStream | CluStream | StreamKM++ | **MicroTEDAclus** |
|---------|-----------|-----------|------------|-------------------|
| ST-D1 | 0.21±0.10 | 0.55±0.03 | 0.62±0.05 | 0.38±0.07 |
| ST-D2 | 0.22±0.13 | 0.36±0.07 | 0.37±0.03 | **0.41±0.05** |
| CSN | 0.39±0.09 | 0.40±0.11 | 0.48±0.18 | 0.42±0.16 |
| STR-B1 | 0.61±0.31 | 0.57±0.33 | 0.66±0.27 | **0.68±0.21** |
| STR-B2 | 0.65±0.13 | 0.65±0.13 | 0.55±0.04 | **0.69±0.12** |
| RBF | 0.42±0.10 | 0.61±0.06 | 0.46±0.03 | 0.50±0.07 |

**Observação:** MicroTEDAclus é competitivo, mas o diferencial é a **robustez de parâmetros**.

### 7.4 Vantagem Principal: Robustez de Parâmetros

| Algoritmo | Parâmetros a ajustar |
|-----------|---------------------|
| DenStream | ε (raio), μ (peso), β (outlier) |
| CluStream | m (micro-clusters), h (janela), t (boundary) |
| StreamKM++ | s (coreset), k (clusters) |
| **MicroTEDAclus** | **r₀ = 0.001 (fixo para todos experimentos!)** |

> "MicroTEDAclus was very robust in terms of parameters. The only variable to be tuned was the variance limit r₀, which stayed in the same value for all the experiments."

---

## 8. Pseudocódigo Completo

### Algorithm 1: Micro-cluster Update

```python
def micro_cluster_update(x_k, r0=0.001):
    if k == 1:
        # Primeiro ponto: criar primeiro micro-cluster
        n = 1
        m[1] = MicroCluster(S=1, μ=x_k, σ²=0)
    else:
        flag = True  # Assume outlier para todos

        for i in range(1, n+1):
            # Calcular threshold dinâmico
            m_val = 3 / (1 + exp(-0.007 * (m[i].S - 100)))

            if m[i].S == 2:
                # Condição especial para k=2
                outlier = (ζ[i](x_k) > (m_val**2 + 1)/(4)) AND (m[i].σ² < r0)
            else:
                # Condição normal
                outlier = ζ[i](x_k) > (m_val**2 + 1) / (2 * m[i].S)

            if not outlier:
                # Atualizar micro-cluster
                update_micro_cluster(m[i], x_k)
                flag = False

        if flag:
            # Criar novo micro-cluster
            n += 1
            m[n] = MicroCluster(S=1, μ=x_k, σ²=0)
```

### Algorithm 2: Macro-cluster Update

```python
def macro_cluster_update(x_k, micro_clusters):
    # 1. Agrupar micro-clusters que se intersectam
    M = group_intersecting(micro_clusters)  # Eq. 17

    # 2. Filtrar: ativar apenas os mais densos
    for j in range(len(M)):
        avg_density = mean([m.D for m in M[j]])
        for m in M[j]:
            m.active = (m.D >= avg_density)

    # 3. Calcular mixture of typicalities para cada macro-cluster
    T = []
    for j in range(len(M)):
        active_micros = [m for m in M[j] if m.active]
        total_D = sum([m.D for m in active_micros])

        T_j = 0
        for m in active_micros:
            w = m.D / total_D
            T_j += w * m.t(x_k)
        T.append(T_j)

    # 4. Atribuir ao macro-cluster com maior T
    return argmax(T)
```

---

## 9. Relação com TEDA (Angelov 2014)

| Aspecto | TEDA | MicroTEDAclus |
|---------|------|---------------|
| Granularidade | Data clouds (pontos) | Micro-clusters (estatísticas) |
| Threshold m | Fixo (m=3) | Dinâmico m(k) |
| Limite de variância | Não definido | r₀ = 0.001 |
| Interseção | "Zona de influência" (vago) | dist < 2(σ_i + σ_j) |
| Densidade | Tipicalidade simples | Mixture of typicalities |
| Macro-clusters | Não formalizado | Grafo de interseção + filtro |

**MicroTEDAclus preenche as lacunas do TEDA:**
1. Define m(k) ao invés de m fixo
2. Define zona de influência: 2σ
3. Define critério de interseção
4. Adiciona filtro de densidade para separar overlapping
5. Formaliza mixture of typicalities

---

## 10. Implicações para Minha Pesquisa

### 10.1 Código Disponível

GitHub: https://github.com/cseveriano/evolving_clustering

### 10.2 Adaptação para IDS IoT

| Aspecto | Aplicação |
|---------|-----------|
| Streaming | Processar pacotes de rede em tempo real |
| Concept drift | Novos tipos de ataque surgindo |
| Mixture of typicalities | Grau de anomalia, não binário |
| Sem parâmetros | Robusto para diferentes cenários IoT |

### 10.3 Possíveis Extensões

1. **Forgetting factor:** Dar menos peso a dados antigos
2. **Merge/delete de micro-clusters:** Reduzir memória
3. **Integração com classificação:** Dois estágios (clustering + label)

---

## 11. Citações Importantes

### Sobre o algoritmo

> "MicroTEDAclus has competitive performance for online clustering of data streams with arbitrary shapes. However, it is worth noting that MicroTEDAclus was very robust in terms of parameters."

### Sobre adaptação a drift

> "MicroTEDAclus tend to adapt to such events over time. The prequential evaluation suggests that the algorithms can adapt to different concept drift events."

### Sobre memória

> "In addition to not requiring storage of information from each sample processed, the algorithm presented a good ability to handle data with higher dimensionality."

---

## 12. Glossário

| Termo | Definição |
|-------|-----------|
| **Micro-cluster** | Estrutura granular com estatísticas (μ, σ², S, D) |
| **Macro-cluster** | Grupo de micro-clusters conectados |
| **Mixture of typicalities** | Soma ponderada de tipicalidades: T_j = Σ w_l × t_l |
| **m(k)** | Threshold dinâmico: 3/(1 + e^{-0.007(k-100)}) |
| **r₀** | Limite de variância para k=2 (default: 0.001) |
| **Active micro-cluster** | Micro-cluster com D ≥ média do macro-cluster |
| **Prequential method** | Testar antes de treinar (streaming evaluation) |
| **ARI** | Adjusted Rand Index — métrica de clustering |

---

## 13. Figuras Importantes

- **Fig. 1:** Tipos de concept drift (sudden, incremental, gradual)
- **Fig. 2:** Ilustração de tipicalidade e excentricidade
- **Fig. 3:** Função m(k) — curva sigmoide de 1 a 3
- **Fig. 4:** Micro-clusters formados
- **Fig. 5:** Processo completo (micro → macro → assignment)
- **Figs. 10-12:** Avaliação prequential nos datasets

---

**Status:** Fichamento completo
**Próximo passo:** Implementar MicroTEDAclus para IDS IoT

