# Fichamento: Outside the Box - An Alternative Data Analytics Framework

**Referência completa:** Angelov, P. (2014). "Outside the box: an alternative data analytics framework." *Journal of Automation, Mobile Robotics and Intelligent Systems*, 8(2), pp.29-35. DOI: 10.14313/JAMRIS_2-2014/16

**Data de leitura:** 2026-01-03
**Área:** Machine Learning (Clustering Evolutivo)
**PDF:** https://www.jamris.org/index.php/JAMRIS/article/view/299/299

---

## 1. Objetivo do Artigo

Propor um framework alternativo para análise de dados chamado **TEDA (Typicality and Eccentricity Data Analytics)** que:
- Não depende das suposições da teoria de probabilidade tradicional
- É baseado em conceitos espaciais de **eccentricidade** e **tipicalidade**
- Pode ser usado para detecção de anomalias, clustering, classificação, predição e controle

---

## 2. Motivação

### 2.1 Crítica à Teoria de Probabilidade Tradicional (Frequentista)

**O que é abordagem frequentista:**
A probabilidade é definida como a **frequência relativa** de um evento quando o experimento é repetido infinitas vezes.

```
P(evento) = lim(n→∞) [número de ocorrências / n tentativas]
```

**Exemplo:** P(cara) = 0.5 significa "se lançar infinitas vezes, metade será cara".

**Três suposições problemáticas** (linhas 74-77):

| Suposição | O que exige | Problema em dados reais |
|-----------|-------------|------------------------|
| Independência | Amostras não se influenciam | Temperatura hoje depende de ontem |
| n → ∞ | Muitas observações | Às vezes temos apenas 10-50 dados |
| Distribuição conhecida | Assumir Gaussiana, etc. | Dados reais raramente são "puros" |

### 2.2 Crítica às Abordagens Alternativas

#### Belief Functions (Teoria de Dempster-Shafer)
- Atribui **graus de crença** a conjuntos de eventos
- Permite expressar incerteza parcial (ex: "60% de crença que é ataque OU falha")
- **Problema:** Requer especialista para definir os graus — é subjetivo

#### Possibility Theory (Teoria da Possibilidade)
- Distingue entre **possibilidade** (Π) e **necessidade** (N)
- Π(A) = "quão possível é A?" — pode somar mais que 1
- N(A) = "quão certo é A?" — N(A) = 1 - Π(não A)
- **Problema:** Também requer definição subjetiva por especialistas

| Abordagem | Pergunta | Soma |
|-----------|----------|------|
| Probabilidade | "Qual a chance?" | = 1 (obrigatório) |
| Possibilidade | "É compatível?" | ≤ n (livre) |
| Necessidade | "É certo?" | ≤ 1 |

#### First Principles Models (Modelos de Primeiros Princípios)
- Derivados de **leis fundamentais** (física, química, etc.)
- Exemplo: y(t) = v₀·sin(θ)·t - ½·g·t² (movimento de projétil)
- **Problema:** Não existe "Lei de Newton" para prever comportamento de rede

#### Expert-Based Models (Modelos Baseados em Especialistas)
- Construídos com **conhecimento humano** (regras, heurísticas)
- Exemplo: "SE pacotes > 10000/s ENTÃO alerta DDoS"
- **Problemas:** Subjetivo, trabalhoso, incompleto, não se adapta a mudanças

### 2.3 O que TEDA Propõe

> "The proposed new framework TEDA is a systematic methodology which does not require prior assumptions" (linhas 44-45)

**"No prior assumptions or kernels"** significa:
- **Sem prior assumptions:** Não assume distribuição (Gaussiana, etc.) antes de ver os dados
- **Sem kernels:** Não precisa escolher função núcleo nem bandwidth para estimar densidade

TEDA calcula tipicalidade **diretamente das distâncias** entre os dados, sem escolhas arbitrárias.

---

## 3. Conceitos Fundamentais

### 3.1 Accumulated Proximity (π)

**Definição:** Soma das distâncias de um ponto para todos os outros.

**Fórmula:**
```
π_j^k = Σ(i=1 to k) d_ij    onde d é distância (Euclidean, Mahalonobis, etc.)
```

### 3.2 Eccentricity (ξ)

**Definição:** Proporção normalizada da proximidade acumulada — mede quão "excêntrico" (longe dos outros) um ponto é.

**Fórmula:**
```
ξ_j^k = (2 × π_j^k) / Σ(i=1 to k) π_i^k
```

**Interpretação:**
- ξ alto → ponto está **longe** dos outros → **anômalo**
- ξ baixo → ponto está **perto** dos outros → **típico**
- Anomalia quando ξ > 1/k

### 3.3 Typicality (τ)

**Definição:** Complemento da eccentricidade — mede quão "típico" um ponto é.

**Fórmula:**
```
τ_j^k = 1 - ξ_j^k
```

**Interpretação:**
- τ alto → ponto é **típico** (próximo ao padrão)
- τ baixo → ponto é **atípico** (candidato a anomalia)
- Típico quando τ > 1/k

### 3.4 Relação entre Eccentricity e Typicality

```
τ = 1 - ξ

Σξ = 2        (soma das eccentricidades)
Στ = k - 2    (soma das tipicalidades)

0 < ξ < 1
0 < τ < 1
```

### 3.5 "Builds Upon Mutual Dependence"

Diferente da probabilidade que **ignora** relações entre amostras, TEDA **usa** essas relações:

| Abordagem | Como trata as amostras |
|-----------|------------------------|
| Probabilidade | Cada amostra é independente — conta apenas frequência |
| TEDA | Cada amostra influencia as outras — mede distâncias entre elas |

**Exemplo:** Dados {10, 12, 11, 25}
- Probabilidade: P(cada) = 1/4 (todos iguais)
- TEDA: τ(10,12,11) alto (próximos), τ(25) baixo (longe) → 25 é anômalo

A "dependência mútua" é a **estrutura espacial** dos dados — quem está perto de quem.

### 3.6 Métricas de Distância

O paper menciona (linhas 148-149): "This distance/proximity measure can be of any form, e.g. **Euclidean, Mahalonobis, cosine, Manhattan/city/L1**, etc."

#### 3.6.1 Euclidean Distance (Distância Euclidiana)

**O que é:** Distância "em linha reta" entre dois pontos.

**Fórmula:**
```
d(A, B) = √[Σᵢ (aᵢ - bᵢ)²]
```

**Exemplo:** A=(1,2), B=(4,6) → d = √[(3)² + (4)²] = 5

**Quando usar:**
- Features na mesma escala
- Dados contínuos
- Distância padrão no TEDA

**Limitação:** Sensível a escala; não considera correlação entre features.

#### 3.6.2 Manhattan Distance (L1 / City Block)

**O que é:** Soma dos deslocamentos em cada eixo — como um táxi em grid.

**Fórmula:**
```
d(A, B) = Σᵢ |aᵢ - bᵢ|
```

**Exemplo:** A=(1,2), B=(4,6) → d = |3| + |4| = 7

**Quando usar:**
- Dados esparsos (muitos zeros)
- Robustez a outliers
- Features independentes

#### 3.6.3 Mahalanobis Distance

**O que é:** Distância que considera **correlação** entre variáveis — mede desvios padrão do centro ajustando pela forma da distribuição.

**Fórmula:**
```
d(x, μ) = √[(x - μ)ᵀ · Σ⁻¹ · (x - μ)]
```
Onde Σ⁻¹ = inversa da matriz de covariância.

**Quando usar:**
- Features **correlacionadas** (ex: bytes_in ~ packets_in)
- Distribuição elíptica
- Detecção de anomalias multivariadas

**Limitação:** Precisa calcular matriz de covariância; requer dados suficientes.

#### 3.6.4 Cosine Distance (Distância do Cosseno)

**O que é:** Mede o **ângulo** entre vetores, ignorando magnitude.

**Fórmula:**
```
similaridade = (A · B) / (||A|| × ||B||)
distância = 1 - similaridade
```

**Exemplo:** A=(3,4), B=(6,8) → mesma direção → distância = 0

**Quando usar:**
- Direção mais importante que magnitude
- Dados de texto (TF-IDF)
- Alta dimensionalidade

#### 3.6.5 Comparação Resumida

| Distância | Fórmula | Sensível a Escala | Considera Correlação | Melhor Para |
|-----------|---------|-------------------|---------------------|-------------|
| **Euclidiana** | √Σ(a-b)² | Sim | Não | Dados contínuos, mesma escala |
| **Manhattan** | Σ\|a-b\| | Sim | Não | Dados esparsos, robustez |
| **Mahalanobis** | √[(x-μ)ᵀΣ⁻¹(x-μ)] | Não | **Sim** | Dados correlacionados |
| **Cosseno** | 1 - cos(θ) | **Não** | Não | Texto, direção > magnitude |

#### 3.6.6 Recomendação para IDS IoT

| Cenário | Distância Recomendada | Justificativa |
|---------|----------------------|---------------|
| Features normalizadas | **Euclidiana** | Simples, eficiente |
| Features correlacionadas | **Mahalanobis** | Captura correlação |
| Features com outliers extremos | **Manhattan** | Mais robusta |
| Embedding de comportamento | **Cosseno** | Padrão > intensidade |

**Nota:** No TEDA, a escolha da distância é a **única decisão** necessária (diferente de probabilidade que exige distribuição + parâmetros).

### 3.7 Normalização: Conceito e Aplicação no TEDA

#### 3.7.1 O que é Normalização nas Ciências Exatas

Normalização é o processo de **transformar valores para uma escala comum**, permitindo comparações justas entre grandezas diferentes.

```
Valor Normalizado = Valor Original / Fator de Escala
```

#### 3.7.2 Tipos Comuns de Normalização

| Tipo | Fórmula | Propriedade | Exemplo |
|------|---------|-------------|---------|
| **Por Soma** | x/Σx | Soma = 1 | Votos: 300/1000 = 30% |
| **Por Máximo** | x/max(x) | Máximo = 1 | Nota: 8/10 = 0.8 |
| **Min-Max** | (x-min)/(max-min) | Range [0,1] | Temp: (25-10)/(40-10) = 0.5 |
| **Z-Score** | (x-μ)/σ | Média=0, σ=1 | Altura: (190-170)/10 = +2.0 |
| **L2 (Unitário)** | x/\|\|x\|\| | Norma = 1 | Vetor: (3,4)/5 = (0.6, 0.8) |

#### 3.7.3 Por que Normalizar?

| Problema sem normalização | Solução com normalização |
|---------------------------|-------------------------|
| Escalas diferentes (km vs mm) | Valores comparáveis |
| Números absolutos sem contexto | Proporções com significado |
| Dominância de features grandes | Contribuição equilibrada |
| Difícil interpretar magnitudes | Fácil interpretar (0-1, %) |

#### 3.7.4 Por que Eccentricity é π Normalizado

**Problema com π (proximidade acumulada):**

π é um valor absoluto que depende de número de pontos, escala e unidade:
```
Dados A: {1, 2, 3}       →  π(2) = |2-1| + |2-3| = 2
Dados B: {10, 20, 30}    →  π(20) = |20-10| + |20-30| = 20
```
O ponto "do meio" tem π=2 em A e π=20 em B, mas **ambos são igualmente típicos**!

**Solução: Normalizar π para obter ξ**

A pergunta que queremos responder: "Quão excêntrico é este ponto **RELATIVO** aos outros?"

```
ξ_j = (2 × π_j) / Σπ_i
```

| Componente | Significado |
|------------|-------------|
| `π_j` | Proximidade acumulada do ponto j |
| `Σπ_i` | Soma de TODAS as proximidades |
| `π_j / Σπ_i` | Fração da proximidade total que j representa |
| `2 ×` | Fator de correção (cada distância contada 2x) |

#### 3.7.5 Exemplo Numérico Completo

```
Dados: {10, 12, 11, 25}

Proximidades acumuladas (π):
π(10) = |10-12| + |10-11| + |10-25| = 2 + 1 + 15 = 18
π(12) = |12-10| + |12-11| + |12-25| = 2 + 1 + 13 = 16
π(11) = |11-10| + |11-12| + |11-25| = 1 + 1 + 14 = 16
π(25) = |25-10| + |25-12| + |25-11| = 15 + 13 + 14 = 42

Σπ = 18 + 16 + 16 + 42 = 92

Eccentricity (ξ) — π normalizado:
ξ(10) = (2 × 18) / 92 = 0.39
ξ(12) = (2 × 16) / 92 = 0.35
ξ(11) = (2 × 16) / 92 = 0.35
ξ(25) = (2 × 42) / 92 = 0.91

Σξ = 0.39 + 0.35 + 0.35 + 0.91 = 2.0 ✓
```

**Interpretação:**
- ξ(25) = 0.91 >> threshold (1/k = 0.25) → **ANOMALIA**
- ξ(10,11,12) ≈ 0.35 ~ threshold → pontos típicos

#### 3.7.6 O Fator 2

Por que multiplicar por 2 no numerador?

Cada distância d(i,j) aparece **duas vezes** na soma total:
- Uma vez em π_i (distância de i para j)
- Uma vez em π_j (distância de j para i)

Então: `Σπ = 2 × (soma de todas as distâncias únicas)`

O fator 2 no numerador garante que `Σξ = 2` (propriedade útil do framework).

#### 3.7.7 Vantagem da Normalização no TEDA

```
SEM normalização:        COM normalização (ξ):
π = 42                   ξ = 0.91
"É muito? Pouco?"        "91% de uma unidade"
Depende do contexto      Sempre comparável
Threshold arbitrário     Threshold = 1/k (universal)
```

**Resultado:** ξ ∈ (0,1) sempre, independente de quantos pontos, qual escala, ou qual unidade. O mesmo threshold (1/k) funciona para qualquer dataset.

---

## 4. Propriedades Estatísticas

### 4.1 Propriedades da Eccentricity e Typicality

- **Bounded:** Valores sempre entre 0 e 1
- **Normalizado:** Soma de ξ = 2, soma de τ = k-2
- **Recursivo:** Pode ser atualizado incrementalmente sem recalcular tudo

### 4.2 Eccentricity e Typicality Normalizadas

```
ζ_j^k = ξ_j^k / 2           (soma = 1, range [0, 0.5])
t_j^k = τ_j^k / (k-2)       (soma = 1, range [0, 1/(k-2)])
```

Normalizadas somam 1 (similar a PDF), mas são **espacialmente conscientes**.

---

## 5. O Framework TEDA

### 5.1 Algoritmo (Algorithm 1 do paper)

```
Inicializar: k=1; x₁; X¹=||x₁||²; m¹=x₁; π₁¹=0

ENQUANTO dados disponíveis:
    1. Ler próximo ponto x_k (k := k+1)
    2. Atualizar:
       a. m^k (média)
       b. X^k (produto escalar)
    3. Para 1 ≤ j ≤ k, computar π_j^k
    4. Atualizar Σπ
    5. Para 1 ≤ j ≤ k, computar:
       a. ξ_j^k (eccentricity)
       b. τ_j^k (typicality)
       c. ζ_j^k (eccentricity normalizada)
       d. t_j^k (typicality normalizada)
FIM ENQUANTO
```

### 5.2 Fórmulas de Atualização Recursiva (Euclidean)

```
π_j^k = k × ||x_j - μ^k||² + X^k - ||μ^k||²

μ^k = ((k-1)/k) × μ^(k-1) + (1/k) × x_k       (média recursiva)

X^k = ((k-1)/k) × X^(k-1) + (1/k) × ||x_k||²  (produto escalar recursivo)

Σπ^k = Σπ^(k-1) + 2×π_k^k                      (soma recursiva)
```

### 5.3 Vantagens do Approach

1. **Sem prior assumptions:** Não assume distribuição
2. **Sem kernels:** Não precisa escolher função/bandwidth
3. **Recursivo:** Computacionalmente eficiente para streams
4. **Funciona com poucos dados:** k ≥ 3 é suficiente
5. **Usa dependência mútua:** Captura estrutura espacial dos dados
6. **Closed-form:** Expressões analíticas fechadas

---

## 6. Aplicações Demonstradas

| Aplicação | Dataset | Resultados |
|-----------|---------|------------|
| Temperatura | {20, 12, 10} °C (exemplo didático) | 20°C identificado como excêntrico (ξ=0.45 > 1/3) |
| Temperatura | {20, 12, 10, 17} °C | Padrão rebalanceado, todos mais típicos |
| Precipitação | Bristol, UK, Jan 2014 (14 dias) | 20.2mm no Ano Novo identificado como atípico |

---

## 7. Seções do Paper

### 7.1 Anomaly Detection (Seção 4)
- Baseado em eccentricity: ξ > 1/k indica anomalia
- Aplicável a: processamento de imagens, detecção de falhas, modelagem de comportamento

### 7.2 Clustering (Seção 5)
- Baseado em typicality: pontos com maior τ são protótipos/centros
- "Data clouds" em vez de clusters tradicionais (sem forma/parâmetros específicos)

### 7.3 Classification (Seção 6)
- Usa valores locais de ξ, τ por classe
- Zero, first ou higher order classifiers

### 7.4 Prediction and Control (Seção 7)
- Multi-model principle com sub-modelos locais simples
- Decomposição do espaço de dados em regiões locais

---

## 8. Limitações Identificadas

### 8.1 Reconhecida pelo Autor

> "TEDA can work efficiently with any data **except pure random processes**" (linhas 99-100)

**Por que TEDA falha em processos puramente aleatórios:**
- TEDA busca **dependência espacial** que **não existe** em dados aleatórios puros
- Para dado justo, cada face tem mesma probabilidade — não há padrão espacial
- TEDA converge para tipicalidade igual para todos, igual à probabilidade
- **Mais trabalho computacional, mesmo resultado**

**Para processos aleatórios puros (dados, moedas), probabilidade clássica é melhor.**

### 8.2 Outras Limitações (implícitas)
- Paper apresenta apenas exemplos didáticos pequenos
- Não há comparação quantitativa com outros métodos
- Escolha da métrica de distância ainda é necessária

---

## 9. Relação com Minha Pesquisa

### 9.1 Base para MicroTEDAclus (Maia et al., 2020)

MicroTEDAclus **estende** TEDA com:
- Micro-clusters em vez de pontos individuais
- Mixture of typicalities
- Operações de merge/split para clusters

As fórmulas de eccentricidade e tipicalidade do MicroTEDAclus derivam diretamente deste paper.

### 9.2 Aplicação em IDS IoT

| Vantagem TEDA | Aplicação em IDS |
|---------------|------------------|
| Sem prior assumptions | Não precisa assumir distribuição do tráfego |
| Recursivo | Adequado para streaming em tempo real |
| Detecta anomalias | Identifica ataques como pontos excêntricos |
| Usa dependência | Captura correlação temporal do tráfego |

**Diferença de regras de especialista:**
- Regras (Snort/Suricata): Detectam ataques **conhecidos**
- TEDA: Pode detectar ataques **novos** (zero-day) como pontos atípicos

### 9.3 Contribuição para Dissertação

Este paper é **fundamental** para:
- **Fundamentação Teórica:** Define os conceitos matemáticos de ξ e τ
- **Metodologia:** Justifica abordagem não-probabilística para dados de rede
- **Implementação:** Fórmulas recursivas para sistema em tempo real

---

## 10. Citações Importantes

> "Unlike purely random processes, such as throwing dices, tossing coins... real life processes of interest **do violate** the main assumptions which the traditional probability theory requires." (linhas 32-36)

> "It does not require independence of the individual data samples; on the contrary, the proposed approach **builds upon their mutual dependence**." (linhas 90-93)

> "The proposed new framework TEDA is a systematic methodology which **does not require prior assumptions** and can be used for development of a range of methods for anomalies and fault detection, image processing, clustering, classification, prediction, control, filtering, regression, etc." (linhas 43-48)

> "For such pure random data the **traditional probability theory is the best tool** to be used. However, for real data processes – which are the majority of the cases – we argue that TEDA is better justified." (linhas 101-106)

---

## 11. Referências Relevantes do Paper

| # | Referência | Por que é relevante |
|---|------------|---------------------|
| [2] | Osherson & Smith (1997) "On typicality and vagueness" | Conceito filosófico de tipicalidade |
| [3] | Angelov (2012) "Autonomous Learning Systems" | Livro com detalhes de TEDA e sistemas evolutivos |
| [7] | Zadeh (1965) "Fuzzy sets" | Comparação com funções de pertinência fuzzy |
| [8] | Angelov & Yager (2012) "AnYa framework" | Introduz conceito de "data clouds" |

---

## 12. Notas Pessoais

### Insights da Leitura

1. **Honestidade científica:** Angelov reconhece que TEDA não é para tudo — probabilidade é melhor para processos puramente aleatórios

2. **Filosofia central:** Probabilidade pergunta "quantas vezes X apareceu?"; TEDA pergunta "onde X está em relação aos outros?"

3. **Para IDS:** Tráfego de rede NÃO é como jogar dados — tem correlação temporal, padrões de uso. TEDA é adequado.

4. **Simplicidade:** As fórmulas são elegantes e computacionalmente eficientes (recursivas)

### Dúvidas Esclarecidas

- **Frequentista:** Probabilidade como frequência relativa no limite infinito
- **Kernels:** Funções para estimar densidade (Gaussiano, Epanechnikov) — TEDA não precisa
- **Mutual dependence:** TEDA usa distâncias entre todos os pares, não trata pontos como independentes

---

## 13. Checklist de Leitura

- [x] Li o abstract
- [x] Li a introdução
- [x] Entendi a definição de eccentricity
- [x] Entendi a definição de typicality
- [x] Copiei as fórmulas principais
- [x] Entendi o algoritmo TEDA
- [x] Li os experimentos
- [x] Li a conclusão
- [ ] Identifiquei relação completa com MicroTEDAclus
- [x] Identifiquei aplicação na minha pesquisa

---

## 14. Glossário de Termos

| Termo | Definição |
|-------|-----------|
| **Frequentista** | Abordagem que define probabilidade como frequência relativa em infinitas repetições |
| **Prior assumption** | Suposição feita antes de ver os dados (ex: assumir distribuição Gaussiana) |
| **Kernel** | Função usada para estimar densidade de dados (ex: Gaussiano, Epanechnikov) |
| **Bandwidth** | Parâmetro que controla a largura do kernel |
| **Eccentricity (ξ)** | Medida de quão "excêntrico" (distante dos outros) um ponto é |
| **Typicality (τ)** | Medida de quão "típico" (próximo ao padrão) um ponto é |
| **Mutual dependence** | Relação espacial entre pontos de dados — quem está perto de quem |
| **Data cloud** | Agrupamento de dados sem forma, parâmetros ou fronteiras específicas |
| **First principles** | Modelos derivados de leis fundamentais (física, química, etc.) |
| **Euclidean distance** | Distância em linha reta: √Σ(a-b)² — sensível a escala |
| **Manhattan distance** | Soma dos deslocamentos por eixo: Σ\|a-b\| — robusta a outliers |
| **Mahalanobis distance** | Distância que considera correlação via matriz de covariância |
| **Cosine distance** | Mede ângulo entre vetores, ignora magnitude: 1 - cos(θ) |
| **Belief functions** | Teoria de Dempster-Shafer — graus de crença em conjuntos de eventos |
| **Possibility theory** | Teoria que distingue possibilidade (Π) de necessidade (N) |
| **Normalização** | Transformar valores para escala comum, permitindo comparações justas |
| **Normalização por soma** | x/Σx — valores como fração do total (soma = 1) |
| **Normalização Min-Max** | (x-min)/(max-min) — mapeia para intervalo [0,1] |
| **Z-Score** | (x-μ)/σ — mede em unidades de desvio padrão |

---

**Status:** Em progresso (~85% completo)
**Última atualização:** 2026-01-03
**Próximos passos:** Relacionar com MicroTEDAclus (Maia 2020)
