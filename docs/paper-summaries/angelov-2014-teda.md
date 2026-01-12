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

#### 4.2.1 O Problema: ξ e τ Não Somam 1

As versões "cruas" têm somas específicas:
```
Σξ = 2        ← soma das eccentricidades de todos os pontos
Στ = k - 2    ← soma das tipicalidades de todos os pontos
```

Isso dificulta a interpretação como "distribuição" — valores não são diretamente comparáveis a probabilidades.

#### 4.2.2 A Solução: Normalização Adicional

O paper propõe (equações 5 e 6):

```
ζ_j = ξ_j / 2           → Σζ = 1   (eccentricity normalizada)
τ̃_j = τ_j / (k - 2)    → Στ̃ = 1   (typicality normalizada)
```

**Propriedades resultantes:**
- **0 < ζ < 1** para cada ponto
- **Σζ = 1** para todos os pontos
- **0 < τ̃ < 1/(k-2)** para cada ponto
- **Στ̃ = 1** para todos os pontos

#### 4.2.3 "Resembles Probability Distribution Function (PDF)"

**Por que a semelhança?**

Uma PDF tem duas propriedades fundamentais:
1. **f(x) ≥ 0** para todo x (não-negativa)
2. **∫f(x)dx = 1** (integral/soma = 1)

A eccentricity normalizada ζ satisfaz ambas:
1. **0 < ζ < 1** (positiva e limitada)
2. **Σζ = 1** (soma = 1)

**Porém, há diferenças conceituais:**

| Aspecto | PDF (Probabilidade) | ζ (TEDA) |
|---------|---------------------|----------|
| Suposição prévia | Distribuição conhecida (Gaussiana, etc.) | Nenhuma |
| Independência | Amostras devem ser independentes | Usa dependência espacial |
| O que mede | Frequência esperada de ocorrência | Distância relativa aos outros |
| Interpretação | "Chance de um novo dado cair aqui" | "Quão excêntrico é este dado" |
| Base | Modelo assumido | Dados observados apenas |

**Citação do paper (linhas 356-362):**
> "Normalised eccentricity and typicality resemble probability distribution function (pdf) in that they sum to 1, but they are different as they do not require the prior assumptions that are a must for the probability theory and they represent both the **spatial distribution pattern** and the **frequency of occurrence** of a data sample."

#### 4.2.4 Exemplo Numérico: ξ vs ζ

```
Dados: {2, 3, 3, 10} (k=4 pontos)

Proximidades acumuladas (π):
π₁ = |2-2| + |2-3| + |2-3| + |2-10| = 0 + 1 + 1 + 8 = 10
π₂ = |3-2| + |3-3| + |3-3| + |3-10| = 1 + 0 + 0 + 7 = 8
π₃ = |3-2| + |3-3| + |3-3| + |3-10| = 1 + 0 + 0 + 7 = 8
π₄ = |10-2| + |10-3| + |10-3| + |10-10| = 8 + 7 + 7 + 0 = 22

Σπ = 10 + 8 + 8 + 22 = 48

Eccentricity crua (ξ):
ξ₁ = (2 × 10) / 48 = 0.417
ξ₂ = (2 × 8) / 48 = 0.333
ξ₃ = (2 × 8) / 48 = 0.333
ξ₄ = (2 × 22) / 48 = 0.917

Σξ = 0.417 + 0.333 + 0.333 + 0.917 = 2.0 ✓

Eccentricity normalizada (ζ):
ζ₁ = ξ₁ / 2 = 0.208
ζ₂ = ξ₂ / 2 = 0.167
ζ₃ = ξ₃ / 2 = 0.167
ζ₄ = ξ₄ / 2 = 0.458

Σζ = 0.208 + 0.167 + 0.167 + 0.458 = 1.0 ✓ (como PDF!)
```

**Interpretação:**
- Se fosse probabilidade uniforme: cada ponto teria 25% (1/4)
- ζ₄ = 0.458 → o ponto 10 "concentra" **45.8%** da excentricidade total
- Isso mostra que ponto 10 é **muito mais excêntrico** que os outros

#### 4.2.5 Diferença Conceitual Fundamental

```
┌─────────────────────────────────────────────────────────────┐
│ PDF responde: "Qual a CHANCE de um novo dado cair aqui?"   │
│   → Baseado em MODELO assumido (Gaussiana, etc.)           │
│                                                             │
│ ζ responde: "Quão EXCÊNTRICO é este dado específico?"      │
│   → Baseado APENAS nas DISTÂNCIAS observadas               │
│                                                             │
│ Ambos somam 1, mas medem coisas diferentes:                │
│   - PDF: distribuição ESPERADA (modelo teórico)            │
│   - ζ: distribuição OBSERVADA (dados reais)                │
└─────────────────────────────────────────────────────────────┘
```

#### 4.2.6 Por Que Isso É Útil?

A semelhança com PDF permite:

1. **Interpretação intuitiva**: "Este ponto tem 45% da excentricidade" é fácil de entender
2. **Comparação justa**: Valores entre 0-1 permitem comparar datasets diferentes
3. **Threshold natural**: 1/k é o valor "esperado" se todos fossem igualmente excêntricos
4. **Compatibilidade**: Pode-se usar ζ em fórmulas que esperam valores tipo-probabilidade

**Sem as limitações da probabilidade:**
- Funciona com k ≥ 3 amostras (não precisa de infinitas)
- Usa a estrutura espacial (não assume independência)
- Não precisa assumir distribuição prévia

#### 4.2.7 Analogia com Histograma (linhas 379-382)

O paper também menciona que tipicalidade é análoga a histogramas:

> "The typicality can also be seen as an analogue to the histograms of distributions, but it is in a **closed analytical form** and does take into account the **mutual influence** of the neighbouring data samples/observations."

| Aspecto | Histograma | Tipicalidade (τ) |
|---------|------------|------------------|
| Forma | Discreto (bins) | Analítico (fórmula) |
| Depende de | Escolha de bins (arbitrário) | Apenas dados |
| Considera vizinhança | Não (conta frequência apenas) | Sim (distâncias) |

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

#### 5.2.1 Fórmulas Intermediárias

```
π_j^k = k × ||x_j - μ^k||² + X^k - ||μ^k||²

μ^k = ((k-1)/k) × μ^(k-1) + (1/k) × x_k       (média recursiva)

X^k = ((k-1)/k) × X^(k-1) + (1/k) × ||x_k||²  (produto escalar recursivo)

Σπ^k = Σπ^(k-1) + 2×π_k^k                      (soma recursiva)
```

#### 5.2.2 Fórmula Direta para Eccentricity

O paper demonstra que, para distância Euclidiana quadrada, a eccentricity pode ser calculada **diretamente** usando apenas média e variância:

```
ξ_j^k = 1/k + ||x_j - μ^k||² / (k × σ²)
```

Onde:
- `k` = número de amostras
- `μ^k` = média de todas as amostras
- `σ²` = variância (dispersão média)
- `||x_j - μ^k||²` = distância quadrada do ponto j ao centro

**Interpretação dos componentes:**

| Componente | Significado |
|------------|-------------|
| `1/k` | Eccentricidade mínima (se ponto estiver exatamente na média) |
| `(x - μ)²` | Quão longe o ponto está do centro |
| `k × σ²` | Fator de escala (normaliza pela dispersão total) |

#### 5.2.3 Derivação Matemática: Da Definição Original à Fórmula Recursiva

**Problema:** A definição original requer O(n²) cálculos de distância. Como chegar à fórmula O(n)?

##### Passo 1: Definição Original

Proximidade acumulada do ponto j:
```
π_j = Σᵢ ||x_j - x_i||²    (soma das distâncias quadradas a todos os outros pontos)
```

##### Passo 2: Identidade Algébrica Fundamental (Teorema de Huygens-Steiner)

Existe uma identidade clássica que relaciona distâncias pairwise com distância à média:

```
┌────────────────────────────────────────────────────────────────┐
│  Σᵢ ||x_j - x_i||² = k·||x_j - μ||² + k·σ²                     │
│                                                                │
│  "Soma das distâncias a todos" = "k × (distância à média)²"   │
│                                   + "k × variância"            │
└────────────────────────────────────────────────────────────────┘
```

**Nomes desta identidade em diferentes áreas:**

| Área | Nome | Forma Clássica |
|------|------|----------------|
| **Física/Mecânica** | Teorema de Steiner (Eixos Paralelos) | I = I_cm + m·d² |
| **Estatística** | Fórmula de König-Huygens | Var(X) = E[X²] - E[X]² |
| **ML/Data Science** | Decomposição do Centróide | Σd_ij² = k·d_μ² + k·σ² |

**Teorema de Huygens-Steiner (Física):**
> "O momento de inércia em relação a qualquer eixo é igual ao momento de inércia em relação ao centro de massa, mais a massa vezes a distância ao quadrado."

Na versão para dados, a "inércia" é a soma das distâncias quadradas, e o "centro de massa" é a média μ.

**Fórmula de König-Huygens (Estatística):**
```
Var(X) = E[X²] - (E[X])²
σ² = X - ||μ||²
```

Esta é exatamente a relação usada no Passo 4 para expressar σ² em termos de X e μ.

##### Passo 3: Prova da Identidade

Reescrevendo cada distância em termos de desvios da média:
```
||x_j - x_i||² = ||(x_j - μ) - (x_i - μ)||²
```

Expandindo o quadrado:
```
= ||x_j - μ||² - 2(x_j - μ)·(x_i - μ) + ||x_i - μ||²
```

Somando sobre todos os i = 1, 2, ..., k:
```
Σᵢ ||x_j - x_i||² = Σᵢ ||x_j - μ||² - 2(x_j - μ)·Σᵢ(x_i - μ) + Σᵢ ||x_i - μ||²
                    └─────────────┘   └────────────────────┘   └────────────────┘
                      k·||x_j - μ||²           = 0                   k·σ²
```

**O termo do meio é ZERO!** Por quê?
```
Σᵢ (x_i - μ) = Σᵢ x_i - k·μ = k·μ - k·μ = 0
```
(A soma dos desvios em relação à média é sempre zero — propriedade fundamental da média!)

**Resultado:**
```
π_j = k·||x_j - μ||² + k·σ²
```

##### Passo 4: O Papel do Produto Escalar X

Precisamos expressar σ² de forma que possa ser atualizada incrementalmente.

A **variância** tem a fórmula:
```
σ² = E[||x||²] - ||E[x]||² = E[||x||²] - ||μ||²
```

O paper define **X** como a média dos quadrados das normas:
```
X = (1/k) · Σᵢ ||x_i||²    (média de ||x||²)
```

Portanto:
```
σ² = X - ||μ||²
```

| Estatística | O que guarda | Atualização |
|-------------|--------------|-------------|
| **μ** (média) | Centro de massa dos dados | O(1) por novo ponto |
| **X** (média de \|\|x\|\|²) | "Energia" média dos dados | O(1) por novo ponto |
| **σ² = X - \|\|μ\|\|²** | Variância (dispersão) | Calculado de μ e X |

##### Passo 5: Substituição

Substituindo σ² = X - ||μ||² na fórmula de π:
```
π_j = k·||x_j - μ||² + k·σ²
    = k·||x_j - μ||² + k·(X - ||μ||²)
    = k·||x_j - μ||² + k·X - k·||μ||²
```

##### Passo 6: Da Fórmula de π para ξ

A eccentricity é definida como:
```
ξ_j = 2π_j / Σπ
```

Após simplificação algébrica (referenciada em [3],[4] do paper), obtém-se:
```
ξ_j = 1/k + ||x_j - μ||² / (k·σ²)
```

##### Resumo da Derivação

```
PASSO 1: Definição
  π_j = Σᵢ ||x_j - x_i||²               (soma de todas as distâncias)

PASSO 2: Identidade algébrica
  Σᵢ ||x_j - x_i||² = k·||x_j - μ||² + k·σ²

PASSO 3: Expressar variância
  σ² = X - ||μ||²                       (X = média de ||x||²)

PASSO 4: Substituir
  π_j = k·||x_j - μ||² + k·(X - ||μ||²)

PASSO 5: Eccentricity simplificada
  ξ_j = 2π_j / Σπ  →  simplifica para  →  ξ_j = 1/k + ||x_j - μ||²/(k·σ²)

RESULTADO:
  ξ depende APENAS de: μ (média), σ² (variância), x_j (ponto atual)
  Complexidade: O(1) para calcular ξ de um novo ponto!
```

##### Visualização: O(n²) vs O(n)

```
ORIGINAL (O(n²)):                    RECURSIVA (O(n)):
    x₁ ←→ x₂                                  x₁
    x₁ ←→ x₃     Calcular TODAS                ↘
    x₁ ←→ x₄     as distâncias          x₂ ──→ μ ←── x₃
    x₂ ←→ x₃     pairwise                      ↗
    x₂ ←→ x₄                                  x₄
    x₃ ←→ x₄
                                       Cada ponto só calcula
    6 distâncias para 4 pontos         distância à MÉDIA (1 cada)
    n(n-1)/2 no geral                  + σ² vem "de brinde" via X
```

#### 5.2.4 Aplicabilidade a Outras Métricas de Distância

A derivação acima usa **propriedades específicas da distância Euclidiana quadrada**:
- Expansão do quadrado: `||a-b||² = ||a||² - 2a·b + ||b||²`
- Cancelamento do termo cruzado via propriedade da média

**Para outras métricas:**

| Métrica | Fórmula Recursiva? | Referência no Paper |
|---------|-------------------|---------------------|
| **Euclidiana²** | ✅ ξ = 1/k + \|\|x-μ\|\|²/(k·σ²) | [3], [4] |
| **Mahalanobis²** | ✅ Similar, com matriz Σ⁻¹ | [6] |
| **Cosseno** | ✅ Usa produto interno normalizado | [5] |
| **Manhattan (L1)** | ❌ Não tem forma fechada simples | Usa O(n²) ou aproximações |

**Citação do paper (linhas 390-395):**
> "It can also be proven that both eccentricity and typicality can be calculated recursively by updating only the global or local mean, μ and scalar product, X for the cases when **Euclidean square distance** [3],[4] is used and similarly if **cosine** [5] or **Mahalonobis square distance** [6] are used."

**Nota:** Para Manhattan, a propriedade algébrica `||a-b||² = ||a||² - 2a·b + ||b||²` não se aplica (pois |a-b| ≠ |a|² - 2ab + |b|²), então a simplificação não é possível.

#### 5.2.5 Implementação Python (Lab de Clustering)

Código do laboratório que implementa a fórmula recursiva:

```python
def calculate_eccentricity_batch(X):
    """
    Calcula eccentricidade para cada ponto em um batch.
    Eccentricidade alta = ponto é outlier/diferente
    """
    n = len(X)
    mean = np.mean(X, axis=0)                           # μ

    # Distância de cada ponto à média
    distances_to_mean = np.sum((X - mean) ** 2, axis=1) # ||x_j - μ||²

    # Variância total
    variance = np.mean(distances_to_mean)               # σ²

    # Eccentricidade: ξ = 1/k + (x - μ)² / (k × σ²)
    if variance > 0:
        eccentricity = (1/n) + (distances_to_mean / (n * variance))
    else:
        eccentricity = np.ones(n) / n  # Caso degenerado

    return eccentricity
```

**Mapeamento código → fórmula:**

| Paper | Código | Significado |
|-------|--------|-------------|
| `k` | `n` | Número de pontos |
| `μ^k` | `mean` | Média do dataset |
| `\|\|x_j - μ\|\|²` | `distances_to_mean` | Distância quadrada ao centro |
| `σ²` | `variance` | Variância (dispersão média) |
| `1/k` | `1/n` | Termo base (eccentricidade mínima) |

#### 5.2.6 Verificação Numérica

Usando o exemplo {2, 3, 3, 10}:

```python
import numpy as np

X = np.array([[2], [3], [3], [10]])
n = 4
mean = np.mean(X)  # = 4.5

distances_to_mean = (X.flatten() - mean) ** 2
# = [6.25, 2.25, 2.25, 30.25]

variance = np.mean(distances_to_mean)  # = 10.25

# Fórmula recursiva: ξ = 1/k + (x - μ)² / (k × σ²)
xi = (1/n) + (distances_to_mean / (n * variance))
# xi = 0.25 + [6.25, 2.25, 2.25, 30.25] / 41.0
# xi = 0.25 + [0.152, 0.055, 0.055, 0.738]
# xi = [0.402, 0.305, 0.305, 0.988]

print(f"ξ = {xi}")       # [0.402, 0.305, 0.305, 0.988]
print(f"Σξ = {sum(xi)}") # 2.0 ✓
```

**Resultado:**
```
ξ(2)  = 0.402  (típico, < 1)
ξ(3)  = 0.305  (típico, < 1)
ξ(3)  = 0.305  (típico, < 1)
ξ(10) = 0.988  (ANOMALIA! >> threshold 1/k = 0.25)
```

#### 5.2.7 Por Que Isso É Importante para Streaming

A fórmula recursiva permite **atualização incremental**:

```python
# Streaming: atualização O(1) por novo ponto
def update_streaming(new_point, k, old_mean, old_variance):
    # Atualiza média incrementalmente
    new_mean = old_mean + (new_point - old_mean) / (k + 1)

    # Atualiza variância incrementalmente (Welford's algorithm)
    # ... (detalhes no MicroTEDAclus)

    # Calcula ξ do novo ponto em O(1)
    dist_sq = (new_point - new_mean) ** 2
    xi_new = 1/(k+1) + dist_sq / ((k+1) * new_variance)

    return xi_new, new_mean, new_variance
```

**Implicação:** TEDA pode processar **milhões de pontos em streaming** sem armazenar histórico, apenas mantendo μ e σ².

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

## 7. Seções do Paper (Detalhado)

### 7.1 Anomaly Detection (Seção 4)

**Princípio:** Detecção baseada em eccentricity — pontos com ξ alto são candidatos a anomalias.

**Regra básica:**
```
SE ζ > 1/k  →  Ponto é ANOMALIA SUSPEITA
SE τ < 1/k  →  Ponto é ATÍPICO
```

**Vantagens sobre probabilidade:**

| Aspecto | Probabilidade | TEDA (Eccentricity) |
|---------|---------------|---------------------|
| Exemplo {20, 12, 10}°C | P(cada) = 1/3 (iguais!) | ξ(20) = 0.45 >> ξ(12) = 0.25 |
| Precisa de distribuição | Sim (Gaussiana, etc.) | Não |
| Quantidade de dados | Muitos (para convergir) | Funciona com k ≥ 3 |
| Forma analítica | Aproximada (KDE, histograma) | Exata (fórmula fechada) |

**Grau de severidade:**
- Não é binário (anomalia/normal)
- ζ indica **quão anômalo** é o ponto
- "How bigger ζ is in comparison with 1/k" (linha 1504)

**Aplicações citadas:**
- Processamento de imagens e vídeo
- Detecção de falhas
- Modelagem de comportamento de usuários
- Eventos extremos (clima, terremotos, terrorismo)

---

### 7.2 Clustering e Data Clouds (Seção 5) — ESSENCIAL PARA MicroTEDAclus

#### 7.2.1 Conceito de "Data Cloud"

**Definição (linhas 1514-1517):**
> "The term 'data cloud' was introduced in the so called AnYa framework [8] and **differs from clusters** by the fact that data clouds have **no specific shape, parameters, and boundaries**."

| Característica | Cluster Tradicional | Data Cloud (TEDA) |
|----------------|--------------------|--------------------|
| Forma | Esférica, elíptica, etc. | Livre (sem forma fixa) |
| Parâmetros | k (número), raio, etc. | Nenhum pré-definido |
| Fronteiras | Definidas (hard/soft) | Flexíveis |
| Centro | Centróide calculado | Ponto com maior τ (focal point) |

#### 7.2.2 Formação de Data Clouds

**Algoritmo conceitual (linhas 1520-1536):**

```
1. Encontrar ponto com MAIOR τ → Primeiro "focal point" (protótipo)
2. Definir "zona de influência" (raio) ao redor do focal point
3. Para pontos FORA da zona de influência:
   - SE τ > 1/k → Candidato a novo focal point
   - Selecionar o de maior τ como próximo protótipo
4. Repetir até não haver mais candidatos
```

**Critério para ser protótipo:** τ > 1/k (tipicalidade acima da média)

#### 7.2.3 Tipicalidade Local vs Global

**Global:** Calculada sobre TODOS os k pontos
```
τ_global = tipicalidade em relação ao dataset inteiro
```

**Local:** Calculada sobre pontos de UMA data cloud específica
```
τ_local = tipicalidade em relação apenas aos membros do cluster
```

Isso permite extrair distribuições separadas por cluster:
```
Data cloud "azul": ξ_blue, τ_blue
Data cloud "vermelha": ξ_red, τ_red
```

#### 7.2.4 Eficiência em Streaming (Memória)

**O que manter por data cloud (linhas 1551-1554):**
```
Para cada data cloud i*, guardar apenas:
- x_i*  (ponto focal/protótipo)
- μ_i*  (média local)
- X_i*  (produto escalar local)
- Σπ    (soma das proximidades)
```

**NÃO precisa guardar todos os pontos!** Apenas as estatísticas agregadas.

#### 7.2.5 Aspectos Dinâmicos (linhas 1581-1584)

> "An important aspect is the **dynamic nature** of the data streams and their **order dependency**. One can chose to have a **forgetting factor** or mechanism to introduce the importance of the time instances when a particular data sample was read."

**Implicações:**
- Data clouds podem **evoluir** com o tempo
- Pontos antigos podem ter peso menor (forgetting factor)
- Ordem de chegada importa (streaming não é batch)

#### 7.2.6 Por que "Fora da Zona + τ > 1/k" para Novo Protótipo?

**O Problema:** Como formar MÚLTIPLOS clusters sem definir k (número de clusters)?

Se apenas pegássemos o ponto com maior τ, teríamos um único cluster. Precisamos de critério para identificar centros de OUTROS grupos.

**A Solução: Duas Condições Simultâneas**

```
Para ser protótipo de um NOVO cluster, o ponto deve:
1. Estar FORA da zona de influência → "Longe o suficiente do primeiro protótipo"
2. Ter τ > 1/k                      → "Típico o suficiente para ser centro"
```

**Por que as duas condições?**

| Condição | O que garante | Sem ela... |
|----------|---------------|------------|
| Fora da zona | Clusters são **separados** | Protótipos muito próximos = mesmo cluster |
| τ > 1/k | Protótipo é **representativo** | Outlier isolado viraria centro (ruim!) |

**Exemplo visual:**
```
Dados: ● ● ● ●    ○ ○ ○ ○ ○        ✗ (outlier)
       grupo A       grupo B

Passo 1: Maior τ global → ● central (protótipo de A)
Passo 2: Zona de influência ao redor do ● central
Passo 3: Fora da zona, quem tem τ > 1/k?
         → ○ central é candidato (τ alto, fora da zona) ✓
         → ✗ outlier NÃO é (τ < 1/k, muito excêntrico) ✗
Passo 4: ○ com maior τ → protótipo de B
```

**⚠️ ATENÇÃO:** O paper NÃO define como calcular a "zona de influência" — é deixado como escolha de design. Ver seção 8.3 para detalhes desta lacuna.

#### 7.2.7 Por que Guardar Apenas {μ, X, Σπ} por Cluster?

**O Problema: Streaming com Milhões de Pontos**

```
Cenário IDS IoT:
- 10.000 pacotes/segundo
- 1 hora = 36 milhões de pontos
- Guardar todos = 💥 MEMÓRIA EXPLODE
```

**A Solução: Estatísticas Suficientes**

Graças ao **Teorema de Huygens-Steiner**, a fórmula recursiva só precisa de:

```
ξ_j = 1/k + ||x_j - μ||² / (k × σ²)

onde σ² = X - ||μ||²
```

**Para calcular ξ de um NOVO ponto, basta ter:**
- **μ** (média) — centro do cluster
- **X** (média de ||x||²) — para calcular σ²
- **k** (contador) — quantos pontos já processados
- **Σπ** (soma de proximidades) — para eccentricity global

**Comparação de Memória:**

| Abordagem | Memória (1M pontos, 10 clusters) |
|-----------|----------------------------------|
| Batch (todos os pontos) | O(1.000.000 × n_features) ≈ GB |
| Streaming (estatísticas) | O(10 × 4) = 40 valores ≈ bytes |

**Por que funciona?** O Teorema de Huygens-Steiner "comprime" toda a informação sobre distâncias pairwise em apenas μ e X:
```
Σᵢ ||x_j - x_i||² = k·||x_j - μ||² + k·σ²
                    └─────────────────────┘
                    Calculável só com μ, X, k
```

#### 7.2.8 O Significado de 1/k como Threshold

**O que 1/k representa?**

Para métricas normalizadas (ζ e τ̃) que somam 1:

```
Σζ = 1  (soma das eccentricidades normalizadas)
Στ̃ = 1  (soma das tipicalidades normalizadas)

Se TODOS os k pontos fossem igualmente típicos/excêntricos:
   Cada um teria exatamente 1/k do total

1/k = "cota justa" = valor esperado sob distribuição uniforme
```

**A Lógica do Threshold:**

| Comparação | Interpretação | Conclusão |
|------------|---------------|-----------|
| ζ > 1/k | Mais excêntrico que a média | Candidato a **anomalia** |
| ζ < 1/k | Menos excêntrico que a média | Ponto **normal** |
| τ̃ > 1/k | Mais típico que a média | Candidato a **protótipo** |
| τ̃ < 1/k | Menos típico que a média | Não serve como centro |

**Exemplo numérico:**
```
Dados: {2, 3, 3, 10}  (k = 4)
Threshold: 1/k = 0.25

ζ(2)  = 0.208 < 0.25 → típico
ζ(3)  = 0.167 < 0.25 → típico
ζ(3)  = 0.167 < 0.25 → típico
ζ(10) = 0.458 > 0.25 → ANÔMALO! (quase 2× a cota justa)
```

**Conexão com a fórmula recursiva:**
```
ξ = 1/k + ||x - μ||² / (k × σ²)
    └─┘   └────────────────────┘
   base    penalidade por distância

- Ponto na média (x = μ): ξ = 1/k (mínimo possível)
- Ponto longe da média: ξ >> 1/k (anômalo)
```

**Por que 1/k é elegante:**
1. **Adapta-se automaticamente** ao tamanho do dataset
2. **Não requer cálculo adicional** — deriva direto de k
3. **Interpretação clara** — "acima/abaixo da média esperada"

#### 7.2.9 Conexão com MicroTEDAclus

| TEDA (Angelov 2014) | MicroTEDAclus (Maia 2020) |
|---------------------|---------------------------|
| Data cloud = conjunto de pontos | Micro-cluster = resumo estatístico |
| Focal point = ponto com maior τ | Centróide do micro-cluster |
| Zona de influência (raio) | Merge/split baseado em tipicalidade |
| Tipicalidade local | "Mixture of typicalities" |
| Streaming básico | Streaming com concept drift |
| Guardar {μ, X, Σπ} | Mesma ideia, mais refinada |
| Threshold 1/k | Chebyshev test adaptativo |

**Esta seção é a PONTE CONCEITUAL para entender MicroTEDAclus!**

---

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

### 8.3 Zona de Influência NÃO Definida (Lacuna Importante)

**O que o paper diz (linhas 1526-1527):**
> "For example, a zone of influence/radius **can be defined**..."

A linguagem "**can be defined**" indica que é uma **escolha de design**, não uma fórmula fixa.

**O que isso significa:**

| Aspecto | O que o paper DIZ | O que NÃO diz |
|---------|-------------------|---------------|
| Conceito | "zona ao redor do protótipo" | Fórmula específica |
| Uso | "pontos fora são candidatos a novos protótipos" | Como calcular o raio |
| Flexibilidade | "different ways to form" (linha 1523) | Valor recomendado |

**Possíveis definições (não especificadas no paper):**

| Abordagem | Fórmula | Característica |
|-----------|---------|----------------|
| Raio fixo | r = constante | Simples, não adapta |
| Baseado em σ | r = c × σ | Adapta à dispersão |
| Baseado em τ | τ_local > threshold | Consistente com TEDA |
| k-vizinhos | dist. ao k-ésimo vizinho | Adapta à densidade local |

**Implicação:** O paper Angelov (2014) é um **framework conceitual**. A implementação prática de clustering requer decisões adicionais que o paper deixa em aberto.

**Provável solução:** MicroTEDAclus (Maia 2020) provavelmente preenche essa lacuna com critérios específicos de formação/merge/split de clusters.

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
| **Teorema de Huygens-Steiner** | Identidade que relaciona distâncias pairwise com distância à média: Σd_ij² = k·d_μ² + k·σ². Base da fórmula recursiva do TEDA |
| **Fórmula de König-Huygens** | Relação estatística Var(X) = E[X²] - E[X]², usada para expressar σ² = X - \|\|μ\|\|² |
| **Produto escalar X** | Média dos quadrados das normas: X = (1/k)·Σ\|\|x_i\|\|². Usado para calcular variância incrementalmente |

---

**Status:** ✅ Completo (~95%)
**Última atualização:** 2026-01-05
**Próximos passos:** Leitura do MicroTEDAclus (Maia 2020) — fichamento separado
