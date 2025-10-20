# Média, Desvio Padrão e Erro Padrão

> **Tipo:** Estatística Descritiva  
> **Complexidade:** ⭐⭐☆☆☆ (Básico-Intermediário)  
> **Aplicação:** Análise de Dados, Agregação de Resultados

---

## 🎯 Visão Geral

Estas três medidas formam a base da análise estatística:
- **Média:** Onde os dados estão centrados?
- **Desvio Padrão:** Quão dispersos estão os dados?
- **Erro Padrão:** Quão incerta é nossa estimativa da média?

---

## 📊 1. Média (Mean)

### Definição

A **média** é a medida de tendência central mais comum.

**Pergunta que responde:**
> "Qual o valor 'típico' dos dados?"

### Fórmula

**Média amostral:**
```
x̄ = (Σᵢ xᵢ) / n = (x₁ + x₂ + ... + xₙ) / n
```

**Média populacional:**
```
μ = (Σᵢ xᵢ) / N
```

**Notação:**
- `x̄` (x-barra): média amostral
- `μ` (mu): média populacional

### Exemplo [[Aplicação_ao_IoT_IDS|IDS]]: Acurácia em 5 Folds

```
Fold 1: 92%
Fold 2: 88%
Fold 3: 91%
Fold 4: 89%
Fold 5: 90%

x̄ = (92 + 88 + 91 + 89 + 90) / 5
  = 450 / 5
  = 90%
```

**Interpretação:** Em média, o modelo tem 90% de [[Acurácia]].

### Propriedades

1. **Sensível a outliers:**
```
Dados: [10, 12, 11, 13, 100]
Média: 29.2 (não representa a maioria!)
```

2. **Minimiza soma dos quadrados:**
```
x̄ minimiza Σ(xᵢ - c)²
```

3. **Linearidade:**
```
E[aX + b] = aE[X] + b
```

### Implementação

```python
import numpy as np

acuracias = [92, 88, 91, 89, 90]

# Método 1: Manual
media = sum(acuracias) / len(acuracias)

# Método 2: NumPy
media = np.mean(acuracias)

print(f"Média: {media}%")
```

---

## 📏 2. Desvio Padrão (Standard Deviation)

### Definição

O **desvio padrão** mede a **dispersão** dos dados em relação à média.

**Pergunta que responde:**
> "Quão longe os dados estão da média, em média?"

### Fórmula

**Desvio padrão amostral (s):**
```
s = √[Σ(xᵢ - x̄)² / (n-1)]
```

**Desvio padrão populacional (σ):**
```
σ = √[Σ(xᵢ - μ)² / N]
```

**Por que (n-1)?** Correção de Bessel para viés - veja [[Inferência_Bayesiana#Estimador Não-viesado|estimadores não-viesados]].

### Variância

```
Variância = s² = Σ(xᵢ - x̄)² / (n-1)
Desvio Padrão = √Variância
```

### Exemplo IDS: Passo a Passo

```
Dados: [92, 88, 91, 89, 90]
Média: 90

Passo 1: Calcular diferenças
92 - 90 = +2
88 - 90 = -2
91 - 90 = +1
89 - 90 = -1
90 - 90 =  0

Passo 2: Elevar ao quadrado
(+2)² = 4
(-2)² = 4
(+1)² = 1
(-1)² = 1
( 0)² = 0

Passo 3: Somar
Σ(xᵢ - x̄)² = 4 + 4 + 1 + 1 + 0 = 10

Passo 4: Dividir por (n-1)
Variância = 10 / (5-1) = 10/4 = 2.5

Passo 5: Raiz quadrada
s = √2.5 ≈ 1.58%
```

**Interpretação:** Os resultados variam tipicamente ±1.58% em torno da média de 90%.

### Interpretação com Distribuição Normal

Se os dados seguem [[Distribuições_de_Probabilidade#Normal|distribuição Normal]]:

```
68% dos dados estão em [μ - σ, μ + σ]
95% dos dados estão em [μ - 2σ, μ + 2σ]
99.7% dos dados estão em [μ - 3σ, μ + 3σ]
```

**Regra 68-95-99.7** (Regra Empírica)

### Implementação

```python
import numpy as np

acuracias = [92, 88, 91, 89, 90]

# Método 1: Manual
media = np.mean(acuracias)
diferencas_quadradas = [(x - media)**2 for x in acuracias]
variancia = sum(diferencas_quadradas) / (len(acuracias) - 1)
desvio = np.sqrt(variancia)

# Método 2: NumPy (usa n-1 por padrão)
desvio = np.std(acuracias, ddof=1)

print(f"Desvio Padrão: {desvio:.2f}%")
```

---

## 📐 3. Erro Padrão (Standard Error)

### Definição

O **erro padrão** mede a **incerteza** sobre a estimativa da **média**.

**Pergunta que responde:**
> "Se eu repetir o experimento, quanto a média vai variar?"

### Diferença Crucial

```
┌──────────────────────────────────────────────┐
│  DESVIO PADRÃO:                              │
│  • Dispersão dos DADOS individuais           │
│  • Propriedade da AMOSTRA                    │
│  • Quanto os valores variam entre si         │
└──────────────────────────────────────────────┘

┌──────────────────────────────────────────────┐
│  ERRO PADRÃO:                                │
│  • Incerteza sobre a MÉDIA                   │
│  • Propriedade do ESTIMADOR                  │
│  • Quanto a média varia entre experimentos   │
└──────────────────────────────────────────────┘
```

### Fórmula

```
SE = s / √n

ou

SE(x̄) = σ / √n
```

Onde:
- **s:** desvio padrão amostral
- **n:** tamanho da amostra

### Propriedade Chave

**Erro padrão diminui com √n:**

```
n = 10  → SE = s/√10 ≈ 0.316s
n = 100 → SE = s/√100 = 0.1s
n = 1000 → SE = s/√1000 ≈ 0.032s
```

Quanto maior a amostra, mais precisa a estimativa da média!

### Exemplo IDS

```
Dados: [92, 88, 91, 89, 90]
Média: 90%
Desvio Padrão: 1.58%
n: 5

SE = s / √n
   = 1.58 / √5
   = 1.58 / 2.236
   ≈ 0.71%
```

**Interpretação:** A verdadeira média da [[Acurácia]] do modelo está provavelmente dentro de ±0.71% de 90%.

### [[Intervalos_de_Confiança|Intervalo de Confiança]] (Aproximação)

```
IC 95% ≈ x̄ ± 1.96 × SE
       ≈ 90 ± 1.96 × 0.71
       ≈ 90 ± 1.39
       ≈ [88.61%, 91.39%]
```

**Problema:** Essa aproximação assume [[Distribuições_de_Probabilidade#Normal|normalidade]] e pode violar limites [0, 100%]. Veja [[The_Balanced_Accuracy_and_Its_Posterior_Distribution#Problema 1|problema no artigo]].

### Implementação

```python
import numpy as np
from scipy import stats

acuracias = [92, 88, 91, 89, 90]
n = len(acuracias)

# Média e desvio
media = np.mean(acuracias)
desvio = np.std(acuracias, ddof=1)

# Erro padrão
erro_padrao = desvio / np.sqrt(n)

# Intervalo de confiança 95% (distribuição t)
ci = stats.t.interval(
    confidence=0.95,
    df=n-1,
    loc=media,
    scale=erro_padrao
)

print(f"Média: {media:.2f}%")
print(f"Desvio Padrão: {desvio:.2f}%")
print(f"Erro Padrão: {erro_padrao:.2f}%")
print(f"IC 95%: [{ci[0]:.2f}%, {ci[1]:.2f}%]")
```

---

## 🔄 Relações Entre as Três

### Tabela Comparativa

| Conceito | Mede | Pergunta | Notação | Depende de n? |
|----------|------|----------|---------|---------------|
| **Média** | Tendência central | Valor típico? | x̄, μ | Não |
| **Desvio Padrão** | Dispersão dos dados | Quão espalhados? | s, σ | Não* |
| **Erro Padrão** | Incerteza da média | Quão preciso? | SE | **Sim! (∝ 1/√n)** |

*Tecnicamente, o estimador de s melhora com n maior, mas s não diminui sistematicamente.

### Visualização

```
         Dados Individuais
         ↓
    ┌────────────────────┐
    │ x₁, x₂, ..., xₙ   │
    └────────┬───────────┘
             │
    ┌────────▼───────────┐
    │    MÉDIA (x̄)      │ ← Valor central
    └────────┬───────────┘
             │
    ┌────────▼───────────┐
    │ DESVIO PADRÃO (s) │ ← Dispersão dos dados
    └────────┬───────────┘
             │
    ┌────────▼───────────┐
    │ ERRO PADRÃO (SE)  │ ← Incerteza sobre x̄
    └────────┬───────────┘
             │
    ┌────────▼───────────┐
    │ INTERVALO (IC)    │ ← [x̄ - t×SE, x̄ + t×SE]
    └────────────────────┘
```

---

## 🎯 Aplicação em Cross-Validation

### Cenário Típico

Você treina um modelo IDS com 10-fold cross-validation:

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Treinar modelo
clf = RandomForestClassifier()
scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')

# Resultados dos 10 folds
# scores = [0.91, 0.89, 0.92, 0.88, 0.90, 
#           0.91, 0.89, 0.93, 0.88, 0.91]

# Estatísticas
media = np.mean(scores)
desvio = np.std(scores, ddof=1)
erro_padrao = desvio / np.sqrt(len(scores))

print(f"Acurácia: {media:.3f} ± {erro_padrao:.3f}")
# Saída: Acurácia: 0.902 ± 0.005
```

### Problema com Esta Abordagem

**[[The_Balanced_Accuracy_and_Its_Posterior_Distribution|Artigo Brodersen et al.]]** mostra que isso é **problemático**:

1. ❌ Pode gerar IC > 100% ou < 0%
2. ❌ Não modela corretamente a natureza probabilística
3. ❌ Ignora que [[Acurácia]] é uma proporção limitada

**Solução:** Usar [[Distribuição_Beta]] e [[Inferência_Bayesiana]]!

---

## 🧮 Relação com [[Distribuições_de_Probabilidade]]

### Média como Esperança

A média é um caso especial da **esperança matemática**:

```
E[X] = Σ x × P(X=x)     (discreto)
E[X] = ∫ x × f(x) dx    (contínuo)
```

Para dados empíricos com P(X=xᵢ) = 1/n:
```
E[X] = Σ xᵢ × (1/n) = x̄
```

### Desvio Padrão como Raiz da Variância

```
Var(X) = E[(X - μ)²] = E[X²] - (E[X])²
σ = √Var(X)
```

### Exemplo: [[Distribuição_Beta]]

Para Beta(α, β):

```
Média: μ = α / (α + β)

Variância: σ² = αβ / [(α+β)²(α+β+1)]

Desvio Padrão: σ = √σ²
```

**Mesmo conceito, fórmula derivada diferente!** Veja explicação em [[Distribuições_de_Probabilidade#Momentos]].

---

## 📚 Referências

### Livros Fundamentais
- **Wasserman, L.** (2004). *All of Statistics*. Springer. [Capítulos 2-3]
- **Rice, J.A.** (2006). *Mathematical Statistics and Data Analysis* (3rd ed.). Duxbury Press. [Capítulo 7]
- **Casella, G. & Berger, R.L.** (2002). *Statistical Inference* (2nd ed.). Duxbury. [Capítulo 5]

### Online
- [Khan Academy: Média e Desvio Padrão](https://pt.khanacademy.org/math/statistics-probability)
- [Seeing Theory: Visualização Interativa](https://seeing-theory.brown.edu/)

Veja [[Referências_Bibliográficas]] para lista completa.

---

## 🔗 Conceitos Relacionados

### Aplicações
- [[Acurácia]] - Agregação de resultados de cross-validation
- [[Intervalos_de_Confiança]] - Usando erro padrão
- [[The_Balanced_Accuracy_and_Its_Posterior_Distribution]] - Crítica à abordagem tradicional

### Teoria
- [[Distribuições_de_Probabilidade]] - Definições universais
- [[Distribuição_Beta]] - Exemplo de média e variância específicas
- [[Métodos_Paramétricos_vs_Não_Paramétricos]] - Estimação de parâmetros

---

## 🎯 Exercícios

Veja [[Exercícios_Práticos#Estatística Descritiva]].

---

## 📌 Resumo Visual

```
┌───────────────────────────────────────────────┐
│                                               │
│  MÉDIA (x̄)                                   │
│  "Centro dos dados"                           │
│  x̄ = Σxᵢ / n                                 │
│                                               │
├───────────────────────────────────────────────┤
│                                               │
│  DESVIO PADRÃO (s)                           │
│  "Dispersão dos dados"                        │
│  s = √[Σ(xᵢ-x̄)² / (n-1)]                    │
│                                               │
├───────────────────────────────────────────────┤
│                                               │
│  ERRO PADRÃO (SE)                            │
│  "Incerteza sobre x̄"                         │
│  SE = s / √n                                  │
│                                               │
└───────────────────────────────────────────────┘

  ↓ Quanto maior n, menor SE! ↓
```

---

**Tags:** #statistics #descriptive-statistics #mean #standard-deviation #standard-error #fundamentals

**Voltar para:** [[INDEX]]  
**Próximo:** [[Intervalos_de_Confiança]]

