# Intervalos de Confiança

> **Tipo:** Inferência Estatística  
> **Complexidade:** ⭐⭐⭐☆☆ (Intermediário)  
> **Aplicação:** Quantificação de Incerteza

---

## 🎯 O que é um Intervalo de Confiança?

Um **intervalo de confiança** (IC) expressa a **incerteza** sobre um parâmetro desconhecido com base em dados observados.

**Pergunta que responde:**
> "Dado que observei estes dados, em que faixa de valores o verdadeiro parâmetro provavelmente está?"

---

## 🧭 Duas Filosofias, Duas Interpretações

### 🔵 Abordagem Frequentista (Clássica)

**Interpretação:**
> "Se repetirmos o experimento infinitas vezes, 95% dos intervalos construídos conterão o verdadeiro parâmetro θ."

**Características:**
- θ é **fixo mas desconhecido**
- O intervalo é **aleatório** (varia entre experimentos)
- Não se pode dizer "95% de probabilidade de θ estar no intervalo"

**Analogia:**
Imagine jogar uma rede de pesca (o intervalo) em um ponto fixo no oceano (θ). Se você jogar 100 redes de tamanhos aleatórios, 95 delas capturarão aquele ponto.

### 🟢 Abordagem Bayesiana (Do [[The_Balanced_Accuracy_and_Its_Posterior_Distribution|Artigo]])

**Interpretação:**
> "Há 95% de probabilidade de θ estar neste intervalo, dada a evidência observada."

**Nome alternativo:** **Intervalo de Credibilidade**

**Características:**
- θ é **variável aleatória** (tem distribuição de probabilidade)
- O intervalo contém 95% da massa da distribuição posterior
- Interpretação probabilística direta!

**Analogia:**
Você tem um "mapa de probabilidade" mostrando onde θ provavelmente está. O intervalo marca a região que contém 95% da probabilidade.

Veja [[Inferência_Bayesiana]] para detalhes do paradigma.

---

## 📊 Construção: Abordagem Frequentista

### Passo a Passo

**1. Coletar dados:** x₁, x₂, ..., xₙ

**2. Assumir distribuição:** Ex: dados são Normal(μ, σ²)

**3. Calcular estatística:** Ex: x̄ (média amostral)

**4. Conhecer distribuição amostral:** Ex: x̄ ~ Normal(μ, σ²/n)

**5. Construir intervalo:**
```
IC_{1-α} = [x̄ - z_{α/2} × SE, x̄ + z_{α/2} × SE]
```

### Exemplo: [[Acurácia]] de IDS

```
Dados: 100 testes, 90 acertos
Estatística: p̂ = 90/100 = 0.90

Distribuição amostral (aproximação Normal):
p̂ ~ Normal(p, p(1-p)/n)

Erro padrão:
SE = √[p̂(1-p̂)/n] = √[0.90×0.10/100] = 0.03

IC 95% (z = 1.96):
[0.90 - 1.96×0.03, 0.90 + 1.96×0.03]
= [0.841, 0.959]
```

### Problema: [[Métodos_Paramétricos_vs_Não_Paramétricos|Abordagem Não-Paramétrica]]

**Se p̂ = 0.98 e SE = 0.02:**
```
IC = [0.98 - 1.96×0.02, 0.98 + 1.96×0.02]
   = [0.941, 1.019]
```

**1.019?! 101.9%?!** 🚨 Acurácia não pode ser > 100%!

**Problema fundamental:** A aproximação Normal **não respeita** os limites naturais [0, 1].

---

## 📈 Construção: Abordagem Bayesiana

### Passo a Passo

**1. Especificar prior:** P(θ) - crença inicial

**2. Coletar dados:** X

**3. Calcular posterior:** P(θ|X) usando Teorema de Bayes

**4. Extrair intervalo:** Região que contém 95% da massa

### Exemplo: [[Distribuição_Beta|Acurácia com Beta]]

```
Prior: θ ~ Beta(1, 1)  # Uniforme
Dados: 90 acertos, 10 erros
Posterior: θ ~ Beta(91, 11)

IC 95% (intervalo de credibilidade):
[L, U] tal que ∫ₗᵘ Beta(x; 91, 11) dx = 0.95

Usando scipy:
>>> from scipy import stats
>>> stats.beta.interval(0.95, 91, 11)
(0.838, 0.946)
```

**Vantagens:**
- ✅ Respeita limites [0, 1] naturalmente
- ✅ Assimétrico quando apropriado
- ✅ Interpretação probabilística direta
- ✅ Funciona para qualquer n (não precisa ser grande)

---

## 🎚️ Tipos de Intervalos Bayesianos

### 1. Equal-Tailed Interval (Massa Central)

**Definição:** Deixa α/2 em cada cauda.

```
P(θ < L) = α/2 = 0.025
P(θ > U) = α/2 = 0.025
```

**Cálculo:**
```python
L = distribuicao.ppf(0.025)  # Quantil 2.5%
U = distribuicao.ppf(0.975)  # Quantil 97.5%
```

### 2. Highest Density Interval (HDI)

**Definição:** Menor intervalo que contém 95% da massa.

**Propriedade:** Todo ponto dentro tem densidade ≥ qualquer ponto fora.

**Vantagem:** Intervalo mais curto possível!

```python
# Requer bibliotecas especiais (PyMC, arviz)
import arviz as az
hdi = az.hdi(samples, hdi_prob=0.95)
```

### Comparação Visual

```
        Posterior Beta(91, 11)
           ╱╲
          ╱  ╲
         ╱    ╲
        ╱      ╲___
       ╱            ╲
  ────┼──────────────┼──── θ
      L_et        U_et    (Equal-tailed)
      
      L_hdi    U_hdi      (HDI - mais curto!)
```

Para distribuições simétricas: Equal-tailed = HDI  
Para assimétricas: HDI é mais informativo

---

## 🔄 Comparação: Frequentista vs. Bayesiano

### Exemplo Lado a Lado

**Dados:** 98 acertos em 100 testes de [[Aplicação_ao_IoT_IDS|IDS]]

#### Frequentista (Normal Approximation)
```
p̂ = 0.98
SE = √[0.98×0.02/100] = 0.014

IC 95% = [0.98 - 1.96×0.014, 0.98 + 1.96×0.014]
       = [0.953, 1.007]  ← 100.7%! ❌
```

#### Frequentista (Wilson Score - corrigido)
```
IC 95% = [0.930, 0.995]  ✅
(mais complexo, mas respeita limites)
```

#### Bayesiano (Beta)
```
Posterior: Beta(99, 3)

IC 95% = [0.932, 0.997]  ✅
(natural, sem artifícios)
```

### Tabela Comparativa

| Aspecto | Frequentista | Bayesiano |
|---------|-------------|-----------|
| θ é... | Fixo desconhecido | Variável aleatória |
| Interpreta 95% como... | % de intervalos que capturam θ | Probabilidade de θ estar lá |
| Requer n grande? | Sim (para CLT) | Não |
| Respeita limites? | Depende do método | Sim (se prior adequado) |
| Incorpora conhecimento prévio? | Não | Sim |
| Computação | Geralmente analítica | Às vezes numérica |

---

## 📐 Relação com [[Distribuições_de_Probabilidade]]

### Por que Precisamos da Distribuição?

**Sem distribuição:**
```
Dados: 90/100
IC: ???  ¯\_(ツ)_/¯
```

**Com distribuição (Beta):**
```
Dados: 90/100
Modelo: θ ~ Beta(91, 11)
IC: [0.838, 0.946]  ✅

Além disso:
P(θ > 0.85) = 0.973
P(θ > 0.90) = 0.503
etc.
```

A distribuição é a **ponte** entre dados e inferência!

### Diferentes Distribuições, Diferentes ICs

**Mesmo dado (90/100), diferentes modelos:**

```python
from scipy import stats

# Beta (para proporções)
beta_ic = stats.beta.interval(0.95, 91, 11)
# [0.838, 0.946]

# Normal (aproximação)
from statsmodels.stats.proportion import proportion_confint
normal_ic = proportion_confint(90, 100, method='normal')
# [0.841, 0.959]

# Wilson Score (melhor aproximação)
wilson_ic = proportion_confint(90, 100, method='wilson')
# [0.831, 0.946]
```

**A escolha da distribuição importa!**

Veja [[Métodos_Paramétricos_vs_Não_Paramétricos]] para discussão sobre escolhas de modelo.

---

## 🚀 Aplicação ao [[The_Balanced_Accuracy_and_Its_Posterior_Distribution|Artigo Principal]]

### Problema 1 do Artigo: ICs Inadequados

**Método tradicional ([[Média_Desvio_Padrão_Erro_Padrão|com erro padrão]]):**
```
Acurácia = 0.98 ± 2×SE
```

**Defeitos:**
- ❌ Simétrico sempre
- ❌ Pode violar [0, 1]
- ❌ Assume normalidade

### Solução do Artigo: Posterior Beta

**Para [[Acurácia]]:**
```
A ~ Beta(C+1, I+1)
IC = quantis da posterior
```

**Para [[Acurácia_Balanceada]]:**
```
BA distribuição via convolução de Betas
IC = quantis da posterior de BA
```

### Exemplo Comparativo do Artigo

**Dataset desbalanceado:** 45 positivos, 10 negativos  
**Classificador enviesado:** 48 pred. positivos, 7 pred. negativos

```
Acurácia Tradicional:
IC 95%: [87%, 97%]  ← Sugere bom desempenho

Acurácia Balanceada (Posterior):
IC 95%: [48%, 54%]  ← Revela nível do acaso!
```

O intervalo da BA **expõe** o problema que acurácia mascara!

---

## 🧮 Implementação Prática

### 1. Frequentista: Proporção

```python
from statsmodels.stats.proportion import proportion_confint

# Dados
sucessos = 90
tentativas = 100

# Método Wilson (recomendado)
ci = proportion_confint(sucessos, tentativas, 
                        alpha=0.05, method='wilson')
print(f"IC 95%: [{ci[0]:.3f}, {ci[1]:.3f}]")
```

### 2. Bayesiano: Beta

```python
from scipy import stats

# Dados
C, I = 90, 10  # corretos, incorretos

# Posterior (prior uniforme)
posterior = stats.beta(C + 1, I + 1)

# Intervalo de credibilidade
ci = posterior.interval(0.95)
print(f"IC 95%: [{ci[0]:.3f}, {ci[1]:.3f}]")

# Estatísticas adicionais
print(f"Média: {posterior.mean():.3f}")
print(f"Mediana: {posterior.median():.3f}")
print(f"Moda: {C/(C+I):.3f}")
```

### 3. Bayesiano: HDI

```python
import numpy as np
from scipy import stats

def hdi(distribution, credibility=0.95):
    """
    Calcula Highest Density Interval.
    
    Algoritmo: busca o menor intervalo que contém
    a massa de probabilidade desejada.
    """
    # Amostragem da posterior
    samples = distribution.rvs(100000)
    samples = np.sort(samples)
    
    # Tamanho do intervalo
    n = len(samples)
    interval_size = int(np.floor(credibility * n))
    
    # Buscar menor intervalo
    n_intervals = n - interval_size
    interval_widths = samples[interval_size:] - samples[:n_intervals]
    
    min_idx = np.argmin(interval_widths)
    hdi_min = samples[min_idx]
    hdi_max = samples[min_idx + interval_size]
    
    return (hdi_min, hdi_max)

# Uso
posterior = stats.beta(91, 11)
interval = hdi(posterior, 0.95)
print(f"HDI 95%: [{interval[0]:.3f}, {interval[1]:.3f}]")
```

### 4. Comparando Múltiplos Modelos

```python
from scipy import stats
import numpy as np

# Dois algoritmos
model_A = stats.beta(85 + 1, 15 + 1)  # 85/100
model_B = stats.beta(90 + 1, 10 + 1)  # 90/100

# Probabilidade de B > A
samples_A = model_A.rvs(100000)
samples_B = model_B.rvs(100000)
prob_B_better = np.mean(samples_B > samples_A)

print(f"P(B > A) = {prob_B_better:.3f}")
# Se > 0.95, B é significativamente melhor!
```

---

## 📚 Referências

### Livros Fundamentais
- **Wasserman, L.** (2004). *All of Statistics*. Springer. [Capítulo 11: "Statistical Inference"]
- **Casella, G. & Berger, R.L.** (2002). *Statistical Inference* (2nd ed.). [Seção 9.2: "Interval Estimators"]
- **Gelman, A., et al.** (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press. [Capítulo 4]

### Papers
- **Brodersen et al.** (2010). [[The_Balanced_Accuracy_and_Its_Posterior_Distribution]]
- **Agresti, A. & Coull, B.A.** (1998). "Approximate is better than 'exact' for interval estimation of binomial proportions". *The American Statistician*, 52(2), 119-126.
- **Brown, L.D., Cai, T.T., & DasGupta, A.** (2001). "Interval estimation for a binomial proportion". *Statistical Science*, 16(2), 101-133.

### Online
- [Seeing Theory: Confidence Intervals](https://seeing-theory.brown.edu/frequentist-inference/index.html)
- [3Blue1Brown: Bayesian Inference](https://www.youtube.com/watch?v=HZGCoVF3YvM)

Veja [[Referências_Bibliográficas]] para lista completa.

---

## 🔗 Conceitos Relacionados

### Fundamentos
- [[Média_Desvio_Padrão_Erro_Padrão]] - Base para construção
- [[Distribuições_de_Probabilidade]] - Framework teórico
- [[Distribuição_Beta]] - Para proporções/acurácias

### Paradigmas
- [[Métodos_Paramétricos_vs_Não_Paramétricos]] - Escolhas de modelagem
- [[Inferência_Bayesiana]] - Paradigma probabilístico

### Aplicações
- [[Acurácia]] - Métrica a ser quantificada
- [[Acurácia_Balanceada]] - Com incerteza
- [[The_Balanced_Accuracy_and_Its_Posterior_Distribution]] - Artigo completo

---

## 🎯 Exercícios

Veja [[Exercícios_Práticos#Intervalos de Confiança]].

---

## 📌 Resumo Visual

```
┌─────────────────────────────────────────────┐
│         INTERVALO DE CONFIANÇA              │
│                                             │
│  "Quantificação de incerteza"               │
│                                             │
│  ┌─────────────────┐  ┌─────────────────┐ │
│  │  FREQUENTISTA   │  │   BAYESIANO     │ │
│  │                 │  │                 │ │
│  │ θ fixo          │  │ θ variável      │ │
│  │ IC varia        │  │ IC fixo p/ dados│ │
│  │ "95% capturam"  │  │ "95% prob."     │ │
│  │                 │  │                 │ │
│  │ Pode violar     │  │ Respeita        │ │
│  │ limites [0,1]   │  │ limites         │ │
│  └─────────────────┘  └─────────────────┘ │
│                                             │
│  Artigo usa BAYESIANO com Beta!            │
│                                             │
└─────────────────────────────────────────────┘
```

---

**Tags:** #statistics #confidence-interval #credible-interval #inference #bayesian #frequentist

**Voltar para:** [[INDEX]]  
**Artigo relacionado:** [[The_Balanced_Accuracy_and_Its_Posterior_Distribution]]  
**Paradigma:** [[Inferência_Bayesiana]]


