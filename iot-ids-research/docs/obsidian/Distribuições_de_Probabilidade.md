# Distribuições de Probabilidade

> **Tipo:** Teoria de Probabilidade  
> **Complexidade:** ⭐⭐⭐☆☆ (Intermediário-Avançado)  
> **Aplicação:** Modelagem de Processos Aleatórios

---

## 🎯 O que é uma Distribuição de Probabilidade?

Uma **distribuição de probabilidade** descreve como a **probabilidade está distribuída** sobre diferentes valores possíveis de uma variável aleatória.

**Pergunta que responde:**
> "Quais valores uma variável pode assumir e com que probabilidade?"

---

## 🧭 Por que Precisamos Delas?

### Sem Distribuições
```
Observei: 90 acertos em 100 tentativas
E agora? 🤷
```

### Com Distribuições
```
Observei: 90 acertos em 100 tentativas
Modelo: Binomial(100, θ)
Inferência: θ ~ Beta(91, 11)

Agora posso:
✅ Calcular P(θ > 0.85) = 0.973
✅ Encontrar IC 95%: [0.838, 0.946]
✅ Comparar com outro modelo
✅ Fazer predições
```

**Distribuições transformam dados em conhecimento probabilístico!**

---

## 📊 Tipos de Distribuições

### Discretas vs. Contínuas

#### **Distribuições Discretas**
Variável assume valores **contáveis** (inteiros).

**Função Massa de Probabilidade (PMF):**
```
P(X = k) = probabilidade de X ser exatamente k
```

**Propriedades:**
```
∑ P(X = k) = 1  (soma de todas as probabilidades = 1)
P(X = k) ∈ [0, 1]
```

#### **Distribuições Contínuas**
Variável assume valores em um **intervalo contínuo**.

**Função Densidade de Probabilidade (PDF):**
```
f(x) = densidade em x
P(a ≤ X ≤ b) = ∫ₐᵇ f(x) dx
```

**Propriedades:**
```
∫₋∞^∞ f(x) dx = 1  (área total = 1)
P(X = x exato) = 0  (probabilidade de ponto exato é zero!)
```

---

## 📐 Momentos de uma Distribuição

### Definição Universal

Para **qualquer** distribuição, os momentos são definidos por:

#### **Esperança (Média):**

**Discreto:**
```
E[X] = μ = ∑ x × P(X=x)
```

**Contínuo:**
```
E[X] = μ = ∫ x × f(x) dx
```

#### **Variância:**
```
Var(X) = σ² = E[(X-μ)²] = E[X²] - (E[X])²
```

#### **Desvio Padrão:**
```
σ = √Var(X)
```

### Por que Fórmulas "Diferentes"?

As fórmulas **derivadas** são diferentes porque cada distribuição tem **f(x) ou P(X=x) diferente**!

**Exemplo:**
- Normal(μ, σ²): f(x) = (1/√2πσ²)exp[-(x-μ)²/2σ²]
- [[Distribuição_Beta|Beta(α,β)]]: f(x) = x^(α-1)(1-x)^(β-1)/B(α,β)

Mesma definição (∫ x f(x) dx), mas integrandos diferentes → resultados diferentes!

Veja [[Média_Desvio_Padrão_Erro_Padrão#Relação com Distribuições|relação com estatística descritiva]].

---

## 📚 Catálogo de Distribuições Importantes

### 1. Binomial(n, p)

**Tipo:** Discreta

**Contexto:** n tentativas independentes, cada com probabilidade p de sucesso.

**PMF:**
```
P(X = k) = C(n,k) × p^k × (1-p)^(n-k)
```

**Momentos:**
```
Média: μ = np
Variância: σ² = np(1-p)
```

**Exemplo [[Aplicação_ao_IoT_IDS|IDS]]:**
```python
from scipy import stats

# 100 classificações, p=0.9 de acertar cada uma
dist = stats.binom(n=100, p=0.9)

# P(acertar exatamente 90)
print(dist.pmf(90))  # 0.132

# P(acertar 85 ou mais)
print(1 - dist.cdf(84))  # 0.942
```

### 2. Normal(μ, σ²)

**Tipo:** Contínua

**Contexto:** Distribuição mais comum na natureza. Teorema Central do Limite.

**PDF:**
```
f(x) = (1/√(2πσ²)) × exp[-(x-μ)²/(2σ²)]
```

**Momentos:**
```
Média: μ (parâmetro)
Variância: σ² (parâmetro)
```

**Propriedades:**
- Simétrica em torno de μ
- Regra 68-95-99.7 (veja [[Média_Desvio_Padrão_Erro_Padrão#Interpretação com Normal|regra empírica]])

**Exemplo:**
```python
from scipy import stats

# Acurácia com μ=0.9, σ=0.05
dist = stats.norm(loc=0.9, scale=0.05)

# P(acurácia > 0.85)
print(1 - dist.cdf(0.85))  # 0.841

# IC 95%
print(dist.interval(0.95))  # (0.802, 0.998)
```

**Problema:** Não respeita limites [0,1] para proporções! Veja [[The_Balanced_Accuracy_and_Its_Posterior_Distribution#Problema 1]].

### 3. [[Distribuição_Beta|Beta(α, β)]]

**Tipo:** Contínua

**Contexto:** Proporções, probabilidades, acurácias. Prior conjugado da Binomial.

**PDF:**
```
f(x) = x^(α-1) × (1-x)^(β-1) / B(α,β)   para x ∈ [0,1]
```

**Momentos:**
```
Média: μ = α/(α+β)
Variância: σ² = αβ/[(α+β)²(α+β+1)]
```

**Por que perfeita para [[Acurácia]]:**
- ✅ Suporte [0,1] - respeita limites naturais!
- ✅ Flexível - várias formas com α, β
- ✅ Conjugada - matemática elegante

Veja página dedicada: [[Distribuição_Beta]]

### 4. Poisson(λ)

**Tipo:** Discreta

**Contexto:** Contagem de eventos raros em intervalo fixo.

**PMF:**
```
P(X = k) = (λ^k × e^(-λ)) / k!
```

**Momentos:**
```
Média: λ
Variância: λ (igual à média!)
```

**Exemplo IDS:**
Número de ataques por hora em rede IoT.

```python
# Média de 2 ataques/hora
dist = stats.poisson(mu=2)

# P(0 ataques em uma hora)
print(dist.pmf(0))  # 0.135

# P(mais de 5 ataques)
print(1 - dist.cdf(5))  # 0.017 (raro!)
```

### 5. Uniforme(a, b)

**Tipo:** Contínua

**Contexto:** Total ignorância - todos os valores igualmente prováveis.

**PDF:**
```
f(x) = 1/(b-a)   para x ∈ [a,b]
```

**Momentos:**
```
Média: (a+b)/2
Variância: (b-a)²/12
```

**Uso em [[Inferência_Bayesiana]]:**
Prior não-informativo para proporções: Uniforme(0,1) = Beta(1,1)

### 6. Exponencial(λ)

**Tipo:** Contínua

**Contexto:** Tempo até o próximo evento (processos de Poisson).

**PDF:**
```
f(x) = λ × e^(-λx)   para x ≥ 0
```

**Momentos:**
```
Média: 1/λ
Variância: 1/λ²
```

**Exemplo IDS:**
Tempo até o próximo ataque.

```python
# Taxa: 0.5 ataques/hora → média 2 horas entre ataques
dist = stats.expon(scale=2)

# P(próximo ataque em menos de 1 hora)
print(dist.cdf(1))  # 0.393
```

---

## 🔄 Relações Entre Distribuições

### Famílias Conjugadas

**Binomial ↔ Beta**
```
Likelihood: Binomial(n, θ)
Prior: Beta(α, β)
→ Posterior: Beta(α + k, β + n - k)
```

Usado no [[The_Balanced_Accuracy_and_Its_Posterior_Distribution|artigo]]!

**Poisson ↔ Gamma**
```
Likelihood: Poisson(λ)
Prior: Gamma(α, β)
→ Posterior: Gamma(α + ∑k, β + n)
```

Veja [[Inferência_Bayesiana#Conjugate Priors|priors conjugados]].

### Aproximações

**Binomial → Normal** (para n grande):
```
Binomial(n, p) ≈ Normal(np, np(1-p))
```

**Poisson → Normal** (para λ grande):
```
Poisson(λ) ≈ Normal(λ, λ)
```

**Beta → Normal** (para α, β grandes):
```
Beta(α, β) ≈ Normal(α/(α+β), ...)
```

---

## 📊 Tabela de Referência Rápida

| Distribuição | Suporte | Parâmetros | Média | Variância | Uso Típico |
|--------------|---------|------------|-------|-----------|------------|
| Binomial | {0,1,...,n} | n, p | np | np(1-p) | Contagem sucessos |
| Poisson | {0,1,2,...} | λ | λ | λ | Eventos raros |
| Normal | (-∞, ∞) | μ, σ² | μ | σ² | Fenômenos naturais |
| **Beta** | [0, 1] | α, β | α/(α+β) | ... | **Proporções/Acurácias** |
| Uniforme | [a, b] | a, b | (a+b)/2 | (b-a)²/12 | Prior ignorância |
| Exponencial | [0, ∞) | λ | 1/λ | 1/λ² | Tempo entre eventos |

---

## 🧮 Escolhendo a Distribuição Certa

### Perguntas a Fazer

1. **Tipo de variável?**
   - Discreta → Binomial, Poisson, etc.
   - Contínua → Normal, Beta, etc.

2. **Suporte (valores possíveis)?**
   - [0,1] → Beta ✅
   - [0,∞) → Exponencial, Gamma
   - (-∞,∞) → Normal

3. **Processo gerador?**
   - n tentativas independentes → Binomial
   - Contagem em intervalo → Poisson
   - Proporção/probabilidade → Beta

4. **Propriedades desejadas?**
   - Simétrico → Normal
   - Assimétrico → Beta (com α≠β), Gamma

### Para [[Acurácia]]/[[Acurácia_Balanceada]]

**Escolha: Beta** ✅

**Razão:**
1. Suporte [0,1] - respeita limites naturalmente
2. Flexível (múltiplas formas)
3. Conjugada à Binomial (classificação é Bernoulli)
4. Interpretação clara dos parâmetros

**NÃO use Normal** para proporções (pode violar [0,1])!

---

## 🎲 Simulação e Visualização

### Comparando Distribuições

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Configurar distribuições
binomial = stats.binom(n=100, p=0.9)
beta = stats.beta(a=91, b=11)
normal = stats.norm(loc=0.9, scale=0.03)

# Visualizar
x_discrete = np.arange(70, 101)
x_continuous = np.linspace(0.7, 1.0, 1000)

plt.figure(figsize=(15, 5))

# Binomial
plt.subplot(131)
plt.stem(x_discrete, binomial.pmf(x_discrete))
plt.title('Binomial(100, 0.9)')
plt.xlabel('Número de acertos')

# Beta
plt.subplot(132)
plt.plot(x_continuous, beta.pdf(x_continuous))
plt.title('Beta(91, 11)')
plt.xlabel('Acurácia')

# Normal (problema!)
plt.subplot(133)
plt.plot(x_continuous, normal.pdf(x_continuous))
plt.axvline(1.0, color='r', linestyle='--', label='Limite [0,1]')
plt.title('Normal(0.9, 0.03²) - Problema!')
plt.xlabel('Acurácia')
plt.legend()

plt.tight_layout()
plt.show()
```

### Amostragem

```python
# Gerar amostras
n_samples = 10000

binomial_samples = binomial.rvs(n_samples) / 100  # converter para [0,1]
beta_samples = beta.rvs(n_samples)
normal_samples = normal.rvs(n_samples)

# Verificar quantos violam [0,1]
print(f"Beta viola [0,1]: {np.sum((beta_samples < 0) | (beta_samples > 1))}")
# 0 ✅

print(f"Normal viola [0,1]: {np.sum((normal_samples < 0) | (normal_samples > 1))}")
# > 0 ❌
```

---

## 📚 Referências

### Livros Fundamentais
- **Casella, G. & Berger, R.L.** (2002). *Statistical Inference* (2nd ed.). Duxbury. [Capítulo 3: "Common Families of Distributions"]
- **DeGroot, M.H. & Schervish, M.J.** (2012). *Probability and Statistics* (4th ed.). Pearson. [Capítulos 3-5]
- **Ross, S.M.** (2014). *Introduction to Probability Models* (11th ed.). Academic Press.

### Para Intuição
- **McElreath, R.** (2020). *Statistical Rethinking* (2nd ed.). CRC Press. [Capítulo 2]
- **Kruschke, J.K.** (2014). *Doing Bayesian Data Analysis* (2nd ed.). Academic Press. [Capítulo 4]

### Online
- [Seeing Theory](https://seeing-theory.brown.edu/) - Visualizações interativas ⭐
- [Distribution Explorer](https://distribution-explorer.github.io/) - Explore distribuições
- [3Blue1Brown: Distributions](https://www.youtube.com/watch?v=zeJD6dqJ5lo)

Veja [[Referências_Bibliográficas]] para lista completa.

---

## 🔗 Conceitos Relacionados

### Fundamentos
- [[Média_Desvio_Padrão_Erro_Padrão]] - Momentos em forma empírica
- [[Intervalos_de_Confiança]] - Aplicação das distribuições

### Específicas
- [[Distribuição_Beta]] - Distribuição central do artigo
- [[Métodos_Paramétricos_vs_Não_Paramétricos]] - Assumir ou não distribuição

### Paradigma
- [[Inferência_Bayesiana]] - Uso de distribuições como priors/posteriors

### Aplicação
- [[Acurácia]] - Modelada como Beta
- [[The_Balanced_Accuracy_and_Its_Posterior_Distribution]] - Artigo usa Beta

---

## 🎯 Exercícios

Veja [[Exercícios_Práticos#Distribuições de Probabilidade]].

---

## 📌 Resumo Visual

```
┌────────────────────────────────────────────┐
│     DISTRIBUIÇÕES DE PROBABILIDADE         │
│                                            │
│  DISCRETAS          CONTÍNUAS             │
│  ────────────       ─────────────         │
│  • Binomial         • Normal              │
│  • Poisson          • Beta ⭐             │
│  • Geométrica       • Exponencial         │
│                     • Gamma               │
│                                            │
│  Para Proporções/Acurácia:                │
│                                            │
│      ┌────────┐                           │
│      │  BETA  │  ← Escolha correta!       │
│      └────────┘                           │
│      • Suporte [0,1]                      │
│      • Flexível                            │
│      • Conjugada                           │
│                                            │
└────────────────────────────────────────────┘
```

---

**Tags:** #probability #distributions #beta #normal #binomial #statistics #theory

**Voltar para:** [[INDEX]]  
**Aprofundar:** [[Distribuição_Beta]]  
**Aplicação:** [[Inferência_Bayesiana]]


