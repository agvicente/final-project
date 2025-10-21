# Inferência Bayesiana

> **Tipo:** Paradigma Estatístico  
> **Complexidade:** ⭐⭐⭐⭐☆ (Avançado)  
> **Aplicação:** Framework do [[The_Balanced_Accuracy_and_Its_Posterior_Distribution|Artigo Principal]]

---

## 🎯 O que é Inferência Bayesiana?

**Inferência Bayesiana** é um paradigma de raciocínio probabilístico onde:
- Parâmetros são tratados como **variáveis aleatórias** (têm distribuições)
- Usamos **Teorema de Bayes** para atualizar crenças com base em evidências
- Resultados são **distribuições de probabilidade**, não pontos

**Contraste com Frequentismo:**
```
FREQUENTISTA:
θ é fixo (mas desconhecido)
Dados são aleatórios
"Probabilidade" = frequência de longo prazo

BAYESIANO:
θ é variável aleatória (tem distribuição)
Dados são fixos (observados)
"Probabilidade" = grau de crença
```

---

## 📐 Teorema de Bayes

### Forma Geral

```
P(θ|dados) = [P(dados|θ) × P(θ)] / P(dados)

Posterior = [Likelihood × Prior] / Evidence
```

**Componentes:**

1. **P(θ):** **Prior** - Crença inicial sobre θ (antes de ver dados)
2. **P(dados|θ):** **Likelihood** - Probabilidade dos dados dado θ
3. **P(dados):** **Evidence/Marginal** - Probabilidade total dos dados
4. **P(θ|dados):** **Posterior** - Crença atualizada sobre θ (após ver dados)

### Evidence (Normalização)

```
P(dados) = ∫ P(dados|θ) × P(θ) dθ
```

Frequentemente difícil de calcular, mas pode ser ignorado:

```
P(θ|dados) ∝ P(dados|θ) × P(θ)

Posterior ∝ Likelihood × Prior
```

---

## 🔄 Ciclo Bayesiano

```
┌────────────────┐
│  1. PRIOR      │  Crença inicial P(θ)
│     P(θ)       │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│  2. DADOS      │  Observações X
│     X          │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│  3. LIKELIHOOD │  P(X|θ) - Modelo dos dados
│     P(X|θ)     │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│  4. BAYES      │  P(θ|X) = P(X|θ) × P(θ) / P(X)
│                │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│  5. POSTERIOR  │  Crença atualizada P(θ|X)
│     P(θ|X)     │
└───────┬────────┘
        │
        │  (Pode virar novo prior!)
        └────────────┐
                     │
                     ▼
              Novos dados...
```

---

## 🎲 Exemplo Completo: [[Acurácia]] de [[Aplicação_ao_IoT_IDS|IDS]]

### Cenário

Você desenvolveu um IDS e quer estimar sua verdadeira acurácia θ.

### Passo 1: Especificar Prior

**Sem conhecimento prévio:**
```
θ ~ Beta(1, 1)  = Uniforme(0, 1)
```

"Qualquer acurácia entre 0% e 100% é igualmente plausível."

**Com conhecimento prévio:**
```
θ ~ Beta(10, 3)  # Média ≈ 0.77
```

"Com base em sistemas similares, espero acurácia em torno de 77%."

### Passo 2: Coletar Dados

```
Teste o IDS em 100 conexões
Resultado: 90 acertos, 10 erros
```

### Passo 3: Definir Likelihood

**Modelo:** Cada classificação é Bernoulli(θ)

```
P(dados|θ) = θ^90 × (1-θ)^10  (ignorando constantes)
```

Mais formalmente:
```
X | θ ~ Binomial(100, θ)
P(X=90|θ) = C(100,90) × θ^90 × (1-θ)^10
```

### Passo 4: Aplicar Bayes

**Com prior não-informativo:**
```
Prior: Beta(1, 1)
Likelihood: Binomial(90 | 100, θ)

Posterior: Beta(1+90, 1+10) = Beta(91, 11)
```

**Com prior informativo:**
```
Prior: Beta(10, 3)
Likelihood: Binomial(90 | 100, θ)

Posterior: Beta(10+90, 3+10) = Beta(100, 13)
```

### Passo 5: Interpretar Posterior

```python
from scipy import stats

# Sem prior
post_uninformative = stats.beta(91, 11)
print(f"Média: {post_uninformative.mean():.3f}")  # 0.892
print(f"IC 95%: {post_uninformative.interval(0.95)}")  # [0.838, 0.946]

# Com prior
post_informative = stats.beta(100, 13)
print(f"Média: {post_informative.mean():.3f}")  # 0.885
print(f"IC 95%: {post_informative.interval(0.95)}")  # [0.835, 0.929]
```

**Interpretação:**
> "Dado que observei 90 acertos em 100 tentativas, há 95% de probabilidade da verdadeira acurácia estar entre 83.8% e 94.6%."

---

## ⚖️ Comparação: Bayesiano vs. Frequentista

### Filosofia

| Aspecto | Frequentista | Bayesiano |
|---------|-------------|-----------|
| **Parâmetro θ** | Fixo, desconhecido | Variável aleatória |
| **Dados** | Aleatórios | Fixos (observados) |
| **Probabilidade** | Frequência longo prazo | Grau de crença |
| **Prior** | Não existe | Essencial |
| **Resultado** | Ponto + IC | Distribuição completa |
| **Interpretação IC** | "95% dos ICs capturam θ" | "95% prob. de θ estar ali" |

### Exemplo: 90 Acertos em 100 Testes

#### Abordagem Frequentista
```
Estimativa: p̂ = 0.90
IC 95%: [0.831, 0.946]  (Wilson score)

Interpretação:
"Se repetirmos o experimento infinitas vezes,
95% dos intervalos conterão o verdadeiro θ."

Não podemos dizer: "95% de probabilidade de θ estar em [0.831, 0.946]"
(θ é fixo, ou está ou não está!)
```

#### Abordagem Bayesiana
```
Posterior: θ ~ Beta(91, 11)
IC 95%: [0.838, 0.946]

Interpretação:
"Há 95% de probabilidade de θ estar em [0.838, 0.946]"

Além disso:
P(θ > 0.85) = 0.973
P(θ > 0.90) = 0.503
```

**Vantagem Bayesiana:** Interpretação probabilística direta!

---

## 🔗 Priors Conjugados

### Definição

Um **prior conjugado** para uma likelihood é aquele cuja posterior tem a **mesma família de distribuições** do prior.

```
Prior: Família F
+ Likelihood: Distribuição D
= Posterior: Família F (mesma!)
```

**Vantagem:** Posterior tem forma analítica fechada!

### Principais Pares Conjugados

| Likelihood | Prior Conjugado | Posterior | Uso |
|------------|-----------------|-----------|-----|
| **Binomial(n, θ)** | **[[Distribuição_Beta\|Beta(α, β)]]** | **Beta(α+k, β+n-k)** | **Proporções** ⭐ |
| Poisson(λ) | Gamma(α, β) | Gamma(α+∑x, β+n) | Contagens |
| Normal(μ, σ²) | Normal(μ₀, τ²) | Normal(...) | Médias |
| Exponencial(λ) | Gamma(α, β) | Gamma(α+n, β+∑x) | Tempos |
| Multinomial | Dirichlet | Dirichlet | Multiclasse |

### Beta-Binomial (Do [[The_Balanced_Accuracy_and_Its_Posterior_Distribution|Artigo]])

**Setup:**
```
Prior: θ ~ Beta(α, β)
Likelihood: X ~ Binomial(n, θ), observamos k sucessos
Posterior: θ | X ~ Beta(α + k, β + n - k)
```

**Por que é mágico:**
```python
# Atualização sequencial é trivial!
alpha, beta = 1, 1  # Prior

# Observação 1: acerto
alpha += 1

# Observação 2: acerto
alpha += 1

# Observação 3: erro
beta += 1

# Posterior: Beta(3, 2)
# Média = 3/(3+2) = 0.6 = 60%
```

---

## 📊 Escolhendo o Prior

### Tipos de Priors

#### 1. Não-Informativo (Uninformative)

**Objetivo:** Deixar dados falarem.

**Exemplos:**
```
Beta(1, 1) = Uniforme(0, 1)
Beta(0.5, 0.5)  # Jeffreys prior
```

**Uso:** Quando não tem conhecimento prévio.

#### 2. Informativo (Informative)

**Objetivo:** Incorporar conhecimento de domínio.

**Exemplo:**
```
# Sei que sistemas similares têm acc ~ 85%
# Com moderada confiança (equivalente a ~20 observações)
Beta(17, 3)  # Média = 17/20 = 0.85
```

**Uso:** Quando tem informação prévia valiosa.

#### 3. Fracamente Informativo (Weakly Informative)

**Objetivo:** Regularização suave, evitar valores absurdos.

**Exemplo:**
```
# Acurácia deve ser razoável, mas sem certeza forte
Beta(2, 2)  # Simétrico, suavemente centrado em 0.5
```

**Uso:** Balanço entre informação e flexibilidade.

### Sensibilidade ao Prior

**Com poucos dados:** Posterior dominado pelo prior

**Com muitos dados:** Dados dominam, prior tem pouco efeito

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Diferentes priors
priors = [
    (1, 1, "Não-informativo"),
    (10, 3, "Informativo (otimista)"),
    (3, 10, "Informativo (pessimista)")
]

# Dados: 90/100 acertos
k, n = 90, 10

x = np.linspace(0, 1, 1000)

for alpha0, beta0, label in priors:
    posterior = stats.beta(alpha0 + k, beta0 + n)
    plt.plot(x, posterior.pdf(x), label=f"{label}: Beta({alpha0+k},{beta0+n})")

plt.xlabel('θ (acurácia)')
plt.ylabel('Densidade')
plt.title('Posteriors com Diferentes Priors (90/100 acertos)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

**Conclusão:** Com 100 dados, todos convergem para região similar!

---

## 🎯 Aplicação no [[The_Balanced_Accuracy_and_Its_Posterior_Distribution|Artigo]]

### Para [[Acurácia]] Simples

```python
from scipy import stats

# Dados agregados de cross-validation
corretos = 93
incorretos = 7

# Prior não-informativo
prior = stats.beta(1, 1)

# Posterior
posterior = stats.beta(corretos + 1, incorretos + 1)

# Estatísticas
print(f"Média: {posterior.mean():.3f}")
print(f"IC 95%: {posterior.interval(0.95)}")

# Probabilidades
print(f"P(acc > 0.85): {1 - posterior.cdf(0.85):.3f}")
```

### Para [[Acurácia_Balanceada]]

**Mais complexo:** Duas distribuições independentes.

```python
# Dados por classe
TP, FN = 450, 50  # Classe positiva
TN, FP = 9300, 200  # Classe negativa

# Posteriors independentes
pos_posterior = stats.beta(TP + 1, FN + 1)
neg_posterior = stats.beta(TN + 1, FP + 1)

# BA via amostragem (convolução)
n_samples = 100000
pos_samples = pos_posterior.rvs(n_samples)
neg_samples = neg_posterior.rvs(n_samples)
ba_samples = 0.5 * (pos_samples + neg_samples)

# Estatísticas
print(f"BA média: {np.mean(ba_samples):.3f}")
print(f"IC 95%: {np.percentile(ba_samples, [2.5, 97.5])}")
```

---

## 🧮 Comparação de Modelos Bayesiana

### Bayes Factor

```
BF = P(dados|Modelo A) / P(dados|Modelo B)
```

**Interpretação:**
- BF > 10: Evidência forte para A
- BF > 100: Evidência muito forte para A
- BF < 0.1: Evidência forte para B

### Probabilidade Direta

Mais simples: amostrar de ambas posteriors!

```python
# Dois algoritmos
model_A_post = stats.beta(90 + 1, 10 + 1)
model_B_post = stats.beta(85 + 1, 15 + 1)

# Amostrar
samples_A = model_A_post.rvs(100000)
samples_B = model_B_post.rvs(100000)

# P(A > B)
prob_A_better = np.mean(samples_A > samples_B)
print(f"P(A > B) = {prob_A_better:.3f}")

if prob_A_better > 0.95:
    print("A é significativamente melhor que B!")
```

---

## 💻 Ferramentas Computacionais

### Python: Scipy (Analítico)

Para casos simples com conjugação:

```python
from scipy import stats

posterior = stats.beta(91, 11)
samples = posterior.rvs(10000)
```

### Python: PyMC (MCMC)

Para modelos complexos:

```python
import pymc as pm

with pm.Model() as model:
    # Prior
    theta = pm.Beta('theta', alpha=1, beta=1)
    
    # Likelihood
    obs = pm.Binomial('obs', n=100, p=theta, observed=90)
    
    # Sample
    trace = pm.sample(2000, return_inferencedata=True)

# Análise
print(pm.summary(trace))
```

### R: rstan, brms

Framework robusto para modelos hierárquicos.

---

## 📚 Referências

### Livros Essenciais
- **McElreath, R.** (2020). *Statistical Rethinking* (2nd ed.). CRC Press. ⭐ **MAIS DIDÁTICO!**
- **Gelman, A., et al.** (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press. ⭐ **REFERÊNCIA DEFINITIVA**
- **Kruschke, J.K.** (2014). *Doing Bayesian Data Analysis* (2nd ed.). Academic Press. **Excelente para iniciantes**

### Fundamentos
- **Sivia, D.S. & Skilling, J.** (2006). *Data Analysis: A Bayesian Tutorial* (2nd ed.). Oxford University Press.
- **Jaynes, E.T.** (2003). *Probability Theory: The Logic of Science*. Cambridge University Press.

### Aplicado
- **Bishop, C.M.** (2006). *Pattern Recognition and Machine Learning*. Springer. [Seção 2.2] **Citado no artigo!**

### Online
- [3Blue1Brown: Bayes Theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM)
- [Seeing Theory: Bayesian Inference](https://seeing-theory.brown.edu/bayesian-inference/)
- [PyMC Documentation](https://www.pymc.io/)

Veja [[Referências_Bibliográficas]] para lista completa.

---

## 🔗 Conceitos Relacionados

### Fundamentos
- [[Distribuições_de_Probabilidade]] - Priors e posteriors
- [[Distribuição_Beta]] - Prior/posterior para proporções
- [[Intervalos_de_Confiança]] - Intervalos de credibilidade

### Filosofia
- [[Métodos_Paramétricos_vs_Não_Paramétricos]] - Bayesiano é paramétrico

### Aplicações
- [[Acurácia]] - Modelagem Bayesiana
- [[Acurácia_Balanceada]] - Com posteriors
- [[The_Balanced_Accuracy_and_Its_Posterior_Distribution]] - Artigo usa framework Bayesiano
- [[Aplicação_ao_IoT_IDS]] - Uso prático

---

## 🎯 Exercícios

Veja [[Exercícios_Práticos#Inferência Bayesiana]].

---

## 📌 Resumo Visual

```
┌───────────────────────────────────────────────┐
│         INFERÊNCIA BAYESIANA                  │
│                                               │
│  Teorema de Bayes:                            │
│  P(θ|dados) = P(dados|θ) × P(θ) / P(dados)   │
│                                               │
│  ┌─────────┐                                 │
│  │  PRIOR  │  Crença inicial                 │
│  └────┬────┘                                 │
│       │                                       │
│       ├──► + LIKELIHOOD (dados)              │
│       │                                       │
│       ▼                                       │
│  ┌──────────┐                                │
│  │ POSTERIOR│  Crença atualizada             │
│  └──────────┘                                │
│                                               │
│  Para Acurácia:                              │
│  • Prior: Beta(α, β)                         │
│  • Likelihood: Binomial                      │
│  • Posterior: Beta(α+k, β+n-k) ✅            │
│                                               │
│  Artigo usa este framework!                  │
│                                               │
└───────────────────────────────────────────────┘
```

---

**Tags:** #bayesian #inference #prior #posterior #likelihood #bayes-theorem #beta-binomial

**Voltar para:** [[INDEX]]  
**Framework:** [[Distribuição_Beta]]  
**Artigo:** [[The_Balanced_Accuracy_and_Its_Posterior_Distribution]]  
**Aplicação:** [[Aplicação_ao_IoT_IDS]]


