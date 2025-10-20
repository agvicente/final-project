# Distribuição Beta

> **Tipo:** Distribuição de Probabilidade Contínua  
> **Complexidade:** ⭐⭐⭐⭐☆ (Avançado)  
> **Aplicação:** Modelagem de Proporções, Probabilidades, [[Acurácia]]

---

## 🎯 O que é a Distribuição Beta?

A **distribuição Beta** é uma família de [[Distribuições_de_Probabilidade|distribuições contínuas]] definidas no intervalo **[0, 1]**, tornando-a **perfeita** para modelar:
- Proporções
- Probabilidades
- Taxas
- **[[Acurácia|Acurácias]] de modelos** ⭐

**Por que é central no [[The_Balanced_Accuracy_and_Its_Posterior_Distribution|artigo]]?**
Resolve o problema de [[Intervalos_de_Confiança|intervalos de confiança]] que violam [0,1]!

---

## 📐 Definição Matemática

### Função Densidade de Probabilidade (PDF)

```
Beta(x; α, β) = [x^(α-1) × (1-x)^(β-1)] / B(α, β)
```

Onde:
- **x ∈ [0, 1]:** Valor da variável (ex: acurácia)
- **α > 0:** Parâmetro de forma ("sucessos" + 1)
- **β > 0:** Parâmetro de forma ("falhas" + 1)
- **B(α, β):** Função Beta (constante normalizadora)

### Função Beta

```
B(α, β) = ∫₀¹ t^(α-1) × (1-t)^(β-1) dt

        = Γ(α) × Γ(β) / Γ(α + β)
```

Onde **Γ** é a função Gamma (generalização do fatorial).

**Para inteiros:**
```
B(α, β) = (α-1)! × (β-1)! / (α+β-1)!
```

---

## 🎨 Interpretação dos Parâmetros

### Forma Intuitiva

**Contexto [[Inferência_Bayesiana|Bayesiano]]:**
```
α = número de "sucessos" + 1
β = número de "falhas" + 1
```

**Exemplo [[Aplicação_ao_IoT_IDS|IDS]]:**
```
95 classificações corretas
5 classificações incorretas

→ Beta(96, 6)

α = 95 + 1 = 96
β = 5 + 1 = 6
```

### Efeito dos Parâmetros

#### α: "Força dos Sucessos"
- ↑ α: distribui massa para **direita** (valores altos)
- ↓ α: distribui massa para **esquerda** (valores baixos)

#### β: "Força das Falhas"  
- ↑ β: distribui massa para **esquerda** (valores baixos)
- ↓ β: distribui massa para **direita** (valores altos)

#### α + β: "Força das Evidências"
- ↑ (α+β): distribuição mais **concentrada** (menos incerteza)
- ↓ (α+β): distribuição mais **espalhada** (mais incerteza)

---

## 📊 Formas da Beta

### Visualização de Formas Clássicas

```
Beta(1, 1): ───────────  Uniforme (total ignorância)

Beta(2, 2):    ╱╲       Simétrica suave
              ╱  ╲

Beta(5, 2):      ╱╲     Pico à direita (alta acurácia)
                ╱  ╲___

Beta(2, 5): ___╱  ╲     Pico à esquerda (baixa acurácia)
               ╲  ╱

Beta(0.5, 0.5): U       Forma de U (valores extremos)

Beta(20, 20):   ▲       Muito concentrada (muita evidência)
```

### Casos Especiais

**Beta(1, 1):**
```
Uniforme em [0,1]
f(x) = 1
```
Prior não-informativo - total ignorância.

**Beta(α, α) com α grande:**
```
Aproxima Normal
Simétrica, concentrada em 0.5
```

**Beta(α, 1) ou Beta(1, β):**
```
Distribuições crescentes/decrescentes
```

---

## 📈 Momentos e Estatísticas

### Média (Esperança)

```
E[X] = μ = α / (α + β)
```

**Exemplo:**
```
Beta(91, 11):
μ = 91 / (91 + 11) = 91/102 ≈ 0.892 = 89.2%
```

### Moda

```
Moda = (α - 1) / (α + β - 2)   se α, β > 1
```

**Exemplo:**
```
Beta(91, 11):
Moda = (91-1) / (91+11-2) = 90/100 = 0.90 = 90%
```

**Interessante:** A moda ≈ acurácia observada (90/100)!

### Mediana

Não tem forma fechada, mas para simétrica (α=β):
```
Mediana = 0.5
```

### Variância

```
Var(X) = σ² = αβ / [(α+β)²(α+β+1)]
```

**Exemplo:**
```
Beta(91, 11):
σ² = 91×11 / [102² × 103]
   ≈ 0.000935
σ ≈ 0.0306 = 3.06%
```

### Desvio Padrão

```
σ = √[αβ / [(α+β)²(α+β+1)]]
```

---

## 🧮 Por que Beta é Perfeita para [[Acurácia]]?

### 1. Suporte Natural [0,1]

**Problema com Normal:**
```
Normal(0.98, 0.02²) pode gerar valores > 1 ❌
```

**Beta sempre respeita limites:**
```
Beta(α, β) só gera valores em [0, 1] ✅
```

### 2. Flexibilidade

Pode representar:
- Ignorância total: Beta(1,1)
- Alta confiança: Beta(100, 10)
- Baixo desempenho: Beta(10, 100)
- Qualquer forma intermediária!

### 3. Conjugação com Binomial

**Classificação binária** = sequência de Bernoulli = Binomial!

```
Prior: θ ~ Beta(α₀, β₀)
Likelihood: k sucessos em n → Binomial(n, θ)
Posterior: θ ~ Beta(α₀+k, β₀+n-k)  ✅ Também é Beta!
```

**Matemática elegante!** Veja [[Inferência_Bayesiana#Conjugate Priors]].

### 4. Interpretação Intuitiva

```
α = "evidência de sucesso"
β = "evidência de falha"
μ = α/(α+β) = "taxa de sucesso"
```

Faz sentido intuitivo! 💡

---

## 🔬 Beta no [[The_Balanced_Accuracy_and_Its_Posterior_Distribution|Artigo Brodersen]]

### Para [[Acurácia]] Simples

**Dados:** C corretos, I incorretos

**Prior não-informativo:** Beta(1, 1)

**Posterior:**
```
A ~ Beta(C+1, I+1)
```

**Exemplo:**
```
90 corretos, 10 incorretos
→ Beta(91, 11)

Média: 89.2%
IC 95%: [83.8%, 94.6%]
```

### Para [[Acurácia_Balanceada]]

**Dados:** TP, FN, TN, FP

**Acurácia em cada classe:**
```
A_pos ~ Beta(TP+1, FN+1)
A_neg ~ Beta(TN+1, FP+1)
```

**Balanced Accuracy:**
```
BA = ½(A_pos + A_neg)
```

**Distribuição de BA:** Convolução! (seção abaixo)

---

## 🌀 Convolução para Balanced Accuracy

### O Problema

Queremos a distribuição de:
```
BA = ½(A_pos + A_neg)
```

Onde A_pos e A_neg são **independentes** e seguem Betas.

### Solução: Convolução

**Do artigo, Equação (7):**

```
p_BA(x; TP, FP, FN, TN) = 
    ∫₀¹ p_A(2(x-z); TP+1, FN+1) × p_A(2z; TN+1, FP+1) dz
```

Onde:
- `p_A(x)` é a PDF da Beta (definida acima)
- `p_BA(x)` é a PDF da balanced accuracy

### Propriedades

- ❌ Não tem forma analítica fechada
- ✅ Pode ser calculada numericamente
- ✅ Respeita [0,1] naturalmente
- ✅ Captura correlação entre classes

### Implementação (Aproximação Monte Carlo)

```python
from scipy import stats
import numpy as np

def balanced_accuracy_distribution(TP, FN, TN, FP, n_samples=100000):
    """
    Aproxima distribuição da balanced accuracy
    via amostragem Monte Carlo.
    """
    # Posteriors para cada classe
    pos_post = stats.beta(TP + 1, FN + 1)
    neg_post = stats.beta(TN + 1, FP + 1)
    
    # Amostrar
    pos_samples = pos_post.rvs(n_samples)
    neg_samples = neg_post.rvs(n_samples)
    
    # Balanced accuracy
    ba_samples = 0.5 * (pos_samples + neg_samples)
    
    return ba_samples

# Exemplo do artigo (C2)
# 45 positivos: 43 TP, 2 FN
# 10 negativos: 3 TN, 7 FP
TP, FN, TN, FP = 43, 2, 3, 7

ba_samples = balanced_accuracy_distribution(TP, FN, TN, FP)

print(f"BA média: {np.mean(ba_samples):.3f}")
print(f"IC 95%: [{np.percentile(ba_samples, 2.5):.3f}, "
      f"{np.percentile(ba_samples, 97.5):.3f}]")

# Resultado aproximado:
# BA média: 0.512
# IC 95%: [0.384, 0.639]
```

---

## 🎲 Simulação e Visualização

### Comparando Diferentes Evidências

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

x = np.linspace(0, 1, 1000)

# Diferentes níveis de evidência
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

configs = [
    (2, 2, "Pouca evidência balanceada"),
    (10, 10, "Evidência moderada balanceada"),
    (91, 11, "Muita evidência (90/100 acertos)"),
    (10, 91, "Muita evidência (10/100 acertos)")
]

for ax, (alpha, beta, title) in zip(axes.flat, configs):
    dist = stats.beta(alpha, beta)
    
    ax.plot(x, dist.pdf(x), 'b-', lw=2)
    ax.fill_between(x, dist.pdf(x), alpha=0.3)
    
    # Estatísticas
    mean = dist.mean()
    ci = dist.interval(0.95)
    
    ax.axvline(mean, color='r', linestyle='--', 
               label=f'Média: {mean:.3f}')
    ax.axvline(ci[0], color='g', linestyle=':', alpha=0.5)
    ax.axvline(ci[1], color='g', linestyle=':', alpha=0.5,
               label=f'IC 95%: [{ci[0]:.3f}, {ci[1]:.3f}]')
    
    ax.set_title(f'{title}\nBeta({alpha}, {beta})')
    ax.set_xlabel('x (acurácia)')
    ax.set_ylabel('Densidade')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('beta_comparison.png', dpi=300)
plt.show()
```

### Atualização Bayesiana Sequencial

```python
# Começar com prior não-informativo
prior = stats.beta(1, 1)

# Observar dados sequencialmente
observations = [(1,0), (1,0), (0,1), (1,0), (1,0)]  # (sucesso, falha)

alpha, beta = 1, 1

for i, (succ, fail) in enumerate(observations):
    # Atualizar
    alpha += succ
    beta += fail
    
    # Posterior atual
    posterior = stats.beta(alpha, beta)
    
    # Visualizar evolução
    x = np.linspace(0, 1, 1000)
    plt.plot(x, posterior.pdf(x), 
             label=f'Após {i+1} obs: Beta({alpha},{beta})')

plt.title('Evolução da Crença Bayesiana')
plt.xlabel('θ (taxa de sucesso)')
plt.ylabel('Densidade')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

---

## 📚 Propriedades Matemáticas Avançadas

### Função Geradora de Momentos

Não tem forma fechada simples, mas momentos podem ser calculados:

```
E[X^n] = (α)_n / (α+β)_n
```

Onde `(a)_n` é o símbolo de Pochhammer (rising factorial).

### Entropia

```
H(X) = ln B(α,β) - (α-1)ψ(α) - (β-1)ψ(β) + (α+β-2)ψ(α+β)
```

Onde ψ é a função digamma.

### Informação de Fisher

```
I(α,β) = matriz 2×2 das segundas derivadas de ln L
```

---

## 🔄 Relações com Outras Distribuições

### Beta e Uniforme

```
Beta(1, 1) = Uniforme(0, 1)
```

### Beta e Binomial

**Conjugação:**
```
Prior: Beta(α, β)
+ Likelihood: Binomial
= Posterior: Beta(α+k, β+n-k)
```

### Beta e Dirichlet

**Generalização multiclasse:**
```
Beta(α, β) é caso especial de Dirichlet(α, β)
```

Para [[Acurácia_Balanceada]] multiclasse!

### Beta e F de Fisher

```
Se X ~ Beta(α, β), então
Y = (X/α) / ((1-X)/β) ~ F(2α, 2β)
```

---

## 💻 Implementação Completa

### Classe Beta Helper

```python
from scipy import stats
import numpy as np

class BetaAccuracy:
    """Helper para trabalhar com acurácias modeladas como Beta."""
    
    def __init__(self, correct, incorrect):
        """
        Args:
            correct: número de classificações corretas
            incorrect: número de classificações incorretas
        """
        self.C = correct
        self.I = incorrect
        self.alpha = correct + 1
        self.beta = incorrect + 1
        self.dist = stats.beta(self.alpha, self.beta)
    
    def mean(self):
        """Acurácia média."""
        return self.dist.mean()
    
    def median(self):
        """Mediana."""
        return self.dist.median()
    
    def mode(self):
        """Moda (MLE)."""
        if self.alpha > 1 and self.beta > 1:
            return (self.alpha - 1) / (self.alpha + self.beta - 2)
        return np.nan
    
    def std(self):
        """Desvio padrão."""
        return self.dist.std()
    
    def interval(self, confidence=0.95):
        """Intervalo de credibilidade."""
        return self.dist.interval(confidence)
    
    def prob_greater_than(self, threshold):
        """P(acurácia > threshold)."""
        return 1 - self.dist.cdf(threshold)
    
    def prob_between(self, lower, upper):
        """P(lower < acurácia < upper)."""
        return self.dist.cdf(upper) - self.dist.cdf(lower)
    
    def compare_with(self, other_beta_acc, n_samples=100000):
        """
        Compara com outro modelo.
        Retorna P(este modelo > outro modelo).
        """
        samples_self = self.dist.rvs(n_samples)
        samples_other = other_beta_acc.dist.rvs(n_samples)
        return np.mean(samples_self > samples_other)
    
    def summary(self):
        """Resumo completo."""
        ci = self.interval(0.95)
        return f"""
        Beta({self.alpha}, {self.beta})
        ─────────────────────────────
        Média:    {self.mean():.4f}
        Mediana:  {self.median():.4f}
        Moda:     {self.mode():.4f}
        Desvio:   {self.std():.4f}
        IC 95%:   [{ci[0]:.4f}, {ci[1]:.4f}]
        """

# Uso
model_A = BetaAccuracy(correct=90, incorrect=10)
print(model_A.summary())

# P(acurácia > 85%)
print(f"P(acc > 0.85): {model_A.prob_greater_than(0.85):.3f}")

# Comparar modelos
model_B = BetaAccuracy(correct=85, incorrect=15)
prob = model_A.compare_with(model_B)
print(f"P(A > B): {prob:.3f}")
```

---

## 📚 Referências

### Livros Específicos
- **Gelman, A., et al.** (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press. [Seção 2.4: "Bayesian inference for Binomial data"]
- **Kruschke, J.K.** (2014). *Doing Bayesian Data Analysis* (2nd ed.). Academic Press. [Capítulo 6: "Inferring a Binomial Probability"]
- **Bishop, C.M.** (2006). *Pattern Recognition and Machine Learning*. Springer. [Seção 2.2, pp. 68-74] ⭐ **Citado no artigo!**

### Papers
- **Brodersen et al.** (2010). [[The_Balanced_Accuracy_and_Its_Posterior_Distribution|"The balanced accuracy and its posterior distribution"]]. ICPR.

### Online
- [Beta Distribution - Wikipedia](https://en.wikipedia.org/wiki/Beta_distribution)
- [Seeing Theory: Beta Distribution](https://seeing-theory.brown.edu/bayesian-inference/index.html)
- [Scipy Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html)

Veja [[Referências_Bibliográficas]] para lista completa.

---

## 🔗 Conceitos Relacionados

### Fundamentos
- [[Distribuições_de_Probabilidade]] - Contexto geral
- [[Média_Desvio_Padrão_Erro_Padrão]] - Momentos
- [[Intervalos_de_Confiança]] - Aplicação principal

### Paradigma
- [[Inferência_Bayesiana]] - Framework de uso
- [[Métodos_Paramétricos_vs_Não_Paramétricos]] - Escolha paramétrica

### Aplicações
- [[Acurácia]] - Modelada como Beta
- [[Acurácia_Balanceada]] - Convolução de Betas
- [[The_Balanced_Accuracy_and_Its_Posterior_Distribution]] - Artigo principal
- [[Aplicação_ao_IoT_IDS]] - Uso prático

---

## 🎯 Exercícios

Veja [[Exercícios_Práticos#Distribuição Beta]].

---

## 📌 Resumo Visual

```
┌───────────────────────────────────────────────┐
│            DISTRIBUIÇÃO BETA                  │
│                                               │
│  Beta(α, β) para x ∈ [0,1]                   │
│                                               │
│  α = "sucessos" + 1                          │
│  β = "falhas" + 1                            │
│                                               │
│  ✅ Suporte [0,1] - perfeito para proporções │
│  ✅ Flexível - múltiplas formas              │
│  ✅ Conjugada à Binomial                     │
│  ✅ Interpretação intuitiva                  │
│                                               │
│  Momentos:                                    │
│  • Média: α/(α+β)                            │
│  • Variância: αβ/[(α+β)²(α+β+1)]            │
│                                               │
│  Uso: Modelar ACURÁCIAS!                     │
│                                               │
└───────────────────────────────────────────────┘
```

---

**Tags:** #beta-distribution #bayesian #probability #accuracy #proportion #conjugate-prior

**Voltar para:** [[INDEX]]  
**Contexto:** [[Distribuições_de_Probabilidade]]  
**Artigo:** [[The_Balanced_Accuracy_and_Its_Posterior_Distribution]]  
**Aplicação:** [[Acurácia_Balanceada]]

