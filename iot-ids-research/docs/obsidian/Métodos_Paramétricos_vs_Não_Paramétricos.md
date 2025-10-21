# Métodos Paramétricos vs. Não-Paramétricos

> **Tipo:** Filosofia Estatística  
> **Complexidade:** ⭐⭐⭐☆☆ (Intermediário-Avançado)  
> **Aplicação:** Escolha de Abordagem de Modelagem

---

## 🎯 A Diferença Fundamental

A distinção entre métodos **paramétricos** e **não-paramétricos** está em **como modelamos os dados**:

```
┌────────────────────────────────────────┐
│  PARAMÉTRICO                           │
│  "Assumo que os dados seguem           │
│   distribuição específica com          │
│   parâmetros fixos"                    │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│  NÃO-PARAMÉTRICO                       │
│  "Não assumo distribuição específica;  │
│   deixo os dados falarem"              │
└────────────────────────────────────────┘
```

---

## 🔵 Métodos Paramétricos

### Definição

Um método é **paramétrico** quando:
1. Assume que os dados seguem uma [[Distribuições_de_Probabilidade|distribuição específica]]
2. A distribuição é caracterizada por um **conjunto finito e fixo de parâmetros**

### Exemplo: [[Acurácia]] como [[Distribuição_Beta|Beta]]

```
Assumo: Acurácia θ ~ Beta(α, β)

Parâmetros: α, β  (apenas 2!)

Toda a distribuição é descrita por α e β
```

### O que Significa "Parâmetro"?

**Parâmetro (θ):**
- Característica da **população** (desconhecida)
- Exemplo: verdadeira acurácia do modelo = θ

**Estatística:**
- Valor calculado da **amostra** (observado)
- Exemplo: acurácia observada = x̄

**Objetivo:** Estimar θ a partir dos dados.

### Vantagens

✅ **Eficiência:** Usa toda a informação dos dados de forma ótima (se modelo correto)

✅ **Poder:** Testes estatísticos mais poderosos

✅ **Precisão:** Estimativas mais precisas com menos dados

✅ **Interpretação:** Parâmetros têm significado claro

✅ **Inferência:** Intervalos de confiança matematicamente rigorosos

### Desvantagens

❌ **Suposição forte:** Se a distribuição assumida estiver errada, resultados podem ser ruins

❌ **Falta de flexibilidade:** Limitado às formas da distribuição escolhida

❌ **Verificação:** Precisa verificar se suposições são válidas

### Exemplos Clássicos

**Teste t de Student:**
```
Assume: dados ~ Normal(μ, σ²)
Parâmetros: μ, σ²
Testa: H₀: μ = μ₀
```

**Regressão Linear:**
```
Assume: y = β₀ + β₁x + ε, onde ε ~ Normal(0, σ²)
Parâmetros: β₀, β₁, σ²
```

**[[The_Balanced_Accuracy_and_Its_Posterior_Distribution|Artigo Brodersen]]:**
```
Assume: Acurácia ~ Beta(α, β)
Parâmetros: α, β
```

---

## 🟢 Métodos Não-Paramétricos

### Definição

Um método é **não-paramétrico** quando:
1. **NÃO assume** distribuição específica dos dados
2. Número de "parâmetros" pode crescer com os dados (ou não usa parâmetros explícitos)

**Nome enganoso:** "Não-paramétrico" não significa "sem parâmetros", mas sim "livre de distribuição"!

### Exemplo: [[Média_Desvio_Padrão_Erro_Padrão|Média ± Erro Padrão]]

```
Não assumo distribuição específica

Apenas calculo:
- x̄ = ∑xᵢ / n
- SE = s / √n
- IC ≈ x̄ ± 2×SE
```

### Vantagens

✅ **Flexibilidade:** Funciona para qualquer distribuição

✅ **Robustez:** Menos sensível a outliers e violações de suposições

✅ **Simplicidade:** Não precisa verificar suposições distributivas

✅ **Segurança:** Menos propenso a erros por modelo incorreto

### Desvantagens

❌ **Eficiência:** Pode precisar de mais dados para mesma precisão

❌ **Poder:** Testes menos poderosos

❌ **[[Intervalos_de_Confiança|Intervalos inadequados]]:** Podem violar limites naturais

❌ **Interpretação:** Menos estrutura teórica

### Exemplos Clássicos

**Teste de Mann-Whitney U:**
```
Não assume distribuição
Compara medianas de dois grupos
```

**Teste de Wilcoxon:**
```
Não assume normalidade
Testa diferença entre pares
```

**Bootstrap:**
```
Reamostragem dos dados
Estima distribuição empiricamente
```

---

## ⚖️ Comparação Direta

### Tabela Comparativa

| Aspecto | Paramétrico | Não-Paramétrico |
|---------|-------------|-----------------|
| **Suposições** | Distribuição específica | Mínimas/Nenhuma |
| **Parâmetros** | Fixos, finitos | Flexíveis, podem crescer |
| **Eficiência** | Alta (se correto) | Menor |
| **Robustez** | Baixa (se errado) | Alta |
| **Dados necessários** | Menos | Mais |
| **Complexidade** | Maior (modelagem) | Menor (cálculos) |
| **Interpretação** | Clara (parâmetros) | Limitada |
| **Limites naturais** | Respeitados (se bem modelado) | Podem violar |

### Exemplo: [[Acurácia]] 98% em 100 Testes

#### Abordagem Não-Paramétrica
```
p̂ = 0.98
SE = √[p̂(1-p̂)/n] = √[0.98×0.02/100] = 0.014

IC 95% = [0.98 - 1.96×0.014, 0.98 + 1.96×0.014]
       = [0.953, 1.007]  ← 100.7%! ❌
```

**Problema:** Viola limite [0,1]!

#### Abordagem Paramétrica (Beta)
```
Modelo: θ ~ Beta(99, 3)

IC 95% = [0.932, 0.997]  ✅
```

**Vantagem:** Respeita limites naturalmente!

---

## 🔄 Quando Usar Cada Um?

### Use Paramétrico Quando:

1. ✅ **Conhece a distribuição:** Processo gerador é bem compreendido
   - Exemplo: Classificação binária → Binomial/Beta

2. ✅ **Dados limitados:** Poucos dados disponíveis
   - Método paramétrico usa informação mais eficientemente

3. ✅ **Limites naturais:** Variável tem restrições (ex: [0,1])
   - Modelo paramétrico respeita isso

4. ✅ **Inferência rigorosa:** Precisa de intervalos de confiança precisos
   - Framework matemático mais sólido

5. ✅ **Paradigma [[Inferência_Bayesiana|Bayesiano]]:** Incorporando conhecimento prévio
   - Precisa especificar prior paramétrico

### Use Não-Paramétrico Quando:

1. ✅ **Distribuição desconhecida:** Não sabe qual distribuição assumir
   - Seguro não fazer suposições

2. ✅ **Violações de suposições:** Dados claramente não seguem distribuição padrão
   - Outliers, assimetrias extremas

3. ✅ **Exploração inicial:** Primeira análise dos dados
   - Entender padrões sem compromisso

4. ✅ **Muitos dados:** Dataset grande
   - Não precisa da eficiência paramétrica

5. ✅ **Simplicidade:** Análise rápida sem modelagem complexa
   - Trade-off entre simplicidade e rigor

---

## 🎯 Contexto do [[The_Balanced_Accuracy_and_Its_Posterior_Distribution|Artigo Brodersen]]

### O Problema Identificado

**Abordagem tradicional (não-paramétrica):**
```python
# Cross-validation tradicional
acuracias = [0.91, 0.89, 0.92, 0.88, 0.90]

# Média e erro padrão
media = np.mean(acuracias)
se = np.std(acuracias, ddof=1) / np.sqrt(len(acuracias))

# IC "aproximado"
ic = [media - 1.96*se, media + 1.96*se]
# Pode violar [0,1]! ❌
```

**Problemas:**
1. ❌ Não respeita limites [0, 1]
2. ❌ Assume normalidade (aproximação CLT)
3. ❌ Sempre simétrico
4. ❌ Depende de escolhas arbitrárias (número de folds)

### A Solução Proposta (Paramétrica)

**Abordagem Bayesiana com Beta:**
```python
from scipy import stats

# Agregar todos os resultados
corretos = 93
incorretos = 7

# Modelo paramétrico
posterior = stats.beta(corretos + 1, incorretos + 1)

# IC correto
ic = posterior.interval(0.95)
# [0.873, 0.971] ✅ Sempre em [0,1]!
```

**Vantagens:**
1. ✅ Respeita limites naturalmente
2. ✅ Não assume normalidade
3. ✅ Assimétrico quando apropriado
4. ✅ Interpretação probabilística direta

### Filosofia do Artigo

> "Substituir ponto estimado **não-paramétrico** por distribuição **paramétrica** completa"

**De:**
```
Acurácia = 90% ± 2%  (não-paramétrico)
```

**Para:**
```
Acurácia ~ Beta(91, 11)  (paramétrico)
```

Ganha toda a rica estrutura da distribuição Beta!

---

## 🧪 Exemplo Comparativo Completo

### Cenário: [[Aplicação_ao_IoT_IDS|Sistema IDS]]

**Dados:** 10-fold cross-validation, acurácias:
```
[0.89, 0.91, 0.88, 0.92, 0.90, 0.89, 0.91, 0.88, 0.93, 0.89]
```

### Análise Não-Paramétrica

```python
import numpy as np
from scipy import stats as sp_stats

acuracias = [0.89, 0.91, 0.88, 0.92, 0.90, 
             0.89, 0.91, 0.88, 0.93, 0.89]

# Estatísticas
media = np.mean(acuracias)
desvio = np.std(acuracias, ddof=1)
se = desvio / np.sqrt(len(acuracias))

# IC com distribuição t
ic_t = sp_stats.t.interval(0.95, df=len(acuracias)-1,
                            loc=media, scale=se)

print("NÃO-PARAMÉTRICO:")
print(f"Média: {media:.3f}")
print(f"SE: {se:.3f}")
print(f"IC 95%: [{ic_t[0]:.3f}, {ic_t[1]:.3f}]")
# IC 95%: [0.884, 0.916]
```

### Análise Paramétrica (Beta)

```python
# Assumir cada fold teve 100 testes
# Acurácias → número de acertos
acertos = [int(a * 100) for a in acuracias]
total_acertos = sum(acertos)
total_testes = len(acertos) * 100

# Modelo Beta
posterior = sp_stats.beta(total_acertos + 1, 
                          total_testes - total_acertos + 1)

# Estatísticas
media_beta = posterior.mean()
ic_beta = posterior.interval(0.95)

print("\nPARAMÉTRICO (Beta):")
print(f"Média: {media_beta:.3f}")
print(f"IC 95%: [{ic_beta[0]:.3f}, {ic_beta[1]:.3f}]")
# IC 95%: [0.884, 0.916]

# Probabilidades adicionais!
print(f"\nP(acc > 0.85): {1 - posterior.cdf(0.85):.3f}")
print(f"P(acc > 0.90): {1 - posterior.cdf(0.90):.3f}")
```

### Comparação

```
Métrica           | Não-Paramétrico | Paramétrico
───────────────────────────────────────────────────
Média             | 0.900           | 0.900
IC 95%            | [0.884, 0.916]  | [0.884, 0.916]
Respeita [0,1]    | Sim (por sorte) | Sim (sempre)
P(acc > 0.85)     | ❌ Não calcula  | ✅ 0.999
P(acc > 0.90)     | ❌ Não calcula  | ✅ 0.503
Comparar modelos  | ❌ Difícil      | ✅ Fácil
```

**Neste caso:** Resultados similares, mas abordagem paramétrica oferece **mais informação**!

---

## 💡 Conceitos Relacionados

### "Semiparamétrico"

Meio-termo: alguns parâmetros, mas distribuição flexível.

**Exemplo:** Regressão de Cox (survival analysis)

### Bootstrap (Não-Paramétrico Avançado)

```python
# Reamostragem para estimar distribuição
from sklearn.utils import resample

bootstrap_means = []
for _ in range(10000):
    sample = resample(acuracias)
    bootstrap_means.append(np.mean(sample))

# IC via percentis
ic_bootstrap = np.percentile(bootstrap_means, [2.5, 97.5])
```

**Vantagem:** Não assume distribuição!  
**Desvantagem:** Computacionalmente intensivo.

---

## 📚 Referências

### Livros
- **Wasserman, L.** (2006). *All of Nonparametric Statistics*. Springer. [Capítulo 1: "Parametric vs Nonparametric"]
- **Sheskin, D.J.** (2011). *Handbook of Parametric and Nonparametric Statistical Procedures* (5th ed.). CRC Press.
- **Hollander, M., Wolfe, D.A., & Chicken, E.** (2013). *Nonparametric Statistical Methods* (3rd ed.). Wiley.

### Papers
- **Brodersen et al.** (2010). [[The_Balanced_Accuracy_and_Its_Posterior_Distribution|"The balanced accuracy and its posterior distribution"]]. ICPR.

### Online
- [Scipy: stats module](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Khan Academy: Hypothesis Testing](https://www.khanacademy.org/math/statistics-probability)

Veja [[Referências_Bibliográficas]] para lista completa.

---

## 🔗 Conceitos Relacionados

### Fundamentos
- [[Distribuições_de_Probabilidade]] - Framework paramétrico
- [[Distribuição_Beta]] - Exemplo paramétrico específico
- [[Média_Desvio_Padrão_Erro_Padrão]] - Estatísticas não-paramétricas

### Paradigmas
- [[Inferência_Bayesiana]] - Sempre paramétrica
- [[Intervalos_de_Confiança]] - Ambas as abordagens

### Aplicações
- [[Acurácia]] - Pode usar ambas
- [[The_Balanced_Accuracy_and_Its_Posterior_Distribution]] - Advocacia pela paramétrica

---

## 🎯 Exercícios

Veja [[Exercícios_Práticos#Paramétrico vs Não-Paramétrico]].

---

## 📌 Resumo Visual

```
┌──────────────────────────────────────────────┐
│    PARAMÉTRICO vs NÃO-PARAMÉTRICO            │
│                                              │
│  PARAMÉTRICO                                 │
│  ─────────────────                           │
│  • Assume distribuição específica            │
│  • Parâmetros fixos                          │
│  • Mais eficiente (se correto)               │
│  • Respeita limites naturais                 │
│  ✅ Artigo usa este!                         │
│                                              │
│  NÃO-PARAMÉTRICO                             │
│  ──────────────────                          │
│  • Sem assumir distribuição                 │
│  • Flexível                                  │
│  • Mais robusto                              │
│  • Pode violar limites                       │
│  ❌ Problema identificado                    │
│                                              │
│  Trade-off: Eficiência vs. Robustez         │
│                                              │
└──────────────────────────────────────────────┘
```

---

**Tags:** #statistics #parametric #nonparametric #modeling #philosophy #methodology

**Voltar para:** [[INDEX]]  
**Fundamento:** [[Distribuições_de_Probabilidade]]  
**Aplicação:** [[The_Balanced_Accuracy_and_Its_Posterior_Distribution]]


