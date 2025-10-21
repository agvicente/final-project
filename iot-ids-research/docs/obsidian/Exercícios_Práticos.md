# Exercícios Práticos

> **Objetivo:** Fixar conceitos através de problemas práticos  
> **Contexto:** [[Aplicação_ao_IoT_IDS|Sistema IDS para IoT]]  
> **Dificuldade:** ⭐ Básico → ⭐⭐⭐⭐⭐ Avançado

---

## 🎯 Como Usar Este Documento

1. **Leia o conceito** relacionado antes de fazer os exercícios
2. **Tente resolver** sem olhar a solução
3. **Compare** sua resposta com a solução fornecida
4. **Implemente** em código Python quando indicado

**Dica:** Use Jupyter Notebook para praticar!

---

## 📊 Seção 1: Estatística Descritiva

### Exercício 1.1: [[Média_Desvio_Padrão_Erro_Padrão|Média e Desvio Padrão]] ⭐

Um modelo IDS foi avaliado em 5 folds com as seguintes acurácias:
```
[0.89, 0.91, 0.88, 0.92, 0.90]
```

**Calcule:**
a) Média  
b) Desvio padrão (amostral, use n-1)  
c) Erro padrão

<details>
<summary>💡 Solução</summary>

**a) Média:**
```
x̄ = (0.89 + 0.91 + 0.88 + 0.92 + 0.90) / 5
  = 4.50 / 5
  = 0.90 = 90%
```

**b) Desvio padrão:**
```
Diferenças: [0.89-0.90, 0.91-0.90, 0.88-0.90, 0.92-0.90, 0.90-0.90]
          = [-0.01, 0.01, -0.02, 0.02, 0.00]

Quadrados: [0.0001, 0.0001, 0.0004, 0.0004, 0.0000]

Soma: 0.0010

Variância: 0.0010 / (5-1) = 0.00025

Desvio padrão: √0.00025 ≈ 0.0158 = 1.58%
```

**c) Erro padrão:**
```
SE = s / √n
   = 0.0158 / √5
   = 0.0158 / 2.236
   ≈ 0.0071 = 0.71%
```

**Código Python:**
```python
import numpy as np

acuracias = [0.89, 0.91, 0.88, 0.92, 0.90]

media = np.mean(acuracias)
desvio = np.std(acuracias, ddof=1)  # ddof=1 para amostral
erro_padrao = desvio / np.sqrt(len(acuracias))

print(f"Média: {media:.2%}")
print(f"Desvio Padrão: {desvio:.2%}")
print(f"Erro Padrão: {erro_padrao:.2%}")
```

</details>

---

### Exercício 1.2: Interpretação ⭐⭐

Dois modelos IDS foram avaliados:

**Modelo A:** Média = 0.90, Desvio Padrão = 0.02  
**Modelo B:** Média = 0.90, Desvio Padrão = 0.05

**Perguntas:**
a) Qual modelo é mais consistente?  
b) Se você pudesse testar apenas uma vez, qual modelo escolheria?  
c) O que o maior desvio padrão do Modelo B pode indicar?

<details>
<summary>💡 Solução</summary>

**a)** Modelo A é mais consistente (menor desvio padrão = menos variação).

**b)** Modelo A - maior probabilidade de resultado próximo à média.

**c)** Possibilidades:
- Desempenho varia muito entre folds (pode ser problema de overfitting)
- Sensível à seleção de dados
- Menos estável para deployment

</details>

---

## 🎲 Seção 2: [[Distribuição_Beta]]

### Exercício 2.1: Parâmetros da Beta ⭐⭐

Seu IDS classificou 100 conexões: 85 acertos, 15 erros.

**Perguntas:**
a) Qual a distribuição posterior com prior não-informativo?  
b) Qual a média desta distribuição?  
c) Qual a moda?

<details>
<summary>💡 Solução</summary>

**a) Distribuição posterior:**
```
Prior: Beta(1, 1)
Dados: 85 acertos, 15 erros

Posterior: Beta(1+85, 1+15) = Beta(86, 16)
```

**b) Média:**
```
μ = α / (α + β)
  = 86 / (86 + 16)
  = 86 / 102
  ≈ 0.843 = 84.3%
```

**c) Moda:**
```
Moda = (α - 1) / (α + β - 2)
     = (86 - 1) / (86 + 16 - 2)
     = 85 / 100
     = 0.85 = 85%
```

**Código Python:**
```python
from scipy import stats

# Dados
corretos = 85
incorretos = 15

# Posterior
posterior = stats.beta(corretos + 1, incorretos + 1)

print(f"Distribuição: Beta({corretos+1}, {incorretos+1})")
print(f"Média: {posterior.mean():.3f}")
print(f"Moda: {corretos/(corretos+incorretos):.3f}")
print(f"Mediana: {posterior.median():.3f}")
```

</details>

---

### Exercício 2.2: Visualização ⭐⭐⭐

Usando o exercício anterior, visualize a distribuição posterior e responda:

a) P(acurácia > 0.80)?  
b) P(acurácia > 0.85)?  
c) Intervalo de credibilidade 95%?

<details>
<summary>💡 Solução</summary>

```python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# Posterior
posterior = stats.beta(86, 16)

# Probabilidades
prob_above_80 = 1 - posterior.cdf(0.80)
prob_above_85 = 1 - posterior.cdf(0.85)
ci_95 = posterior.interval(0.95)

print(f"P(acc > 0.80) = {prob_above_80:.3f}")
print(f"P(acc > 0.85) = {prob_above_85:.3f}")
print(f"IC 95% = [{ci_95[0]:.3f}, {ci_95[1]:.3f}]")

# Visualização
x = np.linspace(0.7, 1.0, 1000)
pdf = posterior.pdf(x)

plt.figure(figsize=(10, 6))
plt.plot(x, pdf, 'b-', linewidth=2, label='Posterior Beta(86, 16)')
plt.fill_between(x, pdf, alpha=0.3)

# Marcar média
mean = posterior.mean()
plt.axvline(mean, color='r', linestyle='--', label=f'Média: {mean:.3f}')

# Marcar IC 95%
plt.axvline(ci_95[0], color='g', linestyle=':', alpha=0.7, label=f'IC 95%')
plt.axvline(ci_95[1], color='g', linestyle=':', alpha=0.7)

plt.xlabel('Acurácia', fontsize=12)
plt.ylabel('Densidade', fontsize=12)
plt.title('Distribuição Posterior da Acurácia\n85 acertos em 100 tentativas', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

**Resultado esperado:**
- P(acc > 0.80) ≈ 0.989 (muito provável!)
- P(acc > 0.85) ≈ 0.462 (não tão certo)
- IC 95% ≈ [0.772, 0.900]

</details>

---

## 📈 Seção 3: [[Acurácia_Balanceada]]

### Exercício 3.1: Cálculo Básico ⭐

Matriz de confusão do seu IDS:

```
                Predito
                Ataque  Normal
Ataque    [     45        5    ]
Real
Normal    [     10      940    ]
```

**Calcule:**
a) Acurácia tradicional  
b) Sensibilidade (recall em ataques)  
c) Especificidade (recall em normais)  
d) Balanced Accuracy

<details>
<summary>💡 Solução</summary>

**Extrair valores:**
```
TP = 45  (ataques detectados)
FN = 5   (ataques perdidos)
FP = 10  (falsos alarmes)
TN = 940 (normais corretos)

Total = 1000
```

**a) Acurácia:**
```
Acc = (TP + TN) / Total
    = (45 + 940) / 1000
    = 0.985 = 98.5%
```

**b) Sensibilidade:**
```
Sens = TP / (TP + FN)
     = 45 / (45 + 5)
     = 45 / 50
     = 0.90 = 90%
```

**c) Especificidade:**
```
Spec = TN / (TN + FP)
     = 940 / (940 + 10)
     = 940 / 950
     ≈ 0.989 = 98.9%
```

**d) Balanced Accuracy:**
```
BA = ½ × (Sens + Spec)
   = ½ × (0.90 + 0.989)
   = ½ × 1.889
   ≈ 0.945 = 94.5%
```

**Código Python:**
```python
import numpy as np
from sklearn.metrics import balanced_accuracy_score

# Dados
TP, FN, FP, TN = 45, 5, 10, 940

# Acurácia
acc = (TP + TN) / (TP + TN + FP + FN)

# Sensibilidade e Especificidade
sens = TP / (TP + FN)
spec = TN / (TN + FP)

# Balanced Accuracy
ba = 0.5 * (sens + spec)

print(f"Acurácia: {acc:.3%}")
print(f"Sensibilidade: {sens:.3%}")
print(f"Especificidade: {spec:.3%}")
print(f"Balanced Accuracy: {ba:.3%}")
```

</details>

---

### Exercício 3.2: Detector Enviesado ⭐⭐⭐

Um modelo "preguiçoso" SEMPRE prevê "Normal":

Dataset: 50 ataques, 950 normais

**Calcule:**
a) Matriz de confusão  
b) Acurácia  
c) Balanced Accuracy  
d) Por que BA é melhor neste caso?

<details>
<summary>💡 Solução</summary>

**a) Matriz:**
```
TP = 0   (nenhum ataque detectado)
FN = 50  (todos os ataques perdidos)
FP = 0   (nunca prevê ataque)
TN = 950 (todos os normais corretos)
```

**b) Acurácia:**
```
Acc = 950 / 1000 = 95%  ← Parece ótimo! (mas é enganoso)
```

**c) Balanced Accuracy:**
```
Sens = 0 / 50 = 0%
Spec = 950 / 950 = 100%

BA = ½ × (0 + 1) = 0.5 = 50%  ← Nível do acaso!
```

**d) Por que BA é melhor:**
BA revela que o modelo não aprendeu nada útil! É equivalente a um classificador aleatório para [[Aplicação_ao_IoT_IDS|IDS]] - completamente inútil.

</details>

---

## 🔵 Seção 4: [[Inferência_Bayesiana]]

### Exercício 4.1: Atualização Bayesiana ⭐⭐⭐

Você começa com prior não-informativo Beta(1,1) e observa dados sequencialmente:

```
Teste 1: Acerto
Teste 2: Acerto  
Teste 3: Erro
Teste 4: Acerto
Teste 5: Acerto
```

**Para cada etapa, calcule:**
- Posterior
- Média posterior

<details>
<summary>💡 Solução</summary>

```python
from scipy import stats

# Começar com prior
alpha, beta = 1, 1

observacoes = [1, 1, 0, 1, 1]  # 1=acerto, 0=erro

print("Atualização Bayesiana Sequencial:")
print("="*50)

for i, obs in enumerate(observacoes, 1):
    # Atualizar
    if obs == 1:
        alpha += 1
    else:
        beta += 1
    
    # Posterior atual
    posterior = stats.beta(alpha, beta)
    mean = posterior.mean()
    
    print(f"Após teste {i}: Beta({alpha}, {beta}), Média = {mean:.3f}")

# Resultado final
print("\n" + "="*50)
print(f"Posterior final: Beta({alpha}, {beta})")
print(f"Média final: {posterior.mean():.3f}")
print(f"IC 95%: {posterior.interval(0.95)}")
```

**Resultado:**
```
Após teste 1: Beta(2, 1), Média = 0.667
Após teste 2: Beta(3, 1), Média = 0.750
Após teste 3: Beta(3, 2), Média = 0.600
Após teste 4: Beta(4, 2), Média = 0.667
Após teste 5: Beta(5, 2), Média = 0.714

Posterior final: Beta(5, 2)
Média final: 0.714
IC 95%: (0.362, 0.945)
```

</details>

---

### Exercício 4.2: Prior Informativo ⭐⭐⭐⭐

Você sabe que sistemas IDS similares têm acurácia ~85%. Modele isso como Beta(17, 3) (média = 17/20 = 0.85).

Você testa seu sistema: 90 acertos, 10 erros.

**Compare:**
a) Posterior com prior não-informativo  
b) Posterior com prior informativo  
c) Como o prior afetou o resultado?

<details>
<summary>💡 Solução</summary>

```python
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

# Dados
k, n = 90, 10  # acertos, erros

# Caso 1: Prior não-informativo
prior_uninf = stats.beta(1, 1)
post_uninf = stats.beta(1 + k, 1 + n)

# Caso 2: Prior informativo
prior_inf = stats.beta(17, 3)
post_inf = stats.beta(17 + k, 3 + n)

# Comparar
print("PRIOR NÃO-INFORMATIVO:")
print(f"  Prior: Beta(1, 1)")
print(f"  Posterior: Beta({1+k}, {1+n})")
print(f"  Média: {post_uninf.mean():.3f}")
print(f"  IC 95%: {post_uninf.interval(0.95)}")

print("\nPRIOR INFORMATIVO:")
print(f"  Prior: Beta(17, 3), média={prior_inf.mean():.3f}")
print(f"  Posterior: Beta({17+k}, {3+n})")
print(f"  Média: {post_inf.mean():.3f}")
print(f"  IC 95%: {post_inf.interval(0.95)}")

# Visualização
x = np.linspace(0.7, 1.0, 1000)

plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.plot(x, prior_uninf.pdf(x), 'gray', linestyle='--', label='Prior')
plt.plot(x, post_uninf.pdf(x), 'b-', linewidth=2, label='Posterior')
plt.title('Prior Não-Informativo')
plt.xlabel('Acurácia')
plt.ylabel('Densidade')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(122)
plt.plot(x, prior_inf.pdf(x), 'gray', linestyle='--', label='Prior')
plt.plot(x, post_inf.pdf(x), 'r-', linewidth=2, label='Posterior')
plt.title('Prior Informativo')
plt.xlabel('Acurácia')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

**c) Impacto do prior:**
Com 100 dados, ambos convergem para região similar, mas:
- Prior informativo "puxa" ligeiramente para 0.85
- Diferença é pequena (dados dominam)
- Com menos dados, prior teria mais influência

</details>

---

## 🎯 Seção 5: [[Intervalos_de_Confiança]]

### Exercício 5.1: Problema do Intervalo >100% ⭐⭐

Dados: 98 acertos em 100 testes.

**Calcule:**
a) IC 95% usando aproximação Normal  
b) IC 95% usando Beta  
c) Qual está errado e por quê?

<details>
<summary>💡 Solução</summary>

```python
from scipy import stats
import numpy as np

# Dados
k, n = 98, 2  # corretos, incorretos
p_hat = 0.98

# a) Aproximação Normal
se = np.sqrt(p_hat * (1-p_hat) / (k+n))
ci_normal = [p_hat - 1.96*se, p_hat + 1.96*se]

print("APROXIMAÇÃO NORMAL:")
print(f"  p̂ = {p_hat:.3f}")
print(f"  SE = {se:.4f}")
print(f"  IC 95% = [{ci_normal[0]:.3f}, {ci_normal[1]:.3f}]")
if ci_normal[1] > 1.0:
    print(f"  ❌ PROBLEMA: Limite superior > 1.0!")

# b) Beta
posterior = stats.beta(k + 1, n + 1)
ci_beta = posterior.interval(0.95)

print("\nDISTRIBUIÇÃO BETA:")
print(f"  Posterior: Beta({k+1}, {n+1})")
print(f"  IC 95% = [{ci_beta[0]:.3f}, {ci_beta[1]:.3f}]")
print(f"  ✅ Sempre respeita [0, 1]")
```

**c) Qual está errado:**
A aproximação Normal viola o limite [0,1]. Beta é a abordagem correta para proporções!

</details>

---

## 🚀 Seção 6: Projeto Completo [[Aplicação_ao_IoT_IDS]]

### Exercício 6.1: Pipeline Completo ⭐⭐⭐⭐⭐

**Desafio:** Implemente avaliação Bayesiana completa para comparar 3 algoritmos.

**Dados simulados:**
```python
# Resultados de 3 algoritmos no CICIoT2023
algorithms = {
    'Random Forest': {'TP': 450, 'FN': 50, 'TN': 9300, 'FP': 200},
    'XGBoost': {'TP': 460, 'FN': 40, 'TN': 9250, 'FP': 250},
    'MoT': {'TP': 440, 'FN': 60, 'TN': 9350, 'FP': 150}
}
```

**Implemente:**
1. Classe para calcular métricas Bayesianas
2. Comparação probabilística entre algoritmos
3. Visualização das distribuições posteriores
4. Recomendação do melhor algoritmo

<details>
<summary>💡 Esqueleto do Código</summary>

```python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

class BayesianEvaluator:
    """Avaliação Bayesiana de modelos IDS."""
    
    def __init__(self, confusion_matrix):
        """
        Args:
            confusion_matrix: dict com TP, FN, TN, FP
        """
        self.cm = confusion_matrix
        self.TP = confusion_matrix['TP']
        self.FN = confusion_matrix['FN']
        self.TN = confusion_matrix['TN']
        self.FP = confusion_matrix['FP']
    
    def balanced_accuracy_posterior(self, n_samples=100000):
        """Distribuição posterior da BA."""
        # Posteriors por classe
        pos_post = stats.beta(self.TP + 1, self.FN + 1)
        neg_post = stats.beta(self.TN + 1, self.FP + 1)
        
        # Amostragem da convolução
        pos_samples = pos_post.rvs(n_samples)
        neg_samples = neg_post.rvs(n_samples)
        ba_samples = 0.5 * (pos_samples + neg_samples)
        
        return ba_samples
    
    def report(self):
        """Relatório completo."""
        ba_samples = self.balanced_accuracy_posterior()
        
        return {
            'mean': np.mean(ba_samples),
            'median': np.median(ba_samples),
            'std': np.std(ba_samples),
            'ci_95': np.percentile(ba_samples, [2.5, 97.5]),
            'samples': ba_samples
        }

# Avaliar algoritmos
algorithms = {
    'Random Forest': {'TP': 450, 'FN': 50, 'TN': 9300, 'FP': 200},
    'XGBoost': {'TP': 460, 'FN': 40, 'TN': 9250, 'FP': 250},
    'MoT': {'TP': 440, 'FN': 60, 'TN': 9350, 'FP': 150}
}

results = {}
for name, cm in algorithms.items():
    evaluator = BayesianEvaluator(cm)
    results[name] = evaluator.report()
    
    print(f"{name}:")
    print(f"  BA média: {results[name]['mean']:.3f}")
    print(f"  IC 95%: {results[name]['ci_95']}")
    print()

# Comparações probabilísticas
print("COMPARAÇÕES:")
for alg1 in results:
    for alg2 in results:
        if alg1 < alg2:
            prob = np.mean(results[alg1]['samples'] > results[alg2]['samples'])
            print(f"P({alg1} > {alg2}) = {prob:.3f}")

# Visualização
fig, ax = plt.subplots(figsize=(12, 6))

for name, result in results.items():
    ax.hist(result['samples'], bins=50, alpha=0.5, label=name, density=True)

ax.set_xlabel('Balanced Accuracy')
ax.set_ylabel('Densidade')
ax.set_title('Distribuições Posteriores - Comparação de Algoritmos IDS')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('algorithm_comparison.png', dpi=300)
plt.show()

# COMPLETE VOCÊ: Adicione mais análises!
```

</details>

---

## ✅ Gabarito Rápido

### Respostas dos Exercícios Principais

**1.1:** média=0.90, desvio=0.0158, SE=0.0071  
**2.1:** Beta(86,16), média=0.843, moda=0.85  
**3.1:** Acc=98.5%, BA=94.5%  
**3.2:** Acc=95%, BA=50% (modelo inútil!)  
**5.1:** Normal viola [0,1], Beta é correto

---

## 📚 Próximos Passos

Depois de completar estes exercícios:

1. Aplique ao seu dataset CICIoT2023 real
2. Integre com seu pipeline DVC
3. Documente no seu projeto
4. Use nas comparações de algoritmos

---

## 🔗 Conceitos Relacionados

- [[Acurácia]]
- [[Acurácia_Balanceada]]
- [[Distribuição_Beta]]
- [[Inferência_Bayesiana]]
- [[Intervalos_de_Confiança]]
- [[Média_Desvio_Padrão_Erro_Padrão]]
- [[The_Balanced_Accuracy_and_Its_Posterior_Distribution]]
- [[Aplicação_ao_IoT_IDS]]
- [[Referências_Bibliográficas]]

---

**Tags:** #exercises #practice #hands-on #python #IDS #balanced-accuracy #bayesian

**Voltar para:** [[INDEX]]


