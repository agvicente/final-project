# Acurácia (Accuracy)

> **Tipo:** Métrica de Avaliação  
> **Complexidade:** ⭐☆☆☆☆ (Básico)  
> **Aplicação:** Classificação em Machine Learning

---

## 🎯 Definição

**Acurácia** é a métrica mais básica e intuitiva para avaliar o desempenho de um modelo de classificação.

**Pergunta que responde:**
> "De todas as previsões que o modelo fez, quantas estavam corretas?"

---

## 📐 Fórmula Matemática

### Definição Simples
```
Acurácia = Número de Previsões Corretas / Número Total de Previsões
```

### Definição Formal (Classificação Binária)

```
Acurácia = (TP + TN) / (TP + TN + FP + FN)
```

Onde:
- **TP (True Positives):** Verdadeiros Positivos - acertou positivo
- **TN (True Negatives):** Verdadeiros Negativos - acertou negativo
- **FP (False Positives):** Falsos Positivos - errou, previu positivo
- **FN (False Negatives):** Falsos Negativos - errou, previu negativo

### Matriz de Confusão

```
                    Predito
                    +       -
           +     [ TP  |  FN ]
Real       
           -     [ FP  |  TN ]

Acurácia = (TP + TN) / Total
```

---

## 💡 Exemplo Prático - [[Aplicação_ao_IoT_IDS|Sistema IDS para IoT]]

### Cenário: Detector de Intrusão

Seu modelo analisa **100 conexões de rede IoT**:

**Dados reais:**
- 90 conexões normais (legítimas)
- 10 ataques (maliciosos)

**Predições do modelo:**
- 85 conexões normais identificadas corretamente (TN)
- 5 conexões normais marcadas como ataque (FP - falsos alarmes)
- 8 ataques detectados corretamente (TP)
- 2 ataques que passaram despercebidos (FN - perigoso!)

**Cálculo:**
```
Acurácia = (TP + TN) / Total
         = (8 + 85) / 100
         = 93 / 100
         = 0.93 = 93%
```

**Interpretação:** O modelo acertou 93% das classificações.

---

## ⚠️ Limitações Críticas

### Problema 1: Datasets Desbalanceados

**Cenário problemático:**

Dataset com 95 conexões normais e 5 ataques.

**Modelo "preguiçoso"** que sempre prevê "normal":

```
Predições: [Normal, Normal, Normal, ..., Normal]
           (100 vezes)

Resultados:
- TP = 0 (nenhum ataque detectado!)
- TN = 95 (todas as normais corretas)
- FP = 0
- FN = 5 (todos os ataques perdidos!)

Acurácia = (0 + 95) / 100 = 95% ✓ Parece excelente!
```

**MAS:** O modelo **NÃO DETECTA NENHUM ATAQUE!** É completamente inútil para IDS! 🚨

### Problema 2: Custos Desiguais de Erros

Em [[Aplicação_ao_IoT_IDS|sistemas IDS]]:
- **FN (ataque não detectado):** 💀 Muito perigoso! Sistema comprometido!
- **FP (falso alarme):** 😐 Irritante, mas não crítico

A acurácia **trata ambos os erros igualmente**, não refletindo o impacto real.

### Problema 3: [[Intervalos_de_Confiança|Intervalos de Confiança]] Inadequados

Reportar apenas "93% de acurácia" não diz nada sobre:
- Quão confiável é essa estimativa?
- Qual a margem de erro?
- O modelo é realmente melhor que outro com 91%?

Veja [[The_Balanced_Accuracy_and_Its_Posterior_Distribution#Problema 1|problema detalhado no artigo]].

---

## ✅ Quando Usar Acurácia?

Acurácia é apropriada quando:

1. **✅ Classes balanceadas:** Aproximadamente mesmo número de exemplos de cada classe
2. **✅ Custos iguais:** Errar em qualquer direção tem o mesmo impacto
3. **✅ Contexto geral:** Você quer uma visão geral simplificada

**Exemplo adequado:** Classificar gatos vs. cachorros em dataset balanceado.

---

## 🚫 Quando NÃO Usar Acurácia?

Evite acurácia quando:

1. **❌ Classes desbalanceadas:** Uma classe muito mais frequente que outra
2. **❌ Custos assimétricos:** Um tipo de erro é mais grave
3. **❌ Contextos críticos:** Saúde, segurança, finanças

**Exemplo inadequado:** Detecção de câncer (poucos casos positivos, FN é fatal).

---

## 🔄 Alternativas e Complementos

### Para Datasets Desbalanceados

Use [[Acurácia_Balanceada]]:
```
Balanced Accuracy = ½(TP/P + TN/N)
```

**Vantagem:** Não é enganada por desbalanceamento!

### Para Foco em Uma Classe

**Precision (Precisão):**
```
Precision = TP / (TP + FP)
```
*"Das previsões positivas, quantas estavam corretas?"*

**Recall/Sensitivity (Sensibilidade):**
```
Recall = TP / (TP + FN)
```
*"Dos positivos reais, quantos foram detectados?"*

**F1-Score (Média Harmônica):**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### Para Análise Completa

- **ROC-AUC:** Curva ROC e área sob a curva
- **Precision-Recall Curve:** Especialmente para datasets desbalanceados
- **Cohen's Kappa:** Acurácia ajustada por concordância ao acaso

---

## 📊 Modelagem Probabilística da Acurácia

### Abordagem Bayesiana (do [[The_Balanced_Accuracy_and_Its_Posterior_Distribution|artigo principal]])

Ao invés de reportar apenas um ponto (ex: 93%), modelar a **distribuição completa** da acurácia.

**Framework:**
```
Dados: C corretos, I incorretos
Prior: Beta(1, 1) - uniforme
Posterior: A ~ Beta(C+1, I+1)
```

Veja [[Distribuição_Beta]] para detalhes.

**Exemplo:**
```python
from scipy import stats

# 93 corretos, 7 incorretos
C, I = 93, 7
posterior = stats.beta(C+1, I+1)

# Estatísticas
print(f"Média: {posterior.mean():.3f}")
print(f"Mediana: {posterior.median():.3f}")
print(f"IC 95%: {posterior.interval(0.95)}")

# Resultado:
# Média: 0.931
# Mediana: 0.932
# IC 95%: (0.873, 0.971)
```

Veja [[Inferência_Bayesiana]] para o paradigma completo.

---

## 🧮 Implementação Prática

### Python (NumPy)

```python
import numpy as np

def accuracy(y_true, y_pred):
    """Calcula acurácia."""
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total

# Exemplo IoT-IDS
y_true = np.array([0,0,0,1,1,0,1,0,0,1])  # 0=normal, 1=ataque
y_pred = np.array([0,0,1,1,1,0,1,0,0,0])  # predições

acc = accuracy(y_true, y_pred)
print(f"Acurácia: {acc:.2%}")  # 70%
```

### Python (Scikit-learn)

```python
from sklearn.metrics import accuracy_score, confusion_matrix

# Dados
y_true = [0,0,0,1,1,0,1,0,0,1]
y_pred = [0,0,1,1,1,0,1,0,0,0]

# Acurácia
acc = accuracy_score(y_true, y_pred)
print(f"Acurácia: {acc:.2%}")

# Matriz de confusão
cm = confusion_matrix(y_true, y_pred)
print("Matriz de Confusão:")
print(cm)
```

### Com Posterior Beta

```python
from scipy import stats
import numpy as np

def accuracy_with_posterior(y_true, y_pred, confidence=0.95):
    """Calcula acurácia com intervalo de credibilidade."""
    correct = np.sum(y_true == y_pred)
    incorrect = len(y_true) - correct
    
    # Posterior Beta
    posterior = stats.beta(correct + 1, incorrect + 1)
    
    # Estatísticas
    mean = posterior.mean()
    median = posterior.median()
    mode = correct / len(y_true)  # MLE
    ci = posterior.interval(confidence)
    
    return {
        'accuracy': mean,
        'median': median,
        'mode': mode,
        'ci': ci,
        'posterior': posterior
    }

# Exemplo
y_true = np.random.randint(0, 2, 100)
y_pred = np.random.randint(0, 2, 100)

result = accuracy_with_posterior(y_true, y_pred)
print(f"Acurácia: {result['accuracy']:.3f}")
print(f"IC 95%: [{result['ci'][0]:.3f}, {result['ci'][1]:.3f}]")
```

---

## 📚 Referências

### Livros
- **Alpaydin, E.** (2020). *Introduction to Machine Learning* (4th ed.). MIT Press. [Capítulo sobre avaliação]
- **Bishop, C.M.** (2006). *Pattern Recognition and Machine Learning*. Springer. [Seção 1.5.4]
- **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning*. Springer. [Capítulo 7]

### Papers
- **Brodersen et al.** (2010). "The balanced accuracy and its posterior distribution". ICPR. [[The_Balanced_Accuracy_and_Its_Posterior_Distribution]]
- **Sokolova, M. & Lapalme, G.** (2009). "A systematic analysis of performance measures for classification tasks". *Information Processing & Management*, 45(4), 427-437.

### Online
- [Scikit-learn: Accuracy Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
- [Confusion Matrix Visualization](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html)

Veja [[Referências_Bibliográficas]] para lista completa.

---

## 🔗 Conceitos Relacionados

### Fundamentos
- [[Média_Desvio_Padrão_Erro_Padrão]] - Como agregar acurácias de vários folds
- [[Intervalos_de_Confiança]] - Incerteza sobre a estimativa

### Alternativas Superiores
- [[Acurácia_Balanceada]] - Solução para desbalanceamento
- [[The_Balanced_Accuracy_and_Its_Posterior_Distribution]] - Artigo principal

### Modelagem Probabilística
- [[Distribuição_Beta]] - Modelo para proporções
- [[Inferência_Bayesiana]] - Paradigma probabilístico

### Aplicação
- [[Aplicação_ao_IoT_IDS]] - Uso em seu projeto de pesquisa

---

## 🎯 Exercícios

Veja [[Exercícios_Práticos#Acurácia]] para problemas práticos.

---

## 📌 Resumo Visual

```
┌─────────────────────────────────────────┐
│           ACURÁCIA                      │
│                                         │
│  "Proporção de acertos totais"         │
│                                         │
│  Fórmula: (TP + TN) / Total            │
│                                         │
│  ✅ Vantagens:                          │
│     • Simples e intuitiva              │
│     • Fácil de explicar                │
│     • Boa para classes balanceadas     │
│                                         │
│  ❌ Limitações:                         │
│     • Enganosa em datasets desbalanceados │
│     • Ignora custos de erros diferentes │
│     • Não quantifica incerteza         │
│                                         │
│  ➡️ Solução: Balanced Accuracy         │
│                                         │
└─────────────────────────────────────────┘
```

---

**Tags:** #metrics #accuracy #classification #machine-learning #evaluation #basic

**Voltar para:** [[INDEX]]  
**Próximo conceito:** [[Acurácia_Balanceada]]


