# Acurácia Balanceada (Balanced Accuracy)

> **Tipo:** Métrica de Avaliação  
> **Complexidade:** ⭐⭐☆☆☆ (Intermediário)  
> **Aplicação:** Classificação com Datasets Desbalanceados

---

## 🎯 Definição

**Acurácia Balanceada** é a média aritmética das acurácias obtidas em **cada classe individualmente**, resolvendo o problema de [[Acurácia|acurácia tradicional]] em datasets desbalanceados.

**Pergunta que responde:**
> "Qual a média do meu desempenho em cada classe?"

**Proposta por:** [[The_Balanced_Accuracy_and_Its_Posterior_Distribution|Brodersen et al. (2010)]]

---

## 📐 Fórmula Matemática

### Classificação Binária

```
Balanced Accuracy = ½ × (Sensibilidade + Especificidade)

                  = ½ × (TP/P + TN/N)
                  
                  = ½ × (TP/(TP+FN) + TN/(TN+FP))
```

Onde:
- **TP/P:** Taxa de acerto na classe **positiva** (Sensibilidade/Recall)
- **TN/N:** Taxa de acerto na classe **negativa** (Especificidade)
- **P = TP + FN:** Total de positivos reais
- **N = TN + FP:** Total de negativos reais

### Forma Geral (Multiclasse)

```
Balanced Accuracy = (1/K) × Σᵢ (Acurácia na classe i)
```

Onde K é o número de classes.

### Com Custos Personalizados

```
Balanced Accuracy = c × (TP/P) + (1-c) × (TN/N)
```

Onde c ∈ [0, 1] é o custo/peso da classe positiva.

---

## 💡 Intuição: Por Que Funciona?

### O Problema com [[Acurácia]] Tradicional

**Cenário:** Dataset com 95 normais e 5 ataques.

**Modelo "preguiçoso"** prevê tudo como "normal":

```
┌──────────────────────────────────────┐
│  Classe Normal (95 exemplos):       │
│  Acertos: 95/95 = 100% ✅           │
│                                      │
│  Classe Ataque (5 exemplos):        │
│  Acertos: 0/5 = 0% ❌               │
└──────────────────────────────────────┘

Acurácia Tradicional:
(95 + 0) / 100 = 95% ← Parece bom! 🎉

Acurácia Balanceada:
½(100% + 0%) = 50% ← Nível do acaso! ⚠️
```

**A acurácia balanceada REVELA a verdade:** o modelo não aprendeu nada útil!

### Analogia do IDS

Imagine um guarda de segurança que nunca aciona o alarme:
- **Estatística enganosa:** "99% de precisão" (quase nunca há intrusos)
- **Realidade:** Inútil! Todos os intrusos passam!

A acurácia balanceada detecta isso porque avalia **separadamente**:
- Quão bem detecta situações normais?
- Quão bem detecta intrusos?

---

## 🧮 Exemplo Passo a Passo - [[Aplicação_ao_IoT_IDS|Sistema IDS]]

### Cenário Realista: CICIoT2023

**Dataset de teste:**
- 100 conexões normais
- 10 ataques

**Modelo detecta:**
- 95 conexões normais corretamente (TN = 95)
- 5 falsos alarmes (FP = 5)
- 8 ataques corretamente (TP = 8)
- 2 ataques perdidos (FN = 2)

### Matriz de Confusão

```
                 Predito
                 Ataque  Normal
           ┌─────────────────────┐
  Ataque   │   8 (TP) │  2 (FN) │  P = 10
  Real     ├──────────┼─────────┤
  Normal   │   5 (FP) │ 95 (TN) │  N = 100
           └─────────────────────┘
```

### Passo 1: Acurácia na Classe Positiva (Ataques)

```
Sensibilidade = TP / (TP + FN)
              = 8 / (8 + 2)
              = 8 / 10
              = 0.80 = 80%
```

**Interpretação:** O modelo detecta **80% dos ataques**.

### Passo 2: Acurácia na Classe Negativa (Normais)

```
Especificidade = TN / (TN + FP)
               = 95 / (95 + 5)
               = 95 / 100
               = 0.95 = 95%
```

**Interpretação:** O modelo classifica corretamente **95% do tráfego normal**.

### Passo 3: Média das Duas

```
Balanced Accuracy = ½ × (80% + 95%)
                  = ½ × 175%
                  = 87.5%
```

### Comparação

```
Acurácia Tradicional = (TP + TN) / Total
                     = (8 + 95) / 110
                     = 103 / 110
                     = 93.6% ✓

Acurácia Balanceada = 87.5% ✓
```

**Qual é mais informativa?**
- Acurácia tradicional: dominada pela classe majoritária (normal)
- Acurácia balanceada: reflete desempenho **equilibrado** em ambas as classes

---

## ✅ Vantagens

### 1. Robusta a Desbalanceamento

Dá **peso igual** a cada classe, independente da quantidade de exemplos.

```
Dataset 1: 50/50 split
Dataset 2: 99/1 split

→ Ambos usam a mesma fórmula!
→ Resultados comparáveis!
```

### 2. Detecta Classificadores Enviesados

Modelos que "trapaceiam" explorando desbalanceamento são penalizados:

```
Modelo que prevê sempre "majoritária":
→ 100% em uma classe
→ 0% na outra
→ BA = 50% (nível do acaso)
```

### 3. Interpretação Clara

**Significa:** "Média de performance nas classes"
- BA = 50%: Nível do acaso
- BA = 100%: Perfeito em ambas
- BA entre 50-100%: Performance real

### 4. Fácil de Comparar Algoritmos

```
Algoritmo A: BA = 85%
Algoritmo B: BA = 78%

→ A é melhor, considerando ambas as classes!
```

---

## ⚠️ Limitações

### 1. Assume Classes Igualmente Importantes

Se detectar ataques é **3x mais importante** que não dar falso alarme, BA simples não reflete isso.

**Solução:** Usar versão ponderada com custos.

### 2. Informação Agregada

BA é um número único, perde detalhes:
- Qual classe tem pior performance?
- Quantos FP vs FN?

**Solução:** Reportar BA **junto com** matriz de confusão e métricas individuais.

### 3. Não Considera Distribuição Real

Em produção, classes podem ter proporção diferente do treino.

**Exemplo:**
- Treino: 50/50
- Produção: 99/1
- BA otimiza para 50/50, pode não ser ideal para 99/1

### 4. Multiclasse Complexo

Com K classes, todas recebem peso 1/K. Pode não ser desejável.

---

## 📊 Distribuição Posterior da Balanced Accuracy

### Abordagem Bayesiana (Contribuição do [[The_Balanced_Accuracy_and_Its_Posterior_Distribution|Artigo]])

Ao invés de reportar apenas um ponto, modelar a **distribuição completa**.

### Matemática (Classificação Binária)

**Acurácia em cada classe:**
```
A_pos ~ Beta(TP+1, FN+1)
A_neg ~ Beta(TN+1, FP+1)
```

**Balanced Accuracy:**
```
BA = ½(A_pos + A_neg)
```

**Distribuição de BA:** Convolução de duas Betas!

```
p_BA(x; TP, FP, FN, TN) = ∫₀¹ p_A(2(x-z); TP+1, FN+1) · p_A(2z; TN+1, FP+1) dz
```

Veja [[Distribuição_Beta#Convolução|seção de convolução]] para detalhes.

### Propriedades

- Não tem forma fechada analítica
- Requer integração numérica
- Permite calcular intervalos de credibilidade

Detalhes completos em [[Inferência_Bayesiana]].

---

## 🧪 Implementação Prática

### Python (NumPy)

```python
import numpy as np

def balanced_accuracy(y_true, y_pred):
    """
    Calcula balanced accuracy para classificação binária.
    
    Args:
        y_true: labels verdadeiros (0/1)
        y_pred: labels preditos (0/1)
    
    Returns:
        balanced accuracy
    """
    # Positivos e negativos
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    
    # Sensibilidade (recall na classe positiva)
    sensitivity = np.sum(y_pred[pos_mask] == 1) / np.sum(pos_mask)
    
    # Especificidade (recall na classe negativa)
    specificity = np.sum(y_pred[neg_mask] == 0) / np.sum(neg_mask)
    
    # Balanced accuracy
    ba = 0.5 * (sensitivity + specificity)
    
    return ba

# Exemplo IoT-IDS
y_true = np.array([0,0,0,1,1,0,1,0,0,1,0,0,0,1])  # 4 ataques, 10 normais
y_pred = np.array([0,0,1,1,1,0,1,0,0,0,0,0,0,1])  # predições

ba = balanced_accuracy(y_true, y_pred)
print(f"Balanced Accuracy: {ba:.2%}")  # 75%
```

### Python (Scikit-learn)

```python
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

y_true = [0,0,0,1,1,0,1,0,0,1,0,0,0,1]
y_pred = [0,0,1,1,1,0,1,0,0,0,0,0,0,1]

# Balanced accuracy
ba = balanced_accuracy_score(y_true, y_pred)
print(f"Balanced Accuracy: {ba:.2%}")

# Para entender o resultado, veja a matriz
cm = confusion_matrix(y_true, y_pred)
print("\nMatriz de Confusão:")
print(cm)
```

### Com Distribuição Posterior

```python
from scipy import stats
import numpy as np

def balanced_accuracy_posterior(y_true, y_pred, n_samples=10000):
    """
    Calcula distribuição posterior da balanced accuracy.
    
    Usa aproximação por amostragem Monte Carlo da convolução.
    """
    # Extrair TP, TN, FP, FN
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    
    TP = np.sum((y_pred == 1) & pos_mask)
    FN = np.sum((y_pred == 0) & pos_mask)
    TN = np.sum((y_pred == 0) & neg_mask)
    FP = np.sum((y_pred == 1) & neg_mask)
    
    # Posteriors para cada classe
    pos_posterior = stats.beta(TP + 1, FN + 1)
    neg_posterior = stats.beta(TN + 1, FP + 1)
    
    # Amostrar da convolução
    pos_samples = pos_posterior.rvs(n_samples)
    neg_samples = neg_posterior.rvs(n_samples)
    ba_samples = 0.5 * (pos_samples + neg_samples)
    
    return {
        'mean': np.mean(ba_samples),
        'median': np.median(ba_samples),
        'std': np.std(ba_samples),
        'ci_95': np.percentile(ba_samples, [2.5, 97.5]),
        'samples': ba_samples
    }

# Exemplo
y_true = np.random.randint(0, 2, 100)
y_pred = (np.random.rand(100) > 0.3).astype(int)

result = balanced_accuracy_posterior(y_true, y_pred)
print(f"BA média: {result['mean']:.3f}")
print(f"IC 95%: [{result['ci_95'][0]:.3f}, {result['ci_95'][1]:.3f}]")
```

### Multiclasse

```python
from sklearn.metrics import balanced_accuracy_score

# 3 classes: normal, DoS, reconnaissance
y_true = [0,0,0,1,1,2,1,0,0,2,0,0,0,1]
y_pred = [0,0,1,1,1,2,1,0,0,1,0,0,0,1]

ba = balanced_accuracy_score(y_true, y_pred)
print(f"Balanced Accuracy (multiclasse): {ba:.2%}")
```

---

## 📊 Quando Usar Balanced Accuracy?

### ✅ Use Quando:

1. **Dataset desbalanceado** (proporções muito diferentes)
2. **Todas as classes são importantes** (não pode ignorar minoritárias)
3. **Comparar algoritmos** em diferentes datasets
4. **Reportar para academia** (métrica bem estabelecida)

### 🎯 Contextos Ideais:

- [[Aplicação_ao_IoT_IDS|Detecção de intrusão]] (ataques raros)
- Detecção de fraude (transações fraudulentas raras)
- Diagnóstico médico (doenças raras)
- Detecção de anomalias (eventos raros)

### ⚠️ Considere Alternativas Quando:

1. **Classes têm importância diferente** → Use F1-Score ponderado ou métrica customizada
2. **Precisa de threshold flexível** → Use ROC-AUC ou PR-AUC
3. **Foco em uma classe específica** → Use Precision/Recall daquela classe

---

## 🔄 Relação com Outras Métricas

### vs. [[Acurácia]] Tradicional

```
Dataset balanceado: BA ≈ Accuracy
Dataset desbalanceado: BA < Accuracy (geralmente)
```

### vs. F1-Score

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

- **F1:** Foca na classe positiva (média harmônica de precision e recall)
- **BA:** Considera ambas as classes igualmente (média aritmética)

**Quando usar qual?**
- F1: Quando classe positiva é mais importante
- BA: Quando ambas as classes são igualmente importantes

### vs. ROC-AUC

```
ROC-AUC: Área sob curva ROC
```

- **ROC-AUC:** Avalia em todos os thresholds possíveis
- **BA:** Avalia em um threshold específico

**Complementares:** Use ambos!

### Tabela Comparativa

| Métrica | Balanceado | Desbalanceado | Classes Iguais | Threshold |
|---------|-----------|---------------|----------------|-----------|
| [[Acurácia]] | ✅ | ❌ | ✅ | Fixo |
| **BA** | ✅ | ✅ | ✅ | Fixo |
| F1 | ✅ | ✅ | Foco em positiva | Fixo |
| ROC-AUC | ✅ | ✅ | ✅ | Todos |

---

## 📚 Referências

### Paper Original
- **Brodersen, K.H., et al.** (2010). "The balanced accuracy and its posterior distribution". *ICPR*. [[The_Balanced_Accuracy_and_Its_Posterior_Distribution]]

### Trabalhos Relacionados
- **Velez et al.** (2007). "A balanced accuracy function for epistasis modeling in imbalanced datasets". *Genetic Epidemiology*, 31(4), 306-315.
- **Japkowicz, N. & Stephen, S.** (2002). "The class imbalance problem: A systematic study". *Intelligent Data Analysis*, 6(5), 429-449.

### Livros
- **He, H. & Ma, Y.** (eds.) (2013). *Imbalanced Learning: Foundations, Algorithms, and Applications*. Wiley-IEEE Press.
- **Alpaydin, E.** (2020). *Introduction to Machine Learning* (4th ed.). MIT Press.

### Online
- [Scikit-learn: Balanced Accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)
- [Imbalanced-learn Library](https://imbalanced-learn.org/)

Veja [[Referências_Bibliográficas]] para lista completa.

---

## 🔗 Conceitos Relacionados

### Pré-requisitos
- [[Acurácia]] - Métrica básica
- [[Média_Desvio_Padrão_Erro_Padrão]] - Para agregar resultados

### Teoria Avançada
- [[Distribuição_Beta]] - Modelagem probabilística
- [[Inferência_Bayesiana]] - Paradigma do artigo
- [[Intervalos_de_Confiança]] - Quantificação de incerteza

### Contexto
- [[The_Balanced_Accuracy_and_Its_Posterior_Distribution]] - Artigo completo
- [[Aplicação_ao_IoT_IDS]] - Aplicação prática
- [[Métodos_Paramétricos_vs_Não_Paramétricos]] - Abordagens de modelagem

---

## 🎯 Exercícios

Veja [[Exercícios_Práticos#Balanced Accuracy]] para problemas.

---

## 📌 Resumo Visual

```
┌──────────────────────────────────────────────┐
│       BALANCED ACCURACY                      │
│                                              │
│  "Média de acurácia em cada classe"         │
│                                              │
│  Fórmula: ½(TP/P + TN/N)                    │
│                                              │
│  ┌─────────────┐      ┌─────────────┐      │
│  │  Classe +   │      │  Classe -   │      │
│  │  TP/P = 80% │  +   │  TN/N = 95% │  ÷ 2 │
│  └─────────────┘      └─────────────┘      │
│         ↓                    ↓              │
│         └──────────┬─────────┘              │
│                    ↓                        │
│              BA = 87.5%                     │
│                                              │
│  ✅ Solução para desbalanceamento           │
│  ✅ Detecta classificadores enviesados      │
│  ✅ Comparável entre datasets               │
│                                              │
└──────────────────────────────────────────────┘
```

---

**Tags:** #metrics #balanced-accuracy #classification #imbalanced-data #evaluation #IDS

**Voltar para:** [[INDEX]]  
**Artigo relacionado:** [[The_Balanced_Accuracy_and_Its_Posterior_Distribution]]  
**Aplicação:** [[Aplicação_ao_IoT_IDS]]

