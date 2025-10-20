# The Balanced Accuracy and Its Posterior Distribution

> **Autores:** Kay H. Brodersen, Cheng Soon Ong, Klaas E. Stephan, Joachim M. Buhmann  
> **Conferência:** ICPR 2010 (International Conference on Pattern Recognition)  
> **Instituições:** ETH Zurich, University of Zurich  
> **Relevância:** ⭐⭐⭐⭐⭐

---

## 📋 Resumo Executivo

Este artigo seminal propõe substituir a [[Acurácia]] tradicional pela [[Acurácia_Balanceada]] e modelar sua [[Distribuição_Beta|distribuição posterior usando Beta]], resolvendo dois problemas críticos:

1. **[[Intervalos_de_Confiança]] inadequados** com a abordagem [[Métodos_Paramétricos_vs_Não_Paramétricos|não-paramétrica]] tradicional
2. **Viés otimista** em datasets desbalanceados

---

## 🏛️ Contexto Histórico

**Ano:** 2010

**Problema na época:** A comunidade de machine learning avaliava modelos usando [[Acurácia|acurácia média]] através de cross-validation, mas essa abordagem tinha limitações graves:

- Impossibilidade de derivar intervalos de confiança significativos
- Falha em detectar classificadores enviesados explorando desbalanceamento de classes
- Uso inadequado de erro padrão da média levando a intervalos acima de 100%

**Impacto:** O artigo formalizou uma solução matemática rigorosa usando [[Inferência_Bayesiana]] e a [[Distribuição_Beta]].

---

## 🎯 Problemas que Resolve

### Problema 1: Intervalos de Confiança Inadequados

**Método tradicional ([[Métodos_Paramétricos_vs_Não_Paramétricos|não-paramétrico]]):**
```
Acurácia média ± 2 × erro_padrão
```

**Defeitos:**
- ❌ Pode gerar intervalos > 100% ou < 0%
- ❌ Sempre simétrico (não reflete realidade próxima aos limites)
- ❌ Depende de escolhas arbitrárias (número de folds)

**Exemplo problemático:**
```
Acurácia: 98%
Desvio padrão: 2%
Intervalo: [94%, 102%] ← 102%?! Impossível!
```

### Problema 2: Viés em Datasets Desbalanceados

**Cenário crítico para [[Aplicação_ao_IoT_IDS|IoT-IDS]]:**

Dataset com 95% tráfego normal, 5% ataques.

Um modelo "preguiçoso" que **sempre prevê "normal":**
- [[Acurácia]]: 95% ✅ (enganoso!)
- [[Acurácia_Balanceada]]: 50% ❌ (revela a verdade!)

A acurácia tradicional **mascara** a incapacidade de detectar ataques.

---

## 🔬 Metodologia e Solução

### Componente 1: Abordagem Bayesiana para Acurácia

Ao invés de ponto estimado, modela a **distribuição posterior** da [[Acurácia]].

**Framework matemático:**

```
Dados: C corretos, I incorretos
Prior: Uniforme Beta(1, 1)
Posterior: A ~ Beta(C+1, I+1)
```

Veja [[Inferência_Bayesiana]] para detalhes do paradigma.

**Vantagens:**
- ✅ Respeita limites naturais [0, 1]
- ✅ Intervalos assimétricos quando apropriado
- ✅ Interpretação probabilística intuitiva

### Componente 2: Acurácia Balanceada

**Definição:**
```
Balanced Accuracy = ½(TP/P + TN/N)
```

- **TP/P:** Sensibilidade (recall na classe positiva)
- **TN/N:** Especificidade (recall na classe negativa)

Detalhes completos em [[Acurácia_Balanceada]].

**Propriedades:**
- Se reduz à acurácia tradicional quando o desempenho é igual em ambas as classes
- Cai para 50% (nível do acaso) quando modelo é enviesado

### Componente 3: Distribuição Posterior da Acurácia Balanceada

**Contribuição principal do artigo:**

Derivar a [[Distribuição_Beta|distribuição posterior]] da acurácia balanceada através da **convolução de duas distribuições Beta**:

```
pB(x; TP, FP, FN, TN) = ∫₀¹ pA(2(x-z); TP+1, FN+1) · pA(2z; TN+1, FP+1) dz
```

Onde:
- `pA(x)` é a densidade da acurácia para cada classe
- `pB(x)` é a densidade da acurácia balanceada

**Veja [[Distribuição_Beta#Convolução para Balanced Accuracy|seção de convolução]]** para detalhes matemáticos.

**Características:**
- Não tem forma analítica fechada
- Requer integração numérica
- Código MATLAB disponibilizado pelos autores

---

## 📊 Resultados

### Experimento 1: Dataset Balanceado

**Configuração:**
- 70 exemplos positivos
- 70 exemplos negativos
- Matriz de confusão: adequada

**Achados:**
- [[Acurácia|Acurácia tradicional]] com 2 SE: intervalo inclui > 100% ❌
- [[Distribuição_Beta|Posterior Beta]]: intervalos assimétricos corretos ✅
- Pouca diferença entre acurácia e acurácia balanceada (dataset equilibrado)

### Experimento 2: Dataset Desbalanceado com Classificador Enviesado

**Configuração:**
- 45 exemplos positivos
- 10 exemplos negativos
- Classificador prevê: 48 positivos, 7 negativos

**Achados críticos:**

| Métrica | Valor | Interpretação |
|---------|-------|---------------|
| [[Acurácia]] média | 92% | Sugere forte desempenho ❌ |
| Posterior [[Acurácia]] | 92% [88%, 95%] | Confirma alto desempenho ❌ |
| [[Acurácia_Balanceada]] | 51% [48%, 54%] | **Revela nível do acaso!** ✅ |

**Conclusão:** Apenas a acurácia balanceada detectou que o modelo estava apenas explorando o desbalanceamento.

### Visualizações

O artigo apresenta:
1. Comparação entre intervalos (Fig. 1)
2. Distribuições posteriores completas (Fig. 2)
3. Estatísticas: média, mediana, moda, intervalos de probabilidade

---

## 💡 Conclusões do Artigo

### Contribuições Principais

1. **Formalização matemática** da posterior da acurácia balanceada
2. **Demonstração empírica** da superioridade sobre acurácia tradicional
3. **Código disponibilizado** para comunidade (MATLAB)
4. **Generalização possível** para cenários multiclasse (usando [[Distribuições_de_Probabilidade#Dirichlet|Dirichlet]])

### Limitações e Trabalhos Futuros

Os autores mencionam:
- Extensão para variáveis correlacionadas com classes
- Comparação com ROC analysis
- Relação com binomial tail inversion

### Aplicabilidade

- ✅ Qualquer número de folds (só precisa da matriz de confusão agregada)
- ✅ Classificadores individuais ou algoritmos completos
- ✅ Cenários com desbalanceamento de classes
- ✅ Quando intervalos de confiança rigorosos são necessários

---

## ✨ Insights Valiosos para [[Aplicação_ao_IoT_IDS|Pesquisa IoT-IDS]]

### 1. Desbalanceamento é Inevitável em IDS

No dataset CICIoT2023:
- Tráfego normal >> Ataques
- [[Acurácia]] pode ser enganosa
- [[Acurácia_Balanceada]] é **essencial**

### 2. Incerteza Quantificável

A [[Distribuição_Beta|distribuição posterior Beta]] permite:
- Comparar algoritmos com rigor estatístico
- Calcular P(Algoritmo A > Algoritmo B)
- Reportar intervalos de credibilidade ao invés de pontos

### 3. Detecção de Overfitting Enviesado

Modelo que "memoriza" classe majoritária:
- [[Acurácia]] alta ✓
- [[Acurácia_Balanceada]] próxima de 50% ✗
- Posterior revela o problema!

### 4. Integração com Outras Métricas

[[Acurácia_Balanceada]] complementa:
- F1-score
- ROC-AUC
- Precision-Recall curves

### 5. Rigor Científico

Para publicações acadêmicas:
- Intervalos de credibilidade ao invés de apenas média
- Base estatística sólida ([[Inferência_Bayesiana]])
- Comparações algorítmicas significativas

### 6. Implementação Prática

```python
# Exemplo com dados do IoT-IDS
from scipy import stats

# Após classificação
TP, FP, FN, TN = 450, 200, 50, 9300

# Posterior para cada classe
pos_posterior = stats.beta(TP+1, FN+1)
neg_posterior = stats.beta(TN+1, FP+1)

# Balanced accuracy (aproximação pela média das médias)
ba_approx = 0.5 * (pos_posterior.mean() + neg_posterior.mean())

# Para distribuição exata, usar convolução (Eq. 7 do artigo)
```

---

## 🔗 Conceitos Relacionados

### Fundamentos Necessários
- [[Média_Desvio_Padrão_Erro_Padrão]] - Base estatística
- [[Distribuições_de_Probabilidade]] - Teoria geral
- [[Distribuição_Beta]] - Distribuição específica usada
- [[Métodos_Paramétricos_vs_Não_Paramétricos]] - Comparação de abordagens

### Paradigma
- [[Inferência_Bayesiana]] - Framework filosófico e matemático
- [[Intervalos_de_Confiança]] - Quantificação de incerteza

### Aplicação
- [[Acurácia]] - Métrica tradicional
- [[Acurácia_Balanceada]] - Métrica proposta
- [[Aplicação_ao_IoT_IDS]] - Seu contexto específico

---

## 📚 Referências do Artigo

### Citadas no Paper
- **Bishop, C.M.** (2006). *Pattern Recognition and Machine Learning*. Springer. [Cap. 2.2, pp. 68-74]
- **Langford, J.** (2005). "Tutorial on practical prediction theory for classification". *JMLR*, 6, 273-306.
- **Chawla et al.** (2002). "SMOTE: synthetic minority over-sampling technique". *JAIR*, 16(3), 321-357.

### Para Aprofundamento
Veja [[Referências_Bibliográficas]] para lista completa de materiais recomendados.

---

## 🎯 Exercícios Práticos

Para fixar o conteúdo, veja [[Exercícios_Práticos#Balanced Accuracy]].

---

## 📌 Metadados

**Tags:** #paper #bayesian #balanced-accuracy #evaluation-metrics #beta-distribution #classification #imbalanced-data

**Citação:**
```bibtex
@inproceedings{brodersen2010balanced,
  title={The balanced accuracy and its posterior distribution},
  author={Brodersen, Kay H and Ong, Cheng Soon and Stephan, Klaas E and Buhmann, Joachim M},
  booktitle={2010 20th International Conference on Pattern Recognition},
  pages={3121--3124},
  year={2010},
  organization={IEEE}
}
```

**Links Externos:**
- [Paper PDF](../papers/The_Balanced_Accuracy_and_Its_Posterior_Distribution.pdf)
- [Paper TXT](../papers/The_Balanced_Accuracy_and_Its_Posterior_Distribution.txt)
- [Código MATLAB (original)](http://people.inf.ethz.ch/bkay/downloads)

---

**Última atualização:** 2025-10-19  
**Voltar para:** [[INDEX]]

