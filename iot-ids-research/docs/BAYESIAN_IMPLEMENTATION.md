# 🔬 Implementação da Abordagem Bayesiana de Brodersen et al. (2010)

## ✅ Implementação Completa

### 📁 Arquivos Criados

1. **`experiments/bayesian_metrics.py`** (185 linhas)
   - Classe `BayesianAccuracyEvaluator`
   - Implementa distribuições posteriores Beta
   - Calcula Balanced Accuracy via convolução (Eq. 7 do artigo)
   - Intervalos de credibilidade Bayesianos
   - Comparações probabilísticas P(A > B)
   - Funções auxiliares para avaliação rápida

2. **`experiments/bayesian_plots.py`** (310 linhas)
   - `plot_posterior_distributions()` - Distribuições posteriores de BA
   - `plot_credibility_intervals()` - Intervalos de credibilidade 95%
   - `plot_probabilistic_comparison_matrix()` - Matriz P(A > B)
   - `generate_bayesian_statistics_table()` - Tabela com estatísticas completas
   - `generate_all_bayesian_plots()` - Função principal

### 📝 Arquivos Modificados

1. **`experiments/algorithm_comparison.py`**
   - Linha 20: Import `BayesianAccuracyEvaluator`
   - Linhas 753-766: Cálculo de métricas Bayesianas
   - Linhas 788-807: Adição de campos Bayesianos no resultado

2. **`experiments/consolidate_results.py`**
   - Linhas 887-895: Integração dos plots Bayesianos

---

## 📊 Saídas Geradas

### Plots (4 novos)
1. **`bayesian_posterior_distributions.png`**
   - Distribuições posteriores de BA para cada algoritmo
   - Densidades via KDE
   - Médias e IC 95% marcados
   - Linhas de referência (0.8, 0.9)

2. **`bayesian_credibility_intervals.png`**
   - Barras horizontais com intervalos de credibilidade
   - P(BA > 0.9) para cada algoritmo
   - Comparação visual direta

3. **`bayesian_comparison_matrix.png`**
   - Heatmap com P(Algoritmo_i > Algoritmo_j)
   - Verde: forte evidência de superioridade
   - Vermelho: forte evidência de inferioridade
   - Amarelo: diferença não significativa

4. **`bayesian_statistics.csv/md`**
   - Tabela completa com estatísticas Bayesianas
   - Médias, medianas, IC 95%, probabilidades

### Campos Adicionados aos Resultados

Cada experimento agora inclui:

```json
{
  "bayesian": {
    "accuracy_posterior": {
      "mean": 0.8992,
      "median": 0.8995,
      "ci_lower": 0.8798,
      "ci_upper": 0.9171,
      "std": 0.0095
    },
    "balanced_accuracy_posterior": {
      "mean": 0.8985,
      "median": 0.8986,
      "std": 0.0095,
      "ci_lower": 0.8791,
      "ci_upper": 0.9165
    },
    "sensitivity": 0.8941,
    "specificity": 0.9061
  }
}
```

---

## 🎯 Uso

### Durante Experimentos

As métricas Bayesianas são calculadas automaticamente durante `algorithm_comparison.py`.

### Durante Consolidação

Os plots Bayesianos são gerados automaticamente durante `consolidate_results.py`.

### Exemplo de Log

```
🔬 Calculando métricas Bayesianas (Brodersen et al., 2010)...
📊 MÉTRICAS BAYESIANAS:
   BA Média: 0.8985
   BA IC 95%: [0.8791, 0.9165]
   BA Mediana: 0.8986
```

---

## 📚 Referência

**Brodersen, K.H., Ong, C.S., Stephan, K.E., & Buhmann, J.M.** (2010).  
"The balanced accuracy and its posterior distribution".  
*2010 20th International Conference on Pattern Recognition (ICPR)*, pp. 3121-3124. IEEE.

### Equação Implementada (Eq. 7)

```
p_BA(x; TP, FP, FN, TN) = ∫₀¹ p_A(2(x-z); TP+1, FN+1) × p_A(2z; TN+1, FP+1) dz
```

Implementada via Monte Carlo sampling com 50,000 amostras durante experimentos e 100,000 durante consolidação.

---

## ✅ Validação

Teste automático executado com sucesso:
- ✅ Distribuições posteriores calculadas corretamente
- ✅ Intervalos de credibilidade respeitam limites [0,1]
- ✅ Probabilidades condicionais funcionando
- ✅ Comparações probabilísticas implementadas

---

**Status:** ✅ Implementação completa e validada  
**Data:** 2025-10-20

