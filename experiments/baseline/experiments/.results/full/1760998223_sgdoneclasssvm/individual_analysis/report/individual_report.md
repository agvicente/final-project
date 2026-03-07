# 📊 Relatório Individual - SGDOneClassSVM

**Modo de Execução**: COMPLETO  
**Data de Geração**: 2025-10-22 23:26:07  
**Total de Configurações**: 15  
**Total de Execuções**: 75

## 🎯 Resumo Executivo

### Melhor Configuração (param_id: 14)
- **F1-Score Médio**: 0.9911 ± 0.0000
- **Accuracy Média**: 0.9827 ± 0.0000
- **Precision Média**: 0.9963 ± 0.0000
- **Recall Médio**: 0.9860 ± 0.0000
- **Tempo de Treinamento Médio**: 0.30s ± 0.05s
- **Execuções**: 5

### Performance Geral (todas as configurações)
- **F1-Score Médio**: 0.9901 ± 0.0008
- **Accuracy Média**: 0.9808 ± 0.0015
- **Precision Média**: 0.9983 ± 0.0012
- **Recall Médio**: 0.9820 ± 0.0026

### Métricas de Qualidade
- **Estabilidade entre Configurações (Desvio F1)**: 0.0008 🟢 Excelente
- **Eficiência Média**: 3.1890 F1/segundo
- **Tempo Médio**: 0.31s ± 0.02s

## 📈 Análise Detalhada

### Distribuição das Métricas (por configuração)

#### Accuracy
- **Mínimo**: 0.9778
- **Q1**: 0.9800
- **Mediana**: 0.9811
- **Q3**: 0.9819
- **Máximo**: 0.9827
- **IQR**: 0.0020

#### Balanced Accuracy
- **Mínimo**: 0.9155
- **Q1**: 0.9384
- **Mediana**: 0.9585
- **Q3**: 0.9747
- **Máximo**: 0.9869
- **IQR**: 0.0363

#### Precision
- **Mínimo**: 0.9963
- **Q1**: 0.9974
- **Mediana**: 0.9984
- **Q3**: 0.9992
- **Máximo**: 0.9999
- **IQR**: 0.0018

#### Recall
- **Mínimo**: 0.9773
- **Q1**: 0.9802
- **Mediana**: 0.9822
- **Q3**: 0.9840
- **Máximo**: 0.9860
- **IQR**: 0.0038

#### F1 Score
- **Mínimo**: 0.9885
- **Q1**: 0.9896
- **Mediana**: 0.9903
- **Q3**: 0.9907
- **Máximo**: 0.9911
- **IQR**: 0.0010

### Análise de Parâmetros


#### max_iter
- **Melhor valor**: 2000 (F1: 0.9911)
- **Variação observada**: 8 valores diferentes
- **Impacto no F1**: 0.0026

#### nu
- **Melhor valor**: 0.5 (F1: 0.9911)
- **Variação observada**: 15 valores diferentes
- **Impacto no F1**: 0.0026

## 🎯 Recomendações

### Pontos Fortes
- ✅ **Excelente performance geral** (F1 > 0.8)
- ✅ **Alta estabilidade** entre execuções
- ✅ **Boa eficiência computacional**

### Áreas de Melhoria

## 📊 Arquivos Gerados

### Gráficos
- `plots/performance_evolution.png` - Evolução das métricas
- `plots/parameter_impact.png` - Impacto dos parâmetros
- `plots/confusion_matrix_analysis.png` - Análise da matriz de confusão
- `plots/metrics_distribution.png` - Distribuição das métricas
- `plots/execution_time_analysis.png` - Análise de tempo

### Tabelas
- `tables/descriptive_statistics.csv` - Estatísticas descritivas
- `tables/detailed_results.csv` - Resultados detalhados
- `tables/execution_ranking.csv` - Ranking por execução

---
*Relatório gerado automaticamente pelo sistema de análise individual*
