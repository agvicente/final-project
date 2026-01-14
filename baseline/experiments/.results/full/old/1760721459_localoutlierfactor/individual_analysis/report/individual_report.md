# 📊 Relatório Individual - LocalOutlierFactor

**Modo de Execução**: COMPLETO  
**Data de Geração**: 2025-10-18 02:28:26  
**Total de Configurações**: 8  
**Total de Execuções**: 8

## 🎯 Resumo Executivo

### Melhor Configuração (param_id: 7)
- **F1-Score Médio**: 0.9909 ± nan
- **Accuracy Média**: 0.9824 ± nan
- **Precision Média**: 0.9952 ± nan
- **Recall Médio**: 0.9867 ± nan
- **Tempo de Treinamento Médio**: 142.09s ± nans
- **Execuções**: 1

### Performance Geral (todas as configurações)
- **F1-Score Médio**: 0.9900 ± 0.0007
- **Accuracy Média**: 0.9807 ± 0.0014
- **Precision Média**: 0.9968 ± 0.0009
- **Recall Médio**: 0.9833 ± 0.0022

### Métricas de Qualidade
- **Estabilidade entre Configurações (Desvio F1)**: 0.0007 🟢 Excelente
- **Eficiência Média**: 0.0071 F1/segundo
- **Tempo Médio**: 139.15s ± 1.59s

## 📈 Análise Detalhada

### Distribuição das Métricas (por configuração)

#### Accuracy
- **Mínimo**: 0.9784
- **Q1**: 0.9798
- **Mediana**: 0.9810
- **Q3**: 0.9816
- **Máximo**: 0.9824
- **IQR**: 0.0019

#### Balanced Accuracy
- **Mínimo**: 0.8936
- **Q1**: 0.9183
- **Mediana**: 0.9286
- **Q3**: 0.9401
- **Máximo**: 0.9415
- **IQR**: 0.0218

#### Precision
- **Mínimo**: 0.9952
- **Q1**: 0.9964
- **Mediana**: 0.9970
- **Q3**: 0.9975
- **Máximo**: 0.9976
- **IQR**: 0.0011

#### Recall
- **Mínimo**: 0.9803
- **Q1**: 0.9817
- **Mediana**: 0.9835
- **Q3**: 0.9848
- **Máximo**: 0.9867
- **IQR**: 0.0030

#### F1 Score
- **Mínimo**: 0.9888
- **Q1**: 0.9896
- **Mediana**: 0.9902
- **Q3**: 0.9905
- **Máximo**: 0.9909
- **IQR**: 0.0010

### Análise de Parâmetros


#### contamination
- **Melhor valor**: 0.2 (F1: 0.9909)
- **Variação observada**: 3 valores diferentes
- **Impacto no F1**: 0.0015

#### n_neighbors
- **Melhor valor**: 50 (F1: 0.9909)
- **Variação observada**: 7 valores diferentes
- **Impacto no F1**: 0.0021

## 🎯 Recomendações

### Pontos Fortes
- ✅ **Excelente performance geral** (F1 > 0.8)
- ✅ **Alta estabilidade** entre execuções

### Áreas de Melhoria
- 🟡 **Tempo de treinamento elevado** - considerar otimizações

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
