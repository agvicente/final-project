# 📊 Relatório Individual - GradientBoostingClassifier

**Modo de Execução**: COMPLETO  
**Data de Geração**: 2025-10-18 01:42:51  
**Total de Configurações**: 10  
**Total de Execuções**: 10

## 🎯 Resumo Executivo

### Melhor Configuração (param_id: 9)
- **F1-Score Médio**: 0.9964 ± nan
- **Accuracy Média**: 0.9930 ± nan
- **Precision Média**: 0.9968 ± nan
- **Recall Médio**: 0.9960 ± nan
- **Tempo de Treinamento Médio**: 4228.19s ± nans
- **Execuções**: 1

### Performance Geral (todas as configurações)
- **F1-Score Médio**: 0.9960 ± 0.0004
- **Accuracy Média**: 0.9922 ± 0.0007
- **Precision Média**: 0.9962 ± 0.0006
- **Recall Médio**: 0.9958 ± 0.0002

### Métricas de Qualidade
- **Estabilidade entre Configurações (Desvio F1)**: 0.0004 🟢 Excelente
- **Eficiência Média**: 0.0012 F1/segundo
- **Tempo Médio**: 1844.23s ± 1433.10s

## 📈 Análise Detalhada

### Distribuição das Métricas (por configuração)

#### Accuracy
- **Mínimo**: 0.9910
- **Q1**: 0.9917
- **Mediana**: 0.9923
- **Q3**: 0.9928
- **Máximo**: 0.9930
- **IQR**: 0.0011

#### Balanced Accuracy
- **Mínimo**: 0.8975
- **Q1**: 0.9095
- **Mediana**: 0.9207
- **Q3**: 0.9282
- **Máximo**: 0.9310
- **IQR**: 0.0187

#### Precision
- **Mínimo**: 0.9952
- **Q1**: 0.9958
- **Mediana**: 0.9963
- **Q3**: 0.9967
- **Máximo**: 0.9968
- **IQR**: 0.0009

#### Recall
- **Mínimo**: 0.9955
- **Q1**: 0.9957
- **Mediana**: 0.9958
- **Q3**: 0.9960
- **Máximo**: 0.9960
- **IQR**: 0.0003

#### F1 Score
- **Mínimo**: 0.9954
- **Q1**: 0.9957
- **Mediana**: 0.9961
- **Q3**: 0.9963
- **Máximo**: 0.9964
- **IQR**: 0.0006

### Análise de Parâmetros


#### learning_rate
- **Melhor valor**: 0.05 (F1: 0.9962)
- **Variação observada**: 4 valores diferentes
- **Impacto no F1**: 0.0008

#### n_estimators
- **Melhor valor**: 250 (F1: 0.9964)
- **Variação observada**: 6 valores diferentes
- **Impacto no F1**: 0.0010

#### max_depth
- **Melhor valor**: 7 (F1: 0.9964)
- **Variação observada**: 5 valores diferentes
- **Impacto no F1**: 0.0010

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
