# 📊 Relatório Individual - IsolationForest

**Modo de Execução**: COMPLETO  
**Data de Geração**: 2025-10-20 00:10:00  
**Total de Configurações**: 15  
**Total de Execuções**: 45

## 🎯 Resumo Executivo

### Melhor Configuração (param_id: 13)
- **F1-Score Médio**: 0.9900 ± 0.0000
- **Accuracy Média**: 0.9805 ± 0.0000
- **Precision Média**: 0.9927 ± 0.0000
- **Recall Médio**: 0.9873 ± 0.0000
- **Tempo de Treinamento Médio**: 11.43s ± 0.49s
- **Execuções**: 3

### Performance Geral (todas as configurações)
- **F1-Score Médio**: 0.9335 ± 0.0629
- **Accuracy Média**: 0.8837 ± 0.1053
- **Precision Média**: 0.9954 ± 0.0017
- **Recall Médio**: 0.8852 ± 0.1096

### Métricas de Qualidade
- **Estabilidade entre Configurações (Desvio F1)**: 0.0629 🔴 Instável
- **Eficiência Média**: 0.1739 F1/segundo
- **Tempo Médio**: 6.91s ± 3.46s

## 📈 Análise Detalhada

### Distribuição das Métricas (por configuração)

#### Accuracy
- **Mínimo**: 0.6900
- **Q1**: 0.7768
- **Mediana**: 0.9351
- **Q3**: 0.9745
- **Máximo**: 0.9805
- **IQR**: 0.1977

#### Balanced Accuracy
- **Mínimo**: 0.8157
- **Q1**: 0.8365
- **Mediana**: 0.8422
- **Q3**: 0.8763
- **Máximo**: 0.8857
- **IQR**: 0.0398

#### Precision
- **Mínimo**: 0.9927
- **Q1**: 0.9942
- **Mediana**: 0.9954
- **Q3**: 0.9966
- **Máximo**: 0.9982
- **IQR**: 0.0023

#### Recall
- **Mínimo**: 0.6838
- **Q1**: 0.7741
- **Mediana**: 0.9379
- **Q3**: 0.9796
- **Máximo**: 0.9873
- **IQR**: 0.2055

#### F1 Score
- **Mínimo**: 0.8116
- **Q1**: 0.8714
- **Mediana**: 0.9658
- **Q3**: 0.9869
- **Máximo**: 0.9900
- **IQR**: 0.1155

### Análise de Parâmetros


#### n_estimators
- **Melhor valor**: 300 (F1: 0.9899)
- **Variação observada**: 6 valores diferentes
- **Impacto no F1**: 0.1448

#### contamination
- **Melhor valor**: 0.3 (F1: 0.9900)
- **Variação observada**: 10 valores diferentes
- **Impacto no F1**: 0.1784

## 🎯 Recomendações

### Pontos Fortes
- ✅ **Excelente performance geral** (F1 > 0.8)
- ✅ **Boa eficiência computacional**

### Áreas de Melhoria
- 🔴 **Alta variabilidade** - resultados inconsistentes

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
