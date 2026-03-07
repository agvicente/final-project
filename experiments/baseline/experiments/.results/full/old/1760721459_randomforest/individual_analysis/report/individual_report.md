# 📊 Relatório Individual - RandomForest

**Modo de Execução**: COMPLETO  
**Data de Geração**: 2025-10-17 20:34:28  
**Total de Configurações**: 12  
**Total de Execuções**: 12

## 🎯 Resumo Executivo

### Melhor Configuração (param_id: 11)
- **F1-Score Médio**: 0.9964 ± nan
- **Accuracy Média**: 0.9929 ± nan
- **Precision Média**: 0.9966 ± nan
- **Recall Médio**: 0.9961 ± nan
- **Tempo de Treinamento Médio**: 2640.90s ± nans
- **Execuções**: 1

### Performance Geral (todas as configurações)
- **F1-Score Médio**: 0.9961 ± 0.0005
- **Accuracy Média**: 0.9923 ± 0.0010
- **Precision Média**: 0.9962 ± 0.0009
- **Recall Médio**: 0.9960 ± 0.0002

### Métricas de Qualidade
- **Estabilidade entre Configurações (Desvio F1)**: 0.0005 🟢 Excelente
- **Eficiência Média**: 0.0039 F1/segundo
- **Tempo Médio**: 915.23s ± 847.42s

## 📈 Análise Detalhada

### Distribuição das Métricas (por configuração)

#### Accuracy
- **Mínimo**: 0.9898
- **Q1**: 0.9922
- **Mediana**: 0.9928
- **Q3**: 0.9929
- **Máximo**: 0.9929
- **IQR**: 0.0007

#### Balanced Accuracy
- **Mínimo**: 0.8608
- **Q1**: 0.9164
- **Mediana**: 0.9275
- **Q3**: 0.9278
- **Máximo**: 0.9286
- **IQR**: 0.0115

#### Precision
- **Mínimo**: 0.9935
- **Q1**: 0.9961
- **Mediana**: 0.9966
- **Q3**: 0.9966
- **Máximo**: 0.9967
- **IQR**: 0.0005

#### Recall
- **Mínimo**: 0.9953
- **Q1**: 0.9960
- **Mediana**: 0.9960
- **Q3**: 0.9961
- **Máximo**: 0.9961
- **IQR**: 0.0001

#### F1 Score
- **Mínimo**: 0.9948
- **Q1**: 0.9960
- **Mediana**: 0.9963
- **Q3**: 0.9964
- **Máximo**: 0.9964
- **IQR**: 0.0004

### Análise de Parâmetros


#### max_depth
- **Melhor valor**: 25 (F1: 0.9964)
- **Variação observada**: 8 valores diferentes
- **Impacto no F1**: 0.0016

#### n_estimators
- **Melhor valor**: 350 (F1: 0.9964)
- **Variação observada**: 10 valores diferentes
- **Impacto no F1**: 0.0016

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
