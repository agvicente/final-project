# 📊 Relatório Individual - GradientBoostingClassifier

**Modo de Execução**: COMPLETO  
**Data de Geração**: 2025-10-14 13:20:18  
**Total de Configurações**: 10  
**Total de Execuções**: 30

## 🎯 Resumo Executivo

### Melhor Configuração (param_id: 9)
- **F1-Score Médio**: 0.9965 ± 0.0000
- **Accuracy Média**: 0.9931 ± 0.0000
- **Precision Média**: 0.9967 ± 0.0000
- **Recall Médio**: 0.9962 ± 0.0000
- **Tempo de Treinamento Médio**: 6776.25s ± 102.46s
- **Execuções**: 3

### Performance Geral (todas as configurações)
- **F1-Score Médio**: 0.9959 ± 0.0005
- **Accuracy Média**: 0.9921 ± 0.0009
- **Precision Média**: 0.9961 ± 0.0007
- **Recall Médio**: 0.9958 ± 0.0003

### Métricas de Qualidade
- **Estabilidade entre Configurações (Desvio F1)**: 0.0005 🟢 Excelente
- **Eficiência Média**: 0.0014 F1/segundo
- **Tempo Médio**: 2091.99s ± 2083.88s

## 📈 Análise Detalhada

### Distribuição das Métricas (por configuração)

#### Accuracy
- **Mínimo**: 0.9904
- **Q1**: 0.9914
- **Mediana**: 0.9923
- **Q3**: 0.9928
- **Máximo**: 0.9931
- **IQR**: 0.0015

#### Balanced Accuracy
- **Mínimo**: 0.8913
- **Q1**: 0.9053
- **Mediana**: 0.9199
- **Q3**: 0.9300
- **Máximo**: 0.9306
- **IQR**: 0.0246

#### Precision
- **Mínimo**: 0.9949
- **Q1**: 0.9956
- **Mediana**: 0.9963
- **Q3**: 0.9967
- **Máximo**: 0.9968
- **IQR**: 0.0012

#### Recall
- **Mínimo**: 0.9953
- **Q1**: 0.9955
- **Mediana**: 0.9959
- **Q3**: 0.9960
- **Máximo**: 0.9962
- **IQR**: 0.0004

#### F1 Score
- **Mínimo**: 0.9951
- **Q1**: 0.9956
- **Mediana**: 0.9960
- **Q3**: 0.9963
- **Máximo**: 0.9965
- **IQR**: 0.0007

### Análise de Parâmetros


#### n_estimators
- **Melhor valor**: 200 (F1: 0.9964)
- **Variação observada**: 6 valores diferentes
- **Impacto no F1**: 0.0013

#### max_depth
- **Melhor valor**: 10 (F1: 0.9965)
- **Variação observada**: 4 valores diferentes
- **Impacto no F1**: 0.0011

#### learning_rate
- **Melhor valor**: 0.05 (F1: 0.9962)
- **Variação observada**: 4 valores diferentes
- **Impacto no F1**: 0.0011

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
