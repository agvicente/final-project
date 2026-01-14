# 📊 Relatório Individual - SGDOneClassSVM

**Modo de Execução**: COMPLETO  
**Data de Geração**: 2025-10-14 17:54:07  
**Total de Configurações**: 10  
**Total de Execuções**: 30

## 🎯 Resumo Executivo

### Melhor Configuração (param_id: 9)
- **F1-Score Médio**: 0.9905 ± 0.0000
- **Accuracy Média**: 0.9815 ± 0.0000
- **Precision Média**: 0.9979 ± 0.0000
- **Recall Médio**: 0.9831 ± 0.0000
- **Tempo de Treinamento Médio**: 0.21s ± 0.01s
- **Execuções**: 3

### Performance Geral (todas as configurações)
- **F1-Score Médio**: 0.9897 ± 0.0006
- **Accuracy Média**: 0.9801 ± 0.0012
- **Precision Média**: 0.9990 ± 0.0007
- **Recall Médio**: 0.9806 ± 0.0019

### Métricas de Qualidade
- **Estabilidade entre Configurações (Desvio F1)**: 0.0006 🟢 Excelente
- **Eficiência Média**: 4.9147 F1/segundo
- **Tempo Médio**: 0.20s ± 0.00s

## 📈 Análise Detalhada

### Distribuição das Métricas (por configuração)

#### Accuracy
- **Mínimo**: 0.9778
- **Q1**: 0.9794
- **Mediana**: 0.9803
- **Q3**: 0.9810
- **Máximo**: 0.9815
- **IQR**: 0.0017

#### Balanced Accuracy
- **Mínimo**: 0.9490
- **Q1**: 0.9597
- **Mediana**: 0.9703
- **Q3**: 0.9798
- **Máximo**: 0.9869
- **IQR**: 0.0201

#### Precision
- **Mínimo**: 0.9979
- **Q1**: 0.9985
- **Mediana**: 0.9990
- **Q3**: 0.9995
- **Máximo**: 0.9999
- **IQR**: 0.0010

#### Recall
- **Mínimo**: 0.9773
- **Q1**: 0.9793
- **Mediana**: 0.9808
- **Q3**: 0.9821
- **Máximo**: 0.9831
- **IQR**: 0.0027

#### F1 Score
- **Mínimo**: 0.9885
- **Q1**: 0.9893
- **Mediana**: 0.9898
- **Q3**: 0.9902
- **Máximo**: 0.9905
- **IQR**: 0.0009

### Análise de Parâmetros


#### max_iter
- **Melhor valor**: 1500 (F1: 0.9903)
- **Variação observada**: 3 valores diferentes
- **Impacto no F1**: 0.0014

#### nu
- **Melhor valor**: 0.25 (F1: 0.9905)
- **Variação observada**: 10 valores diferentes
- **Impacto no F1**: 0.0020

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
