# 📊 Relatório Individual - SGDClassifier

**Modo de Execução**: COMPLETO  
**Data de Geração**: 2025-10-14 17:53:14  
**Total de Configurações**: 10  
**Total de Execuções**: 30

## 🎯 Resumo Executivo

### Melhor Configuração (param_id: 8)
- **F1-Score Médio**: 0.9933 ± 0.0000
- **Accuracy Média**: 0.9869 ± 0.0000
- **Precision Média**: 0.9918 ± 0.0000
- **Recall Médio**: 0.9948 ± 0.0000
- **Tempo de Treinamento Médio**: 12.31s ± 0.06s
- **Execuções**: 3

### Performance Geral (todas as configurações)
- **F1-Score Médio**: 0.9931 ± 0.0003
- **Accuracy Média**: 0.9865 ± 0.0006
- **Precision Média**: 0.9918 ± 0.0007
- **Recall Médio**: 0.9944 ± 0.0006

### Métricas de Qualidade
- **Estabilidade entre Configurações (Desvio F1)**: 0.0003 🟢 Excelente
- **Eficiência Média**: 0.0976 F1/segundo
- **Tempo Médio**: 10.98s ± 3.29s

## 📈 Análise Detalhada

### Distribuição das Métricas (por configuração)

#### Accuracy
- **Mínimo**: 0.9850
- **Q1**: 0.9865
- **Mediana**: 0.9867
- **Q3**: 0.9869
- **Máximo**: 0.9869
- **IQR**: 0.0004

#### Balanced Accuracy
- **Mínimo**: 0.8027
- **Q1**: 0.8175
- **Mediana**: 0.8219
- **Q3**: 0.8350
- **Máximo**: 0.8463
- **IQR**: 0.0175

#### Precision
- **Mínimo**: 0.9907
- **Q1**: 0.9914
- **Mediana**: 0.9916
- **Q3**: 0.9923
- **Máximo**: 0.9928
- **IQR**: 0.0008

#### Recall
- **Mínimo**: 0.9937
- **Q1**: 0.9938
- **Mediana**: 0.9945
- **Q3**: 0.9949
- **Máximo**: 0.9950
- **IQR**: 0.0011

#### F1 Score
- **Mínimo**: 0.9923
- **Q1**: 0.9931
- **Mediana**: 0.9932
- **Q3**: 0.9933
- **Máximo**: 0.9933
- **IQR**: 0.0002

### Análise de Parâmetros


#### alpha
- **Melhor valor**: 0.001 (F1: 0.9933)
- **Variação observada**: 4 valores diferentes
- **Impacto no F1**: 0.0009

#### loss
- **Melhor valor**: hinge (F1: 0.9931)
- **Variação observada**: 3 valores diferentes
- **Impacto no F1**: 0.0002

#### max_iter
- **Melhor valor**: 1000 (F1: 0.9932)
- **Variação observada**: 2 valores diferentes
- **Impacto no F1**: 0.0002

#### penalty
- **Melhor valor**: l1 (F1: 0.9933)
- **Variação observada**: 3 valores diferentes
- **Impacto no F1**: 0.0002

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
