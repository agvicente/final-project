# 📊 Relatório Individual - SGDClassifier

**Modo de Execução**: COMPLETO  
**Data de Geração**: 2025-10-22 23:23:38  
**Total de Configurações**: 20  
**Total de Execuções**: 100

## 🎯 Resumo Executivo

### Melhor Configuração (param_id: 15)
- **F1-Score Médio**: 0.9934 ± 0.0000
- **Accuracy Média**: 0.9871 ± 0.0000
- **Precision Média**: 0.9925 ± 0.0000
- **Recall Médio**: 0.9944 ± 0.0000
- **Tempo de Treinamento Médio**: 24.37s ± 0.79s
- **Execuções**: 5

### Performance Geral (todas as configurações)
- **F1-Score Médio**: 0.9929 ± 0.0007
- **Accuracy Média**: 0.9861 ± 0.0013
- **Precision Média**: 0.9921 ± 0.0018
- **Recall Médio**: 0.9937 ± 0.0018

### Métricas de Qualidade
- **Estabilidade entre Configurações (Desvio F1)**: 0.0007 🟢 Excelente
- **Eficiência Média**: 0.0620 F1/segundo
- **Tempo Médio**: 27.83s ± 24.19s

## 📈 Análise Detalhada

### Distribuição das Métricas (por configuração)

#### Accuracy
- **Mínimo**: 0.9817
- **Q1**: 0.9861
- **Mediana**: 0.9867
- **Q3**: 0.9869
- **Máximo**: 0.9871
- **IQR**: 0.0007

#### Balanced Accuracy
- **Mínimo**: 0.7370
- **Q1**: 0.8182
- **Mediana**: 0.8346
- **Q3**: 0.8532
- **Máximo**: 0.8914
- **IQR**: 0.0350

#### Precision
- **Mínimo**: 0.9876
- **Q1**: 0.9915
- **Mediana**: 0.9922
- **Q3**: 0.9932
- **Máximo**: 0.9950
- **IQR**: 0.0017

#### Recall
- **Mínimo**: 0.9881
- **Q1**: 0.9932
- **Mediana**: 0.9939
- **Q3**: 0.9947
- **Máximo**: 0.9962
- **IQR**: 0.0015

#### F1 Score
- **Mínimo**: 0.9906
- **Q1**: 0.9929
- **Mediana**: 0.9932
- **Q3**: 0.9933
- **Máximo**: 0.9934
- **IQR**: 0.0004

### Análise de Parâmetros


#### alpha
- **Melhor valor**: 0.0005 (F1: 0.9934)
- **Variação observada**: 10 valores diferentes
- **Impacto no F1**: 0.0015

#### loss
- **Melhor valor**: log_loss (F1: 0.9930)
- **Variação observada**: 3 valores diferentes
- **Impacto no F1**: 0.0008

#### penalty
- **Melhor valor**: l1 (F1: 0.9933)
- **Variação observada**: 3 valores diferentes
- **Impacto no F1**: 0.0005

#### l1_ratio
- **Melhor valor**: 0.3 (F1: 0.9934)
- **Variação observada**: 3 valores diferentes
- **Impacto no F1**: 0.0002

#### max_iter
- **Melhor valor**: 700 (F1: 0.9934)
- **Variação observada**: 7 valores diferentes
- **Impacto no F1**: 0.0011

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
