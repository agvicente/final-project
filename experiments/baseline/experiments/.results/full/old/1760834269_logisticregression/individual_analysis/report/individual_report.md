# 📊 Relatório Individual - LogisticRegression

**Modo de Execução**: COMPLETO  
**Data de Geração**: 2025-10-19 01:09:15  
**Total de Configurações**: 20  
**Total de Execuções**: 60

## 🎯 Resumo Executivo

### Melhor Configuração (param_id: 7)
- **F1-Score Médio**: 0.9935 ± 0.0000
- **Accuracy Média**: 0.9872 ± 0.0000
- **Precision Média**: 0.9931 ± 0.0000
- **Recall Médio**: 0.9938 ± 0.0000
- **Tempo de Treinamento Médio**: 34.40s ± 0.66s
- **Execuções**: 3

### Performance Geral (todas as configurações)
- **F1-Score Médio**: 0.9934 ± 0.0002
- **Accuracy Média**: 0.9870 ± 0.0004
- **Precision Média**: 0.9929 ± 0.0004
- **Recall Médio**: 0.9938 ± 0.0001

### Métricas de Qualidade
- **Estabilidade entre Configurações (Desvio F1)**: 0.0002 🟢 Excelente
- **Eficiência Média**: 0.0352 F1/segundo
- **Tempo Médio**: 29.74s ± 5.63s

## 📈 Análise Detalhada

### Distribuição das Métricas (por configuração)

#### Accuracy
- **Mínimo**: 0.9857
- **Q1**: 0.9871
- **Mediana**: 0.9872
- **Q3**: 0.9872
- **Máximo**: 0.9872
- **IQR**: 0.0001

#### Balanced Accuracy
- **Mínimo**: 0.8234
- **Q1**: 0.8493
- **Mediana**: 0.8516
- **Q3**: 0.8525
- **Máximo**: 0.8535
- **IQR**: 0.0032

#### Precision
- **Mínimo**: 0.9917
- **Q1**: 0.9929
- **Mediana**: 0.9931
- **Q3**: 0.9931
- **Máximo**: 0.9931
- **IQR**: 0.0002

#### Recall
- **Mínimo**: 0.9937
- **Q1**: 0.9938
- **Mediana**: 0.9938
- **Q3**: 0.9938
- **Máximo**: 0.9939
- **IQR**: 0.0001

#### F1 Score
- **Mínimo**: 0.9927
- **Q1**: 0.9934
- **Mediana**: 0.9934
- **Q3**: 0.9934
- **Máximo**: 0.9935
- **IQR**: 0.0000

### Análise de Parâmetros


#### C
- **Melhor valor**: 0.5 (F1: 0.9935)
- **Variação observada**: 20 valores diferentes
- **Impacto no F1**: 0.0008

#### max_iter
- **Melhor valor**: 700 (F1: 0.9935)
- **Variação observada**: 8 valores diferentes
- **Impacto no F1**: 0.0006

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
