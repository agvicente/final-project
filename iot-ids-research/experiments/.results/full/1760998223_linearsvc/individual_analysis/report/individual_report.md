# 📊 Relatório Individual - LinearSVC

**Modo de Execução**: COMPLETO  
**Data de Geração**: 2025-10-22 22:34:51  
**Total de Configurações**: 18  
**Total de Execuções**: 90

## 🎯 Resumo Executivo

### Melhor Configuração (param_id: 5)
- **F1-Score Médio**: 0.9934 ± 0.0000
- **Accuracy Média**: 0.9871 ± 0.0000
- **Precision Média**: 0.9929 ± 0.0000
- **Recall Médio**: 0.9939 ± 0.0000
- **Tempo de Treinamento Médio**: 265.57s ± 21.53s
- **Execuções**: 5

### Performance Geral (todas as configurações)
- **F1-Score Médio**: 0.9934 ± 0.0001
- **Accuracy Média**: 0.9870 ± 0.0002
- **Precision Média**: 0.9928 ± 0.0002
- **Recall Médio**: 0.9939 ± 0.0000

### Métricas de Qualidade
- **Estabilidade entre Configurações (Desvio F1)**: 0.0001 🟢 Excelente
- **Eficiência Média**: 0.0073 F1/segundo
- **Tempo Médio**: 162.64s ± 60.48s

## 📈 Análise Detalhada

### Distribuição das Métricas (por configuração)

#### Accuracy
- **Mínimo**: 0.9862
- **Q1**: 0.9871
- **Mediana**: 0.9871
- **Q3**: 0.9871
- **Máximo**: 0.9871
- **IQR**: 0.0000

#### Balanced Accuracy
- **Mínimo**: 0.8301
- **Q1**: 0.8485
- **Mediana**: 0.8485
- **Q3**: 0.8486
- **Máximo**: 0.8489
- **IQR**: 0.0001

#### Precision
- **Mínimo**: 0.9920
- **Q1**: 0.9929
- **Mediana**: 0.9929
- **Q3**: 0.9929
- **Máximo**: 0.9929
- **IQR**: 0.0000

#### Recall
- **Mínimo**: 0.9939
- **Q1**: 0.9939
- **Mediana**: 0.9939
- **Q3**: 0.9939
- **Máximo**: 0.9940
- **IQR**: 0.0000

#### F1 Score
- **Mínimo**: 0.9930
- **Q1**: 0.9934
- **Mediana**: 0.9934
- **Q3**: 0.9934
- **Máximo**: 0.9934
- **IQR**: 0.0000

### Análise de Parâmetros


#### max_iter
- **Melhor valor**: 1000 (F1: 0.9934)
- **Variação observada**: 8 valores diferentes
- **Impacto no F1**: 0.0003

#### C
- **Melhor valor**: 0.05 (F1: 0.9934)
- **Variação observada**: 18 valores diferentes
- **Impacto no F1**: 0.0004

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
