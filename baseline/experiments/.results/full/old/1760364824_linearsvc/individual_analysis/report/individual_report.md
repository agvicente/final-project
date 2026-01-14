# 📊 Relatório Individual - LinearSVC

**Modo de Execução**: COMPLETO  
**Data de Geração**: 2025-10-14 17:46:58  
**Total de Configurações**: 10  
**Total de Execuções**: 30

## 🎯 Resumo Executivo

### Melhor Configuração (param_id: 3)
- **F1-Score Médio**: 0.9934 ± 0.0000
- **Accuracy Média**: 0.9871 ± 0.0000
- **Precision Média**: 0.9929 ± 0.0000
- **Recall Médio**: 0.9939 ± 0.0000
- **Tempo de Treinamento Médio**: 170.73s ± 36.87s
- **Execuções**: 3

### Performance Geral (todas as configurações)
- **F1-Score Médio**: 0.9934 ± 0.0000
- **Accuracy Média**: 0.9871 ± 0.0001
- **Precision Média**: 0.9929 ± 0.0001
- **Recall Médio**: 0.9939 ± 0.0000

### Métricas de Qualidade
- **Estabilidade entre Configurações (Desvio F1)**: 0.0000 🟢 Excelente
- **Eficiência Média**: 0.0071 F1/segundo
- **Tempo Médio**: 146.49s ± 29.12s

## 📈 Análise Detalhada

### Distribuição das Métricas (por configuração)

#### Accuracy
- **Mínimo**: 0.9869
- **Q1**: 0.9871
- **Mediana**: 0.9871
- **Q3**: 0.9871
- **Máximo**: 0.9871
- **IQR**: 0.0000

#### Balanced Accuracy
- **Mínimo**: 0.8438
- **Q1**: 0.8485
- **Mediana**: 0.8485
- **Q3**: 0.8486
- **Máximo**: 0.8488
- **IQR**: 0.0001

#### Precision
- **Mínimo**: 0.9927
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
- **Máximo**: 0.9939
- **IQR**: 0.0000

#### F1 Score
- **Mínimo**: 0.9933
- **Q1**: 0.9934
- **Mediana**: 0.9934
- **Q3**: 0.9934
- **Máximo**: 0.9934
- **IQR**: 0.0000

### Análise de Parâmetros


#### C
- **Melhor valor**: 0.5 (F1: 0.9934)
- **Variação observada**: 10 valores diferentes
- **Impacto no F1**: 0.0001

#### max_iter
- **Melhor valor**: 1000 (F1: 0.9934)
- **Variação observada**: 3 valores diferentes
- **Impacto no F1**: 0.0000

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
