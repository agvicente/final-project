# 📊 Relatório Final de Experimentos - IoT Anomaly Detection (MODO TESTE)

## 🎯 Resumo Executivo

- **Modo de Execução**: 🧪 TESTE (dados reduzidos)
- **Total de Algoritmos Testados**: 5
- **Total de Experimentos**: 14
- **Tempo Total de Execução**: 42.01 segundos (0.7 minutos)
- **Coeficiente de Variação Accuracy**: 0.045 (baixa variabilidade)
- **Coeficiente de Variação F1-Score**: 0.024 (baixa variabilidade)

## 🏆 Melhores Resultados

### 🎯 Melhor Accuracy
- **Algoritmo**: RandomForestClassifier
- **Accuracy**: 0.9840 (±0.0040)
- **F1-Score**: 0.9918
- **Tempo**: 8.66s

### 🎯 Melhor F1-Score
- **Algoritmo**: RandomForestClassifier
- **F1-Score**: 0.9918 (±0.0021)
- **Accuracy**: 0.9840
- **Tempo**: 8.66s

### ⚡ Mais Rápido
- **Algoritmo**: SVC
- **Tempo**: 7.56s
- **Accuracy**: 0.9720
- **F1-Score**: 0.9856
- **Eficiência**: 0.1305 F1/segundo

## 📋 Resultados Detalhados

| Algoritmo | Best Accuracy | Mean Accuracy | Best F1 | Mean F1 | Tempo (s) | Experimentos | Eficiência |
|-----------|---------------|---------------|---------|---------|-----------|--------------|------------|
| LogisticRegression | 0.9720 | 0.9720 | 0.9856 | 0.9856 | 8.5 | 4 | 0.1165 |
| RandomForestClassifier | 0.9840 | 0.9800 | 0.9918 | 0.9897 | 8.7 | 4 | 0.1146 |
| OneClassSVM | 0.9720 | 0.9720 | 0.9855 | 0.9855 | 9.6 | 2 | 0.1022 |
| IsolationForest | 0.8800 | 0.8800 | 0.9345 | 0.9345 | 7.7 | 2 | 0.1214 |
| SVC | 0.9720 | 0.9720 | 0.9856 | 0.9856 | 7.6 | 2 | 0.1305 |

## 📊 Análise Estatística Avançada

### Métricas de Performance
- **Accuracy Média Geral**: 0.9560 ± 0.0428
- **F1-Score Médio Geral**: 0.9766 ± 0.0237
- **Algoritmo mais Consistente (menor CV)**: LogisticRegression

### Métricas de Eficiência
- **Tempo Médio por Algoritmo**: 8.40s ± 0.84s
- **Total de Experimentos Executados**: 14
- **Experimentos por Minuto**: 20.0

### Rankings
1. **Por Performance (F1)**: RandomForestClassifier, LogisticRegression, SVC
2. **Por Velocidade**: SVC, IsolationForest, LogisticRegression
3. **Por Eficiência (F1/tempo)**: SVC, IsolationForest, LogisticRegression

## 🔧 Configuração dos Experimentos

- **Configurações por Algoritmo**: 1.4 (média)
- **Execuções por Configuração**: 2.0 (média)
- **Rigor Estatístico**: ✅ Múltiplas execuções para cada configuração
- **Validação**: ✅ Holdout test set independente

## 📈 Gráficos e Análises Geradas

1. **Gráficos Básicos**: Comparações de accuracy, F1-score, tempo de execução
2. **Análises Avançadas**: 
   - 📊 Matrizes de confusão agregadas
   - 📦 Boxplots de distribuições
   - 🔥 Heatmap de correlações
   - ⚡ Análises de performance detalhadas
   - 🔧 Impacto de parâmetros
   - 🔍 Análise específica de detecção de anomalias

## 💡 Recomendações

### Para Produção
- **Melhor Performance**: Use **RandomForestClassifier** (F1: 0.9918)
- **Melhor Velocidade**: Use **SVC** (7.56s)
- **Balanceado**: Considere trade-off entre performance e velocidade

### Para Pesquisa
- Investigar parâmetros que causaram maior variabilidade
- Comparar com outros datasets de IoT
- Analisar interpretabilidade dos modelos

---
*Relatório gerado automaticamente pelo pipeline DVC avançado de experimentos de detecção de anomalias em IoT*
*Data: 2025-10-01 22:52:44*
