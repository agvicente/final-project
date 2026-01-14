# 📊 Relatório Final de Experimentos - IoT Anomaly Detection (MODO COMPLETO)

## 🎯 Resumo Executivo

- **Modo de Execução**: 🚀 COMPLETO (dataset completo)
- **Total de Algoritmos Testados**: 9
- **Total de Experimentos**: 270
- **Tempo Total de Execução**: 97682.84 segundos (1628.0 minutos)
- **Coeficiente de Variação Accuracy**: 0.005 (baixa variabilidade)
- **Coeficiente de Variação F1-Score**: 0.003 (baixa variabilidade)

## 🏆 Melhores Resultados

### 🎯 Melhor Accuracy
- **Algoritmo**: GradientBoostingClassifier
- **Accuracy**: 0.9931 (±0.0010)
- **F1-Score**: 0.9965
- **Tempo**: 62848.24s

### 🎯 Melhor F1-Score
- **Algoritmo**: GradientBoostingClassifier
- **F1-Score**: 0.9965 (±0.0005)
- **Accuracy**: 0.9931
- **Tempo**: 62848.24s

### ⚡ Mais Rápido
- **Algoritmo**: SGDOneClassSVM
- **Tempo**: 35.86s
- **Accuracy**: 0.9815
- **F1-Score**: 0.9905
- **Eficiência**: 0.0276 F1/segundo

## 📋 Resultados Detalhados

| Algoritmo | Best Accuracy | Mean Accuracy | Best F1 | Mean F1 | Tempo (s) | Experimentos | Eficiência |
|-----------|---------------|---------------|---------|---------|-----------|--------------|------------|
| RandomForest | 0.9929 | 0.9922 | 0.9964 | 0.9960 | 17508.0 | 30 | 0.0001 |
| LocalOutlierFactor | 0.9824 | 0.9798 | 0.9909 | 0.9896 | 8676.4 | 30 | 0.0001 |
| IsolationForest | 0.9794 | 0.8415 | 0.9894 | 0.9087 | 317.2 | 30 | 0.0031 |
| EllipticEnvelope | 0.9829 | 0.9799 | 0.9912 | 0.9896 | 2535.3 | 30 | 0.0004 |
| LinearSVC | 0.9871 | 0.9871 | 0.9934 | 0.9934 | 4425.6 | 30 | 0.0002 |
| GradientBoostingClassifier | 0.9931 | 0.9921 | 0.9965 | 0.9959 | 62848.2 | 30 | 0.0000 |
| LogisticRegression | 0.9872 | 0.9871 | 0.9935 | 0.9934 | 978.3 | 30 | 0.0010 |
| SGDClassifier | 0.9869 | 0.9865 | 0.9933 | 0.9931 | 358.0 | 30 | 0.0028 |
| SGDOneClassSVM | 0.9815 | 0.9801 | 0.9905 | 0.9897 | 35.9 | 30 | 0.0276 |

## 📊 Análise Estatística Avançada

### Métricas de Performance
- **Accuracy Média Geral**: 0.9859 ± 0.0048
- **F1-Score Médio Geral**: 0.9928 ± 0.0025
- **Algoritmo mais Consistente (menor CV)**: LinearSVC

### Métricas de Eficiência
- **Tempo Médio por Algoritmo**: 10853.65s ± 20306.08s
- **Total de Experimentos Executados**: 270
- **Experimentos por Minuto**: 0.2

### Rankings
1. **Por Performance (F1)**: GradientBoostingClassifier, RandomForest, LogisticRegression
2. **Por Velocidade**: SGDOneClassSVM, IsolationForest, SGDClassifier
3. **Por Eficiência (F1/tempo)**: SGDOneClassSVM, IsolationForest, SGDClassifier

## 🔧 Configuração dos Experimentos

- **Configurações por Algoritmo**: 10.0 (média)
- **Execuções por Configuração**: 3.0 (média)
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
- **Melhor Performance**: Use **GradientBoostingClassifier** (F1: 0.9965)
- **Melhor Velocidade**: Use **SGDOneClassSVM** (35.86s)
- **Balanceado**: Considere trade-off entre performance e velocidade

### Para Pesquisa
- Investigar parâmetros que causaram maior variabilidade
- Comparar com outros datasets de IoT
- Analisar interpretabilidade dos modelos

---
*Relatório gerado automaticamente pelo pipeline DVC avançado de experimentos de detecção de anomalias em IoT*
*Data: 2025-10-15 14:28:28*
