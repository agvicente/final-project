# 📊 Relatório Final de Experimentos - IoT Anomaly Detection (MODO COMPLETO)

## 🎯 Resumo Executivo

- **Modo de Execução**: 🚀 COMPLETO (dataset completo)
- **Total de Algoritmos Testados**: 20
- **Total de Experimentos**: 564
- **Tempo Total de Execução**: 175856.48 segundos (2930.9 minutos)
- **Coeficiente de Variação Accuracy**: 0.004 (baixa variabilidade)
- **Coeficiente de Variação F1-Score**: 0.002 (baixa variabilidade)

## 🏆 Melhores Resultados

### 🎯 Melhor Accuracy
- **Algoritmo**: GradientBoostingClassifier
- **Accuracy**: 0.9930 (±0.0008)
- **F1-Score**: 0.9964
- **Tempo**: 52887.03s

### 🎯 Melhor F1-Score
- **Algoritmo**: GradientBoostingClassifier
- **F1-Score**: 0.9964 (±0.0004)
- **Accuracy**: 0.9930
- **Tempo**: 52887.03s

### ⚡ Mais Rápido
- **Algoritmo**: SGDOneClassSVM
- **Tempo**: 24.25s
- **Accuracy**: 0.9827
- **F1-Score**: 0.9911
- **Eficiência**: 0.0409 F1/segundo

## 📋 Resultados Detalhados

| Algoritmo | Best Accuracy | Mean Accuracy | Best F1 | Mean F1 | Tempo (s) | Experimentos | Eficiência |
|-----------|---------------|---------------|---------|---------|-----------|--------------|------------|
| LogisticRegression | 0.9872 | 0.9870 | 0.9935 | 0.9934 | 1869.0 | 60 | 0.0005 |
| LocalOutlierFactor | 0.9824 | 0.9807 | 0.9909 | 0.9900 | 6538.7 | 24 | 0.0002 |
| GradientBoostingClassifier | 0.9930 | 0.9922 | 0.9964 | 0.9960 | 52887.0 | 30 | 0.0000 |
| LinearSVC | 0.9871 | 0.9870 | 0.9934 | 0.9934 | 7895.5 | 54 | 0.0001 |
| SGDOneClassSVM | 0.9827 | 0.9808 | 0.9911 | 0.9901 | 73.3 | 45 | 0.0135 |
| IsolationForest | 0.9805 | 0.8837 | 0.9900 | 0.9335 | 699.8 | 45 | 0.0014 |
| EllipticEnvelope | 0.9842 | 0.9811 | 0.9919 | 0.9902 | 767.1 | 45 | 0.0013 |
| RandomForest | 0.9929 | 0.9923 | 0.9964 | 0.9961 | 29190.0 | 36 | 0.0000 |
| MLPClassifier | 0.9913 | 0.9909 | 0.9956 | 0.9953 | 28433.6 | 24 | 0.0000 |
| SGDClassifier | 0.9871 | 0.9861 | 0.9934 | 0.9929 | 1822.1 | 60 | 0.0005 |
| EllipticEnvelope | 0.9842 | 0.9811 | 0.9919 | 0.9902 | 257.2 | 15 | 0.0039 |
| MLPClassifier | 0.9913 | 0.9909 | 0.9956 | 0.9953 | 9458.1 | 8 | 0.0001 |
| GradientBoostingClassifier | 0.9930 | 0.9922 | 0.9964 | 0.9960 | 18479.7 | 10 | 0.0001 |
| LocalOutlierFactor | 0.9824 | 0.9807 | 0.9909 | 0.9900 | 2167.8 | 8 | 0.0005 |
| SGDClassifier | 0.9871 | 0.9861 | 0.9934 | 0.9929 | 574.5 | 20 | 0.0017 |
| LinearSVC | 0.9871 | 0.9870 | 0.9934 | 0.9934 | 2710.1 | 18 | 0.0004 |
| RandomForest | 0.9929 | 0.9923 | 0.9964 | 0.9961 | 11139.4 | 12 | 0.0001 |
| IsolationForest | 0.9805 | 0.8837 | 0.9900 | 0.9335 | 238.5 | 15 | 0.0042 |
| LogisticRegression | 0.9872 | 0.9870 | 0.9935 | 0.9934 | 630.7 | 20 | 0.0016 |
| SGDOneClassSVM | 0.9827 | 0.9808 | 0.9911 | 0.9901 | 24.2 | 15 | 0.0409 |

## 📊 Análise Estatística Avançada

### Métricas de Performance
- **Accuracy Média Geral**: 0.9868 ± 0.0043
- **F1-Score Médio Geral**: 0.9933 ± 0.0022
- **Algoritmo mais Consistente (menor CV)**: LinearSVC

### Métricas de Eficiência
- **Tempo Médio por Algoritmo**: 8792.82s ± 13750.06s
- **Total de Experimentos Executados**: 564
- **Experimentos por Minuto**: 0.2

### Rankings
1. **Por Performance (F1)**: GradientBoostingClassifier, GradientBoostingClassifier, RandomForest
2. **Por Velocidade**: SGDOneClassSVM, SGDOneClassSVM, IsolationForest
3. **Por Eficiência (F1/tempo)**: SGDOneClassSVM, SGDOneClassSVM, IsolationForest

## 🔧 Configuração dos Experimentos

- **Configurações por Algoritmo**: 14.1 (média)
- **Execuções por Configuração**: 2.0 (média)
- **Rigor Estatístico**: ✅ Múltiplas execuções (5 runs) para cada configuração
- **Validação**: ✅ Holdout test set independente

### 🎛️ Estratégia Adaptativa de Configurações (Opção C)

**Racional**: O número de configurações varia por algoritmo conforme sua complexidade computacional,
mantendo o tempo total de execução em ~24h e garantindo cobertura abrangente do espaço de hiperparâmetros.

**Distribuição por Complexidade**:
- ⚡ **Algoritmos Rápidos (20 configs)**: LogisticRegression, SGDClassifier
- 🔄 **Algoritmos Médios (12-18 configs)**: RandomForest(12), LinearSVC(18), IsolationForest(15), EllipticEnvelope(15), SGDOneClassSVM(15)
- 🐢 **Algoritmos Pesados (8-10 configs)**: GradientBoosting(10), LocalOutlierFactor(8), MLPClassifier(8)

**Totais**: 141 configurações × 5 runs = 705 experimentos | Tempo estimado: ~30h

**Estratégia de Amostragem**: Cada algoritmo possui configurações organizadas em 4 faixas:
1. **LEVES (20%)**: Modelos muito simples, deployable em edge devices
2. **SWEET SPOT (40%)**: Range ideal para IoT, balanceando performance e recursos
3. **MÉDIAS (20%)**: Configurações moderadas, para edge servers
4. **PESADAS (20%)**: Limite da capacidade IoT, para gateways e fog nodes

**Comparabilidade**: Apesar do número variável, todos os algoritmos são comparáveis pois:
- Utilizam 5 runs cada para rigor estatístico
- Compartilham o mesmo train/test split (random_state=42)
- Incluem configurações leves e pesadas para análise de trade-offs
- Focam no sweet spot IoT (40% das configurações)

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
- **Melhor Performance**: Use **GradientBoostingClassifier** (F1: 0.9964)
- **Melhor Velocidade**: Use **SGDOneClassSVM** (24.25s)
- **Balanceado**: Considere trade-off entre performance e velocidade

### Para Pesquisa
- Investigar parâmetros que causaram maior variabilidade
- Comparar com outros datasets de IoT
- Analisar interpretabilidade dos modelos

---
*Relatório gerado automaticamente pelo pipeline DVC avançado de experimentos de detecção de anomalias em IoT*
*Data: 2025-10-20 20:59:41*
