# 📊 Estatísticas Detalhadas dos Experimentos

## Resumo por Algoritmo

| algorithm                  |   Accuracy_Mean |   Accuracy_Std |   F1_Mean |   F1_Std |   Training_Time |   Experiments |
|:---------------------------|----------------:|---------------:|----------:|---------:|----------------:|--------------:|
| EllipticEnvelope           |          0.9811 |         0.0025 |    0.9902 |   0.0014 |         14.6027 |       60.0000 |
| GradientBoostingClassifier |          0.9922 |         0.0007 |    0.9960 |   0.0004 |       1780.4975 |       40.0000 |
| IsolationForest            |          0.8837 |         0.1026 |    0.9335 |   0.0613 |          6.9024 |       60.0000 |
| LinearSVC                  |          0.9870 |         0.0002 |    0.9934 |   0.0001 |        145.9778 |       72.0000 |
| LocalOutlierFactor         |          0.9807 |         0.0013 |    0.9900 |   0.0007 |        139.8023 |       32.0000 |
| LogisticRegression         |          0.9870 |         0.0004 |    0.9934 |   0.0002 |         29.8120 |       80.0000 |
| MLPClassifier              |          0.9909 |         0.0004 |    0.9953 |   0.0002 |       1180.9674 |       32.0000 |
| RandomForest               |          0.9923 |         0.0010 |    0.9961 |   0.0005 |        827.6984 |       48.0000 |
| SGDClassifier              |          0.9861 |         0.0013 |    0.9929 |   0.0007 |         28.6500 |       80.0000 |
| SGDOneClassSVM             |          0.9808 |         0.0014 |    0.9901 |   0.0008 |          0.3265 |       60.0000 |

## Legenda
- **Accuracy/F1 Mean**: Valor médio across todas as execuções
- **Accuracy/F1 Std**: Desvio padrão (estabilidade)
- **Training_Time**: Tempo médio de treinamento (s)
- **Experiments**: Total de experimentos executados
