# 📊 Estatísticas Detalhadas dos Experimentos

## Resumo por Algoritmo

| algorithm                  |   Accuracy_Mean |   Accuracy_Std |   F1_Mean |   F1_Std |   Training_Time |   Experiments |
|:---------------------------|----------------:|---------------:|----------:|---------:|----------------:|--------------:|
| EllipticEnvelope           |          0.9811 |         0.0025 |    0.9902 |   0.0013 |         15.1522 |       75.0000 |
| GradientBoostingClassifier |          0.9922 |         0.0007 |    0.9960 |   0.0004 |       1760.2190 |       50.0000 |
| IsolationForest            |          0.8837 |         0.1024 |    0.9335 |   0.0612 |          6.9024 |       75.0000 |
| LinearSVC                  |          0.9870 |         0.0002 |    0.9934 |   0.0001 |        162.6445 |       90.0000 |
| LocalOutlierFactor         |          0.9807 |         0.0013 |    0.9900 |   0.0007 |        140.5623 |       40.0000 |
| LogisticRegression         |          0.9870 |         0.0004 |    0.9934 |   0.0002 |         30.1801 |      100.0000 |
| MLPClassifier              |          0.9909 |         0.0004 |    0.9953 |   0.0002 |       1199.7022 |       40.0000 |
| RandomForest               |          0.9923 |         0.0010 |    0.9961 |   0.0005 |        879.5958 |       60.0000 |
| SGDClassifier              |          0.9861 |         0.0013 |    0.9929 |   0.0007 |         27.8349 |      100.0000 |
| SGDOneClassSVM             |          0.9808 |         0.0014 |    0.9901 |   0.0008 |          0.3112 |       75.0000 |

## Legenda
- **Accuracy/F1 Mean**: Valor médio across todas as execuções
- **Accuracy/F1 Std**: Desvio padrão (estabilidade)
- **Training_Time**: Tempo médio de treinamento (s)
- **Experiments**: Total de experimentos executados
