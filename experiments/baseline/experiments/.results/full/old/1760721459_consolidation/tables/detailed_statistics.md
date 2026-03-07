# 📊 Estatísticas Detalhadas dos Experimentos

## Resumo por Algoritmo

| algorithm                  |   Accuracy_Mean |   Accuracy_Std |   F1_Mean |   F1_Std |   Training_Time |   Experiments |
|:---------------------------|----------------:|---------------:|----------:|---------:|----------------:|--------------:|
| EllipticEnvelope           |          0.9811 |         0.0026 |    0.9902 |   0.0014 |         14.6592 |       15.0000 |
| GradientBoostingClassifier |          0.9922 |         0.0007 |    0.9960 |   0.0004 |       1844.2263 |       10.0000 |
| IsolationForest            |          0.8837 |         0.1053 |    0.9335 |   0.0629 |          6.8906 |       15.0000 |
| LinearSVC                  |          0.9870 |         0.0002 |    0.9934 |   0.0001 |        149.1727 |       18.0000 |
| LocalOutlierFactor         |          0.9807 |         0.0014 |    0.9900 |   0.0007 |        139.1547 |        8.0000 |
| LogisticRegression         |          0.9870 |         0.0004 |    0.9934 |   0.0002 |         30.0191 |       20.0000 |
| MLPClassifier              |          0.9909 |         0.0004 |    0.9953 |   0.0002 |       1179.2406 |        8.0000 |
| RandomForest               |          0.9923 |         0.0010 |    0.9961 |   0.0005 |        915.2278 |       12.0000 |
| SGDClassifier              |          0.9861 |         0.0013 |    0.9929 |   0.0007 |         27.4730 |       20.0000 |
| SGDOneClassSVM             |          0.9808 |         0.0015 |    0.9901 |   0.0008 |          0.3040 |       15.0000 |

## Legenda
- **Accuracy/F1 Mean**: Valor médio across todas as execuções
- **Accuracy/F1 Std**: Desvio padrão (estabilidade)
- **Training_Time**: Tempo médio de treinamento (s)
- **Experiments**: Total de experimentos executados
