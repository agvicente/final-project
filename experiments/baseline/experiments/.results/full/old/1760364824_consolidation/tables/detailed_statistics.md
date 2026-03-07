# 📊 Estatísticas Detalhadas dos Experimentos

## Resumo por Algoritmo

| algorithm                  |   Accuracy_Mean |   Accuracy_Std |   F1_Mean |   F1_Std |   Training_Time |   Experiments |
|:---------------------------|----------------:|---------------:|----------:|---------:|----------------:|--------------:|
| EllipticEnvelope           |          0.9799 |         0.0023 |    0.9896 |   0.0012 |         82.5365 |       30.0000 |
| GradientBoostingClassifier |          0.9921 |         0.0009 |    0.9959 |   0.0005 |       2091.9866 |       30.0000 |
| IsolationForest            |          0.8415 |         0.1009 |    0.9087 |   0.0607 |          5.3464 |       30.0000 |
| LinearSVC                  |          0.9871 |         0.0001 |    0.9934 |   0.0000 |        146.4941 |       30.0000 |
| LocalOutlierFactor         |          0.9798 |         0.0035 |    0.9896 |   0.0018 |        150.6161 |       30.0000 |
| LogisticRegression         |          0.9871 |         0.0002 |    0.9934 |   0.0001 |         31.4622 |       30.0000 |
| RandomForest               |          0.9922 |         0.0010 |    0.9960 |   0.0005 |        575.3636 |       30.0000 |
| SGDClassifier              |          0.9865 |         0.0005 |    0.9931 |   0.0003 |         10.9777 |       30.0000 |
| SGDOneClassSVM             |          0.9801 |         0.0012 |    0.9897 |   0.0006 |          0.2015 |       30.0000 |

## Legenda
- **Accuracy/F1 Mean**: Valor médio across todas as execuções
- **Accuracy/F1 Std**: Desvio padrão (estabilidade)
- **Training_Time**: Tempo médio de treinamento (s)
- **Experiments**: Total de experimentos executados
