# Probabilistic Comparison Matrix (Bayesian)

## P(Row Algorithm > Column Algorithm)

|                            |   LogisticRegression |   EllipticEnvelope |   RandomForest |   MLPClassifier |   SGDOneClassSVM |   SGDClassifier |   GradientBoostingClassifier |   LocalOutlierFactor |   LinearSVC |   IsolationForest |
|:---------------------------|---------------------:|-------------------:|---------------:|----------------:|-----------------:|----------------:|-----------------------------:|---------------------:|------------:|------------------:|
| LogisticRegression         |                0.500 |              0.000 |          0.000 |           0.000 |            0.000 |           1.000 |                        0.000 |                0.000 |       1.000 |             0.000 |
| EllipticEnvelope           |                1.000 |              0.500 |          0.000 |           0.000 |            0.000 |           1.000 |                        0.000 |                0.000 |       1.000 |             1.000 |
| RandomForest               |                1.000 |              1.000 |          0.500 |           1.000 |            0.000 |           1.000 |                        0.000 |                0.000 |       1.000 |             1.000 |
| MLPClassifier              |                1.000 |              1.000 |          0.000 |           0.500 |            0.000 |           1.000 |                        0.000 |                0.000 |       1.000 |             1.000 |
| SGDOneClassSVM             |                1.000 |              1.000 |          1.000 |           1.000 |            0.500 |           1.000 |                        1.000 |                1.000 |       1.000 |             1.000 |
| SGDClassifier              |                0.000 |              0.000 |          0.000 |           0.000 |            0.000 |           0.500 |                        0.000 |                0.000 |       0.000 |             0.000 |
| GradientBoostingClassifier |                1.000 |              1.000 |          1.000 |           1.000 |            0.000 |           1.000 |                        0.500 |                0.000 |       1.000 |             1.000 |
| LocalOutlierFactor         |                1.000 |              1.000 |          1.000 |           1.000 |            0.000 |           1.000 |                        1.000 |                0.500 |       1.000 |             1.000 |
| LinearSVC                  |                0.000 |              0.000 |          0.000 |           0.000 |            0.000 |           1.000 |                        0.000 |                0.000 |       0.500 |             0.000 |
| IsolationForest            |                1.000 |              0.000 |          0.000 |           0.000 |            0.000 |           1.000 |                        0.000 |                0.000 |       1.000 |             0.500 |

## Interpretation

- **P > 0.95**: Strong evidence that Row algorithm is better than Column
- **P < 0.05**: Strong evidence that Column algorithm is better than Row
- **0.05 < P < 0.95**: Inconclusive or small difference
- **P ≈ 0.5**: No difference between algorithms

Reference: Brodersen et al. (2010) - Bayesian posterior distributions
