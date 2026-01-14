# Bayesian Statistics (Brodersen et al., 2010)

## Balanced Accuracy - Posterior Distributions

| algorithm                  |   ba_mean |   ba_median |   ba_std |   ba_ci_lower |   ba_ci_upper |   sensitivity_mean |   specificity_mean |   prob_ba_above_80 |   prob_ba_above_85 |   prob_ba_above_90 |   prob_ba_above_95 |
|:---------------------------|----------:|------------:|---------:|--------------:|--------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|
| SGDOneClassSVM             |    0.9560 |      0.9560 |   0.0001 |        0.9558 |        0.9562 |             0.9820 |             0.9299 |             1.0000 |             1.0000 |             1.0000 |             1.0000 |
| LocalOutlierFactor         |    0.9261 |      0.9261 |   0.0002 |        0.9257 |        0.9265 |             0.9833 |             0.8689 |             1.0000 |             1.0000 |             1.0000 |             0.0000 |
| GradientBoostingClassifier |    0.9183 |      0.9183 |   0.0002 |        0.9180 |        0.9187 |             0.9958 |             0.8408 |             1.0000 |             1.0000 |             1.0000 |             0.0000 |
| RandomForest               |    0.9175 |      0.9175 |   0.0002 |        0.9172 |        0.9178 |             0.9960 |             0.8391 |             1.0000 |             1.0000 |             1.0000 |             0.0000 |
| MLPClassifier              |    0.9073 |      0.9073 |   0.0002 |        0.9069 |        0.9077 |             0.9950 |             0.8196 |             1.0000 |             1.0000 |             1.0000 |             0.0000 |
| EllipticEnvelope           |    0.9017 |      0.9017 |   0.0002 |        0.9014 |        0.9020 |             0.9850 |             0.8185 |             1.0000 |             1.0000 |             1.0000 |             0.0000 |
| IsolationForest            |    0.8537 |      0.8537 |   0.0002 |        0.8534 |        0.8540 |             0.8852 |             0.8221 |             1.0000 |             1.0000 |             0.0000 |             0.0000 |
| LogisticRegression         |    0.8489 |      0.8489 |   0.0002 |        0.8486 |        0.8492 |             0.9938 |             0.7040 |             1.0000 |             0.0000 |             0.0000 |             0.0000 |
| LinearSVC                  |    0.8468 |      0.8468 |   0.0002 |        0.8464 |        0.8471 |             0.9939 |             0.6996 |             1.0000 |             0.0000 |             0.0000 |             0.0000 |
| SGDClassifier              |    0.8317 |      0.8317 |   0.0002 |        0.8314 |        0.8320 |             0.9937 |             0.6698 |             1.0000 |             0.0000 |             0.0000 |             0.0000 |

## Columns Description

- **ba_mean**: Mean of BA posterior distribution
- **ba_median**: Median of BA posterior
- **ba_std**: Standard deviation of BA posterior
- **ba_ci_lower/upper**: 95% Bayesian credibility interval
- **sensitivity_mean**: Mean sensitivity (TP rate)
- **specificity_mean**: Mean specificity (TN rate)
- **prob_ba_above_X**: Probability that BA > X threshold

Reference: Brodersen, K.H., et al. (2010). 'The balanced accuracy and its posterior distribution'. ICPR.
