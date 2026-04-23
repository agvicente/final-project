"""Gaussian stream generator for synthetic experiments.

Generates isotropic Gaussian normal data with injected anomalies
at configurable scale for dimensional sensitivity analysis.
"""

import numpy as np


class GaussianStreamGenerator:
    """Generate synthetic d-dimensional Gaussian data with injected anomalies.

    Normal points are drawn from N(0, I_d).
    Anomalies are drawn from N(0, anomaly_scale^2 * I_d) -- outliers at anomaly_scale sigma.

    Args:
        d: Dimensionality of the feature space.
        n_normal: Number of normal samples (default: 950).
        n_anomalies: Number of anomaly samples (default: 50).
        anomaly_scale: Standard deviation multiplier for anomalies (default: 5.0).
        seed: Random seed for reproducibility (default: 42).
    """

    def __init__(
        self,
        d: int,
        n_normal: int = 950,
        n_anomalies: int = 50,
        anomaly_scale: float = 5.0,
        seed: int = 42,
    ):
        self.d = d
        self.n_normal = n_normal
        self.n_anomalies = n_anomalies
        self.anomaly_scale = anomaly_scale
        self.seed = seed

    def generate(self):
        """Generate synthetic data stream with labeled anomalies.

        Returns:
            Tuple of (X, y_true) where:
                X: ndarray of shape (n_normal + n_anomalies, d) -- feature vectors
                y_true: ndarray of shape (n_total,) -- 0 for normal, 1 for anomaly
        """
        rng = np.random.default_rng(self.seed)

        # Normal samples: N(0, I_d)
        X_normal = rng.standard_normal((self.n_normal, self.d))
        y_normal = np.zeros(self.n_normal, dtype=int)

        # Anomalies: N(0, anomaly_scale^2 * I_d)
        X_anomaly = rng.standard_normal((self.n_anomalies, self.d)) * self.anomaly_scale
        y_anomaly = np.ones(self.n_anomalies, dtype=int)

        # Concatenate and shuffle
        X = np.concatenate([X_normal, X_anomaly], axis=0)
        y = np.concatenate([y_normal, y_anomaly], axis=0)

        shuffle_idx = rng.permutation(len(X))
        X = X[shuffle_idx]
        y = y[shuffle_idx]

        return X, y
