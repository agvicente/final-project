"""Gaussian stream generator for synthetic experiments.

Generates Gaussian normal data with injected anomalies, supporting
isotropic or per-feature scaling for dimensional / regime sensitivity
analysis.
"""

from typing import Optional, Union

import numpy as np


class GaussianStreamGenerator:
    """Generate synthetic d-dimensional Gaussian data with injected anomalies.

    Normal points are drawn from N(0, normal_scale^2 * Sigma) where Sigma is
    either I_d (default) or diag(feature_scales^2).
    Anomalies are drawn from the same distribution scaled additionally by
    `anomaly_scale` -- so the contrast (anomaly_scale * normal_scale) is
    preserved relative to normal points.

    Args:
        d: Dimensionality of the feature space.
        n_normal: Number of normal samples (default: 950).
        n_anomalies: Number of anomaly samples (default: 50).
        anomaly_scale: Standard deviation multiplier for anomalies relative
            to normal points (default: 5.0).
        normal_scale: Isotropic standard deviation of normal points
            (default: 1.0). Multiplies the unit-variance N(0, I_d) base.
        feature_scales: Optional per-feature std array of shape (d,).
            If provided, overrides normal_scale isotropy: each dimension j
            is scaled by feature_scales[j]. Useful for testing detector
            robustness to heterogeneous feature scales.
        seed: Random seed for reproducibility (default: 42).

    Notes:
        Backward-compat: with default parameters (normal_scale=1.0,
        feature_scales=None), behavior matches the original isotropic
        N(0, I_d) generator.
    """

    def __init__(
        self,
        d: int,
        n_normal: int = 950,
        n_anomalies: int = 50,
        anomaly_scale: float = 5.0,
        normal_scale: float = 1.0,
        feature_scales: Optional[Union[np.ndarray, list]] = None,
        seed: int = 42,
    ):
        self.d = d
        self.n_normal = n_normal
        self.n_anomalies = n_anomalies
        self.anomaly_scale = anomaly_scale
        self.normal_scale = float(normal_scale)
        if feature_scales is not None:
            feature_scales = np.asarray(feature_scales, dtype=np.float64)
            if feature_scales.shape != (d,):
                raise ValueError(
                    f"feature_scales must have shape ({d},), got {feature_scales.shape}"
                )
        self.feature_scales = feature_scales
        self.seed = seed

    def _effective_scale(self) -> np.ndarray:
        """Return the (d,) array of effective per-feature std for normal points."""
        if self.feature_scales is not None:
            return self.feature_scales * self.normal_scale
        return np.full(self.d, self.normal_scale, dtype=np.float64)

    def generate(self):
        """Generate synthetic data stream with labeled anomalies.

        Returns:
            Tuple of (X, y_true) where:
                X: ndarray of shape (n_normal + n_anomalies, d).
                y_true: ndarray of shape (n_total,) -- 0 for normal, 1 anomaly.
        """
        rng = np.random.default_rng(self.seed)
        scale = self._effective_scale()  # shape (d,)

        # Normal samples
        X_normal = rng.standard_normal((self.n_normal, self.d)) * scale
        y_normal = np.zeros(self.n_normal, dtype=int)

        # Anomalies: same distribution shape, scaled additionally by anomaly_scale
        X_anomaly = (
            rng.standard_normal((self.n_anomalies, self.d))
            * scale
            * self.anomaly_scale
        )
        y_anomaly = np.ones(self.n_anomalies, dtype=int)

        # Concatenate and shuffle (preserves seed determinism via single rng)
        X = np.concatenate([X_normal, X_anomaly], axis=0)
        y = np.concatenate([y_normal, y_anomaly], axis=0)

        shuffle_idx = rng.permutation(len(X))
        X = X[shuffle_idx]
        y = y[shuffle_idx]

        return X, y
