"""Tests for GaussianStreamGenerator extension (normal_scale, feature_scales).

Verifies:
1. Default behavior is unchanged (backward-compat).
2. normal_scale=10 produces normal points with sample variance ~100.
3. Anomalies preserve contrast (variance ~ (anomaly_scale * normal_scale)^2).
4. Same seed yields identical output (reproducibility).
5. feature_scales overrides isotropy (per-dimension control).
6. Invalid feature_scales shape raises.
"""

import numpy as np
import pytest

from teda_hd.generators.gaussian import GaussianStreamGenerator


class TestBackwardCompat:
    """Default parameters reproduce original isotropic N(0, I_d) behavior."""

    def test_default_normal_variance_is_one(self):
        gen = GaussianStreamGenerator(d=10, n_normal=10_000, n_anomalies=0, seed=0)
        X, y = gen.generate()
        # Sample variance of normal points should be ~1 per dimension
        var_per_dim = np.var(X[y == 0], axis=0, ddof=1)
        np.testing.assert_allclose(var_per_dim.mean(), 1.0, atol=0.05)

    def test_default_shape_and_labels(self):
        gen = GaussianStreamGenerator(d=5, n_normal=100, n_anomalies=20, seed=1)
        X, y = gen.generate()
        assert X.shape == (120, 5)
        assert y.shape == (120,)
        assert np.sum(y == 0) == 100
        assert np.sum(y == 1) == 20


class TestNormalScale:
    """normal_scale multiplies the std of normal points isotropically."""

    @pytest.mark.parametrize("scale", [0.1, 1.0, 10.0, 100.0])
    def test_normal_variance_scales_quadratically(self, scale):
        gen = GaussianStreamGenerator(
            d=10,
            n_normal=10_000,
            n_anomalies=0,
            normal_scale=scale,
            seed=42,
        )
        X, y = gen.generate()
        # Expected var per dim = scale^2; allow 10% tolerance for sampling noise
        var_per_dim = np.var(X[y == 0], axis=0, ddof=1)
        expected = scale**2
        np.testing.assert_allclose(var_per_dim.mean(), expected, rtol=0.05)


class TestAnomalyContrast:
    """Anomaly variance equals (anomaly_scale * normal_scale)^2."""

    def test_anomaly_variance_preserves_contrast(self):
        gen = GaussianStreamGenerator(
            d=10,
            n_normal=0,
            n_anomalies=10_000,
            anomaly_scale=5.0,
            normal_scale=10.0,
            seed=42,
        )
        X, y = gen.generate()
        var_anom = np.var(X[y == 1], axis=0, ddof=1)
        expected = (5.0 * 10.0) ** 2  # = 2500
        np.testing.assert_allclose(var_anom.mean(), expected, rtol=0.05)

    def test_relative_contrast_5sigma(self):
        """At equal seeds, ratio of std(anomaly) / std(normal) ≈ anomaly_scale."""
        gen = GaussianStreamGenerator(
            d=10,
            n_normal=10_000,
            n_anomalies=10_000,
            anomaly_scale=5.0,
            normal_scale=2.5,
            seed=42,
        )
        X, y = gen.generate()
        std_normal = np.std(X[y == 0], axis=0, ddof=1).mean()
        std_anom = np.std(X[y == 1], axis=0, ddof=1).mean()
        np.testing.assert_allclose(std_anom / std_normal, 5.0, rtol=0.05)


class TestReproducibility:
    """Same seed yields bit-exact same data."""

    def test_seed_determinism(self):
        kw = dict(d=17, n_normal=950, n_anomalies=50, normal_scale=3.0, seed=7)
        gen1 = GaussianStreamGenerator(**kw)
        gen2 = GaussianStreamGenerator(**kw)
        X1, y1 = gen1.generate()
        X2, y2 = gen2.generate()
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_differ(self):
        kw = dict(d=17, n_normal=950, n_anomalies=50)
        X_a, _ = GaussianStreamGenerator(seed=1, **kw).generate()
        X_b, _ = GaussianStreamGenerator(seed=2, **kw).generate()
        # Should NOT be identical
        assert not np.array_equal(X_a, X_b)


class TestFeatureScales:
    """feature_scales overrides isotropy."""

    def test_per_dimension_scaling(self):
        d = 5
        scales = np.array([1.0, 2.0, 5.0, 10.0, 0.5])
        gen = GaussianStreamGenerator(
            d=d,
            n_normal=20_000,
            n_anomalies=0,
            feature_scales=scales,
            seed=42,
        )
        X, y = gen.generate()
        var_per_dim = np.var(X[y == 0], axis=0, ddof=1)
        expected = scales**2
        np.testing.assert_allclose(var_per_dim, expected, rtol=0.05)

    def test_feature_scales_combined_with_normal_scale(self):
        d = 5
        scales = np.array([1.0, 2.0, 5.0, 10.0, 0.5])
        normal_scale = 3.0
        gen = GaussianStreamGenerator(
            d=d,
            n_normal=20_000,
            n_anomalies=0,
            feature_scales=scales,
            normal_scale=normal_scale,
            seed=42,
        )
        X, y = gen.generate()
        var_per_dim = np.var(X[y == 0], axis=0, ddof=1)
        expected = (scales * normal_scale) ** 2
        np.testing.assert_allclose(var_per_dim, expected, rtol=0.05)

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="feature_scales must have shape"):
            GaussianStreamGenerator(d=5, feature_scales=np.array([1.0, 2.0]))
