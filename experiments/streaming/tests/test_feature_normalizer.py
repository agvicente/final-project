"""Tests for FeatureNormalizer.

Verifies:
1. Warmup phase returns raw samples unchanged (no leakage).
2. Z-score: post-fit data has mean ~ 0 and std ~ 1.
3. Min-max: post-fit data is in [0, 1].
4. Reproducibility: same sequence -> same output.
5. Edge case: constant feature does not produce NaN/inf.
6. Reset clears state and resumes warmup.
"""

import numpy as np
import pytest

from src.utils.feature_normalizer import FeatureNormalizer


class TestWarmup:
    def test_warmup_returns_raw(self):
        rng = np.random.default_rng(0)
        norm = FeatureNormalizer(mode="zscore", warmup_size=50)
        for _ in range(49):
            x = rng.standard_normal(5) * 10 + 100
            y = norm.update_and_transform(x)
            np.testing.assert_array_equal(y, x)
        assert not norm.is_fitted()

    def test_fit_at_warmup_boundary(self):
        rng = np.random.default_rng(0)
        norm = FeatureNormalizer(mode="zscore", warmup_size=50)
        for _ in range(50):
            x = rng.standard_normal(5)
            norm.update_and_transform(x)
        assert norm.is_fitted()


class TestZScore:
    def test_warmup_data_normalized_to_unit_stats(self):
        """The warmup data itself, when transformed via the fitted normalizer,
        should have mean=0 and std=1 by construction."""
        rng = np.random.default_rng(42)
        scale = 100.0
        offset = 50.0
        warmup = 500

        norm = FeatureNormalizer(mode="zscore", warmup_size=warmup)
        warmup_data = rng.standard_normal((warmup, 5)) * scale + offset

        # Fill warmup
        for x in warmup_data:
            norm.update_and_transform(x)
        assert norm.is_fitted()

        # Transform the same warmup data: mean and std should be exact
        out = np.array([norm.transform(x) for x in warmup_data])
        np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(out.std(axis=0, ddof=1), 1.0, rtol=1e-10)

    def test_fresh_data_approximately_normalized(self):
        """Fresh draws from the same distribution should be approximately
        unit-variance, with tolerance proportional to sampling noise."""
        rng = np.random.default_rng(42)
        scale = 100.0
        offset = 50.0
        warmup = 1000  # large warmup -> tighter SE on stats

        norm = FeatureNormalizer(mode="zscore", warmup_size=warmup)
        for _ in range(warmup):
            x = rng.standard_normal(5) * scale + offset
            norm.update_and_transform(x)

        n_test = 5000
        out = np.array([norm.transform(rng.standard_normal(5) * scale + offset)
                        for _ in range(n_test)])
        # SE of mean ~ 1/sqrt(warmup) ~ 0.03; std of fresh data ~ 1
        np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=0.1)
        np.testing.assert_allclose(out.std(axis=0, ddof=1), 1.0, rtol=0.1)


class TestMinMax:
    def test_minmax_in_unit_interval(self):
        rng = np.random.default_rng(0)
        norm = FeatureNormalizer(mode="minmax", warmup_size=200)
        # Use a long warmup to capture the range
        warmup_data = rng.uniform(-10, 10, (200, 4))
        for x in warmup_data:
            norm.update_and_transform(x)

        # Within-range samples should map to [0, 1]
        for x in warmup_data[:50]:
            y = norm.transform(x)
            assert (y >= 0).all() and (y <= 1).all()


class TestReproducibility:
    def test_same_seed_same_output(self):
        rng1 = np.random.default_rng(7)
        rng2 = np.random.default_rng(7)
        norm1 = FeatureNormalizer(mode="zscore", warmup_size=50)
        norm2 = FeatureNormalizer(mode="zscore", warmup_size=50)

        outs1, outs2 = [], []
        for _ in range(100):
            outs1.append(norm1.update_and_transform(rng1.standard_normal(3)))
            outs2.append(norm2.update_and_transform(rng2.standard_normal(3)))
        np.testing.assert_array_equal(np.array(outs1), np.array(outs2))


class TestEdgeCases:
    def test_constant_feature_no_nan_zscore(self):
        norm = FeatureNormalizer(mode="zscore", warmup_size=10)
        # Feature 1 is constant 0; feature 0 varies
        for i in range(10):
            x = np.array([float(i), 5.0])
            norm.update_and_transform(x)

        out = norm.transform(np.array([0.0, 5.0]))
        # Should not produce NaN/inf even with std=0 -> 1e-8 fallback
        assert np.all(np.isfinite(out))

    def test_constant_feature_no_nan_minmax(self):
        norm = FeatureNormalizer(mode="minmax", warmup_size=10)
        for i in range(10):
            x = np.array([float(i), 5.0])
            norm.update_and_transform(x)

        out = norm.transform(np.array([0.0, 5.0]))
        # Constant column produces 0, not NaN
        assert np.isfinite(out).all()
        assert out[1] == 0.0

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            FeatureNormalizer(mode="invalid")

    def test_warmup_size_too_small_raises(self):
        with pytest.raises(ValueError, match="warmup_size"):
            FeatureNormalizer(warmup_size=1)


class TestReset:
    def test_reset_resumes_warmup(self):
        rng = np.random.default_rng(0)
        norm = FeatureNormalizer(mode="zscore", warmup_size=10)
        for _ in range(10):
            norm.update_and_transform(rng.standard_normal(3))
        assert norm.is_fitted()

        norm.reset()
        assert not norm.is_fitted()

        # After reset, should accumulate again
        for _ in range(5):
            x = rng.standard_normal(3)
            y = norm.update_and_transform(x)
            np.testing.assert_array_equal(y, x)
        assert not norm.is_fitted()
