"""Tests for Welford-based corrected variance.

Verifies that:
1. Welford matches numpy.var() for batch data
2. Corrected implementation is dimension-agnostic
3. Ablation variants toggle correctly
"""

import numpy as np
import pytest
from teda_hd.algorithms.corrected import CorrectedMicroCluster, CorrectedMicroTEDAclus
from teda_hd.algorithms.variants import create_variant, create_all_variants


class TestWelfordMicroCluster:
    """Test Welford variance in the CorrectedMicroCluster."""

    def test_matches_numpy_var_2d(self):
        """Welford variance should match numpy.var(ddof=1) for 2D data."""
        rng = np.random.default_rng(42)
        points = rng.standard_normal((100, 2))

        mc = CorrectedMicroCluster(1, points[0])
        for x in points[1:]:
            mc.update(x)

        # Welford gives total variance (sum of per-dim variances / (n-1))
        # which equals np.sum(np.var(points, axis=0, ddof=1))... no.
        # Actually _var_sum accumulates dot(delta, delta2) which sums all dimensions.
        # variance = _var_sum / (n-1) = sample trace-variance / (n-1)... no.
        # Let me think: Welford accumulates sum of (x_k - mu_{k-1}) . (x_k - mu_k)
        # This equals sum_{i=1}^n (x_i - mu_n)^2 for each dimension summed.
        # So variance = sum over dims of var_per_dim * (n-1) / (n-1) = trace of cov.
        # Actually: _var_sum = sum_k dot(delta_pre, delta_post)
        # = sum_k sum_j (x_k_j - mu_{k-1}_j)(x_k_j - mu_k_j)
        # = sum_j sum_k (x_k_j - mu_{k-1}_j)(x_k_j - mu_k_j)
        # = sum_j S_j where S_j = sum of Welford terms for dimension j
        # And S_j / (n-1) = sample variance of dimension j
        # So variance = _var_sum / (n-1) = sum_j var_j = trace of sample covariance
        expected = np.sum(np.var(points, axis=0, ddof=1))
        assert mc.variance == pytest.approx(expected, rel=1e-6)

    def test_matches_numpy_var_17d(self):
        """Same test at d=17."""
        rng = np.random.default_rng(42)
        points = rng.standard_normal((200, 17))

        mc = CorrectedMicroCluster(1, points[0])
        for x in points[1:]:
            mc.update(x)

        expected = np.sum(np.var(points, axis=0, ddof=1))
        assert mc.variance == pytest.approx(expected, rel=1e-6)

    def test_variance_scales_linearly_with_d(self):
        """For isotropic Gaussian, Welford variance ≈ d (one unit per dimension)."""
        rng = np.random.default_rng(42)

        for d in [2, 5, 10, 17, 50]:
            points = rng.standard_normal((500, d))
            mc = CorrectedMicroCluster(1, points[0])
            for x in points[1:]:
                mc.update(x)

            # variance ≈ d * 1.0 = d (trace of identity covariance)
            assert mc.variance == pytest.approx(d, rel=0.2), (
                f"d={d}: Welford variance {mc.variance:.2f} should be ≈ {d}"
            )


class TestCorrectedVsOriginalAtD2:
    """At d=2, both should produce similar anomaly rates."""

    def test_d2_similar_anomaly_rates(self):
        """V0 (all flags off) and V7 (all on) should behave similarly at d=2.

        Using r0=100 so the n<3 test doesn't dominate — allows clusters
        to grow and the Chebyshev test (n>=3) to be the main mechanism.
        """
        rng = np.random.default_rng(42)
        points = rng.standard_normal((300, 2))

        v0 = create_variant("V0_original", r0=100.0)
        v7 = create_variant("V7_full_corrected", r0=100.0)

        anom_v0 = sum(1 for x in points if v0.process(x).is_anomaly)
        anom_v7 = sum(1 for x in points if v7.process(x).is_anomaly)

        rate_v0 = anom_v0 / len(points)
        rate_v7 = anom_v7 / len(points)

        # Both should have reasonable rates at d=2
        assert rate_v0 < 0.50, f"V0 at d=2: {rate_v0:.1%}"
        assert rate_v7 < 0.50, f"V7 at d=2: {rate_v7:.1%}"


class TestCorrectedFixesHighD:
    """The full correction should fix the high-d anomaly rate."""

    def test_d17_corrected_low_anomaly_rate(self):
        """V7 (full corrected) should have low anomaly rate at d=17."""
        rng = np.random.default_rng(42)
        points = rng.standard_normal((500, 17))

        v7 = create_variant("V7_full_corrected", r0=0.001)
        anom = sum(1 for x in points if v7.process(x).is_anomaly)
        rate = anom / len(points)

        assert rate < 0.25, (
            f"V7 at d=17: anomaly rate {rate:.1%} should be < 25% (corrected)"
        )

    def test_d17_original_high_anomaly_rate(self):
        """V0 (original flags) should have high anomaly rate at d=17."""
        rng = np.random.default_rng(42)
        points = rng.standard_normal((500, 17))

        v0 = create_variant("V0_original", r0=0.001)
        anom = sum(1 for x in points if v0.process(x).is_anomaly)
        rate = anom / len(points)

        assert rate > 0.30, (
            f"V0 at d=17: anomaly rate {rate:.1%} should be > 30% (bug present)"
        )


class TestAblationVariants:
    """Test that ablation variants are correctly configured."""

    def test_all_variants_created(self):
        """create_all_variants should return 8 variants."""
        variants = create_all_variants()
        assert len(variants) == 8

    def test_v0_all_flags_off(self):
        """V0 should have all flags off."""
        v0 = create_variant("V0_original")
        assert not v0.use_welford_variance
        assert not v0.use_consistent_eccentricity
        assert not v0.use_selective_update
        assert not v0.use_n1_guard
        assert not v0.use_n2_guard

    def test_v7_all_flags_on(self):
        """V7 should have all flags on."""
        v7 = create_variant("V7_full_corrected")
        assert v7.use_welford_variance
        assert v7.use_consistent_eccentricity
        assert v7.use_selective_update
        assert v7.use_n1_guard
        assert v7.use_n2_guard

    def test_v1_only_welford(self):
        """V1 should only have Welford variance on."""
        v1 = create_variant("V1_welford_var")
        assert v1.use_welford_variance
        assert not v1.use_consistent_eccentricity
        assert not v1.use_selective_update
        assert not v1.use_n1_guard
        assert not v1.use_n2_guard

    def test_unknown_variant_raises(self):
        """Unknown variant name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown variant"):
            create_variant("V99_nonexistent")
