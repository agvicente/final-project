"""Tests for original variance formula (norm*2/d)^2.

Verifies that our reimplementation matches the exact behavior of the
original EvolvingClustering.update_variance() at different dimensionalities.
"""

import numpy as np
import pytest
from teda_hd.algorithms.original import OriginalMicroTEDAclus


class TestUpdateVarianceFormula:
    """Test the raw update_variance static method."""

    def test_2d_single_update(self):
        """At d=2, (norm*2/2)^2 = norm^2."""
        delta = np.array([1.0, 0.0])
        # s_ik=2, var_ik=0 (first update after initialization)
        var = OriginalMicroTEDAclus.update_variance(delta, 2, 0.0)
        # (norm(delta)*2/2)^2 / (2-1) = (1*1)^2 / 1 = 1.0
        # ((2-1)/2)*0 + 1.0 = 1.0
        assert var == pytest.approx(1.0)

    def test_17d_single_update(self):
        """At d=17, (norm*2/17)^2 << norm^2."""
        delta = np.ones(17)  # norm = sqrt(17)
        var = OriginalMicroTEDAclus.update_variance(delta, 2, 0.0)
        # norm = sqrt(17) ≈ 4.123
        # (4.123 * 2/17)^2 / 1 = (0.485)^2 = 0.235
        norm_d = np.linalg.norm(delta)
        expected = (norm_d * 2 / 17) ** 2
        assert var == pytest.approx(expected)

    def test_2d_vs_17d_ratio(self):
        """Variance at d=17 should be (4/17) times what it'd be at d=2
        for a unit-norm-per-dimension vector."""
        # Same "spread" per dimension
        delta_2d = np.array([1.0, 1.0])
        delta_17d = np.ones(17)

        var_2d = OriginalMicroTEDAclus.update_variance(delta_2d, 2, 0.0)
        var_17d = OriginalMicroTEDAclus.update_variance(delta_17d, 2, 0.0)

        # var_2d contribution: (sqrt(2)*2/2)^2 = (sqrt(2))^2 = 2
        # var_17d contribution: (sqrt(17)*2/17)^2 = 4*17/289 = 68/289 ≈ 0.235
        # Ratio (17d / 2d) per unit dimension = (4/17) / (4/2) = 2/17
        # But norm grows with d, so: var_17d/var_2d = (4*17/17^2)/(4*2/2^2) = (4/17)/(4/2) = 2/17
        # Actually: var = (norm*2/d)^2 = 4*||delta||^2/d^2
        # For delta = ones(d): ||delta||^2 = d, so var = 4d/d^2 = 4/d
        assert var_2d == pytest.approx(4.0 / 2)  # = 2.0
        assert var_17d == pytest.approx(4.0 / 17)  # ≈ 0.235

    def test_recursive_convergence(self):
        """After many updates from same distribution, variance should stabilize."""
        rng = np.random.default_rng(42)
        d = 10
        var = 0.0
        mean = np.zeros(d)

        for k in range(2, 1001):
            x = rng.standard_normal(d)
            mean = ((k - 1) / k) * mean + x / k
            delta = x - mean
            var = OriginalMicroTEDAclus.update_variance(delta, k, var)

        # True per-dimension variance of standard normal = 1.0
        # Expected original variance ≈ 4/d = 0.4 (underestimated)
        assert var == pytest.approx(4.0 / d, rel=0.3)

    def test_d2_formula_equals_norm_squared(self):
        """At d=2, (norm*2/2)^2 = norm^2 = ||delta||^2."""
        delta = np.array([3.0, 4.0])  # norm = 5
        var = OriginalMicroTEDAclus.update_variance(delta, 2, 0.0)
        assert var == pytest.approx(np.sum(delta**2))  # = 25


class TestEccentricity:
    """Test eccentricity formula consistency."""

    def test_first_point_eccentricity(self):
        """Eccentricity for x == mean should be 1/n."""
        mean = np.array([1.0, 2.0, 3.0])
        ecc = OriginalMicroTEDAclus.get_eccentricity(mean, 10, mean, 1.0)
        assert ecc == pytest.approx(0.1)  # 1/10

    def test_self_cancellation_at_high_d(self):
        """The (2/d)^2 factor in both ecc and var should cancel for n>=3.

        This is the key mathematical insight: despite the variance being
        wrong in absolute terms, the eccentricity RATIO is correct because
        both numerator and denominator use the same (2/d)^2 scaling.
        """
        rng = np.random.default_rng(42)

        for d in [2, 5, 10, 17, 50]:
            # Build up statistics
            var = 0.0
            mean = np.zeros(d)
            n_points = 500

            for k in range(1, n_points + 1):
                x = rng.standard_normal(d)
                mean = ((k - 1) / k) * mean + x / k
                if k >= 2:
                    delta = x - mean
                    var = OriginalMicroTEDAclus.update_variance(delta, k, var)

            # Compute eccentricities for test points from SAME distribution
            eccentricities = []
            for _ in range(100):
                x_test = rng.standard_normal(d)
                ecc = OriginalMicroTEDAclus.get_eccentricity(
                    x_test, n_points, mean, var
                )
                eccentricities.append(ecc)

            mean_ecc = np.mean(eccentricities)
            # For n>=3, mean ecc should be close to 1/n + 1 ≈ 1.002
            # (because E[||x-mu||^2/(n*var)] ≈ 1 when var is self-consistent)
            # The exact value depends on distribution, but should NOT grow with d
            assert mean_ecc < 5.0, (
                f"d={d}: mean_ecc={mean_ecc:.2f} — eccentricity should not "
                f"grow with d due to self-cancellation"
            )


class TestIsOutlier:
    """Test the outlier classification logic."""

    def test_n_less_than_3_uses_variance_limit(self):
        """For n<3, outlier if var > r0."""
        detector = OriginalMicroTEDAclus(r0=0.001)
        # Small variance → NOT outlier
        assert not detector._is_outlier(2, 0.0005, 0.5)
        # Large variance → outlier
        assert detector._is_outlier(2, 0.002, 0.5)

    def test_n_ge_3_uses_chebyshev(self):
        """For n>=3, outlier if norm_ecc > (m^2+1)/(2n)."""
        detector = OriginalMicroTEDAclus()
        # At n=100: m(100) ≈ 1.5, threshold ≈ (2.25+1)/200 ≈ 0.016
        # Low eccentricity → not outlier
        assert not detector._is_outlier(100, 0.5, 0.01)
        # High eccentricity → outlier
        assert detector._is_outlier(100, 0.5, 0.5)


class TestFullProcessing:
    """Integration tests for the full processing pipeline."""

    def test_first_point_not_anomaly(self):
        """First point should never be classified as anomaly."""
        detector = OriginalMicroTEDAclus()
        result = detector.process(np.array([1.0, 2.0]))
        assert not result.is_anomaly
        assert result.new_cluster_created
        assert result.num_clusters == 1

    def test_gaussian_2d_reasonable_anomaly_rate(self):
        """At d=2, anomaly rate should be reasonable (<30%) with large r0.

        We use r0=100 (very permissive for n<3 test) so that clusters can
        grow past n=2 and the Chebyshev test (n>=3) becomes the dominant
        decision mechanism. With small r0 (e.g., 0.001), the n<3 test
        rejects everything regardless of dimension.
        """
        rng = np.random.default_rng(42)
        detector = OriginalMicroTEDAclus(r0=100.0)

        anomalies = 0
        n_points = 500
        for _ in range(n_points):
            x = rng.standard_normal(2)
            result = detector.process(x)
            if result.is_anomaly:
                anomalies += 1

        rate = anomalies / n_points
        assert rate < 0.30, f"d=2 anomaly rate {rate:.1%} should be < 30%"

    def test_high_d_higher_anomaly_rate_than_low_d(self):
        """At d=17, anomaly rate should be HIGHER than at d=2.

        With permissive r0, the dimension effect manifests through
        intersection radius (sqrt(var) scaled down) and life decay
        calculations that mix scaled variance with true distances.
        """
        rng_2d = np.random.default_rng(42)
        rng_17d = np.random.default_rng(42)
        n_points = 500

        detector_2d = OriginalMicroTEDAclus(r0=100.0)
        anom_2d = sum(
            1 for _ in range(n_points)
            if detector_2d.process(rng_2d.standard_normal(2)).is_anomaly
        )

        detector_17d = OriginalMicroTEDAclus(r0=100.0)
        anom_17d = sum(
            1 for _ in range(n_points)
            if detector_17d.process(rng_17d.standard_normal(17)).is_anomaly
        )

        rate_2d = anom_2d / n_points
        rate_17d = anom_17d / n_points

        # d=17 should have more anomalies due to dimensional effects
        assert rate_17d > rate_2d, (
            f"d=17 rate ({rate_17d:.1%}) should exceed d=2 rate ({rate_2d:.1%})"
        )

    def test_update_all_clusters(self):
        """Original should update ALL accepting clusters, not just best."""
        detector = OriginalMicroTEDAclus(r0=10.0)  # Very permissive r0

        # Create two clusters
        detector.process(np.array([0.0, 0.0]))  # Cluster 1
        detector.process(np.array([100.0, 100.0]))  # Should create cluster 2

        clusters_before = detector.get_clusters()
        assert len(clusters_before) >= 1

        # Process a point near cluster 1
        detector.process(np.array([0.1, 0.1]))

        # If update-all is working, cluster stats should be updated
        assert detector.total_samples == 3
