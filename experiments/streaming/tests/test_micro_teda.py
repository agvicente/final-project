"""
Unit tests for MicroTEDAclus.

Tests cover:
1. MicroCluster basic functionality
2. Dynamic threshold m(k)
3. Chebyshev acceptance/rejection
4. MicroTEDAclus orchestrator
5. Contamination resistance (key improvement over basic TEDA)

Run with: pytest tests/test_micro_teda.py -v
"""

import pytest
import numpy as np
from src.detector.micro_teda import MicroCluster, MicroTEDAclus, MicroTEDAResult


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def micro_cluster():
    """Create a fresh MicroCluster."""
    return MicroCluster(cluster_id=0, initial_point=np.array([5.0, 5.0]))


@pytest.fixture
def trained_micro_cluster():
    """Create a MicroCluster trained with 10 points around [5, 5].

    After training, the cluster should have:
    - n=10
    - mean close to [5, 5]
    - variance ~0.25 (from std=0.5)
    """
    mc = MicroCluster(cluster_id=0, initial_point=np.array([5.0, 5.0]))
    np.random.seed(42)
    for _ in range(9):  # Already has 1 point
        point = np.random.randn(2) * 0.5 + [5, 5]
        mc.update(point)
    return mc


@pytest.fixture
def detector():
    """Create a fresh MicroTEDAclus detector.

    Note: r0=0.1 is appropriate for non-normalized data with variance ~0.25.
    For normalized data (mean=0, std=1), use r0=0.001 as in the paper.
    """
    return MicroTEDAclus(r0=0.1, min_samples=3)


@pytest.fixture
def trained_detector():
    """Create a MicroTEDAclus trained with 10 normal samples around [5, 5].

    Note: r0=0.1 allows proper cluster formation for data with variance ~0.25.
    """
    detector = MicroTEDAclus(r0=0.1, min_samples=3)
    np.random.seed(42)
    normal_data = np.random.randn(10, 2) * 0.5 + [5, 5]
    for point in normal_data:
        detector.process(point)
    return detector


# ============================================================
# MICRO CLUSTER TESTS
# ============================================================

class TestMicroClusterBasic:
    """Tests for MicroCluster basic functionality."""

    def test_initialization(self, micro_cluster):
        """MicroCluster initializes correctly."""
        assert micro_cluster.cluster_id == 0
        assert micro_cluster.n == 1
        np.testing.assert_array_equal(micro_cluster.mean, [5.0, 5.0])
        assert micro_cluster.variance == 0.0

    def test_update_mean(self, micro_cluster):
        """Mean updates correctly."""
        micro_cluster.update(np.array([7.0, 7.0]))

        # Mean should be [6.0, 6.0]
        np.testing.assert_array_almost_equal(micro_cluster.mean, [6.0, 6.0])
        assert micro_cluster.n == 2

    def test_update_variance(self, micro_cluster):
        """Variance updates correctly."""
        micro_cluster.update(np.array([7.0, 7.0]))
        micro_cluster.update(np.array([3.0, 3.0]))

        assert micro_cluster.variance > 0
        assert micro_cluster.n == 3


class TestDynamicThreshold:
    """Tests for dynamic threshold m(k)."""

    def test_m_small_k(self):
        """For small k, m(k) is close to 1 (restrictive)."""
        mc = MicroCluster(0, np.array([0.0, 0.0]))
        # n=1
        m = mc.dynamic_m()

        # m(1) = 3 / (1 + e^{-0.007(1-100)}) = 3 / (1 + e^{0.693}) ≈ 1.0
        assert m < 1.5  # Should be restrictive

    def test_m_large_k(self):
        """For large k, m(k) approaches 3 (permissive)."""
        mc = MicroCluster(0, np.array([0.0, 0.0]))

        # Simulate large cluster
        mc.n = 1000
        m = mc.dynamic_m()

        # m(1000) ≈ 3
        assert m > 2.9
        assert m <= 3.0

    def test_m_medium_k(self):
        """For medium k, m(k) is between 1 and 3."""
        mc = MicroCluster(0, np.array([0.0, 0.0]))
        mc.n = 100
        m = mc.dynamic_m()

        # m(100) = 3 / (1 + e^0) = 3/2 = 1.5
        assert 1.4 < m < 1.6


class TestChebyshevAcceptance:
    """Tests for Chebyshev acceptance/rejection."""

    def test_accepts_typical_point(self, trained_micro_cluster):
        """Typical point is accepted."""
        typical = np.array([5.1, 4.9])
        assert trained_micro_cluster.chebyshev_accepts(typical)

    def test_rejects_outlier(self, trained_micro_cluster):
        """Outlier is rejected."""
        outlier = np.array([100.0, 100.0])
        assert not trained_micro_cluster.chebyshev_accepts(outlier)

    def test_threshold_decreases_with_n(self):
        """Threshold decreases as cluster grows."""
        mc = MicroCluster(0, np.array([0.0, 0.0]))

        thresholds = []
        for i in range(20):
            mc.update(np.array([float(i % 3), float(i % 3)]))
            t = mc.chebyshev_threshold()
            if t != float('inf'):
                thresholds.append(t)

        # Threshold should decrease
        assert thresholds[-1] < thresholds[0]


class TestMicroClusterTypicality:
    """Tests for typicality calculation."""

    def test_typicality_typical_point(self, trained_micro_cluster):
        """Typical point has high typicality."""
        typical = np.array([5.0, 5.0])
        typ = trained_micro_cluster.calculate_typicality(typical)

        assert typ > 0.5

    def test_typicality_outlier(self, trained_micro_cluster):
        """Outlier has low typicality."""
        outlier = np.array([100.0, 100.0])
        typ = trained_micro_cluster.calculate_typicality(outlier)

        assert typ < 0


# ============================================================
# MICRO TEDA CLUS TESTS
# ============================================================

class TestMicroTEDAclusBasic:
    """Tests for MicroTEDAclus basic functionality."""

    def test_first_point_creates_cluster(self, detector):
        """First point creates first cluster."""
        result = detector.process(np.array([1.0, 2.0]))

        assert result.num_clusters == 1
        assert result.new_cluster_created == True
        assert result.is_anomaly == False  # First point never anomaly
        assert detector.total_samples == 1

    def test_typical_point_joins_cluster(self, trained_detector):
        """Typical point joins existing cluster."""
        initial_clusters = len(trained_detector.micro_clusters)
        result = trained_detector.process(np.array([5.0, 5.0]))

        assert result.new_cluster_created == False
        assert result.is_anomaly == False
        assert len(trained_detector.micro_clusters) == initial_clusters

    def test_outlier_creates_new_cluster(self, trained_detector):
        """Outlier creates new cluster."""
        initial_clusters = len(trained_detector.micro_clusters)
        result = trained_detector.process(np.array([100.0, 100.0]))

        assert result.new_cluster_created == True
        assert result.is_anomaly == True
        assert len(trained_detector.micro_clusters) == initial_clusters + 1


class TestContaminationResistance:
    """Tests proving MicroTEDAclus resists contamination.

    This is the KEY advantage over basic TEDA.
    """

    def test_outlier_does_not_contaminate_existing_cluster(self, trained_detector):
        """Outlier does NOT contaminate existing cluster statistics.

        In basic TEDA, outlier would shift mean and explode variance.
        In MicroTEDAclus, outlier creates separate cluster.
        """
        # Get original cluster stats
        original_cluster = trained_detector.micro_clusters[0]
        mean_before = original_cluster.mean.copy()
        var_before = original_cluster.variance
        n_before = original_cluster.n

        # Add extreme outlier
        trained_detector.process(np.array([100.0, 100.0]))

        # Original cluster should be UNCHANGED
        np.testing.assert_array_almost_equal(
            original_cluster.mean, mean_before,
            err_msg="Outlier contaminated the mean!"
        )
        assert original_cluster.variance == var_before, "Outlier contaminated variance!"
        assert original_cluster.n == n_before, "Outlier was added to wrong cluster!"

    def test_second_outlier_still_detected(self, trained_detector):
        """Second outlier is still detected (unlike basic TEDA).

        In basic TEDA, after first outlier contaminates stats,
        second outlier might not be detected.
        """
        # First outlier
        result1 = trained_detector.process(np.array([100.0, 100.0]))
        assert result1.is_anomaly == True

        # Second outlier in opposite direction
        result2 = trained_detector.process(np.array([-100.0, -100.0]))

        # Should STILL be detected as anomaly!
        assert result2.is_anomaly == True
        assert result2.new_cluster_created == True

    def test_normal_point_after_outliers_still_normal(self, trained_detector):
        """Normal point after outliers is still classified as normal.

        In basic TEDA, after contamination, normal points might
        appear as anomalies. MicroTEDAclus preserves normal clusters.
        """
        # Record which clusters existed before outliers
        normal_cluster_ids = {mc.cluster_id for mc in trained_detector.micro_clusters}

        # Add several outliers
        trained_detector.process(np.array([100.0, 100.0]))
        trained_detector.process(np.array([-100.0, -100.0]))
        trained_detector.process(np.array([50.0, -50.0]))

        # Normal point should STILL be normal
        result = trained_detector.process(np.array([5.0, 5.0]))

        assert result.is_anomaly == False
        assert result.new_cluster_created == False
        # Should join one of the original normal clusters (not outlier clusters)
        assert result.cluster_id in normal_cluster_ids

    def test_multiple_outliers_create_separate_clusters(self, trained_detector):
        """Different outliers create separate clusters."""
        initial_clusters = len(trained_detector.micro_clusters)

        # Three different outliers
        trained_detector.process(np.array([100.0, 100.0]))
        trained_detector.process(np.array([-100.0, -100.0]))
        trained_detector.process(np.array([100.0, -100.0]))

        # Should have 3 new clusters (4 total)
        assert len(trained_detector.micro_clusters) == initial_clusters + 3


class TestClusterAssignment:
    """Tests for cluster assignment logic."""

    def test_point_assigned_to_highest_typicality(self):
        """Point is assigned to cluster with highest typicality."""
        detector = MicroTEDAclus(r0=0.1, min_samples=3)

        # Create two well-separated clusters
        for _ in range(10):
            detector.process(np.array([0.0, 0.0]))

        for _ in range(10):
            detector.process(np.array([100.0, 100.0]))

        # Point closer to first cluster should go to cluster with highest typicality
        result = detector.process(np.array([1.0, 1.0]))

        # Verify it went to the cluster with highest typicality
        if result.cluster_typicalities:
            best_cluster = max(result.cluster_typicalities,
                               key=result.cluster_typicalities.get)
            assert result.cluster_id == best_cluster

    def test_typicalities_returned_for_all_clusters(self, trained_detector):
        """Result includes typicalities for all clusters."""
        initial_cluster_count = len(trained_detector.micro_clusters)

        # Create additional cluster with outlier
        trained_detector.process(np.array([100.0, 100.0]))

        result = trained_detector.process(np.array([5.0, 5.0]))

        assert result.cluster_typicalities is not None
        # Should have typicalities for all clusters (initial + outlier cluster)
        assert len(result.cluster_typicalities) == initial_cluster_count + 1


class TestPredictWithoutUpdate:
    """Tests for predict() method."""

    def test_predict_does_not_change_state(self, trained_detector):
        """Predict does not change detector state."""
        clusters_before = len(trained_detector.micro_clusters)
        samples_before = trained_detector.total_samples

        trained_detector.predict(np.array([100.0, 100.0]))

        assert len(trained_detector.micro_clusters) == clusters_before
        assert trained_detector.total_samples == samples_before

    def test_predict_returns_valid_result(self, trained_detector):
        """Predict returns valid MicroTEDAResult."""
        result = trained_detector.predict(np.array([5.0, 5.0]))

        assert isinstance(result, MicroTEDAResult)
        assert result.is_anomaly == False


class TestEdgeCases:
    """Tests for edge cases."""

    def test_reset(self, trained_detector):
        """Reset clears all state."""
        trained_detector.reset()

        assert len(trained_detector.micro_clusters) == 0
        assert trained_detector.total_samples == 0
        assert trained_detector.anomaly_count == 0

    def test_single_feature(self):
        """Works with single feature (1D)."""
        detector = MicroTEDAclus()

        for i in range(10):
            result = detector.process(np.array([float(i)]))
            assert isinstance(result, MicroTEDAResult)

    def test_many_features(self):
        """Works with many features (high dimensional)."""
        detector = MicroTEDAclus()

        for i in range(10):
            result = detector.process(np.random.randn(100))
            assert isinstance(result, MicroTEDAResult)

    def test_get_statistics(self, trained_detector):
        """get_statistics returns valid dict."""
        stats = trained_detector.get_statistics()

        assert 'total_samples' in stats
        assert 'num_clusters' in stats
        assert 'anomaly_count' in stats
        assert 'clusters' in stats

    def test_get_cluster_centers(self, trained_detector):
        """get_cluster_centers returns array of centers."""
        centers = trained_detector.get_cluster_centers()

        assert len(centers) == len(trained_detector.micro_clusters)

    def test_get_cluster_sizes(self, trained_detector):
        """get_cluster_sizes returns dict of sizes."""
        sizes = trained_detector.get_cluster_sizes()

        assert len(sizes) == len(trained_detector.micro_clusters)
        assert sum(sizes.values()) == trained_detector.total_samples


class TestMicroTEDAResult:
    """Tests for MicroTEDAResult dataclass."""

    def test_to_dict(self, detector):
        """MicroTEDAResult converts to dict correctly."""
        result = detector.process(np.array([1.0, 2.0]))
        d = result.to_dict()

        assert isinstance(d, dict)
        assert d['eccentricity'] == result.eccentricity
        assert d['cluster_id'] == result.cluster_id
        assert d['is_anomaly'] == result.is_anomaly
        assert d['new_cluster_created'] == result.new_cluster_created


# ============================================================
# COMPARISON WITH BASIC TEDA
# ============================================================

class TestComparisonWithBasicTEDA:
    """Tests comparing MicroTEDAclus with basic TEDA.

    These tests demonstrate the key advantages of MicroTEDAclus.
    """

    def test_contamination_comparison(self):
        """Direct comparison of contamination behavior.

        This test shows why MicroTEDAclus is better for IDS.
        """
        from src.detector.teda import TEDADetector

        # Setup both detectors with same data
        np.random.seed(42)
        normal_data = np.random.randn(10, 2) * 0.5 + [5, 5]

        basic_teda = TEDADetector(m=3.0, min_samples=3)
        micro_teda = MicroTEDAclus(r0=0.1, min_samples=3)  # r0=0.1 for non-normalized data

        for point in normal_data:
            basic_teda.update(point)
            micro_teda.process(point)

        # Record stats before outlier
        basic_var_before = basic_teda.variance
        # With r0=0.1, all 10 points should be in cluster 0
        micro_var_before = micro_teda.micro_clusters[0].variance

        # Add outlier
        outlier = np.array([100.0, 100.0])
        basic_teda.update(outlier)
        micro_teda.process(outlier)

        # Basic TEDA: variance exploded
        assert basic_teda.variance > basic_var_before * 50

        # MicroTEDAclus: original cluster variance UNCHANGED
        assert micro_teda.micro_clusters[0].variance == micro_var_before

    def test_detection_after_contamination(self):
        """Compare detection capability after outliers.

        MicroTEDAclus maintains detection capability.
        """
        from src.detector.teda import TEDADetector

        np.random.seed(42)
        normal_data = np.random.randn(10, 2) * 0.5 + [5, 5]

        basic_teda = TEDADetector(m=3.0, min_samples=3)
        micro_teda = MicroTEDAclus(r0=0.1, min_samples=3)  # r0=0.1 for non-normalized data

        for point in normal_data:
            basic_teda.update(point)
            micro_teda.process(point)

        # First outlier - both should detect
        outlier1 = np.array([100.0, 100.0])
        basic_result1 = basic_teda.update(outlier1)
        micro_result1 = micro_teda.process(outlier1)

        assert basic_result1.is_anomaly == True
        assert micro_result1.is_anomaly == True

        # Second outlier - MicroTEDAclus should still detect
        outlier2 = np.array([-100.0, -100.0])
        basic_result2 = basic_teda.update(outlier2)
        micro_result2 = micro_teda.process(outlier2)

        # MicroTEDAclus: definitely detects
        assert micro_result2.is_anomaly == True

        # Basic TEDA: might not detect (documented limitation)
        # We don't assert on basic_result2.is_anomaly as it's
        # inconsistent due to contamination
