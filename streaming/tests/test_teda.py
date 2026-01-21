"""
Unit tests for TEDADetector.

Tests cover:
1. Basic functionality (initialization, first sample, updates)
2. Mathematical correctness (eccentricity, typicality, threshold)
3. Anomaly detection behavior
4. Edge cases (reset, predict without update, identical samples)

Run with: pytest tests/test_teda.py -v
"""

import pytest
import numpy as np
from src.detector.teda import TEDADetector, TEDAResult, calculate_eccentricity_batch


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def detector():
    """Create a fresh TEDADetector with default parameters."""
    return TEDADetector(m=3.0, min_samples=3)


@pytest.fixture
def trained_detector():
    """Create a detector trained with 10 normal samples around [5, 5]."""
    detector = TEDADetector(m=3.0, min_samples=3)
    np.random.seed(42)
    normal_data = np.random.randn(10, 2) * 0.5 + [5, 5]
    for point in normal_data:
        detector.update(point)
    return detector


# ============================================================
# BASIC FUNCTIONALITY TESTS
# ============================================================

class TestInitialization:
    """Tests for detector initialization."""

    def test_default_parameters(self, detector):
        """Detector initializes with correct default parameters."""
        assert detector.m == 3.0
        assert detector.min_samples == 3
        assert detector.k == 0
        assert detector.mean is None
        assert detector.variance == 0.0
        assert detector.anomaly_count == 0

    def test_custom_parameters(self):
        """Detector accepts custom parameters."""
        detector = TEDADetector(m=2.0, min_samples=5)
        assert detector.m == 2.0
        assert detector.min_samples == 5


class TestFirstSample:
    """Tests for first sample behavior."""

    def test_first_sample_eccentricity_is_one(self, detector):
        """First sample always has eccentricity = 1."""
        x = np.array([1.0, 2.0, 3.0])
        result = detector.update(x)

        assert result.eccentricity == 1.0
        assert result.typicality == 0.0
        assert result.sample_count == 1

    def test_first_sample_sets_mean(self, detector):
        """First sample becomes the mean."""
        x = np.array([1.0, 2.0, 3.0])
        detector.update(x)

        np.testing.assert_array_equal(detector.mean, x)

    def test_first_sample_variance_zero(self, detector):
        """Variance is zero after first sample."""
        x = np.array([1.0, 2.0, 3.0])
        detector.update(x)

        assert detector.variance == 0.0

    def test_first_sample_not_anomaly(self, detector):
        """First sample is never classified as anomaly (min_samples)."""
        x = np.array([1.0, 2.0, 3.0])
        result = detector.update(x)

        assert result.is_anomaly == False
        assert result.threshold == float('inf')


class TestStatisticsUpdate:
    """Tests for recursive statistics update."""

    def test_mean_updates_correctly(self, detector):
        """Mean is updated correctly with each sample."""
        detector.update(np.array([0.0, 0.0]))
        detector.update(np.array([2.0, 2.0]))

        # Mean should be [1.0, 1.0]
        np.testing.assert_array_almost_equal(detector.mean, [1.0, 1.0])

    def test_variance_updates_correctly(self, detector):
        """Variance is calculated correctly."""
        # Add points at known locations
        detector.update(np.array([0.0, 0.0]))
        detector.update(np.array([2.0, 0.0]))
        detector.update(np.array([1.0, 0.0]))

        # Mean should be [1.0, 0.0]
        np.testing.assert_array_almost_equal(detector.mean, [1.0, 0.0])

        # Variance should be positive
        assert detector.variance > 0

    def test_sample_count_increments(self, detector):
        """Sample count increments with each update."""
        for i in range(5):
            result = detector.update(np.array([float(i), float(i)]))
            assert result.sample_count == i + 1

        assert detector.k == 5


# ============================================================
# MATHEMATICAL CORRECTNESS TESTS
# ============================================================

class TestEccentricity:
    """Tests for eccentricity calculation."""

    def test_eccentricity_decreases_for_typical_points(self, detector):
        """Eccentricity decreases as more similar points are added."""
        eccentricities = []

        # Add 10 identical points
        for _ in range(10):
            result = detector.update(np.array([5.0, 5.0]))
            eccentricities.append(result.eccentricity)

        # After first, eccentricity should decrease (approaching 1/k)
        assert eccentricities[0] == 1.0  # First sample
        assert eccentricities[-1] < eccentricities[1]  # Later < earlier

    def test_eccentricity_high_for_outlier(self, trained_detector):
        """Outlier has high eccentricity."""
        # Trained detector has mean around [5, 5]
        outlier = np.array([100.0, 100.0])
        result = trained_detector.update(outlier)

        assert result.eccentricity > 0.5  # Should be high

    def test_eccentricity_formula(self, detector):
        """Eccentricity follows the formula: ξ = 1/k + d²/(k*σ²)."""
        # Add some points to establish statistics
        points = [
            np.array([0.0, 0.0]),
            np.array([2.0, 0.0]),
            np.array([1.0, 1.0]),
        ]
        for p in points:
            detector.update(p)

        # Now check a new point
        x = np.array([1.0, 0.0])
        result = detector.update(x)

        # Manually calculate expected eccentricity
        k = detector.k
        mean = detector.mean
        variance = detector.variance

        diff = x - mean
        d_squared = np.sum(diff ** 2)
        expected_ecc = (1.0 / k) + (d_squared / (k * variance)) if variance > 0 else 1.0 / k

        assert abs(result.eccentricity - expected_ecc) < 1e-10


class TestTypicality:
    """Tests for typicality calculation."""

    def test_typicality_equals_one_minus_eccentricity(self, detector):
        """Typicality is always 1 - eccentricity."""
        x = np.array([1.0, 2.0])
        result = detector.update(x)

        assert abs(result.typicality - (1.0 - result.eccentricity)) < 1e-10

    def test_typicality_high_for_typical_points(self, trained_detector):
        """Typical points have high typicality."""
        # Point close to mean [5, 5]
        typical = np.array([5.1, 4.9])
        result = trained_detector.update(typical)

        assert result.typicality > 0.5  # Should be high


class TestThreshold:
    """Tests for threshold calculation."""

    def test_threshold_infinite_before_min_samples(self, detector):
        """Threshold is infinite before min_samples are collected."""
        detector.update(np.array([1.0, 1.0]))
        result = detector.update(np.array([2.0, 2.0]))

        # With min_samples=3, threshold should be inf for k < 3
        assert result.threshold == float('inf')

    def test_threshold_decreases_with_samples(self, detector):
        """Threshold decreases as more samples are added."""
        thresholds = []

        for i in range(20):
            result = detector.update(np.array([float(i % 5), float(i % 5)]))
            if result.threshold != float('inf'):
                thresholds.append(result.threshold)

        # Threshold should decrease
        assert thresholds[-1] < thresholds[0]

    def test_threshold_formula(self, detector):
        """Threshold follows formula: (m² + 1) / (2k)."""
        # Add enough samples to pass min_samples
        for i in range(10):
            result = detector.update(np.array([float(i), float(i)]))

        k = detector.k
        m = detector.m
        expected_threshold = (m ** 2 + 1) / (2 * k)

        assert abs(result.threshold - expected_threshold) < 1e-10


# ============================================================
# ANOMALY DETECTION TESTS
# ============================================================

class TestAnomalyDetection:
    """Tests for anomaly detection behavior."""

    def test_outlier_detected_as_anomaly(self, trained_detector):
        """Outlier is detected as anomaly."""
        # Trained with points around [5, 5]
        outlier = np.array([50.0, 50.0])
        result = trained_detector.update(outlier)

        assert result.is_anomaly == True

    def test_normal_point_not_anomaly(self, trained_detector):
        """Normal point is not classified as anomaly."""
        # Point within normal range
        normal = np.array([5.0, 5.0])
        result = trained_detector.update(normal)

        assert result.is_anomaly == False

    def test_min_samples_prevents_early_anomalies(self):
        """No anomalies detected before min_samples."""
        detector = TEDADetector(m=3.0, min_samples=5)

        for i in range(4):
            # Even extreme points shouldn't be flagged
            result = detector.update(np.array([1000.0 * i, 1000.0 * i]))
            assert result.is_anomaly == False

    def test_anomaly_count_tracked(self, trained_detector):
        """Anomaly count is tracked correctly."""
        initial_count = trained_detector.anomaly_count

        # Add an extreme outlier - should definitely be anomaly
        result = trained_detector.update(np.array([100.0, 100.0]))

        # Note: TEDA is adaptive. After adding first outlier, variance increases
        # dramatically, making the algorithm more tolerant. This is expected
        # behavior for evolutionary algorithms - they adapt to the data stream.
        assert result.is_anomaly == True
        assert trained_detector.anomaly_count >= initial_count + 1


class TestPredictWithoutUpdate:
    """Tests for predict() method."""

    def test_predict_does_not_update_statistics(self, trained_detector):
        """Predict does not change detector state."""
        k_before = trained_detector.k
        mean_before = trained_detector.mean.copy()
        var_before = trained_detector.variance

        trained_detector.predict(np.array([100.0, 100.0]))

        assert trained_detector.k == k_before
        np.testing.assert_array_equal(trained_detector.mean, mean_before)
        assert trained_detector.variance == var_before

    def test_predict_returns_valid_result(self, trained_detector):
        """Predict returns valid TEDAResult."""
        result = trained_detector.predict(np.array([5.0, 5.0]))

        assert isinstance(result, TEDAResult)
        assert 0 <= result.eccentricity <= 2  # Reasonable range
        assert result.sample_count == trained_detector.k


# ============================================================
# EDGE CASES
# ============================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_reset(self, trained_detector):
        """Reset clears all state."""
        trained_detector.reset()

        assert trained_detector.k == 0
        assert trained_detector.mean is None
        assert trained_detector.variance == 0.0
        assert trained_detector.anomaly_count == 0

    def test_single_feature(self):
        """Works with single feature (1D)."""
        detector = TEDADetector()

        for i in range(10):
            result = detector.update(np.array([float(i)]))
            assert isinstance(result, TEDAResult)

    def test_many_features(self):
        """Works with many features (high dimensional)."""
        detector = TEDADetector()

        for i in range(10):
            result = detector.update(np.random.randn(100))
            assert isinstance(result, TEDAResult)

    def test_identical_points(self, detector):
        """Handles identical points (zero variance)."""
        for _ in range(10):
            result = detector.update(np.array([5.0, 5.0]))
            assert not np.isnan(result.eccentricity)
            assert not np.isinf(result.eccentricity)

    def test_list_input_converted(self, detector):
        """List input is converted to numpy array."""
        result = detector.update([1.0, 2.0, 3.0])
        assert isinstance(result, TEDAResult)

    def test_get_statistics(self, trained_detector):
        """get_statistics returns valid dict."""
        stats = trained_detector.get_statistics()

        assert 'sample_count' in stats
        assert 'mean' in stats
        assert 'variance' in stats
        assert 'anomaly_count' in stats
        assert 'current_threshold' in stats

        assert stats['sample_count'] == trained_detector.k


# ============================================================
# BATCH ECCENTRICITY TESTS
# ============================================================

class TestBatchEccentricity:
    """Tests for calculate_eccentricity_batch function."""

    def test_batch_eccentricity_shape(self):
        """Returns array of correct shape."""
        X = np.random.randn(100, 5)
        ecc = calculate_eccentricity_batch(X)

        assert ecc.shape == (100,)

    def test_batch_outlier_high_eccentricity(self):
        """Outlier has highest eccentricity in batch."""
        # Normal cluster + one outlier
        X = np.vstack([
            np.random.randn(10, 2) * 0.1,  # Tight cluster
            np.array([[10.0, 10.0]])  # Outlier
        ])
        ecc = calculate_eccentricity_batch(X)

        # Outlier (last point) should have highest eccentricity
        assert ecc[-1] == np.max(ecc)

    def test_batch_single_point(self):
        """Single point returns eccentricity of 1."""
        X = np.array([[1.0, 2.0]])
        ecc = calculate_eccentricity_batch(X)

        assert len(ecc) == 1
        assert ecc[0] == 1.0


# ============================================================
# RESULT CONVERSION TESTS
# ============================================================

class TestStatisticsContamination:
    """Tests demonstrating statistics contamination in basic TEDA.

    These tests document the known limitation of single-center TEDA:
    outliers contaminate the global statistics, potentially causing
    subsequent anomalies to go undetected.

    MicroTEDAclus addresses this by rejecting outliers into separate
    micro-clusters instead of updating the main statistics.
    """

    def test_outlier_contaminates_variance(self, trained_detector):
        """Demonstrate how one outlier dramatically increases variance."""
        var_before = trained_detector.variance
        mean_before = trained_detector.mean.copy()

        # Add extreme outlier
        trained_detector.update(np.array([100.0, 100.0]))

        var_after = trained_detector.variance
        mean_after = trained_detector.mean

        # Variance explodes (typically 100x-1000x increase)
        assert var_after > var_before * 50

        # Mean shifts significantly toward outlier
        assert np.linalg.norm(mean_after - mean_before) > 5

    def test_second_outlier_may_not_be_detected(self, trained_detector):
        """After one outlier, second outlier may go undetected.

        This is the key limitation that MicroTEDAclus addresses.
        """
        # First outlier - should be detected
        result1 = trained_detector.update(np.array([100.0, 100.0]))
        assert result1.is_anomaly == True

        # Second outlier in opposite direction
        # Due to contaminated statistics, may NOT be detected
        result2 = trained_detector.update(np.array([-100.0, -100.0]))

        # Document the behavior (not asserting failure, just documenting)
        # In basic TEDA, this often returns False (false negative)
        # MicroTEDAclus would detect this as anomaly
        if not result2.is_anomaly:
            # This is the expected problematic behavior in basic TEDA
            pass  # Documented limitation

    def test_normal_point_after_contamination(self, trained_detector):
        """After contamination, normal points may appear as outliers.

        This inversion effect is particularly dangerous in IDS.
        """
        # Contaminate with extreme outlier
        trained_detector.update(np.array([1000.0, 1000.0]))

        # Now a "normal" point (close to original mean [5,5]) might
        # have unusual eccentricity due to the shifted statistics
        result = trained_detector.predict(np.array([5.0, 5.0]))

        # The eccentricity is no longer near 1/k as expected for typical points
        # because the mean has shifted toward [1000, 1000]
        assert result.distance_to_mean > 50  # Far from new mean


class TestTEDAResult:
    """Tests for TEDAResult dataclass."""

    def test_to_dict(self, detector):
        """TEDAResult converts to dict correctly."""
        result = detector.update(np.array([1.0, 2.0]))
        d = result.to_dict()

        assert isinstance(d, dict)
        assert d['eccentricity'] == result.eccentricity
        assert d['typicality'] == result.typicality
        assert d['is_anomaly'] == result.is_anomaly
        assert d['sample_count'] == result.sample_count
