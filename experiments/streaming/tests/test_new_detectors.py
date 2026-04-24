"""
Tests for new streaming detectors (Half-Space Trees, LOF).

Both use river library and are genuinely incremental (no buffer/retrain).
Tests verify the adapter pattern: process() returns MicroTEDAResult,
prequential evaluation (test-then-train), reset, and basic detection.

Run with: pytest tests/test_new_detectors.py -v
"""

import numpy as np
import pytest

from src.detector.halfspace_trees_detector import HalfSpaceTreesDetector
from src.detector.lof_detector import LOFDetector
from src.detector.micro_teda import MicroTEDAResult


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def hst_detector():
    """Create a fresh HalfSpaceTreesDetector."""
    return HalfSpaceTreesDetector(
        n_trees=10, height=8, window_size=50, seed=42,
        threshold=0.5, min_samples=5,
    )


@pytest.fixture
def lof_detector():
    """Create a fresh LOFDetector."""
    return LOFDetector(
        n_neighbors=10, threshold=1.5, min_samples=5,
    )


def _generate_normal(n=200, dim=10, seed=42):
    """Generate n normal samples around origin."""
    rng = np.random.RandomState(seed)
    return rng.randn(n, dim) * 0.5


def _generate_outliers(n=10, dim=10, seed=99):
    """Generate n outlier samples far from origin."""
    rng = np.random.RandomState(seed)
    return rng.randn(n, dim) * 0.5 + 50.0  # Far from normal data


# ============================================================
# HALF-SPACE TREES
# ============================================================

class TestHalfSpaceTreesDetector:

    def test_process_returns_micro_teda_result(self, hst_detector):
        """process() must return MicroTEDAResult."""
        x = np.random.randn(10)
        result = hst_detector.process(x)
        assert isinstance(result, MicroTEDAResult)

    def test_result_fields(self, hst_detector):
        """Returned result has all expected fields."""
        x = np.random.randn(10)
        result = hst_detector.process(x)
        assert hasattr(result, 'eccentricity')
        assert hasattr(result, 'typicality')
        assert hasattr(result, 'cluster_id')
        assert hasattr(result, 'is_anomaly')
        assert hasattr(result, 'num_clusters')
        assert hasattr(result, 'sample_count')
        assert result.sample_count == 1

    def test_sample_count_increments(self, hst_detector):
        """sample_count must increment with each call."""
        for i in range(10):
            result = hst_detector.process(np.random.randn(5))
            assert result.sample_count == i + 1

    def test_no_anomaly_during_warmup(self, hst_detector):
        """No anomalies should be reported during warmup (< min_samples)."""
        for i in range(hst_detector.min_samples - 1):
            result = hst_detector.process(np.random.randn(10))
            assert not result.is_anomaly, (
                f"Anomaly detected at sample {i+1}, before min_samples={hst_detector.min_samples}"
            )

    def test_anomaly_detection_on_outliers(self):
        """HST should detect at least some outliers after training on normal data."""
        det = HalfSpaceTreesDetector(
            n_trees=25, height=15, window_size=100, seed=42,
            threshold=0.5, min_samples=10,
        )
        normal = _generate_normal(200, dim=10)
        outliers = _generate_outliers(20, dim=10)

        # Train on normal
        for x in normal:
            det.process(x)

        # Test on outliers
        anomaly_hits = 0
        for x in outliers:
            result = det.process(x)
            if result.is_anomaly:
                anomaly_hits += 1

        assert anomaly_hits > 0, (
            f"HST did not detect any of {len(outliers)} outliers "
            f"(anomaly_count={det.anomaly_count})"
        )

    def test_reset(self, hst_detector):
        """reset() clears all counters."""
        for _ in range(20):
            hst_detector.process(np.random.randn(5))
        assert hst_detector.total_samples == 20

        hst_detector.reset()
        assert hst_detector.total_samples == 0
        assert hst_detector.anomaly_count == 0

    def test_get_statistics(self, hst_detector):
        """get_statistics() returns expected keys."""
        stats = hst_detector.get_statistics()
        assert "total_samples" in stats
        assert "anomaly_count" in stats
        assert "n_trees" in stats
        assert "threshold" in stats

    def test_eccentricity_range(self, hst_detector):
        """Eccentricity (HST score) should be in [0, 1]."""
        normal = _generate_normal(50, dim=5)
        for x in normal:
            result = hst_detector.process(x)
            assert 0.0 <= result.eccentricity <= 1.0, (
                f"eccentricity={result.eccentricity} out of [0,1]"
            )

    def test_to_dict(self, hst_detector):
        """MicroTEDAResult.to_dict() must work."""
        result = hst_detector.process(np.random.randn(5))
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "is_anomaly" in d


# ============================================================
# LOCAL OUTLIER FACTOR
# ============================================================

class TestLOFDetector:

    def test_process_returns_micro_teda_result(self, lof_detector):
        """process() must return MicroTEDAResult."""
        x = np.random.randn(10)
        result = lof_detector.process(x)
        assert isinstance(result, MicroTEDAResult)

    def test_result_fields(self, lof_detector):
        """Returned result has all expected fields."""
        x = np.random.randn(10)
        result = lof_detector.process(x)
        assert hasattr(result, 'eccentricity')
        assert hasattr(result, 'typicality')
        assert hasattr(result, 'is_anomaly')
        assert result.sample_count == 1

    def test_sample_count_increments(self, lof_detector):
        """sample_count must increment with each call."""
        for i in range(10):
            result = lof_detector.process(np.random.randn(5))
            assert result.sample_count == i + 1

    def test_no_anomaly_during_warmup(self, lof_detector):
        """No anomalies during warmup."""
        for i in range(lof_detector.min_samples - 1):
            result = lof_detector.process(np.random.randn(10))
            assert not result.is_anomaly

    def test_reset(self, lof_detector):
        """reset() clears all counters."""
        for _ in range(20):
            lof_detector.process(np.random.randn(5))
        lof_detector.reset()
        assert lof_detector.total_samples == 0
        assert lof_detector.anomaly_count == 0

    def test_get_statistics(self, lof_detector):
        """get_statistics() returns expected keys."""
        stats = lof_detector.get_statistics()
        assert "total_samples" in stats
        assert "n_neighbors" in stats
        assert "threshold" in stats

    def test_to_dict(self, lof_detector):
        """MicroTEDAResult.to_dict() must work."""
        result = lof_detector.process(np.random.randn(5))
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "is_anomaly" in d


# ============================================================
# CROSS-DETECTOR CHECKS
# ============================================================

class TestDetectorConsistency:
    """Verify both detectors conform to the same adapter interface."""

    @pytest.mark.parametrize("DetectorClass,kwargs", [
        (HalfSpaceTreesDetector, {"n_trees": 5, "height": 5, "window_size": 20}),
        (LOFDetector, {"n_neighbors": 5}),
    ])
    def test_interface_contract(self, DetectorClass, kwargs):
        """Both detectors must implement process(), reset(), get_statistics()."""
        det = DetectorClass(**kwargs)
        assert callable(getattr(det, 'process', None))
        assert callable(getattr(det, 'reset', None))
        assert callable(getattr(det, 'get_statistics', None))

    @pytest.mark.parametrize("DetectorClass,kwargs", [
        (HalfSpaceTreesDetector, {"n_trees": 5, "height": 5, "window_size": 20}),
        (LOFDetector, {"n_neighbors": 5}),
    ])
    def test_process_with_different_dimensions(self, DetectorClass, kwargs):
        """Detectors must handle arbitrary feature dimensions."""
        det = DetectorClass(**kwargs)
        for dim in [3, 10, 25]:
            result = det.process(np.random.randn(dim))
            assert isinstance(result, MicroTEDAResult)
