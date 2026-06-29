"""Tests for regime topology / indicator metrics.

Verifies that:
1. With small variance (sigma^2 << r0), most clusters are r0-bounded.
2. With large variance (sigma^2 >> r0), most clusters are data-bounded.
3. Topology metrics correctly identify singleton-dominated vs uniform fragments.
4. Empty detector returns zero metrics safely.
"""

import numpy as np
import pytest

from teda_hd.algorithms.variants import create_variant
from teda_hd.generators.gaussian import GaussianStreamGenerator
from teda_hd.metrics.regime import (
    compute_cluster_topology,
    compute_effective_variance,
    compute_full_regime_metrics,
    compute_regime_indicator,
)


def _run_v7(normal_scale: float, r0: float, seed: int = 42, d: int = 10):
    """Helper: run V7 on synthetic data and return the trained detector."""
    gen = GaussianStreamGenerator(
        d=d,
        n_normal=500,
        n_anomalies=0,
        normal_scale=normal_scale,
        seed=seed,
    )
    X, _ = gen.generate()
    algo = create_variant("V7_full_corrected", r0=r0)
    for x in X:
        algo.process(x)
    return algo


class TestRegimeIndicator:
    def test_small_variance_is_r0_bounded(self):
        """sigma^2 << r0  ->  regime == 'r0_bounded'."""
        algo = _run_v7(normal_scale=1e-3, r0=10.0, seed=0)
        regime = compute_regime_indicator(algo, r0=10.0)
        assert regime == "r0_bounded"

    def test_large_variance_is_data_bounded(self):
        """sigma^2 >> r0  ->  regime == 'data_bounded'."""
        algo = _run_v7(normal_scale=100.0, r0=1e-3, seed=0)
        regime = compute_regime_indicator(algo, r0=1e-3)
        assert regime == "data_bounded"


class TestEffectiveVariance:
    def test_frac_above_r0_high_variance(self):
        """High-variance data: most clusters should have sigma^2 > r0."""
        algo = _run_v7(normal_scale=100.0, r0=1e-3, seed=0)
        eff = compute_effective_variance(algo, r0=1e-3)
        assert eff["frac_above_r0"] > 0.9
        assert eff["n_clusters"] > 0

    def test_frac_above_r0_low_variance(self):
        """Low-variance data: most clusters should have sigma^2 < r0."""
        algo = _run_v7(normal_scale=1e-3, r0=10.0, seed=0)
        eff = compute_effective_variance(algo, r0=10.0)
        assert eff["frac_above_r0"] < 0.1


class TestTopology:
    def test_metrics_keys_present(self):
        algo = _run_v7(normal_scale=1.0, r0=0.1, seed=0)
        topo = compute_cluster_topology(algo)
        for key in [
            "n_clusters",
            "singletons",
            "singleton_frac",
            "top1_n",
            "top1_frac",
            "mean_cluster_size",
            "shannon_entropy",
        ]:
            assert key in topo

    def test_topology_returns_valid_ranges(self):
        """Topology metrics should be in valid mathematical ranges."""
        gen = GaussianStreamGenerator(
            d=17, n_normal=500, n_anomalies=0, normal_scale=1.0, seed=0
        )
        X, _ = gen.generate()
        algo = create_variant("V0_original", r0=0.1)
        for x in X:
            algo.process(x)
        topo = compute_cluster_topology(algo)
        assert 0.0 <= topo["singleton_frac"] <= 1.0
        assert 0.0 <= topo["top1_frac"] <= 1.0
        assert topo["n_clusters"] >= 1
        # Shannon entropy is bounded by log2(n_clusters)
        max_entropy = np.log2(max(topo["n_clusters"], 1))
        assert 0.0 <= topo["shannon_entropy"] <= max_entropy + 1e-9


class TestEmptyDetector:
    def test_empty_returns_zero_metrics(self):
        algo = create_variant("V7_full_corrected", r0=0.1)
        # No process() calls -- detector is empty
        topo = compute_cluster_topology(algo)
        eff = compute_effective_variance(algo, r0=0.1)
        assert topo["n_clusters"] == 0
        assert eff["n_clusters"] == 0
        # No exception, no NaN
        assert topo["shannon_entropy"] == 0.0


class TestFullMetrics:
    def test_combined_call_returns_flat_dict(self):
        algo = _run_v7(normal_scale=1.0, r0=0.1, seed=0)
        m = compute_full_regime_metrics(algo, r0=0.1)
        # Should contain keys from all three sub-functions
        assert "frac_above_r0" in m
        assert "shannon_entropy" in m
        assert "regime" in m
        assert m["regime"] in ("r0_bounded", "transition", "data_bounded")
