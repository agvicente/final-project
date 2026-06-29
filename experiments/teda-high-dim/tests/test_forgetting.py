"""Tests for the life-based forgetting/pruning ported from the original
EvolvingClustering (Maia 2020), exposed via enable_pruning / the V7_forgetting variant.

Verifies that:
1. With pruning OFF (the prior default), life stays at 1.0 and no cluster is pruned.
2. With pruning ON, an inactive cluster decays and is removed, while an active one survives.
3. Enabling pruning does not alter the 5 adaptations (A1-A5).
"""

import numpy as np
import pytest

from teda_hd.algorithms.corrected import CorrectedMicroCluster, CorrectedMicroTEDAclus
from teda_hd.algorithms.variants import create_variant, VARIANT_CONFIGS


class TestLifeUpdate:
    """The CorrectedMicroCluster.update() life mechanics."""

    def test_life_inert_without_fading(self):
        """fading_factor=0 (default) leaves life at 1.0 — prior behavior preserved."""
        mc = CorrectedMicroCluster(1, np.zeros(5))
        for _ in range(50):
            mc.update(np.full(5, 10.0))  # far points, but fading off
        assert mc.life == 1.0

    def test_life_rises_for_close_points(self):
        """A point inside the radius (dist < sqrt(var)) increases life."""
        mc = CorrectedMicroCluster(1, np.zeros(3))
        # Build up some variance with a couple of moderate points.
        mc.update(np.array([1.0, 0.0, 0.0]), fading_factor=0.01)
        mc.update(np.array([-1.0, 0.0, 0.0]), fading_factor=0.01)
        life_before = mc.life
        # A point at the mean (dist≈0 < rt) should push life up.
        mc.update(np.zeros(3), fading_factor=0.01)
        assert mc.life >= life_before

    def test_life_falls_for_far_points(self):
        """A point well outside the radius (dist >> sqrt(var)) decreases life."""
        mc = CorrectedMicroCluster(1, np.zeros(3))
        mc.update(np.array([0.1, 0.0, 0.0]), fading_factor=0.01)  # tiny variance
        life_before = mc.life
        mc.update(np.array([100.0, 0.0, 0.0]), fading_factor=0.01)  # far away
        assert mc.life < life_before


class TestPruning:
    """Cluster removal driven by life decay."""

    def test_no_pruning_when_disabled(self):
        """enable_pruning=False keeps every cluster that gets created."""
        det = CorrectedMicroTEDAclus(r0=0.1, enable_pruning=False)
        rng = np.random.default_rng(0)
        # Two well-separated regions → at least two clusters, none pruned.
        for _ in range(200):
            det.process(rng.normal(0, 0.3, 4))
        for _ in range(200):
            det.process(rng.normal(20, 0.3, 4))
        # Without pruning the inert region's clusters persist.
        n_no_prune = len(det.get_clusters())
        assert n_no_prune >= 2

    def test_pruning_removes_inactive_region(self):
        """With pruning ON, abandoning a region lets its clusters decay and be removed."""
        det = CorrectedMicroTEDAclus(r0=0.1, decay=20, enable_pruning=True)
        rng = np.random.default_rng(1)
        # Phase 1: populate region A.
        for _ in range(150):
            det.process(rng.normal(0, 0.3, 4))
        n_after_a = len(det.get_clusters())
        # Phase 2: move permanently to region B; region A clusters should decay/prune.
        for _ in range(400):
            det.process(rng.normal(20, 0.3, 4))
        n_after_b = len(det.get_clusters())
        # Pruning must keep the cluster count bounded, not let it grow unbounded.
        det_no_prune = CorrectedMicroTEDAclus(r0=0.1, enable_pruning=False)
        rng2 = np.random.default_rng(1)
        for _ in range(150):
            det_no_prune.process(rng2.normal(0, 0.3, 4))
        for _ in range(400):
            det_no_prune.process(rng2.normal(20, 0.3, 4))
        n_no_prune = len(det_no_prune.get_clusters())
        # The pruned detector ends with fewer clusters than the un-pruned one.
        assert n_after_b < n_no_prune
        assert n_after_a >= 1


class TestAdaptationsIntact:
    """Enabling pruning must not change the 5 ablation flags (A1-A5)."""

    def test_v7_forgetting_flags_match_v7(self):
        cfg7 = VARIANT_CONFIGS["V7_full_corrected"]
        cfgf = VARIANT_CONFIGS["V7_forgetting"]
        for flag in (
            "use_welford_variance",
            "use_consistent_eccentricity",
            "use_selective_update",
            "use_n1_guard",
            "use_n2_guard",
        ):
            assert cfg7[flag] == cfgf[flag]
        # The only difference is pruning.
        assert cfgf.get("enable_pruning") is True
        assert "enable_pruning" not in cfg7  # V7 keeps the (False) default

    def test_v7_forgetting_constructs(self):
        det = create_variant("V7_forgetting", r0=0.1)
        assert det.enable_pruning is True
        assert det.use_welford_variance is True
