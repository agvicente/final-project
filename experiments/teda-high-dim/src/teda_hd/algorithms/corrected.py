"""Corrected MicroTEDAclus with Welford variance and 5 adaptations.

This implementation fixes the high-dimensional variance underestimation
by applying 5 modifications to the original MicroTEDAclus:

    1. Welford variance:    dot(delta_pre, delta_post) accumulation
    2. Consistent ecc:      ||x-mu||^2 instead of (norm*2/d)^2
    3. Selective update:     Update ONLY the best accepting cluster
    4. n=1 guard:            Permissive threshold=13 for singleton clusters
    5. n=2 guard:            Dual condition: zeta > threshold AND var >= r0

Each modification can be toggled independently via constructor flags
to support ablation studies.
"""

import math
import numpy as np
from typing import List
from .base import EvolvingClusteringBase, ClusteringResult, MicroClusterStats


class CorrectedMicroCluster:
    """A single micro-cluster with Welford variance tracking."""

    def __init__(self, cluster_id: int, initial_point: np.ndarray):
        self.cluster_id = cluster_id
        self.n = 1
        self.mean = initial_point.copy().astype(np.float64)
        self.variance = 0.0
        self._var_sum = 0.0  # Welford accumulator
        self.life = 1.0
        self.density = 0.0   # 1/norm_ecc, set on update (original define_activations input)
        self.active = True   # density-filter activation flag (original)

    def update(self, x: np.ndarray, fading_factor: float = 0.0) -> None:
        """Update cluster stats with Welford's online algorithm.

        If ``fading_factor > 0``, also updates ``life`` exactly as the original
        EvolvingClustering (Maia 2020): life rises when the new point falls
        within the cluster radius (sqrt(var)) and falls otherwise, so inactive
        clusters decay and become eligible for pruning. Uses the PRE-update
        mean/variance, matching the reference implementation.
        """
        x = np.asarray(x, dtype=np.float64)

        # Life update uses the state BEFORE absorbing x (as in the original).
        if fading_factor > 0.0:
            prev_var = self.variance
            if prev_var > 0.0:
                dist = float(np.linalg.norm(x - self.mean))
                rt = math.sqrt(prev_var)
                self.life += ((rt - dist) / rt) * fading_factor
            else:
                self.life = 1.0

        self.n += 1

        # Mean update (same as original)
        delta = x - self.mean
        self.mean = self.mean + delta / self.n

        # Variance: Welford accumulation
        delta2 = x - self.mean
        self._var_sum += np.dot(delta, delta2)

        if self.n > 1:
            self.variance = self._var_sum / (self.n - 1)


class CorrectedMicroTEDAclus(EvolvingClusteringBase):
    """MicroTEDAclus with corrected variance and optional adaptations.

    Each adaptation can be toggled for ablation studies:

    Args:
        r0: Variance limit (default: 0.001).
        decay: Life decay factor denominator (fading = 1/decay).
        use_welford_variance: Use Welford instead of (norm*2/d)^2.
        use_consistent_eccentricity: Use ||diff||^2 instead of (norm*2/d)^2 in ecc.
        use_selective_update: Update only best cluster (not all accepting).
        use_n1_guard: Use permissive threshold=13 for n=1 clusters.
        use_n2_guard: Use dual condition (zeta > threshold AND var >= r0) for n=2.
        enable_pruning: Whether to prune dead clusters.
    """

    def __init__(
        self,
        r0: float = 0.001,
        decay: int = 100,
        use_welford_variance: bool = True,
        use_consistent_eccentricity: bool = True,
        use_selective_update: bool = True,
        use_n1_guard: bool = True,
        use_n2_guard: bool = True,
        enable_pruning: bool = False,
    ):
        self.r0 = r0
        self.fading_factor = 1.0 / decay
        self.use_welford_variance = use_welford_variance
        self.use_consistent_eccentricity = use_consistent_eccentricity
        self.use_selective_update = use_selective_update
        self.use_n1_guard = use_n1_guard
        self.use_n2_guard = use_n2_guard
        self.enable_pruning = enable_pruning

        self._clusters: List[CorrectedMicroCluster] = []
        self._next_id = 1
        self._total_samples = 0
        self._anomaly_count = 0

        # Shadow clusters for original-style update when Welford is off
        # (needed for ablation variants that use original variance)
        self._cluster_variances: dict = {}  # id -> original-style variance

    # ── Interface ────────────────────────────────────────────

    @property
    def total_samples(self) -> int:
        return self._total_samples

    @property
    def anomaly_count(self) -> int:
        return self._anomaly_count

    def reset(self) -> None:
        self._clusters = []
        self._next_id = 1
        self._total_samples = 0
        self._anomaly_count = 0
        self._cluster_variances = {}

    def get_clusters(self) -> List[MicroClusterStats]:
        return [
            MicroClusterStats(
                cluster_id=mc.cluster_id,
                n=mc.n,
                mean=mc.mean.copy(),
                variance=self._get_variance(mc),
            )
            for mc in self._clusters
        ]

    def process(self, x: np.ndarray) -> ClusteringResult:
        x = np.asarray(x, dtype=np.float64)
        self._total_samples += 1

        # First sample
        if not self._clusters:
            mc = self._create_cluster(x)
            return ClusteringResult(
                is_anomaly=False,
                cluster_id=mc.cluster_id,
                eccentricity=1.0,
                typicality=0.0,
                new_cluster_created=True,
                num_clusters=1,
                sample_count=self._total_samples,
            )

        # Find accepting clusters
        accepting = []
        for mc in self._clusters:
            if self._chebyshev_accepts(mc, x):
                accepting.append(mc)

        if accepting:
            # Pass fading_factor only when pruning is enabled, so life actually
            # decays; otherwise update() leaves life at 1.0 (inert, the prior behavior).
            ff = self.fading_factor if self.enable_pruning else 0.0
            if self.use_selective_update:
                # ADAPTATION 3: Update ONLY best cluster
                best = max(accepting, key=lambda mc: self._typicality(mc, x))
                ecc = self._eccentricity(best, x)
                best.update(x, fading_factor=ff)
                if not self.use_welford_variance:
                    self._update_original_variance(best, x)
                # density = 1/norm_ecc (original update_micro_cluster), input to the filter
                best.density = (1.0 / (ecc / 2.0)) if ecc > 0 else 0.0
            else:
                # Original behavior: update ALL accepting clusters
                best = max(accepting, key=lambda mc: self._typicality(mc, x))
                ecc = self._eccentricity(best, x)
                for mc in accepting:
                    e = self._eccentricity(mc, x)
                    mc.update(x, fading_factor=ff)
                    if not self.use_welford_variance:
                        self._update_original_variance(mc, x)
                    mc.density = (1.0 / (e / 2.0)) if e > 0 else 0.0

            if self.enable_pruning:
                # Original chain: rebuild density-based activation, then decay
                # the life of INACTIVE clusters and remove the dead ones.
                self._define_activations()
                self._prune()

            return ClusteringResult(
                is_anomaly=False,
                cluster_id=best.cluster_id,
                eccentricity=ecc,
                typicality=1.0 - ecc,
                new_cluster_created=False,
                num_clusters=len(self._clusters),
                sample_count=self._total_samples,
            )
        else:
            # Rejected by all → new cluster
            mc = self._create_cluster(x)
            is_anomaly = self._total_samples >= 3
            if is_anomaly:
                self._anomaly_count += 1

            return ClusteringResult(
                is_anomaly=is_anomaly,
                cluster_id=mc.cluster_id,
                eccentricity=1.0,
                typicality=0.0,
                new_cluster_created=True,
                num_clusters=len(self._clusters),
                sample_count=self._total_samples,
            )

    # ── Variance ─────────────────────────────────────────────

    def _get_variance(self, mc: CorrectedMicroCluster) -> float:
        """Get variance using the selected method."""
        if self.use_welford_variance:
            return mc.variance
        return self._cluster_variances.get(mc.cluster_id, 0.0)

    def _update_original_variance(
        self, mc: CorrectedMicroCluster, x: np.ndarray
    ) -> None:
        """Update original-style variance for ablation variants."""
        d = len(x)
        delta = x - mc.mean
        norm_delta = np.linalg.norm(delta)
        old_var = self._cluster_variances.get(mc.cluster_id, 0.0)
        if mc.n > 1:
            new_var = ((mc.n - 1) / mc.n) * old_var + (
                ((norm_delta * 2 / d) ** 2) / (mc.n - 1)
            )
        else:
            new_var = 0.0
        self._cluster_variances[mc.cluster_id] = new_var

    # ── Eccentricity ─────────────────────────────────────────

    def _eccentricity(self, mc: CorrectedMicroCluster, x: np.ndarray) -> float:
        """Compute eccentricity using selected method."""
        var = self._get_variance(mc)
        if var <= 0 and mc.n > 1:
            return 1.0 / mc.n

        diff = x - mc.mean
        if self.use_consistent_eccentricity:
            # ADAPTATION 2: Use ||diff||^2 directly
            dist_sq = np.sum(diff**2)
        else:
            # Original: (norm*2/d)^2
            d = len(diff)
            dist_sq = (np.linalg.norm(diff) * 2 / d) ** 2

        effective_var = max(var, self.r0) if var > 0 else self.r0
        return (1.0 / mc.n) + (dist_sq / (mc.n * effective_var))

    def _normalized_eccentricity(
        self, mc: CorrectedMicroCluster, x: np.ndarray
    ) -> float:
        return self._eccentricity(mc, x) / 2.0

    def _typicality(self, mc: CorrectedMicroCluster, x: np.ndarray) -> float:
        return 1.0 - self._eccentricity(mc, x)

    # ── Chebyshev test ───────────────────────────────────────

    def _dynamic_m(self, n: int) -> float:
        """Dynamic threshold m(k) = 3 / (1 + exp(-0.007*(k-100)))."""
        return 3.0 / (1.0 + math.exp(-0.007 * (n - 100)))

    def _chebyshev_accepts(self, mc: CorrectedMicroCluster, x: np.ndarray) -> bool:
        """Test if cluster accepts point via Chebyshev bound."""
        # Compute hypothetical values after adding x
        hyp_n = mc.n + 1
        hyp_mean = mc.mean + (x - mc.mean) / hyp_n
        delta_hyp = x - hyp_mean

        if self.use_welford_variance:
            delta_pre = x - mc.mean
            hyp_var_sum = mc._var_sum + np.dot(delta_pre, delta_hyp)
            hyp_var = hyp_var_sum / (hyp_n - 1) if hyp_n > 1 else 0.0
        else:
            old_var = self._cluster_variances.get(mc.cluster_id, 0.0)
            d = len(x)
            norm_d = np.linalg.norm(delta_hyp)
            hyp_var = ((hyp_n - 1) / hyp_n) * old_var + (
                ((norm_d * 2 / d) ** 2) / (hyp_n - 1)
            ) if hyp_n > 1 else 0.0

        # Compute eccentricity with hypothetical values
        if hyp_var <= 0 and hyp_n > 1:
            hyp_norm_ecc = (1.0 / hyp_n) / 2.0
        else:
            diff = x - hyp_mean
            if self.use_consistent_eccentricity:
                dist_sq = np.sum(diff**2)
            else:
                d = len(diff)
                dist_sq = (np.linalg.norm(diff) * 2 / d) ** 2

            eff_var = max(hyp_var, self.r0) if hyp_var > 0 else self.r0
            hyp_ecc = (1.0 / hyp_n) + (dist_sq / (hyp_n * eff_var))
            hyp_norm_ecc = hyp_ecc / 2.0

        # ADAPTATION 4: n=1 guard
        if self.use_n1_guard and mc.n == 1:
            return hyp_norm_ecc <= 13.0

        # ADAPTATION 5: n=2 guard
        if self.use_n2_guard and mc.n == 2:
            m = self._dynamic_m(hyp_n)
            threshold = (m**2 + 1) / (2 * hyp_n)
            return not (hyp_norm_ecc > threshold and hyp_var >= self.r0)

        # Original n<3 test
        if hyp_n < 3:
            return not (hyp_var > self.r0)

        # Standard Chebyshev test (n >= 3)
        m = self._dynamic_m(hyp_n)
        threshold = (m**2 + 1) / (2 * hyp_n)
        return hyp_norm_ecc <= threshold

    # ── Internal ─────────────────────────────────────────────

    def _define_activations(self) -> None:
        """Density-based activation filter, faithful to the original
        EvolvingClustering.define_activations (Maia 2020).

        Builds macro-clusters as connected components under the intersection
        rule ``dist(mu_i, mu_j) <= 2*(sqrt(var_i)+sqrt(var_j))``; within each
        macro-cluster a micro-cluster is *active* iff ``n > 2`` and its density
        is at least the macro-cluster mean density. Inactive clusters are the
        ones the pruning step decays.
        """
        clusters = self._clusters
        n = len(clusters)
        if n == 0:
            return

        # Union-find over the intersection graph (replaces NetworkX components).
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri

        for i in range(n):
            for j in range(i + 1, n):
                a, b = clusters[i], clusters[j]
                dist = float(np.linalg.norm(a.mean - b.mean))
                deviation = 2.0 * (math.sqrt(max(a.variance, 0.0)) + math.sqrt(max(b.variance, 0.0)))
                if dist <= deviation:
                    union(i, j)

        # Group indices by macro-cluster root.
        groups: dict = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        for members in groups.values():
            mean_density = sum(clusters[i].density for i in members) / len(members)
            for i in members:
                mc = clusters[i]
                mc.active = (mc.n > 2) and (mc.density >= mean_density)

    def _prune(self) -> None:
        """Decay the life of INACTIVE clusters and remove the dead ones,
        faithful to the original prune_micro_clusters (decay every step)."""
        survivors = []
        for mc in self._clusters:
            if not mc.active:
                mc.life -= self.fading_factor
                if mc.life < 0:
                    continue  # pruned
            survivors.append(mc)
        self._clusters = survivors

    # ── Internal ─────────────────────────────────────────────

    def _create_cluster(self, x: np.ndarray) -> CorrectedMicroCluster:
        mc = CorrectedMicroCluster(self._next_id, x)
        self._next_id += 1
        self._clusters.append(mc)
        self._cluster_variances[mc.cluster_id] = 0.0
        return mc
