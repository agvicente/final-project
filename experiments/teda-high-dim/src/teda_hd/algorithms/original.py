"""Faithful reimplementation of MicroTEDAclus (Maia et al., 2020).

This reproduces the EXACT behavior of the original EvolvingClustering code at:
    https://github.com/cseveriano/evolving_clustering

Key formulas replicated from the original code (NOT the paper):
    - Variance:     ((n-1)/n)*var + ((norm(delta)*2/d)^2 / (n-1))
    - Eccentricity: 1/n + (norm(a)*2/d)^2 / (n*var)
    - is_outlier:   n<3 -> var > r0;  n>=3 -> norm_ecc > (m^2+1)/(2n)
    - Update:       ALL accepting clusters (not just best)
    - Intersection:  dist < 2*(sqrt(var_i) + sqrt(var_j))

IMPORTANT: The original code uses `(norm*2/d)^2` which differs from what the
paper describes as `||x-mu||^2`. This discrepancy is the subject of our analysis.
At d=2 the factor (2/d)^2 = 1, making it harmless. At d=17, it becomes 0.014,
underestimating variance by ~70x in absolute terms.

No numba/JIT dependency — pure NumPy for portability and debuggability.
"""

import math
import numpy as np
from typing import List, Optional
from .base import EvolvingClusteringBase, ClusteringResult, MicroClusterStats


class OriginalMicroTEDAclus(EvolvingClusteringBase):
    """Original MicroTEDAclus with the (norm*2/d)^2 variance formula.

    Args:
        r0: Variance limit for the n<3 outlier test (default: 0.001).
        decay: Controls fading factor for life decay (fading = 1/decay).
        enable_pruning: Whether to prune dead micro-clusters.
        enable_macro_clusters: Whether to compute macro-cluster intersection.
    """

    def __init__(
        self,
        r0: float = 0.001,
        decay: int = 100,
        enable_pruning: bool = True,
        enable_macro_clusters: bool = False,
    ):
        self.r0 = r0
        self.fading_factor = 1.0 / decay
        self.enable_pruning = enable_pruning
        self.enable_macro_clusters = enable_macro_clusters

        self._clusters: List[dict] = []
        self._next_id = 1
        self._total_samples = 0
        self._anomaly_count = 0

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

    def get_clusters(self) -> List[MicroClusterStats]:
        return [
            MicroClusterStats(
                cluster_id=mc["id"],
                n=mc["num_samples"],
                mean=mc["mean"].copy(),
                variance=mc["variance"],
            )
            for mc in self._clusters
        ]

    def process(self, x: np.ndarray) -> ClusteringResult:
        x = np.asarray(x, dtype=np.float64)
        self._total_samples += 1

        # First sample: create first micro-cluster
        if self._total_samples == 1:
            mc = self._create_cluster(x)
            return ClusteringResult(
                is_anomaly=False,
                cluster_id=mc["id"],
                eccentricity=1.0,
                typicality=0.0,
                new_cluster_created=True,
                num_clusters=1,
                sample_count=self._total_samples,
            )

        # Try to add to existing clusters
        new_cluster_needed = True

        for mc in self._clusters:
            s_ik = mc["num_samples"]
            mu_ik = mc["mean"]
            var_ik = mc["variance"]

            # Compute hypothetical updated values
            new_s, new_mean, new_var, new_norm_ecc = self._get_updated_values(
                x, s_ik, mu_ik, var_ik
            )

            if not self._is_outlier(new_s, new_var, new_norm_ecc):
                # ORIGINAL BEHAVIOR: update ALL accepting clusters
                self._update_life(x, mc)
                mc["num_samples"] = new_s
                mc["mean"] = new_mean
                mc["variance"] = new_var
                mc["density"] = 1.0 / new_norm_ecc if new_norm_ecc > 0 else 0.0
                new_cluster_needed = False

        if new_cluster_needed:
            mc = self._create_cluster(x)
            is_anomaly = self._total_samples >= 3  # min_samples warmup
            if is_anomaly:
                self._anomaly_count += 1

            return ClusteringResult(
                is_anomaly=is_anomaly,
                cluster_id=mc["id"],
                eccentricity=1.0,
                typicality=0.0,
                new_cluster_created=True,
                num_clusters=len(self._clusters),
                sample_count=self._total_samples,
            )

        # Pruning
        if self.enable_pruning:
            self._prune()

        # Find best cluster for reporting
        best_id, best_ecc = self._find_best_cluster(x)

        return ClusteringResult(
            is_anomaly=False,
            cluster_id=best_id,
            eccentricity=best_ecc,
            typicality=1.0 - best_ecc,
            new_cluster_created=False,
            num_clusters=len(self._clusters),
            sample_count=self._total_samples,
        )

    # ── Core formulas (faithful to original code) ────────────

    @staticmethod
    def update_variance(delta: np.ndarray, s_ik: int, var_ik: float) -> float:
        """Original variance formula: ((n-1)/n)*var + ((norm*2/d)^2 / (n-1)).

        Source: EvolvingClustering.py line 99.
        """
        d = len(delta)
        norm_delta = np.linalg.norm(delta)
        variance = ((s_ik - 1) / s_ik) * var_ik + (
            ((norm_delta * 2 / d) ** 2) / (s_ik - 1)
        )
        return variance

    @staticmethod
    def get_eccentricity(
        x: np.ndarray, num_samples: int, mean: np.ndarray, var: float
    ) -> float:
        """Original eccentricity: 1/n + (norm(a)*2/d)^2 / (n*var).

        Source: EvolvingClustering.py lines 109-116.
        """
        if var == 0 and num_samples > 1:
            return 1.0 / num_samples

        a = mean - x
        d = len(a)
        norm_a = np.linalg.norm(a)
        return (1.0 / num_samples) + ((norm_a * 2 / d) ** 2 / (num_samples * var))

    @staticmethod
    def get_normalized_eccentricity(
        x: np.ndarray, num_samples: int, mean: np.ndarray, var: float
    ) -> float:
        """Normalized eccentricity = ecc / 2."""
        return OriginalMicroTEDAclus.get_eccentricity(x, num_samples, mean, var) / 2.0

    def _is_outlier(self, s_ik: int, var_ik: float, norm_ecc: float) -> bool:
        """Original outlier test.

        Source: EvolvingClustering.py lines 48-57.

        For n < 3: outlier if var > r0
        For n >= 3: outlier if norm_ecc > (m(k)^2 + 1) / (2k)
        """
        if s_ik < 3:
            return var_ik > self.r0
        else:
            m_k = 3.0 / (1.0 + math.exp(-0.007 * (s_ik - 100)))
            threshold = (m_k**2 + 1) / (2 * s_ik)
            return norm_ecc > threshold

    # ── Intersection (for macro-clusters) ────────────────────

    @staticmethod
    def has_intersection(mc_i: dict, mc_j: dict) -> bool:
        """Original intersection test: dist < 2*(sqrt(var_i) + sqrt(var_j)).

        Source: EvolvingClustering.py lines 306-318.

        CRITICAL: dist uses TRUE Euclidean distance, but sqrt(var) uses the
        scaled-down variance. At d=17, radii are ~(2/17) of correct values.
        """
        dist = np.linalg.norm(mc_i["mean"] - mc_j["mean"])
        deviation = 2 * (np.sqrt(mc_i["variance"]) + np.sqrt(mc_j["variance"]))
        return dist <= deviation

    # ── Internal helpers ─────────────────────────────────────

    def _get_updated_values(
        self, x: np.ndarray, s_ik: int, mu_ik: np.ndarray, var_ik: float
    ):
        """Compute hypothetical cluster stats after adding point x.

        Source: EvolvingClustering.py lines 84-94.
        """
        new_s = s_ik + 1
        new_mean = ((new_s - 1) / new_s) * mu_ik + (x / new_s)
        delta = x - new_mean
        new_var = self.update_variance(delta, new_s, var_ik)
        new_norm_ecc = self.get_normalized_eccentricity(x, new_s, new_mean, new_var)
        return new_s, new_mean, new_var, new_norm_ecc

    def _create_cluster(self, x: np.ndarray) -> dict:
        mc = {
            "id": self._next_id,
            "num_samples": 1,
            "mean": x.copy(),
            "variance": 0.0,
            "density": 0.0,
            "active": True,
            "life": 1.0,
        }
        self._next_id += 1
        self._clusters.append(mc)
        return mc

    def _update_life(self, x: np.ndarray, mc: dict) -> None:
        """Original life update.

        Source: EvolvingClustering.py lines 72-81.

        CRITICAL: Uses sqrt(variance) as radius (scaled-down) but
        Euclidean distance to mean (true scale). Mixed scales.
        """
        prev_var = mc["variance"]
        if prev_var > 0:
            dist = np.linalg.norm(x - mc["mean"])
            rt = np.sqrt(prev_var)
            mc["life"] += ((rt - dist) / rt) * self.fading_factor
        else:
            mc["life"] = 1.0

    def _prune(self) -> None:
        """Remove micro-clusters with life < 0."""
        for mc in self._clusters:
            if not mc.get("active", True):
                mc["life"] -= self.fading_factor
        self._clusters = [mc for mc in self._clusters if mc["life"] >= 0]

    def _find_best_cluster(self, x: np.ndarray) -> tuple:
        """Find cluster with lowest eccentricity for point x."""
        best_id = -1
        best_ecc = float("inf")
        for mc in self._clusters:
            if mc["num_samples"] > 0:
                ecc = self.get_eccentricity(
                    x, mc["num_samples"], mc["mean"], mc["variance"]
                )
                if ecc < best_ecc:
                    best_ecc = ecc
                    best_id = mc["id"]
        return best_id, best_ecc
