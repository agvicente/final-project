"""Regime topology and indicator metrics for MicroTEDAclus variants.

Instrumentation utilities that read post-stream cluster state via the
public `get_clusters()` API of any `EvolvingClusteringBase` implementation
and produce metrics that characterize the operational regime
(r0-bounded vs data-bounded) and the cluster topology
(silent collapse vs hyper-fragmented vs long-tail).

These are READ-ONLY observers — they do NOT modify the detector.
"""

from typing import List

import numpy as np

from teda_hd.algorithms.base import EvolvingClusteringBase, MicroClusterStats


def compute_effective_variance(
    detector: EvolvingClusteringBase, r0: float
) -> dict:
    """Summarize the effective variance max(sigma^2, r0) across clusters.

    Args:
        detector: detector after processing a stream.
        r0: the variance floor used by the detector.

    Returns:
        Dictionary with:
            mean_var: mean of raw cluster variances (sigma^2)
            mean_eff_var: mean of max(sigma^2, r0)
            median_var: median of raw cluster variances
            n_clusters: total cluster count
            n_above_r0: number of clusters with sigma^2 > r0
            frac_above_r0: fraction of clusters with sigma^2 > r0 (regime indicator)
    """
    clusters = detector.get_clusters()
    if not clusters:
        return {
            "mean_var": 0.0,
            "mean_eff_var": r0,
            "median_var": 0.0,
            "n_clusters": 0,
            "n_above_r0": 0,
            "frac_above_r0": 0.0,
        }

    variances = np.array([mc.variance for mc in clusters], dtype=np.float64)
    eff_var = np.maximum(variances, r0)
    above = variances > r0

    return {
        "mean_var": float(variances.mean()),
        "mean_eff_var": float(eff_var.mean()),
        "median_var": float(np.median(variances)),
        "n_clusters": len(clusters),
        "n_above_r0": int(above.sum()),
        "frac_above_r0": float(above.mean()),
    }


def compute_cluster_topology(detector: EvolvingClusteringBase) -> dict:
    """Topology features distinguishing silent / paranoid / long-tail regimes.

    Args:
        detector: detector after processing a stream.

    Returns:
        Dictionary with:
            n_clusters: total cluster count
            singletons: number of clusters with n == 1
            singleton_frac: fraction of clusters that are singletons
            top1_n: sample count of the largest cluster
            top1_frac: fraction of total samples in the largest cluster
            mean_cluster_size: mean n across clusters
            shannon_entropy: Shannon entropy of cluster size distribution (bits).
                Low: one dominant cluster (long-tail / silent collapse).
                High: uniform fragmentation (paranoid).
    """
    clusters: List[MicroClusterStats] = detector.get_clusters()
    n_clusters = len(clusters)

    if n_clusters == 0:
        return {
            "n_clusters": 0,
            "singletons": 0,
            "singleton_frac": 0.0,
            "top1_n": 0,
            "top1_frac": 0.0,
            "mean_cluster_size": 0.0,
            "shannon_entropy": 0.0,
        }

    sizes = np.array([mc.n for mc in clusters], dtype=np.float64)
    total = float(sizes.sum())
    singletons = int((sizes == 1).sum())
    top1_n = int(sizes.max())

    # Shannon entropy in bits over normalized cluster size distribution
    if total > 0:
        p = sizes / total
        # Avoid log(0) -- contributions of zero-prob bins are zero by convention
        nonzero = p > 0
        entropy = -float(np.sum(p[nonzero] * np.log2(p[nonzero])))
    else:
        entropy = 0.0

    return {
        "n_clusters": n_clusters,
        "singletons": singletons,
        "singleton_frac": singletons / n_clusters,
        "top1_n": top1_n,
        "top1_frac": top1_n / total if total > 0 else 0.0,
        "mean_cluster_size": float(sizes.mean()),
        "shannon_entropy": entropy,
    }


def compute_regime_indicator(
    detector: EvolvingClusteringBase, r0: float
) -> str:
    """Classify operational regime via simple heuristics.

    Heuristic decision tree (post-stream snapshot):
        - frac_above_r0 < 0.1  -> 'r0_bounded'    (variance dominated by floor)
        - frac_above_r0 > 0.9  -> 'data_bounded'  (variance dominated by data)
        - else                  -> 'transition'    (mixed)

    Returns:
        One of: 'r0_bounded', 'transition', 'data_bounded'.
    """
    eff = compute_effective_variance(detector, r0)
    f = eff["frac_above_r0"]
    if f < 0.1:
        return "r0_bounded"
    if f > 0.9:
        return "data_bounded"
    return "transition"


def compute_full_regime_metrics(
    detector: EvolvingClusteringBase, r0: float
) -> dict:
    """Combined call returning all regime + topology metrics.

    Convenience for experiment loops -- returns a flat dict suitable for
    direct insertion into a results CSV row.
    """
    eff = compute_effective_variance(detector, r0)
    topo = compute_cluster_topology(detector)
    regime = compute_regime_indicator(detector, r0)
    out = {**eff, **topo, "regime": regime}
    # n_clusters appears in both -- they should agree by construction
    return out
