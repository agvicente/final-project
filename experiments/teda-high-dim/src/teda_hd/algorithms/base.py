"""Base interface for evolving clustering algorithms."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
import numpy as np


@dataclass
class ClusteringResult:
    """Result of processing a single data point."""

    is_anomaly: bool
    cluster_id: int
    eccentricity: float
    typicality: float
    new_cluster_created: bool
    num_clusters: int
    sample_count: int


@dataclass
class MicroClusterStats:
    """Statistics of a single micro-cluster."""

    cluster_id: int
    n: int
    mean: np.ndarray
    variance: float


class EvolvingClusteringBase(ABC):
    """Abstract base for evolving clustering implementations.

    All variants (original, corrected, ablation) implement this interface
    to allow fair comparison with identical calling code.
    """

    @abstractmethod
    def process(self, x: np.ndarray) -> ClusteringResult:
        """Process a single data point from the stream.

        Args:
            x: Feature vector (d-dimensional).

        Returns:
            ClusteringResult with anomaly decision and metadata.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset all state to initial conditions."""
        ...

    @abstractmethod
    def get_clusters(self) -> List[MicroClusterStats]:
        """Return current micro-cluster statistics."""
        ...

    @property
    @abstractmethod
    def total_samples(self) -> int:
        """Total number of samples processed."""
        ...

    @property
    @abstractmethod
    def anomaly_count(self) -> int:
        """Total number of anomalies detected."""
        ...
