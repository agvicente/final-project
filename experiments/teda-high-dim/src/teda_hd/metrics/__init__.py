"""Metrics for evaluating variance estimation and clustering quality."""

from .detection import compute_metrics, anomaly_rate

__all__ = ["compute_metrics", "anomaly_rate"]
