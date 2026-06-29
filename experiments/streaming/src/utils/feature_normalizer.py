"""Online feature normalization with warmup.

Implements z-score (StandardScaler-equivalent) and min-max normalization
for the streaming pipeline, *without* the sklearn dependency. Used by
Campaign-06 to validate the regime-transition hypothesis on real IoT data:
when raw features are projected into unit variance space, V0 and V7 should
collapse into the same operational regime (data-bounded), as predicted by
the synthetic Exp 3 in `experiments/teda-high-dim/`.

Design:
    - Warmup phase: accumulates the first N samples raw and returns them
      unmodified (the detector sees raw data) — avoids leaking statistics
      from a single sample.
    - Fit phase: at sample N+1, computes population mean/std over the
      warmup buffer.
    - Transform phase: subsequent samples are normalized using the fitted
      statistics. Statistics are NOT updated post-fit (to keep determinism
      across runs).

Edge cases:
    - Constant feature (std=0): replaced with 1e-8 fallback to avoid
      divide-by-zero.
    - Min-max with degenerate range (min==max): output 0 for that feature.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


class FeatureNormalizer:
    """Online feature normalizer with deferred fit.

    Args:
        mode: 'zscore' (subtract mean, divide by std) or 'minmax' (rescale
            to [0, 1]). Default 'zscore'.
        warmup_size: Number of raw samples to accumulate before fitting
            statistics. Default 100.

    Behavior:
        Calls `update_and_transform(x)` for each incoming sample:
            * If `is_fitted()` is False (still in warmup): appends `x` to
              the buffer, returns `x` unchanged. Once the buffer reaches
              `warmup_size`, fits statistics on the buffer.
            * If `is_fitted()` is True: applies the normalization and
              returns the transformed vector.

    Reproducibility: deterministic — given the same sample sequence,
    output is bit-exact across runs.
    """

    def __init__(
        self,
        mode: Literal["zscore", "minmax"] = "zscore",
        warmup_size: int = 100,
    ):
        if mode not in ("zscore", "minmax"):
            raise ValueError(f"Unknown mode '{mode}'. Use 'zscore' or 'minmax'.")
        if warmup_size < 2:
            raise ValueError(f"warmup_size must be >= 2, got {warmup_size}.")

        self.mode = mode
        self.warmup_size = warmup_size
        self._buffer: list[np.ndarray] = []
        self._fitted = False

        # Statistics, populated on fit
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None

    def is_fitted(self) -> bool:
        return self._fitted

    def _fit_from_buffer(self) -> None:
        if len(self._buffer) < 2:
            raise RuntimeError("Not enough samples in warmup buffer to fit.")
        X = np.asarray(self._buffer, dtype=np.float64)
        if self.mode == "zscore":
            self._mean = X.mean(axis=0)
            std = X.std(axis=0, ddof=1)
            # Replace zero std with small fallback to avoid divide-by-zero
            std[std == 0] = 1e-8
            self._std = std
        else:  # minmax
            self._min = X.min(axis=0)
            self._max = X.max(axis=0)
        self._fitted = True

    def update_and_transform(self, x: np.ndarray) -> np.ndarray:
        """Process a single sample: accumulate during warmup, transform after.

        Returns the (raw or normalized) feature vector. Always returns a
        new array — does not modify `x` in place.
        """
        x = np.asarray(x, dtype=np.float64)

        if not self._fitted:
            self._buffer.append(x.copy())
            if len(self._buffer) >= self.warmup_size:
                self._fit_from_buffer()
            # During warmup: return raw, even when buffer just filled
            return x.copy()

        return self._transform(x)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform a single sample (must be fitted)."""
        if not self._fitted:
            raise RuntimeError("FeatureNormalizer not fitted; call update_and_transform first.")
        x = np.asarray(x, dtype=np.float64)
        return self._transform(x)

    def _transform(self, x: np.ndarray) -> np.ndarray:
        if self.mode == "zscore":
            return (x - self._mean) / self._std

        # minmax
        rng = self._max - self._min
        # Constant feature: output 0 (no information)
        out = np.zeros_like(x, dtype=np.float64)
        nonzero = rng != 0
        out[nonzero] = (x[nonzero] - self._min[nonzero]) / rng[nonzero]
        return out

    def reset(self) -> None:
        """Reset internal state (e.g. between experiment runs)."""
        self._buffer = []
        self._fitted = False
        self._mean = None
        self._std = None
        self._min = None
        self._max = None
