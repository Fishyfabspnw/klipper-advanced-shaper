"""Small deterministic statistical helpers."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np


def bootstrap_confidence_interval(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: Optional[int] = 0,
) -> Tuple[float, float]:
    """Return a percentile bootstrap interval for independent repeat metrics."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size < 2 or not np.all(np.isfinite(data)):
        raise ValueError("at least two finite scalar observations are required")
    if not 0 < confidence < 1 or resamples < 100:
        raise ValueError("invalid confidence or resample count")
    rng = np.random.default_rng(seed)
    draws = rng.choice(data, size=(resamples, data.size), replace=True)
    estimates = np.asarray([statistic(row) for row in draws], dtype=np.float64)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return float(low), float(high)


def attenuation_improvement_ci(
    baseline_energy: np.ndarray,
    candidate_energy: np.ndarray,
    *,
    confidence: float = 0.95,
    resamples: int = 5000,
    seed: Optional[int] = 0,
) -> Tuple[float, float]:
    """Paired bootstrap CI for fractional energy reduction."""
    base = np.asarray(baseline_energy, dtype=np.float64)
    candidate = np.asarray(candidate_energy, dtype=np.float64)
    if base.shape != candidate.shape or base.ndim != 1 or base.size < 2:
        raise ValueError("matching arrays with at least two held-out repeats are required")
    if np.any(base <= 0) or not np.all(np.isfinite(base)) or not np.all(np.isfinite(candidate)):
        raise ValueError("energies must be finite and baseline energy positive")
    improvements = 1.0 - candidate / base
    return bootstrap_confidence_interval(
        improvements, confidence=confidence, resamples=resamples, seed=seed
    )
