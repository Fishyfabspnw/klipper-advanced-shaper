"""Modal peak and damping estimation without SciPy dependencies."""

from __future__ import annotations

from typing import List

import numpy as np

from .models import ModeEstimate


def find_modes(
    frequencies: np.ndarray,
    spectrum: np.ndarray,
    *,
    min_frequency: float = 5.0,
    max_frequency: float = 200.0,
    min_prominence_ratio: float = 0.05,
    min_separation_hz: float = 3.0,
) -> List[ModeEstimate]:
    """Find separated local maxima and estimate half-power damping."""
    f = np.asarray(frequencies, dtype=np.float64)
    p = np.asarray(spectrum, dtype=np.float64)
    if f.ndim != 1 or p.shape != f.shape or f.size < 3:
        raise ValueError("frequency and spectrum arrays must be matching 1-D arrays")
    mask = (f >= min_frequency) & (f <= max_frequency) & np.isfinite(p)
    indices = np.flatnonzero(mask)
    if indices.size < 3:
        return []
    floor = float(np.median(p[indices]))
    ceiling = float(np.max(p[indices]))
    candidates = []
    for i in indices[1:-1]:
        if p[i] > p[i - 1] and p[i] >= p[i + 1]:
            prominence = max(0.0, float(p[i] - floor))
            if prominence >= max(ceiling - floor, 0.0) * min_prominence_ratio:
                candidates.append((i, prominence))
    accepted = []
    for i, prominence in sorted(candidates, key=lambda item: p[item[0]], reverse=True):
        if all(abs(f[i] - f[j]) >= min_separation_hz for j, _ in accepted):
            accepted.append((i, prominence))

    modes = []
    for i, prominence in sorted(accepted, key=lambda item: f[item[0]]):
        half = p[i] / 2.0
        left = i
        while left > 0 and p[left] > half:
            left -= 1
        right = i
        while right < p.size - 1 and p[right] > half:
            right += 1
        damping = None
        if left < i < right and p[left] <= half and p[right] <= half and f[i] > 0:
            damping = float((f[right] - f[left]) / (2.0 * f[i]))
        modes.append(ModeEstimate(float(f[i]), float(p[i]), prominence, damping))
    return modes
