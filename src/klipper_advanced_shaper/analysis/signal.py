"""Timestamp normalization and capture quality checks."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .models import QualityIssue, QualityReport


def _vectors(timestamps: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(timestamps, dtype=np.float64)
    x = np.asarray(values, dtype=np.float64)
    if t.ndim != 1 or x.shape[0] != t.size:
        raise ValueError("timestamps must be 1-D and match the first value dimension")
    if t.size < 4:
        raise ValueError("at least four samples are required")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(x)):
        raise ValueError("capture contains non-finite samples")
    if np.any(np.diff(t) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    return t, x


def infer_sample_rate(timestamps: np.ndarray) -> float:
    """Estimate sample rate robustly from the median timestamp interval."""
    t, _ = _vectors(timestamps, np.zeros(len(timestamps)))
    return float(1.0 / np.median(np.diff(t)))


def resample_uniform(
    timestamps: np.ndarray,
    values: np.ndarray,
    sample_rate: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Linearly resample timestamped samples onto a uniform, endpoint-safe grid."""
    t, x = _vectors(timestamps, values)
    rate = infer_sample_rate(t) if sample_rate is None else float(sample_rate)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("sample_rate must be positive and finite")
    count = int(np.floor((t[-1] - t[0]) * rate)) + 1
    if count < 4:
        raise ValueError("resampled capture would be too short")
    uniform_t = t[0] + np.arange(count, dtype=np.float64) / rate
    if x.ndim == 1:
        uniform_x = np.interp(uniform_t, t, x)
    else:
        flat = x.reshape((x.shape[0], -1))
        uniform_x = np.column_stack(
            [np.interp(uniform_t, t, flat[:, index]) for index in range(flat.shape[1])]
        ).reshape((count,) + x.shape[1:])
    return uniform_t, uniform_x, rate


def assess_quality(
    timestamps: np.ndarray,
    values: np.ndarray,
    *,
    expected_band_max: Optional[float] = None,
    clip_limit: Optional[float] = None,
    max_jitter_ratio: float = 0.12,
    max_dropout_ratio: float = 0.01,
    max_clipped_fraction: float = 0.001,
    max_noise_ratio: float = 0.35,
    min_nyquist_margin: float = 1.25,
) -> QualityReport:
    """Apply conservative, explainable capture QC gates.

    Noise is estimated from successive-difference energy relative to centered signal
    energy. It is a screening metric, not a physical sensor noise measurement.
    """
    t, x = _vectors(timestamps, values)
    dt = np.diff(t)
    median_dt = float(np.median(dt))
    rate = 1.0 / median_dt
    jitter = float(np.median(np.abs(dt - median_dt)) / median_dt)
    dropouts = float(np.mean(dt > median_dt * 1.5))
    centered = x - np.mean(x, axis=0)
    signal_rms = float(np.sqrt(np.mean(centered * centered)))
    noise_rms = float(np.sqrt(np.mean(np.diff(x, axis=0) ** 2) / 2.0))
    noise_ratio = noise_rms / max(signal_rms, np.finfo(float).eps)
    if clip_limit is None:
        clipped = 0.0
    else:
        clipped = float(np.mean(np.abs(x) >= abs(float(clip_limit)) * 0.999))
    nyquist_margin = float("inf") if not expected_band_max else rate / (2.0 * expected_band_max)

    issues = []
    if jitter > max_jitter_ratio:
        issues.append(QualityIssue("timestamp_jitter", "timestamp jitter exceeds limit"))
    if dropouts > max_dropout_ratio:
        issues.append(QualityIssue("sample_dropout", "sample dropout rate exceeds limit"))
    if clipped > max_clipped_fraction:
        issues.append(QualityIssue("sensor_clipping", "sensor clipping exceeds limit"))
    if noise_ratio > max_noise_ratio:
        issues.append(QualityIssue("excess_noise", "successive-difference noise exceeds limit"))
    if nyquist_margin < min_nyquist_margin:
        issues.append(QualityIssue("aliasing_risk", "Nyquist margin is insufficient"))
    return QualityReport(
        passed=not issues,
        sample_rate=rate,
        jitter_ratio=jitter,
        dropout_ratio=dropouts,
        clipped_fraction=clipped,
        noise_ratio=noise_ratio,
        nyquist_margin=nyquist_margin,
        issues=tuple(issues),
    )
