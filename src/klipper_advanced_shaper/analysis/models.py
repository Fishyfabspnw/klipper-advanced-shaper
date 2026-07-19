"""Data models shared by the numerical analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Spectrum:
    frequencies: np.ndarray
    values: np.ndarray
    sample_rate: float
    segments: int


@dataclass(frozen=True)
class TransferMetrics:
    frequencies: np.ndarray
    transfer: np.ndarray
    coherence: np.ndarray
    sample_rate: float
    segments: int


@dataclass(frozen=True)
class QualityIssue:
    code: str
    message: str
    severity: str = "error"


@dataclass(frozen=True)
class QualityReport:
    passed: bool
    sample_rate: float
    jitter_ratio: float
    dropout_ratio: float
    clipped_fraction: float
    noise_ratio: float
    nyquist_margin: float
    issues: Tuple[QualityIssue, ...] = ()


@dataclass(frozen=True)
class ModeEstimate:
    frequency: float
    amplitude: float
    prominence: float
    damping_ratio: Optional[float]


@dataclass(frozen=True)
class CandidateScore:
    name: str
    frequency: float
    residual_vibration: float
    smoothing: float
    max_accel: float
    repeatability: float
    cross_axis_energy: float
    sensitivity: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    candidate_id: Optional[str] = None


@dataclass(frozen=True)
class SelectionProfile:
    name: str
    weights: Dict[str, float]
    maximum_residual: Optional[float] = None
    maximum_cross_axis: Optional[float] = None
    minimum_parameterized_smoothing_gain: Optional[float] = None


@dataclass(frozen=True)
class SelectionResult:
    selected: Optional[CandidateScore]
    frontier: List[CandidateScore]
    utilities: Dict[str, float]
    abstention_reason: Optional[str] = None
