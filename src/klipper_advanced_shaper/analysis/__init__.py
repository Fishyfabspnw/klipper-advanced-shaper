"""Numerical primitives for Klipper Advanced Shaper."""

from .facade import analyze_calibration
from .models import (
    CandidateScore,
    ModeEstimate,
    QualityIssue,
    QualityReport,
    SelectionProfile,
    SelectionResult,
    Spectrum,
    TransferMetrics,
)
from .modes import find_modes
from .selection import PROFILES, pareto_frontier, select_candidate
from .signal import assess_quality, infer_sample_rate, resample_uniform
from .spectral import (
    aggregate_spectra,
    integrated_band_energy,
    project_axes,
    transfer_coherence,
    welch_psd,
)
from .statistics import attenuation_improvement_ci, bootstrap_confidence_interval

__all__ = [
    "CandidateScore",
    "ModeEstimate",
    "PROFILES",
    "QualityIssue",
    "QualityReport",
    "SelectionProfile",
    "SelectionResult",
    "Spectrum",
    "TransferMetrics",
    "aggregate_spectra",
    "analyze_calibration",
    "assess_quality",
    "attenuation_improvement_ci",
    "bootstrap_confidence_interval",
    "find_modes",
    "infer_sample_rate",
    "pareto_frontier",
    "project_axes",
    "integrated_band_energy",
    "resample_uniform",
    "select_candidate",
    "transfer_coherence",
    "welch_psd",
]
