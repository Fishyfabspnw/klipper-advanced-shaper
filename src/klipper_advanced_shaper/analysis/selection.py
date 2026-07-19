"""Multi-objective candidate filtering and profile-based selection."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

from .models import CandidateScore, SelectionProfile, SelectionResult

MINIMIZE = ("residual_vibration", "smoothing", "repeatability", "cross_axis_energy", "sensitivity")
MAXIMIZE = ("max_accel",)

PROFILES: Dict[str, SelectionProfile] = {
    "quality": SelectionProfile(
        "quality",
        {
            "residual_vibration": 0.42,
            "cross_axis_energy": 0.20,
            "repeatability": 0.16,
            "sensitivity": 0.12,
            "smoothing": 0.07,
            "max_accel": 0.03,
        },
        maximum_residual=0.10,
    ),
    "balanced": SelectionProfile(
        "balanced",
        {
            "residual_vibration": 0.28,
            "max_accel": 0.23,
            "smoothing": 0.18,
            "repeatability": 0.12,
            "cross_axis_energy": 0.11,
            "sensitivity": 0.08,
        },
        maximum_residual=0.15,
    ),
    "performance": SelectionProfile(
        "performance",
        {
            "max_accel": 0.38,
            "smoothing": 0.24,
            "residual_vibration": 0.18,
            "repeatability": 0.08,
            "cross_axis_energy": 0.07,
            "sensitivity": 0.05,
        },
        maximum_residual=0.20,
    ),
    "experimental_mzv": SelectionProfile(
        "experimental_mzv",
        {
            "max_accel": 0.38,
            "smoothing": 0.24,
            "residual_vibration": 0.18,
            "repeatability": 0.08,
            "cross_axis_energy": 0.07,
            "sensitivity": 0.05,
        },
        maximum_residual=0.10,
        minimum_parameterized_smoothing_gain=0.05,
    ),
    "adaptive_stock": SelectionProfile(
        "adaptive_stock",
        {
            "max_accel": 0.34,
            "smoothing": 0.22,
            "residual_vibration": 0.22,
            "repeatability": 0.09,
            "cross_axis_energy": 0.08,
            "sensitivity": 0.07,
        },
        maximum_residual=0.10,
        minimum_parameterized_smoothing_gain=0.05,
    ),
}


def _dominates(left: CandidateScore, right: CandidateScore) -> bool:
    no_worse = all(getattr(left, key) <= getattr(right, key) for key in MINIMIZE)
    no_worse = no_worse and all(getattr(left, key) >= getattr(right, key) for key in MAXIMIZE)
    better = any(getattr(left, key) < getattr(right, key) for key in MINIMIZE)
    better = better or any(getattr(left, key) > getattr(right, key) for key in MAXIMIZE)
    return no_worse and better


def pareto_frontier(candidates: Iterable[CandidateScore]) -> List[CandidateScore]:
    values = list(candidates)
    return [
        candidate
        for candidate in values
        if not any(_dominates(other, candidate) for other in values if other is not candidate)
    ]


def eligible_candidates(
    candidates: Iterable[CandidateScore], profile: SelectionProfile
) -> List[CandidateScore]:
    """Return candidates that pass the profile's common safety gates."""
    return [
        candidate
        for candidate in candidates
        if (
            profile.maximum_residual is None
            or candidate.residual_vibration <= profile.maximum_residual
        )
        and (
            profile.maximum_cross_axis is None
            or candidate.cross_axis_energy <= profile.maximum_cross_axis
        )
    ]


def _candidate_key(candidate: CandidateScore) -> str:
    return candidate.candidate_id or candidate.name


def select_candidate(
    candidates: Iterable[CandidateScore], profile: SelectionProfile
) -> SelectionResult:
    values = list(candidates)
    if not values:
        return SelectionResult(None, [], {}, "no candidates")
    if len({_candidate_key(candidate) for candidate in values}) != len(values):
        return SelectionResult(
            None, pareto_frontier(values), {}, "candidate identities must be unique"
        )
    eligible = eligible_candidates(values, profile)
    if not eligible:
        return SelectionResult(
            None, pareto_frontier(values), {}, "no candidate passed profile safety gates"
        )
    frontier = pareto_frontier(eligible)
    utilities = {_candidate_key(candidate): 0.0 for candidate in frontier}
    for metric, weight in profile.weights.items():
        raw = np.asarray([getattr(candidate, metric) for candidate in frontier], dtype=float)
        if not np.all(np.isfinite(raw)):
            return SelectionResult(
                None, frontier, {}, "candidate metrics contain non-finite values"
            )
        spread = float(np.ptp(raw))
        normalized = np.ones_like(raw) if spread == 0 else (raw - np.min(raw)) / spread
        if metric in MINIMIZE:
            normalized = 1.0 - normalized
        for candidate, score in zip(frontier, normalized):
            utilities[_candidate_key(candidate)] += float(weight * score)
    selected = max(
        frontier,
        key=lambda item: (
            utilities[_candidate_key(item)],
            -item.residual_vibration,
            item.max_accel,
            _candidate_key(item),
        ),
    )
    return SelectionResult(selected, frontier, utilities)
