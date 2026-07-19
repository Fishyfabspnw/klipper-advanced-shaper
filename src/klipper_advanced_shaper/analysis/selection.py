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


def select_candidate(
    candidates: Iterable[CandidateScore], profile: SelectionProfile
) -> SelectionResult:
    values = list(candidates)
    if not values:
        return SelectionResult(None, [], {}, "no candidates")
    if len({candidate.name for candidate in values}) != len(values):
        return SelectionResult(None, pareto_frontier(values), {}, "candidate names must be unique")
    eligible = [
        c
        for c in values
        if (profile.maximum_residual is None or c.residual_vibration <= profile.maximum_residual)
        and (
            profile.maximum_cross_axis is None or c.cross_axis_energy <= profile.maximum_cross_axis
        )
    ]
    if not eligible:
        return SelectionResult(
            None, pareto_frontier(values), {}, "no candidate passed profile safety gates"
        )
    frontier = pareto_frontier(eligible)
    utilities = {candidate.name: 0.0 for candidate in frontier}
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
            utilities[candidate.name] += float(weight * score)
    selected = max(
        frontier,
        key=lambda item: (
            utilities[item.name],
            -item.residual_vibration,
            item.max_accel,
            item.name,
        ),
    )
    return SelectionResult(selected, frontier, utilities)
