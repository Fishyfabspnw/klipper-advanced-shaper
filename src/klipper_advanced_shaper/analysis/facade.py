"""End-to-end calibration orchestration built from the numerical primitives."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

import numpy as np

from klipper_advanced_shaper import __version__

from .models import CandidateScore, Spectrum
from .modes import find_modes
from .selection import PROFILES, select_candidate
from .signal import assess_quality, resample_uniform
from .spectral import aggregate_spectra, integrated_band_energy, welch_psd
from .statistics import attenuation_improvement_ci

_MAX_NATIVE_BINS = 1024
_MAX_SPECTROGRAM_FREQUENCIES = 256
_MAX_SPECTROGRAM_TIMES = 192


def _samples(capture: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(capture["samples"], dtype=float)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("capture samples must be rows of time,x,y,z")
    return values[:, 0], values[:, 1:]


def _axis_spectra(
    captures: Sequence[Mapping[str, Any]], axis: str, expected_max: float = 200.0
) -> tuple[list[Spectrum], list[Spectrum], list[dict[str, Any]]]:
    direction = np.array([1.0, 0.0, 0.0] if axis == "X" else [0.0, 1.0, 0.0])
    along_spectra: list[Spectrum] = []
    cross_spectra: list[Spectrum] = []
    quality = []
    for capture in captures:
        timestamps, acceleration = _samples(capture)
        clip_limit = capture.get("metadata", {}).get("clip_limit")
        qc = assess_quality(
            timestamps,
            acceleration,
            expected_band_max=expected_max,
            clip_limit=clip_limit,
            max_noise_ratio=1.1,
        )
        quality_row = asdict(qc)
        if clip_limit is None:
            quality_row["passed"] = False
            quality_row["issues"] = list(quality_row["issues"]) + [
                {
                    "code": "unknown_clip_limit",
                    "message": "sensor full-scale range is unknown",
                    "severity": "error",
                }
            ]
        quality.append(quality_row)
        if not quality_row["passed"]:
            continue
        _, uniform, rate = resample_uniform(timestamps, acceleration)
        axis_index = int(np.argmax(direction))
        along = uniform[:, axis_index]
        orthogonal = np.delete(uniform, axis_index, axis=1)
        nperseg = max(8, min(len(along), 1 << max(3, int(np.ceil(np.log2(rate * 0.5))))))
        along_spectra.append(welch_psd(along, rate, nperseg=nperseg))
        component_spectra = [
            welch_psd(orthogonal[:, index], rate, nperseg=nperseg)
            for index in range(orthogonal.shape[1])
        ]
        cross_spectra.append(
            Spectrum(
                component_spectra[0].frequencies,
                sum(item.values for item in component_spectra),
                rate,
                min(item.segments for item in component_spectra),
            )
        )
    return along_spectra, cross_spectra, quality


def _candidate_scores(
    captures: Sequence[Mapping[str, Any]], repeatability: float, cross_ratio: float
) -> list[CandidateScore]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for capture in captures:
        for item in capture.get("native_candidates", []):
            grouped.setdefault(str(item["name"]).lower(), []).append(item)
    scores = []
    for name, rows in grouped.items():
        frequency = np.asarray([float(row["frequency"]) for row in rows])
        scores.append(
            CandidateScore(
                name=name,
                frequency=float(np.median(frequency)),
                residual_vibration=float(np.median([row["residual_vibration"] for row in rows])),
                smoothing=float(np.median([row["smoothing"] for row in rows])),
                max_accel=float(np.median([row["max_accel"] for row in rows])),
                repeatability=max(
                    repeatability, float(np.std(frequency) / max(np.mean(frequency), 1e-9))
                ),
                cross_axis_energy=cross_ratio,
                sensitivity=float(np.ptp(frequency) / max(np.mean(frequency), 1e-9)),
            )
        )
    return scores


def _common_grid(rows: Sequence[Mapping[str, Any]], frequency_key: str) -> np.ndarray:
    grids = [np.asarray(row[frequency_key], dtype=float) for row in rows]
    lower = max(float(grid[0]) for grid in grids)
    upper = min(float(grid[-1]) for grid in grids)
    reference = min(grids, key=lambda grid: np.count_nonzero((grid >= lower) & (grid <= upper)))
    grid = reference[(reference >= lower) & (reference <= upper)]
    if grid.size < 2:
        raise ValueError("native spectra have no common frequency range")
    if grid.size > _MAX_NATIVE_BINS:
        indices = np.linspace(0, grid.size - 1, _MAX_NATIVE_BINS, dtype=int)
        grid = grid[indices]
    return grid


def _native_spectrum_summary(captures: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    required = ("frequency_hz", "psd_sum", "psd_x", "psd_y", "psd_z")
    rows = [capture.get("native_spectrum") for capture in captures]
    if not rows or any(not isinstance(row, Mapping) for row in rows):
        return {"available": False, "reason": "native CalibrationData spectrum unavailable"}
    unavailable = [
        str(row.get("reason", "native spectrum unavailable"))
        for row in rows
        if not row.get("available", True)
    ]
    if unavailable:
        return {"available": False, "reason": unavailable[0]}
    try:
        for row in rows:
            frequency = np.asarray(row["frequency_hz"], dtype=float)
            if frequency.ndim != 1 or frequency.size < 2 or np.any(np.diff(frequency) <= 0):
                raise ValueError("invalid native frequency bins")
            for key in required[1:]:
                values = np.asarray(row[key], dtype=float)
                if values.shape != frequency.shape or not np.all(np.isfinite(values)):
                    raise ValueError("invalid native PSD component")
        grid = _common_grid(rows, "frequency_hz")
        result: dict[str, Any] = {
            "available": True,
            "source": "running_klipper_normalized_calibration_data",
            "frequency_hz": grid.tolist(),
            "repeat_count": len(rows),
        }
        for key in required[1:]:
            matrix = np.vstack(
                [
                    np.interp(grid, row["frequency_hz"], np.asarray(row[key], dtype=float))
                    for row in rows
                ]
            )
            result[key] = np.median(matrix, axis=0).tolist()
        return result
    except (KeyError, TypeError, ValueError) as error:
        return {"available": False, "reason": str(error)}


def _native_candidate_summary(captures: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for capture in captures:
        for item in capture.get("native_candidates", []):
            grouped.setdefault(str(item["name"]).lower(), []).append(item)
    result = []
    for name, rows in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "name": name,
            "frequency_hz": float(np.median([row["frequency"] for row in rows])),
            "residual_vibration": float(np.median([row["residual_vibration"] for row in rows])),
            "smoothing": float(np.median([row["smoothing"] for row in rows])),
            "max_accel": float(np.median([row["max_accel"] for row in rows])),
            "repeat_count": len(rows),
        }
        responses = [row.get("native_frequency_response") for row in rows]
        if responses and all(isinstance(response, Mapping) for response in responses):
            try:
                grid = _common_grid(responses, "frequency_hz")
                matrix = np.vstack(
                    [
                        np.interp(
                            grid,
                            response["frequency_hz"],
                            np.asarray(response["response_ratio"], dtype=float),
                        )
                        for response in responses
                    ]
                )
                if np.all(np.isfinite(matrix)):
                    summary["native_frequency_response"] = {
                        "frequency_hz": grid.tolist(),
                        "response_ratio": np.median(matrix, axis=0).tolist(),
                    }
            except (KeyError, TypeError, ValueError):
                pass
        result.append(summary)
    return result


def _spectrogram(capture: Mapping[str, Any], axis: str) -> dict[str, Any]:
    if int(capture.get("dataset_count", 1)) != 1:
        return {
            "available": False,
            "reason": "multiple probe sweeps do not share one excitation time axis",
        }
    recipe = capture.get("metadata", {}).get("test_recipe", {})
    try:
        start_frequency = float(recipe["freq_start"])
        end_frequency = float(recipe["freq_end"])
        sweep_rate = float(recipe["hz_per_sec"])
    except (KeyError, TypeError, ValueError):
        return {"available": False, "reason": "timestamped sweep recipe unavailable"}
    if start_frequency < 0 or end_frequency <= start_frequency or sweep_rate <= 0:
        return {"available": False, "reason": "invalid sweep recipe"}
    try:
        timestamps, acceleration = _samples(capture)
        uniform_time, uniform, rate = resample_uniform(timestamps, acceleration)
        axis_index = 0 if axis == "X" else 1
        signal = uniform[:, axis_index]
        target = max(64, int(rate * 0.25))
        size = min(1024, 1 << max(6, int(np.floor(np.log2(target)))))
        if signal.size < 2 * size:
            raise ValueError("capture is too short for a stable spectrogram")
        step = max(1, size // 4)
        starts = np.arange(0, signal.size - size + 1, step, dtype=int)
        window = np.hanning(size)
        scale = rate * np.sum(window**2)
        columns = []
        for offset in starts:
            segment = signal[offset : offset + size]
            transformed = np.fft.rfft((segment - np.mean(segment)) * window)
            power = np.abs(transformed) ** 2 / scale
            power[1:-1] *= 2.0
            columns.append(power)
        frequencies = np.fft.rfftfreq(size, 1.0 / rate)
        matrix = np.asarray(columns, dtype=float).T
        frequency_mask = frequencies <= min(rate * 0.5, end_frequency + 20.0)
        frequencies = frequencies[frequency_mask]
        matrix = matrix[frequency_mask]
        times = uniform_time[starts + size // 2] - uniform_time[0]
        if frequencies.size > _MAX_SPECTROGRAM_FREQUENCIES:
            indices = np.linspace(
                0, frequencies.size - 1, _MAX_SPECTROGRAM_FREQUENCIES, dtype=int
            )
            frequencies = frequencies[indices]
            matrix = matrix[indices]
        if times.size > _MAX_SPECTROGRAM_TIMES:
            indices = np.linspace(0, times.size - 1, _MAX_SPECTROGRAM_TIMES, dtype=int)
            times = times[indices]
            matrix = matrix[:, indices]
        if not np.all(np.isfinite(matrix)):
            raise ValueError("spectrogram contains non-finite power")
        return {
            "available": True,
            "source": "timestamp_resampled_sweep",
            "axis": axis,
            "frequency_hz": frequencies.tolist(),
            "time_s": times.tolist(),
            "power": matrix.tolist(),
            "power_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
            "test_recipe": {
                "freq_start": start_frequency,
                "freq_end": end_frequency,
                "hz_per_sec": sweep_rate,
            },
        }
    except (KeyError, TypeError, ValueError) as error:
        return {"available": False, "reason": str(error)}


def _energies(
    captures: Sequence[Mapping[str, Any]], axis: str, bands: Sequence[tuple[float, float]]
) -> tuple[np.ndarray, np.ndarray]:
    along, cross, quality = _axis_spectra(captures, axis)
    if len(along) != len(captures) or any(not row["passed"] for row in quality):
        raise ValueError("held-out capture failed quality checks")
    along_energy = np.asarray(
        [sum(integrated_band_energy(item, low, high) for low, high in bands) for item in along]
    )
    cross_energy = np.asarray(
        [sum(integrated_band_energy(item, low, high) for low, high in bands) for item in cross]
    )
    return along_energy, cross_energy


def analyze_calibration(
    *,
    captures: Mapping[str, Sequence[Mapping[str, Any]]],
    axes: Sequence[str],
    profile: str,
    snapshot: Any,
    held_out_captures: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    validation_captures: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    prior_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze training captures or judge shaped captures against held-out baselines."""
    if validation_captures is not None:
        details = {}
        passed = True
        for axis in axes:
            modes = (prior_report or {}).get("axes", {}).get(axis, {}).get("modes", [])
            bands = [
                (max(5.0, m["frequency"] - 5.0), min(200.0, m["frequency"] + 5.0)) for m in modes
            ]
            if not bands:
                return {"validation": {"passed": False, "reason": "no modal bands"}}
            baseline, baseline_cross = _energies(held_out_captures[axis], axis, bands)
            shaped, shaped_cross = _energies(validation_captures[axis], axis, bands)
            low, high = attenuation_improvement_ci(baseline, shaped)
            cross_regression = float(
                (np.mean(shaped_cross) - np.mean(baseline_cross))
                / max(np.mean(baseline_cross), 1e-12)
            )
            axis_passed = low >= 0.10 and cross_regression <= 0.05
            passed = passed and axis_passed
            details[axis] = {
                "baseline_energy": float(np.mean(baseline)),
                "shaped_energy": float(np.mean(shaped)),
                "improvement_ci_95": [low, high],
                "reference_cross_axis_energy": float(np.mean(baseline_cross)),
                "candidate_cross_axis_energy": float(np.mean(shaped_cross)),
                "cross_axis_regression": cross_regression,
                "passed": axis_passed,
            }
        return {
            "validation": {
                "passed": passed,
                "axes": details,
                "reason": None
                if passed
                else "10% attenuation or 5% cross-axis regression gate not met",
            }
        }

    report: dict[str, Any] = {
        "schema_version": "1.0.0-alpha.1",
        "engine": "robust_v1+running_klipper_reference",
        "plugin_version": __version__,
        "provenance": dict(captures[axes[0]][0].get("metadata", {})),
        "profile": profile,
        "square_corner_velocity": float(snapshot.square_corner_velocity),
        "axes": {},
        "selections": [],
    }
    for axis in axes:
        along, cross, quality = _axis_spectra(captures[axis], axis)
        if len(along) != len(captures[axis]):
            return {"abstain": True, "reason": "%s capture failed QC" % axis, "qc": quality}
        aggregate, mad = aggregate_spectra(along)
        cross_aggregate, _ = aggregate_spectra(cross)
        modes = find_modes(aggregate.frequencies, aggregate.values)
        if not modes:
            return {"abstain": True, "reason": "%s has no identifiable modes" % axis}
        main_energy = integrated_band_energy(aggregate, 5.0, min(200.0, aggregate.frequencies[-1]))
        cross_energy = integrated_band_energy(
            cross_aggregate, 5.0, min(200.0, cross_aggregate.frequencies[-1])
        )
        scores = _candidate_scores(
            captures[axis], float(np.median(mad)), cross_energy / max(main_energy, 1e-12)
        )
        if not scores:
            return {"abstain": True, "reason": "%s native candidate data unavailable" % axis}
        chosen = select_candidate(scores, PROFILES[profile])
        if chosen.selected is None:
            return {"abstain": True, "reason": "%s: %s" % (axis, chosen.abstention_reason)}
        damping = 0.1
        stride = max(1, len(aggregate.frequencies) // 512)
        report["selections"].append(
            {
                "axis": axis,
                "shaper_type": chosen.selected.name,
                "frequency_hz": chosen.selected.frequency,
                "damping_ratio": damping,
            }
        )
        report["axes"][axis] = {
            "qc": quality,
            "modes": [asdict(mode) for mode in modes],
            "candidates": [asdict(item) for item in scores],
            "native_candidates": _native_candidate_summary(captures[axis]),
            "pareto": [item.name for item in chosen.frontier],
            "selected": chosen.selected.name,
            "native_spectrum": _native_spectrum_summary(captures[axis]),
            "spectrogram": _spectrogram(captures[axis][0], axis),
            "spectrum": {
                "frequency_hz": aggregate.frequencies[::stride].tolist(),
                "psd": aggregate.values[::stride].tolist(),
                "relative_mad": mad[::stride].tolist(),
            },
        }
    report["native_command_preview"] = "SET_INPUT_SHAPER " + " ".join(
        "SHAPER_TYPE_%s=%s SHAPER_FREQ_%s=%.3f DAMPING_RATIO_%s=%.4f"
        % (
            item["axis"],
            item["shaper_type"],
            item["axis"],
            item["frequency_hz"],
            item["axis"],
            item["damping_ratio"],
        )
        for item in report["selections"]
    )
    return report
