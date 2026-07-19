"""End-to-end calibration orchestration built from the numerical primitives."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from klipper_advanced_shaper import __version__
from klipper_advanced_shaper.shapers import NATIVE_SHAPER_ORDER, parse_shaper_identifier

from .experimental import (
    damping_samples,
    generalized_mzv_pulses,
    optimize_generalized_mzv,
    oscillator_response,
)
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
    captures: Sequence[Mapping[str, Any]],
    repeatability: float,
    cross_ratio: float,
    comparison_spectrum: Optional[Spectrum] = None,
    cross_comparison_spectrum: Optional[Spectrum] = None,
) -> list[CandidateScore]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for capture in captures:
        for item in capture.get("native_candidates", []):
            grouped.setdefault(str(item["name"]).lower(), []).append(item)
    scores = []
    for name, rows in grouped.items():
        frequency = np.asarray([float(row["frequency"]) for row in rows])
        residuals = [float(row["residual_vibration"]) for row in rows]
        residual_metric = "klipper_native_vibration_fraction"
        candidate_cross_axis_energy = cross_ratio
        cross_axis_metadata = {
            "cross_axis_metric": "measured_unshaped_cross_to_main_energy_ratio",
        }
        sensitivity = float(np.ptp(frequency) / max(np.mean(frequency), 1e-9))
        if comparison_spectrum is not None:
            common_residuals = []
            power = np.asarray(comparison_spectrum.values, dtype=float)
            threshold = float(np.max(power)) / 20.0
            denominator = float(np.sum(np.maximum(power - threshold, 0.0)))
            if denominator <= 0.0:
                continue
            for row in rows:
                response = row.get("native_frequency_response")
                if not isinstance(response, Mapping):
                    common_residuals = []
                    break
                try:
                    values = np.interp(
                        comparison_spectrum.frequencies,
                        np.asarray(response["frequency_hz"], dtype=float),
                        np.asarray(response["response_ratio"], dtype=float),
                    )
                except (KeyError, TypeError, ValueError):
                    common_residuals = []
                    break
                if not np.all(np.isfinite(values)) or np.any(values < 0.0):
                    common_residuals = []
                    break
                common_residuals.append(
                    float(np.sum(np.maximum(values * power - threshold, 0.0)))
                    / denominator
                )
            if len(common_residuals) != len(rows):
                continue
            residuals = common_residuals
            residual_metric = "common_klipper_thresholded_vibration_fraction"
            sensitivity = float(np.ptp(common_residuals))
            if cross_comparison_spectrum is None:
                continue
            cross_residuals = []
            for row in rows:
                response = row.get("native_frequency_response")
                assert isinstance(response, Mapping)
                try:
                    cross_residuals.append(
                        _response_weighted_residual(
                            cross_comparison_spectrum,
                            np.asarray(response["frequency_hz"], dtype=float),
                            np.asarray(response["response_ratio"], dtype=float),
                            empty_value=0.0,
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    cross_residuals = []
                    break
            if len(cross_residuals) != len(rows):
                continue
            candidate_cross_axis_energy = float(np.median(cross_residuals))
            cross_axis_metadata = {
                "cross_axis_metric": "predicted_cross_axis_residual_fraction",
                "cross_axis_model": (
                    "native_frequency_response_weighted_training_cross_psd"
                ),
                "cross_axis_aggregation": "median_across_training_captures",
                "unshaped_cross_to_main_energy_ratio": cross_ratio,
            }
        scores.append(
            CandidateScore(
                name=name,
                frequency=float(np.median(frequency)),
                residual_vibration=float(np.median(residuals)),
                smoothing=float(np.median([row["smoothing"] for row in rows])),
                max_accel=float(np.median([row["max_accel"] for row in rows])),
                repeatability=max(
                    repeatability, float(np.std(frequency) / max(np.mean(frequency), 1e-9))
                ),
                cross_axis_energy=candidate_cross_axis_energy,
                sensitivity=sensitivity,
                metadata={
                    "family": "native",
                    "parameterized": False,
                    "residual_metric": residual_metric,
                    "upstream_residual_vibration": float(
                        np.median([row["residual_vibration"] for row in rows])
                    ),
                    "design_damping_ratio": (
                        float(np.median([row["design_damping_ratio"] for row in rows]))
                        if all("design_damping_ratio" in row for row in rows)
                        else None
                    ),
                    "theoretical_smoothing_acceleration_mm_s2": float(
                        np.median([row["max_accel"] for row in rows])
                    ),
                    "resonance_validated_acceleration_mm_s2": None,
                    "print_validated_acceleration_mm_s2": None,
                    "acceleration_evidence": "theoretical",
                    **cross_axis_metadata,
                },
            )
        )
    return scores


def _response_weighted_residual(
    spectrum: Spectrum,
    response_frequency_hz: np.ndarray,
    response_ratio: np.ndarray,
    *,
    empty_value: Optional[float] = None,
) -> float:
    """Return response-weighted residual power on one measured spectrum."""
    frequencies = np.asarray(response_frequency_hz, dtype=float)
    response = np.asarray(response_ratio, dtype=float)
    if (
        frequencies.ndim != 1
        or response.shape != frequencies.shape
        or frequencies.size < 2
        or not np.all(np.isfinite(frequencies))
        or not np.all(np.isfinite(response))
        or np.any(response < 0.0)
    ):
        raise ValueError("candidate response is not a finite non-negative curve")
    power = np.asarray(spectrum.values, dtype=float)
    threshold = float(np.max(power)) / 20.0
    meaningful = np.maximum(power - threshold, 0.0)
    denominator = float(np.sum(meaningful))
    if denominator <= 0.0:
        if empty_value is not None:
            return float(empty_value)
        raise ValueError("comparison spectrum has no meaningful power")
    interpolated = np.interp(spectrum.frequencies, frequencies, response)
    return float(np.sum(np.maximum(interpolated * power - threshold, 0.0))) / denominator


def _generalized_cross_axis_residual(
    spectrum: Spectrum,
    candidate: Mapping[str, Any],
    damping_uncertainty: Sequence[float],
) -> tuple[float, dict[str, Any]]:
    """Conservatively predict generalized-MZV cross residual over damping uncertainty."""
    amplitudes, times = generalized_mzv_pulses(
        int(candidate["pulse_count"]),
        float(candidate["spacing"]),
        float(candidate["frequency_hz"]),
        float(candidate["design_damping_ratio"]),
    )
    values = []
    samples = [float(value) for value in damping_uncertainty]
    if not samples:
        raise ValueError("generalized cross-axis model requires damping uncertainty")
    for damping in samples:
        response = oscillator_response(
            amplitudes,
            times,
            spectrum.frequencies,
            damping,
        )
        values.append(
            _response_weighted_residual(
                spectrum,
                spectrum.frequencies,
                response,
                empty_value=0.0,
            )
        )
    q95 = float(np.quantile(np.asarray(values, dtype=float), 0.95))
    return q95, {
        "cross_axis_metric": "predicted_cross_axis_residual_fraction_q95",
        "cross_axis_model": "oscillator_response_weighted_training_cross_psd",
        "cross_axis_aggregation": "damping_uncertainty_q95",
        "cross_axis_residual_median": float(np.median(values)),
        "cross_axis_residual_q95": q95,
        "cross_axis_damping_sample_count": len(samples),
    }


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
        if all("design_damping_ratio" in row for row in rows):
            summary["design_damping_ratio"] = float(
                np.median([row["design_damping_ratio"] for row in rows])
            )
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
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], list[dict[str, Any]]]:
    along, cross, quality = _axis_spectra(captures, axis)
    if len(along) != len(captures) or any(not row["passed"] for row in quality):
        return None, None, quality
    along_energy = np.asarray(
        [sum(integrated_band_energy(item, low, high) for low, high in bands) for item in along]
    )
    cross_energy = np.asarray(
        [sum(integrated_band_energy(item, low, high) for low, high in bands) for item in cross]
    )
    return along_energy, cross_energy, quality


def analyze_calibration(
    *,
    captures: Mapping[str, Sequence[Mapping[str, Any]]],
    axes: Sequence[str],
    profile: str,
    snapshot: Any,
    held_out_captures: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
    validation_captures: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
    validation_pair_ids: Optional[Mapping[str, Sequence[str]]] = None,
    prior_report: Optional[Mapping[str, Any]] = None,
    experimental_mode: bool = False,
    executor_pulse_limit: int = 10,
    peak_lock: bool = False,
) -> dict[str, Any]:
    """Analyze training captures or judge shaped captures against held-out baselines."""
    if peak_lock and not experimental_mode:
        return {
            "abstain": True,
            "reason": "strongest-peak locking is only supported in adaptive stock modes",
        }
    if validation_captures is not None:
        details = {}
        passed = True
        for axis in axes:
            explicit_pairing = validation_pair_ids is not None
            modes = (prior_report or {}).get("axes", {}).get(axis, {}).get("modes", [])
            bands = [
                (max(5.0, m["frequency"] - 5.0), min(200.0, m["frequency"] + 5.0)) for m in modes
            ]
            if not bands:
                return {"validation": {"passed": False, "reason": "no modal bands"}}
            try:
                pair_ids = (
                    list(validation_pair_ids[axis])
                    if validation_pair_ids is not None
                    else [
                        "%s-%02d" % (axis, index + 1)
                        for index in range(len(held_out_captures[axis]))
                    ]
                )
                if (
                    len(pair_ids) != len(held_out_captures[axis])
                    or len(pair_ids) != len(validation_captures[axis])
                    or len(set(pair_ids)) != len(pair_ids)
                ):
                    raise ValueError("validation pair IDs must uniquely match both capture lists")
                baseline, baseline_cross, reference_qc = _energies(
                    held_out_captures[axis], axis, bands
                )
                shaped, shaped_cross, candidate_qc = _energies(
                    validation_captures[axis], axis, bands
                )
            except (KeyError, TypeError, ValueError) as error:
                details[axis] = {
                    "passed": False,
                    "qc_passed": False,
                    "reason": str(error),
                }
                passed = False
                continue
            if baseline is None or shaped is None:
                details[axis] = {
                    "passed": False,
                    "qc_passed": False,
                    "reason": "held-out capture failed quality checks",
                    "reference_qc": reference_qc,
                    "candidate_qc": candidate_qc,
                }
                passed = False
                continue
            assert baseline_cross is not None and shaped_cross is not None
            low, high = attenuation_improvement_ci(baseline, shaped)
            cross_regression = float(
                (np.mean(shaped_cross) - np.mean(baseline_cross))
                / max(np.mean(baseline_cross), 1e-12)
            )
            axis_passed = low >= 0.10 and cross_regression <= 0.05
            passed = passed and axis_passed
            details[axis] = {
                "energy_units": "acceleration_squared",
                "baseline_energy": float(np.mean(baseline)),
                "shaped_energy": float(np.mean(shaped)),
                "reference_energy_samples": baseline.tolist(),
                "candidate_energy_samples": shaped.tolist(),
                "improvement_ci_95": [low, high],
                "reference_cross_axis_energy": float(np.mean(baseline_cross)),
                "candidate_cross_axis_energy": float(np.mean(shaped_cross)),
                "reference_cross_axis_energy_samples": baseline_cross.tolist(),
                "candidate_cross_axis_energy_samples": shaped_cross.tolist(),
                "cross_axis_regression": cross_regression,
                "qc_passed": True,
                "reference_qc": reference_qc,
                "candidate_qc": candidate_qc,
                "pair_ids": pair_ids,
                "pair_count": len(pair_ids),
                "paired_energy_observations": [
                    {
                        "pair_id": pair_id,
                        "reference_energy": float(reference_energy),
                        "candidate_energy": float(candidate_energy),
                        "reference_cross_axis_energy": float(reference_cross),
                        "candidate_cross_axis_energy": float(candidate_cross),
                    }
                    for pair_id, reference_energy, candidate_energy,
                    reference_cross, candidate_cross in zip(
                        pair_ids, baseline, shaped, baseline_cross, shaped_cross
                    )
                ],
                "capture_design": (
                    "paired_interleaved_ab"
                    if explicit_pairing
                    else "paired_by_list_position_order_unverified"
                ),
                "passed": axis_passed,
            }
        return {
            "validation": {
                "passed": passed,
                "axes": details,
                "reason": None
                if passed
                else "QC, 10% attenuation, or 5% cross-axis regression gate not met",
            }
        }

    report: dict[str, Any] = {
        "schema_version": "1.0.0-alpha.2",
        "engine": "robust_v1+running_klipper_reference",
        "plugin_version": __version__,
        "provenance": dict(captures[axes[0]][0].get("metadata", {})),
        "profile": profile,
        "experimental_mode": bool(experimental_mode),
        "peak_lock": bool(peak_lock),
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
            captures[axis],
            float(np.median(mad)),
            cross_energy / max(main_energy, 1e-12),
            aggregate if profile == "adaptive_stock" else None,
            cross_aggregate if profile == "adaptive_stock" else None,
        )
        if not scores:
            return {"abstain": True, "reason": "%s native candidate data unavailable" % axis}
        measured_damping = [
            float(mode.damping_ratio)
            for mode in modes
            if mode.damping_ratio is not None and 0.0 < float(mode.damping_ratio) < 1.0
        ]
        snapshot_damping = float(getattr(snapshot, "damping_ratio_" + axis.lower()))
        measured_design_damping = (
            float(np.median(measured_damping)) if measured_damping else snapshot_damping
        )
        damping_uncertainty_samples = (
            damping_samples([asdict(mode) for mode in modes]).tolist()
            if measured_damping
            else [snapshot_damping]
        )
        generalized_report: Optional[dict[str, Any]] = None
        if experimental_mode:
            if not 3 <= int(executor_pulse_limit) <= 10:
                return {
                    "abstain": True,
                    "reason": "%s installed executor pulse limit is unsupported" % axis,
                }
            if not measured_damping:
                return {
                    "abstain": True,
                    "reason": "%s experimental fitting requires measured damping" % axis,
                }
            try:
                strongest_mode = max(modes, key=lambda mode: float(mode.amplitude))
                peak_frequency = float(strongest_mode.frequency) if peak_lock else None
                generalized_report = optimize_generalized_mzv(
                    aggregate.frequencies,
                    aggregate.values,
                    [asdict(mode) for mode in modes],
                    float(snapshot.square_corner_velocity),
                    pulse_counts=range(3, int(executor_pulse_limit) + 1),
                    fixed_frequency_hz=peak_frequency,
                )
            except (TypeError, ValueError, np.linalg.LinAlgError) as error:
                return {
                    "abstain": True,
                    "reason": "%s generalized-MZV optimization failed: %s" % (axis, error),
                }
            generalized_by_name: dict[str, Mapping[str, Any]] = {}
            for item in generalized_report.get("pareto", []):
                identifier = parse_shaper_identifier(str(item["shaper_type"]))
                current = generalized_by_name.get(identifier.canonical)
                if current is None or (
                    float(item["smoothing_max_accel"]),
                    -float(item["residual_energy_q95"]),
                    -float(item["sensitivity"]),
                ) > (
                    float(current["smoothing_max_accel"]),
                    -float(current["residual_energy_q95"]),
                    -float(current["sensitivity"]),
                ):
                    generalized_by_name[identifier.canonical] = item
            generalized_report["selection_candidate_count"] = len(generalized_by_name)
            generalized_report["peak_lock_enabled"] = bool(peak_lock)
            generalized_report["strongest_measured_peak_hz"] = float(
                max(modes, key=lambda mode: float(mode.amplitude)).frequency
            )
            for canonical, item in generalized_by_name.items():
                theoretical = float(item["smoothing_max_accel"])
                try:
                    candidate_cross_axis_energy, cross_axis_metadata = (
                        _generalized_cross_axis_residual(
                            cross_aggregate,
                            item,
                            generalized_report["measured_damping_samples"],
                        )
                    )
                except (KeyError, TypeError, ValueError) as error:
                    return {
                        "abstain": True,
                        "reason": "%s generalized cross-axis modeling failed: %s"
                        % (axis, error),
                    }
                scores.append(
                    CandidateScore(
                        name=canonical,
                        frequency=float(item["frequency_hz"]),
                        residual_vibration=float(
                            item["klipper_remaining_vibration"]
                            if profile == "adaptive_stock"
                            else item["residual_energy_q95"]
                        ),
                        smoothing=float(item["path_error_at_5000"]),
                        max_accel=theoretical,
                        repeatability=float(np.median(mad)),
                        cross_axis_energy=candidate_cross_axis_energy,
                        sensitivity=float(item["sensitivity"]),
                        metadata={
                            "family": "generalized_mzv",
                            "parameterized": True,
                            "residual_metric": (
                                "common_klipper_thresholded_vibration_fraction"
                                if profile == "adaptive_stock"
                                else "psd_weighted_squared_response_q95"
                            ),
                            "robust_residual_energy_q95": float(
                                item["residual_energy_q95"]
                            ),
                            "pulse_count": int(item["pulse_count"]),
                            "spacing": float(item["spacing"]),
                            "design_damping_ratio": float(item["design_damping_ratio"]),
                            "damping_uncertainty_samples": list(
                                generalized_report["measured_damping_samples"]
                            ),
                            "theoretical_smoothing_acceleration_mm_s2": theoretical,
                            "resonance_validated_acceleration_mm_s2": None,
                            "print_validated_acceleration_mm_s2": None,
                            "acceleration_evidence": "theoretical",
                            "frequency_strategy": generalized_report["frequency_strategy"],
                            "strongest_measured_peak_hz": generalized_report[
                                "strongest_measured_peak_hz"
                            ],
                            **cross_axis_metadata,
                        },
                    )
                )
        selection_pool = (
            [item for item in scores if item.metadata.get("parameterized")]
            if profile == "experimental_mzv"
            else scores
        )
        if profile == "experimental_mzv" and not selection_pool:
            return {
                "abstain": True,
                "reason": "%s has no generalized-MZV candidate after robust gates" % axis,
            }
        chosen = select_candidate(selection_pool, PROFILES[profile])
        if chosen.selected is None:
            return {"abstain": True, "reason": "%s: %s" % (axis, chosen.abstention_reason)}
        selected_metadata = chosen.selected.metadata
        if selected_metadata.get("parameterized"):
            damping = float(
                selected_metadata.get("design_damping_ratio", measured_design_damping)
            )
            damping_source = "measured_modes"
        else:
            native_damping = selected_metadata.get("design_damping_ratio")
            damping = (
                float(native_damping)
                if native_damping is not None
                else snapshot_damping
            )
            damping_source = (
                "active_input_shaper_status"
                if native_damping is not None
                else "snapshot_configured"
            )
        stride = max(1, len(aggregate.frequencies) // 512)
        report["selections"].append(
            {
                "axis": axis,
                "shaper_type": chosen.selected.name,
                "frequency_hz": chosen.selected.frequency,
                "damping_ratio": damping,
                "damping_source": damping_source,
                "damping_uncertainty_samples": damping_uncertainty_samples,
            }
        )
        report["axes"][axis] = {
            "qc": quality,
            "modes": [asdict(mode) for mode in modes],
            "candidates": [asdict(item) for item in scores],
            "native_candidates": _native_candidate_summary(captures[axis]),
            "pareto": [item.name for item in chosen.frontier],
            "selected": chosen.selected.name,
            "design_damping_ratio": damping,
            "measured_design_damping_ratio": (
                measured_design_damping if measured_damping else None
            ),
            "damping_source": damping_source,
            "damping_uncertainty_samples": damping_uncertainty_samples,
            "generalized_mzv": generalized_report,
            "acceleration_limits": {
                "theoretical_smoothing_mm_s2": float(chosen.selected.max_accel),
                "resonance_validated_mm_s2": None,
                "print_validated_mm_s2": None,
                "evidence_level": "theoretical",
                "note": (
                    "The smoothing model is not proof of mechanically or print-safe "
                    "acceleration."
                ),
            },
            "native_spectrum": _native_spectrum_summary(captures[axis]),
            "spectrogram": _spectrogram(captures[axis][0], axis),
            "spectrum": {
                "frequency_hz": aggregate.frequencies[::stride].tolist(),
                "psd": aggregate.values[::stride].tolist(),
                "relative_mad": mad[::stride].tolist(),
            },
        }
    if profile == "adaptive_stock":
        report["runtime_contract"] = {
            "interface": "stock_set_input_shaper",
            "families": list(NATIVE_SHAPER_ORDER),
            "parameterized_family": "mzv",
            "arbitrary_pulse_vectors": False,
            "installed_capability_required": True,
            "held_out_validation_required": True,
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
