"""Generalized-MZV optimization and evidence-bounded acceleration envelopes.

This module deliberately has no Klippy imports and cannot apply a shaper.  It
searches a public Klipper shaper parameterization, then emits evidence that the
normal pipeline may promote only after runtime compatibility and held-out
validation gates pass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from klipper_advanced_shaper.shapers import parse_shaper_identifier

_NATIVE_PULSE_COUNTS = {
    "zv": 2,
    "mzv": 3,
    "zvd": 3,
    "ei": 3,
    "2hump_ei": 4,
    "3hump_ei": 5,
}


@dataclass(frozen=True)
class GeneralizedMZVCandidate:
    shaper_type: str
    pulse_count: int
    spacing: float
    frequency_hz: float
    design_damping_ratio: float
    pulse_amplitudes: tuple[float, ...]
    pulse_times_s: tuple[float, ...]
    residual_energy_q95: float
    residual_energy_median: float
    klipper_remaining_vibration: float
    sensitivity: float
    path_error_at_5000: float
    smoothing_max_accel: float

    def to_report(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AccelerationEnvelope:
    acceleration_mm_s2: float
    evidence_level: str
    bounds_mm_s2: Mapping[str, float]
    limiting_bound: str
    notes: tuple[str, ...]


def generalized_mzv_pulses(
    pulse_count: int,
    spacing: float,
    frequency_hz: float,
    damping_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce Klipper's positive-pulse generalized MZV construction.

    ``spacing`` is Klipper's dimensionless ``t`` parameter, not seconds.
    The returned amplitudes are normalized because the native executor also
    normalizes them. Invalid or negative-pulse designs fail closed.
    """
    n = int(pulse_count)
    t = float(spacing)
    frequency = float(frequency_hz)
    damping = float(damping_ratio)
    if n < 3 or n > 10:
        raise ValueError("generalized MZV requires 3..10 pulses")
    if not 0.0 < t < 0.5 * (n - 1):
        raise ValueError("spacing must be positive and below (n - 1) / 2")
    if not frequency > 0.0:
        raise ValueError("frequency must be positive")
    if not 0.0 <= damping < 1.0:
        raise ValueError("damping ratio must be in [0, 1)")

    projected = t * (n - 2.0) / (n - 2.0 * t - 1.0)
    times = np.arange(n, dtype=float) * t / (n - 1.0)
    matrix: list[list[float]] = [[1.0] * n]
    target = [1.0]
    for index in range(n - 1):
        phase = 2.0 * np.pi * (1.0 + index / projected) * times
        matrix.extend([np.cos(phase).tolist(), np.sin(phase).tolist()])
        target.extend([0.0, 0.0])
    amplitudes = np.linalg.pinv(np.asarray(matrix, dtype=float)) @ np.asarray(target)
    if np.any(amplitudes < -1e-5) or not np.all(np.isfinite(amplitudes)):
        raise ValueError("generalized MZV design contains negative or invalid pulses")
    amplitudes = np.maximum(amplitudes, 0.0)

    damped_frequency = frequency * np.sqrt(1.0 - damping**2)
    decay = np.exp(-2.0 * t * damping * np.pi / ((n - 1.0) * np.sqrt(1.0 - damping**2)))
    pulse_times = times / damped_frequency
    amplitudes *= decay ** np.arange(n, dtype=float)
    amplitudes /= np.sum(amplitudes)
    return amplitudes, pulse_times


def oscillator_response(
    amplitudes: Sequence[float],
    pulse_times_s: Sequence[float],
    frequencies_hz: Sequence[float],
    damping_ratio: float,
) -> np.ndarray:
    """Return the residual vibration amplitude using Klipper's convention."""
    amplitudes_array = np.asarray(amplitudes, dtype=float)
    times = np.asarray(pulse_times_s, dtype=float)
    frequencies = np.asarray(frequencies_hz, dtype=float)
    damping = float(damping_ratio)
    if amplitudes_array.ndim != 1 or times.shape != amplitudes_array.shape:
        raise ValueError("pulse amplitudes and times must be matching vectors")
    if frequencies.ndim != 1 or np.any(frequencies < 0.0):
        raise ValueError("frequencies must be a non-negative vector")
    if not 0.0 <= damping < 1.0:
        raise ValueError("damping ratio must be in [0, 1)")
    omega = 2.0 * np.pi * frequencies
    damped_omega = omega * np.sqrt(1.0 - damping**2)
    weights = amplitudes_array * np.exp(
        np.outer(-damping * omega, times[-1] - times)
    )
    sine = weights * np.sin(np.outer(damped_omega, times))
    cosine = weights * np.cos(np.outer(damped_omega, times))
    return np.sqrt(np.sum(sine, axis=1) ** 2 + np.sum(cosine, axis=1) ** 2)


def path_error_proxy(
    amplitudes: Sequence[float], pulse_times_s: Sequence[float], accel: float, scv: float
) -> float:
    """Klipper's native 90/180-degree smoothing proxy."""
    weights = np.asarray(amplitudes, dtype=float)
    times = np.asarray(pulse_times_s, dtype=float)
    total = float(np.sum(weights))
    if total <= 0.0 or accel < 0.0 or scv < 0.0:
        raise ValueError("invalid pulse, acceleration, or SCV values")
    center = float(np.sum(weights * times) / total)
    delta = times - center
    half_accel = 0.5 * accel
    positive = delta >= 0.0
    offset_90 = np.sum(
        weights[positive]
        * (scv + half_accel * delta[positive])
        * delta[positive]
    )
    offset_90 *= np.sqrt(2.0) / total
    offset_180 = float(np.sum(weights * half_accel * delta**2) / total)
    return float(max(offset_90, offset_180))


def smoothing_max_accel(
    amplitudes: Sequence[float],
    pulse_times_s: Sequence[float],
    scv: float,
    target_path_error: float = 0.12,
) -> float:
    """Solve the same 0.12 path-error limit used by Klipper calibration."""
    if target_path_error <= 0.0:
        raise ValueError("target path error must be positive")
    if path_error_proxy(amplitudes, pulse_times_s, 0.0, scv) > target_path_error:
        return 0.0
    lower, upper = 0.0, 1.0
    while path_error_proxy(amplitudes, pulse_times_s, upper, scv) <= target_path_error:
        upper *= 2.0
        if upper > 1e8:
            raise ValueError("could not bracket smoothing acceleration")
    for _ in range(64):
        middle = 0.5 * (lower + upper)
        if path_error_proxy(amplitudes, pulse_times_s, middle, scv) <= target_path_error:
            lower = middle
        else:
            upper = middle
    return lower


def damping_samples(
    modes: Sequence[Mapping[str, Any]], uncertainty: float = 0.02
) -> np.ndarray:
    """Build an uncertainty set from measured modal damping; never assume 0.1."""
    measured = []
    for mode in modes:
        value = mode.get("damping_ratio")
        if value is not None and np.isfinite(float(value)) and 0.0 < float(value) < 1.0:
            measured.append(float(value))
    if not measured:
        raise ValueError("measured modal damping is required for experimental fitting")
    spread = max(float(uncertainty), float(np.std(measured)))
    samples = []
    for value in measured:
        samples.extend([value - spread, value, value + spread])
    return np.unique(np.clip(samples, 0.005, 0.40))


def _integral(values: np.ndarray, coordinates: np.ndarray) -> float:
    """Trapezoidal integration compatible with NumPy 1.x and 2.x."""
    return float(np.sum(0.5 * (values[:-1] + values[1:]) * np.diff(coordinates)))


def klipper_remaining_vibration(
    amplitudes: Sequence[float],
    pulse_times_s: Sequence[float],
    frequencies_hz: Sequence[float],
    psd: Sequence[float],
    test_damping_ratios: Sequence[float] = (0.075, 0.10, 0.15),
) -> float:
    """Reproduce Klipper's thresholded worst-damping vibration fraction."""
    frequencies = np.asarray(frequencies_hz, dtype=float)
    power = np.asarray(psd, dtype=float)
    if frequencies.shape != power.shape or frequencies.ndim != 1:
        raise ValueError("frequency and PSD vectors must match")
    if power.size < 2 or np.any(power < 0.0) or not np.all(np.isfinite(power)):
        raise ValueError("PSD must be finite, non-negative, and non-empty")
    responses = np.vstack(
        [
            oscillator_response(
                amplitudes, pulse_times_s, frequencies, float(damping_ratio)
            )
            for damping_ratio in test_damping_ratios
        ]
    )
    worst_response = np.max(responses, axis=0)
    threshold = float(np.max(power)) / 20.0
    denominator = float(np.sum(np.maximum(power - threshold, 0.0)))
    if denominator <= 0.0:
        raise ValueError("PSD has no energy above Klipper's vibration threshold")
    numerator = float(np.sum(np.maximum(worst_response * power - threshold, 0.0)))
    return numerator / denominator


def _pareto(candidates: Sequence[GeneralizedMZVCandidate]) -> list[GeneralizedMZVCandidate]:
    result = []
    for item in candidates:
        dominated = any(
            other is not item
            and other.residual_energy_q95 <= item.residual_energy_q95
            and other.sensitivity <= item.sensitivity
            and other.smoothing_max_accel >= item.smoothing_max_accel
            and (
                other.residual_energy_q95 < item.residual_energy_q95
                or other.sensitivity < item.sensitivity
                or other.smoothing_max_accel > item.smoothing_max_accel
            )
            for other in candidates
        )
        if not dominated:
            result.append(item)
    return sorted(result, key=lambda item: (-item.smoothing_max_accel, item.residual_energy_q95))


def optimize_generalized_mzv(
    frequencies_hz: Sequence[float],
    psd: Sequence[float],
    modes: Sequence[Mapping[str, Any]],
    square_corner_velocity: float,
    *,
    pulse_counts: Iterable[int] = range(3, 8),
    spacing_values: Optional[Iterable[float]] = None,
    frequency_values: Optional[Iterable[float]] = None,
    fixed_frequency_hz: Optional[float] = None,
    damping_uncertainty: float = 0.02,
    maximum_residual_q95: float = 0.10,
) -> dict[str, Any]:
    """Search positive-pulse generalized MZV and return research candidates.

    Residual energy is the PSD-weighted squared oscillator response. The 95th
    percentile across measured damping uncertainty is the acceptance metric.
    This is an offline model result and is never runtime-applicable by itself.
    """
    frequencies = np.asarray(frequencies_hz, dtype=float)
    power = np.asarray(psd, dtype=float)
    if frequencies.ndim != 1 or power.shape != frequencies.shape or frequencies.size < 3:
        raise ValueError("frequency and PSD vectors must match")
    if np.any(np.diff(frequencies) <= 0.0) or np.any(power < 0.0) or not np.all(np.isfinite(power)):
        raise ValueError("frequency and PSD vectors must be finite, ordered, and non-negative")
    total_power = _integral(power, frequencies)
    if total_power <= 0.0:
        raise ValueError("PSD must contain positive energy")
    damping = damping_samples(modes, damping_uncertainty)
    design_damping = float(np.median(damping))
    modal_frequencies = [float(mode["frequency"]) for mode in modes]
    if fixed_frequency_hz is not None:
        fixed_frequency = float(fixed_frequency_hz)
        if not np.isfinite(fixed_frequency) or fixed_frequency <= 0.0:
            raise ValueError("fixed generalized-MZV frequency must be finite and positive")
        if frequency_values is not None:
            raise ValueError("fixed_frequency_hz and frequency_values are mutually exclusive")
        frequency_values = (fixed_frequency,)
    elif frequency_values is None:
        center = np.asarray(modal_frequencies, dtype=float)
        frequency_values = np.unique(
            np.concatenate(
                [center * scale for scale in (0.85, 0.95, 1.0, 1.05, 1.15)]
            )
        )
    if spacing_values is None:
        spacing_values = np.linspace(0.45, 1.35, 10)

    candidates: list[GeneralizedMZVCandidate] = []
    for n in pulse_counts:
        for spacing in spacing_values:
            if not 0.0 < float(spacing) < 0.5 * (int(n) - 1):
                continue
            for frequency in frequency_values:
                try:
                    amplitudes, times = generalized_mzv_pulses(
                        int(n), float(spacing), float(frequency), design_damping
                    )
                except (ValueError, np.linalg.LinAlgError):
                    continue
                residuals = []
                for damping_ratio in damping:
                    response = oscillator_response(amplitudes, times, frequencies, damping_ratio)
                    residuals.append(
                        _integral(power * response**2, frequencies) / total_power
                    )
                residual_array = np.asarray(residuals)
                accel = smoothing_max_accel(amplitudes, times, square_corner_velocity)
                native_metric = klipper_remaining_vibration(
                    amplitudes, times, frequencies, power
                )
                candidates.append(
                    GeneralizedMZVCandidate(
                        shaper_type="mzv(n=%d,t=%.6f)" % (int(n), float(spacing)),
                        pulse_count=int(n),
                        spacing=float(spacing),
                        frequency_hz=float(frequency),
                        design_damping_ratio=design_damping,
                        pulse_amplitudes=tuple(float(value) for value in amplitudes),
                        pulse_times_s=tuple(float(value) for value in times),
                        residual_energy_q95=float(np.quantile(residual_array, 0.95)),
                        residual_energy_median=float(np.median(residual_array)),
                        klipper_remaining_vibration=native_metric,
                        sensitivity=float(np.ptp(residual_array)),
                        path_error_at_5000=path_error_proxy(
                            amplitudes, times, 5000.0, square_corner_velocity
                        ),
                        smoothing_max_accel=accel,
                    )
                )
    eligible = [item for item in candidates if item.residual_energy_q95 <= maximum_residual_q95]
    frontier = _pareto(eligible)
    return {
        "status": "research_only",
        "runtime_applicable": False,
        "family": "klipper_generalized_mzv",
        "frequency_strategy": (
            "strongest_measured_peak" if fixed_frequency_hz is not None else "modal_search_grid"
        ),
        "fixed_frequency_hz": (
            float(fixed_frequency_hz) if fixed_frequency_hz is not None else None
        ),
        "measured_damping_samples": damping.tolist(),
        "evaluated_count": len(candidates),
        "eligible_count": len(eligible),
        "pareto": [item.to_report() for item in frontier],
        "reason": None if frontier else "no design passed the robust residual gate",
    }


def acceleration_envelope(
    smoothing_bound: float,
    *,
    repeatability_cv_q95: float,
    model_sensitivity_q95: float,
    vibration_confidence_bound: Optional[float] = None,
    hardware_validated_bound: Optional[float] = None,
    print_validated_bound: Optional[float] = None,
) -> AccelerationEnvelope:
    """Return a non-inflating minimum-of-evidence acceleration envelope.

    Repeatability and model sensitivity only derate the native-compatible
    smoothing bound. Hardware bounds enter only when measured at an explicit
    acceleration; normalized resonance attenuation alone is not converted to
    acceleration because that conversion would be dimensionally unjustified.
    """
    values = {
        "smoothing_path_error": float(smoothing_bound),
        "repeatability_penalty": float(smoothing_bound) / (1.0 + repeatability_cv_q95),
        "uncertainty_penalty": float(smoothing_bound) / (1.0 + model_sensitivity_q95),
    }
    optional = {
        "vibration_confidence": vibration_confidence_bound,
        "hardware_validated": hardware_validated_bound,
        "print_validated": print_validated_bound,
    }
    for name, value in optional.items():
        if value is not None:
            values[name] = float(value)
    if any(not np.isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError("acceleration bounds and penalties must be finite and non-negative")
    limiting = min(values, key=values.get)
    if print_validated_bound is not None:
        level = "print_validated"
    elif vibration_confidence_bound is not None or hardware_validated_bound is not None:
        level = "resonance_validated"
    else:
        level = "theoretical"
    notes = (
        "This envelope never changes Klipper max_accel.",
        "Higher output must come from a less-smoothing validated pulse design, "
        "not formula inflation.",
    )
    return AccelerationEnvelope(values[limiting], level, values, limiting, notes)


def prove_runtime_generalized_mzv(
    shaper_defs_module: Any,
    syntax: str = "mzv(n=4,t=0.800000)",
    frequency: float = 60.0,
    damping_ratio: float = 0.08,
) -> dict[str, Any]:
    """Prove that installed Klipper realizes an exact allowlisted MZV form."""
    try:
        identifier = parse_shaper_identifier(syntax, allow_parameterized=True)
        if not identifier.parameterized or identifier.family != "mzv":
            raise ValueError("runtime generalized-MZV proof requires parameterized MZV")
        canonical = identifier.canonical
        config = shaper_defs_module.get_shaper_cfg(canonical)
        pulses = shaper_defs_module.init_shaper(canonical, frequency, damping_ratio)
        amplitudes, times = pulses
        expected_amplitudes, expected_times = generalized_mzv_pulses(
            int(identifier.argument_map()["n"]),
            identifier.mzv_spacing(),
            frequency,
            damping_ratio,
        )
        runtime_amplitudes = np.asarray(amplitudes, dtype=float)
        runtime_times = np.asarray(times, dtype=float)
        if np.sum(runtime_amplitudes) > 0.0:
            runtime_amplitudes = runtime_amplitudes / np.sum(runtime_amplitudes)
        passed = (
            config is not None
            and 3 <= len(amplitudes) <= 10
            and len(amplitudes) == len(times)
            and all(float(value) >= -1e-5 for value in amplitudes)
            and all(
                float(times[index]) <= float(times[index + 1])
                for index in range(len(times) - 1)
            )
            and np.allclose(runtime_amplitudes, expected_amplitudes, rtol=1e-6, atol=1e-8)
            and np.allclose(runtime_times, expected_times, rtol=1e-6, atol=1e-9)
        )
        return {
            "passed": bool(passed),
            "syntax": canonical,
            "pulse_count": len(amplitudes),
            "reason": None if passed else "runtime returned an invalid pulse sequence",
        }
    except Exception as error:  # Klipper raises version-specific config errors.
        return {"passed": False, "syntax": syntax, "reason": str(error)}


def prove_runtime_native_shapers(
    shaper_defs_module: Any,
    *,
    families: Sequence[str] = tuple(_NATIVE_PULSE_COUNTS),
    frequency: float = 60.0,
    damping_ratio: float = 0.10,
    executor_pulse_limit: int = 10,
) -> dict[str, Any]:
    """Prove that installed Klipper realizes every requested native family."""
    proofs = []
    for raw_family in families:
        family = str(raw_family).lower()
        try:
            identifier = parse_shaper_identifier(family, allow_parameterized=False)
            expected_count = _NATIVE_PULSE_COUNTS[identifier.family]
            config = shaper_defs_module.get_shaper_cfg(identifier.canonical)
            amplitudes, times = shaper_defs_module.init_shaper(
                identifier.canonical, float(frequency), float(damping_ratio)
            )
            amplitude_array = np.asarray(amplitudes, dtype=float)
            time_array = np.asarray(times, dtype=float)
            passed = (
                config is not None
                and len(amplitudes) == expected_count
                and len(amplitudes) == len(times)
                and len(amplitudes) <= int(executor_pulse_limit)
                and np.all(np.isfinite(amplitude_array))
                and np.all(amplitude_array >= -1e-5)
                and float(np.sum(amplitude_array)) > 0.0
                and np.all(np.isfinite(time_array))
                and np.all(np.diff(time_array) >= 0.0)
            )
            proofs.append(
                {
                    "passed": bool(passed),
                    "syntax": identifier.canonical,
                    "pulse_count": len(amplitudes),
                    "reason": None
                    if passed
                    else "runtime returned an invalid native pulse sequence",
                }
            )
        except Exception as error:  # Klipper raises version-specific config errors.
            proofs.append({"passed": False, "syntax": family, "reason": str(error)})
    failed = next((proof for proof in proofs if not proof["passed"]), None)
    return {
        "passed": failed is None,
        "family": "stock_native_allowlist",
        "proofs": proofs,
        "reason": None if failed is None else failed["reason"],
    }
