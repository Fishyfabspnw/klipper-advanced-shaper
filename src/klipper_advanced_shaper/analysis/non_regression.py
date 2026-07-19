"""Conservative model screen against the exact configured Klipper shaper.

This module performs no motion and makes no validation or acceleration claim.  It
only prevents an experimental candidate from reaching shaped held-out motion when
the installed-Klipper pulse model predicts a material regression in a meaningful
band of the measured training spectrum.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from klipper_advanced_shaper.shapers import parse_shaper_identifier

from .experimental import oscillator_response

_LOW_FREQUENCY_HZ = 5.0
_HIGH_FREQUENCY_HZ = 200.0
_BAND_WIDTH_HZ = 5.0
_MIN_RAW_BAND_FRACTION = 0.001
_REFERENCE_NUMERICAL_FLOOR_FRACTION = 64.0 * np.finfo(float).eps
_MAX_BAND_REGRESSION_RATIO = 1.10


def _integral(values: np.ndarray, frequencies: np.ndarray) -> float:
    return float(
        np.sum(0.5 * (values[:-1] + values[1:]) * np.diff(frequencies))
    )


def _validated_model(
    raw: Mapping[str, Any], axis: str, role: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        model_axis = str(raw["axis"]).upper()
        identifier = parse_shaper_identifier(str(raw["shaper_type"]))
        frequency = float(raw["frequency_hz"])
        design_damping = float(raw["design_damping_ratio"])
        amplitudes = np.asarray(raw["pulse_amplitudes_normalized"], dtype=float)
        times = np.asarray(raw["pulse_times_s"], dtype=float)
        pulse_count = int(raw["pulse_count"])
        executor_limit = int(raw["executor_pulse_limit"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("%s %s model is malformed" % (axis, role)) from error
    if (
        model_axis != axis
        or not raw.get("api_signature_verified")
        or raw.get("source") != "installed_klipper_shaper_defs.init_shaper"
        or raw.get("theoretical_model_only") is not True
        or raw.get("live_c_executor_readback") is not False
        or not np.isfinite(frequency)
        or frequency <= 0.0
        or not np.isfinite(design_damping)
        or not 0.0 <= design_damping < 1.0
        or amplitudes.ndim != 1
        or times.shape != amplitudes.shape
        or pulse_count != amplitudes.size
        or not 2 <= pulse_count <= executor_limit <= 10
        or not np.all(np.isfinite(amplitudes))
        or np.any(amplitudes < -1e-5)
        or not np.isclose(float(np.sum(amplitudes)), 1.0, rtol=1e-9, atol=1e-10)
        or not np.all(np.isfinite(times))
        or np.any(np.diff(times) < 0.0)
    ):
        raise ValueError("%s %s model failed strict installed-source checks" % (axis, role))
    return amplitudes, times, {
        "axis": axis,
        "shaper_type": identifier.canonical,
        "frequency_hz": frequency,
        "design_damping_ratio": design_damping,
        "pulse_count": pulse_count,
        "source": str(raw["source"]),
        "source_module": raw.get("source_module"),
        "source_file": raw.get("source_file"),
        "api_signature_verified": True,
    }


def _evaluate_channel(
    *,
    axis: str,
    channel: str,
    spectrum: Mapping[str, Any],
    modes: Sequence[float],
    damping: np.ndarray,
    reference_amplitudes: np.ndarray,
    reference_times: np.ndarray,
    candidate_amplitudes: np.ndarray,
    candidate_times: np.ndarray,
) -> dict[str, Any]:
    try:
        frequencies = np.asarray(spectrum["frequency_hz"], dtype=float)
        power = np.asarray(spectrum["psd"], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("%s %s spectrum is malformed" % (axis, channel)) from error
    if (
        frequencies.ndim != 1
        or frequencies.size < 3
        or power.shape != frequencies.shape
        or not np.all(np.isfinite(frequencies))
        or np.any(np.diff(frequencies) <= 0.0)
        or not np.all(np.isfinite(power))
        or np.any(power < 0.0)
    ):
        raise ValueError("%s %s spectrum failed numeric checks" % (axis, channel))
    upper = min(_HIGH_FREQUENCY_HZ, float(frequencies[-1]))
    mask = (frequencies >= _LOW_FREQUENCY_HZ) & (frequencies <= upper)
    screen_frequency = frequencies[mask]
    screen_power = power[mask]
    if screen_frequency.size < 3:
        raise ValueError("%s %s spectrum does not cover the screen range" % (axis, channel))
    total_energy = _integral(screen_power, screen_frequency)
    if not np.isfinite(total_energy) or total_energy <= 0.0:
        raise ValueError("%s %s spectrum contains no positive energy" % (axis, channel))

    reference_responses = [
        oscillator_response(
            reference_amplitudes, reference_times, screen_frequency, float(sample)
        )
        for sample in damping
    ]
    candidate_responses = [
        oscillator_response(
            candidate_amplitudes, candidate_times, screen_frequency, float(sample)
        )
        for sample in damping
    ]
    bands = []
    low = _LOW_FREQUENCY_HZ
    while low < upper:
        high = min(low + _BAND_WIDTH_HZ, upper)
        if high == upper:
            band_mask = (screen_frequency >= low) & (screen_frequency <= high)
        else:
            band_mask = (screen_frequency >= low) & (screen_frequency < high)
        band_frequency = screen_frequency[band_mask]
        band_power = screen_power[band_mask]
        if band_frequency.size >= 2:
            raw_energy = _integral(band_power, band_frequency)
            raw_fraction = raw_energy / total_energy
            contains_mode = channel == "along_axis" and any(
                low <= mode <= high for mode in modes
            )
            if raw_fraction >= _MIN_RAW_BAND_FRACTION or contains_mode:
                ratios = []
                reference_energies = []
                candidate_energies = []
                for reference_response, candidate_response in zip(
                    reference_responses, candidate_responses
                ):
                    reference_energy = _integral(
                        band_power * reference_response[band_mask] ** 2,
                        band_frequency,
                    )
                    candidate_energy = _integral(
                        band_power * candidate_response[band_mask] ** 2,
                        band_frequency,
                    )
                    denominator = max(
                        reference_energy,
                        _REFERENCE_NUMERICAL_FLOOR_FRACTION * raw_energy,
                        np.finfo(float).tiny,
                    )
                    reference_energies.append(reference_energy)
                    candidate_energies.append(candidate_energy)
                    ratios.append(candidate_energy / denominator)
                worst_index = int(np.argmax(ratios))
                worst_ratio = float(ratios[worst_index])
                bands.append(
                    {
                        "low_hz": float(low),
                        "high_hz": float(high),
                        "raw_energy_fraction": float(raw_fraction),
                        "contains_detected_mode": bool(contains_mode),
                        "worst_candidate_to_guarded_reference_ratio": worst_ratio,
                        "worst_damping_ratio": float(damping[worst_index]),
                        "reference_energy_at_worst_damping": float(
                            reference_energies[worst_index]
                        ),
                        "candidate_energy_at_worst_damping": float(
                            candidate_energies[worst_index]
                        ),
                        "reference_floor_energy": float(
                            _REFERENCE_NUMERICAL_FLOOR_FRACTION * raw_energy
                        ),
                        "passed": worst_ratio <= _MAX_BAND_REGRESSION_RATIO,
                    }
                )
        low += _BAND_WIDTH_HZ
    if not bands:
        raise ValueError("%s %s has no meaningful measured spectral bands" % (axis, channel))
    worst_band = max(
        bands, key=lambda item: item["worst_candidate_to_guarded_reference_ratio"]
    )
    return {
        "passed": all(item["passed"] for item in bands),
        "meaningful_band_count": len(bands),
        "worst_band": worst_band,
        "bands": bands,
    }


def theoretical_spectral_non_regression(
    *,
    training_report: Mapping[str, Any],
    axes: Sequence[str],
    reference_models: Mapping[str, Mapping[str, Any]],
    candidate_models: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare worst measured-band response against the configured baseline.

    Bands are non-overlapping 5 Hz intervals.  A band is meaningful when it
    contains at least 0.1% of measured unshaped training energy, or contains an
    identified mode.  For every measured damping sample, candidate and reference
    squared responses are integrated with the measured PSD.  The worst band may
    not exceed 110% of the exact configured-reference model energy.

    The only denominator floor is 64 machine epsilons of unshaped band energy;
    it handles floating-point zero and does not relax a physical/model regression.
    The per-band maximum, rather than a whole-spectrum average, preserves
    secondary-mode regressions that a dominant primary peak could otherwise hide.
    """
    normalized_axes = tuple(str(axis).upper() for axis in axes)
    if (
        not normalized_axes
        or len(set(normalized_axes)) != len(normalized_axes)
        or any(axis not in {"X", "Y"} for axis in normalized_axes)
    ):
        raise ValueError("spectral screen axes must be unique X and/or Y")
    if set(reference_models) != set(normalized_axes):
        raise ValueError("configured reference models do not exactly match requested axes")
    if set(candidate_models) != set(normalized_axes):
        raise ValueError("candidate models do not exactly match requested axes")

    report_axes = training_report.get("axes")
    selections = training_report.get("selections")
    if not isinstance(report_axes, Mapping) or not isinstance(selections, Sequence):
        raise ValueError("training report lacks axis spectra or selections")
    selected_by_axis = {
        str(item.get("axis", "")).upper(): item
        for item in selections
        if isinstance(item, Mapping)
    }
    if set(selected_by_axis) != set(normalized_axes):
        raise ValueError("training report selections do not exactly match requested axes")

    details: dict[str, Any] = {}
    passed = True
    failure_reasons = []
    for axis in normalized_axes:
        reference_amplitudes, reference_times, reference_identity = _validated_model(
            reference_models[axis], axis, "reference"
        )
        candidate_amplitudes, candidate_times, candidate_identity = _validated_model(
            candidate_models[axis], axis, "candidate"
        )
        selected = selected_by_axis[axis]
        selected_identifier = parse_shaper_identifier(str(selected.get("shaper_type", "")))
        try:
            selected_frequency = float(selected["frequency_hz"])
            selected_damping = float(selected["damping_ratio"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("%s training selection is malformed" % axis) from error
        if (
            candidate_identity["shaper_type"] != selected_identifier.canonical
            or not np.isclose(
                candidate_identity["frequency_hz"], selected_frequency, rtol=0.0, atol=1e-9
            )
            or not np.isclose(
                candidate_identity["design_damping_ratio"],
                selected_damping,
                rtol=0.0,
                atol=1e-9,
            )
        ):
            raise ValueError("%s candidate model does not exactly match selection" % axis)

        axis_report = report_axes.get(axis)
        if not isinstance(axis_report, Mapping):
            raise ValueError("%s training report is missing" % axis)
        spectrum = axis_report.get("spectrum")
        cross_spectrum = axis_report.get("cross_spectrum")
        if not isinstance(spectrum, Mapping) or not isinstance(cross_spectrum, Mapping):
            raise ValueError("%s along-axis or cross-axis training spectrum is missing" % axis)
        try:
            damping = np.asarray(axis_report["damping_uncertainty_samples"], dtype=float)
            modes = [float(item["frequency"]) for item in axis_report.get("modes", [])]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("%s spectral-screen evidence is malformed" % axis) from error
        if (
            damping.ndim != 1
            or damping.size < 1
            or not np.all(np.isfinite(damping))
            or np.any((damping <= 0.0) | (damping >= 1.0))
            or any(not np.isfinite(mode) for mode in modes)
        ):
            raise ValueError("%s spectral-screen evidence failed numeric checks" % axis)

        channels = {
            "along_axis": _evaluate_channel(
                axis=axis,
                channel="along_axis",
                spectrum=spectrum,
                modes=modes,
                damping=damping,
                reference_amplitudes=reference_amplitudes,
                reference_times=reference_times,
                candidate_amplitudes=candidate_amplitudes,
                candidate_times=candidate_times,
            ),
            "cross_axis": _evaluate_channel(
                axis=axis,
                channel="cross_axis",
                spectrum=cross_spectrum,
                modes=(),
                damping=damping,
                reference_amplitudes=reference_amplitudes,
                reference_times=reference_times,
                candidate_amplitudes=candidate_amplitudes,
                candidate_times=candidate_times,
            ),
        }
        worst_channel_name, worst_channel = max(
            channels.items(),
            key=lambda item: item[1]["worst_band"][
                "worst_candidate_to_guarded_reference_ratio"
            ],
        )
        worst_band = worst_channel["worst_band"]
        axis_passed = all(channel["passed"] for channel in channels.values())
        passed = passed and axis_passed
        if not axis_passed:
            failure_reasons.append(
                "%s %s %.1f-%.1f Hz ratio %.3f exceeds %.3f"
                % (
                    axis,
                    worst_channel_name,
                    worst_band["low_hz"],
                    worst_band["high_hz"],
                    worst_band["worst_candidate_to_guarded_reference_ratio"],
                    _MAX_BAND_REGRESSION_RATIO,
                )
            )
        details[axis] = {
            "passed": axis_passed,
            "reference": reference_identity,
            "candidate": candidate_identity,
            "damping_uncertainty_samples": damping.tolist(),
            "worst_channel": worst_channel_name,
            "worst_band": worst_band,
            "channels": channels,
        }

    return {
        "passed": passed,
        "status": "passed" if passed else "rejected",
        "evidence_level": "theoretical_preflight_screen",
        "purpose": "post-training_pre-held-out-motion_non_regression",
        "validation": False,
        "held_out_validation_still_required": True,
        "physical_acceleration_claim": False,
        "live_c_executor_verified": False,
        "method": {
            "response": "installed_klipper_pulses_impulse_ringdown_oscillator_response_squared",
            "not_a_direct_shaped_sweep_filter_model": True,
            "training_input": "measured_unshaped_along_and_cross_axis_psd",
            "band_width_hz": _BAND_WIDTH_HZ,
            "meaningful_raw_energy_fraction": _MIN_RAW_BAND_FRACTION,
            "reference_numerical_floor_fraction_of_unshaped_band": (
                _REFERENCE_NUMERICAL_FLOOR_FRACTION
            ),
            "reference_floor_purpose": "floating_point_zero_only_not_noise_allowance",
            "maximum_candidate_to_guarded_reference_ratio": _MAX_BAND_REGRESSION_RATIO,
            "aggregation": "worst_meaningful_band_across_measured_damping_samples",
        },
        "axes": details,
        "reason": None if passed else "; ".join(failure_reasons),
    }
