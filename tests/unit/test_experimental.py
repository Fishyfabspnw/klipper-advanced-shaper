import numpy as np
import pytest

from klipper_advanced_shaper.analysis.experimental import (
    acceleration_envelope,
    damping_samples,
    generalized_mzv_pulses,
    optimize_generalized_mzv,
    oscillator_response,
    path_error_proxy,
    prove_runtime_generalized_mzv,
    smoothing_max_accel,
)


def test_generalized_mzv_is_positive_normalized_and_notches_design_mode():
    amplitudes, times = generalized_mzv_pulses(3, 0.75, 60.0, 0.08)
    assert len(amplitudes) == 3
    assert np.sum(amplitudes) == pytest.approx(1.0)
    assert np.all(amplitudes >= 0.0)
    assert np.all(np.diff(times) >= 0.0)
    response = oscillator_response(amplitudes, times, [60.0], 0.08)
    assert response[0] < 1e-7


def test_generalized_mzv_matches_pinned_upstream_pulse_definition():
    # Frozen from Klipper shaper_defs.py at 7046bd00ef5c30dec6febc724f8d22967433c45c.
    amplitudes, times = generalized_mzv_pulses(4, 0.8, 72.0, 0.04)
    assert amplitudes == pytest.approx(
        [0.2496491005879188, 0.28225831610016416, 0.2639470043708108, 0.20414557894110627]
    )
    assert times == pytest.approx(
        [0.0, 0.00370667022696961, 0.00741334045393922, 0.011120010680908832]
    )


def test_smoothing_acceleration_solves_klipper_path_error_limit():
    amplitudes, times = generalized_mzv_pulses(4, 0.8, 70.0, 0.05)
    estimate = smoothing_max_accel(amplitudes, times, scv=7.0)
    assert path_error_proxy(amplitudes, times, estimate, 7.0) == pytest.approx(0.12)
    assert path_error_proxy(amplitudes, times, estimate * 1.01, 7.0) > 0.12


def test_optimizer_requires_measured_damping_and_returns_research_only_frontier():
    frequencies = np.linspace(5.0, 150.0, 600)
    psd = np.exp(-0.5 * ((frequencies - 72.0) / 2.0) ** 2)
    with pytest.raises(ValueError, match="measured modal damping"):
        optimize_generalized_mzv(frequencies, psd, [{"frequency": 72.0}], 7.0)

    report = optimize_generalized_mzv(
        frequencies,
        psd,
        [{"frequency": 72.0, "damping_ratio": 0.04}],
        7.0,
        pulse_counts=[3, 4],
        spacing_values=[0.65, 0.8],
        frequency_values=[68.0, 72.0, 76.0],
        maximum_residual_q95=1.0,
    )
    assert report["status"] == "research_only"
    assert report["runtime_applicable"] is False
    assert report["evaluated_count"] > 0
    assert report["pareto"]
    assert report["measured_damping_samples"] != [0.1]


def test_damping_uncertainty_is_derived_from_all_measured_modes():
    samples = damping_samples(
        [
            {"damping_ratio": 0.03},
            {"damping_ratio": 0.08},
            {"damping_ratio": None},
        ],
        uncertainty=0.01,
    )
    assert samples.min() < 0.03
    assert samples.max() > 0.08
    assert 0.03 in samples
    assert 0.08 in samples


def test_acceleration_envelope_can_only_derate_and_labels_evidence():
    theoretical = acceleration_envelope(
        20_000.0, repeatability_cv_q95=0.05, model_sensitivity_q95=0.10
    )
    assert theoretical.evidence_level == "theoretical"
    assert theoretical.acceleration_mm_s2 < 20_000.0

    validated = acceleration_envelope(
        20_000.0,
        repeatability_cv_q95=0.05,
        model_sensitivity_q95=0.10,
        vibration_confidence_bound=17_500.0,
        print_validated_bound=16_000.0,
    )
    assert validated.evidence_level == "print_validated"
    assert validated.acceleration_mm_s2 == 16_000.0
    assert validated.limiting_bound == "print_validated"


def test_runtime_capability_requires_parameterized_parser_and_valid_pulses():
    class Supported:
        @staticmethod
        def get_shaper_cfg(name):
            return object() if name.startswith("mzv(") else None

        @staticmethod
        def init_shaper(name, frequency, damping):
            return generalized_mzv_pulses(4, 0.8, frequency, damping)

    class Unsupported:
        @staticmethod
        def get_shaper_cfg(name):
            return None

        @staticmethod
        def init_shaper(name, frequency, damping):
            raise ValueError("unsupported")

    assert prove_runtime_generalized_mzv(Supported)["passed"] is True
    assert prove_runtime_generalized_mzv(Unsupported)["passed"] is False
