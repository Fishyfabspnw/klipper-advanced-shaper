import numpy as np
import pytest

from klipper_advanced_shaper.analysis.experimental import generalized_mzv_pulses
from klipper_advanced_shaper.analysis.non_regression import (
    theoretical_spectral_non_regression,
)


def _model(axis, shaper_type, frequency, damping, amplitudes, times):
    return {
        "axis": axis,
        "shaper_type": shaper_type,
        "frequency_hz": frequency,
        "design_damping_ratio": damping,
        "pulse_count": len(amplitudes),
        "pulse_amplitudes_normalized": np.asarray(amplitudes).tolist(),
        "pulse_times_s": np.asarray(times).tolist(),
        "source": "installed_klipper_shaper_defs.init_shaper",
        "source_module": "extras.shaper_defs",
        "source_file": "shaper_defs.py",
        "api_signature_verified": True,
        "executor_pulse_limit": 10,
        "theoretical_model_only": True,
        "live_c_executor_readback": False,
    }


def _live_like_report(candidate_type, candidate_frequency, candidate_damping):
    frequencies = np.linspace(5.0, 135.0, 521)
    primary = np.exp(-0.5 * ((frequencies - 72.5) / 2.2) ** 2)
    secondary = 0.004 * np.exp(-0.5 * ((frequencies - 128.0) / 2.0) ** 2)
    along = primary + secondary + 1e-6
    cross = 0.03 * primary + 0.012 * np.exp(
        -0.5 * ((frequencies - 128.0) / 2.0) ** 2
    ) + 1e-6
    return {
        "selections": [
            {
                "axis": "X",
                "shaper_type": candidate_type,
                "frequency_hz": candidate_frequency,
                "damping_ratio": candidate_damping,
            }
        ],
        "axes": {
            "X": {
                "modes": [{"frequency": 72.5}],
                "damping_uncertainty_samples": [0.0669565, 0.0869565, 0.1069565],
                "spectrum": {
                    "frequency_hz": frequencies.tolist(),
                    "psd": along.tolist(),
                },
                "cross_spectrum": {
                    "frequency_hz": frequencies.tolist(),
                    "psd": cross.tolist(),
                },
            }
        },
    }


def test_live_like_n10_candidate_is_rejected_for_secondary_band_regression():
    reference_amplitudes, reference_times = generalized_mzv_pulses(
        3, 0.75, 75.6, 0.038
    )
    candidate_amplitudes, candidate_times = generalized_mzv_pulses(
        10, 0.65, 72.454637, 0.0869565
    )
    reference = _model(
        "X", "mzv", 75.6, 0.038, reference_amplitudes, reference_times
    )
    candidate = _model(
        "X",
        "mzv(n=10,t=0.650000)",
        72.454637,
        0.0869565,
        candidate_amplitudes,
        candidate_times,
    )

    result = theoretical_spectral_non_regression(
        training_report=_live_like_report(
            "mzv(n=10,t=0.650000)", 72.454637, 0.0869565
        ),
        axes=("X",),
        reference_models={"X": reference},
        candidate_models={"X": candidate},
    )

    assert result["passed"] is False
    assert result["validation"] is False
    assert result["held_out_validation_still_required"] is True
    assert result["physical_acceleration_claim"] is False
    assert result["live_c_executor_verified"] is False
    channels = result["axes"]["X"]["channels"]
    assert channels["along_axis"]["passed"] is False
    assert channels["cross_axis"]["passed"] is False
    assert any(
        not band["passed"] and 120.0 <= band["low_hz"] <= 130.0
        for channel in channels.values()
        for band in channel["bands"]
    )
    assert result["method"]["not_a_direct_shaped_sweep_filter_model"] is True
    assert "cross_axis" in result["reason"] or "along_axis" in result["reason"]


def test_exact_configured_candidate_passes_but_remains_theoretical_only():
    amplitudes, times = generalized_mzv_pulses(3, 0.75, 75.6, 0.038)
    model = _model("X", "mzv", 75.6, 0.038, amplitudes, times)

    result = theoretical_spectral_non_regression(
        training_report=_live_like_report("mzv", 75.6, 0.038),
        axes=("X",),
        reference_models={"X": model},
        candidate_models={"X": model},
    )

    assert result["passed"] is True
    assert result["status"] == "passed"
    assert result["evidence_level"] == "theoretical_preflight_screen"
    assert result["held_out_validation_still_required"] is True
    assert all(
        channel["worst_band"]["worst_candidate_to_guarded_reference_ratio"]
        <= 1.0 + 1e-9
        for channel in result["axes"]["X"]["channels"].values()
    )


def test_screen_rejects_unproved_model_source_and_missing_cross_spectrum():
    amplitudes, times = generalized_mzv_pulses(3, 0.75, 75.6, 0.038)
    model = _model("X", "mzv", 75.6, 0.038, amplitudes, times)
    unproved = dict(model, api_signature_verified=False)
    with pytest.raises(ValueError, match="strict installed-source checks"):
        theoretical_spectral_non_regression(
            training_report=_live_like_report("mzv", 75.6, 0.038),
            axes=("X",),
            reference_models={"X": unproved},
            candidate_models={"X": model},
        )

    missing_cross = _live_like_report("mzv", 75.6, 0.038)
    del missing_cross["axes"]["X"]["cross_spectrum"]
    with pytest.raises(ValueError, match="cross-axis training spectrum"):
        theoretical_spectral_non_regression(
            training_report=missing_cross,
            axes=("X",),
            reference_models={"X": model},
            candidate_models={"X": model},
        )
