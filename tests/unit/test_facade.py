from types import SimpleNamespace

import numpy as np
import pytest

from klipper_advanced_shaper.analysis import analyze_calibration
from klipper_advanced_shaper.analysis.facade import _candidate_scores
from klipper_advanced_shaper.analysis.models import Spectrum
from klipper_advanced_shaper.analysis.selection import PROFILES, select_candidate


def _capture(axis="X", scale=1.0, repeat=0, cross_scale=0.0, display_data=True):
    rate = 1000.0
    t = np.arange(0, 4, 1 / rate)
    signal = scale * (np.sin(2 * np.pi * 74 * t) + 0.35 * np.sin(2 * np.pi * 111 * t))
    xyz = np.zeros((t.size, 3))
    xyz[:, 0 if axis == "X" else 1] = signal
    xyz[:, 1 if axis == "X" else 0] = cross_scale * np.sin(2 * np.pi * 74 * t)
    candidates = [
        {
            "name": "mzv",
            "frequency": 74.2 + 0.1 * repeat,
            "residual_vibration": 0.04,
            "smoothing": 0.08,
            "max_accel": 17000,
            "design_damping_ratio": 0.08,
        },
        {
            "name": "ei",
            "frequency": 86.0,
            "residual_vibration": 0.02,
            "smoothing": 0.13,
            "max_accel": 12000,
            "design_damping_ratio": 0.08,
        },
    ]
    result = {
        "samples": np.column_stack((t, xyz)),
        "native_candidates": candidates,
        "metadata": {
            "clip_limit": 100.0,
            "test_recipe": {
                "freq_start": 5.0,
                "freq_end": 115.0,
                "hz_per_sec": 30.0,
            },
        },
    }
    if display_data:
        frequency = np.linspace(0.0, 200.0, 401)
        psd_x = 1.0 + frequency * 0.01 + repeat * 0.1
        psd_y = 2.0 + frequency * 0.02 + repeat * 0.1
        psd_z = 3.0 + frequency * 0.03 + repeat * 0.1
        result["native_spectrum"] = {
            "frequency_hz": frequency.tolist(),
            "psd_x": psd_x.tolist(),
            "psd_y": psd_y.tolist(),
            "psd_z": psd_z.tolist(),
            "psd_sum": (psd_x + psd_y + psd_z).tolist(),
        }
        response_frequency = np.linspace(5.0, 150.0, 100)
        for candidate in candidates:
            candidate["native_frequency_response"] = {
                "frequency_hz": response_frequency.tolist(),
                "response_ratio": np.exp(-response_frequency / candidate["frequency"]).tolist(),
            }
    return result


def test_facade_selects_native_candidate_and_reports_modes():
    captures = {"X": [_capture(repeat=index) for index in range(3)]}
    report = analyze_calibration(
        captures=captures,
        axes=("X",),
        profile="performance",
        snapshot=SimpleNamespace(square_corner_velocity=7.0, damping_ratio_x=0.08),
    )
    assert not report.get("abstain")
    assert report["selections"][0]["shaper_type"] == "mzv"
    assert report["selections"][0]["damping_ratio"] == 0.08
    assert report["selections"][0]["damping_source"] == "active_input_shaper_status"
    assert report["axes"]["X"]["modes"]
    assert report["square_corner_velocity"] == 7.0


def test_facade_exposes_native_components_responses_and_bounded_spectrogram():
    captures = {"X": [_capture(repeat=index) for index in range(3)]}
    report = analyze_calibration(
        captures=captures,
        axes=("X",),
        profile="performance",
        snapshot=SimpleNamespace(square_corner_velocity=7.0, damping_ratio_x=0.08),
    )
    details = report["axes"]["X"]
    native = details["native_spectrum"]

    assert native["available"] is True
    assert native["repeat_count"] == 3
    assert native["psd_x"][20] == pytest.approx(1.0 + native["frequency_hz"][20] * 0.01 + 0.1)
    assert len(native["frequency_hz"]) <= 1024
    assert details["native_candidates"][0]["native_frequency_response"]
    spectrogram = details["spectrogram"]
    assert spectrogram["available"] is True
    assert spectrogram["power_shape"] == [
        len(spectrogram["frequency_hz"]),
        len(spectrogram["time_s"]),
    ]
    assert spectrogram["power_shape"][0] <= 256
    assert spectrogram["power_shape"][1] <= 192


def test_display_data_does_not_change_candidate_selection():
    with_display = analyze_calibration(
        captures={"X": [_capture(repeat=index) for index in range(3)]},
        axes=("X",),
        profile="performance",
        snapshot=SimpleNamespace(square_corner_velocity=7.0, damping_ratio_x=0.08),
    )
    without_display = analyze_calibration(
        captures={
            "X": [_capture(repeat=index, display_data=False) for index in range(3)]
        },
        axes=("X",),
        profile="performance",
        snapshot=SimpleNamespace(square_corner_velocity=7.0, damping_ratio_x=0.08),
    )
    assert with_display["selections"] == without_display["selections"]
    assert without_display["axes"]["X"]["native_spectrum"]["available"] is False


def test_facade_requires_statistically_lower_held_out_energy():
    training = {"X": [_capture(repeat=index) for index in range(3)]}
    first = analyze_calibration(
        captures=training,
        axes=("X",),
        profile="performance",
        snapshot=SimpleNamespace(square_corner_velocity=7.0, damping_ratio_x=0.08),
    )
    result = analyze_calibration(
        captures=training,
        held_out_captures={"X": [_capture(scale=1.0, repeat=index) for index in range(3)]},
        validation_captures={"X": [_capture(scale=0.75, repeat=index) for index in range(3)]},
        axes=("X",),
        profile="performance",
        snapshot=SimpleNamespace(square_corner_velocity=7.0, damping_ratio_x=0.08),
        prior_report=first,
        validation_pair_ids={"X": ["X-01", "X-02", "X-03"]},
    )
    assert result["validation"]["passed"]
    assert result["validation"]["axes"]["X"]["improvement_ci_95"][0] > 0.10


def test_facade_fast_two_repeat_validation_keeps_ci_qc_and_cross_axis_gates():
    training = {"X": [_capture(repeat=index) for index in range(2)]}
    first = analyze_calibration(
        captures=training,
        axes=("X",),
        profile="performance",
        snapshot=SimpleNamespace(square_corner_velocity=7.0, damping_ratio_x=0.08),
    )
    result = analyze_calibration(
        captures=training,
        held_out_captures={"X": [_capture(scale=1.0, repeat=index) for index in range(2)]},
        validation_captures={
            "X": [_capture(scale=0.75, repeat=index) for index in range(2)]
        },
        axes=("X",),
        profile="performance",
        snapshot=SimpleNamespace(square_corner_velocity=7.0, damping_ratio_x=0.08),
        prior_report=first,
        validation_pair_ids={"X": ["X-01", "X-02"]},
    )

    evidence = result["validation"]["axes"]["X"]
    assert result["validation"]["passed"] is True
    assert evidence["qc_passed"] is True
    assert len(evidence["reference_qc"]) == 2
    assert len(evidence["candidate_qc"]) == 2
    assert evidence["improvement_ci_95"][0] > 0.10
    assert evidence["cross_axis_regression"] <= 0.05
    assert evidence["pair_ids"] == ["X-01", "X-02"]
    assert evidence["pair_count"] == 2
    assert evidence["capture_design"] == "paired_interleaved_ab"
    assert evidence["energy_units"] == "acceleration_squared"
    assert len(evidence["reference_energy_samples"]) == 2
    assert len(evidence["candidate_energy_samples"]) == 2
    assert len(evidence["paired_energy_observations"]) == 2
    assert evidence["paired_energy_observations"][0]["pair_id"] == "X-01"
    assert all(
        np.isfinite(row[key])
        for row in evidence["paired_energy_observations"]
        for key in (
            "reference_energy",
            "candidate_energy",
            "reference_cross_axis_energy",
            "candidate_cross_axis_energy",
        )
    )


def test_facade_rejects_cross_axis_regression_even_with_main_axis_improvement():
    training = {"X": [_capture(repeat=index) for index in range(3)]}
    first = analyze_calibration(
        captures=training,
        axes=("X",),
        profile="performance",
        snapshot=SimpleNamespace(square_corner_velocity=7.0, damping_ratio_x=0.08),
    )
    result = analyze_calibration(
        captures=training,
        held_out_captures={"X": [_capture(cross_scale=0.2, repeat=index) for index in range(3)]},
        validation_captures={
            "X": [_capture(scale=0.75, cross_scale=0.3, repeat=index) for index in range(3)]
        },
        axes=("X",),
        profile="performance",
        snapshot=SimpleNamespace(square_corner_velocity=7.0, damping_ratio_x=0.08),
        prior_report=first,
    )
    assert not result["validation"]["passed"]
    assert result["validation"]["axes"]["X"]["cross_axis_regression"] > 0.05


def test_validation_qc_failure_retains_reference_and_candidate_diagnostics():
    training = {"X": [_capture(repeat=index) for index in range(3)]}
    first = analyze_calibration(
        captures=training,
        axes=("X",),
        profile="performance",
        snapshot=SimpleNamespace(square_corner_velocity=7.0, damping_ratio_x=0.08),
    )
    invalid = [_capture(scale=0.7, repeat=index) for index in range(3)]
    invalid[1]["metadata"]["clip_limit"] = None
    result = analyze_calibration(
        captures=training,
        held_out_captures={"X": [_capture(repeat=index) for index in range(3)]},
        validation_captures={"X": invalid},
        axes=("X",),
        profile="performance",
        snapshot=SimpleNamespace(square_corner_velocity=7.0, damping_ratio_x=0.08),
        prior_report=first,
    )
    evidence = result["validation"]["axes"]["X"]
    assert result["validation"]["passed"] is False
    assert evidence["qc_passed"] is False
    assert len(evidence["reference_qc"]) == 3
    assert len(evidence["candidate_qc"]) == 3
    assert any(not row["passed"] for row in evidence["candidate_qc"])


def test_experimental_profile_promotes_generalized_mzv_with_measured_damping():
    report = analyze_calibration(
        captures={"X": [_capture(repeat=index) for index in range(3)]},
        axes=("X",),
        profile="experimental_mzv",
        snapshot=SimpleNamespace(square_corner_velocity=7.0, damping_ratio_x=0.09),
        experimental_mode=True,
    )
    assert not report.get("abstain")
    details = report["axes"]["X"]
    parameterized = [item for item in details["candidates"] if "(" in item["name"]]
    assert parameterized
    assert report["selections"][0]["shaper_type"].startswith("mzv(")
    assert details["generalized_mzv"]["selection_candidate_count"] == len(parameterized)
    assert details["damping_source"] == "measured_modes"
    assert details["design_damping_ratio"] != 0.1
    assert report["selections"][0]["damping_ratio"] == details["design_damping_ratio"]
    assert details["acceleration_limits"]["evidence_level"] == "theoretical"
    assert details["acceleration_limits"]["resonance_validated_mm_s2"] is None
    assert details["acceleration_limits"]["print_validated_mm_s2"] is None


def test_adaptive_stock_compares_native_and_parameterized_stock_candidates():
    report = analyze_calibration(
        captures={"X": [_capture(repeat=index) for index in range(3)]},
        axes=("X",),
        profile="adaptive_stock",
        snapshot=SimpleNamespace(square_corner_velocity=7.0, damping_ratio_x=0.09),
        experimental_mode=True,
    )

    assert not report.get("abstain")
    candidates = report["axes"]["X"]["candidates"]
    assert any(not item["metadata"].get("parameterized") for item in candidates)
    assert any(item["metadata"].get("parameterized") for item in candidates)
    assert all(
        item["metadata"]["cross_axis_metric"].startswith("predicted_cross_axis")
        for item in candidates
    )
    assert {
        item["metadata"]["cross_axis_model"]
        for item in candidates
        if item["metadata"].get("parameterized")
    } == {"oscillator_response_weighted_training_cross_psd"}
    assert {
        item["metadata"]["design_damping_ratio"]
        for item in candidates
        if not item["metadata"].get("parameterized")
    } == {0.08}
    selection = report["selections"][0]
    assert selection["shaper_type"] in {item["name"] for item in candidates}
    assert report["runtime_contract"] == {
        "interface": "stock_set_input_shaper",
        "families": ["zv", "mzv", "zvd", "ei", "2hump_ei", "3hump_ei"],
        "parameterized_family": "mzv",
        "arbitrary_pulse_vectors": False,
        "installed_capability_required": True,
        "held_out_validation_required": True,
    }
    assert report["native_command_preview"].startswith("SET_INPUT_SHAPER ")


def test_candidate_specific_cross_axis_response_can_change_selection():
    frequencies = np.asarray([5.0, 74.0, 111.0, 150.0])
    along = Spectrum(frequencies, np.asarray([0.0, 10.0, 0.0, 0.0]), 1000.0, 8)
    cross = Spectrum(frequencies, np.asarray([0.0, 0.0, 10.0, 0.0]), 1000.0, 8)
    base = {
        "frequency": 74.0,
        "residual_vibration": 0.04,
        "smoothing": 0.1,
        "max_accel": 10000.0,
        "design_damping_ratio": 0.08,
    }
    capture = {
        "native_candidates": [
            {
                **base,
                "name": "mzv",
                "native_frequency_response": {
                    "frequency_hz": frequencies.tolist(),
                    "response_ratio": [1.0, 0.05, 0.01, 1.0],
                },
            },
            {
                **base,
                "name": "zv",
                "native_frequency_response": {
                    "frequency_hz": frequencies.tolist(),
                    "response_ratio": [1.0, 0.05, 0.90, 1.0],
                },
            },
        ]
    }

    shared = _candidate_scores([capture], 0.01, 0.2)
    modeled = _candidate_scores([capture], 0.01, 0.2, along, cross)

    assert select_candidate(shared, PROFILES["adaptive_stock"]).selected.name == "zv"
    assert select_candidate(modeled, PROFILES["adaptive_stock"]).selected.name == "mzv"
    modeled_by_name = {item.name: item for item in modeled}
    assert modeled_by_name["mzv"].cross_axis_energy < modeled_by_name["zv"].cross_axis_energy


def test_experimental_optimizer_respects_installed_executor_pulse_limit():
    report = analyze_calibration(
        captures={"X": [_capture(repeat=index) for index in range(3)]},
        axes=("X",),
        profile="experimental_mzv",
        snapshot=SimpleNamespace(square_corner_velocity=7.0, damping_ratio_x=0.04),
        experimental_mode=True,
        executor_pulse_limit=5,
    )
    assert not report.get("abstain")
    parameterized = [
        item
        for item in report["axes"]["X"]["candidates"]
        if item.get("metadata", {}).get("parameterized")
    ]
    assert parameterized
    assert max(item["metadata"]["pulse_count"] for item in parameterized) <= 5


def test_experimental_peak_lock_uses_highest_psd_mode_exactly():
    report = analyze_calibration(
        captures={"X": [_capture(repeat=index) for index in range(3)]},
        axes=("X",),
        profile="experimental_mzv",
        snapshot=SimpleNamespace(square_corner_velocity=7.0, damping_ratio_x=0.04),
        experimental_mode=True,
        peak_lock=True,
    )
    assert not report.get("abstain")
    details = report["axes"]["X"]
    target = max(details["modes"], key=lambda mode: mode["amplitude"])["frequency"]
    assert details["generalized_mzv"]["frequency_strategy"] == "strongest_measured_peak"
    assert details["generalized_mzv"]["strongest_measured_peak_hz"] == target
    assert report["selections"][0]["frequency_hz"] == target
    parameterized = [
        item for item in details["candidates"] if item["metadata"].get("parameterized")
    ]
    assert parameterized
    assert {item["frequency"] for item in parameterized} == {target}
