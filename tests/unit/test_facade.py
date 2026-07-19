from types import SimpleNamespace

import numpy as np
import pytest

from klipper_advanced_shaper.analysis import analyze_calibration


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
        },
        {
            "name": "ei",
            "frequency": 86.0,
            "residual_vibration": 0.02,
            "smoothing": 0.13,
            "max_accel": 12000,
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
        snapshot=SimpleNamespace(square_corner_velocity=7.0),
    )
    assert not report.get("abstain")
    assert report["selections"][0]["shaper_type"] == "mzv"
    assert report["axes"]["X"]["modes"]
    assert report["square_corner_velocity"] == 7.0


def test_facade_exposes_native_components_responses_and_bounded_spectrogram():
    captures = {"X": [_capture(repeat=index) for index in range(3)]}
    report = analyze_calibration(
        captures=captures,
        axes=("X",),
        profile="performance",
        snapshot=SimpleNamespace(square_corner_velocity=7.0),
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
        snapshot=SimpleNamespace(square_corner_velocity=7.0),
    )
    without_display = analyze_calibration(
        captures={
            "X": [_capture(repeat=index, display_data=False) for index in range(3)]
        },
        axes=("X",),
        profile="performance",
        snapshot=SimpleNamespace(square_corner_velocity=7.0),
    )
    assert with_display["selections"] == without_display["selections"]
    assert without_display["axes"]["X"]["native_spectrum"]["available"] is False


def test_facade_requires_statistically_lower_held_out_energy():
    training = {"X": [_capture(repeat=index) for index in range(3)]}
    first = analyze_calibration(
        captures=training,
        axes=("X",),
        profile="performance",
        snapshot=SimpleNamespace(square_corner_velocity=7.0),
    )
    result = analyze_calibration(
        captures=training,
        held_out_captures={"X": [_capture(scale=1.0, repeat=index) for index in range(3)]},
        validation_captures={"X": [_capture(scale=0.75, repeat=index) for index in range(3)]},
        axes=("X",),
        profile="performance",
        snapshot=SimpleNamespace(square_corner_velocity=7.0),
        prior_report=first,
    )
    assert result["validation"]["passed"]
    assert result["validation"]["axes"]["X"]["improvement_ci_95"][0] > 0.10


def test_facade_rejects_cross_axis_regression_even_with_main_axis_improvement():
    training = {"X": [_capture(repeat=index) for index in range(3)]}
    first = analyze_calibration(
        captures=training,
        axes=("X",),
        profile="performance",
        snapshot=SimpleNamespace(square_corner_velocity=7.0),
    )
    result = analyze_calibration(
        captures=training,
        held_out_captures={"X": [_capture(cross_scale=0.2, repeat=index) for index in range(3)]},
        validation_captures={
            "X": [_capture(scale=0.75, cross_scale=0.3, repeat=index) for index in range(3)]
        },
        axes=("X",),
        profile="performance",
        snapshot=SimpleNamespace(square_corner_velocity=7.0),
        prior_report=first,
    )
    assert not result["validation"]["passed"]
    assert result["validation"]["axes"]["X"]["cross_axis_regression"] > 0.05
