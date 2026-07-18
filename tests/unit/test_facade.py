from types import SimpleNamespace

import numpy as np

from klipper_advanced_shaper.analysis import analyze_calibration


def _capture(axis="X", scale=1.0, repeat=0, cross_scale=0.0):
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
    return {
        "samples": np.column_stack((t, xyz)),
        "native_candidates": candidates,
        "metadata": {"clip_limit": 100.0},
    }


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
