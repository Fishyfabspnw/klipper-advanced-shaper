import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from klipper_advanced_shaper.analysis import (
    PROFILES,
    CandidateScore,
    aggregate_spectra,
    assess_quality,
    attenuation_improvement_ci,
    find_modes,
    integrated_band_energy,
    pareto_frontier,
    project_axes,
    resample_uniform,
    select_candidate,
    transfer_coherence,
    welch_psd,
)


def test_timestamp_resampling_recovers_frequency():
    rng = np.random.default_rng(4)
    dt = (1 / 1000) * (1 + rng.normal(0, 0.025, 4000))
    t = np.cumsum(dt)
    x = np.sin(2 * np.pi * 73 * t)
    rt, rx, rate = resample_uniform(t, x)
    spectrum = welch_psd(rx, rate, nperseg=1000)
    peak = spectrum.frequencies[np.argmax(spectrum.values)]
    assert np.allclose(np.diff(rt), 1 / rate)
    assert peak == pytest.approx(73, abs=1)


def test_welch_density_integrates_to_variance():
    rng = np.random.default_rng(9)
    x = rng.normal(size=8192)
    spectrum = welch_psd(x, 2000, nperseg=1024)
    assert np.trapezoid(spectrum.values, spectrum.frequencies) == pytest.approx(np.var(x), rel=0.08)
    assert spectrum.segments == 15


def test_repeat_aggregation_handles_different_fft_grids():
    rate = 1000
    t = np.arange(4096) / rate
    first = welch_psd(np.sin(2 * np.pi * 70 * t), rate, nperseg=512)
    second = welch_psd(1.02 * np.sin(2 * np.pi * 70 * t), rate, nperseg=1024)
    aggregate, relative_mad = aggregate_spectra([first, second])
    assert aggregate.segments == 2
    assert aggregate.frequencies[np.argmax(aggregate.values)] == pytest.approx(70, abs=2)
    assert np.median(relative_mad) >= 0
    assert integrated_band_energy(aggregate, 60, 80) > 0.4


def test_transfer_and_coherence_for_known_linear_response():
    rng = np.random.default_rng(2)
    excitation = rng.normal(size=16384)
    response = 2.5 * excitation + rng.normal(scale=0.1, size=excitation.size)
    metrics = transfer_coherence(excitation, response, 2000, nperseg=1024)
    band = (metrics.frequencies > 20) & (metrics.frequencies < 500)
    assert np.median(np.abs(metrics.transfer[band])) == pytest.approx(2.5, rel=0.02)
    assert np.median(metrics.coherence[band]) > 0.99


def test_axis_projection_separates_cross_axis_energy():
    values = np.array([[1.0, 3.0, 4.0], [2.0, 0.0, 0.0]])
    along, cross = project_axes(values, np.array([1.0, 0.0, 0.0]))
    assert np.array_equal(along, [1, 2])
    assert np.allclose(cross, [5, 0])


def test_quality_flags_dropout_clipping_alias_and_noise():
    t = np.arange(1000) / 1000
    t[501:] += 0.01
    x = np.tile([10.0, -10.0], 500)
    report = assess_quality(t, x, expected_band_max=450, clip_limit=10, max_dropout_ratio=0)
    codes = {issue.code for issue in report.issues}
    assert not report.passed
    assert {"sample_dropout", "sensor_clipping", "aliasing_risk", "excess_noise"} <= codes


def test_quality_accepts_clean_low_frequency_capture():
    t = np.arange(4000) / 2000
    x = np.sin(2 * np.pi * 50 * t)
    report = assess_quality(t, x, expected_band_max=200, max_noise_ratio=0.5)
    assert report.passed


def test_modal_estimation_finds_multimodal_peaks_and_damping():
    f = np.linspace(0, 200, 2001)
    p = 0.01 + 3 / (1 + ((f - 50) / 2) ** 2) + 1.5 / (1 + ((f - 91) / 4) ** 2)
    modes = find_modes(f, p, min_prominence_ratio=0.1)
    assert [m.frequency for m in modes] == pytest.approx([50, 91], abs=0.2)
    assert modes[0].damping_ratio == pytest.approx(0.04, abs=0.005)


def _candidate(name, residual, smoothing, accel, repeat=0.03, cross=0.04, sensitivity=0.03):
    return CandidateScore(name, 60, residual, smoothing, accel, repeat, cross, sensitivity)


def test_pareto_and_profiles_make_distinct_defensible_choices():
    quality = _candidate("quality", 0.02, 0.25, 6000)
    balanced = _candidate("balanced", 0.07, 0.12, 12000)
    performance = _candidate("performance", 0.17, 0.04, 20000)
    dominated = _candidate("dominated", 0.20, 0.30, 5000)
    candidates = [quality, balanced, performance, dominated]
    assert {c.name for c in pareto_frontier(candidates)} == {"quality", "balanced", "performance"}
    assert select_candidate(candidates, PROFILES["quality"]).selected.name == "quality"
    assert select_candidate(candidates, PROFILES["performance"]).selected.name == "performance"


def test_selection_abstains_when_safety_gate_fails():
    result = select_candidate([_candidate("risky", 0.25, 0.01, 30000)], PROFILES["balanced"])
    assert result.selected is None
    assert result.abstention_reason


def test_held_out_attenuation_bootstrap_requires_three_repeats():
    low, high = attenuation_improvement_ci(
        np.array([10, 11, 9, 10]), np.array([7, 7.7, 6.3, 7]), seed=1
    )
    assert low == pytest.approx(0.3)
    assert high == pytest.approx(0.3)
    with pytest.raises(ValueError):
        attenuation_improvement_ci(np.ones(2), np.ones(2))
