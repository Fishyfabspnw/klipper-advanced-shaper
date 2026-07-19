from pathlib import Path

import pytest

from klipper_advanced_shaper.klippy.capture import _Command
from klipper_advanced_shaper.klippy.excitation import (
    check_motion_budget,
    check_sweep_rate,
)
from klipper_advanced_shaper.klippy.plugin import (
    parse_accel_per_hz,
    parse_hz_per_sec,
    parse_square_corner_velocity,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("CONFIG", None),
        ("config", None),
        ("20", 20.0),
        ("20.125", 20.125),
        ("45.5", 45.5),
        (60, 60.0),
        ("349.999", 349.999),
        (350, 350.0),
    ],
)
def test_accel_per_hz_parser_accepts_config_or_bounded_decimals(value, expected):
    assert parse_accel_per_hz(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        " 30",
        "30 ",
        "+30",
        "-30",
        "030",
        ".30",
        "3e1",
        "NaN",
        "inf",
        "30,45",
        "19.999",
        "350.001",
        "HIGH_INTENSITY",
    ],
)
def test_accel_per_hz_parser_rejects_out_of_range_or_non_decimal_values(value):
    with pytest.raises(ValueError, match="ACCEL_PER_HZ"):
        parse_accel_per_hz(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("CONFIG", None),
        ("config", None),
        ("0.1", 0.1),
        ("1", 1.0),
        ("1.75", 1.75),
        (2, 2.0),
    ],
)
def test_hz_per_sec_parser_accepts_config_or_bounded_decimals(value, expected):
    assert parse_hz_per_sec(value) == expected


@pytest.mark.parametrize(
    "value",
    ["", " 1", "1 ", "+1", "-1", "01", ".5", "1e0", "NaN", "inf", "0.099", "2.001"],
)
def test_hz_per_sec_parser_rejects_out_of_range_or_non_decimal_values(value):
    with pytest.raises(ValueError, match="HZ_PER_SEC"):
        parse_hz_per_sec(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), ("CONFIG", None), ("0.1", 0.1), ("15", 15.0), (50, 50.0)],
)
def test_scv_parser_accepts_config_or_bounded_decimals(value, expected):
    assert parse_square_corner_velocity(value) == expected


@pytest.mark.parametrize("value", ["0", "0.099", "50.001", "+15", "015", "1e1"])
def test_scv_parser_rejects_unsafe_or_noncanonical_values(value):
    with pytest.raises(ValueError, match="SCV"):
        parse_square_corner_velocity(value)


def test_effective_configured_sweep_rate_is_checked_fail_closed():
    assert check_sweep_rate(2.0) == 2.0
    with pytest.raises(RuntimeError, match="effective .* hz_per_sec"):
        check_sweep_rate(2.1)


def test_motion_budget_includes_sweeping_accel_and_eighty_percent_margin():
    result = check_motion_budget(
        accel_per_hz=100.0,
        max_frequency_hz=100.0,
        printer_max_accel=20_000.0,
        sweeping_accel=400.0,
    )

    assert result["pulse_peak_accel_mm_s2"] == 10_000.0
    assert result["estimated_peak_accel_mm_s2"] == 10_400.0
    assert result["allowed_peak_accel_mm_s2"] == 16_000.0
    assert result["motion_limit_fraction"] == 0.8


def test_motion_budget_rejects_bounded_value_that_exceeds_printer_limit():
    with pytest.raises(RuntimeError, match="80% motion budget"):
        check_motion_budget(
            accel_per_hz=350.0,
            max_frequency_hz=135.0,
            printer_max_accel=20_000.0,
            sweeping_accel=400.0,
        )


def test_configured_effective_value_is_also_range_checked():
    with pytest.raises(RuntimeError, match=r"effective \[resonance_tester\]"):
        check_motion_budget(351.0, 100.0, 50_000.0)


def test_capture_command_overrides_only_explicit_recipe_parameters():
    inherited = _Command(validation=False, responder=lambda _message: None)
    overridden = _Command(
        validation=True,
        responder=lambda _message: None,
        accel_per_hz=45.5,
        hz_per_sec=2.0,
    )

    assert inherited.get_float("ACCEL_PER_HZ", 52.5) == 52.5
    assert overridden.get_float("ACCEL_PER_HZ", 52.5) == 45.5
    assert inherited.get_float("HZ_PER_SEC", 1.25) == 1.25
    assert overridden.get_float("HZ_PER_SEC", 1.25, maxval=2.0) == 2.0
    assert overridden.get_int("INPUT_SHAPING") == 1


def test_direct_numeric_calibration_is_the_only_mainsail_visible_macro():
    macros = (ROOT / "config" / "advanced_shaper_macros.cfg").read_text(encoding="utf-8")
    sections = [
        line.removeprefix("[gcode_macro ").removesuffix("]")
        for line in macros.splitlines()
        if line.startswith("[gcode_macro ")
    ]

    assert [name for name in sections if not name.startswith("_")] == [
        "ADV_SHAPER_UI_CALIBRATE"
    ]
    assert 'params.ACCEL_PER_HZ|default("CONFIG")' in macros
    assert 'params.HZ_PER_SEC|default("CONFIG")' in macros
    assert 'params.SCV|default("CONFIG")' in macros
    assert "params.FAST_VALIDATION|default(0)" in macros
    assert "params.PEAK_LOCK|default(0)" in macros
    assert "PEAK_LOCK={peak_lock}" in macros
    assert "REPEATS=2 VALIDATE=1 HZ_PER_SEC=2" in macros
    assert "CONFIG or 20..350" in macros
    for explanation in (
        "AXIS chooses X, Y, or ALL",
        "PROFILE chooses the selection tradeoff",
        "REPEATS controls statistical confidence",
        "VALIDATE enables held-out comparison",
        "ACCEL_PER_HZ sets excitation intensity",
        "HZ_PER_SEC sets sweep speed",
        "SCV sets temporary square-corner velocity",
        "FAST_VALIDATION=1 runs one training, two reference, and two candidate sweeps",
        "PEAK_LOCK=1 fixes generalized MZV frequency",
    ):
        assert explanation in macros
    assert "action:prompt_" not in macros
    assert "[respond]" not in macros


def test_calibration_macro_never_applies_stages_or_saves():
    macros = (ROOT / "config" / "advanced_shaper_macros.cfg").read_text(encoding="utf-8")
    calibration = macros.split("[gcode_macro ADV_SHAPER_UI_CALIBRATE]", 1)[1].split(
        "[gcode_macro _ADV_SHAPER_UI_STATUS]", 1
    )[0]

    assert "ADV_SHAPER_CALIBRATE AXIS={axis}" in calibration
    assert "ADV_SHAPER_APPLY" not in calibration
    assert "ADV_SHAPER_STAGE" not in calibration
    assert "SAVE_CONFIG" not in calibration
    assert "will not apply, stage, or save" in calibration
