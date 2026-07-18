from dataclasses import dataclass, field

import pytest

from klipper_advanced_shaper.klippy.adapter import (
    KlipperPrinterAdapter,
    PrinterSnapshot,
    ShaperSelection,
)
from klipper_advanced_shaper.klippy.plugin import AdvancedInputShaper
from klipper_advanced_shaper.klippy.state import CalibrationCancelled

SNAPSHOT = PrinterSnapshot(
    shaper_type_x="mzv",
    shaper_freq_x=50.0,
    shaper_type_y="ei",
    shaper_freq_y=42.0,
    max_velocity=300.0,
    max_accel=5000.0,
    square_corner_velocity=5.0,
)


@dataclass
class FakeAdapter:
    calls: list = field(default_factory=list)
    fail_capture_at: int = -1
    controller: object = None
    fail_restore: bool = False

    def preflight(self, axes):
        self.calls.append(("preflight", tuple(axes)))

    def snapshot(self):
        self.calls.append(("snapshot",))
        return SNAPSHOT

    def capture(self, axis, repeat, validation=False):
        self.calls.append(("capture", axis, repeat, validation))
        capture_count = sum(call[0] == "capture" for call in self.calls)
        if capture_count == self.fail_capture_at:
            raise RuntimeError("accelerometer disconnected")
        if self.controller is not None and capture_count == 1:
            self.controller.cancel()
        return {"axis": axis, "repeat": repeat, "validation": validation}

    def apply_temporary(self, selections):
        self.calls.append(("apply", tuple(selections)))

    def restore(self, snapshot):
        self.calls.append(("restore", snapshot))
        if self.fail_restore:
            raise RuntimeError("restore failed")

    def stage(self, selections):
        self.calls.append(("stage", tuple(selections)))

    def respond(self, message):
        self.calls.append(("respond", message))


def analyzer(**kwargs):
    if "validation_captures" in kwargs:
        return {"validation": {"passed": True, "axes": {}}}
    return {
        "selections": [
            {"axis": axis, "shaper_type": "mzv", "frequency_hz": 71.5} for axis in kwargs["axes"]
        ],
        "qc": {"passed": True},
    }


def test_calibration_validates_held_out_data_and_always_restores():
    adapter = FakeAdapter()
    plugin = AdvancedInputShaper(adapter=adapter, analyzer=analyzer, id_factory=lambda: "result-1")

    result = plugin.calibrate(("X",), repeats=3, validate=True)

    assert result.result_id == "result-1"
    training = [c for c in adapter.calls if c[0] == "capture" and not c[3]]
    shaped_comparisons = [c for c in adapter.calls if c[0] == "capture" and c[3]]
    assert len(training) == 3
    assert len(shaped_comparisons) == 6
    assert len([c for c in adapter.calls if c[0] == "apply"]) == 2
    assert adapter.calls[-1] == ("restore", SNAPSHOT)
    assert plugin.status()["state"] == "review"


def test_capture_failure_restores_snapshot_and_records_failure():
    adapter = FakeAdapter(fail_capture_at=2)
    plugin = AdvancedInputShaper(adapter=adapter, analyzer=analyzer)

    with pytest.raises(RuntimeError, match="accelerometer disconnected"):
        plugin.calibrate(("Y",), repeats=3)

    assert adapter.calls[-1] == ("restore", SNAPSHOT)
    assert plugin.status()["state"] == "failed"


def test_cancellation_restores_snapshot():
    adapter = FakeAdapter()
    plugin = AdvancedInputShaper(adapter=adapter, analyzer=analyzer)
    adapter.controller = plugin

    with pytest.raises(CalibrationCancelled):
        plugin.calibrate(("X",), repeats=3)

    assert adapter.calls[-1] == ("restore", SNAPSHOT)
    assert plugin.status()["state"] == "cancelled"


def test_apply_is_runtime_only_and_stage_is_explicit():
    adapter = FakeAdapter()
    plugin = AdvancedInputShaper(adapter=adapter, analyzer=analyzer, id_factory=lambda: "approved")
    plugin.calibrate(("X",), repeats=1, validate=False)

    assert not [call for call in adapter.calls if call[0] == "stage"]
    plugin.apply("approved")
    assert plugin.status()["state"] == "runtime_applied"
    assert len([call for call in adapter.calls if call[0] == "apply"]) == 1
    plugin.stage("approved")
    assert plugin.status()["state"] == "staged"
    assert len([call for call in adapter.calls if call[0] == "stage"]) == 1


def test_analysis_abstention_never_applies_candidate():
    adapter = FakeAdapter()
    plugin = AdvancedInputShaper(
        adapter=adapter,
        analyzer=lambda **_: {"abstain": True, "reason": "aliased signal"},
    )

    with pytest.raises(RuntimeError, match="aliased signal"):
        plugin.calibrate(("X",), repeats=1)

    assert not [call for call in adapter.calls if call[0] in {"apply", "stage"}]
    assert adapter.calls[-1] == ("restore", SNAPSHOT)


def test_preflight_failure_does_not_attempt_restore_without_snapshot():
    class RefusingAdapter(FakeAdapter):
        def preflight(self, axes):
            raise RuntimeError("printer is active")

    adapter = RefusingAdapter()
    plugin = AdvancedInputShaper(adapter=adapter, analyzer=analyzer)

    with pytest.raises(RuntimeError, match="printer is active"):
        plugin.calibrate(("X",), repeats=1)

    assert not [call for call in adapter.calls if call[0] == "restore"]
    assert plugin.status()["state"] == "failed"


def test_restore_failure_never_publishes_result():
    adapter = FakeAdapter(fail_restore=True)
    plugin = AdvancedInputShaper(adapter=adapter, analyzer=analyzer, id_factory=lambda: "unsafe")

    with pytest.raises(RuntimeError, match="restore failed"):
        plugin.calibrate(("X",), repeats=1, validate=True)

    assert plugin.status()["state"] == "failed"
    assert "unsafe" not in plugin.results


def test_real_adapter_uses_native_shaper_command_and_stage_does_not_save():
    class GCode:
        def __init__(self):
            self.scripts = []

        def run_script_from_command(self, command):
            self.scripts.append(command)

    class ConfigFile:
        def __init__(self):
            self.values = []

        def set(self, section, key, value):
            self.values.append((section, key, value))

    class Printer:
        def __init__(self):
            self.objects = {"gcode": GCode(), "configfile": ConfigFile()}

        def lookup_object(self, name, default=None):
            return self.objects.get(name, default)

    class Config:
        def __init__(self):
            self.printer = Printer()

        def get_printer(self):
            return self.printer

    config = Config()
    adapter = KlipperPrinterAdapter(config)
    selections = (
        ShaperSelection("mzv", 74.4, "X"),
        ShaperSelection("2hump_ei", 76.4, "Y"),
    )

    adapter.apply_temporary(selections)
    adapter.stage(selections)

    scripts = config.printer.objects["gcode"].scripts
    assert scripts == [
        "SET_INPUT_SHAPER DAMPING_RATIO_X=0.100000 DAMPING_RATIO_Y=0.100000 "
        "SHAPER_FREQ_X=74.400000 SHAPER_FREQ_Y=76.400000 "
        "SHAPER_TYPE_X=mzv SHAPER_TYPE_Y=2hump_ei"
    ]
    assert config.printer.objects["configfile"].values == [
        ("input_shaper", "shaper_type_x", "mzv"),
        ("input_shaper", "shaper_freq_x", "74.400"),
        ("input_shaper", "damping_ratio_x", "0.1000"),
        ("input_shaper", "shaper_type_y", "2hump_ei"),
        ("input_shaper", "shaper_freq_y", "76.400"),
        ("input_shaper", "damping_ratio_y", "0.1000"),
    ]
    assert all("SAVE_CONFIG" not in script for script in scripts)
