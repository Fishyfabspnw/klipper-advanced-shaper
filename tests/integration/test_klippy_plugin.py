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
    damping_ratio_x=0.08,
    damping_ratio_y=0.06,
)


@dataclass
class FakeAdapter:
    calls: list = field(default_factory=list)
    fail_capture_at: int = -1
    controller: object = None
    fail_restore: bool = False

    def configure_capture_profile(self, profile):
        self.capture_profile = profile

    def preflight(self, axes):
        self.calls.append(("preflight", tuple(axes)))

    def preflight_excitation(self, axes, accel_per_hz, hz_per_sec):
        self.calls.append(
            ("excitation_preflight", tuple(axes), accel_per_hz, hz_per_sec)
        )
        effective = 60.0 if accel_per_hz is None else accel_per_hz
        effective_rate = 1.0 if hz_per_sec is None else hz_per_sec
        return {
            "accel_per_hz": effective,
            "min_frequency_hz": 5.0,
            "max_frequency_hz": 100.0,
            "estimated_peak_accel_mm_s2": effective * 100.0,
            "allowed_peak_accel_mm_s2": 8000.0,
            "motion_limit_fraction": 0.8,
            "hz_per_sec": effective_rate,
            "hz_per_sec_source": "command" if hz_per_sec is not None else "resonance_tester",
        }

    def preflight_transient(self, axes):
        self.calls.append(("transient_preflight", tuple(axes)))
        return {
            "passed": True,
            "protocol": "finite_reversal_ringdown_v1",
            "speed_mm_s": 80.0,
            "max_accel_mm_s2": 5000.0,
            "estimated_base_motion_seconds_per_capture_upper_bound": 4.0,
            "estimated_motion_seconds_per_capture_upper_bound": 4.0,
            "plans": {
                axis: {
                    "axis": axis,
                    "original_position_mm": 50.0,
                    "anchor_position_mm": 50.0,
                    "start_position_mm": 46.0,
                    "reversal_position_mm": 54.0,
                    "return_position_mm": 46.0,
                }
                for axis in axes
            },
        }

    def snapshot(self):
        self.calls.append(("snapshot",))
        return SNAPSHOT

    def capture(
        self,
        axis,
        repeat,
        validation=False,
        accel_per_hz=None,
        hz_per_sec=None,
        max_vibrations=None,
    ):
        self.calls.append(
            (
                "capture",
                axis,
                repeat,
                validation,
                accel_per_hz,
                hz_per_sec,
                max_vibrations,
            )
        )
        capture_count = sum(
            call[0] in {"capture", "transient"} for call in self.calls
        )
        if capture_count == self.fail_capture_at:
            raise RuntimeError("accelerometer disconnected")
        if self.controller is not None and capture_count == 1:
            self.controller.cancel()
        return {"axis": axis, "repeat": repeat, "validation": validation}

    def apply_temporary(self, selections):
        self.calls.append(("apply", tuple(selections)))

    def verify_live_python_pulses(self, selections):
        self.calls.append(("verify_live_pulses", tuple(selections)))
        return {
            "passed": True,
            "live_c_executor_readback": False,
            "axes": {
                item.axis: {
                    "post_command_guard_seconds": 0.03,
                    "pulse_span_seconds": 0.01,
                }
                for item in selections
            },
        }

    def capture_transient(
        self,
        axis,
        repeat,
        plan,
        max_accel_mm_s2,
        speed_mm_s,
        post_command_guard_seconds,
    ):
        self.calls.append(
            (
                "transient",
                axis,
                repeat,
                True,
                None,
                None,
                None,
                "finite_reversal_ringdown",
                plan,
                max_accel_mm_s2,
                speed_mm_s,
                post_command_guard_seconds,
            )
        )
        capture_count = sum(
            call[0] in {"capture", "transient"} for call in self.calls
        )
        if capture_count == self.fail_capture_at:
            raise RuntimeError("accelerometer disconnected")
        return {
            "axis": axis,
            "repeat": repeat,
            "validation": True,
            "metadata": {
                "promotion_eligible": True,
                "protocol": "finite_reversal_ringdown_v1",
                "validation_capture_kind": "finite_reversal_ringdown",
            },
        }

    def set_test_square_corner_velocity(self, value):
        self.calls.append(("set_scv", float(value)))

    def set_test_max_accel(self, value):
        self.calls.append(("set_max_accel", float(value)))

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
            {
                "axis": axis,
                "shaper_type": "mzv",
                "frequency_hz": 71.5,
                "damping_ratio": 0.07,
            }
            for axis in kwargs["axes"]
        ],
        "qc": {"passed": True},
    }


def rejecting_analyzer(**kwargs):
    if "validation_captures" in kwargs:
        return {
            "qc": {"passed": True, "held_out_repeats": 2},
            "validation": {
                "passed": False,
                "reason": "confidence interval below target",
                "axes": {
                    "X": {
                        "baseline_energy": 12.0,
                        "shaped_energy": 11.4,
                        "improvement_ci_95": [0.01, 0.09],
                        "cross_axis_regression": 0.02,
                        "passed": False,
                    }
                },
            }
        }
    return {
        "selections": [
            {
                "axis": "X",
                "shaper_type": "mzv",
                "frequency_hz": 71.5,
                "damping_ratio": 0.07,
            }
        ],
        "qc": {"passed": True, "dropout_ratio": 0.0},
        "provenance": {"engine": "test-native"},
        "axes": {"X": {"selected": "mzv", "candidates": []}},
    }


@dataclass
class RecordingArtifactWriter:
    adapter: FakeAdapter
    fail: bool = False
    calls: list = field(default_factory=list)

    def write(self, result_id, report, raw_groups=None):
        self.adapter.calls.append(("artifact", result_id))
        self.calls.append(
            {"result_id": result_id, "report": report, "raw_groups": raw_groups}
        )
        if self.fail:
            raise RuntimeError("artifact disk full")
        return {"json": "/private/result.json", "raw": "/private/captures.npz"}


def test_calibration_validates_held_out_data_and_always_restores():
    adapter = FakeAdapter()
    plugin = AdvancedInputShaper(adapter=adapter, analyzer=analyzer, id_factory=lambda: "result-1")

    result = plugin.calibrate(("X",), repeats=3, validate=True)

    assert result.result_id == "result-1"
    training = [c for c in adapter.calls if c[0] == "capture" and not c[3]]
    shaped_comparisons = [c for c in adapter.calls if c[0] == "capture" and c[3]]
    assert len(training) == 3
    assert len(shaped_comparisons) == 6
    assert {call[6] for call in adapter.calls if call[0] == "capture"} == {None}
    validation_actions = [
        call for call in adapter.calls if call[0] in {"apply", "capture"}
    ][3:]
    assert [call[0] for call in validation_actions] == [
        "apply", "capture", "apply", "capture",
        "apply", "capture", "apply", "capture",
        "apply", "capture", "apply", "capture",
    ]
    assert [
        call[1][0].frequency for call in validation_actions if call[0] == "apply"
    ] == [50.0, 71.5, 50.0, 71.5, 50.0, 71.5]
    protocol = result.report["validation_protocol"]
    assert protocol["capture_design"] == "paired_interleaved_ab"
    assert protocol["pair_ids_by_axis"] == {"X": ["X-01", "X-02", "X-03"]}
    assert [row["condition_label"] for row in protocol["capture_order"]] == [
        "A", "B", "A", "B", "A", "B"
    ]
    assert adapter.calls[-1] == ("restore", SNAPSHOT)
    assert plugin.status()["state"] == "review"


def test_default_native_validation_does_not_require_transient_private_apis():
    adapter = FakeAdapter()
    adapter.preflight_transient = None
    adapter.capture_transient = None
    adapter.verify_live_python_pulses = None
    adapter.set_test_max_accel = None
    plugin = AdvancedInputShaper(
        adapter=adapter, analyzer=analyzer, id_factory=lambda: "legacy-native"
    )

    result = plugin.calibrate(("X",), profile="balanced", repeats=2, validate=True)

    assert result.report["validation_protocol"]["capture_design"] == (
        "paired_interleaved_ab"
    )
    assert len([call for call in adapter.calls if call[0] == "capture"]) == 6
    assert not [call for call in adapter.calls if call[0] == "transient"]


def test_failure_during_interleaved_pair_restores_exact_snapshot():
    adapter = FakeAdapter(fail_capture_at=4)
    plugin = AdvancedInputShaper(adapter=adapter, analyzer=analyzer)

    with pytest.raises(RuntimeError, match="accelerometer disconnected"):
        plugin.calibrate(("X",), repeats=2, validate=True)

    validation_actions = [
        call for call in adapter.calls if call[0] in {"apply", "capture"}
    ][2:]
    assert [call[0] for call in validation_actions] == [
        "apply", "capture", "apply", "capture"
    ]
    assert adapter.calls[-1] == ("restore", SNAPSHOT)


def test_bounded_numeric_acceleration_is_used_for_training_and_held_out_validation():
    adapter = FakeAdapter()
    plugin = AdvancedInputShaper(adapter=adapter, analyzer=analyzer)

    plugin.calibrate(("X",), repeats=2, validate=True, accel_per_hz="45.5")

    captures = [call for call in adapter.calls if call[0] == "capture"]
    assert len(captures) == 6
    assert {call[4] for call in captures} == {45.5}
    assert plugin.results[next(iter(plugin.results))].report["excitation_preflight"] == {
        "accel_per_hz": 45.5,
        "min_frequency_hz": 5.0,
        "max_frequency_hz": 100.0,
        "estimated_peak_accel_mm_s2": 4550.0,
        "allowed_peak_accel_mm_s2": 8000.0,
        "motion_limit_fraction": 0.8,
        "hz_per_sec": 1.0,
        "hz_per_sec_source": "resonance_tester",
    }
    assert adapter.calls[-1] == ("restore", SNAPSHOT)


def test_inherited_acceleration_does_not_override_resonance_tester_default():
    adapter = FakeAdapter()
    plugin = AdvancedInputShaper(adapter=adapter, analyzer=analyzer)

    plugin.calibrate(("Y",), repeats=1, validate=False, accel_per_hz="CONFIG")

    capture = next(call for call in adapter.calls if call[0] == "capture")
    assert capture[4] is None


def test_bounded_numeric_350_propagates_to_both_axes():
    adapter = FakeAdapter()
    plugin = AdvancedInputShaper(adapter=adapter, analyzer=analyzer)

    plugin.calibrate(
        ("X", "Y"),
        repeats=1,
        validate=False,
        accel_per_hz="350",
    )

    captures = [call for call in adapter.calls if call[0] == "capture"]
    assert [(call[1], call[4]) for call in captures] == [("X", 350.0), ("Y", 350.0)]
    assert not [call for call in adapter.calls if call[0] == "stage"]
    assert adapter.calls[-1] == ("restore", SNAPSHOT)


def test_explicit_scv_is_applied_after_snapshot_and_restored_exactly():
    seen = []

    def scv_analyzer(**kwargs):
        seen.append(kwargs["snapshot"].square_corner_velocity)
        return analyzer(**kwargs)

    adapter = FakeAdapter()
    plugin = AdvancedInputShaper(adapter=adapter, analyzer=scv_analyzer)

    result = plugin.calibrate(
        ("X",), repeats=1, validate=False, square_corner_velocity="15"
    )

    assert adapter.calls.index(("snapshot",)) < adapter.calls.index(("set_scv", 15.0))
    assert adapter.calls[-1] == ("restore", SNAPSHOT)
    assert seen == [15.0]
    assert result.report["validation_protocol"]["square_corner_velocity"] == 15.0
    assert result.report["validation_protocol"]["square_corner_velocity_source"] == "command"


def test_dynamic_excitation_rejection_happens_before_snapshot_or_capture():
    class MotionLimitedAdapter(FakeAdapter):
        def preflight_excitation(self, axes, accel_per_hz, hz_per_sec):
            self.calls.append(
                ("excitation_preflight", tuple(axes), accel_per_hz, hz_per_sec)
            )
            raise RuntimeError("estimated peak exceeds the 80% motion budget")

    adapter = MotionLimitedAdapter()
    plugin = AdvancedInputShaper(adapter=adapter, analyzer=analyzer)

    with pytest.raises(RuntimeError, match="80% motion budget"):
        plugin.calibrate(("X", "Y"), repeats=1, validate=False, accel_per_hz="350")

    assert adapter.calls == [
        ("preflight", ("X", "Y")),
        ("excitation_preflight", ("X", "Y"), 350.0, None),
    ]


def test_numeric_sweep_rate_propagates_to_training_reference_and_candidate_sweeps():
    adapter = FakeAdapter()
    plugin = AdvancedInputShaper(adapter=adapter, analyzer=analyzer)

    plugin.calibrate(
        ("X", "Y"),
        repeats=3,
        validate=True,
        accel_per_hz="150",
        hz_per_sec="2",
    )

    captures = [call for call in adapter.calls if call[0] == "capture"]
    assert len(captures) == 18
    assert {call[5] for call in captures} == {2.0}
    assert adapter.calls[1] == (
        "excitation_preflight",
        ("X", "Y"),
        150.0,
        2.0,
    )
    result = plugin.results[next(iter(plugin.results))]
    assert result.report["excitation_preflight"]["hz_per_sec"] == 2.0
    assert result.report["excitation_preflight"]["hz_per_sec_source"] == "command"


@pytest.mark.parametrize(
    "value",
    ["19.999", "350.001", "nan", "1e2", "+30", "30 junk", -1],
)
def test_invalid_acceleration_fails_before_preflight_or_motion(value):
    adapter = FakeAdapter()
    plugin = AdvancedInputShaper(adapter=adapter, analyzer=analyzer)

    with pytest.raises(ValueError, match="ACCEL_PER_HZ"):
        plugin.calibrate(("X",), repeats=1, accel_per_hz=value)

    assert adapter.calls == []


@pytest.mark.parametrize(
    "value",
    ["0.099", "2.001", "nan", "1e0", "+1", "1 junk", -1],
)
def test_invalid_sweep_rate_fails_before_preflight_or_motion(value):
    adapter = FakeAdapter()
    plugin = AdvancedInputShaper(adapter=adapter, analyzer=analyzer)

    with pytest.raises(ValueError, match="HZ_PER_SEC"):
        plugin.calibrate(("X",), repeats=1, hz_per_sec=value)

    assert adapter.calls == []


def test_capture_failure_restores_snapshot_and_records_failure():
    adapter = FakeAdapter(fail_capture_at=2)
    plugin = AdvancedInputShaper(adapter=adapter, analyzer=analyzer)

    with pytest.raises(RuntimeError, match="accelerometer disconnected"):
        plugin.calibrate(("Y",), repeats=3)

    assert adapter.calls[-1] == ("restore", SNAPSHOT)
    assert plugin.status()["state"] == "failed"


def test_failed_calibration_can_retry_without_klipper_restart():
    adapter = FakeAdapter(fail_capture_at=1)
    plugin = AdvancedInputShaper(
        adapter=adapter, analyzer=analyzer, id_factory=lambda: "retry-result"
    )

    with pytest.raises(RuntimeError, match="accelerometer disconnected"):
        plugin.calibrate(("X",), repeats=1, validate=False)

    result = plugin.calibrate(("X",), repeats=1, validate=False)

    assert result.result_id == "retry-result"
    assert plugin.status()["state"] == "review"
    assert len([call for call in adapter.calls if call[0] == "restore"]) == 2


def test_rejected_validation_is_written_only_after_rollback_with_full_diagnostics():
    adapter = FakeAdapter()
    writer = RecordingArtifactWriter(adapter)
    plugin = AdvancedInputShaper(
        adapter=adapter,
        analyzer=rejecting_analyzer,
        artifact_writer=writer,
        id_factory=lambda: "rejected-attempt",
    )

    with pytest.raises(RuntimeError, match="confidence interval below target"):
        plugin.calibrate(("X",), repeats=2, validate=True)

    assert adapter.calls.index(("restore", SNAPSHOT)) < adapter.calls.index(
        ("artifact", "rejected-attempt")
    )
    written = writer.calls[0]
    assert written["report"]["status"] == "rejected"
    assert written["report"]["attempt_id"] == "rejected-attempt"
    assert written["report"]["qc"]["passed"] is True
    assert written["report"]["provenance"]["engine"] == "test-native"
    assert written["report"]["validation"]["axes"]["X"]["baseline_energy"] == 12.0
    assert written["report"]["validation_report"]["qc"]["held_out_repeats"] == 2
    assert set(written["raw_groups"]) == {"training", "reference", "candidate"}
    assert all(len(written["raw_groups"][group]["X"]) == 2 for group in written["raw_groups"])
    assert "rejected-attempt" not in plugin.results
    with pytest.raises(ValueError, match="unknown result"):
        plugin.apply("rejected-attempt")
    with pytest.raises(ValueError, match="unknown result"):
        plugin.stage("rejected-attempt")
    status = dict(plugin.status())
    protocol = status.pop("validation_protocol")
    assert status == {
        "state": "failed",
        "result_id": None,
        "attempt_id": "rejected-attempt",
        "attempt_status": "rejected",
        "artifacts": {"json": "/private/result.json", "raw": "/private/captures.npz"},
        "cancel_requested": False,
        "error": "candidate failed held-out validation: confidence interval below target",
        "experimental_generalized_mzv_enabled": False,
    }
    assert protocol["capture_design"] == "paired_interleaved_ab"
    assert protocol["pair_ids_by_axis"] == {"X": ["X-01", "X-02"]}
    assert [row["condition_label"] for row in protocol["capture_order"]] == [
        "A", "B", "A", "B"
    ]
    assert written["report"]["validation_protocol"] == protocol


def test_new_rejected_attempt_does_not_expose_stale_accepted_result_id():
    adapter = FakeAdapter()
    writer = RecordingArtifactWriter(adapter)
    attempts = iter(("accepted-result", "rejected-result"))
    plugin = AdvancedInputShaper(
        adapter=adapter,
        analyzer=rejecting_analyzer,
        artifact_writer=writer,
        id_factory=lambda: next(attempts),
    )
    plugin.calibrate(("X",), repeats=1, validate=False)
    assert plugin.status()["result_id"] == "accepted-result"

    with pytest.raises(RuntimeError, match="failed held-out validation"):
        plugin.calibrate(("X",), repeats=1, validate=True)

    assert "accepted-result" in plugin.results
    assert plugin.status()["result_id"] is None
    assert plugin.status()["attempt_id"] == "rejected-result"


def test_rejection_artifact_failure_reports_both_errors_after_rollback():
    adapter = FakeAdapter()
    writer = RecordingArtifactWriter(adapter, fail=True)
    plugin = AdvancedInputShaper(
        adapter=adapter,
        analyzer=rejecting_analyzer,
        artifact_writer=writer,
        id_factory=lambda: "artifact-failure",
    )

    with pytest.raises(RuntimeError) as raised:
        plugin.calibrate(("X",), repeats=1, validate=True)

    assert "confidence interval below target" in str(raised.value)
    assert "artifact disk full" in str(raised.value)
    assert adapter.calls.index(("restore", SNAPSHOT)) < adapter.calls.index(
        ("artifact", "artifact-failure")
    )
    assert plugin.status()["attempt_status"] == "failed"
    assert "artifact disk full" in plugin.status()["error"]
    assert "artifact-failure" not in plugin.results


def test_rejection_rollback_failure_prevents_artifact_write_and_reports_both_errors():
    adapter = FakeAdapter(fail_restore=True)
    writer = RecordingArtifactWriter(adapter)
    plugin = AdvancedInputShaper(
        adapter=adapter,
        analyzer=rejecting_analyzer,
        artifact_writer=writer,
        id_factory=lambda: "rollback-failure",
    )

    with pytest.raises(RuntimeError) as raised:
        plugin.calibrate(("X",), repeats=1, validate=True)

    assert "confidence interval below target" in str(raised.value)
    assert "restoration also failed: restore failed" in str(raised.value)
    assert writer.calls == []
    assert "restore failed" in plugin.status()["error"]
    assert plugin.status()["attempt_status"] == "failed"


def test_capture_failure_does_not_persist_partial_raw_data():
    adapter = FakeAdapter(fail_capture_at=2)
    writer = RecordingArtifactWriter(adapter)
    plugin = AdvancedInputShaper(
        adapter=adapter,
        analyzer=analyzer,
        artifact_writer=writer,
        id_factory=lambda: "partial-capture",
    )

    with pytest.raises(RuntimeError, match="accelerometer disconnected"):
        plugin.calibrate(("X",), repeats=3, validate=True)

    assert writer.calls == []
    assert plugin.status()["artifacts"] is None


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
    adapter.verify_applied = lambda selections: None
    selections = (
        ShaperSelection("mzv", 74.4, "X", 0.08),
        ShaperSelection("2hump_ei", 76.4, "Y", 0.12),
    )

    adapter.apply_temporary(selections)
    adapter.stage(selections)

    scripts = config.printer.objects["gcode"].scripts
    assert scripts == [
        "SET_INPUT_SHAPER DAMPING_RATIO_X=0.08 DAMPING_RATIO_Y=0.12 "
        "SHAPER_FREQ_X=74.4 SHAPER_FREQ_Y=76.4 "
        "SHAPER_TYPE_X=mzv SHAPER_TYPE_Y=2hump_ei"
    ]
    assert config.printer.objects["configfile"].values == [
        ("input_shaper", "shaper_type_x", "mzv"),
        ("input_shaper", "shaper_freq_x", "74.4"),
        ("input_shaper", "damping_ratio_x", "0.08"),
        ("input_shaper", "shaper_type_y", "2hump_ei"),
        ("input_shaper", "shaper_freq_y", "76.4"),
        ("input_shaper", "damping_ratio_y", "0.12"),
    ]
    assert all("SAVE_CONFIG" not in script for script in scripts)


def test_stale_frequency_readback_blocks_validation_capture_and_restores_snapshot():
    shaping_status = {
        "shaper_type_x": SNAPSHOT.shaper_type_x,
        "shaper_freq_x": SNAPSHOT.shaper_freq_x,
        "damping_ratio_x": SNAPSHOT.damping_ratio_x,
        "shaper_type_y": SNAPSHOT.shaper_type_y,
        "shaper_freq_y": SNAPSHOT.shaper_freq_y,
        "damping_ratio_y": SNAPSHOT.damping_ratio_y,
    }
    velocity_status = {
        "max_velocity": SNAPSHOT.max_velocity,
        "max_accel": SNAPSHOT.max_accel,
        "square_corner_velocity": SNAPSHOT.square_corner_velocity,
    }

    class Reactor:
        @staticmethod
        def monotonic():
            return 1.0

    class InputShaper:
        @staticmethod
        def get_status(_eventtime):
            return dict(shaping_status)

    class Toolhead:
        @staticmethod
        def get_status(_eventtime):
            return dict(velocity_status)

    class GCode:
        def __init__(self):
            self.scripts = []

        def run_script_from_command(self, command):
            self.scripts.append(command)
            arguments = dict(
                token.split("=", 1) for token in command.split()[1:]
            )
            if command.startswith("SET_INPUT_SHAPER"):
                for axis in ("X", "Y"):
                    suffix = axis.lower()
                    shaper_type = arguments.get("SHAPER_TYPE_" + axis)
                    damping = arguments.get("DAMPING_RATIO_" + axis)
                    if shaper_type is not None:
                        shaping_status["shaper_type_" + suffix] = shaper_type
                    if damping is not None:
                        shaping_status["damping_ratio_" + suffix] = float(damping)
                    # Reproduce Klipper regression 77d5d94: the command accepts
                    # type and damping but silently retains the old frequency.
            elif command.startswith("SET_VELOCITY_LIMIT"):
                velocity_status["max_velocity"] = float(arguments["VELOCITY"])
                velocity_status["max_accel"] = float(arguments["ACCEL"])
                velocity_status["square_corner_velocity"] = float(
                    arguments["SQUARE_CORNER_VELOCITY"]
                )

    class Printer:
        def __init__(self, gcode):
            self.objects = {
                "gcode": gcode,
                "input_shaper": InputShaper(),
                "toolhead": Toolhead(),
            }

        @staticmethod
        def get_reactor():
            return Reactor()

        def lookup_object(self, name, default=None):
            return self.objects.get(name, default)

    gcode = GCode()
    adapter = KlipperPrinterAdapter.__new__(KlipperPrinterAdapter)
    adapter.gcode = gcode
    adapter.printer = Printer(gcode)
    adapter.capture_provider = None
    adapter._capture_native_shapers = None
    adapter._prove_selection = lambda _selection: {"passed": True}
    adapter.preflight = lambda _axes: None
    adapter.preflight_excitation = lambda _axes, _accel, _rate: {
        "min_frequency_hz": 5.0,
        "max_frequency_hz": 100.0,
        "hz_per_sec": 1.0,
    }
    captures = []

    def capture(axis, repeat, validation=False, **_kwargs):
        captures.append((axis, repeat, validation))
        return {"axis": axis, "repeat": repeat, "validation": validation}

    adapter.capture = capture
    plugin = AdvancedInputShaper(adapter=adapter, analyzer=analyzer)

    with pytest.raises(RuntimeError, match="X-axis frequency readback mismatch"):
        plugin.calibrate(("X",), repeats=1, validate=True)

    # Training and the held-out reference complete, but stale candidate
    # frequency readback prevents its validation motion from starting.
    assert captures == [("X", 0, False), ("X", 0, True)]
    assert shaping_status == {
        "shaper_type_x": SNAPSHOT.shaper_type_x,
        "shaper_freq_x": SNAPSHOT.shaper_freq_x,
        "damping_ratio_x": SNAPSHOT.damping_ratio_x,
        "shaper_type_y": SNAPSHOT.shaper_type_y,
        "shaper_freq_y": SNAPSHOT.shaper_freq_y,
        "damping_ratio_y": SNAPSHOT.damping_ratio_y,
    }
    assert velocity_status == {
        "max_velocity": SNAPSHOT.max_velocity,
        "max_accel": SNAPSHOT.max_accel,
        "square_corner_velocity": SNAPSHOT.square_corner_velocity,
    }
    assert gcode.scripts[-1].startswith("SET_VELOCITY_LIMIT")


def experimental_analyzer(**kwargs):
    if "validation_captures" in kwargs:
        return {
            "validation": {
                "passed": True,
                "axes": {
                    axis: {
                        "passed": True,
                        "qc_passed": True,
                        "improvement_ci_95": [0.12, 0.25],
                        "cross_axis_regression": 0.01,
                        "validation_evidence_kind": "finite_reversal_ringdown_v1",
                        "paired_window_fairness": {"passed": True},
                        "measured_spectral_non_regression": {"passed": True},
                    }
                    for axis in kwargs["axes"]
                },
            }
        }
    assert kwargs["experimental_mode"] is True
    candidate_id = (
        "mzv(n=4,t=0.800000)@frequency_hz=72.25,"
        "damping_ratio=0.040000000000000001"
    )
    return {
        "selections": [
            {
                "axis": axis,
                "shaper_type": "mzv(n=4,t=.8)",
                "candidate_id": candidate_id,
                "frequency_hz": 72.25,
                "damping_ratio": 0.04,
            }
            for axis in kwargs["axes"]
        ],
        "axes": {
            axis: {
                "selected": "mzv(n=4,t=0.800000)",
                "selected_candidate_id": candidate_id,
                "candidates": [
                    {
                        "name": "mzv",
                        "candidate_id": None,
                        "frequency": 70.0,
                        "max_accel": 15000.0,
                    },
                    {
                        "name": "mzv(n=4,t=0.800000)",
                        "candidate_id": candidate_id,
                        "frequency": 72.25,
                        "damping_ratio": 0.04,
                        "max_accel": 18000.0,
                        "metadata": {
                            "parameterized": True,
                            "design_damping_ratio": 0.04,
                        },
                    }
                ],
                "modes": [{"frequency": 72.0}],
                "damping_uncertainty_samples": [0.04, 0.06],
                "spectrum": {
                    "frequency_hz": [5.0, 10.0, 15.0, 20.0],
                    "psd": [1.0, 2.0, 2.0, 1.0],
                },
                "cross_spectrum": {
                    "frequency_hz": [5.0, 10.0, 15.0, 20.0],
                    "psd": [0.1, 0.2, 0.2, 0.1],
                },
            }
            for axis in kwargs["axes"]
        },
    }


RETRY_CANDIDATES = (
    {
        "name": "mzv(n=4,t=0.800000)",
        "candidate_id": (
            "mzv(n=4,t=0.800000)@frequency_hz=72.25,"
            "damping_ratio=0.040000000000000001"
        ),
        "frequency": 72.25,
        "damping_ratio": 0.04,
        "max_accel": 18000.0,
        "metadata": {
            "parameterized": True,
            "design_damping_ratio": 0.04,
        },
    },
    {
        "name": "mzv(n=5,t=0.900000)",
        "candidate_id": (
            "mzv(n=5,t=0.900000)@frequency_hz=70.5,"
            "damping_ratio=0.050000000000000003"
        ),
        "frequency": 70.5,
        "damping_ratio": 0.05,
        "max_accel": 17500.0,
        "metadata": {
            "parameterized": True,
            "design_damping_ratio": 0.05,
        },
    },
)

RETRY_NATIVE_CANDIDATE = {
    "name": "mzv",
    "candidate_id": None,
    "frequency": 70.0,
    "max_accel": 15000.0,
}


def retrying_experimental_analyzer(**kwargs):
    if "validation_captures" in kwargs:
        return experimental_analyzer(**kwargs)
    excluded = set(kwargs.get("excluded_candidate_ids", {}).get("X", []))
    selected = next(
        (item for item in RETRY_CANDIDATES if item["candidate_id"] not in excluded),
        None,
    )
    if selected is None:
        return {
            "abstain": True,
            "reason": "X has no parameterized candidate meeting the required uplift",
            "profile": "experimental_mzv",
        }
    return {
        "selections": [
            {
                "axis": axis,
                "shaper_type": selected["name"],
                "candidate_id": selected["candidate_id"],
                "frequency_hz": selected["frequency"],
                "damping_ratio": selected["damping_ratio"],
            }
            for axis in kwargs["axes"]
        ],
        "axes": {
            axis: {
                "selected": selected["name"],
                "selected_candidate_id": selected["candidate_id"],
                "candidates": [
                    dict(RETRY_NATIVE_CANDIDATE),
                    *(dict(item) for item in RETRY_CANDIDATES),
                ],
                "modes": [{"frequency": 72.0}],
                "damping_uncertainty_samples": [0.04, 0.06],
                "spectrum": {
                    "frequency_hz": [5.0, 10.0, 15.0, 20.0],
                    "psd": [1.0, 2.0, 2.0, 1.0],
                },
                "cross_spectrum": {
                    "frequency_hz": [5.0, 10.0, 15.0, 20.0],
                    "psd": [0.1, 0.2, 0.2, 0.1],
                },
            }
            for axis in kwargs["axes"]
        },
    }


class ExperimentalAdapter(FakeAdapter):
    def preflight_experimental(self, selections=(), max_vibrations=None):
        self.calls.append(("capability", tuple(selections)))
        return {
            "passed": True,
            "syntax": "mzv(n=4,t=0.800000)",
            "pulse_count": 4,
            "executor_pulse_limit": 10,
            "native_fitting": {
                "passed": True,
                "parameter": "max_vibrations",
                "fraction": max_vibrations,
                "percent": max_vibrations * 100.0,
            },
        }

    def build_shaper_models(self, selections):
        self.calls.append(("models", tuple(selections)))
        return {
            selection.axis: {
                "axis": selection.axis,
                "shaper_type": selection.shaper_type,
                "family": selection.shaper_type.split("(", 1)[0],
                "frequency_hz": selection.frequency,
                "design_damping_ratio": selection.damping_ratio,
                "pulse_count": 2,
                "pulse_amplitudes_normalized": [0.5, 0.5],
                "pulse_times_s": [0.0, 0.01],
                "source": "installed_klipper_shaper_defs.init_shaper",
                "source_module": "extras.shaper_defs",
                "source_file": "shaper_defs.py",
                "api_signature_verified": True,
                "executor_pulse_limit": 10,
                "theoretical_model_only": True,
                "live_c_executor_readback": False,
            }
            for selection in selections
        }


def test_second_stage_abstention_is_artifacted_without_candidate_motion():
    seen = {}

    def no_upgrade_analyzer(**kwargs):
        seen.update(kwargs)
        return {
            "abstain": True,
            "reason": "X has no parameterized candidate that improves the stock baseline",
            "profile": "experimental_mzv",
            "axes": {
                "X": {
                    "baseline_comparison": {
                        "status": "no_upgrade",
                        "physical_acceleration_claim": False,
                    }
                }
            },
        }

    adapter = ExperimentalAdapter()
    writer = RecordingArtifactWriter(adapter)
    plugin = AdvancedInputShaper(
        adapter=adapter,
        analyzer=no_upgrade_analyzer,
        artifact_writer=writer,
        experimental_enabled=True,
        id_factory=lambda: "no-upgrade",
    )

    with pytest.raises(RuntimeError, match="no parameterized candidate"):
        plugin.calibrate(
            ("X",), profile="experimental_mzv", repeats=3, validate=True
        )

    assert seen["reference_models"]["X"]["shaper_type"] == "mzv"
    assert len([call for call in adapter.calls if call[0] == "capture"]) == 3
    assert not [call for call in adapter.calls if call[0] in {"apply", "transient"}]
    assert writer.calls[0]["result_id"] == "no-upgrade"
    assert writer.calls[0]["report"]["status"] == "rejected"
    assert set(writer.calls[0]["raw_groups"]) == {"training"}
    assert adapter.calls[-1] == ("artifact", "no-upgrade")
    assert plugin.status()["attempt_status"] == "rejected"


def adaptive_native_analyzer(**kwargs):
    if "validation_captures" in kwargs:
        return {
            "validation": {
                "passed": True,
                "axes": {
                    axis: {
                        "passed": True,
                        "qc_passed": True,
                        "improvement_ci_95": [0.20, 0.40],
                        "cross_axis_regression": 0.0,
                        "validation_evidence_kind": "finite_reversal_ringdown_v1",
                        "paired_window_fairness": {"passed": True},
                        "measured_spectral_non_regression": {"passed": True},
                    }
                    for axis in kwargs["axes"]
                },
            }
        }
    return {
        "profile": "adaptive_stock",
        "selections": [
            {
                "axis": axis,
                "shaper_type": "zvd",
                "candidate_id": "zvd",
                "frequency_hz": 68.0,
                "damping_ratio": 0.07,
            }
            for axis in kwargs["axes"]
        ],
        "axes": {
            axis: {
                "selected": "zvd",
                "selected_candidate_id": "zvd",
                "candidates": [
                    {
                        "name": "zvd",
                        "candidate_id": "zvd",
                        "frequency": 68.0,
                        "max_accel": 16000.0,
                    }
                ],
                "modes": [{"frequency": 68.0}],
                "damping_uncertainty_samples": [0.05, 0.07],
                "spectrum": {
                    "frequency_hz": [5.0, 10.0, 15.0, 20.0],
                    "psd": [1.0, 2.0, 2.0, 1.0],
                },
                "cross_spectrum": {
                    "frequency_hz": [5.0, 10.0, 15.0, 20.0],
                    "psd": [0.1, 0.2, 0.2, 0.1],
                },
            }
            for axis in kwargs["axes"]
        },
        "runtime_contract": {
            "interface": "stock_set_input_shaper",
            "arbitrary_pulse_vectors": False,
        },
    }


def test_adaptive_stock_native_winner_keeps_capability_and_validation_gates():
    adapter = ExperimentalAdapter()
    plugin = AdvancedInputShaper(
        adapter=adapter,
        analyzer=adaptive_native_analyzer,
        experimental_enabled=True,
        id_factory=lambda: "adaptive-stock-result",
    )
    plugin.minimum_max_accel["X"] = 15000.0

    result = plugin.calibrate(
        ("X",), profile="adaptive_stock", repeats=3, validate=True
    )

    assert adapter.capture_profile == "adaptive_stock"
    assert adapter.calls.index(("capability", ())) < next(
        index for index, call in enumerate(adapter.calls) if call[0] == "capture"
    )
    assert result.selections == (ShaperSelection("zvd", 68.0, "X", 0.07),)
    assert result.report["runtime_capability"]["passed"] is True
    assert result.report["runtime_capability"]["native_fitting"]["fraction"] == 0.10
    assert result.report["validation_protocol"]["native_fit_max_vibrations"] == {
        "fraction": 0.10,
        "percent": 10.0,
        "source": "selection_profile.maximum_residual",
        "upstream_parameter": "max_vibrations",
    }
    training = [call for call in adapter.calls if call[0] == "capture"]
    transients = [call for call in adapter.calls if call[0] == "transient"]
    assert {call[6] for call in training} == {0.10}
    assert {call[7] for call in transients} == {"finite_reversal_ringdown"}
    assert result.report["validation"]["passed"] is True
    assert len(training) == 3
    assert len(transients) == 6
    plugin.apply("adaptive-stock-result")
    plugin.stage("adaptive-stock-result")
    assert adapter.calls[-1][0] == "stage"


def test_adaptive_stock_is_opt_in_and_never_allows_unvalidated_motion():
    adapter = ExperimentalAdapter()
    with pytest.raises(ValueError, match="enable_experimental"):
        AdvancedInputShaper(adapter=adapter, analyzer=adaptive_native_analyzer).calibrate(
            ("X",), profile="adaptive_stock", repeats=3, validate=True
        )
    with pytest.raises(ValueError, match="mandatory held-out validation"):
        AdvancedInputShaper(
            adapter=adapter,
            analyzer=adaptive_native_analyzer,
            experimental_enabled=True,
        ).calibrate(("X",), profile="adaptive_stock", repeats=3, validate=False)
    assert adapter.calls == []


def test_experimental_fast_validation_uses_one_sweep_and_four_transients():
    adapter = ExperimentalAdapter()
    plugin = AdvancedInputShaper(
        adapter=adapter,
        analyzer=experimental_analyzer,
        experimental_enabled=True,
        id_factory=lambda: "fast-generalized-result",
    )

    result = plugin.calibrate(
        ("X",),
        profile="experimental_mzv",
        repeats=2,
        validate=True,
        hz_per_sec="2",
        fast_validation=1,
    )

    training = [call for call in adapter.calls if call[0] == "capture"]
    transients = [call for call in adapter.calls if call[0] == "transient"]
    assert len(training) == 1
    assert len(transients) == 4
    assert {call[5] for call in training} == {2.0}
    assert {call[7] for call in transients} == {"finite_reversal_ringdown"}
    protocol = result.report["validation_protocol"]
    assert protocol["mode"] == "fast_lower_confidence_1_train_2_held_out"
    assert protocol["lower_confidence"] is True
    assert protocol["repeats_per_group"] == 2
    assert protocol["training_repeats"] == 1
    assert protocol["reference_repeats"] == 2
    assert protocol["candidate_repeats"] == 2
    assert protocol["full_sweeps_per_axis"] == 1
    assert protocol["paired_transients_per_axis"] == 4
    assert protocol["common_post_command_guard_seconds_by_axis"] == {"X": 0.03}
    assert {
        row["post_command_guard_seconds"] for row in protocol["capture_order"]
    } == {0.03}
    assert all(
        row["capture_kind"] == "finite_reversal_ringdown"
        for row in protocol["capture_order"]
    )
    assert plugin.status()["validation_protocol"] == protocol


def test_peak_lock_is_propagated_through_training_and_validation():
    calls = []

    def recording_analyzer(**kwargs):
        calls.append(kwargs)
        return experimental_analyzer(**kwargs)

    adapter = ExperimentalAdapter()
    plugin = AdvancedInputShaper(
        adapter=adapter,
        analyzer=recording_analyzer,
        experimental_enabled=True,
        id_factory=lambda: "peak-locked-result",
    )
    result = plugin.calibrate(
        ("X",),
        profile="experimental_mzv",
        repeats=2,
        validate=True,
        hz_per_sec="2",
        fast_validation=1,
        peak_lock=1,
    )
    assert [call["peak_lock"] for call in calls] == [True, True]
    assert result.report["validation_protocol"]["peak_lock"] is True
    assert result.report["validation_protocol"]["frequency_strategy"] == (
        "strongest_measured_peak"
    )


def test_peak_lock_is_rejected_outside_experimental_mode_before_motion():
    adapter = ExperimentalAdapter()
    plugin = AdvancedInputShaper(adapter=adapter, analyzer=experimental_analyzer)
    with pytest.raises(ValueError, match="PEAK_LOCK is only supported"):
        plugin.calibrate(("X",), profile="balanced", peak_lock=1)
    assert adapter.calls == []


@pytest.mark.parametrize(
    ("repeats", "validate", "hz_per_sec", "fast_validation", "message"),
    [
        (1, True, "2", 1, "exactly REPEATS=2"),
        (2, False, "2", 1, "mandatory held-out"),
        (2, True, "1", 1, "explicit HZ_PER_SEC=2"),
        (2, True, "2", 0, "at least three repeats"),
        (3, True, "2", 1, "exactly REPEATS=2"),
    ],
)
def test_experimental_fast_validation_cannot_weaken_required_protocol(
    repeats, validate, hz_per_sec, fast_validation, message
):
    adapter = ExperimentalAdapter()
    plugin = AdvancedInputShaper(
        adapter=adapter,
        analyzer=experimental_analyzer,
        experimental_enabled=True,
    )

    with pytest.raises(ValueError, match=message):
        plugin.calibrate(
            ("X",),
            profile="experimental_mzv",
            repeats=repeats,
            validate=validate,
            hz_per_sec=hz_per_sec,
            fast_validation=fast_validation,
        )
    assert adapter.calls == []


def test_experimental_profile_is_opt_in_and_requires_validation_before_motion():
    adapter = ExperimentalAdapter()
    plugin = AdvancedInputShaper(adapter=adapter, analyzer=experimental_analyzer)

    with pytest.raises(ValueError, match="enable_experimental"):
        plugin.calibrate(("X",), profile="experimental_mzv", repeats=3, validate=True)
    with pytest.raises(ValueError, match="mandatory held-out"):
        AdvancedInputShaper(
            adapter=adapter,
            analyzer=experimental_analyzer,
            experimental_enabled=True,
        ).calibrate(("X",), profile="experimental_mzv", repeats=3, validate=False)
    assert not [call for call in adapter.calls if call[0] == "capture"]


def test_experimental_transient_api_gap_abstains_before_snapshot_or_motion():
    adapter = ExperimentalAdapter()
    adapter.preflight_transient = None
    plugin = AdvancedInputShaper(
        adapter=adapter,
        analyzer=experimental_analyzer,
        experimental_enabled=True,
    )

    with pytest.raises(RuntimeError, match="finite transient validation support"):
        plugin.calibrate(
            ("X",), profile="experimental_mzv", repeats=3, validate=True
        )
    assert not [
        call
        for call in adapter.calls
        if call[0] in {"snapshot", "capture", "transient", "apply"}
    ]


def test_post_analysis_readiness_change_blocks_temporary_apply_and_transient_motion():
    class StateChangingAdapter(ExperimentalAdapter):
        def __init__(self):
            super().__init__()
            self.preflight_count = 0

        def preflight(self, axes):
            self.preflight_count += 1
            super().preflight(axes)
            if self.preflight_count == 2:
                raise RuntimeError("printer started printing during analysis")

    adapter = StateChangingAdapter()
    plugin = AdvancedInputShaper(
        adapter=adapter,
        analyzer=experimental_analyzer,
        experimental_enabled=True,
    )

    with pytest.raises(RuntimeError, match="started printing during analysis"):
        plugin.calibrate(
            ("X",), profile="experimental_mzv", repeats=3, validate=True
        )

    assert len([call for call in adapter.calls if call[0] == "capture"]) == 3
    assert not [
        call for call in adapter.calls if call[0] in {"apply", "transient"}
    ]
    assert adapter.calls[-1] == ("restore", SNAPSHOT)


def test_parameterized_candidate_runs_full_validation_and_preserves_identifier():
    adapter = ExperimentalAdapter()
    plugin = AdvancedInputShaper(
        adapter=adapter,
        analyzer=experimental_analyzer,
        experimental_enabled=True,
        id_factory=lambda: "generalized-result",
    )
    result = plugin.calibrate(
        ("X",),
        profile="experimental_mzv",
        repeats=3,
        validate=True,
        accel_per_hz="30.25",
    )

    assert adapter.calls.index(("capability", ())) < next(
        index for index, call in enumerate(adapter.calls) if call[0] == "capture"
    )
    assert result.selections[0].shaper_type == "mzv(n=4,t=0.800000)"
    assert result.report["runtime_capability"]["passed"] is True
    assert result.report["theoretical_spectral_non_regression"]["passed"] is True
    assert result.report["theoretical_spectral_non_regression"]["validation"] is False
    assert (
        result.report["theoretical_spectral_non_regression"][
            "held_out_validation_still_required"
        ]
        is True
    )
    assert result.report["runtime_capability"][
        "configured_reference_model_proofs"
    ][0]["shaper_type"] == "mzv"
    assert result.report["runtime_capability"][
        "selected_candidate_model_proofs"
    ][0]["shaper_type"] == "mzv(n=4,t=0.800000)"
    assert result.report["validation"]["passed"] is True
    training = [call for call in adapter.calls if call[0] == "capture"]
    transients = [call for call in adapter.calls if call[0] == "transient"]
    assert len(training) == 3
    assert len(transients) == 6
    assert {call[4] for call in training} == {30.25}
    assert {call[6] for call in training} == {0.10}
    assert {call[7] for call in transients} == {"finite_reversal_ringdown"}
    plugin.apply("generalized-result")
    plugin.stage("generalized-result")
    assert adapter.calls[-1][0] == "stage"


@pytest.mark.parametrize(
    "evidence_kind", [None, "native_compatibility_validation_sweep"]
)
def test_experimental_boundary_rejects_passed_looking_non_transient_evidence(
    evidence_kind,
):
    def wrong_evidence_analyzer(**kwargs):
        result = experimental_analyzer(**kwargs)
        if "validation_captures" in kwargs:
            axis_evidence = result["validation"]["axes"]["X"]
            if evidence_kind is None:
                axis_evidence.pop("validation_evidence_kind", None)
            else:
                axis_evidence["validation_evidence_kind"] = evidence_kind
        return result

    adapter = ExperimentalAdapter()
    plugin = AdvancedInputShaper(
        adapter=adapter,
        analyzer=wrong_evidence_analyzer,
        experimental_enabled=True,
    )

    with pytest.raises(RuntimeError, match="not finite-ringdown evidence"):
        plugin.calibrate(
            ("X",), profile="experimental_mzv", repeats=3, validate=True
        )

    assert plugin.results == {}
    assert adapter.calls[-1] == ("restore", SNAPSHOT)


def test_theoretical_screen_rejection_blocks_all_held_out_motion_and_apply():
    adapter = ExperimentalAdapter()
    seen = {}
    writer = RecordingArtifactWriter(adapter)

    def exhausting_analyzer(**kwargs):
        excluded = kwargs.get("excluded_candidate_ids", {}).get("X", [])
        if excluded:
            return {
                "abstain": True,
                "reason": "X has no parameterized candidate meeting the required uplift",
                "profile": "experimental_mzv",
            }
        return experimental_analyzer(**kwargs)

    def rejecting_screen(**kwargs):
        seen.update(kwargs)
        return {
            "passed": False,
            "validation": False,
            "held_out_validation_still_required": True,
            "evidence_level": "theoretical_preflight_screen",
            "reason": "X along_axis 125.0-130.0 Hz ratio 46.996 exceeds 1.100",
            "axes": {"X": {"passed": False}},
        }

    plugin = AdvancedInputShaper(
        adapter=adapter,
        analyzer=exhausting_analyzer,
        spectral_screener=rejecting_screen,
        artifact_writer=writer,
        experimental_enabled=True,
        id_factory=lambda: "screen-rejected",
    )

    with pytest.raises(RuntimeError, match="no parameterized candidate"):
        plugin.calibrate(
            ("X",), profile="experimental_mzv", repeats=3, validate=True
        )

    captures = [call for call in adapter.calls if call[0] == "capture"]
    assert len(captures) == 3
    assert all(call[3] is False for call in captures)
    first_model_index = next(
        index for index, call in enumerate(adapter.calls) if call[0] == "models"
    )
    first_capture_index = next(
        index for index, call in enumerate(adapter.calls) if call[0] == "capture"
    )
    assert first_model_index < first_capture_index
    assert not [call for call in adapter.calls if call[0] == "apply"]
    assert adapter.calls[-2:] == [
        ("restore", SNAPSHOT),
        ("artifact", "screen-rejected"),
    ]
    assert plugin.status()["attempt_status"] == "rejected"
    rejected = writer.calls[0]["report"]
    assert rejected["status"] == "rejected"
    assert len(rejected["theoretical_spectral_screen_attempts"]) == 1
    assert set(writer.calls[0]["raw_groups"]) == {"training"}
    assert seen["reference_models"]["X"]["shaper_type"] == "mzv"
    assert seen["candidate_models"]["X"]["shaper_type"] == (
        "mzv(n=4,t=0.800000)"
    )


def test_theoretical_screen_retries_offline_and_only_passing_candidate_moves():
    adapter = ExperimentalAdapter()

    def screen_first_candidate_only(**kwargs):
        selected = kwargs["candidate_models"]["X"]["shaper_type"]
        passed = selected == RETRY_CANDIDATES[1]["name"]
        return {
            "passed": passed,
            "validation": False,
            "held_out_validation_still_required": True,
            "evidence_level": "theoretical_preflight_screen",
            "reason": None if passed else "first candidate regresses a secondary band",
            "axes": {"X": {"passed": passed}},
        }

    plugin = AdvancedInputShaper(
        adapter=adapter,
        analyzer=retrying_experimental_analyzer,
        spectral_screener=screen_first_candidate_only,
        experimental_enabled=True,
        id_factory=lambda: "offline-retry-passed",
    )

    result = plugin.calibrate(
        ("X",), profile="experimental_mzv", repeats=3, validate=True
    )

    assert result.selections[0].shaper_type == RETRY_CANDIDATES[1]["name"]
    attempts = result.report["theoretical_spectral_screen_attempts"]
    assert [item["screen"]["passed"] for item in attempts] == [False, True]
    assert [item["candidates"][0]["candidate_id"] for item in attempts] == [
        RETRY_CANDIDATES[0]["candidate_id"],
        RETRY_CANDIDATES[1]["candidate_id"],
    ]
    assert all(item["candidates"][0]["model_proof"] for item in attempts)
    training = [call for call in adapter.calls if call[0] == "capture"]
    transients = [call for call in adapter.calls if call[0] == "transient"]
    assert len(training) == 3
    assert len(transients) == 6
    applied_parameterized = [
        selection.shaper_type
        for call in adapter.calls
        if call[0] == "apply"
        for selection in call[1]
        if selection.parameterized
    ]
    assert set(applied_parameterized) == {RETRY_CANDIDATES[1]["name"]}


def test_all_theoretical_screen_candidates_fail_with_one_training_set_and_artifact():
    adapter = ExperimentalAdapter()
    writer = RecordingArtifactWriter(adapter)

    def reject_every_screen(**kwargs):
        return {
            "passed": False,
            "validation": False,
            "held_out_validation_still_required": True,
            "evidence_level": "theoretical_preflight_screen",
            "reason": "candidate regresses a measured band",
            "axes": {axis: {"passed": False} for axis in kwargs["axes"]},
        }

    plugin = AdvancedInputShaper(
        adapter=adapter,
        analyzer=retrying_experimental_analyzer,
        spectral_screener=reject_every_screen,
        artifact_writer=writer,
        experimental_enabled=True,
        id_factory=lambda: "offline-retry-exhausted",
    )

    with pytest.raises(RuntimeError, match="no parameterized candidate"):
        plugin.calibrate(
            ("X",), profile="experimental_mzv", repeats=3, validate=True
        )

    assert len([call for call in adapter.calls if call[0] == "capture"]) == 3
    assert not [call for call in adapter.calls if call[0] in {"apply", "transient"}]
    rejected = writer.calls[0]
    assert set(rejected["raw_groups"]) == {"training"}
    attempts = rejected["report"]["theoretical_spectral_screen_attempts"]
    assert len(attempts) == 2
    assert all(item["screen"]["passed"] is False for item in attempts)
    assert rejected["report"]["excluded_candidate_ids"]["X"] == [
        RETRY_CANDIDATES[0]["candidate_id"],
        RETRY_CANDIDATES[1]["candidate_id"],
    ]


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("missing", "candidate ID is missing or malformed"),
        ("malformed", "parameterized candidate ID is malformed"),
        ("duplicate", "candidate IDs are missing or duplicated"),
    ],
)
def test_experimental_candidate_identity_defects_fail_closed(defect, message):
    def defective_analyzer(**kwargs):
        report = experimental_analyzer(**kwargs)
        if "validation_captures" in kwargs:
            return report
        if defect == "missing":
            report["selections"][0].pop("candidate_id")
        elif defect == "malformed":
            report["selections"][0]["candidate_id"] = "not-an-opaque-id"
            report["axes"]["X"]["selected_candidate_id"] = "not-an-opaque-id"
            report["axes"]["X"]["candidates"][1]["candidate_id"] = (
                "not-an-opaque-id"
            )
        else:
            report["axes"]["X"]["candidates"].append(
                dict(report["axes"]["X"]["candidates"][1])
            )
        return report

    adapter = ExperimentalAdapter()
    plugin = AdvancedInputShaper(
        adapter=adapter,
        analyzer=defective_analyzer,
        experimental_enabled=True,
    )

    with pytest.raises(RuntimeError, match=message):
        plugin.calibrate(
            ("X",), profile="experimental_mzv", repeats=3, validate=True
        )

    assert len([call for call in adapter.calls if call[0] == "capture"]) == 3
    assert not [call for call in adapter.calls if call[0] in {"apply", "transient"}]
    assert adapter.calls[-1] == ("restore", SNAPSHOT)


def test_parameterized_candidate_damping_mismatch_fails_before_model_or_motion():
    def mismatched_damping_analyzer(**kwargs):
        report = experimental_analyzer(**kwargs)
        if "validation_captures" not in kwargs:
            # The selected row and opaque ID prove the analyzed candidate at
            # damping 0.04, while the runtime selection requests 0.08.
            report["selections"][0]["damping_ratio"] = 0.08
        return report

    adapter = ExperimentalAdapter()
    plugin = AdvancedInputShaper(
        adapter=adapter,
        analyzer=mismatched_damping_analyzer,
        experimental_enabled=True,
    )

    with pytest.raises(
        RuntimeError,
        match="selected candidate ID does not exactly match selection",
    ):
        plugin.calibrate(
            ("X",), profile="experimental_mzv", repeats=3, validate=True
        )

    # The first model call proves the configured stock reference before the
    # training sweeps. No selected-candidate model, SET_INPUT_SHAPER, or
    # finite-transient validation is permitted after this identity mismatch.
    assert len([call for call in adapter.calls if call[0] == "models"]) == 1
    assert len([call for call in adapter.calls if call[0] == "capture"]) == 3
    assert not [call for call in adapter.calls if call[0] in {"apply", "transient"}]
    assert adapter.calls[-1] == ("restore", SNAPSHOT)


def test_mismatched_configured_reference_model_fails_before_motion_or_scv_change():
    class MismatchedReferenceAdapter(ExperimentalAdapter):
        def build_shaper_models(self, selections):
            models = super().build_shaper_models(selections)
            first = selections[0]
            models[first.axis]["frequency_hz"] = first.frequency + 1.0
            return models

    adapter = MismatchedReferenceAdapter()
    plugin = AdvancedInputShaper(
        adapter=adapter,
        analyzer=experimental_analyzer,
        experimental_enabled=True,
    )

    with pytest.raises(RuntimeError, match="reference model does not exactly match"):
        plugin.calibrate(
            ("X",),
            profile="experimental_mzv",
            repeats=3,
            validate=True,
            square_corner_velocity=15.0,
        )

    assert not [
        call for call in adapter.calls if call[0] in {"capture", "apply", "set_scv"}
    ]
    assert adapter.calls[-1] == ("restore", SNAPSHOT)


@pytest.mark.parametrize("profile", ["experimental_mzv", "adaptive_stock"])
def test_unsupported_installed_klipper_abstains_before_capture_or_snapshot(profile):
    class UnsupportedAdapter(ExperimentalAdapter):
        def preflight_experimental(self, selections=(), max_vibrations=None):
            del max_vibrations
            self.calls.append(("capability", tuple(selections)))
            raise RuntimeError("legacy shaper_defs has no parameterized parser")

    adapter = UnsupportedAdapter()
    plugin = AdvancedInputShaper(
        adapter=adapter,
        analyzer=experimental_analyzer,
        experimental_enabled=True,
    )
    with pytest.raises(RuntimeError, match="legacy shaper_defs"):
        plugin.calibrate(("X",), profile=profile, repeats=3, validate=True)
    assert not [call for call in adapter.calls if call[0] in {"snapshot", "capture", "apply"}]
