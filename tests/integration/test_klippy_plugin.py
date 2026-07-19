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
    ):
        self.calls.append(
            ("capture", axis, repeat, validation, accel_per_hz, hz_per_sec)
        )
        capture_count = sum(call[0] == "capture" for call in self.calls)
        if capture_count == self.fail_capture_at:
            raise RuntimeError("accelerometer disconnected")
        if self.controller is not None and capture_count == 1:
            self.controller.cancel()
        return {"axis": axis, "repeat": repeat, "validation": validation}

    def apply_temporary(self, selections):
        self.calls.append(("apply", tuple(selections)))

    def set_test_square_corner_velocity(self, value):
        self.calls.append(("set_scv", float(value)))

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
    assert len([c for c in adapter.calls if c[0] == "apply"]) == 2
    assert adapter.calls[-1] == ("restore", SNAPSHOT)
    assert plugin.status()["state"] == "review"


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
    assert plugin.status() == {
        "state": "failed",
        "result_id": None,
        "attempt_id": "rejected-attempt",
        "attempt_status": "rejected",
        "artifacts": {"json": "/private/result.json", "raw": "/private/captures.npz"},
        "cancel_requested": False,
        "error": "candidate failed held-out validation: confidence interval below target",
        "experimental_generalized_mzv_enabled": False,
        "validation_protocol": {
            "mode": "native_validation",
            "lower_confidence": True,
            "repeats_per_group": 2,
            "validation_enabled": True,
                "full_sweeps_per_axis": 6,
                "motion_time_excludes_host_analysis_and_artifact_time": True,
                "square_corner_velocity_source": "printer_snapshot",
                "estimated_motion_seconds_per_axis": 570.0,
                "hz_per_sec": 1.0,
                "square_corner_velocity": 5.0,
            },
    }


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
        "SET_INPUT_SHAPER DAMPING_RATIO_X=0.080000 DAMPING_RATIO_Y=0.120000 "
        "SHAPER_FREQ_X=74.400000 SHAPER_FREQ_Y=76.400000 "
        "SHAPER_TYPE_X=mzv SHAPER_TYPE_Y=2hump_ei"
    ]
    assert config.printer.objects["configfile"].values == [
        ("input_shaper", "shaper_type_x", "mzv"),
        ("input_shaper", "shaper_freq_x", "74.400000"),
        ("input_shaper", "damping_ratio_x", "0.080000"),
        ("input_shaper", "shaper_type_y", "2hump_ei"),
        ("input_shaper", "shaper_freq_y", "76.400000"),
        ("input_shaper", "damping_ratio_y", "0.120000"),
    ]
    assert all("SAVE_CONFIG" not in script for script in scripts)


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
                    }
                    for axis in kwargs["axes"]
                },
            }
        }
    assert kwargs["experimental_mode"] is True
    return {
        "selections": [
            {
                "axis": axis,
                "shaper_type": "mzv(n=4,t=.8)",
                "frequency_hz": 72.25,
                "damping_ratio": 0.04,
            }
            for axis in kwargs["axes"]
        ],
        "axes": {
            axis: {
                "selected": "mzv(n=4,t=0.800000)",
                "candidates": [
                    {
                        "name": "mzv(n=4,t=0.800000)",
                        "max_accel": 18000.0,
                    }
                ],
            }
            for axis in kwargs["axes"]
        },
    }


class ExperimentalAdapter(FakeAdapter):
    def preflight_experimental(self, selections=()):
        self.calls.append(("capability", tuple(selections)))
        return {
            "passed": True,
            "syntax": "mzv(n=4,t=0.800000)",
            "pulse_count": 4,
            "executor_pulse_limit": 10,
        }


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
                "frequency_hz": 68.0,
                "damping_ratio": 0.07,
            }
            for axis in kwargs["axes"]
        ],
        "axes": {
            axis: {
                "selected": "zvd",
                "candidates": [{"name": "zvd", "max_accel": 16000.0}],
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

    result = plugin.calibrate(
        ("X",), profile="adaptive_stock", repeats=3, validate=True
    )

    assert adapter.capture_profile == "adaptive_stock"
    assert adapter.calls.index(("capability", ())) < next(
        index for index, call in enumerate(adapter.calls) if call[0] == "capture"
    )
    assert result.selections == (ShaperSelection("zvd", 68.0, "X", 0.07),)
    assert result.report["runtime_capability"]["passed"] is True
    assert result.report["validation"]["passed"] is True
    assert len([call for call in adapter.calls if call[0] == "capture"]) == 9
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


def test_experimental_fast_validation_keeps_two_held_out_pairs_in_five_sweeps():
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

    captures = [call for call in adapter.calls if call[0] == "capture"]
    assert len(captures) == 5
    assert sum(not call[3] for call in captures) == 1
    assert sum(call[3] for call in captures) == 4
    assert {call[5] for call in captures} == {2.0}
    protocol = result.report["validation_protocol"]
    assert protocol["mode"] == "fast_lower_confidence_1_train_2_held_out"
    assert protocol["lower_confidence"] is True
    assert protocol["repeats_per_group"] == 2
    assert protocol["training_repeats"] == 1
    assert protocol["reference_repeats"] == 2
    assert protocol["candidate_repeats"] == 2
    assert protocol["full_sweeps_per_axis"] == 5
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
    assert result.report["validation"]["passed"] is True
    captures = [call for call in adapter.calls if call[0] == "capture"]
    assert len(captures) == 9
    assert {call[4] for call in captures} == {30.25}
    plugin.apply("generalized-result")
    plugin.stage("generalized-result")
    assert adapter.calls[-1][0] == "stage"


@pytest.mark.parametrize("profile", ["experimental_mzv", "adaptive_stock"])
def test_unsupported_installed_klipper_abstains_before_capture_or_snapshot(profile):
    class UnsupportedAdapter(ExperimentalAdapter):
        def preflight_experimental(self, selections=()):
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
