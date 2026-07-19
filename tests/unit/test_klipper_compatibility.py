import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np
import pytest

from klipper_advanced_shaper.analysis import analyze_calibration
from klipper_advanced_shaper.analysis.experimental import (
    generalized_mzv_pulses,
    prove_runtime_generalized_mzv,
)
from klipper_advanced_shaper.artifacts import ArtifactWriter
from klipper_advanced_shaper.klippy.adapter import KlipperPrinterAdapter, ShaperSelection
from klipper_advanced_shaper.klippy.capture import (
    NativeResonanceCaptureProvider,
    _CaptureHelper,
    _native_candidate,
    _native_spectrum,
)
from klipper_advanced_shaper.klippy.worker import SupervisedWorker
from klipper_advanced_shaper.shapers import parse_shaper_identifier
from klipper_advanced_shaper.worker_child import (
    diagnostic_failure,
    diagnostic_numpy_payload,
    diagnostic_sleep,
    diagnostic_sum,
)


class _Samples:
    def get_samples(self):
        return [[0.0, 1.0, 2.0, 3.0], [0.1, 2.0, 3.0, 4.0]]


class _V013NativeHelper:
    def process_accelerometer_data(self, data):
        assert isinstance(data, _Samples)
        return "native-v013"


def test_native_excitation_preflight_resolves_config_and_printer_motion_budget():
    pulse = type(
        "Pulse",
        (),
        {
            "accel_per_hz": 60.0,
            "hz_per_sec": 1.0,
            "min_freq": 5.0,
            "max_freq": 135.0,
        },
    )()
    generator = type(
        "Generator",
        (),
        {"vibration_generator": pulse, "sweeping_accel": 400.0},
    )()
    tester = type("Tester", (), {"generator": generator})()

    class Reactor:
        def monotonic(self):
            return 1.0

    class Toolhead:
        def __init__(self, max_accel):
            self.max_accel = max_accel

        def get_status(self, _eventtime):
            return {"max_accel": self.max_accel}

    class Printer:
        def __init__(self):
            self.toolhead = Toolhead(30_000.0)

        def get_reactor(self):
            return Reactor()

        def lookup_object(self, name):
            assert name == "toolhead"
            return self.toolhead

    provider = NativeResonanceCaptureProvider.__new__(NativeResonanceCaptureProvider)
    provider.printer = Printer()
    provider._tester = lambda: tester

    inherited = provider.preflight_excitation(("X", "Y"), None, None)
    assert inherited["source"] == "resonance_tester"
    assert inherited["accel_per_hz"] == 60.0
    assert inherited["estimated_peak_accel_mm_s2"] == 8500.0
    assert inherited["hz_per_sec"] == 1.0
    assert inherited["hz_per_sec_source"] == "resonance_tester"

    explicit = provider.preflight_excitation(("X", "Y"), 150.0, 2.0)
    assert explicit["source"] == "command"
    assert explicit["estimated_peak_accel_mm_s2"] == 20_650.0
    assert explicit["hz_per_sec"] == 2.0
    assert explicit["hz_per_sec_source"] == "command"

    provider.printer.toolhead.max_accel = 20_000.0
    with pytest.raises(RuntimeError, match="80% motion budget"):
        provider.preflight_excitation(("X", "Y"), 150.0, 2.0)


def test_native_capture_records_actual_explicit_sweep_rate(monkeypatch):
    class TestAxis:
        def __init__(self, axis):
            self.axis = axis

    class NativeData:
        freq_bins = np.array([5.0, 70.0, 135.0])
        psd_sum = np.array([1.0, 2.0, 1.0])
        psd_x = np.array([1.0, 2.0, 1.0])
        psd_y = np.array([0.0, 0.0, 0.0])
        psd_z = np.array([0.0, 0.0, 0.0])

        def normalize_to_frequencies(self):
            return None

    class NativeHelper:
        fitting_damping = None

        def __init__(self, _printer):
            pass

        def find_best_shaper(self, *_args, **_kwargs):
            NativeHelper.fitting_damping = _kwargs.get("damping_ratio")
            candidate = type(
                "Candidate",
                (),
                {
                    "name": "mzv",
                    "freq": 70.0,
                    "vibrs": 0.05,
                    "smoothing": 0.1,
                    "max_accel": 10_000.0,
                },
            )()
            return candidate, [candidate]

    pulse = type(
        "Pulse",
        (),
        {
            "accel_per_hz": 60.0,
            "hz_per_sec": 1.0,
            "min_freq": 5.0,
            "max_freq": 135.0,
        },
    )()

    class Tester:
        generator = type("Generator", (), {"vibration_generator": pulse})()
        max_smoothing = None
        accel_chips = [("x", type("Chip", (), {"name": "adxl345"})())]
        probe_points = [(100.0, 100.0, 20.0)]

        def _run_test(self, gcmd, axes, helper, name_suffix):
            del helper, name_suffix
            pulse.freq_start = pulse.min_freq
            pulse.freq_end = pulse.max_freq
            pulse.test_accel_per_hz = gcmd.get_float(
                "ACCEL_PER_HZ", pulse.accel_per_hz
            )
            pulse.test_hz_per_sec = gcmd.get_float(
                "HZ_PER_SEC", pulse.hz_per_sec, maxval=2.0
            )
            return {
                axes[0]: {
                    "native_data": NativeData(),
                    "samples": np.array(
                        [[0.0, 1.0, 0.0, 0.0], [0.1, 2.0, 0.0, 0.0]]
                    ),
                }
            }

        def _get_max_calibration_freq(self):
            return 202.5

    class Reactor:
        def monotonic(self):
            return 1.0

    class Toolhead:
        def get_status(self, _eventtime):
            return {"square_corner_velocity": 5.0}

    class Printer:
        def get_reactor(self):
            return Reactor()

        def lookup_object(self, name):
            assert name == "toolhead"
            return Toolhead()

        def get_start_args(self):
            return {"software_version": "test"}

    class GCode:
        @staticmethod
        def respond_info(_message):
            return None

    module_name = Tester.__module__

    def import_module(name):
        if name == module_name:
            return type("ResonanceModule", (), {"TestAxis": TestAxis})
        if name == "extras.shaper_calibrate":
            return type("NativeModule", (), {"ShaperCalibrate": NativeHelper})
        raise AssertionError(name)

    monkeypatch.setattr(
        "klipper_advanced_shaper.klippy.capture.importlib.import_module",
        import_module,
    )
    provider = NativeResonanceCaptureProvider.__new__(NativeResonanceCaptureProvider)
    provider.printer = Printer()
    provider.gcode = GCode()
    provider._tester = lambda: Tester()

    result = provider.capture(
        "X",
        0,
        validation=True,
        accel_per_hz=150.0,
        hz_per_sec=2.0,
        design_damping_ratio=0.08,
    )

    recipe = result["metadata"]["test_recipe"]
    assert recipe == {
        "freq_start": 5.0,
        "freq_end": 135.0,
        "accel_per_hz": 150.0,
        "hz_per_sec": 2.0,
    }
    assert NativeHelper.fitting_damping == 0.08
    assert result["metadata"]["native_design_damping_ratio"] == 0.08
    assert result["native_candidates"][0]["design_damping_ratio"] == 0.08


def test_capture_helper_supports_v013_single_argument_callback():
    result = _CaptureHelper(_V013NativeHelper()).process_accelerometer_data(_Samples())
    assert result["native_data"] == "native-v013"
    assert np.asarray(result["samples"]).shape == (2, 4)


def test_adapter_passes_active_axis_damping_into_native_fitting():
    class Reactor:
        def monotonic(self):
            return 1.0

    class InputShaper:
        def get_status(self, _eventtime):
            return {"damping_ratio_x": "0.043000"}

    class Printer:
        def get_reactor(self):
            return Reactor()

        def lookup_object(self, name, default=None):
            return InputShaper() if name == "input_shaper" else default

    class Provider:
        def capture(self, **kwargs):
            return kwargs

    adapter = KlipperPrinterAdapter.__new__(KlipperPrinterAdapter)
    adapter.printer = Printer()
    adapter.capture_provider = Provider()

    result = adapter.capture("X", 2, validation=True, accel_per_hz=75.0, hz_per_sec=2.0)

    assert result["design_damping_ratio"] == 0.043
    assert result["axis"] == "X"
    assert result["repeat"] == 2


def test_capture_result_combines_multiple_probe_points_without_timestamp_gap():
    helper = _CaptureHelper(_V013NativeHelper())
    first = helper.process_accelerometer_data(_Samples())
    second = helper.process_accelerometer_data(_Samples())
    first["native_data"] = type("Native", (), {"add_data": lambda self, other: None})()
    second["native_data"] = object()
    first.add_data(second)
    assert first["dataset_count"] == 2
    assert np.all(np.diff(first["samples"][:, 0]) > 0)


def test_native_calibration_components_preserve_fidelity_with_bounded_downsampling():
    size = 2501
    frequency = np.linspace(0.0, 250.0, size)

    class CalibrationData:
        freq_bins = frequency
        psd_x = frequency + 1.0
        psd_y = frequency + 2.0
        psd_z = frequency + 3.0
        psd_sum = psd_x + psd_y + psd_z

    result = _native_spectrum(CalibrationData())
    indices = np.linspace(0, size - 1, 1024, dtype=int)

    assert result["normalized"] is True
    assert result["available"] is True
    assert result["source_bins"] == size
    assert result["reported_bins"] == 1024
    assert result["frequency_hz"] == pytest.approx(frequency[indices])
    assert result["psd_x"] == pytest.approx((frequency + 1.0)[indices])
    assert np.asarray(result["psd_sum"]) == pytest.approx(
        np.asarray(result["psd_x"])
        + np.asarray(result["psd_y"])
        + np.asarray(result["psd_z"])
    )


def test_missing_native_display_components_are_optional():
    class IncompatibleCalibrationData:
        freq_bins = np.arange(4.0)
        psd_sum = np.ones(4)
        psd_x = np.ones(4)
        psd_y = np.ones(4)

    result = _native_spectrum(IncompatibleCalibrationData())
    assert result["available"] is False
    assert "missing display fields: psd_z" in result["reason"]

    class MalformedCalibrationData:
        freq_bins = np.arange(4.0)
        psd_sum = np.ones(4)
        psd_x = np.ones(4)
        psd_y = np.ones(4)
        psd_z = np.ones(3)

    malformed = _native_spectrum(MalformedCalibrationData())
    assert malformed["available"] is False
    assert "mismatched shapes" in malformed["reason"]


def test_current_native_candidate_uses_its_own_response_grid():
    calibration_frequency = np.linspace(5.0, 200.0, 1200)
    candidate_frequency = np.linspace(10.0, 140.0, 321)
    candidate = type(
        "Candidate",
        (),
        {
            "name": "mzv",
            "freq": 74.2,
            "vibrs": 0.04,
            "smoothing": 0.08,
            "max_accel": 17000.0,
            "freq_bins": candidate_frequency,
            "vals": np.exp(-candidate_frequency / 100.0),
        },
    )()
    result = _native_candidate(candidate, calibration_frequency, 150.0)
    response = result["native_frequency_response"]
    assert response["frequency_hz"] == pytest.approx(candidate_frequency)
    assert response["response_ratio"] == pytest.approx(candidate.vals)


def test_v013_native_candidate_uses_calibration_grid_filtered_to_max_frequency():
    calibration_frequency = np.linspace(5.0, 200.0, 1200)
    max_frequency = 150.0
    filtered = calibration_frequency[calibration_frequency <= max_frequency]
    candidate = type(
        "LegacyCandidate",
        (),
        {
            "name": "mzv",
            "freq": 74.2,
            "vibrs": 0.04,
            "smoothing": 0.08,
            "max_accel": 17000.0,
            "vals": np.exp(-filtered / 100.0),
        },
    )()
    result = _native_candidate(candidate, calibration_frequency, max_frequency)
    response = result["native_frequency_response"]
    assert response["frequency_hz"] == pytest.approx(filtered)
    assert response["response_ratio"] == pytest.approx(candidate.vals)

    candidate.vals = np.ones(3)
    assert "native_frequency_response" not in _native_candidate(
        candidate, calibration_frequency, max_frequency
    )
    assert "native_frequency_response" not in _native_candidate(candidate)


def test_snapshot_falls_back_to_v013_axis_params_and_restore_velocity():
    class Params:
        def __init__(self, kind, frequency, damping):
            self.values = {
                "shaper_type": kind,
                "shaper_freq": str(frequency),
                "damping_ratio": str(damping),
            }

        def get_status(self):
            return self.values

    class Axis:
        def __init__(self, name, params):
            self.axis = name
            self.params = params

    class InputShaper:
        def get_shapers(self):
            return [Axis("x", Params("mzv", 74.4, 0.08)), Axis("y", Params("ei", 48, 0.1))]

    class Toolhead:
        def get_status(self, _eventtime):
            return {
                "max_velocity": 900,
                "max_accel": 70000,
                "square_corner_velocity": 7,
                "minimum_cruise_ratio": 0.5,
            }

    class Reactor:
        def monotonic(self):
            return 1.0

    class GCode:
        def __init__(self):
            self.commands = []

        def run_script_from_command(self, command):
            self.commands.append(command)

    class Printer:
        def __init__(self):
            self.gcode = GCode()

        def lookup_object(self, name, default=None):
            return {
                "gcode": self.gcode,
                "toolhead": Toolhead(),
                "input_shaper": InputShaper(),
            }.get(name, default)

        def get_reactor(self):
            return Reactor()

    class Config:
        def __init__(self):
            self.printer = Printer()

        def get_printer(self):
            return self.printer

    config = Config()
    adapter = KlipperPrinterAdapter(config)
    snapshot = adapter.snapshot()
    assert snapshot.shaper_freq_x == 74.4
    assert snapshot.damping_ratio_x == 0.08
    adapter.restore(snapshot)
    assert config.printer.gcode.commands[-1].startswith("SET_VELOCITY_LIMIT VELOCITY=900")
    assert "ACCEL=70000" in config.printer.gcode.commands[-1]


def test_printer_legacy_shaper_defs_abstain_from_generalized_mzv():
    class LegacyPrinterShaperDefs:
        @staticmethod
        def get_mzv_shaper(frequency, damping):
            return [1.0, 1.0, 1.0], [0.0, 0.01, 0.02]

    proof = prove_runtime_generalized_mzv(LegacyPrinterShaperDefs)
    assert proof["passed"] is False
    assert "get_shaper_cfg" in proof["reason"]


def test_installed_executor_capacity_is_discovered_and_enforced(tmp_path):
    extras = tmp_path / "klippy" / "extras"
    chelper = tmp_path / "klippy" / "chelper"
    extras.mkdir(parents=True)
    chelper.mkdir()
    module_path = extras / "shaper_defs.py"
    module_path.write_text("# fixture\n", encoding="utf-8")
    (chelper / "kin_shaper.c").write_text(
        "struct shaper_pulses { double t; } pulses[5];\n", encoding="utf-8"
    )
    module = type("InstalledDefs", (), {"__file__": str(module_path)})
    adapter = KlipperPrinterAdapter.__new__(KlipperPrinterAdapter)
    adapter._shaper_defs_module = module
    adapter._executor_pulse_limit = None
    assert adapter._get_executor_pulse_limit() == 5

    with pytest.raises(RuntimeError, match="supports 5 pulses"):
        adapter._prove_selection(ShaperSelection("mzv(n=6,t=.8)", 70.0, "X", 0.04))


def test_applied_parameterized_status_requires_exact_axis_name_frequency_and_damping():
    class Params:
        def __init__(self, values):
            self.values = values

        def get_status(self):
            return self.values

    class Axis:
        def __init__(self, axis, values):
            self.axis = axis
            self.params = Params(values)

    class InputShaper:
        def __init__(self):
            self.axes = [
                Axis(
                    "x",
                    {
                        "shaper_type": "mzv(n=4,t=.8)",
                        "shaper_freq": "72.250",
                        "damping_ratio": "0.040000",
                    },
                ),
                Axis(
                    "y",
                    {
                        "shaper_type": "ei",
                        "shaper_freq": "50.000",
                        "damping_ratio": "0.080000",
                    },
                ),
            ]

        def get_shapers(self):
            return self.axes

    class Reactor:
        def monotonic(self):
            return 1.0

    class Printer:
        def __init__(self):
            self.input_shaper = InputShaper()

        def lookup_object(self, name, default=None):
            return self.input_shaper if name == "input_shaper" else default

        def get_reactor(self):
            return Reactor()

    class Config:
        def __init__(self):
            self.printer = Printer()

        def get_printer(self):
            return self.printer

    adapter = KlipperPrinterAdapter.__new__(KlipperPrinterAdapter)
    adapter.printer = Config().printer
    expected = ShaperSelection("mzv(n=4,t=.8)", 72.25, "X", 0.04)
    adapter.verify_applied((expected,))

    adapter.printer.input_shaper.axes[0].params.values["shaper_type"] = "mzv(n=5,t=.8)"
    with pytest.raises(RuntimeError, match="readback mismatch"):
        adapter.verify_applied((expected,))


def test_parameterized_snapshot_and_exact_restore_use_installed_capability():
    shaping = {
        "x": {
            "shaper_type": "mzv(n=4,tau=1.2)",
            "shaper_freq": "72.250",
            "damping_ratio": "0.040000",
        },
        "y": {
            "shaper_type": "ei",
            "shaper_freq": "50.000",
            "damping_ratio": "0.080000",
        },
    }
    velocity = {
        "max_velocity": 900.0,
        "max_accel": 70000.0,
        "square_corner_velocity": 7.0,
        "minimum_cruise_ratio": 0.5,
    }

    class Params:
        def __init__(self, axis):
            self.axis = axis

        def get_status(self):
            return shaping[self.axis]

    class Axis:
        def __init__(self, axis):
            self.axis = axis
            self.params = Params(axis)

    class InputShaper:
        def get_shapers(self):
            return [Axis("x"), Axis("y")]

    class Toolhead:
        def get_status(self, _eventtime):
            return dict(velocity)

    class Reactor:
        def monotonic(self):
            return 1.0

    class GCode:
        def __init__(self):
            self.commands = []

        def run_script_from_command(self, command):
            self.commands.append(command)
            values = dict(token.split("=", 1) for token in command.split()[1:])
            if command.startswith("SET_INPUT_SHAPER"):
                for axis in ("X", "Y"):
                    suffix = axis.lower()
                    if "SHAPER_TYPE_" + axis in values:
                        shaping[suffix]["shaper_type"] = values["SHAPER_TYPE_" + axis]
                        shaping[suffix]["shaper_freq"] = values["SHAPER_FREQ_" + axis]
                        shaping[suffix]["damping_ratio"] = values["DAMPING_RATIO_" + axis]
            elif command.startswith("SET_VELOCITY_LIMIT"):
                velocity["max_velocity"] = float(values["VELOCITY"])
                velocity["max_accel"] = float(values["ACCEL"])
                if "MINIMUM_CRUISE_RATIO" in values:
                    velocity["minimum_cruise_ratio"] = float(
                        values["MINIMUM_CRUISE_RATIO"]
                    )

    class Printer:
        def __init__(self):
            self.gcode = GCode()
            self.input_shaper = InputShaper()
            self.toolhead = Toolhead()

        def lookup_object(self, name, default=None):
            return {
                "gcode": self.gcode,
                "input_shaper": self.input_shaper,
                "toolhead": self.toolhead,
            }.get(name, default)

        def get_reactor(self):
            return Reactor()

    class Config:
        def __init__(self):
            self.printer = Printer()

        def get_printer(self):
            return self.printer

    class CurrentShaperDefs:
        @staticmethod
        def get_shaper_cfg(name):
            return object() if name.startswith("mzv(") else None

        @staticmethod
        def init_shaper(name, frequency, damping):
            identifier = parse_shaper_identifier(name)
            return generalized_mzv_pulses(
                int(identifier.argument_map()["n"]),
                identifier.mzv_spacing(),
                frequency,
                damping,
            )

    config = Config()
    adapter = KlipperPrinterAdapter(
        config, shaper_defs_module=CurrentShaperDefs, executor_pulse_limit=10
    )
    snapshot = adapter.snapshot()
    assert snapshot.shaper_type_x == "mzv(n=4,tau=1.200000)"

    shaping["x"].update(
        {"shaper_type": "mzv", "shaper_freq": "60.000", "damping_ratio": "0.100000"}
    )
    velocity.update({"max_velocity": 100.0, "max_accel": 1000.0})
    adapter.restore(snapshot)
    assert shaping["x"]["shaper_type"] == "mzv(n=4,tau=1.200000)"
    assert float(shaping["x"]["shaper_freq"]) == snapshot.shaper_freq_x
    assert float(shaping["x"]["damping_ratio"]) == snapshot.damping_ratio_x
    assert velocity["max_velocity"] == snapshot.max_velocity
    assert velocity["max_accel"] == snapshot.max_accel


def test_restore_attempts_velocity_when_shaper_restore_fails():
    class GCode:
        def __init__(self):
            self.commands = []

        def run_script_from_command(self, command):
            self.commands.append(command)
            if command.startswith("SET_INPUT_SHAPER"):
                raise RuntimeError("shaper restore failed")

    class Printer:
        def __init__(self):
            self.gcode = GCode()

        def lookup_object(self, name, default=None):
            return self.gcode if name == "gcode" else default

    class Config:
        def __init__(self):
            self.printer = Printer()

        def get_printer(self):
            return self.printer

    adapter = KlipperPrinterAdapter(Config())
    snapshot = type(
        "Snapshot",
        (),
        {
            "shaper_type_x": "mzv",
            "shaper_freq_x": 74.4,
            "damping_ratio_x": 0.1,
            "shaper_type_y": "ei",
            "shaper_freq_y": 48.0,
            "damping_ratio_y": 0.1,
            "max_velocity": 900.0,
            "max_accel": 70000.0,
            "minimum_cruise_ratio": 0.5,
        },
    )()
    with pytest.raises(RuntimeError, match="shaper restore failed"):
        adapter.restore(snapshot)
    assert adapter.gcode.commands[1].startswith("SET_VELOCITY_LIMIT")


def test_supervised_worker_runs_callable_out_of_process():
    class Reactor:
        def monotonic(self):
            return time.monotonic()

        def pause(self, until):
            time.sleep(max(0.0, min(0.02, until - time.monotonic())))

    result = SupervisedWorker(Reactor(), timeout=5, memory_mb=0, cpu_seconds=5).run(
        diagnostic_sum, {"values": [1, 2, 3]}, lambda: None
    )
    assert result == 6


def test_supervised_worker_handles_large_numpy_result_in_external_interpreter(tmp_path):

    class Reactor:
        def monotonic(self):
            return time.monotonic()

        def pause(self, until):
            time.sleep(max(0.0, min(0.02, until - time.monotonic())))

    size = 1024 * 1024
    result = SupervisedWorker(
        Reactor(), timeout=10, memory_mb=0, cpu_seconds=5, temporary_root=str(tmp_path)
    ).run(
        diagnostic_numpy_payload, {"size": size}, lambda: None
    )
    assert result["pid"] != os.getpid()
    assert result["payload"].nbytes == 8 * 1024 * 1024
    assert abs(result["mean"]) < 1e-6
    assert list(tmp_path.iterdir()) == []


def test_external_worker_callables_are_picklable(tmp_path):
    writer = ArtifactWriter(tmp_path)

    assert pickle.loads(pickle.dumps(analyze_calibration)) is analyze_calibration
    restored_write = pickle.loads(pickle.dumps(writer.write))
    assert restored_write.__self__.root == tmp_path
    assert restored_write.__self__.keep_raw is True


def test_supervised_worker_runs_artifact_writer_bound_method(tmp_path):
    class Reactor:
        def monotonic(self):
            return time.monotonic()

        def pause(self, until):
            time.sleep(max(0.0, min(0.02, until - time.monotonic())))

    temporary_root = tmp_path / "worker-temp"
    temporary_root.mkdir()
    writer = ArtifactWriter(tmp_path / "artifacts", keep_raw=False)
    artifacts = SupervisedWorker(
        Reactor(), timeout=15, memory_mb=0, cpu_seconds=10, temporary_root=str(temporary_root)
    ).run(
        writer.write,
        {"result_id": "diagnostic", "report": {"schema_version": "test", "axes": {}}},
        lambda: None,
    )

    assert temporary_root.exists()
    assert list(temporary_root.iterdir()) == []
    assert os.path.isfile(artifacts["json"])
    assert os.path.isfile(artifacts["html"])


def test_supervised_worker_returns_child_error_and_cleans_temp(tmp_path):
    class Reactor:
        def monotonic(self):
            return time.monotonic()

        def pause(self, until):
            time.sleep(max(0.0, min(0.02, until - time.monotonic())))

    worker = SupervisedWorker(
        Reactor(), timeout=5, memory_mb=0, cpu_seconds=5, temporary_root=str(tmp_path)
    )
    with pytest.raises(RuntimeError, match="intentional diagnostic failure"):
        worker.run(
            diagnostic_failure,
            {"message": "intentional diagnostic failure"},
            lambda: None,
        )
    assert list(tmp_path.iterdir()) == []


def test_supervised_worker_timeout_terminates_and_cleans_temp(tmp_path):
    class Reactor:
        def monotonic(self):
            return time.monotonic()

        def pause(self, until):
            time.sleep(max(0.0, min(0.02, until - time.monotonic())))

    worker = SupervisedWorker(
        Reactor(), timeout=0.1, memory_mb=0, cpu_seconds=5, temporary_root=str(tmp_path)
    )
    with pytest.raises(RuntimeError, match="timed out"):
        worker.run(diagnostic_sleep, {"seconds": 10.0}, lambda: None)
    assert list(tmp_path.iterdir()) == []


def test_supervised_worker_cancellation_terminates_and_cleans_temp(tmp_path):
    class Reactor:
        def monotonic(self):
            return time.monotonic()

        def pause(self, until):
            time.sleep(max(0.0, min(0.02, until - time.monotonic())))

    calls = 0

    def checkpoint():
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise RuntimeError("cancelled by test")

    worker = SupervisedWorker(
        Reactor(), timeout=5, memory_mb=0, cpu_seconds=5, temporary_root=str(tmp_path)
    )
    with pytest.raises(RuntimeError, match="cancelled by test"):
        worker.run(diagnostic_sleep, {"seconds": 10.0}, checkpoint)
    assert list(tmp_path.iterdir()) == []


def test_worker_child_direct_diagnostic_cli():
    completed = subprocess.run(
        [sys.executable, "-m", "klipper_advanced_shaper.worker_child", "--diagnostic"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert payload["boundary"] == "external-interpreter"
    assert payload["pid"] != os.getpid()
    assert payload["numpy_samples"] == 32768
    assert abs(payload["mean"]) < 1e-5
