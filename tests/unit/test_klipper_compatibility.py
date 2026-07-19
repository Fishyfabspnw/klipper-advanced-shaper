import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np
import pytest

from klipper_advanced_shaper.analysis import analyze_calibration
from klipper_advanced_shaper.artifacts import ArtifactWriter
from klipper_advanced_shaper.klippy.adapter import KlipperPrinterAdapter
from klipper_advanced_shaper.klippy.capture import (
    _CaptureHelper,
    _native_candidate,
    _native_spectrum,
)
from klipper_advanced_shaper.klippy.worker import SupervisedWorker
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


def test_capture_helper_supports_v013_single_argument_callback():
    result = _CaptureHelper(_V013NativeHelper()).process_accelerometer_data(_Samples())
    assert result["native_data"] == "native-v013"
    assert np.asarray(result["samples"]).shape == (2, 4)


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
