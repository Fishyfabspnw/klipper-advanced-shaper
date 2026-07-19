import json
import os
import pickle
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from klipper_advanced_shaper.analysis import analyze_calibration
from klipper_advanced_shaper.analysis.experimental import (
    generalized_mzv_pulses,
    prove_runtime_generalized_mzv,
)
from klipper_advanced_shaper.artifacts import ArtifactWriter
from klipper_advanced_shaper.klippy.adapter import (
    KlipperPrinterAdapter,
    PrinterSnapshot,
    ShaperSelection,
)
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


@pytest.mark.parametrize("validation", [False, True])
def test_native_capture_records_actual_explicit_sweep_rate(monkeypatch, validation):
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
        fitting_max_vibrations = None

        def __init__(self, _printer):
            pass

        def find_best_shaper(
            self, *_args, max_vibrations=None, **_kwargs
        ):
            NativeHelper.fitting_damping = _kwargs.get("damping_ratio")
            NativeHelper.fitting_max_vibrations = max_vibrations
            assert _kwargs["shapers"] == (
                "zv",
                "mzv",
                "zvd",
                "ei",
                "2hump_ei",
                "3hump_ei",
            )
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
            timestamps = np.arange(4096, dtype=float) / 1000.0
            return {
                axes[0]: {
                    "native_data": NativeData(),
                    "samples": np.column_stack(
                        (
                            timestamps,
                            np.sin(2.0 * np.pi * 70.0 * timestamps),
                            0.1 * np.sin(2.0 * np.pi * 70.0 * timestamps),
                            0.05 * np.sin(2.0 * np.pi * 70.0 * timestamps),
                        )
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
        validation=validation,
        accel_per_hz=150.0,
        hz_per_sec=2.0,
        design_damping_ratio=0.08,
        native_shapers=("zv", "mzv", "zvd", "ei", "2hump_ei", "3hump_ei"),
        max_vibrations=0.10,
    )

    recipe = result["metadata"]["test_recipe"]
    assert recipe == {
        "freq_start": 5.0,
        "freq_end": 135.0,
        "accel_per_hz": 150.0,
        "hz_per_sec": 2.0,
    }
    assert result["metadata"]["native_design_damping_ratio"] == 0.08
    assert result["metadata"]["native_fit_max_vibrations"] == 0.10
    assert result["native_spectrum"]["available"] is True
    if not validation:
        assert NativeHelper.fitting_damping == 0.08
        assert NativeHelper.fitting_max_vibrations == 0.10
        assert result["native_candidates"][0]["design_damping_ratio"] == 0.08
        assert "native_fitting_performed" not in result["metadata"]
        assert "native_fitting_status" not in result["metadata"]
        return

    assert NativeHelper.fitting_damping is None
    assert NativeHelper.fitting_max_vibrations is None
    assert result["native_candidates"] == []
    assert result["metadata"]["native_fitting_performed"] is False
    assert (
        result["metadata"]["native_fitting_status"]
        == "skipped_held_out_validation"
    )

    # Validation analysis consumes the raw samples and QC metadata, not fitted
    # candidates. A skipped-fit capture remains a valid held-out input.
    validation_result = analyze_calibration(
        captures={"X": []},
        held_out_captures={"X": [result, result]},
        validation_captures={"X": [result, result]},
        validation_pair_ids={"X": ["X-01", "X-02"]},
        axes=("X",),
        profile="performance",
        snapshot=SimpleNamespace(square_corner_velocity=5.0, damping_ratio_x=0.08),
        prior_report={"axes": {"X": {"modes": [{"frequency": 70.0}]}}},
    )
    evidence = validation_result["validation"]["axes"]["X"]
    assert evidence["qc_passed"] is True
    assert evidence["pair_count"] == 2


def test_native_fitting_preflight_rejects_legacy_api_without_max_vibrations():
    class LegacyHelper:
        def find_best_shaper(self, calibration_data, damping_ratio=None):
            del calibration_data, damping_ratio

    provider = NativeResonanceCaptureProvider.__new__(NativeResonanceCaptureProvider)
    provider._native_fitting_method = lambda: LegacyHelper.find_best_shaper

    with pytest.raises(RuntimeError, match="lacks max_vibrations support"):
        provider.preflight_native_fitting(0.10)


@pytest.mark.parametrize("validation", [False, True])
def test_native_capture_proves_max_vibrations_support_before_resonance_motion(
    validation,
):
    class LegacyHelper:
        def find_best_shaper(self, calibration_data, damping_ratio=None):
            del calibration_data, damping_ratio

    provider = NativeResonanceCaptureProvider.__new__(NativeResonanceCaptureProvider)
    provider._native_fitting_method = lambda: LegacyHelper.find_best_shaper
    provider._tester = lambda: pytest.fail("resonance motion must not start")

    with pytest.raises(RuntimeError, match="lacks max_vibrations support"):
        provider.capture(
            "X",
            0,
            validation=validation,
            design_damping_ratio=0.08,
            max_vibrations=0.10,
        )


@pytest.mark.parametrize("value", [None, float("nan"), -0.01, 1.01, "ten-percent"])
def test_native_fitting_preflight_rejects_unsafe_thresholds(value):
    class ModernHelper:
        def find_best_shaper(
            self, calibration_data, max_vibrations=None
        ):
            del calibration_data, max_vibrations

    provider = NativeResonanceCaptureProvider.__new__(NativeResonanceCaptureProvider)
    provider._native_fitting_method = lambda: ModernHelper.find_best_shaper

    with pytest.raises(RuntimeError, match="max_vibrations threshold"):
        provider.preflight_native_fitting(value)


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
    adapter._capture_native_shapers = None

    result = adapter.capture("X", 2, validation=True, accel_per_hz=75.0, hz_per_sec=2.0)

    assert result["design_damping_ratio"] == 0.043
    assert result["axis"] == "X"
    assert result["repeat"] == 2
    assert result["native_shapers"] is None
    assert result["max_vibrations"] is None

    adapter.configure_capture_profile("adaptive_stock")
    adaptive = adapter.capture("X", 3, validation=True, max_vibrations=0.10)
    assert adaptive["native_shapers"] == (
        "zv",
        "mzv",
        "zvd",
        "ei",
        "2hump_ei",
        "3hump_ei",
    )
    assert adaptive["max_vibrations"] == 0.10


def test_adapter_builds_exact_normalized_model_from_installed_klipper_source():
    class Config:
        name = "mzv"

    class InstalledShaperDefs:
        __name__ = "extras.shaper_defs"
        __file__ = "/opt/klipper/klippy/extras/shaper_defs.py"

        @staticmethod
        def get_shaper_cfg(name):
            assert name == "mzv(n=4,t=0.800000)"
            return Config()

        @staticmethod
        def init_shaper(name, frequency, damping_ratio, error=None):
            del error
            assert name == "mzv(n=4,t=0.800000)"
            amplitudes, times = generalized_mzv_pulses(
                4, 0.8, frequency, damping_ratio
            )
            return amplitudes * 7.0, times

    adapter = KlipperPrinterAdapter.__new__(KlipperPrinterAdapter)
    adapter._shaper_defs_module = InstalledShaperDefs
    adapter._executor_pulse_limit = 10
    selection = ShaperSelection("mzv(n=4,t=.8)", 72.25, "X", 0.04)

    models = adapter.build_shaper_models((selection,))
    model = models["X"]

    assert model["shaper_type"] == "mzv(n=4,t=0.800000)"
    assert model["frequency_hz"] == 72.25
    assert model["design_damping_ratio"] == 0.04
    assert model["pulse_count"] == 4
    assert sum(model["pulse_amplitudes_normalized"]) == pytest.approx(1.0)
    assert model["source"] == "installed_klipper_shaper_defs.init_shaper"
    assert model["source_file"] == "shaper_defs.py"
    assert model["api_signature_verified"] is True
    assert model["theoretical_model_only"] is True
    assert model["live_c_executor_readback"] is False


def test_adapter_rejects_incompatible_or_oversized_installed_model_before_motion():
    class KeywordOnly:
        @staticmethod
        def get_shaper_cfg(*, name):
            del name

        @staticmethod
        def init_shaper(*, name, frequency, damping_ratio):
            del name, frequency, damping_ratio

    adapter = KlipperPrinterAdapter.__new__(KlipperPrinterAdapter)
    adapter._shaper_defs_module = KeywordOnly
    adapter._executor_pulse_limit = 10
    selection = ShaperSelection("mzv", 70.0, "X", 0.08)
    with pytest.raises(RuntimeError, match="model API is unavailable or incompatible"):
        adapter.build_shaper_models((selection,))

    class Config:
        name = "mzv"

    class Oversized:
        @staticmethod
        def get_shaper_cfg(_name):
            return Config()

        @staticmethod
        def init_shaper(_name, _frequency, _damping_ratio):
            return np.ones(11), np.arange(11, dtype=float) * 0.001

    adapter._shaper_defs_module = Oversized
    with pytest.raises(RuntimeError, match="unsafe pulse model"):
        adapter.build_shaper_models((selection,))


def test_capture_result_combines_multiple_probe_points_without_timestamp_gap():
    helper = _CaptureHelper(_V013NativeHelper())
    first = helper.process_accelerometer_data(_Samples())
    second = helper.process_accelerometer_data(_Samples())
    first["native_data"] = type("Native", (), {"add_data": lambda self, other: None})()
    second["native_data"] = object()
    first.add_data(second)
    assert first["dataset_count"] == 2
    assert np.all(np.diff(first["samples"][:, 0]) > 0)


def _transient_provider_fixture(*, max_accel=5000.0, valid_samples=True):
    class Client:
        def __init__(self):
            self.finished = 0

        def finish_measurements(self):
            self.finished += 1

        def has_valid_samples(self):
            return valid_samples

        def get_samples(self):
            timestamps = np.arange(0.0, 1.301, 0.001)
            decay = np.exp(-8.0 * np.maximum(timestamps - 0.55, 0.0))
            return np.column_stack(
                (
                    timestamps,
                    decay * np.sin(2.0 * np.pi * 70.0 * timestamps),
                    0.1 * decay * np.sin(2.0 * np.pi * 70.0 * timestamps),
                    np.zeros_like(timestamps),
                )
            )

    class Chip:
        name = "adxl345"

        def __init__(self):
            self.clients = []

        def start_internal_client(self):
            client = Client()
            self.clients.append(client)
            return client

    class Kinematics:
        def get_status(self, _eventtime):
            return {
                "axis_minimum": (0.0, 0.0, 0.0),
                "axis_maximum": (100.0, 100.0, 100.0),
                "homed_axes": "xyz",
            }

    class Toolhead:
        def __init__(self):
            self.position = [50.0, 50.0, 10.0, 0.0]
            self.print_time = 0.0
            self.moves = []

        def get_position(self):
            return list(self.position)

        def get_last_move_time(self):
            return self.print_time

        def manual_move(self, coordinate, speed):
            self.moves.append((list(coordinate), speed))
            for index, value in enumerate(coordinate):
                if value is not None:
                    self.position[index] = value
            self.print_time += 0.1

        def wait_moves(self):
            return None

        def dwell(self, seconds):
            self.print_time += seconds

        def get_status(self, _eventtime):
            return {
                "max_velocity": 300.0,
                "max_accel": max_accel,
            }

        def get_kinematics(self):
            return Kinematics()

    class Reactor:
        def monotonic(self):
            return 1.0

    chip = Chip()
    toolhead = Toolhead()
    tester = type("Tester", (), {"accel_chips": [("xy", chip)]})()

    class Printer:
        def lookup_object(self, name, default=None):
            return toolhead if name == "toolhead" else default

        def get_reactor(self):
            return Reactor()

    provider = NativeResonanceCaptureProvider.__new__(NativeResonanceCaptureProvider)
    provider.printer = Printer()
    provider._tester = lambda: tester
    return provider, toolhead, chip


def test_finite_reversal_transient_is_bounded_and_returns_post_command_raw_window():
    provider, toolhead, chip = _transient_provider_fixture()
    proof = provider.preflight_transient(("X",))
    plan = proof["plans"]["X"]

    assert proof["protocol"] == "finite_reversal_ringdown_v1"
    assert proof["max_accel_mm_s2"] == 5000.0
    assert plan["start_position_mm"] == 46.0
    assert plan["reversal_position_mm"] == 54.0
    assert toolhead.moves == []

    result = provider.capture_transient(
        "X",
        0,
        plan,
        proof["max_accel_mm_s2"],
        proof["speed_mm_s"],
    )

    assert [move[0] for move in toolhead.moves] == [
        [46.0, None],
        [54.0, None],
        [46.0, None],
        [50.0, None],
    ]
    assert chip.clients[0].finished == 1
    assert result["metadata"]["promotion_eligible"] is True
    assert result["metadata"]["validation_capture_kind"] == (
        "finite_reversal_ringdown"
    )
    assert result["metadata"]["ringdown_duration_seconds"] >= 0.50
    assert result["metadata"]["clip_limit"] == pytest.approx(
        16.0 * 9.80665 * 1000.0
    )
    assert result["metadata"]["ringdown_start_offset_seconds"] <= result[
        "metadata"
    ]["ringdown_boundary_tolerance_seconds"]
    assert result["metadata"]["ringdown_end_offset_seconds"] <= result[
        "metadata"
    ]["ringdown_boundary_tolerance_seconds"]
    samples = np.asarray(result["samples"])
    assert samples[0, 0] >= result["metadata"]["command_end_time"]
    assert samples.shape[1] == 4
    assert result["metadata"]["cross_axis_channels_retained"] is True
    assert result["metadata"]["position_restored"] is True
    assert result["metadata"]["pre_capture_axis_position_mm"] == 50.0
    assert result["metadata"]["post_capture_axis_position_mm"] == 50.0


def test_transient_preflight_rejects_unsupported_sensor_before_motion():
    provider, toolhead, _chip = _transient_provider_fixture()
    provider._tester = lambda: type(
        "Tester",
        (),
        {"accel_chips": [("x", type("UnsupportedChip", (), {"name": "bad"})())]},
    )()

    with pytest.raises(RuntimeError, match="start_internal_client API"):
        provider.preflight_transient(("X",))
    assert toolhead.moves == []


def test_transient_preflight_rejects_unknown_full_scale_before_motion():
    provider, toolhead, chip = _transient_provider_fixture()
    chip.name = "mystery_accelerometer"

    with pytest.raises(RuntimeError, match="full-scale range cannot be proven"):
        provider.preflight_transient(("X",))
    assert toolhead.moves == []


def test_transient_rejects_timestamp_window_missing_requested_end():
    provider, _toolhead, chip = _transient_provider_fixture()
    proof = provider.preflight_transient(("X",))
    original_start = chip.start_internal_client

    def truncated_client():
        client = original_start()
        original_samples = client.get_samples
        client.get_samples = lambda: original_samples()[
            original_samples()[:, 0] <= 1.25
        ]
        return client

    chip.start_internal_client = truncated_client
    with pytest.raises(RuntimeError, match="ring-down window is incomplete"):
        provider.capture_transient(
            "X",
            0,
            proof["plans"]["X"],
            proof["max_accel_mm_s2"],
            proof["speed_mm_s"],
        )


def test_real_transient_capture_metadata_passes_end_to_end_facade_qc():
    provider, _toolhead, _chip = _transient_provider_fixture()
    proof = provider.preflight_transient(("X",))
    captured = provider.capture_transient(
        "X",
        0,
        proof["plans"]["X"],
        proof["max_accel_mm_s2"],
        proof["speed_mm_s"],
    )

    references = []
    candidates = []
    for scale in (1.0, 1.01):
        reference = dict(captured)
        reference["metadata"] = dict(captured["metadata"])
        reference["samples"] = np.asarray(captured["samples"]).copy()
        reference["samples"][:, 1:] *= scale
        candidate = dict(captured)
        candidate["metadata"] = dict(captured["metadata"])
        candidate["samples"] = np.asarray(captured["samples"]).copy()
        candidate["samples"][:, 1:] *= 0.7 * scale
        references.append(reference)
        candidates.append(candidate)

    result = analyze_calibration(
        captures={"X": references},
        held_out_captures={"X": references},
        validation_captures={"X": candidates},
        validation_pair_ids={"X": ["X-01", "X-02"]},
        axes=("X",),
        profile="experimental_mzv",
        snapshot=SimpleNamespace(square_corner_velocity=7.0, damping_ratio_x=0.08),
        prior_report={"axes": {"X": {"modes": [{"frequency": 70.0}]}}},
        experimental_mode=True,
    )

    evidence = result["validation"]["axes"]["X"]
    assert result["validation"]["passed"] is True
    assert evidence["qc_passed"] is True
    assert evidence["paired_window_fairness"]["passed"] is True
    assert evidence["measured_spectral_non_regression"]["passed"] is True


def test_transient_acceleration_readback_mismatch_abstains_before_sensor_or_motion():
    provider, toolhead, chip = _transient_provider_fixture(max_accel=4000.0)
    proof = provider.preflight_transient(("X",))

    with pytest.raises(RuntimeError, match="max_accel readback mismatch"):
        provider.capture_transient(
            "X", 0, proof["plans"]["X"], 3999.0, proof["speed_mm_s"]
        )
    assert chip.clients == []
    assert toolhead.moves == []


def test_transient_invalid_samples_finish_sensor_before_abstaining():
    provider, toolhead, chip = _transient_provider_fixture(valid_samples=False)
    proof = provider.preflight_transient(("X",))

    with pytest.raises(RuntimeError, match="measured no valid data"):
        provider.capture_transient(
            "X",
            0,
            proof["plans"]["X"],
            proof["max_accel_mm_s2"],
            proof["speed_mm_s"],
        )
    assert chip.clients[0].finished == 1
    assert len(toolhead.moves) == 4


def test_transient_motion_failure_finishes_sensor_without_commanding_return_move():
    provider, toolhead, chip = _transient_provider_fixture()
    proof = provider.preflight_transient(("X",))
    calls = []

    def fail_first_move(coordinate, speed):
        calls.append((coordinate, speed))
        raise RuntimeError("MCU move rejected")

    toolhead.manual_move = fail_first_move
    with pytest.raises(RuntimeError, match="MCU move rejected"):
        provider.capture_transient(
            "X",
            0,
            proof["plans"]["X"],
            proof["max_accel_mm_s2"],
            proof["speed_mm_s"],
        )
    assert len(calls) == 1
    assert chip.clients[0].finished == 1


def test_transient_failure_after_approach_attempts_exact_position_recovery():
    provider, toolhead, chip = _transient_provider_fixture()
    proof = provider.preflight_transient(("X",))
    original_move = toolhead.manual_move
    calls = []

    def fail_reversal(coordinate, speed):
        calls.append((list(coordinate), speed))
        if len(calls) == 2:
            raise RuntimeError("reversal move rejected")
        original_move(coordinate, speed)

    toolhead.manual_move = fail_reversal
    with pytest.raises(RuntimeError, match="reversal move rejected"):
        provider.capture_transient(
            "X",
            0,
            proof["plans"]["X"],
            proof["max_accel_mm_s2"],
            proof["speed_mm_s"],
        )

    assert [call[0] for call in calls] == [
        [46.0, None],
        [54.0, None],
        [50.0, None],
    ]
    assert toolhead.position[0] == 50.0
    assert chip.clients[0].finished == 1


def test_transient_sensor_finalize_failure_still_restores_original_position():
    provider, toolhead, chip = _transient_provider_fixture()
    proof = provider.preflight_transient(("X",))
    original_start = chip.start_internal_client

    def failing_client():
        client = original_start()

        def fail_finish():
            client.finished += 1
            raise RuntimeError("sensor finalization failed")

        client.finish_measurements = fail_finish
        return client

    chip.start_internal_client = failing_client
    with pytest.raises(RuntimeError, match="sensor finalization failed"):
        provider.capture_transient(
            "X",
            0,
            proof["plans"]["X"],
            proof["max_accel_mm_s2"],
            proof["speed_mm_s"],
        )
    assert toolhead.position[0] == 50.0
    assert [move[0] for move in toolhead.moves][-1] == [50.0, None]


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
            self.shaper_type = kind
            self.shaper_freq = frequency
            self.damping_ratio = damping
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
            self.n = 2
            self.A = [0.5, 0.5]
            self.T = [0.0, 0.01]
            self.saved = None

        def is_enabled(self):
            return True

    class InputShaper:
        input_shaper_stepper_kinematics = [object()]

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

    class ShaperDefs:
        @staticmethod
        def init_shaper(name, frequency, damping):
            del name, frequency, damping
            return [0.5, 0.5], [0.0, 0.01]

    config = Config()
    adapter = KlipperPrinterAdapter(config, shaper_defs_module=ShaperDefs)
    snapshot = adapter.snapshot()
    assert snapshot.shaper_freq_x == 74.4
    assert snapshot.damping_ratio_x == 0.08
    adapter.restore(snapshot)
    assert config.printer.gcode.commands[-1].startswith("SET_VELOCITY_LIMIT VELOCITY=900")
    assert "ACCEL=70000" in config.printer.gcode.commands[-1]
    assert "SQUARE_CORNER_VELOCITY=7" in config.printer.gcode.commands[-1]


def test_restore_preserves_non_six_decimal_velocity_snapshot_exactly():
    values = {
        "max_velocity": 300.0000004,
        "max_accel": 5000.0000004,
        "square_corner_velocity": 7.0000004,
        "minimum_cruise_ratio": 0.5000004,
    }

    class GCode:
        commands = []

        @classmethod
        def run_script_from_command(cls, command):
            cls.commands.append(command)
            arguments = dict(token.split("=", 1) for token in command.split()[1:])
            values.update(
                {
                    "max_velocity": float(arguments["VELOCITY"]),
                    "max_accel": float(arguments["ACCEL"]),
                    "square_corner_velocity": float(
                        arguments["SQUARE_CORNER_VELOCITY"]
                    ),
                    "minimum_cruise_ratio": float(arguments["MINIMUM_CRUISE_RATIO"]),
                }
            )

    class Toolhead:
        @staticmethod
        def get_status(_eventtime):
            return dict(values)

    class Reactor:
        @staticmethod
        def monotonic():
            return 1.0

    class Printer:
        @staticmethod
        def get_reactor():
            return Reactor()

        @staticmethod
        def lookup_object(name, default=None):
            return Toolhead() if name == "toolhead" else default

    snapshot = PrinterSnapshot(
        shaper_type_x="mzv",
        shaper_freq_x=75.6000004,
        shaper_type_y="ei",
        shaper_freq_y=48.2000004,
        max_velocity=values["max_velocity"],
        max_accel=values["max_accel"],
        square_corner_velocity=values["square_corner_velocity"],
        damping_ratio_x=0.0380004,
        damping_ratio_y=0.0810004,
        minimum_cruise_ratio=values["minimum_cruise_ratio"],
    )
    adapter = KlipperPrinterAdapter.__new__(KlipperPrinterAdapter)
    adapter.gcode = GCode()
    adapter.printer = Printer()
    adapter.apply_temporary = lambda selections: None

    adapter.restore(snapshot)

    assert values["max_velocity"] == snapshot.max_velocity
    assert values["max_accel"] == snapshot.max_accel
    assert values["square_corner_velocity"] == snapshot.square_corner_velocity
    assert values["minimum_cruise_ratio"] == snapshot.minimum_cruise_ratio
    assert "SQUARE_CORNER_VELOCITY=7.0000004" in GCode.commands[-1]


def test_printer_legacy_shaper_defs_abstain_from_generalized_mzv():
    class LegacyPrinterShaperDefs:
        @staticmethod
        def get_mzv_shaper(frequency, damping):
            return [1.0, 1.0, 1.0], [0.0, 0.01, 0.02]

    proof = prove_runtime_generalized_mzv(LegacyPrinterShaperDefs)
    assert proof["passed"] is False
    assert "get_shaper_cfg" in proof["reason"]


def test_experimental_preflight_rejects_rounded_only_shaper_status():
    adapter = KlipperPrinterAdapter.__new__(KlipperPrinterAdapter)
    adapter._load_shaper_defs = lambda: object()
    adapter._get_executor_pulse_limit = lambda: 10
    adapter._shaping_status = lambda: {
        "shaper_type_x": "mzv",
        "shaper_freq_x": "75.600",
        "damping_ratio_x": "0.038000",
        "shaper_type_y": "ei",
        "shaper_freq_y": "48.200",
        "damping_ratio_y": "0.081000",
        "_advanced_shaper_raw_params": False,
    }

    with pytest.raises(RuntimeError, match="raw input-shaper parameters"):
        adapter.preflight_experimental()


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


def test_live_python_pulse_proof_matches_enabled_axis_to_installed_source():
    class Params:
        shaper_type = "mzv(n=4,t=0.800000)"
        shaper_freq = 72.25
        damping_ratio = 0.04

    class Axis:
        axis = "x"
        params = Params()
        n = 4
        A = [0.1, 0.2, 0.3, 0.4]
        T = [0.0, 0.01, 0.02, 0.03]
        saved = None

        @staticmethod
        def is_enabled():
            return True

    class InputShaper:
        input_shaper_stepper_kinematics = [object()]

        @staticmethod
        def get_shapers():
            return [Axis()]

    class Printer:
        @staticmethod
        def lookup_object(name, default=None):
            return InputShaper() if name == "input_shaper" else default

    class ShaperDefs:
        __name__ = "extras.shaper_defs"

        @staticmethod
        def init_shaper(name, frequency, damping):
            assert name == "mzv(n=4,t=0.800000)"
            assert frequency == 72.25
            assert damping == 0.04
            return [0.1, 0.2, 0.3, 0.4], [0.0, 0.01, 0.02, 0.03]

    adapter = KlipperPrinterAdapter.__new__(KlipperPrinterAdapter)
    adapter.printer = Printer()
    adapter._shaper_defs_module = ShaperDefs
    adapter.last_live_python_pulse_proof = None
    expected = ShaperSelection("mzv(n=4,t=.8)", 72.25, "X", 0.04)
    assert expected.frequency == 72.25
    assert expected.damping_ratio == 0.04

    proof = adapter.verify_live_python_pulses((expected,))

    assert proof["passed"] is True
    assert proof["active_axis_verified"] is True
    assert proof["live_c_executor_readback"] is False
    assert proof["axes"]["X"]["pulse_count"] == 4
    assert proof["axes"]["X"]["post_command_guard_seconds"] == pytest.approx(
        0.05
    )
    InputShaper.input_shaper_stepper_kinematics = []
    with pytest.raises(RuntimeError, match="no active stepper-kinematics wrappers"):
        adapter.verify_live_python_pulses((expected,))


@pytest.mark.parametrize(
    ("enabled", "amplitudes", "message"),
    [
        (False, [0.1, 0.2, 0.3, 0.4], "not actively enabled"),
        (True, [0.2, 0.2, 0.2, 0.4], "pulse amplitude mismatch"),
    ],
)
def test_live_python_pulse_proof_rejects_disabled_or_mismatched_axis(
    enabled, amplitudes, message
):
    class Params:
        shaper_type = "mzv"
        shaper_freq = 70.0
        damping_ratio = 0.05

    class Axis:
        axis = "x"
        params = Params()
        n = 4
        A = amplitudes
        T = [0.0, 0.01, 0.02, 0.03]
        saved = None

        @staticmethod
        def is_enabled():
            return enabled

    input_shaper = type(
        "InputShaper",
        (),
        {
            "input_shaper_stepper_kinematics": [object()],
            "get_shapers": lambda self: [Axis()],
        },
    )()
    printer = type(
        "Printer",
        (),
        {
            "lookup_object": lambda self, name, default=None: (
                input_shaper if name == "input_shaper" else default
            )
        },
    )()
    shaper_defs = type(
        "Defs",
        (),
        {
            "init_shaper": staticmethod(
                lambda name, frequency, damping: (
                    [0.1, 0.2, 0.3, 0.4],
                    [0.0, 0.01, 0.02, 0.03],
                )
            )
        },
    )
    adapter = KlipperPrinterAdapter.__new__(KlipperPrinterAdapter)
    adapter.printer = printer
    adapter._shaper_defs_module = shaper_defs
    adapter.last_live_python_pulse_proof = None

    with pytest.raises(RuntimeError, match=message):
        adapter.verify_live_python_pulses(
            (ShaperSelection("mzv", 70.0, "X", 0.05),)
        )


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

        @property
        def shaper_type(self):
            return shaping[self.axis]["shaper_type"]

        @property
        def shaper_freq(self):
            return shaping[self.axis]["shaper_freq"]

        @property
        def damping_ratio(self):
            return shaping[self.axis]["damping_ratio"]

        def get_status(self):
            return shaping[self.axis]

    class Axis:
        def __init__(self, axis):
            self.axis = axis
            self.params = Params(axis)
            self.saved = None

        def is_enabled(self):
            return True

        @property
        def _pulses(self):
            if self.axis == "x":
                return CurrentShaperDefs.init_shaper(
                    self.params.shaper_type,
                    float(self.params.shaper_freq),
                    float(self.params.damping_ratio),
                )
            return [0.5, 0.5], [0.0, 0.01]

        @property
        def A(self):
            return self._pulses[0]

        @property
        def T(self):
            return self._pulses[1]

        @property
        def n(self):
            return len(self.A)

    class InputShaper:
        input_shaper_stepper_kinematics = [object()]

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
                if "VELOCITY" in values:
                    velocity["max_velocity"] = float(values["VELOCITY"])
                if "ACCEL" in values:
                    velocity["max_accel"] = float(values["ACCEL"])
                if "SQUARE_CORNER_VELOCITY" in values:
                    velocity["square_corner_velocity"] = float(
                        values["SQUARE_CORNER_VELOCITY"]
                    )
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
            if not name.startswith("mzv("):
                return [0.5, 0.5], [0.0, 0.01]
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

    adapter.set_test_square_corner_velocity(15.0)
    assert velocity["square_corner_velocity"] == 15.0

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
    assert velocity["square_corner_velocity"] == snapshot.square_corner_velocity


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
                "square_corner_velocity": 7.0,
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
