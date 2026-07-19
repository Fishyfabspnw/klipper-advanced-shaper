"""Version-isolated bridge to Klipper's native resonance motion and sensors."""

from __future__ import annotations

import importlib
import inspect
import time
from typing import Any

import numpy as np

_MAX_REPORT_FREQUENCY_BINS = 1024


def _bounded_indices(size: int, limit: int = _MAX_REPORT_FREQUENCY_BINS) -> np.ndarray:
    if size <= 0:
        raise RuntimeError("native calibration spectrum is empty")
    return np.linspace(0, size - 1, min(size, limit), dtype=int)


def _native_spectrum(data: Any) -> dict[str, Any]:
    """Best-effort copy of display-only normalized CalibrationData arrays."""
    required = ("freq_bins", "psd_sum", "psd_x", "psd_y", "psd_z")
    missing = [name for name in required if not hasattr(data, name)]
    if missing:
        return {"available": False, "reason": "missing display fields: %s" % ",".join(missing)}
    try:
        arrays = {name: np.asarray(getattr(data, name), dtype=float) for name in required}
        frequencies = arrays["freq_bins"]
        if frequencies.ndim != 1 or frequencies.size < 2 or np.any(np.diff(frequencies) <= 0):
            raise ValueError("invalid frequency bins")
        if any(value.shape != frequencies.shape for value in arrays.values()):
            raise ValueError("display fields have mismatched shapes")
        if any(not np.all(np.isfinite(value)) for value in arrays.values()):
            raise ValueError("display fields contain non-finite values")
        indices = _bounded_indices(frequencies.size)
        return {
            "available": True,
            "schema": "klipper_calibration_data_v1",
            "normalized": True,
            "source_bins": int(frequencies.size),
            "reported_bins": int(indices.size),
            "frequency_hz": frequencies[indices].tolist(),
            "psd_sum": arrays["psd_sum"][indices].tolist(),
            "psd_x": arrays["psd_x"][indices].tolist(),
            "psd_y": arrays["psd_y"][indices].tolist(),
            "psd_z": arrays["psd_z"][indices].tolist(),
        }
    except (TypeError, ValueError) as error:
        return {"available": False, "reason": str(error)}


def _native_candidate(
    item: Any, calibration_frequency_bins: Any = None, max_frequency: Any = None
) -> dict[str, Any]:
    result = {
        "name": item.name,
        "frequency": float(item.freq),
        "residual_vibration": float(item.vibrs),
        "smoothing": float(item.smoothing),
        "max_accel": float(item.max_accel),
    }
    if hasattr(item, "vals"):
        try:
            candidate_bins = getattr(item, "freq_bins", None)
            if candidate_bins is not None:
                frequencies = np.asarray(candidate_bins, dtype=float)
            elif calibration_frequency_bins is not None:
                frequencies = np.asarray(calibration_frequency_bins, dtype=float)
                if frequencies.ndim == 1 and max_frequency is not None:
                    cutoff = float(max_frequency)
                    if not np.isfinite(cutoff):
                        raise ValueError("invalid maximum calibration frequency")
                    frequencies = frequencies[frequencies <= cutoff]
            else:
                frequencies = np.asarray([], dtype=float)
            response = np.asarray(item.vals, dtype=float)
            if (
                frequencies.ndim == 1
                and frequencies.size >= 2
                and response.shape == frequencies.shape
                and np.all(np.isfinite(frequencies))
                and np.all(np.isfinite(response))
                and np.all(np.diff(frequencies) > 0)
            ):
                indices = _bounded_indices(frequencies.size)
                result["native_frequency_response"] = {
                    "frequency_hz": frequencies[indices].tolist(),
                    "response_ratio": response[indices].tolist(),
                    "source_bins": int(frequencies.size),
                }
        except (TypeError, ValueError, OverflowError):
            pass
    return result


class _Command:
    def __init__(self, validation: bool, responder: Any) -> None:
        self.validation = validation
        self.responder = responder

    def get(self, name: str, default: Any = None) -> Any:
        if name == "INPUT_SHAPING":
            return "1" if self.validation else "0"
        return default

    def get_int(self, name: str, default: int = 0, **_: Any) -> int:
        if name == "INPUT_SHAPING":
            return int(self.validation)
        return int(default)

    def get_float(self, _name: str, default: float, **_: Any) -> float:
        return float(default)

    def respond_info(self, message: str) -> None:
        self.responder(message)

    @staticmethod
    def error(message: str) -> RuntimeError:
        return RuntimeError(message)


class _CaptureResult(dict):
    def add_data(self, other: "_CaptureResult") -> None:
        left = np.asarray(self["samples"], dtype=float)
        right = np.asarray(other["samples"], dtype=float).copy()
        if left.size and right.size:
            dt = float(np.median(np.diff(left[:, 0])))
            right[:, 0] += left[-1, 0] + dt - right[0, 0]
        self["samples"] = np.vstack((left, right))
        self["native_data"].add_data(other["native_data"])
        self["dataset_count"] = int(self.get("dataset_count", 1)) + int(
            other.get("dataset_count", 1)
        )


class _CaptureHelper:
    def __init__(self, native_helper: Any) -> None:
        self.native_helper = native_helper

    def process_accelerometer_data(self, *args: Any) -> dict[str, Any]:
        data = args[-1]
        name = args[0] if len(args) > 1 else None
        samples = np.asarray(data.get_samples(), dtype=float)
        parameters = inspect.signature(self.native_helper.process_accelerometer_data).parameters
        if len(parameters) == 1:
            native_data = self.native_helper.process_accelerometer_data(data)
        else:
            native_data = self.native_helper.process_accelerometer_data(name, data)
        return _CaptureResult(
            samples=samples,
            native_data=native_data,
            dataset_count=1,
        )


class NativeResonanceCaptureProvider:
    """Capture one bounded native resonance sweep without persisting configuration.

    Klipper does not expose a stable third-party capture ABI. This adapter checks the
    expected private signature and fails closed when it changes.
    """

    def __init__(self, config: Any) -> None:
        self.printer = config.get_printer()
        self.gcode = self.printer.lookup_object("gcode")

    def _tester(self) -> Any:
        tester = self.printer.lookup_object("resonance_tester", None)
        if tester is None or not hasattr(tester, "_run_test"):
            raise RuntimeError("compatible [resonance_tester] is required")
        required = {"gcmd", "axes", "helper"}
        if not required <= set(inspect.signature(tester._run_test).parameters):
            raise RuntimeError("unsupported Klipper resonance_tester API")
        return tester

    def preflight(self, axes: tuple[str, ...]) -> None:
        tester = self._tester()
        eventtime = self.printer.get_reactor().monotonic()
        toolhead = self.printer.lookup_object("toolhead")
        homed = str(toolhead.get_kinematics().get_status(eventtime).get("homed_axes", ""))
        missing = [axis for axis in axes if axis.lower() not in homed.lower()]
        if missing:
            raise RuntimeError("requested axes are not homed: %s" % ",".join(missing))
        if not getattr(tester, "accel_chips", None):
            raise RuntimeError("no connected resonance accelerometer")

    def capture(self, axis: str, repeat: int, validation: bool = False) -> dict[str, Any]:
        tester = self._tester()
        module = importlib.import_module(tester.__class__.__module__)
        test_axis = module.TestAxis(axis.lower())
        native_module = importlib.import_module("extras.shaper_calibrate")
        native_helper = native_module.ShaperCalibrate(self.printer)
        helper = _CaptureHelper(native_helper)
        command = _Command(validation, self.gcode.respond_info)
        run_parameters = inspect.signature(tester._run_test).parameters
        run_kwargs = {}
        if "name_suffix" in run_parameters:
            run_kwargs["name_suffix"] = "advanced_%s_%d_%d" % (
                axis.lower(),
                repeat,
                int(time.time()),
            )
        result = tester._run_test(command, [test_axis], helper, **run_kwargs)[test_axis]
        data = result["native_data"]
        data.normalize_to_frequencies()
        result["native_spectrum"] = _native_spectrum(data)
        eventtime = self.printer.get_reactor().monotonic()
        scv = self.printer.lookup_object("toolhead").get_status(eventtime)["square_corner_velocity"]
        max_frequency = tester._get_max_calibration_freq()
        _best, candidates = native_helper.find_best_shaper(
            data,
            max_smoothing=getattr(tester, "max_smoothing", None),
            scv=scv,
            max_freq=max_frequency,
            logger=lambda _message: None,
        )
        result["axis"] = axis.upper()
        result["validation"] = bool(validation)
        result["native_candidates"] = [
            _native_candidate(
                item,
                getattr(data, "freq_bins", None),
                max_frequency,
            )
            for item in candidates
        ]
        start_args = (
            self.printer.get_start_args() if hasattr(self.printer, "get_start_args") else {}
        )
        pulse = getattr(tester.generator, "vibration_generator", tester.generator)
        chip_names = [str(chip.name) for _chip_axis, chip in tester.accel_chips]
        known_16g = any(
            token in name.lower() for name in chip_names for token in ("adxl", "lis2", "lis3")
        )
        result["metadata"] = {
            "klipper_version": str(start_args.get("software_version", "unknown")),
            "sensor_names": chip_names,
            "clip_limit": 16.0 * 9.80665 * 1000.0 if known_16g else None,
            "probe_points": [list(point) for point in tester.probe_points],
            "test_recipe": {
                "freq_start": float(getattr(pulse, "freq_start", 0.0)),
                "freq_end": float(getattr(pulse, "freq_end", 0.0)),
                "accel_per_hz": float(getattr(pulse, "test_accel_per_hz", 0.0)),
                "hz_per_sec": float(getattr(pulse, "test_hz_per_sec", 0.0)),
            },
        }
        result.pop("native_data", None)
        return result
