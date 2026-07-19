"""Version-isolated bridge to Klipper's native resonance motion and sensors."""

from __future__ import annotations

import importlib
import inspect
import time
from typing import Any, Optional

import numpy as np

from klipper_advanced_shaper.shapers import NATIVE_SHAPER_ORDER

from .excitation import check_motion_budget, check_sweep_rate

_MAX_REPORT_FREQUENCY_BINS = 1024
_TRANSIENT_HALF_TRAVEL_MM = 4.0
_TRANSIENT_EDGE_MARGIN_MM = 1.0
_TRANSIENT_SPEED_MM_S = 80.0
_TRANSIENT_MAX_ACCEL_MM_S2 = 5000.0
_TRANSIENT_SETTLE_SECONDS = 0.25
_TRANSIENT_RINGDOWN_SECONDS = 0.75
_TRANSIENT_MIN_SAMPLES = 128
_KNOWN_16G_SENSOR_TOKENS = ("adxl345", "lis2dw", "lis3dh")


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
    item: Any,
    calibration_frequency_bins: Any = None,
    max_frequency: Any = None,
    design_damping_ratio: Any = None,
) -> dict[str, Any]:
    result = {
        "name": item.name,
        "frequency": float(item.freq),
        "residual_vibration": float(item.vibrs),
        "smoothing": float(item.smoothing),
        "max_accel": float(item.max_accel),
    }
    if design_damping_ratio is not None:
        result["design_damping_ratio"] = float(design_damping_ratio)
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
    def __init__(
        self,
        validation: bool,
        responder: Any,
        accel_per_hz: Optional[float] = None,
        hz_per_sec: Optional[float] = None,
    ) -> None:
        self.validation = validation
        self.responder = responder
        self.accel_per_hz = accel_per_hz
        self.hz_per_sec = hz_per_sec

    def get(self, name: str, default: Any = None) -> Any:
        if name == "INPUT_SHAPING":
            return "1" if self.validation else "0"
        return default

    def get_int(self, name: str, default: int = 0, **_: Any) -> int:
        if name == "INPUT_SHAPING":
            return int(self.validation)
        return int(default)

    def get_float(self, name: str, default: float, **_: Any) -> float:
        if name == "ACCEL_PER_HZ" and self.accel_per_hz is not None:
            return self.accel_per_hz
        if name == "HZ_PER_SEC" and self.hz_per_sec is not None:
            return self.hz_per_sec
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

    def preflight_excitation(
        self,
        axes: tuple[str, ...],
        accel_per_hz: Optional[float],
        hz_per_sec: Optional[float],
    ) -> dict[str, Any]:
        if not axes or any(axis not in {"X", "Y"} for axis in axes):
            raise RuntimeError("resonance excitation preflight supports only X and Y")
        tester = self._tester()
        generator = tester.generator
        pulse = getattr(generator, "vibration_generator", generator)
        effective = (
            accel_per_hz
            if accel_per_hz is not None
            else getattr(pulse, "accel_per_hz", None)
        )
        max_frequency = getattr(pulse, "max_freq", getattr(pulse, "freq_end", None))
        min_frequency = getattr(pulse, "min_freq", None)
        sweeping_accel = getattr(generator, "sweeping_accel", 0.0)
        eventtime = self.printer.get_reactor().monotonic()
        motion_limit = self.printer.lookup_object("toolhead").get_status(eventtime).get(
            "max_accel"
        )
        result = dict(
            check_motion_budget(
                effective,
                max_frequency,
                motion_limit,
                sweeping_accel,
            )
        )
        result["source"] = "command" if accel_per_hz is not None else "resonance_tester"
        effective_hz_per_sec = (
            hz_per_sec
            if hz_per_sec is not None
            else getattr(pulse, "hz_per_sec", None)
        )
        result["hz_per_sec"] = check_sweep_rate(effective_hz_per_sec)
        try:
            result["min_frequency_hz"] = float(min_frequency)
        except (TypeError, ValueError) as error:
            raise RuntimeError("resonance minimum frequency is unavailable") from error
        if (
            not np.isfinite(result["min_frequency_hz"])
            or result["min_frequency_hz"] <= 0.0
            or result["min_frequency_hz"] >= result["max_frequency_hz"]
        ):
            raise RuntimeError("resonance frequency range is invalid")
        result["hz_per_sec_source"] = (
            "command" if hz_per_sec is not None else "resonance_tester"
        )
        result["axes"] = list(axes)
        return result

    @staticmethod
    def _require_bound_call(target: Any, name: str, *arguments: Any) -> Any:
        method = getattr(target, name, None)
        if method is None or not callable(method):
            raise RuntimeError("stock Klipper %s API is unavailable" % name)
        try:
            inspect.signature(method).bind(*arguments)
        except (TypeError, ValueError) as error:
            raise RuntimeError("unsupported stock Klipper %s API" % name) from error
        return method

    @staticmethod
    def _coordinate_component(value: Any, axis: str) -> float:
        index = 0 if axis == "X" else 1
        try:
            component = getattr(value, axis.lower())
        except AttributeError:
            try:
                component = value[index]
            except (IndexError, KeyError, TypeError) as error:
                raise RuntimeError(
                    "Klipper kinematics did not expose %s-axis bounds" % axis
                ) from error
        try:
            result = float(component)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "Klipper kinematics returned an invalid %s-axis bound" % axis
            ) from error
        if not np.isfinite(result):
            raise RuntimeError(
                "Klipper kinematics returned a non-finite %s-axis bound" % axis
            )
        return result

    def _transient_chips(self, axis: str) -> list[tuple[str, Any]]:
        tester = self._tester()
        matches = [
            (str(chip_axis), chip)
            for chip_axis, chip in getattr(tester, "accel_chips", ())
            if axis.lower() in str(chip_axis).lower()
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "%s-axis transient validation requires exactly one matching "
                "accelerometer, found %d" % (axis, len(matches))
            )
        chip_axis, chip = matches[0]
        self._require_bound_call(chip, "start_internal_client")
        if not getattr(chip, "name", None):
            raise RuntimeError("transient accelerometer has no stable Klipper name")
        return [(chip_axis, chip)]

    @staticmethod
    def _transient_clip_limit(chip: Any) -> float:
        identity = " ".join(
            (
                str(getattr(chip, "name", "")),
                type(chip).__name__,
                type(chip).__module__,
            )
        ).lower()
        if not any(token in identity for token in _KNOWN_16G_SENSOR_TOKENS):
            raise RuntimeError(
                "transient accelerometer full-scale range cannot be proven"
            )
        return 16.0 * 9.80665 * 1000.0

    def preflight_transient(self, axes: tuple[str, ...]) -> dict[str, Any]:
        """Prove and plan a short, bounded stock-toolhead reversal transient.

        This intentionally uses only Klipper's Python toolhead and accelerometer
        interfaces.  It does not inspect or claim readback from the C executor.
        """
        if not axes or any(axis not in {"X", "Y"} for axis in axes):
            raise RuntimeError("transient validation supports only X and Y")
        toolhead = self.printer.lookup_object("toolhead", None)
        if toolhead is None:
            raise RuntimeError("stock Klipper toolhead is unavailable")
        self._require_bound_call(toolhead, "get_position")
        self._require_bound_call(toolhead, "get_last_move_time")
        self._require_bound_call(toolhead, "manual_move", [None, None], 1.0)
        self._require_bound_call(toolhead, "wait_moves")
        self._require_bound_call(toolhead, "dwell", 0.1)

        eventtime = self.printer.get_reactor().monotonic()
        status = toolhead.get_status(eventtime)
        kinematics = toolhead.get_kinematics()
        kinematics_status = kinematics.get_status(eventtime)
        try:
            current = list(toolhead.get_position())
            max_velocity = float(status["max_velocity"])
            configured_max_accel = float(status["max_accel"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                "Klipper toolhead motion limits are unavailable for transient validation"
            ) from error
        if len(current) < 2 or any(not np.isfinite(float(value)) for value in current[:2]):
            raise RuntimeError("Klipper toolhead position is invalid")
        if (
            not np.isfinite(max_velocity)
            or not np.isfinite(configured_max_accel)
            or max_velocity <= 0.0
            or configured_max_accel <= 0.0
        ):
            raise RuntimeError("Klipper toolhead motion limits are invalid")

        speed = min(_TRANSIENT_SPEED_MM_S, max_velocity)
        if speed < 10.0:
            raise RuntimeError("printer max_velocity is too low for transient validation")
        target_accel = min(_TRANSIENT_MAX_ACCEL_MM_S2, configured_max_accel)
        plans: dict[str, Any] = {}
        for axis in axes:
            _chip_axis, chip = self._transient_chips(axis)[0]
            clip_limit = self._transient_clip_limit(chip)
            index = 0 if axis == "X" else 1
            minimum = self._coordinate_component(
                kinematics_status.get("axis_minimum"), axis
            )
            maximum = self._coordinate_component(
                kinematics_status.get("axis_maximum"), axis
            )
            usable_minimum = minimum + _TRANSIENT_EDGE_MARGIN_MM
            usable_maximum = maximum - _TRANSIENT_EDGE_MARGIN_MM
            if usable_maximum - usable_minimum < 2.0 * _TRANSIENT_HALF_TRAVEL_MM:
                raise RuntimeError(
                    "%s-axis travel is too small for bounded transient validation" % axis
                )
            anchor = min(
                max(float(current[index]), usable_minimum + _TRANSIENT_HALF_TRAVEL_MM),
                usable_maximum - _TRANSIENT_HALF_TRAVEL_MM,
            )
            start = anchor - _TRANSIENT_HALF_TRAVEL_MM
            reversal = anchor + _TRANSIENT_HALF_TRAVEL_MM
            if not usable_minimum <= start < reversal <= usable_maximum:
                raise RuntimeError(
                    "%s-axis transient geometry is outside kinematic limits" % axis
                )
            plans[axis] = {
                "axis": axis,
                "original_position_mm": float(current[index]),
                "anchor_position_mm": anchor,
                "start_position_mm": start,
                "reversal_position_mm": reversal,
                "return_position_mm": start,
                "half_travel_mm": _TRANSIENT_HALF_TRAVEL_MM,
                "transient_excitation_travel_mm": 4.0 * _TRANSIENT_HALF_TRAVEL_MM,
                "approach_travel_mm": abs(float(current[index]) - start),
                "position_restore_travel_mm": abs(start - float(current[index])),
                "total_capture_travel_mm": (
                    4.0 * _TRANSIENT_HALF_TRAVEL_MM
                    + 2.0 * abs(float(current[index]) - start)
                ),
                "axis_minimum_mm": minimum,
                "axis_maximum_mm": maximum,
                "edge_margin_mm": _TRANSIENT_EDGE_MARGIN_MM,
                "sensor_name": str(chip.name),
                "clip_limit": clip_limit,
            }
        return {
            "passed": True,
            "protocol": "finite_reversal_ringdown_v1",
            "promotion_role": "mandatory_paired_held_out_validation",
            "stock_klipper_python_interfaces_only": True,
            "motion_planner_modified": False,
            "live_c_executor_readback": False,
            "speed_mm_s": speed,
            "max_accel_mm_s2": target_accel,
            "settle_seconds": _TRANSIENT_SETTLE_SECONDS,
            "ringdown_seconds": _TRANSIENT_RINGDOWN_SECONDS,
            "estimated_base_motion_seconds_per_capture_upper_bound": 4.0,
            "maximum_supported_post_command_guard_seconds": 10.0,
            "estimated_motion_seconds_per_capture_upper_bound": 14.0,
            "plans": plans,
        }

    def capture_transient(
        self,
        axis: str,
        repeat: int,
        plan: Any,
        max_accel_mm_s2: Any,
        speed_mm_s: Any,
        post_command_guard_seconds: Any = 0.0,
    ) -> dict[str, Any]:
        """Capture raw post-command ring-down after a finite reversal."""
        axis = str(axis).upper()
        preflight = self.preflight_transient((axis,))
        try:
            start = float(plan["start_position_mm"])
            reversal = float(plan["reversal_position_mm"])
            returned = float(plan["return_position_mm"])
            original = float(plan["original_position_mm"])
            anchor = float(plan["anchor_position_mm"])
            expected_accel = float(max_accel_mm_s2)
            speed = float(speed_mm_s)
            guard = float(post_command_guard_seconds)
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("transient validation plan is malformed") from error
        current_plan = preflight["plans"][axis]
        if abs(float(current_plan["original_position_mm"]) - original) > 0.001:
            raise RuntimeError(
                "%s-axis position changed after transient validation preflight" % axis
            )
        for name in (
            "anchor_position_mm", "start_position_mm", "reversal_position_mm",
            "return_position_mm",
        ):
            if abs(float(current_plan[name]) - float(plan[name])) > 1e-9:
                raise RuntimeError("transient validation plan is stale")
        minimum = float(current_plan["axis_minimum_mm"])
        maximum = float(current_plan["axis_maximum_mm"])
        if not minimum <= original <= maximum:
            raise RuntimeError("transient original position is outside current axis limits")
        for value in (anchor, start, reversal, returned):
            if not (
                minimum + _TRANSIENT_EDGE_MARGIN_MM
                <= value
                <= maximum - _TRANSIENT_EDGE_MARGIN_MM
            ):
                raise RuntimeError("transient validation plan is outside current axis limits")
        if returned != start or reversal <= start:
            raise RuntimeError("transient validation plan is not a finite reversal")
        if abs((reversal - start) - 2.0 * _TRANSIENT_HALF_TRAVEL_MM) > 1e-9:
            raise RuntimeError("transient validation plan has unexpected geometry")
        if (
            not np.isfinite(expected_accel)
            or expected_accel <= 0.0
            or expected_accel > _TRANSIENT_MAX_ACCEL_MM_S2
        ):
            raise RuntimeError("transient validation acceleration is unsafe")
        if (
            not np.isfinite(speed)
            or speed < 10.0
            or speed > min(_TRANSIENT_SPEED_MM_S, preflight["speed_mm_s"])
        ):
            raise RuntimeError("transient validation speed is unsafe")
        if not np.isfinite(guard) or not 0.0 <= guard <= 10.0:
            raise RuntimeError("transient post-command guard is unsafe")

        toolhead = self.printer.lookup_object("toolhead")
        eventtime = self.printer.get_reactor().monotonic()
        actual_accel = float(toolhead.get_status(eventtime)["max_accel"])
        if abs(actual_accel - expected_accel) > 0.0005:
            raise RuntimeError(
                "transient max_accel readback mismatch: expected %.6f, got %.6f"
                % (expected_accel, actual_accel)
            )

        _chip_axis, chip = self._transient_chips(axis)[0]
        clip_limit = self._transient_clip_limit(chip)
        if (
            str(chip.name) != str(plan.get("sensor_name"))
            or abs(clip_limit - float(plan.get("clip_limit", float("nan")))) > 1e-9
        ):
            raise RuntimeError("transient accelerometer changed after preflight")
        client = chip.start_internal_client()
        for name in ("finish_measurements", "has_valid_samples", "get_samples"):
            self._require_bound_call(client, name)
        finished = False
        position_may_have_changed = False
        try:
            coordinate = [None, None]
            coordinate[0 if axis == "X" else 1] = start
            toolhead.manual_move(coordinate, speed)
            position_may_have_changed = True
            toolhead.wait_moves()
            toolhead.dwell(_TRANSIENT_SETTLE_SECONDS)
            transient_start = float(toolhead.get_last_move_time())

            coordinate[0 if axis == "X" else 1] = reversal
            toolhead.manual_move(coordinate, speed)
            coordinate[0 if axis == "X" else 1] = returned
            toolhead.manual_move(coordinate, speed)
            motion_end = float(toolhead.get_last_move_time())
            if not np.isfinite(motion_end) or motion_end <= transient_start:
                raise RuntimeError("Klipper did not queue the transient motion")
            # Klipper's toolhead print_time marks the nominal trapq command end,
            # not a public C-executor tail timestamp.  Wait through the complete
            # installed Python pulse span supplied by the post-SET pulse proof,
            # plus its conservative margin, before starting the ring-down window.
            toolhead.dwell(guard + _TRANSIENT_RINGDOWN_SECONDS)
            capture_end = float(toolhead.get_last_move_time())
            client.finish_measurements()
            finished = True
        finally:
            if not finished:
                try:
                    client.finish_measurements()
                except Exception:
                    pass
            if position_may_have_changed:
                coordinate = [None, None]
                coordinate[0 if axis == "X" else 1] = original
                toolhead.manual_move(coordinate, speed)
                toolhead.wait_moves()
        position_after = float(toolhead.get_position()[0 if axis == "X" else 1])
        if abs(position_after - original) > 0.001:
            raise RuntimeError(
                "%s-axis transient position restore mismatch: expected %.6f, got %.6f"
                % (axis, original, position_after)
            )
        if not client.has_valid_samples():
            raise RuntimeError("transient accelerometer measured no valid data")
        samples = np.asarray(client.get_samples(), dtype=float)
        if (
            samples.ndim != 2
            or samples.shape[1] != 4
            or samples.shape[0] < _TRANSIENT_MIN_SAMPLES
            or not np.all(np.isfinite(samples))
            or np.any(np.diff(samples[:, 0]) <= 0.0)
        ):
            raise RuntimeError("transient accelerometer samples are malformed")
        sample_interval = float(np.median(np.diff(samples[:, 0])))
        if not np.isfinite(sample_interval) or sample_interval <= 0.0:
            raise RuntimeError("transient accelerometer sample interval is invalid")
        requested_start = motion_end + guard
        requested_end = capture_end
        window = samples[
            (samples[:, 0] >= requested_start) & (samples[:, 0] <= requested_end)
        ]
        if window.shape[0] < _TRANSIENT_MIN_SAMPLES:
            raise RuntimeError("post-command transient ring-down window is incomplete")
        start_offset = float(window[0, 0] - requested_start)
        end_offset = float(requested_end - window[-1, 0])
        boundary_tolerance = 2.5 * sample_interval
        if (
            start_offset < -1e-12
            or end_offset < -1e-12
            or start_offset > boundary_tolerance
            or end_offset > boundary_tolerance
            or float(window[-1, 0] - window[0, 0])
            < _TRANSIENT_RINGDOWN_SECONDS - 2.0 * boundary_tolerance
        ):
            raise RuntimeError("post-command transient ring-down window is incomplete")
        sample_rate = 1.0 / float(np.median(np.diff(window[:, 0])))
        if not np.isfinite(sample_rate) or not 100.0 <= sample_rate <= 10000.0:
            raise RuntimeError("transient accelerometer sample rate is invalid")
        return {
            "samples": window,
            "axis": axis,
            "repeat": int(repeat),
            "validation": True,
            "native_candidates": [],
            "native_spectrum": {
                "available": False,
                "reason": "finite transient uses raw post-command samples",
            },
            "metadata": {
                "validation_capture_kind": "finite_reversal_ringdown",
                "protocol": "finite_reversal_ringdown_v1",
                "promotion_eligible": True,
                "sample_semantics": "raw_accelerometer_post_command_ringdown",
                "timebase": "accelerometer_timestamps_aligned_to_toolhead_print_time",
                "transient_start_time": transient_start,
                "command_end_time": motion_end,
                "post_command_guard_seconds": guard,
                "post_command_guard_basis": (
                    "full_live_python_pulse_span_plus_conservative_margin"
                ),
                "ringdown_window_start_time": float(window[0, 0]),
                "ringdown_window_end_time": float(window[-1, 0]),
                "ringdown_requested_start_time": requested_start,
                "ringdown_requested_end_time": requested_end,
                "ringdown_start_offset_seconds": start_offset,
                "ringdown_end_offset_seconds": end_offset,
                "ringdown_boundary_tolerance_seconds": boundary_tolerance,
                "ringdown_duration_seconds": float(window[-1, 0] - window[0, 0]),
                "sample_rate_hz": sample_rate,
                "ringdown_sample_count": int(window.shape[0]),
                "full_capture_sample_count": int(samples.shape[0]),
                "sensor_names": [str(chip.name)],
                "clip_limit": clip_limit,
                "pre_capture_axis_position_mm": original,
                "post_capture_axis_position_mm": position_after,
                "position_restore_tolerance_mm": 0.001,
                "position_restored": True,
                "motion_recipe": {
                    "axis": axis,
                    "original_position_mm": original,
                    "anchor_position_mm": anchor,
                    "start_position_mm": start,
                    "reversal_position_mm": reversal,
                    "return_position_mm": returned,
                    "speed_mm_s": speed,
                    "max_accel_mm_s2": expected_accel,
                    "settle_seconds": _TRANSIENT_SETTLE_SECONDS,
                    "ringdown_seconds": _TRANSIENT_RINGDOWN_SECONDS,
                    "post_command_guard_seconds": guard,
                    "transient_excitation_travel_mm": 2.0 * (reversal - start),
                    "approach_travel_mm": abs(original - start),
                    "position_restore_travel_mm": abs(start - original),
                    "total_capture_travel_mm": (
                        2.0 * (reversal - start) + 2.0 * abs(original - start)
                    ),
                },
                "cross_axis_channels_retained": True,
                "qc_required": True,
                "stock_klipper_python_interfaces_only": True,
                "motion_planner_modified": False,
                "live_c_executor_readback": False,
            },
        }

    @staticmethod
    def _validate_max_vibrations(value: Any) -> float:
        try:
            threshold = float(value)
        except (TypeError, ValueError) as error:
            raise RuntimeError("max_vibrations threshold is not numeric") from error
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise RuntimeError("max_vibrations threshold must be a fraction within [0, 1]")
        return threshold

    @staticmethod
    def _native_fitting_method() -> Any:
        try:
            native_module = importlib.import_module("extras.shaper_calibrate")
            return native_module.ShaperCalibrate.find_best_shaper
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                "installed Klipper native shaper fitting API is unavailable"
            ) from error

    def preflight_native_fitting(self, max_vibrations: Any) -> dict[str, Any]:
        """Prove the opt-in upstream residual threshold is explicitly supported."""
        threshold = self._validate_max_vibrations(max_vibrations)
        try:
            parameters = inspect.signature(self._native_fitting_method()).parameters
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "cannot inspect installed Klipper native shaper fitting API"
            ) from error
        if "max_vibrations" not in parameters:
            raise RuntimeError(
                "installed Klipper find_best_shaper lacks max_vibrations support"
            )
        return {
            "passed": True,
            "parameter": "max_vibrations",
            "fraction": threshold,
            "percent": threshold * 100.0,
        }

    def capture(
        self,
        axis: str,
        repeat: int,
        validation: bool = False,
        accel_per_hz: Optional[float] = None,
        hz_per_sec: Optional[float] = None,
        design_damping_ratio: Optional[float] = None,
        native_shapers: Optional[tuple[str, ...]] = None,
        max_vibrations: Optional[float] = None,
    ) -> dict[str, Any]:
        if design_damping_ratio is None or not np.isfinite(design_damping_ratio):
            raise RuntimeError("active input-shaper damping is required for native fitting")
        if not 0.0 <= float(design_damping_ratio) < 1.0:
            raise RuntimeError("active input-shaper damping must be within [0, 1)")
        if native_shapers is not None and tuple(native_shapers) != NATIVE_SHAPER_ORDER:
            raise RuntimeError("native shaper override is outside the stock allowlist")
        fitting_proof = (
            self.preflight_native_fitting(max_vibrations)
            if max_vibrations is not None
            else None
        )
        tester = self._tester()
        module = importlib.import_module(tester.__class__.__module__)
        test_axis = module.TestAxis(axis.lower())
        native_module = importlib.import_module("extras.shaper_calibrate")
        native_helper = native_module.ShaperCalibrate(self.printer)
        helper = _CaptureHelper(native_helper)
        command = _Command(
            validation,
            self.gcode.respond_info,
            accel_per_hz,
            hz_per_sec,
        )
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
        result["axis"] = axis.upper()
        result["validation"] = bool(validation)
        if validation:
            # Held-out captures are judged from their raw samples and QC only.
            # Retain normalized display evidence, but do not repeat Klipper's
            # candidate search after every reference/candidate motion.
            result["native_candidates"] = []
        else:
            eventtime = self.printer.get_reactor().monotonic()
            scv = self.printer.lookup_object("toolhead").get_status(eventtime)[
                "square_corner_velocity"
            ]
            max_frequency = tester._get_max_calibration_freq()
            fit_arguments = {
                "damping_ratio": float(design_damping_ratio),
                "max_smoothing": getattr(tester, "max_smoothing", None),
                "scv": scv,
                "max_freq": max_frequency,
                "logger": lambda _message: None,
            }
            if native_shapers is not None:
                fit_arguments["shapers"] = NATIVE_SHAPER_ORDER
            if fitting_proof is not None:
                fit_arguments["max_vibrations"] = fitting_proof["fraction"]
            _best, candidates = native_helper.find_best_shaper(data, **fit_arguments)
            result["native_candidates"] = [
                _native_candidate(
                    item,
                    getattr(data, "freq_bins", None),
                    max_frequency,
                    design_damping_ratio,
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
            "native_design_damping_ratio": float(design_damping_ratio),
            "native_design_damping_source": "active_input_shaper_status",
            "native_fit_max_vibrations": (
                fitting_proof["fraction"] if fitting_proof is not None else None
            ),
        }
        if validation:
            result["metadata"]["native_fitting_performed"] = False
            result["metadata"][
                "native_fitting_status"
            ] = "skipped_held_out_validation"
            result["metadata"][
                "validation_capture_kind"
            ] = "native_compatibility_validation_sweep"
            result["metadata"]["experimental_promotion_eligible"] = False
            result["metadata"][
                "experimental_promotion_exclusion_reason"
            ] = "experimental profiles require finite-reversal ring-down evidence"
        result.pop("native_data", None)
        return result
