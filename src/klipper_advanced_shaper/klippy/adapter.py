"""Klipper boundary adapters.

Only this module knows about Klipper object names and g-code.  The controller is
kept dependency-injected and therefore importable without Klipper installed.
"""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence

from klipper_advanced_shaper.shapers import NATIVE_SHAPER_ORDER, parse_shaper_identifier


@dataclass(frozen=True)
class PrinterSnapshot:
    shaper_type_x: str
    shaper_freq_x: float
    shaper_type_y: str
    shaper_freq_y: float
    max_velocity: float
    max_accel: float
    square_corner_velocity: float
    damping_ratio_x: float
    damping_ratio_y: float
    minimum_cruise_ratio: Optional[float] = None


@dataclass(frozen=True)
class ShaperSelection:
    shaper_type: str
    frequency: float
    axis: str
    damping_ratio: float

    def __post_init__(self) -> None:
        axis = self.axis.upper()
        kind = self.shaper_type.lower()
        if axis not in {"X", "Y"}:
            raise ValueError("axis must be X or Y")
        if not 1.0 <= float(self.frequency) <= 200.0:
            raise ValueError("shaper frequency must be between 1 and 200 Hz")
        identifier = parse_shaper_identifier(kind, allow_parameterized=True)
        limits = {"ei": 0.4, "2hump_ei": 0.3, "3hump_ei": 0.2}
        maximum_damping = limits.get(identifier.family, 0.99)
        if not 0.0 <= float(self.damping_ratio) <= maximum_damping:
            raise ValueError(
                "damping ratio must be between 0 and %.2f for %s"
                % (maximum_damping, identifier.family)
            )
        object.__setattr__(self, "shaper_type", identifier.canonical)
        object.__setattr__(self, "axis", axis)

    @property
    def parameterized(self) -> bool:
        return parse_shaper_identifier(self.shaper_type).parameterized

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "axis": self.axis,
            "shaper_type": self.shaper_type,
            "frequency_hz": self.frequency,
            "damping_ratio": self.damping_ratio,
        }
        return result


class PrinterAdapter(Protocol):
    def preflight(self, axes: Sequence[str]) -> None: ...
    def preflight_excitation(
        self,
        axes: Sequence[str],
        accel_per_hz: Optional[float],
        hz_per_sec: Optional[float],
    ) -> Mapping[str, Any]: ...
    def preflight_experimental(
        self,
        selections: Sequence[ShaperSelection] = (),
        max_vibrations: Optional[float] = None,
    ) -> Mapping[str, Any]: ...
    def snapshot(self) -> PrinterSnapshot: ...
    def capture(
        self,
        axis: str,
        repeat: int,
        validation: bool = False,
        accel_per_hz: Optional[float] = None,
        hz_per_sec: Optional[float] = None,
        max_vibrations: Optional[float] = None,
    ) -> Any: ...
    def apply_temporary(self, selections: Sequence[ShaperSelection]) -> None: ...
    def set_test_square_corner_velocity(self, value: float) -> None: ...
    def restore(self, snapshot: PrinterSnapshot) -> None: ...
    def stage(self, selections: Sequence[ShaperSelection]) -> None: ...
    def respond(self, message: str) -> None: ...


class KlipperPrinterAdapter:
    """Thin adapter around stable Klippy objects and native g-code commands.

    A capture provider is intentionally injected/registered separately.  It is
    responsible for using Klipper's accelerometer/resonance APIs and must not
    move the toolhead outside the bounded calibration command.
    """

    def __init__(
        self,
        config: Any,
        *,
        shaper_defs_module: Any = None,
        executor_pulse_limit: Optional[int] = None,
    ) -> None:
        self.config = config
        self.printer = config.get_printer()
        self.gcode = self.printer.lookup_object("gcode")
        self.capture_provider = self.printer.lookup_object("advanced_shaper_capture", None)
        self._shaper_defs_module = shaper_defs_module
        self._executor_pulse_limit = executor_pulse_limit
        self._capture_native_shapers: Optional[Sequence[str]] = None
        self.last_capability: Optional[Mapping[str, Any]] = None

    def configure_capture_profile(self, profile: str) -> None:
        self._capture_native_shapers = (
            NATIVE_SHAPER_ORDER if str(profile).lower() == "adaptive_stock" else None
        )

    def _load_shaper_defs(self) -> Any:
        if self._shaper_defs_module is not None:
            return self._shaper_defs_module
        errors = []
        for name in ("extras.shaper_defs", "klippy.extras.shaper_defs", "shaper_defs"):
            try:
                self._shaper_defs_module = importlib.import_module(name)
                return self._shaper_defs_module
            except ImportError as error:
                errors.append(str(error))
        raise RuntimeError(
            "installed Klipper shaper definitions are unavailable: %s" % "; ".join(errors)
        )

    def _get_executor_pulse_limit(self) -> int:
        if self._executor_pulse_limit is not None:
            return int(self._executor_pulse_limit)
        module = self._load_shaper_defs()
        module_path = getattr(module, "__file__", None)
        if not module_path:
            raise RuntimeError("cannot locate installed Klipper executor source")
        source = Path(module_path).resolve().parent.parent / "chelper" / "kin_shaper.c"
        try:
            content = source.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise RuntimeError("cannot inspect installed Klipper executor: %s" % error) from error
        match = re.search(r"\bpulses\s*\[\s*(\d+)\s*\]\s*;", content)
        if match is None:
            raise RuntimeError("cannot prove installed Klipper executor pulse capacity")
        limit = int(match.group(1))
        if not 3 <= limit <= 10:
            raise RuntimeError("unsupported installed Klipper executor pulse capacity: %d" % limit)
        self._executor_pulse_limit = limit
        return limit

    def preflight_experimental(
        self,
        selections: Sequence[ShaperSelection] = (),
        max_vibrations: Optional[float] = None,
    ) -> Mapping[str, Any]:
        from klipper_advanced_shaper.analysis.experimental import (
            prove_runtime_generalized_mzv,
            prove_runtime_native_shapers,
        )

        module = self._load_shaper_defs()
        executor_limit = self._get_executor_pulse_limit()
        upper_probe = min(executor_limit, 10)
        proofs = [
            prove_runtime_generalized_mzv(module, syntax)
            for syntax in (
                "mzv(n=3,t=0.750000)",
                "mzv(n=4,tau=1.000000)",
                "mzv(n=%d,t=1.000000)" % upper_probe,
            )
        ]
        failed = next((item for item in proofs if not item.get("passed")), None)
        native_proof = prove_runtime_native_shapers(
            module,
            families=NATIVE_SHAPER_ORDER,
            executor_pulse_limit=executor_limit,
        )
        if not native_proof.get("passed"):
            raise RuntimeError(
                "installed Klipper does not safely support the native allowlist: %s"
                % native_proof.get("reason", "capability proof failed")
            )
        proof: dict[str, Any] = {
            "passed": failed is None,
            "family": "generalized_mzv",
            "executor_pulse_limit": executor_limit,
            "proofs": proofs,
            "native_proof": native_proof,
            "reason": None if failed is None else failed.get("reason"),
        }
        if failed is not None:
            raise RuntimeError(
                "installed Klipper does not safely support generalized MZV: %s"
                % failed.get("reason", "capability proof failed")
            )
        if max_vibrations is not None:
            if self.capture_provider is None:
                self.capture_provider = self.printer.lookup_object(
                    "advanced_shaper_capture", None
                )
            fitting_probe = getattr(
                self.capture_provider, "preflight_native_fitting", None
            )
            if fitting_probe is None:
                raise RuntimeError(
                    "capture provider cannot prove max_vibrations fitting support"
                )
            proof["native_fitting"] = fitting_probe(max_vibrations)
        proof["selection_proofs"] = [
            self._prove_selection(selection)
            for selection in selections
            if selection.parameterized
        ]
        self.last_capability = proof
        return proof

    def _prove_selection(self, selection: ShaperSelection) -> Mapping[str, Any]:
        if not selection.parameterized:
            return {"passed": True, "syntax": selection.shaper_type, "parameterized": False}
        from klipper_advanced_shaper.analysis.experimental import prove_runtime_generalized_mzv

        identifier = parse_shaper_identifier(selection.shaper_type)
        pulse_count = int(identifier.argument_map()["n"])
        executor_limit = self._get_executor_pulse_limit()
        if pulse_count > executor_limit:
            raise RuntimeError(
                "installed Klipper executor supports %d pulses, but %s requires %d"
                % (executor_limit, selection.shaper_type, pulse_count)
            )
        proof = prove_runtime_generalized_mzv(
            self._load_shaper_defs(),
            selection.shaper_type,
            selection.frequency,
            selection.damping_ratio,
        )
        self.last_capability = proof
        if not proof.get("passed"):
            raise RuntimeError(
                "installed Klipper rejected exact shaper %s: %s"
                % (selection.shaper_type, proof.get("reason", "capability proof failed"))
            )
        return proof

    def preflight(self, axes: Sequence[str]) -> None:
        eventtime = self.printer.get_reactor().monotonic()
        idle_timeout = self.printer.lookup_object("idle_timeout", None)
        status = idle_timeout.get_status(eventtime) if idle_timeout is not None else {}
        if str(status.get("state", "")).lower() == "printing":
            raise RuntimeError("calibration is not allowed while printing")
        print_stats = self.printer.lookup_object("print_stats", None)
        print_state = print_stats.get_status(eventtime) if print_stats is not None else {}
        if str(print_state.get("state", "")).lower() in {"printing", "paused"}:
            raise RuntimeError("calibration is not allowed while printing or paused")
        if self.capture_provider is None:
            # Config sections may be loaded after this extra.
            self.capture_provider = self.printer.lookup_object("advanced_shaper_capture", None)
        if self.capture_provider is None:
            raise RuntimeError("advanced_shaper_capture provider is not registered")
        self.capture_provider.preflight(tuple(axes))

    def preflight_excitation(
        self,
        axes: Sequence[str],
        accel_per_hz: Optional[float],
        hz_per_sec: Optional[float],
    ) -> Mapping[str, Any]:
        if self.capture_provider is None or not hasattr(
            self.capture_provider, "preflight_excitation"
        ):
            raise RuntimeError("capture provider cannot prove resonance excitation limits")
        return self.capture_provider.preflight_excitation(
            tuple(axes), accel_per_hz, hz_per_sec
        )

    def snapshot(self) -> PrinterSnapshot:
        eventtime = self.printer.get_reactor().monotonic()
        toolhead = self.printer.lookup_object("toolhead")
        velocity = toolhead.get_status(eventtime)
        shaping = self._shaping_status(eventtime)
        required = {
            "shaper_type_x", "shaper_freq_x", "damping_ratio_x",
            "shaper_type_y", "shaper_freq_y", "damping_ratio_y",
        }
        missing = sorted(required.difference(shaping))
        if missing:
            raise RuntimeError("Klipper input-shaper status is missing: %s" % ", ".join(missing))
        return PrinterSnapshot(
            shaper_type_x=parse_shaper_identifier(str(shaping["shaper_type_x"])).canonical,
            shaper_freq_x=float(shaping["shaper_freq_x"]),
            shaper_type_y=parse_shaper_identifier(str(shaping["shaper_type_y"])).canonical,
            shaper_freq_y=float(shaping["shaper_freq_y"]),
            damping_ratio_x=float(shaping["damping_ratio_x"]),
            damping_ratio_y=float(shaping["damping_ratio_y"]),
            max_velocity=float(velocity["max_velocity"]),
            max_accel=float(velocity["max_accel"]),
            square_corner_velocity=float(velocity["square_corner_velocity"]),
            minimum_cruise_ratio=(
                float(velocity["minimum_cruise_ratio"])
                if "minimum_cruise_ratio" in velocity
                else None
            ),
        )

    def capture(
        self,
        axis: str,
        repeat: int,
        validation: bool = False,
        accel_per_hz: Optional[float] = None,
        hz_per_sec: Optional[float] = None,
        max_vibrations: Optional[float] = None,
    ) -> Any:
        shaping = self._shaping_status()
        damping_key = "damping_ratio_" + axis.lower()
        try:
            design_damping_ratio = float(shaping[damping_key])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                "cannot determine active %s-axis design damping: %s" % (axis, error)
            ) from error
        if not 0.0 <= design_damping_ratio < 1.0:
            raise RuntimeError(
                "active %s-axis design damping is outside [0, 1)" % axis
            )
        return self.capture_provider.capture(
            axis=axis,
            repeat=repeat,
            validation=validation,
            accel_per_hz=accel_per_hz,
            hz_per_sec=hz_per_sec,
            design_damping_ratio=design_damping_ratio,
            native_shapers=self._capture_native_shapers,
            max_vibrations=max_vibrations,
        )

    def _shaping_status(self, eventtime: Optional[float] = None) -> dict[str, Any]:
        input_shaper = self.printer.lookup_object("input_shaper", None)
        if input_shaper is None:
            raise RuntimeError("Klipper input_shaper status is unavailable")
        if eventtime is None:
            eventtime = self.printer.get_reactor().monotonic()
        if hasattr(input_shaper, "get_status"):
            return dict(input_shaper.get_status(eventtime))
        shaping: dict[str, Any] = {}
        for shaper in input_shaper.get_shapers():
            suffix = str(shaper.axis).lower()
            for key, value in shaper.params.get_status().items():
                shaping[key + "_" + suffix] = value
        return shaping

    def verify_applied(self, selections: Sequence[ShaperSelection]) -> None:
        shaping = self._shaping_status()
        for item in selections:
            suffix = item.axis.lower()
            try:
                actual_type = parse_shaper_identifier(
                    str(shaping["shaper_type_" + suffix])
                ).canonical
                actual_frequency = float(shaping["shaper_freq_" + suffix])
                actual_damping = float(shaping["damping_ratio_" + suffix])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    "cannot read back applied %s-axis shaper: %s" % (item.axis, error)
                )
            if actual_type != item.shaper_type:
                raise RuntimeError(
                    "%s-axis shaper readback mismatch: expected %s, got %s"
                    % (item.axis, item.shaper_type, actual_type)
                )
            if abs(actual_frequency - item.frequency) > 0.0005:
                raise RuntimeError(
                    "%s-axis frequency readback mismatch: expected %.6f, got %.6f"
                    % (item.axis, item.frequency, actual_frequency)
                )
            if abs(actual_damping - item.damping_ratio) > 0.0000005:
                raise RuntimeError(
                    "%s-axis damping readback mismatch: expected %.6f, got %.6f"
                    % (item.axis, item.damping_ratio, actual_damping)
                )

    def apply_temporary(self, selections: Sequence[ShaperSelection]) -> None:
        values: dict[str, str] = {}
        for item in selections:
            self._prove_selection(item)
            suffix = item.axis.upper()
            values["SHAPER_TYPE_" + suffix] = item.shaper_type
            values["SHAPER_FREQ_" + suffix] = "%.6f" % item.frequency
            values["DAMPING_RATIO_" + suffix] = "%.6f" % item.damping_ratio
        if values:
            command = "SET_INPUT_SHAPER " + " ".join(
                "%s=%s" % pair for pair in sorted(values.items())
            )
            self.gcode.run_script_from_command(command)
        self.verify_applied(selections)

    def set_test_square_corner_velocity(self, value: float) -> None:
        target = float(value)
        if not 0.1 <= target <= 50.0:
            raise RuntimeError("test square-corner velocity is outside 0.1..50 mm/s")
        self.gcode.run_script_from_command(
            "SET_VELOCITY_LIMIT SQUARE_CORNER_VELOCITY=%.6f" % target
        )
        eventtime = self.printer.get_reactor().monotonic()
        status = self.printer.lookup_object("toolhead").get_status(eventtime)
        actual = status.get("square_corner_velocity")
        if actual is None or abs(float(actual) - target) > 0.0000005:
            raise RuntimeError(
                "SCV readback mismatch: expected %.6f, got %s"
                % (target, actual if actual is not None else "unavailable")
            )

    def restore(self, snapshot: PrinterSnapshot) -> None:
        errors = []
        try:
            self.apply_temporary(
                (
                    ShaperSelection(
                        snapshot.shaper_type_x,
                        snapshot.shaper_freq_x,
                        "X",
                        snapshot.damping_ratio_x,
                    ),
                    ShaperSelection(
                        snapshot.shaper_type_y,
                        snapshot.shaper_freq_y,
                        "Y",
                        snapshot.damping_ratio_y,
                    ),
                )
            )
        except BaseException as error:
            errors.append(error)
        velocity = (
            "SET_VELOCITY_LIMIT VELOCITY=%.6f ACCEL=%.6f "
            "SQUARE_CORNER_VELOCITY=%.6f"
        ) % (
            snapshot.max_velocity,
            snapshot.max_accel,
            snapshot.square_corner_velocity,
        )
        if snapshot.minimum_cruise_ratio is not None:
            velocity += " MINIMUM_CRUISE_RATIO=%.6f" % snapshot.minimum_cruise_ratio
        try:
            self.gcode.run_script_from_command(velocity)
            self._verify_velocity(snapshot)
        except BaseException as error:
            errors.append(error)
        if errors:
            raise RuntimeError(
                "failed to restore printer state: %s" % "; ".join(str(error) for error in errors)
            )

    def _verify_velocity(self, snapshot: PrinterSnapshot) -> None:
        eventtime = self.printer.get_reactor().monotonic()
        toolhead = self.printer.lookup_object("toolhead", None)
        if toolhead is None:
            raise RuntimeError("Klipper toolhead status is unavailable for rollback verification")
        status = toolhead.get_status(eventtime)
        expected = {
            "max_velocity": snapshot.max_velocity,
            "max_accel": snapshot.max_accel,
            "square_corner_velocity": snapshot.square_corner_velocity,
        }
        if snapshot.minimum_cruise_ratio is not None:
            expected["minimum_cruise_ratio"] = snapshot.minimum_cruise_ratio
        for name, value in expected.items():
            if name not in status or abs(float(status[name]) - float(value)) > 0.0000005:
                raise RuntimeError(
                    "rollback readback mismatch for %s: expected %.6f, got %s"
                    % (name, value, status.get(name, "unavailable"))
                )

    def stage(self, selections: Sequence[ShaperSelection]) -> None:
        configfile = self.printer.lookup_object("configfile")
        for item in selections:
            self._prove_selection(item)
            suffix = item.axis.lower()
            configfile.set("input_shaper", "shaper_type_" + suffix, item.shaper_type)
            configfile.set("input_shaper", "shaper_freq_" + suffix, "%.6f" % item.frequency)
            configfile.set(
                "input_shaper", "damping_ratio_" + suffix, "%.6f" % item.damping_ratio
            )

    def respond(self, message: str) -> None:
        self.gcode.respond_info(message)


def selection_from_mapping(value: Mapping[str, Any]) -> ShaperSelection:
    if "damping_ratio" not in value:
        raise ValueError("analysis selection is missing measured design damping")
    return ShaperSelection(
        shaper_type=str(value["shaper_type"]),
        frequency=float(value.get("frequency", value.get("frequency_hz"))),
        axis=str(value["axis"]),
        damping_ratio=float(value["damping_ratio"]),
    )
