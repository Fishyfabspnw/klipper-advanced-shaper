"""Klipper boundary adapters.

Only this module knows about Klipper object names and g-code.  The controller is
kept dependency-injected and therefore importable without Klipper installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, Sequence


@dataclass(frozen=True)
class PrinterSnapshot:
    shaper_type_x: str
    shaper_freq_x: float
    shaper_type_y: str
    shaper_freq_y: float
    max_velocity: float
    max_accel: float
    square_corner_velocity: float
    damping_ratio_x: float = 0.1
    damping_ratio_y: float = 0.1
    minimum_cruise_ratio: Optional[float] = None


@dataclass(frozen=True)
class ShaperSelection:
    shaper_type: str
    frequency: float
    axis: str
    damping_ratio: float = 0.1

    def __post_init__(self) -> None:
        axis = self.axis.upper()
        kind = self.shaper_type.lower()
        if axis not in {"X", "Y"}:
            raise ValueError("axis must be X or Y")
        if kind not in {"zv", "mzv", "ei", "2hump_ei", "3hump_ei", "zvd"}:
            raise ValueError("unsupported shaper type: %s" % self.shaper_type)
        if not 1.0 <= float(self.frequency) <= 200.0:
            raise ValueError("shaper frequency must be between 1 and 200 Hz")
        if not 0.01 <= float(self.damping_ratio) <= 1.0:
            raise ValueError("damping ratio must be between 0.01 and 1.0")
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "shaper_type", kind)


class PrinterAdapter(Protocol):
    def preflight(self, axes: Sequence[str]) -> None: ...
    def snapshot(self) -> PrinterSnapshot: ...
    def capture(self, axis: str, repeat: int, validation: bool = False) -> Any: ...
    def apply_temporary(self, selections: Sequence[ShaperSelection]) -> None: ...
    def restore(self, snapshot: PrinterSnapshot) -> None: ...
    def stage(self, selections: Sequence[ShaperSelection]) -> None: ...
    def respond(self, message: str) -> None: ...


class KlipperPrinterAdapter:
    """Thin adapter around stable Klippy objects and native g-code commands.

    A capture provider is intentionally injected/registered separately.  It is
    responsible for using Klipper's accelerometer/resonance APIs and must not
    move the toolhead outside the bounded calibration command.
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self.printer = config.get_printer()
        self.gcode = self.printer.lookup_object("gcode")
        self.capture_provider = self.printer.lookup_object("advanced_shaper_capture", None)

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

    def snapshot(self) -> PrinterSnapshot:
        eventtime = self.printer.get_reactor().monotonic()
        toolhead = self.printer.lookup_object("toolhead")
        velocity = toolhead.get_status(eventtime)
        input_shaper = self.printer.lookup_object("input_shaper")
        if hasattr(input_shaper, "get_status"):
            shaping = input_shaper.get_status(eventtime)
        else:
            shaping = {}
            for shaper in input_shaper.get_shapers():
                suffix = shaper.axis.lower()
                for key, value in shaper.params.get_status().items():
                    shaping[key + "_" + suffix] = value
        return PrinterSnapshot(
            shaper_type_x=str(shaping.get("shaper_type_x", "mzv")),
            shaper_freq_x=float(shaping.get("shaper_freq_x", 0.0)),
            shaper_type_y=str(shaping.get("shaper_type_y", "mzv")),
            shaper_freq_y=float(shaping.get("shaper_freq_y", 0.0)),
            damping_ratio_x=float(shaping.get("damping_ratio_x", 0.1)),
            damping_ratio_y=float(shaping.get("damping_ratio_y", 0.1)),
            max_velocity=float(velocity["max_velocity"]),
            max_accel=float(velocity["max_accel"]),
            square_corner_velocity=float(velocity["square_corner_velocity"]),
            minimum_cruise_ratio=(
                float(velocity["minimum_cruise_ratio"])
                if "minimum_cruise_ratio" in velocity
                else None
            ),
        )

    def capture(self, axis: str, repeat: int, validation: bool = False) -> Any:
        return self.capture_provider.capture(axis=axis, repeat=repeat, validation=validation)

    def apply_temporary(self, selections: Sequence[ShaperSelection]) -> None:
        values: dict[str, str] = {}
        for item in selections:
            suffix = item.axis.upper()
            values["SHAPER_TYPE_" + suffix] = item.shaper_type
            values["SHAPER_FREQ_" + suffix] = "%.6f" % item.frequency
            values["DAMPING_RATIO_" + suffix] = "%.6f" % item.damping_ratio
        command = "SET_INPUT_SHAPER " + " ".join("%s=%s" % pair for pair in sorted(values.items()))
        self.gcode.run_script_from_command(command)

    def restore(self, snapshot: PrinterSnapshot) -> None:
        errors = []
        try:
            self.gcode.run_script_from_command(
                "SET_INPUT_SHAPER SHAPER_TYPE_X=%s SHAPER_FREQ_X=%.6f DAMPING_RATIO_X=%.6f "
                "SHAPER_TYPE_Y=%s SHAPER_FREQ_Y=%.6f DAMPING_RATIO_Y=%.6f"
                % (
                    snapshot.shaper_type_x,
                    snapshot.shaper_freq_x,
                    snapshot.damping_ratio_x,
                    snapshot.shaper_type_y,
                    snapshot.shaper_freq_y,
                    snapshot.damping_ratio_y,
                )
            )
        except BaseException as error:
            errors.append(error)
        velocity = "SET_VELOCITY_LIMIT VELOCITY=%.6f ACCEL=%.6f" % (
            snapshot.max_velocity,
            snapshot.max_accel,
        )
        if snapshot.minimum_cruise_ratio is not None:
            velocity += " MINIMUM_CRUISE_RATIO=%.6f" % snapshot.minimum_cruise_ratio
        try:
            self.gcode.run_script_from_command(velocity)
        except BaseException as error:
            errors.append(error)
        if errors:
            raise RuntimeError(
                "failed to restore printer state: %s" % "; ".join(str(error) for error in errors)
            )

    def stage(self, selections: Sequence[ShaperSelection]) -> None:
        configfile = self.printer.lookup_object("configfile")
        for item in selections:
            suffix = item.axis.lower()
            configfile.set("input_shaper", "shaper_type_" + suffix, item.shaper_type)
            configfile.set("input_shaper", "shaper_freq_" + suffix, "%.3f" % item.frequency)
            configfile.set("input_shaper", "damping_ratio_" + suffix, "%.4f" % item.damping_ratio)

    def respond(self, message: str) -> None:
        self.gcode.respond_info(message)


def selection_from_mapping(value: Mapping[str, Any]) -> ShaperSelection:
    return ShaperSelection(
        shaper_type=str(value["shaper_type"]),
        frequency=float(value.get("frequency", value.get("frequency_hz"))),
        axis=str(value["axis"]),
        damping_ratio=float(value.get("damping_ratio", 0.1)),
    )
