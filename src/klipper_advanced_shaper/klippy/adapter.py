"""Klipper boundary adapters.

Only this module knows about Klipper object names and g-code.  The controller is
kept dependency-injected and therefore importable without Klipper installed.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import math
import re
import textwrap
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
        object.__setattr__(self, "frequency", float(self.frequency))
        object.__setattr__(self, "damping_ratio", float(self.damping_ratio))

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
    def preflight_transient(self, axes: Sequence[str]) -> Mapping[str, Any]: ...
    def build_shaper_models(
        self, selections: Sequence[ShaperSelection]
    ) -> Mapping[str, Mapping[str, Any]]: ...
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
    def capture_transient(
        self,
        axis: str,
        repeat: int,
        plan: Mapping[str, Any],
        max_accel_mm_s2: float,
        speed_mm_s: float,
        post_command_guard_seconds: float,
    ) -> Any: ...
    def apply_temporary(self, selections: Sequence[ShaperSelection]) -> None: ...
    def set_test_square_corner_velocity(self, value: float) -> None: ...
    def set_test_max_accel(self, value: float) -> None: ...
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
        self._executor_single_pass_proof: Optional[Mapping[str, Any]] = None
        self._capture_native_shapers: Optional[Sequence[str]] = None
        self.last_capability: Optional[Mapping[str, Any]] = None
        self.last_live_python_pulse_proof: Optional[Mapping[str, Any]] = None

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
        if limit == 10:
            single_pass_patterns = {
                "p_ind_member": r"\bint\s+num_pulses\s*,\s*p_ind\s*;",
                "p_ind_assignment": r"\bsp->p_ind\s*=\s*i\s*;",
                "previous_move_loop": r"\bsp->p_ind\s*-\s*1",
                "next_move_loop": r"\bi\s*=\s*sp->p_ind\s*;",
            }
            missing = [
                name
                for name, pattern in single_pass_patterns.items()
                if re.search(pattern, content) is None
            ]
            if missing:
                raise RuntimeError(
                    "cannot prove installed Klipper 10-pulse single-pass executor: %s"
                    % ", ".join(missing)
                )
            self._executor_single_pass_proof = {
                "passed": True,
                "feature": "ten_pulse_single_pass_executor",
                "method": "read_only_c_source_signature",
                "source_file": str(source),
                "signatures": sorted(single_pass_patterns),
            }
        self._executor_pulse_limit = limit
        return limit

    def _prove_parameter_frequency_assignment(self) -> Mapping[str, Any]:
        """Prove the installed Klipper contains the post-April frequency fix.

        The initial parameterized-shaper implementation validated a requested
        frequency but did not assign it to ``self.shaper_freq``.  Exact status
        readback catches that after a temporary apply; this read-only source
        proof moves the incompatibility failure ahead of experimental motion.
        """
        input_shaper = self.printer.lookup_object("input_shaper", None)
        get_shapers = getattr(input_shaper, "get_shapers", None)
        if get_shapers is None:
            raise RuntimeError(
                "installed Klipper cannot expose input-shaper parameter objects"
            )
        shapers = list(get_shapers())
        if not shapers:
            raise RuntimeError("installed Klipper exposes no input-shaper axes")
        checked: dict[type[Any], dict[str, Any]] = {}
        for shaper in shapers:
            params = getattr(shaper, "params", None)
            update = getattr(type(params), "update", None) if params is not None else None
            if update is None:
                raise RuntimeError(
                    "installed Klipper input-shaper parameters lack update()"
                )
            parameter_type = type(params)
            if parameter_type in checked:
                continue
            try:
                source = textwrap.dedent(inspect.getsource(update))
                tree = ast.parse(source)
            except (OSError, TypeError, IndentationError, SyntaxError) as error:
                raise RuntimeError(
                    "cannot inspect installed Klipper input-shaper frequency update"
                ) from error
            fixed = any(
                isinstance(node, (ast.Assign, ast.AnnAssign))
                and isinstance(getattr(node, "value", None), ast.Name)
                and node.value.id == "shaper_freq"
                and any(
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and target.attr == "shaper_freq"
                    for target in (
                        node.targets if isinstance(node, ast.Assign) else [node.target]
                    )
                )
                for node in ast.walk(tree)
            )
            if not fixed:
                raise RuntimeError(
                    "installed Klipper lacks the SET_INPUT_SHAPER frequency-assignment fix"
                )
            checked[parameter_type] = {
                "class": parameter_type.__name__,
                "module": str(getattr(update, "__module__", "")),
                "source_file": inspect.getsourcefile(update),
            }
        return {
            "passed": True,
            "feature": "set_input_shaper_frequency_assignment",
            "method": "read_only_ast_source_proof",
            "parameter_types": list(checked.values()),
        }

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
        shaping = self._shaping_status()
        if shaping.get("_advanced_shaper_raw_params") is not True:
            raise RuntimeError(
                "installed Klipper cannot provide raw input-shaper parameters "
                "required for exact experimental readback and rollback"
            )
        exact_status: dict[str, Any] = {
            "source": "input_shaper.get_shapers raw params"
        }
        for axis in ("X", "Y"):
            suffix = axis.lower()
            try:
                identifier = parse_shaper_identifier(
                    str(shaping["shaper_type_" + suffix]),
                    allow_parameterized=True,
                ).canonical
                frequency = float(shaping["shaper_freq_" + suffix])
                damping = float(shaping["damping_ratio_" + suffix])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    "installed Klipper raw %s-axis shaper status is malformed" % axis
                ) from error
            if (
                not math.isfinite(frequency)
                or not 1.0 <= frequency <= 200.0
                or not math.isfinite(damping)
                or not 0.0 <= damping < 1.0
            ):
                raise RuntimeError(
                    "installed Klipper raw %s-axis shaper status is unsafe" % axis
                )
            exact_status[axis] = {
                "canonical_shaper": identifier,
                "frequency_hz": frequency,
                "damping_ratio": damping,
            }
        frequency_assignment_proof = self._prove_parameter_frequency_assignment()
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
            "executor_single_pass_proof": getattr(
                self, "_executor_single_pass_proof", None
            ),
            "proofs": proofs,
            "native_proof": native_proof,
            "exact_status_readback": exact_status,
            "frequency_assignment_proof": frequency_assignment_proof,
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

    def build_shaper_models(
        self, selections: Sequence[ShaperSelection]
    ) -> Mapping[str, Mapping[str, Any]]:
        """Realize exact selections with the installed Klipper shaper source.

        These pulse models are used only for a conservative theoretical screen.
        They neither read back the live C executor nor replace held-out validation.
        """
        module = self._load_shaper_defs()
        executor_limit = self._get_executor_pulse_limit()
        try:
            get_config = module.get_shaper_cfg
            initialize = module.init_shaper
            get_signature = inspect.signature(get_config)
            init_signature = inspect.signature(initialize)
            get_signature.bind("mzv")
            init_signature.bind("mzv", 60.0, 0.1)
        except (AttributeError, TypeError, ValueError) as error:
            raise RuntimeError(
                "installed Klipper shaper model API is unavailable or incompatible"
            ) from error

        models: dict[str, Mapping[str, Any]] = {}
        for selection in selections:
            identifier = parse_shaper_identifier(
                selection.shaper_type, allow_parameterized=True
            )
            try:
                config = get_config(identifier.canonical)
                pulses = initialize(
                    identifier.canonical,
                    float(selection.frequency),
                    float(selection.damping_ratio),
                )
                amplitudes, times = pulses
                amplitude_values = [float(value) for value in amplitudes]
                time_values = [float(value) for value in times]
            except Exception as error:  # Klipper uses version-specific errors.
                raise RuntimeError(
                    "installed Klipper could not realize exact %s-axis reference %s"
                    % (selection.axis, identifier.canonical)
                ) from error
            config_name = str(getattr(config, "name", "")).lower()
            if config is None or config_name != identifier.family:
                raise RuntimeError(
                    "installed Klipper did not resolve exact shaper family %s"
                    % identifier.family
                )
            if (
                not 2 <= len(amplitude_values) <= executor_limit
                or len(amplitude_values) != len(time_values)
                or any(not math.isfinite(value) for value in amplitude_values)
                or any(value < -1e-5 for value in amplitude_values)
                or any(not math.isfinite(value) for value in time_values)
                or any(
                    time_values[index] > time_values[index + 1]
                    for index in range(len(time_values) - 1)
                )
            ):
                raise RuntimeError(
                    "installed Klipper returned an unsafe pulse model for %s"
                    % identifier.canonical
                )
            amplitude_sum = float(sum(amplitude_values))
            if not math.isfinite(amplitude_sum) or amplitude_sum <= 0.0:
                raise RuntimeError(
                    "installed Klipper returned a non-normalizable pulse model for %s"
                    % identifier.canonical
                )
            models[selection.axis] = {
                "axis": selection.axis,
                "shaper_type": identifier.canonical,
                "family": identifier.family,
                "frequency_hz": float(selection.frequency),
                "design_damping_ratio": float(selection.damping_ratio),
                "pulse_count": len(amplitude_values),
                "pulse_amplitudes_normalized": [
                    value / amplitude_sum for value in amplitude_values
                ],
                "pulse_times_s": time_values,
                "source": "installed_klipper_shaper_defs.init_shaper",
                "source_module": str(getattr(module, "__name__", type(module).__name__)),
                "source_file": (
                    Path(str(module.__file__)).name
                    if getattr(module, "__file__", None)
                    else None
                ),
                "api_signature_verified": True,
                "executor_pulse_limit": executor_limit,
                "theoretical_model_only": True,
                "live_c_executor_readback": False,
            }
        if set(models) != {selection.axis for selection in selections}:
            raise RuntimeError("exact installed-Klipper shaper models are incomplete")
        return models

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

    def preflight_transient(self, axes: Sequence[str]) -> Mapping[str, Any]:
        if self.capture_provider is None or not hasattr(
            self.capture_provider, "preflight_transient"
        ):
            raise RuntimeError(
                "capture provider cannot prove finite transient validation support"
            )
        return self.capture_provider.preflight_transient(tuple(axes))

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

    def capture_transient(
        self,
        axis: str,
        repeat: int,
        plan: Mapping[str, Any],
        max_accel_mm_s2: float,
        speed_mm_s: float,
        post_command_guard_seconds: float,
    ) -> Any:
        if self.capture_provider is None or not hasattr(
            self.capture_provider, "capture_transient"
        ):
            raise RuntimeError(
                "capture provider cannot perform finite transient validation"
            )
        return self.capture_provider.capture_transient(
            axis=axis,
            repeat=repeat,
            plan=plan,
            max_accel_mm_s2=max_accel_mm_s2,
            speed_mm_s=speed_mm_s,
            post_command_guard_seconds=post_command_guard_seconds,
        )

    def _shaping_status(self, eventtime: Optional[float] = None) -> dict[str, Any]:
        input_shaper = self.printer.lookup_object("input_shaper", None)
        if input_shaper is None:
            raise RuntimeError("Klipper input_shaper status is unavailable")
        if eventtime is None:
            eventtime = self.printer.get_reactor().monotonic()
        # Current upstream AxisInputShaper.params.get_status() deliberately
        # formats frequency to three decimals.  Prefer the raw parameter
        # attributes so a snapshot can be restored without quantizing an
        # existing value such as 75.6004 Hz.
        get_shapers = getattr(input_shaper, "get_shapers", None)
        if callable(get_shapers):
            shaping: dict[str, Any] = {}
            raw = True
            for shaper in get_shapers():
                suffix = str(shaper.axis).lower()
                params = shaper.params
                if all(
                    hasattr(params, name)
                    for name in ("shaper_type", "shaper_freq", "damping_ratio")
                ):
                    shaping["shaper_type_" + suffix] = params.shaper_type
                    shaping["shaper_freq_" + suffix] = params.shaper_freq
                    shaping["damping_ratio_" + suffix] = params.damping_ratio
                else:
                    raw = False
                    for key, value in params.get_status().items():
                        shaping[key + "_" + suffix] = value
            shaping["_advanced_shaper_raw_params"] = raw
            return shaping
        if hasattr(input_shaper, "get_status"):
            shaping = dict(input_shaper.get_status(eventtime))
            shaping["_advanced_shaper_raw_params"] = False
            return shaping
        shaping: dict[str, Any] = {}
        raise RuntimeError("Klipper input_shaper status API is unavailable")

    def verify_applied(self, selections: Sequence[ShaperSelection]) -> None:
        shaping = self._shaping_status()
        exact = shaping.get("_advanced_shaper_raw_params") is True
        frequency_tolerance = 5e-12 if exact else 0.0005
        damping_tolerance = 5e-12 if exact else 0.0000005
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
            if abs(actual_frequency - item.frequency) > frequency_tolerance:
                raise RuntimeError(
                    "%s-axis frequency readback mismatch: expected %.6f, got %.6f"
                    % (item.axis, item.frequency, actual_frequency)
                )
            if abs(actual_damping - item.damping_ratio) > damping_tolerance:
                raise RuntimeError(
                    "%s-axis damping readback mismatch: expected %.6f, got %.6f"
                    % (item.axis, item.damping_ratio, actual_damping)
                )

    def verify_live_python_pulses(
        self, selections: Sequence[ShaperSelection]
    ) -> Mapping[str, Any]:
        """Verify live AxisInputShaper Python pulse state after SET.

        Klipper has no C-struct getter.  This is therefore a strong no-motion
        Python-layer proof, not a claim that C executor memory was read back.
        """
        input_shaper = self.printer.lookup_object("input_shaper", None)
        get_shapers = getattr(input_shaper, "get_shapers", None)
        if get_shapers is None or not callable(get_shapers):
            raise RuntimeError("Klipper input_shaper.get_shapers API is unavailable")
        try:
            inspect.signature(get_shapers).bind()
            shapers = list(get_shapers())
        except (TypeError, ValueError) as error:
            raise RuntimeError("unsupported Klipper input_shaper.get_shapers API") from error
        wrappers = getattr(
            input_shaper, "input_shaper_stepper_kinematics", None
        )
        if not isinstance(wrappers, (list, tuple)) or not wrappers:
            raise RuntimeError(
                "Klipper input shaper has no active stepper-kinematics wrappers"
            )
        module = self._load_shaper_defs()
        initialize = getattr(module, "init_shaper", None)
        if initialize is None or not callable(initialize):
            raise RuntimeError("installed Klipper init_shaper API is unavailable")
        try:
            inspect.signature(initialize).bind("mzv", 60.0, 0.1)
        except (TypeError, ValueError) as error:
            raise RuntimeError("unsupported installed Klipper init_shaper API") from error

        axes: dict[str, Any] = {}
        for selection in selections:
            suffix = selection.axis.lower()
            matches = [item for item in shapers if str(getattr(item, "axis", "")).lower() == suffix]
            if len(matches) != 1:
                raise RuntimeError(
                    "%s-axis live Python shaper is not uniquely available" % selection.axis
                )
            shaper = matches[0]
            enabled = getattr(shaper, "is_enabled", None)
            if enabled is None or not callable(enabled):
                raise RuntimeError(
                    "%s-axis live Python shaper lacks is_enabled()" % selection.axis
                )
            try:
                inspect.signature(enabled).bind()
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    "%s-axis live Python shaper has unsupported is_enabled API"
                    % selection.axis
                ) from error
            required = ("n", "A", "T", "saved", "params")
            missing = [name for name in required if not hasattr(shaper, name)]
            if missing:
                raise RuntimeError(
                    "%s-axis live Python shaper is missing pulse fields: %s"
                    % (selection.axis, ",".join(missing))
                )
            if not enabled() or shaper.saved is not None:
                raise RuntimeError(
                    "%s-axis input shaping is not actively enabled" % selection.axis
                )
            params = shaper.params
            try:
                actual_identifier = parse_shaper_identifier(
                    str(params.shaper_type), allow_parameterized=True
                ).canonical
                actual_frequency = float(params.shaper_freq)
                actual_damping = float(params.damping_ratio)
                actual_n = int(shaper.n)
                actual_amplitudes = [float(value) for value in shaper.A]
                actual_times = [float(value) for value in shaper.T]
                commanded_frequency = float(selection.frequency)
                commanded_damping = float(selection.damping_ratio)
                expected_amplitudes, expected_times = initialize(
                    selection.shaper_type,
                    commanded_frequency,
                    commanded_damping,
                )
                expected_amplitudes = [float(value) for value in expected_amplitudes]
                expected_times = [float(value) for value in expected_times]
            except Exception as error:  # Klipper raises version-specific errors.
                raise RuntimeError(
                    "cannot inspect %s-axis live Python pulse state" % selection.axis
                ) from error
            if actual_identifier != selection.shaper_type:
                raise RuntimeError(
                    "%s-axis live Python shaper identifier mismatch" % selection.axis
                )
            if abs(actual_frequency - commanded_frequency) > 0.0000000005:
                raise RuntimeError(
                    "%s-axis live Python shaper frequency mismatch" % selection.axis
                )
            if abs(actual_damping - commanded_damping) > 0.0000000005:
                raise RuntimeError(
                    "%s-axis live Python shaper damping mismatch" % selection.axis
                )
            if (
                actual_n <= 0
                or actual_n != len(actual_amplitudes)
                or actual_n != len(actual_times)
                or actual_n != len(expected_amplitudes)
                or actual_n != len(expected_times)
            ):
                raise RuntimeError(
                    "%s-axis live Python pulse count mismatch" % selection.axis
                )
            if any(
                not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
                for actual, expected in zip(actual_amplitudes, expected_amplitudes)
            ):
                raise RuntimeError(
                    "%s-axis live Python pulse amplitude mismatch" % selection.axis
                )
            if any(
                not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
                for actual, expected in zip(actual_times, expected_times)
            ):
                raise RuntimeError(
                    "%s-axis live Python pulse timing mismatch" % selection.axis
                )
            axes[selection.axis] = {
                "axis": selection.axis,
                "canonical_shaper": selection.shaper_type,
                "frequency_hz": actual_frequency,
                "damping_ratio": actual_damping,
                "pulse_count": actual_n,
                "pulse_span_seconds": float(max(actual_times) - min(actual_times)),
                "post_command_guard_seconds": float(
                    max(actual_times) - min(actual_times) + 0.020
                ),
                "enabled": True,
                "saved_disabled_state_present": False,
                "amplitudes_match_installed_init_shaper": True,
                "times_match_installed_init_shaper": True,
            }
        proof: dict[str, Any] = {
            "passed": True,
            "layer": "live_klippy_axis_input_shaper_python_state",
            "source": "input_shaper.get_shapers plus installed shaper_defs.init_shaper",
            "active_axis_verified": True,
            "python_axis_state_verified": True,
            "input_shaper_kinematics_wrapper_presence_verified": True,
            "active_c_attachment_verified": False,
            "input_shaper_stepper_kinematics_count": len(wrappers),
            "live_c_executor_readback": False,
            "c_struct_state_claimed": False,
            "axes": axes,
        }
        self.last_live_python_pulse_proof = proof
        return proof

    def apply_temporary(self, selections: Sequence[ShaperSelection]) -> None:
        values: dict[str, str] = {}
        for item in selections:
            self._prove_selection(item)
            suffix = item.axis.upper()
            values["SHAPER_TYPE_" + suffix] = item.shaper_type
            values["SHAPER_FREQ_" + suffix] = repr(float(item.frequency))
            values["DAMPING_RATIO_" + suffix] = repr(float(item.damping_ratio))
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

    def set_test_max_accel(self, value: float) -> None:
        target = float(value)
        if not 1.0 <= target <= 5000.0:
            raise RuntimeError("transient test max_accel is outside 1..5000 mm/s^2")
        self.gcode.run_script_from_command(
            "SET_VELOCITY_LIMIT ACCEL=%.6f" % target
        )
        eventtime = self.printer.get_reactor().monotonic()
        status = self.printer.lookup_object("toolhead").get_status(eventtime)
        actual = status.get("max_accel")
        if actual is None or abs(float(actual) - target) > 0.0005:
            raise RuntimeError(
                "transient max_accel readback mismatch: expected %.6f, got %s"
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
        velocity = "SET_VELOCITY_LIMIT VELOCITY=%s ACCEL=%s SQUARE_CORNER_VELOCITY=%s" % (
            repr(float(snapshot.max_velocity)),
            repr(float(snapshot.max_accel)),
            repr(float(snapshot.square_corner_velocity)),
        )
        if snapshot.minimum_cruise_ratio is not None:
            velocity += " MINIMUM_CRUISE_RATIO=%s" % repr(
                float(snapshot.minimum_cruise_ratio)
            )
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
            if name not in status or not math.isclose(
                float(status[name]), float(value), rel_tol=0.0, abs_tol=5e-12
            ):
                raise RuntimeError(
                    "rollback readback mismatch for %s: expected %s, got %s"
                    % (
                        name,
                        repr(float(value)),
                        status.get(name, "unavailable"),
                    )
                )

    def stage(self, selections: Sequence[ShaperSelection]) -> None:
        configfile = self.printer.lookup_object("configfile")
        for item in selections:
            self._prove_selection(item)
            suffix = item.axis.lower()
            configfile.set("input_shaper", "shaper_type_" + suffix, item.shaper_type)
            configfile.set(
                "input_shaper", "shaper_freq_" + suffix,
                repr(float(item.frequency)),
            )
            configfile.set(
                "input_shaper", "damping_ratio_" + suffix,
                repr(float(item.damping_ratio)),
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
