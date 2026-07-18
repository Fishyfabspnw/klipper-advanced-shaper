"""Klippy command surface and fail-closed calibration controller."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from .adapter import (
    KlipperPrinterAdapter,
    PrinterAdapter,
    ShaperSelection,
    selection_from_mapping,
)
from .state import CalibrationCancelled, CalibrationState, StateMachine

SUPPORTED_PROFILES = {"quality", "balanced", "performance"}


@dataclass(frozen=True)
class CalibrationResult:
    result_id: str
    selections: tuple[ShaperSelection, ...]
    report: Mapping[str, Any]


class AdvancedInputShaper:
    """Controller backing the ``[advanced_input_shaper]`` Klippy extra."""

    def __init__(
        self,
        config: Any = None,
        *,
        adapter: Optional[PrinterAdapter] = None,
        analyzer: Optional[Callable[..., Mapping[str, Any]]] = None,
        id_factory: Optional[Callable[[], str]] = None,
        artifact_writer: Any = None,
    ) -> None:
        if adapter is None:
            if config is None:
                raise ValueError("config or adapter is required")
            adapter = KlipperPrinterAdapter(config)
        self.adapter = adapter
        self.analyzer = analyzer or self._load_default_analyzer()
        self.id_factory = id_factory or (lambda: uuid.uuid4().hex[:12])
        self.artifact_writer = artifact_writer
        self.worker = None
        self.minimum_max_accel = {"X": 0.0, "Y": 0.0}
        self.machine = StateMachine()
        self.results: dict[str, CalibrationResult] = {}
        self.current_result_id: Optional[str] = None
        if config is not None:
            self.minimum_max_accel = {
                "X": config.getfloat("minimum_max_accel_x", 0.0, minval=0.0),
                "Y": config.getfloat("minimum_max_accel_y", 0.0, minval=0.0),
            }
            from .worker import SupervisedWorker

            self.worker = SupervisedWorker(
                config.get_printer().get_reactor(),
                timeout=config.getfloat("analysis_timeout", 600.0, above=0.0),
                memory_mb=config.getint("worker_memory_mb", 1536, minval=256),
                cpu_seconds=config.getint("worker_cpu_seconds", 300, minval=10),
            )
            if self.artifact_writer is None:
                from klipper_advanced_shaper.artifacts import ArtifactWriter

                self.artifact_writer = ArtifactWriter(
                    config.get(
                        "result_folder",
                        "~/printer_data/config/AdvancedShaper_results",
                    ),
                    config.getboolean("keep_raw_data", True),
                )
            printer = config.get_printer()
            if printer.lookup_object("advanced_shaper_capture", None) is None:
                from .capture import NativeResonanceCaptureProvider

                printer.add_object(
                    "advanced_shaper_capture", NativeResonanceCaptureProvider(config)
                )
            self._register_commands(config.get_printer().lookup_object("gcode"))

    @staticmethod
    def _load_default_analyzer() -> Callable[..., Mapping[str, Any]]:
        def unavailable(**_: Any) -> Mapping[str, Any]:
            raise RuntimeError("analysis engine is unavailable")

        try:
            from klipper_advanced_shaper.analysis import analyze_calibration

            return analyze_calibration
        except (ImportError, AttributeError):
            return unavailable

    def _register_commands(self, gcode: Any) -> None:
        commands = {
            "ADV_SHAPER_CALIBRATE": self.cmd_ADV_SHAPER_CALIBRATE,
            "ADV_SHAPER_STATUS": self.cmd_ADV_SHAPER_STATUS,
            "ADV_SHAPER_CANCEL": self.cmd_ADV_SHAPER_CANCEL,
            "ADV_SHAPER_APPLY": self.cmd_ADV_SHAPER_APPLY,
            "ADV_SHAPER_STAGE": self.cmd_ADV_SHAPER_STAGE,
        }
        for name, callback in commands.items():
            gcode.register_command(name, callback)

    def calibrate(
        self,
        axes: Sequence[str],
        profile: str = "balanced",
        repeats: int = 3,
        validate: bool = True,
    ) -> CalibrationResult:
        normalized_axes = tuple(dict.fromkeys(axis.upper() for axis in axes))
        if not normalized_axes or any(axis not in {"X", "Y"} for axis in normalized_axes):
            raise ValueError("axes must contain X and/or Y")
        profile = profile.lower()
        if profile not in SUPPORTED_PROFILES:
            raise ValueError("unsupported profile: %s" % profile)
        if not 1 <= repeats <= 20:
            raise ValueError("repeats must be between 1 and 20")

        self.machine.begin()
        snapshot = None
        result = None
        captures: dict[str, list[Any]] = {axis: [] for axis in normalized_axes}
        try:
            self.adapter.preflight(normalized_axes)
            self.machine.checkpoint()
            snapshot = self.adapter.snapshot()
            self.machine.transition(CalibrationState.BASELINE_CAPTURE)
            for axis in normalized_axes:
                for repeat in range(repeats):
                    self.machine.checkpoint()
                    captures[axis].append(self.adapter.capture(axis, repeat, False))

            self.machine.transition(CalibrationState.ANALYSIS)
            self.machine.checkpoint()
            report = self._invoke(
                self.analyzer,
                captures=captures,
                axes=normalized_axes,
                profile=profile,
                snapshot=snapshot,
            )
            if report.get("abstain"):
                raise RuntimeError("analysis abstained: %s" % report.get("reason", "quality gate"))
            for axis in normalized_axes:
                target = self.minimum_max_accel[axis]
                details = report.get("axes", {}).get(axis, {})
                selected_name = details.get("selected")
                selected = next(
                    (
                        item
                        for item in details.get("candidates", [])
                        if item.get("name") == selected_name
                    ),
                    None,
                )
                if target and (selected is None or float(selected["max_accel"]) <= target):
                    raise RuntimeError(
                        "%s candidate did not exceed required %.0f mm/s^2" % (axis, target)
                    )
            selections = tuple(selection_from_mapping(value) for value in report["selections"])
            selected_axes = {item.axis for item in selections}
            if selected_axes != set(normalized_axes) or len(selections) != len(normalized_axes):
                raise RuntimeError("analysis did not return exactly one selection per axis")

            if validate:
                self.machine.transition(CalibrationState.TEMPORARY_VALIDATION)
                reference = tuple(
                    ShaperSelection(
                        getattr(snapshot, "shaper_type_" + axis.lower()),
                        getattr(snapshot, "shaper_freq_" + axis.lower()),
                        axis,
                        getattr(snapshot, "damping_ratio_" + axis.lower()),
                    )
                    for axis in normalized_axes
                )
                self.adapter.apply_temporary(reference)
                held_out: dict[str, list[Any]] = {axis: [] for axis in normalized_axes}
                for axis in normalized_axes:
                    for repeat in range(repeats):
                        self.machine.checkpoint()
                        held_out[axis].append(self.adapter.capture(axis, repeat, validation=True))
                self.adapter.apply_temporary(selections)
                validation: dict[str, list[Any]] = {axis: [] for axis in normalized_axes}
                for axis in normalized_axes:
                    for repeat in range(repeats):
                        self.machine.checkpoint()
                        validation[axis].append(self.adapter.capture(axis, repeat, validation=True))
                validation_report = self._invoke(
                    self.analyzer,
                    captures=captures,
                    held_out_captures=held_out,
                    validation_captures=validation,
                    axes=normalized_axes,
                    profile=profile,
                    snapshot=snapshot,
                    prior_report=report,
                )
                if not validation_report.get("validation", {}).get("passed", False):
                    raise RuntimeError(
                        "candidate failed held-out validation: %s"
                        % validation_report.get("validation", {}).get("reason", "attenuation gate")
                    )
                report = dict(report)
                report["reference"] = [
                    {
                        "axis": item.axis,
                        "shaper_type": item.shaper_type,
                        "frequency_hz": item.frequency,
                        "damping_ratio": item.damping_ratio,
                    }
                    for item in reference
                ]
                report["validation"] = validation_report["validation"]

            result = CalibrationResult(self.id_factory(), selections, report)
        except CalibrationCancelled:
            self.machine.cancelled()
            raise
        except BaseException as error:
            self.machine.failed(error)
            raise
        finally:
            if snapshot is not None:
                try:
                    self.adapter.restore(snapshot)
                except BaseException as restore_error:
                    self.machine.failed(restore_error)
                    raise

        # A result is not reviewable until restoration has succeeded.
        assert result is not None
        if self.artifact_writer is not None:
            try:
                raw_groups = {"training": captures}
                if validate:
                    raw_groups.update({"held_out": held_out, "validation": validation})
                artifacts = self._invoke(
                    self.artifact_writer.write,
                    result_id=result.result_id,
                    report=result.report,
                    raw_groups=raw_groups,
                )
                enriched = dict(result.report)
                enriched["artifacts"] = artifacts
                result = CalibrationResult(result.result_id, result.selections, enriched)
            except BaseException as artifact_error:
                self.machine.failed(artifact_error)
                raise
        self.results[result.result_id] = result
        self.current_result_id = result.result_id
        self.machine.transition(CalibrationState.REVIEW)
        return result

    def _invoke(self, function: Callable[..., Any], **arguments: Any) -> Any:
        if self.worker is None:
            return function(**arguments)
        return self.worker.run(function, arguments, self.machine.checkpoint)

    def apply(self, result_id: str) -> CalibrationResult:
        result = self._get_result(result_id)
        if self.machine.state not in {
            CalibrationState.REVIEW,
            CalibrationState.RUNTIME_APPLIED,
            CalibrationState.STAGED,
        }:
            raise RuntimeError("a reviewed result is required before apply")
        self.adapter.apply_temporary(result.selections)
        if self.machine.state == CalibrationState.REVIEW:
            self.machine.transition(CalibrationState.RUNTIME_APPLIED)
        return result

    def stage(self, result_id: str) -> CalibrationResult:
        result = self._get_result(result_id)
        if self.machine.state not in {
            CalibrationState.REVIEW,
            CalibrationState.RUNTIME_APPLIED,
            CalibrationState.STAGED,
        }:
            raise RuntimeError("a reviewed result is required before stage")
        self.adapter.stage(result.selections)
        if self.machine.state != CalibrationState.STAGED:
            self.machine.transition(CalibrationState.STAGED)
        return result

    def cancel(self) -> bool:
        return self.machine.request_cancel()

    def status(self) -> Mapping[str, Any]:
        return {
            "state": self.machine.state.value,
            "result_id": self.current_result_id,
            "cancel_requested": self.machine.cancel_requested,
            "error": self.machine.error,
        }

    def _get_result(self, result_id: str) -> CalibrationResult:
        try:
            return self.results[result_id]
        except KeyError as error:
            raise ValueError("unknown result: %s" % result_id) from error

    def cmd_ADV_SHAPER_CALIBRATE(self, gcmd: Any) -> None:
        axis = gcmd.get("AXIS", "ALL").upper()
        axes = ("X", "Y") if axis == "ALL" else (axis,)
        try:
            result = self.calibrate(
                axes,
                profile=gcmd.get("PROFILE", "balanced"),
                repeats=gcmd.get_int("REPEATS", 3, minval=1, maxval=20),
                validate=bool(gcmd.get_int("VALIDATE", 1, minval=0, maxval=1)),
            )
        except Exception as error:
            raise gcmd.error(str(error))
        gcmd.respond_info("Advanced shaper result ready: %s" % result.result_id)

    def cmd_ADV_SHAPER_STATUS(self, gcmd: Any) -> None:
        gcmd.respond_info(json.dumps(self.status(), sort_keys=True))

    def cmd_ADV_SHAPER_CANCEL(self, gcmd: Any) -> None:
        if self.cancel():
            gcmd.respond_info("Advanced shaper cancellation requested")
        else:
            gcmd.respond_info("No active advanced shaper calibration")

    def cmd_ADV_SHAPER_APPLY(self, gcmd: Any) -> None:
        try:
            result = self.apply(gcmd.get("RESULT"))
        except Exception as error:
            raise gcmd.error(str(error))
        gcmd.respond_info("Applied result %s for this runtime" % result.result_id)

    def cmd_ADV_SHAPER_STAGE(self, gcmd: Any) -> None:
        try:
            result = self.stage(gcmd.get("RESULT"))
        except Exception as error:
            raise gcmd.error(str(error))
        gcmd.respond_info(
            "Staged result %s; run SAVE_CONFIG separately to persist" % result.result_id
        )


def load_config(config: Any) -> AdvancedInputShaper:
    return AdvancedInputShaper(config)
