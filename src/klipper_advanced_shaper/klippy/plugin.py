"""Klippy command surface and fail-closed calibration controller."""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Optional, Sequence

from .adapter import (
    KlipperPrinterAdapter,
    PrinterAdapter,
    ShaperSelection,
    selection_from_mapping,
)
from .excitation import (
    parse_accel_per_hz,
    parse_hz_per_sec,
    parse_square_corner_velocity,
)
from .state import CalibrationCancelled, CalibrationState, StateMachine

SUPPORTED_PROFILES = {
    "quality",
    "balanced",
    "performance",
    "experimental_mzv",
    "adaptive_stock",
}


def _parse_fast_validation(value: Any) -> bool:
    if value in (None, False, 0, "0"):
        return False
    if value in (True, 1, "1"):
        return True
    raise ValueError("FAST_VALIDATION must be numeric 0 or 1")


def _parse_peak_lock(value: Any) -> bool:
    if value in (None, False, 0, "0"):
        return False
    if value in (True, 1, "1"):
        return True
    raise ValueError("PEAK_LOCK must be numeric 0 or 1")


def _profile_max_vibrations(profile: str) -> Optional[float]:
    if profile not in {"experimental_mzv", "adaptive_stock"}:
        return None
    from klipper_advanced_shaper.analysis.selection import PROFILES

    value = PROFILES[profile].maximum_residual
    if value is None:
        raise RuntimeError("experimental profile lacks a native fitting residual limit")
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise RuntimeError(
            "experimental profile native fitting residual limit must be within [0, 1]"
        )
    return threshold


def _analysis_unavailable(**_: Any) -> Mapping[str, Any]:
    """Spawn-picklable fallback used when the analysis package cannot load."""
    raise RuntimeError("analysis engine is unavailable")


def _screen_unavailable(**_: Any) -> Mapping[str, Any]:
    """Spawn-picklable fallback used when the spectral screen cannot load."""
    raise RuntimeError("theoretical spectral non-regression screen is unavailable")


def _model_proof_summary(models: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "axis",
        "shaper_type",
        "frequency_hz",
        "design_damping_ratio",
        "pulse_count",
        "source",
        "source_module",
        "source_file",
        "api_signature_verified",
        "executor_pulse_limit",
        "theoretical_model_only",
        "live_c_executor_readback",
    )
    return [{key: model.get(key) for key in keys} for model in models.values()]


def _assert_models_match_selections(
    models: Mapping[str, Mapping[str, Any]],
    selections: Sequence[ShaperSelection],
    role: str,
) -> None:
    if set(models) != {selection.axis for selection in selections}:
        raise RuntimeError("exact installed-Klipper %s models are incomplete" % role)
    for selection in selections:
        model = models[selection.axis]
        try:
            modeled_type = str(model["shaper_type"])
            modeled_frequency = float(model["frequency_hz"])
            modeled_damping = float(model["design_damping_ratio"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                "%s-axis installed-Klipper %s model is malformed"
                % (selection.axis, role)
            ) from error
        if (
            modeled_type != selection.shaper_type
            or not math.isclose(
                modeled_frequency, selection.frequency, rel_tol=0.0, abs_tol=1e-9
            )
            or not math.isclose(
                modeled_damping, selection.damping_ratio, rel_tol=0.0, abs_tol=1e-9
            )
        ):
            raise RuntimeError(
                "%s-axis installed-Klipper %s model does not exactly match selection"
                % (selection.axis, role)
            )


def _require_transient_preflight(
    proof: Any, axes: Sequence[str]
) -> Mapping[str, Any]:
    if (
        not isinstance(proof, Mapping)
        or proof.get("passed") is not True
        or proof.get("protocol") != "finite_reversal_ringdown_v1"
        or not isinstance(proof.get("plans"), Mapping)
        or set(proof["plans"]) != set(axes)
    ):
        raise RuntimeError(
            "finite transient validation preflight returned malformed proof"
        )
    try:
        speed = float(proof["speed_mm_s"])
        max_accel = float(proof["max_accel_mm_s2"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "finite transient validation preflight returned malformed proof"
        ) from error
    if not math.isfinite(speed) or speed <= 0.0 or not math.isfinite(max_accel) or max_accel <= 0.0:
        raise RuntimeError(
            "finite transient validation preflight returned malformed proof"
        )
    return proof


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
        spectral_screener: Optional[Callable[..., Mapping[str, Any]]] = None,
        id_factory: Optional[Callable[[], str]] = None,
        artifact_writer: Any = None,
        experimental_enabled: Optional[bool] = None,
    ) -> None:
        if adapter is None:
            if config is None:
                raise ValueError("config or adapter is required")
            adapter = KlipperPrinterAdapter(config)
        self.adapter = adapter
        self.analyzer = analyzer or self._load_default_analyzer()
        self.spectral_screener = spectral_screener or self._load_default_screener()
        self.id_factory = id_factory or (lambda: uuid.uuid4().hex[:12])
        self.artifact_writer = artifact_writer
        self.worker = None
        self.minimum_max_accel = {"X": 0.0, "Y": 0.0}
        self.experimental_enabled = bool(experimental_enabled)
        self.machine = StateMachine()
        self.results: dict[str, CalibrationResult] = {}
        self.current_result_id: Optional[str] = None
        self.current_attempt_id: Optional[str] = None
        self.current_attempt_status: Optional[str] = None
        self.current_attempt_artifacts: Optional[Mapping[str, Any]] = None
        self.current_validation_protocol: Optional[Mapping[str, Any]] = None
        if config is not None:
            self.experimental_enabled = config.getboolean(
                "enable_experimental_generalized_mzv", False
            )
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
        try:
            from klipper_advanced_shaper.analysis import analyze_calibration

            return analyze_calibration
        except (ImportError, AttributeError):
            return _analysis_unavailable

    @staticmethod
    def _load_default_screener() -> Callable[..., Mapping[str, Any]]:
        try:
            from klipper_advanced_shaper.analysis import (
                theoretical_spectral_non_regression,
            )

            return theoretical_spectral_non_regression
        except (ImportError, AttributeError):
            return _screen_unavailable

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
        accel_per_hz: Any = None,
        hz_per_sec: Any = None,
        square_corner_velocity: Any = None,
        fast_validation: Any = False,
        peak_lock: Any = False,
    ) -> CalibrationResult:
        normalized_axes = tuple(dict.fromkeys(axis.upper() for axis in axes))
        if not normalized_axes or any(axis not in {"X", "Y"} for axis in normalized_axes):
            raise ValueError("axes must contain X and/or Y")
        profile = profile.lower()
        if profile not in SUPPORTED_PROFILES:
            raise ValueError("unsupported profile: %s" % profile)
        if not 1 <= repeats <= 20:
            raise ValueError("repeats must be between 1 and 20")
        selected_accel_per_hz = parse_accel_per_hz(accel_per_hz)
        selected_hz_per_sec = parse_hz_per_sec(hz_per_sec)
        selected_scv = parse_square_corner_velocity(square_corner_velocity)
        fast_validation_enabled = _parse_fast_validation(fast_validation)
        peak_lock_enabled = _parse_peak_lock(peak_lock)
        experimental_mode = profile in {"experimental_mzv", "adaptive_stock"}
        max_vibrations = _profile_max_vibrations(profile)
        if experimental_mode and not self.experimental_enabled:
            raise ValueError(
                "%s requires enable_experimental_generalized_mzv: True" % profile
            )
        if fast_validation_enabled and not experimental_mode:
            raise ValueError("FAST_VALIDATION is only supported for adaptive stock modes")
        if peak_lock_enabled and not experimental_mode:
            raise ValueError("PEAK_LOCK is only supported for adaptive stock modes")
        if experimental_mode and not validate:
            raise ValueError("%s requires mandatory held-out validation" % profile)
        if fast_validation_enabled and repeats != 2:
            raise ValueError("FAST_VALIDATION requires exactly REPEATS=2")
        if fast_validation_enabled and selected_hz_per_sec != 2.0:
            raise ValueError("FAST_VALIDATION requires explicit HZ_PER_SEC=2")
        if experimental_mode and not fast_validation_enabled and repeats < 3:
            raise ValueError("%s requires at least three repeats" % profile)

        training_repeats = 1 if fast_validation_enabled else repeats
        reference_repeats = repeats if validate else 0
        candidate_repeats = repeats if validate else 0

        attempt_id = self.id_factory()
        self.machine.begin()
        self.current_attempt_id = attempt_id
        self.current_attempt_status = "running"
        self.current_attempt_artifacts = None
        if fast_validation_enabled:
            protocol_mode = "fast_lower_confidence_1_train_2_held_out"
        elif experimental_mode:
            protocol_mode = "full_confidence_default"
        elif validate:
            protocol_mode = "native_validation"
        else:
            protocol_mode = "native_unvalidated_capture"
        validation_protocol: dict[str, Any] = {
            "mode": protocol_mode,
            "lower_confidence": fast_validation_enabled or (validate and repeats < 3),
            "repeats_per_group": repeats,
            "validation_enabled": bool(validate),
            "full_sweeps_per_axis": training_repeats + (
                0 if experimental_mode else reference_repeats + candidate_repeats
            ),
            "motion_time_excludes_host_analysis_and_artifact_time": True,
            "square_corner_velocity_source": (
                "command" if selected_scv is not None else "printer_snapshot"
            ),
        }
        if max_vibrations is not None:
            validation_protocol["native_fit_max_vibrations"] = {
                "fraction": max_vibrations,
                "percent": max_vibrations * 100.0,
                "source": "selection_profile.maximum_residual",
                "upstream_parameter": "max_vibrations",
            }
        if validate:
            pair_ids_by_axis = {
                axis: ["%s-%02d" % (axis, index + 1) for index in range(repeats)]
                for axis in normalized_axes
            }
            validation_protocol.update(
                {
                    "capture_design": (
                        "paired_interleaved_ab_finite_reversal_ringdown"
                        if experimental_mode
                        else "paired_interleaved_ab"
                    ),
                    "condition_labels": {"A": "reference", "B": "candidate"},
                    "pair_count_per_axis": repeats,
                    "total_pair_count": repeats * len(normalized_axes),
                    "pair_ids_by_axis": pair_ids_by_axis,
                    "capture_order": [],
                    "temporary_apply_readback": (
                        "exact_status_and_live_python_pulses_before_every_capture"
                        if experimental_mode
                        else "exact_status_before_every_capture"
                    ),
                }
            )
            if experimental_mode:
                validation_protocol.update(
                    {
                        "paired_transients_per_axis": (
                            reference_repeats + candidate_repeats
                        ),
                        "promotion_gate": "finite_reversal_ringdown_v1",
                        "filtered_sweep_screen": "diagnostic_only_non_promotional",
                        "shaped_resonance_sweep_promotion_eligible": False,
                    }
                )
        else:
            pair_ids_by_axis = {}
        if fast_validation_enabled:
            validation_protocol.update(
                {
                    "training_repeats": training_repeats,
                    "reference_repeats": reference_repeats,
                    "candidate_repeats": candidate_repeats,
                }
            )
        if peak_lock_enabled:
            validation_protocol.update(
                {
                    "peak_lock": True,
                    "frequency_strategy": "strongest_measured_peak",
                }
            )
        self.current_validation_protocol = validation_protocol
        # A previous accepted result remains addressable in ``results``, but it
        # must not appear to be the outcome of this new attempt.
        self.current_result_id = None
        snapshot = None
        result = None
        operation_error: Optional[BaseException] = None
        restore_error: Optional[BaseException] = None
        rejection_report: Optional[Mapping[str, Any]] = None
        rejection_raw_groups: Optional[Mapping[str, Mapping[str, list[Any]]]] = None
        captures: dict[str, list[Any]] = {axis: [] for axis in normalized_axes}
        runtime_capability: Optional[Mapping[str, Any]] = None
        reference_models: Optional[Mapping[str, Mapping[str, Any]]] = None
        candidate_models: Optional[Mapping[str, Mapping[str, Any]]] = None
        excitation_preflight: Optional[Mapping[str, Any]] = None
        transient_preflight: Optional[Mapping[str, Any]] = None
        executor_pulse_limit = 10
        try:
            capture_profile = getattr(self.adapter, "configure_capture_profile", None)
            if profile == "adaptive_stock" and capture_profile is None:
                raise RuntimeError(
                    "adapter cannot request the complete stock shaper allowlist"
                )
            if capture_profile is not None:
                capture_profile(profile)
            self.adapter.preflight(normalized_axes)
            excitation_probe = getattr(self.adapter, "preflight_excitation", None)
            if excitation_probe is None:
                raise RuntimeError("adapter cannot prove resonance excitation motion budget")
            excitation_preflight = excitation_probe(
                normalized_axes, selected_accel_per_hz, selected_hz_per_sec
            )
            if validate and experimental_mode:
                transient_probe = getattr(self.adapter, "preflight_transient", None)
                if transient_probe is None:
                    raise RuntimeError(
                        "adapter cannot prove finite transient validation support"
                    )
                transient_preflight = _require_transient_preflight(
                    transient_probe(normalized_axes), normalized_axes
                )
            sweep_span = float(excitation_preflight["max_frequency_hz"]) - float(
                excitation_preflight["min_frequency_hz"]
            )
            sweep_rate = float(excitation_preflight["hz_per_sec"])
            validation_protocol["estimated_motion_seconds_per_axis"] = (
                (
                    training_repeats * sweep_span / sweep_rate
                    + (reference_repeats + candidate_repeats)
                    * float(
                        (transient_preflight or {}).get(
                            "estimated_motion_seconds_per_capture_upper_bound", 0.0
                        )
                    )
                )
                if experimental_mode and validate
                else validation_protocol["full_sweeps_per_axis"]
                * sweep_span
                / sweep_rate
            )
            validation_protocol["hz_per_sec"] = sweep_rate
            if transient_preflight is not None:
                validation_protocol["transient_preflight"] = dict(
                    transient_preflight
                )
            self.current_validation_protocol = dict(validation_protocol)
            if experimental_mode:
                capability_probe = getattr(self.adapter, "preflight_experimental", None)
                if capability_probe is None:
                    raise RuntimeError("adapter cannot prove generalized-MZV runtime support")
                runtime_capability = capability_probe(
                    max_vibrations=max_vibrations
                )
                executor_pulse_limit = int(runtime_capability["executor_pulse_limit"])
            self.machine.checkpoint()
            snapshot = self.adapter.snapshot()
            analysis_snapshot = snapshot
            snapshot_selections = tuple(
                ShaperSelection(
                    getattr(snapshot, "shaper_type_" + axis.lower()),
                    getattr(snapshot, "shaper_freq_" + axis.lower()),
                    axis,
                    getattr(snapshot, "damping_ratio_" + axis.lower()),
                )
                for axis in normalized_axes
            )
            if experimental_mode:
                model_builder = getattr(self.adapter, "build_shaper_models", None)
                if model_builder is None:
                    raise RuntimeError(
                        "adapter cannot derive exact installed-Klipper reference models"
                    )
                reference_models = model_builder(snapshot_selections)
                _assert_models_match_selections(
                    reference_models, snapshot_selections, "reference"
                )
            if any(item.parameterized for item in snapshot_selections):
                capability_probe = getattr(self.adapter, "preflight_experimental", None)
                if capability_probe is None:
                    raise RuntimeError("adapter cannot prove existing parameterized shaper support")
                runtime_capability = capability_probe(
                    snapshot_selections, max_vibrations=max_vibrations
                )
                executor_pulse_limit = int(runtime_capability["executor_pulse_limit"])
            if selected_scv is not None:
                scv_setter = getattr(
                    self.adapter, "set_test_square_corner_velocity", None
                )
                if scv_setter is None:
                    raise RuntimeError("adapter cannot set and verify test SCV")
                scv_setter(selected_scv)
                analysis_snapshot = replace(
                    snapshot, square_corner_velocity=selected_scv
                )
            validation_protocol["square_corner_velocity"] = float(
                analysis_snapshot.square_corner_velocity
            )
            self.current_validation_protocol = dict(validation_protocol)
            self.machine.transition(CalibrationState.BASELINE_CAPTURE)
            for axis in normalized_axes:
                for repeat in range(training_repeats):
                    self.machine.checkpoint()
                    captures[axis].append(
                        self.adapter.capture(
                            axis,
                            repeat,
                            False,
                            accel_per_hz=selected_accel_per_hz,
                            hz_per_sec=selected_hz_per_sec,
                            max_vibrations=max_vibrations,
                        )
                    )

            self.machine.transition(CalibrationState.ANALYSIS)
            self.machine.checkpoint()
            report = self._invoke(
                self.analyzer,
                captures=captures,
                axes=normalized_axes,
                profile=profile,
                snapshot=analysis_snapshot,
                experimental_mode=experimental_mode,
                executor_pulse_limit=executor_pulse_limit,
                peak_lock=peak_lock_enabled,
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
            has_parameterized = any(item.parameterized for item in selections)
            if has_parameterized and not experimental_mode:
                raise RuntimeError(
                    "analysis returned a parameterized shaper outside experimental mode"
                )
            if has_parameterized and not validate:
                raise RuntimeError("parameterized shapers require held-out validation")
            if has_parameterized:
                capability_probe = getattr(self.adapter, "preflight_experimental", None)
                if capability_probe is None:
                    raise RuntimeError("adapter cannot prove exact parameterized selection")
                runtime_capability = capability_probe(
                    selections, max_vibrations=max_vibrations
                )
                executor_pulse_limit = int(runtime_capability["executor_pulse_limit"])
            report = dict(report)
            if experimental_mode:
                assert reference_models is not None
                model_builder = getattr(self.adapter, "build_shaper_models", None)
                if model_builder is None:
                    raise RuntimeError(
                        "adapter cannot derive exact installed-Klipper candidate models"
                    )
                candidate_models = model_builder(selections)
                _assert_models_match_selections(
                    candidate_models, selections, "candidate"
                )
                screen = self._invoke(
                    self.spectral_screener,
                    training_report=report,
                    axes=normalized_axes,
                    reference_models=reference_models,
                    candidate_models=candidate_models,
                )
                if not isinstance(screen, Mapping) or screen.get("passed") not in {
                    True,
                    False,
                }:
                    raise RuntimeError(
                        "theoretical spectral non-regression screen returned malformed evidence"
                    )
                report["theoretical_spectral_non_regression"] = dict(screen)
                capability_details = dict(runtime_capability or {})
                capability_details["configured_reference_model_proofs"] = (
                    _model_proof_summary(reference_models)
                )
                capability_details["selected_candidate_model_proofs"] = (
                    _model_proof_summary(candidate_models)
                )
                runtime_capability = capability_details
                validation_protocol["theoretical_spectral_non_regression"] = {
                    "required": True,
                    "passed": bool(screen["passed"]),
                    "evidence_level": "theoretical_preflight_screen",
                    "held_out_validation_still_required": True,
                }
                self.current_validation_protocol = dict(validation_protocol)
                if not screen["passed"]:
                    rejection_error = RuntimeError(
                        "candidate failed theoretical spectral non-regression screen: %s"
                        % screen.get("reason", "meaningful-band regression")
                    )
                    report.update(
                        {
                            "runtime_capability": runtime_capability,
                            "excitation_preflight": excitation_preflight,
                            "validation_protocol": dict(validation_protocol),
                            "attempt_id": attempt_id,
                            "status": "rejected",
                            "reason": str(rejection_error),
                        }
                    )
                    rejection_report = report
                    rejection_raw_groups = {"training": captures}
                    raise rejection_error
            report["runtime_capability"] = runtime_capability
            report["excitation_preflight"] = excitation_preflight
            report["validation_protocol"] = dict(validation_protocol)

            if validate:
                if experimental_mode:
                    # Training sweeps and worker analysis yield control for long
                    # enough that printer state and position must be proven again
                    # before any temporary shaper application or held-out motion.
                    self.adapter.preflight(normalized_axes)
                    transient_probe = getattr(
                        self.adapter, "preflight_transient", None
                    )
                    if transient_probe is None:
                        raise RuntimeError(
                            "adapter cannot prove finite transient validation support"
                        )
                    transient_preflight = _require_transient_preflight(
                        transient_probe(normalized_axes), normalized_axes
                    )
                    validation_protocol["transient_preflight"] = dict(
                        transient_preflight
                    )
                    validation_protocol["post_analysis_preflight_refreshed"] = True
                    self.current_validation_protocol = dict(validation_protocol)
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
                common_guards: dict[str, float] = {}
                if experimental_mode:
                    assert transient_preflight is not None
                    accel_setter = getattr(self.adapter, "set_test_max_accel", None)
                    if accel_setter is None:
                        raise RuntimeError(
                            "adapter cannot set and verify bounded transient acceleration"
                        )
                    accel_setter(float(transient_preflight["max_accel_mm_s2"]))
                    validation_protocol["transient_motion_limit"] = {
                        "max_accel_mm_s2": float(
                            transient_preflight["max_accel_mm_s2"]
                        ),
                        "source": "finite_transient_preflight_cap",
                        "readback_verified": True,
                        "restored_from_exact_snapshot_in_finally": True,
                    }
                    model_builder = getattr(self.adapter, "build_shaper_models", None)
                    if model_builder is None:
                        raise RuntimeError(
                            "adapter cannot derive exact transient pulse-span guards"
                        )
                    if reference_models is None:
                        reference_models = model_builder(reference)
                        _assert_models_match_selections(
                            reference_models, reference, "reference"
                        )
                    if candidate_models is None:
                        candidate_models = model_builder(selections)
                        _assert_models_match_selections(
                            candidate_models, selections, "candidate"
                        )
                    for axis in normalized_axes:
                        try:
                            reference_times = [
                                float(value)
                                for value in reference_models[axis]["pulse_times_s"]
                            ]
                            candidate_times = [
                                float(value)
                                for value in candidate_models[axis]["pulse_times_s"]
                            ]
                        except (KeyError, TypeError, ValueError) as error:
                            raise RuntimeError(
                                "%s-axis installed pulse timings are unavailable" % axis
                            ) from error
                        if not reference_times or not candidate_times:
                            raise RuntimeError(
                                "%s-axis installed pulse timings are empty" % axis
                            )
                        common_guards[axis] = max(
                            max(reference_times) - min(reference_times),
                            max(candidate_times) - min(candidate_times),
                        ) + 0.020
                    validation_protocol[
                        "common_post_command_guard_seconds_by_axis"
                    ] = dict(common_guards)
                    validation_protocol["common_window_fairness"] = (
                        "A and B discard the same longest pulse span plus 20 ms; "
                        "initial response is intentionally excluded"
                    )
                    validation_protocol["estimated_motion_seconds_per_axis"] = (
                        training_repeats * sweep_span / sweep_rate
                        + (reference_repeats + candidate_repeats)
                        * (
                            float(
                                transient_preflight[
                                    "estimated_base_motion_seconds_per_capture_upper_bound"
                                ]
                            )
                            + max(common_guards.values())
                        )
                    )
                held_out: dict[str, list[Any]] = {axis: [] for axis in normalized_axes}
                validation: dict[str, list[Any]] = {axis: [] for axis in normalized_axes}
                capture_order: list[dict[str, Any]] = []
                for axis in normalized_axes:
                    for repeat, pair_id in enumerate(pair_ids_by_axis[axis]):
                        for condition, active, destination in (
                            ("reference", reference, held_out),
                            ("candidate", selections, validation),
                        ):
                            self.machine.checkpoint()
                            if experimental_mode:
                                # Fail before SET_INPUT_SHAPER if printing,
                                # homing, or readiness changed between A/B runs.
                                self.adapter.preflight((axis,))
                            # KlipperPrinterAdapter.apply_temporary performs exact
                            # type/frequency/damping/axis status readback before
                            # this capture is allowed to start.
                            self.adapter.apply_temporary(active)
                            self.machine.checkpoint()
                            capture_row: dict[str, Any] = {
                                "sequence": len(capture_order) + 1,
                                "axis": axis,
                                "pair_id": pair_id,
                                "condition": condition,
                                "condition_label": (
                                    "A" if condition == "reference" else "B"
                                ),
                                "repeat_index": repeat,
                            }
                            if experimental_mode:
                                pulse_verifier = getattr(
                                    self.adapter, "verify_live_python_pulses", None
                                )
                                if pulse_verifier is None:
                                    raise RuntimeError(
                                        "adapter cannot prove live Python pulse state"
                                    )
                                pulse_proof = pulse_verifier(active)
                                if (
                                    not isinstance(pulse_proof, Mapping)
                                    or pulse_proof.get("passed") is not True
                                    or pulse_proof.get("live_c_executor_readback")
                                    is not False
                                    or axis not in pulse_proof.get("axes", {})
                                ):
                                    raise RuntimeError(
                                        "%s-axis live Python pulse proof is malformed"
                                        % axis
                                    )
                                try:
                                    live_pulse_guard = float(
                                        pulse_proof["axes"][axis][
                                            "post_command_guard_seconds"
                                        ]
                                    )
                                except (KeyError, TypeError, ValueError) as error:
                                    raise RuntimeError(
                                        "%s-axis live Python pulse guard is unavailable"
                                        % axis
                                    ) from error
                                pulse_guard = float(common_guards[axis])
                                if live_pulse_guard > pulse_guard + 1e-9:
                                    raise RuntimeError(
                                        "%s-axis live pulse span exceeds common paired guard"
                                        % axis
                                    )
                                transient_capture = self.adapter.capture_transient(
                                    axis,
                                    repeat,
                                    plan=transient_preflight["plans"][axis],
                                    max_accel_mm_s2=float(
                                        transient_preflight["max_accel_mm_s2"]
                                    ),
                                    speed_mm_s=float(
                                        transient_preflight["speed_mm_s"]
                                    ),
                                    post_command_guard_seconds=pulse_guard,
                                )
                                metadata = transient_capture.get("metadata", {})
                                if (
                                    not isinstance(metadata, Mapping)
                                    or metadata.get("promotion_eligible") is not True
                                    or metadata.get("protocol")
                                    != "finite_reversal_ringdown_v1"
                                    or metadata.get("validation_capture_kind")
                                    != "finite_reversal_ringdown"
                                ):
                                    raise RuntimeError(
                                        "%s-axis capture is not promotion-eligible "
                                        "transient evidence" % axis
                                    )
                                destination[axis].append(transient_capture)
                                capture_row.update(
                                    {
                                        "capture_kind": "finite_reversal_ringdown",
                                        "post_command_guard_seconds": pulse_guard,
                                        "live_python_pulse_proof": dict(pulse_proof),
                                    }
                                )
                            else:
                                destination[axis].append(
                                    self.adapter.capture(
                                        axis,
                                        repeat,
                                        validation=True,
                                        accel_per_hz=selected_accel_per_hz,
                                        hz_per_sec=selected_hz_per_sec,
                                        max_vibrations=max_vibrations,
                                    )
                                )
                                capture_row["capture_kind"] = (
                                    "native_compatibility_validation_sweep"
                                )
                            capture_order.append(capture_row)
                            validation_protocol["capture_order"] = list(capture_order)
                            self.current_validation_protocol = dict(validation_protocol)
                report["validation_protocol"] = dict(validation_protocol)
                validation_report = self._invoke(
                    self.analyzer,
                    captures=captures,
                    held_out_captures=held_out,
                    validation_captures=validation,
                    validation_pair_ids=pair_ids_by_axis,
                    axes=normalized_axes,
                    profile=profile,
                    snapshot=analysis_snapshot,
                    prior_report=report,
                    experimental_mode=experimental_mode,
                    executor_pulse_limit=executor_pulse_limit,
                    peak_lock=peak_lock_enabled,
                )
                report = dict(report)
                report["reference"] = [item.to_mapping() for item in reference]
                report["validation"] = validation_report.get("validation", {})
                report["validation_report"] = dict(validation_report)
                if experimental_mode:
                    try:
                        self._validate_generalized_evidence(
                            report["validation"], normalized_axes
                        )
                    except RuntimeError as evidence_error:
                        invalid_validation = dict(report["validation"])
                        invalid_validation["passed"] = False
                        invalid_validation["reason"] = str(evidence_error)
                        report["validation"] = invalid_validation
                        validation_report = dict(validation_report)
                        validation_report["validation"] = invalid_validation
                        report["validation_report"] = validation_report
                if not validation_report.get("validation", {}).get("passed", False):
                    rejection_error = RuntimeError(
                        "candidate failed held-out validation: %s"
                        % validation_report.get("validation", {}).get("reason", "attenuation gate")
                    )
                    report.update(
                        {
                            "attempt_id": attempt_id,
                            "status": "rejected",
                            "reason": str(rejection_error),
                        }
                    )
                    rejection_report = report
                    rejection_raw_groups = {
                        "training": captures,
                        "reference": held_out,
                        "candidate": validation,
                    }
                    raise rejection_error

            report = dict(report)
            report.update({"attempt_id": attempt_id, "status": "accepted"})
            result = CalibrationResult(attempt_id, selections, report)
        except CalibrationCancelled as error:
            operation_error = error
        except BaseException as error:
            operation_error = error
        finally:
            if snapshot is not None:
                try:
                    self.adapter.restore(snapshot)
                except BaseException as error:
                    restore_error = error

        if restore_error is not None:
            if operation_error is not None:
                failure = RuntimeError(
                    "%s; printer state restoration also failed: %s"
                    % (operation_error, restore_error)
                )
            else:
                failure = restore_error
            self.current_attempt_status = "failed"
            self.machine.failed(failure)
            raise failure from restore_error

        if isinstance(operation_error, CalibrationCancelled):
            self.current_attempt_status = "cancelled"
            self.machine.cancelled()
            raise operation_error

        if operation_error is not None:
            if rejection_report is not None and self.artifact_writer is not None:
                try:
                    self.current_attempt_artifacts = self._invoke(
                        self.artifact_writer.write,
                        result_id=attempt_id,
                        report=rejection_report,
                        raw_groups=rejection_raw_groups,
                    )
                except BaseException as artifact_error:
                    failure = RuntimeError(
                        "%s; rejected-attempt artifact write failed: %s"
                        % (operation_error, artifact_error)
                    )
                    self.current_attempt_status = "failed"
                    self.machine.failed(failure)
                    raise failure from artifact_error
            self.current_attempt_status = "rejected" if rejection_report is not None else "failed"
            self.machine.failed(operation_error)
            raise operation_error

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
                self.current_attempt_artifacts = artifacts
            except BaseException as artifact_error:
                self.current_attempt_status = "failed"
                self.machine.failed(artifact_error)
                raise
        self.results[result.result_id] = result
        self.current_result_id = result.result_id
        self.current_attempt_status = "accepted"
        self.machine.transition(CalibrationState.REVIEW)
        return result

    def _invoke(self, function: Callable[..., Any], **arguments: Any) -> Any:
        if self.worker is None:
            return function(**arguments)
        return self.worker.run(function, arguments, self.machine.checkpoint)

    def apply(self, result_id: str) -> CalibrationResult:
        result = self._get_result(result_id)
        self._assert_runtime_eligible(result)
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
        self._assert_runtime_eligible(result)
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
            "attempt_id": self.current_attempt_id,
            "attempt_status": self.current_attempt_status,
            "artifacts": self.current_attempt_artifacts,
            "cancel_requested": self.machine.cancel_requested,
            "error": self.machine.error,
            "experimental_generalized_mzv_enabled": self.experimental_enabled,
            "validation_protocol": self.current_validation_protocol,
        }

    def get_status(self, _eventtime: Any = None) -> Mapping[str, Any]:
        """Expose review state to Klipper templates and Moonraker."""
        return self.status()

    def _get_result(self, result_id: str) -> CalibrationResult:
        try:
            return self.results[result_id]
        except KeyError as error:
            raise ValueError("unknown result: %s" % result_id) from error

    @staticmethod
    def _assert_runtime_eligible(result: CalibrationResult) -> None:
        if result.report.get("status") != "accepted":
            raise RuntimeError("only accepted results are runtime eligible")
        if any(item.parameterized for item in result.selections):
            validation = result.report.get("validation", {})
            capability = result.report.get("runtime_capability", {})
            if not isinstance(validation, Mapping) or not validation.get("passed"):
                raise RuntimeError("parameterized result lacks accepted held-out validation")
            if not isinstance(capability, Mapping) or not capability.get("passed"):
                raise RuntimeError("parameterized result lacks installed-Klipper capability proof")
        if result.report.get("profile") == "adaptive_stock":
            validation = result.report.get("validation", {})
            capability = result.report.get("runtime_capability", {})
            if not isinstance(validation, Mapping) or not validation.get("passed"):
                raise RuntimeError("adaptive-stock result lacks accepted held-out validation")
            if not isinstance(capability, Mapping) or not capability.get("passed"):
                raise RuntimeError(
                    "adaptive-stock result lacks installed-Klipper capability proof"
                )

    @staticmethod
    def _validate_generalized_evidence(
        validation: Any, axes: Sequence[str]
    ) -> None:
        if not isinstance(validation, Mapping) or not validation.get("passed"):
            return
        details = validation.get("axes")
        if not isinstance(details, Mapping) or set(details) != set(axes):
            raise RuntimeError("generalized validation is missing exact per-axis evidence")
        for axis in axes:
            values = details[axis]
            if not isinstance(values, Mapping) or values.get("qc_passed") is not True:
                raise RuntimeError("%s generalized validation did not pass QC" % axis)
            confidence = values.get("improvement_ci_95")
            if not isinstance(confidence, (list, tuple)) or len(confidence) != 2:
                raise RuntimeError("%s generalized attenuation confidence is insufficient" % axis)
            try:
                confidence_low = float(confidence[0])
                cross_axis_regression = float(values["cross_axis_regression"])
            except (KeyError, TypeError, ValueError):
                raise RuntimeError(
                    "%s generalized validation contains malformed evidence" % axis
                ) from None
            if not math.isfinite(confidence_low) or confidence_low < 0.10:
                raise RuntimeError("%s generalized attenuation confidence is insufficient" % axis)
            if not math.isfinite(cross_axis_regression) or cross_axis_regression > 0.05:
                raise RuntimeError("%s generalized cross-axis regression is unsafe" % axis)
            if values.get("passed") is not True:
                raise RuntimeError("%s generalized validation gate was not accepted" % axis)
            if values.get("validation_evidence_kind") != "finite_reversal_ringdown_v1":
                raise RuntimeError(
                    "%s generalized validation is not finite-ringdown evidence" % axis
                )
            fairness = values.get("paired_window_fairness")
            spectral = values.get("measured_spectral_non_regression")
            if not isinstance(fairness, Mapping) or fairness.get("passed") is not True:
                raise RuntimeError(
                    "%s generalized paired-window fairness is insufficient" % axis
                )
            if not isinstance(spectral, Mapping) or spectral.get("passed") is not True:
                raise RuntimeError(
                    "%s generalized measured spectral non-regression is unsafe" % axis
                )

    def cmd_ADV_SHAPER_CALIBRATE(self, gcmd: Any) -> None:
        axis = gcmd.get("AXIS", "ALL").upper()
        axes = ("X", "Y") if axis == "ALL" else (axis,)
        try:
            result = self.calibrate(
                axes,
                profile=gcmd.get("PROFILE", "balanced"),
                repeats=gcmd.get_int("REPEATS", 3, minval=1, maxval=20),
                validate=bool(gcmd.get_int("VALIDATE", 1, minval=0, maxval=1)),
                accel_per_hz=gcmd.get("ACCEL_PER_HZ", None),
                hz_per_sec=gcmd.get("HZ_PER_SEC", None),
                square_corner_velocity=gcmd.get("SCV", None),
                fast_validation=gcmd.get_int(
                    "FAST_VALIDATION", 0, minval=0, maxval=1
                ),
                peak_lock=gcmd.get_int("PEAK_LOCK", 0, minval=0, maxval=1),
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
