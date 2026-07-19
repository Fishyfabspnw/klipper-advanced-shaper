"""Fail-closed calibration state machine."""

from __future__ import annotations

from enum import Enum
from threading import RLock
from typing import Optional


class CalibrationState(str, Enum):
    IDLE = "idle"
    PREFLIGHT = "preflight"
    BASELINE_CAPTURE = "baseline_capture"
    ANALYSIS = "analysis"
    TEMPORARY_VALIDATION = "temporary_validation"
    REVIEW = "review"
    RUNTIME_APPLIED = "runtime_applied"
    STAGED = "staged"
    CANCELLED = "cancelled"
    FAILED = "failed"


class InvalidTransition(RuntimeError):
    pass


class CalibrationCancelled(RuntimeError):
    pass


_ALLOWED = {
    CalibrationState.IDLE: {CalibrationState.PREFLIGHT},
    CalibrationState.PREFLIGHT: {
        CalibrationState.BASELINE_CAPTURE,
        CalibrationState.CANCELLED,
        CalibrationState.FAILED,
    },
    CalibrationState.BASELINE_CAPTURE: {
        CalibrationState.ANALYSIS,
        CalibrationState.CANCELLED,
        CalibrationState.FAILED,
    },
    CalibrationState.ANALYSIS: {
        CalibrationState.TEMPORARY_VALIDATION,
        CalibrationState.REVIEW,
        CalibrationState.CANCELLED,
        CalibrationState.FAILED,
    },
    CalibrationState.TEMPORARY_VALIDATION: {
        CalibrationState.REVIEW,
        CalibrationState.CANCELLED,
        CalibrationState.FAILED,
    },
    CalibrationState.REVIEW: {
        CalibrationState.RUNTIME_APPLIED,
        CalibrationState.STAGED,
        CalibrationState.PREFLIGHT,
    },
    CalibrationState.RUNTIME_APPLIED: {
        CalibrationState.STAGED,
        CalibrationState.PREFLIGHT,
    },
    CalibrationState.STAGED: {CalibrationState.PREFLIGHT},
    CalibrationState.CANCELLED: {CalibrationState.PREFLIGHT, CalibrationState.FAILED},
    CalibrationState.FAILED: {CalibrationState.PREFLIGHT},
}


class StateMachine:
    """Small thread-safe state machine with cooperative cancellation."""

    def __init__(self) -> None:
        self._state = CalibrationState.IDLE
        self._cancel_requested = False
        self._error: Optional[str] = None
        self._lock = RLock()

    @property
    def state(self) -> CalibrationState:
        with self._lock:
            return self._state

    @property
    def error(self) -> Optional[str]:
        with self._lock:
            return self._error

    @property
    def cancel_requested(self) -> bool:
        with self._lock:
            return self._cancel_requested

    def transition(self, target: CalibrationState) -> None:
        with self._lock:
            if target not in _ALLOWED[self._state]:
                raise InvalidTransition(
                    "cannot transition from %s to %s" % (self._state.value, target.value)
                )
            self._state = target

    def begin(self) -> None:
        with self._lock:
            self._cancel_requested = False
            self._error = None
        self.transition(CalibrationState.PREFLIGHT)

    def request_cancel(self) -> bool:
        with self._lock:
            if self._state not in {
                CalibrationState.PREFLIGHT,
                CalibrationState.BASELINE_CAPTURE,
                CalibrationState.ANALYSIS,
                CalibrationState.TEMPORARY_VALIDATION,
            }:
                return False
            self._cancel_requested = True
            return True

    def checkpoint(self) -> None:
        with self._lock:
            if self._cancel_requested:
                raise CalibrationCancelled("calibration cancelled")

    def cancelled(self) -> None:
        with self._lock:
            if CalibrationState.CANCELLED not in _ALLOWED[self._state]:
                raise InvalidTransition("current operation is not cancellable")
            self._state = CalibrationState.CANCELLED

    def failed(self, error: BaseException) -> None:
        with self._lock:
            self._error = str(error)
            if self._state is CalibrationState.FAILED:
                return
            if CalibrationState.FAILED not in _ALLOWED[self._state]:
                raise InvalidTransition("current operation cannot fail")
            self._state = CalibrationState.FAILED
