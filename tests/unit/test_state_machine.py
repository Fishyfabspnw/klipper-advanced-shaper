import pytest

from klipper_advanced_shaper.klippy.state import (
    CalibrationCancelled,
    CalibrationState,
    InvalidTransition,
    StateMachine,
)


def test_happy_path_to_review():
    machine = StateMachine()
    machine.begin()
    machine.transition(CalibrationState.BASELINE_CAPTURE)
    machine.transition(CalibrationState.ANALYSIS)
    machine.transition(CalibrationState.TEMPORARY_VALIDATION)
    machine.transition(CalibrationState.REVIEW)

    assert machine.state is CalibrationState.REVIEW
    assert machine.error is None


def test_invalid_transition_fails_closed():
    machine = StateMachine()

    with pytest.raises(InvalidTransition):
        machine.transition(CalibrationState.RUNTIME_APPLIED)

    assert machine.state is CalibrationState.IDLE


def test_cancel_is_cooperative_and_only_active_during_calibration():
    machine = StateMachine()
    assert machine.request_cancel() is False

    machine.begin()
    assert machine.request_cancel() is True
    with pytest.raises(CalibrationCancelled):
        machine.checkpoint()
    machine.cancelled()

    assert machine.state is CalibrationState.CANCELLED


def test_new_run_clears_failure_and_cancel_flags():
    machine = StateMachine()
    machine.begin()
    machine.failed(RuntimeError("sensor failed"))
    assert machine.error == "sensor failed"

    machine.begin()

    assert machine.state is CalibrationState.PREFLIGHT
    assert machine.error is None
    assert machine.cancel_requested is False
