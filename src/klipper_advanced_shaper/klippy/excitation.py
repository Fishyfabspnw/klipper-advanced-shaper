"""Strict resonance-excitation parsing and motion-budget checks."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Optional

MIN_ACCEL_PER_HZ = 20.0
MAX_ACCEL_PER_HZ = 150.0
MIN_HZ_PER_SEC = 0.1
MAX_HZ_PER_SEC = 2.0
MOTION_LIMIT_FRACTION = 0.80
_STRICT_DECIMAL = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z")


def parse_accel_per_hz(value: Any) -> Optional[float]:
    """Parse CONFIG or an unsigned decimal in the bounded public range."""
    if value is None:
        return None
    text = str(value)
    if text.upper() == "CONFIG":
        return None
    if not _STRICT_DECIMAL.fullmatch(text):
        raise ValueError(
            "ACCEL_PER_HZ must be CONFIG or an unsigned decimal between "
            "20 and 150 mm/s^2/Hz"
        )
    parsed = float(text)
    if not math.isfinite(parsed) or not MIN_ACCEL_PER_HZ <= parsed <= MAX_ACCEL_PER_HZ:
        raise ValueError("ACCEL_PER_HZ must be between 20 and 150 mm/s^2/Hz")
    return parsed


def parse_hz_per_sec(value: Any) -> Optional[float]:
    """Parse CONFIG or an unsigned decimal in the bounded sweep-rate range."""
    if value is None:
        return None
    text = str(value)
    if text.upper() == "CONFIG":
        return None
    if not _STRICT_DECIMAL.fullmatch(text):
        raise ValueError(
            "HZ_PER_SEC must be CONFIG or an unsigned decimal between 0.1 and 2 Hz/s"
        )
    parsed = float(text)
    if not math.isfinite(parsed) or not MIN_HZ_PER_SEC <= parsed <= MAX_HZ_PER_SEC:
        raise ValueError("HZ_PER_SEC must be between 0.1 and 2 Hz/s")
    return parsed


def check_sweep_rate(hz_per_sec: Any) -> float:
    """Resolve and validate the effective configured or command sweep rate."""
    try:
        effective = float(hz_per_sec)
    except (TypeError, ValueError) as error:
        raise RuntimeError("effective [resonance_tester] hz_per_sec is unavailable") from error
    if not math.isfinite(effective):
        raise RuntimeError("effective [resonance_tester] hz_per_sec is non-finite")
    if not MIN_HZ_PER_SEC <= effective <= MAX_HZ_PER_SEC:
        raise RuntimeError(
            "effective [resonance_tester] hz_per_sec must be between 0.1 and 2 Hz/s"
        )
    return effective


def check_motion_budget(
    accel_per_hz: Any,
    max_frequency_hz: Any,
    printer_max_accel: Any,
    sweeping_accel: Any = 0.0,
) -> Mapping[str, float]:
    """Fail unless peak resonance excitation fits a conservative motion budget."""
    try:
        effective = float(accel_per_hz)
        max_frequency = float(max_frequency_hz)
        motion_limit = float(printer_max_accel)
        sweep = abs(float(sweeping_accel))
    except (TypeError, ValueError) as error:
        raise RuntimeError("resonance excitation limits are unavailable") from error
    values = (effective, max_frequency, motion_limit, sweep)
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("resonance excitation limits are non-finite")
    if not MIN_ACCEL_PER_HZ <= effective <= MAX_ACCEL_PER_HZ:
        raise RuntimeError(
            "effective [resonance_tester] accel_per_hz must be between "
            "20 and 150 mm/s^2/Hz"
        )
    if max_frequency <= 0.0 or motion_limit <= 0.0 or sweep < 0.0:
        raise RuntimeError("resonance excitation limits must be positive")
    pulse_peak = effective * max_frequency
    estimated_peak = pulse_peak + sweep
    allowed_peak = motion_limit * MOTION_LIMIT_FRACTION
    if estimated_peak > allowed_peak:
        raise RuntimeError(
            "resonance excitation preflight rejected %.3f mm/s^2/Hz: "
            "estimated peak %.0f mm/s^2 exceeds the 80%% motion budget %.0f "
            "mm/s^2 (max_freq %.1f Hz, printer max_accel %.0f mm/s^2)"
            % (effective, estimated_peak, allowed_peak, max_frequency, motion_limit)
        )
    return {
        "accel_per_hz": effective,
        "max_frequency_hz": max_frequency,
        "pulse_peak_accel_mm_s2": pulse_peak,
        "sweeping_accel_mm_s2": sweep,
        "estimated_peak_accel_mm_s2": estimated_peak,
        "printer_max_accel_mm_s2": motion_limit,
        "motion_limit_fraction": MOTION_LIMIT_FRACTION,
        "allowed_peak_accel_mm_s2": allowed_peak,
    }
