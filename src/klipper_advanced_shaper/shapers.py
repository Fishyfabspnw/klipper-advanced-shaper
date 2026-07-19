"""Strict, canonical input-shaper identifiers shared by analysis and Klippy.

Klipper's upstream parser is intentionally permissive.  This project accepts a
smaller allowlisted language so report data can be passed to g-code without
turning arbitrary strings into shaper arguments.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Mapping, Union

NATIVE_SHAPER_ORDER = ("zv", "mzv", "zvd", "ei", "2hump_ei", "3hump_ei")
NATIVE_SHAPERS = frozenset(NATIVE_SHAPER_ORDER)
MAX_EXECUTOR_PULSES = 10

PARAMETERIZED_ARGUMENTS: Mapping[str, frozenset[str]] = {
    "mzv": frozenset({"n", "t", "tau"}),
}

_IDENTIFIER = re.compile(r"^\s*([a-z0-9_]+)\s*(?:\((.*)\))?\s*$", re.IGNORECASE)
_ARGUMENT = re.compile(
    r"^\s*([a-z_][a-z0-9_]*)\s*=\s*(\d+(?:\.\d*)?|\.\d+)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ShaperIdentifier:
    family: str
    canonical: str
    arguments: tuple[tuple[str, Union[float, int]], ...] = ()

    @property
    def parameterized(self) -> bool:
        return bool(self.arguments)

    def argument_map(self) -> dict[str, Union[float, int]]:
        return dict(self.arguments)

    def mzv_spacing(self) -> float:
        """Return upstream's dimensionless ``t`` for a validated MZV form."""
        values = self.argument_map()
        n = int(values["n"])
        if "t" in values:
            return float(values["t"])
        tau = float(values["tau"])
        return tau * (n - 1.0) / (n + 2.0 * tau - 2.0)


def _canonical_number(value: float) -> str:
    return "%.6f" % value


def parse_shaper_identifier(value: str, *, allow_parameterized: bool = True) -> ShaperIdentifier:
    """Parse a supported shaper name and return its canonical representation.

    Only named generalized-MZV arguments are accepted.  Positional, duplicate,
    mixed ``t``/``tau``, exponent, sign, NaN/Inf, and unknown arguments fail.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("shaper type must be a non-empty string")
    match = _IDENTIFIER.fullmatch(value)
    if match is None:
        raise ValueError("malformed shaper identifier")
    family = match.group(1).lower()
    body = match.group(2)
    if family not in NATIVE_SHAPERS:
        raise ValueError("unsupported shaper type: %s" % family)
    if body is None:
        return ShaperIdentifier(family, family)
    if not allow_parameterized:
        raise ValueError("parameterized shapers require experimental mode")
    if family not in PARAMETERIZED_ARGUMENTS:
        raise ValueError("parameterized %s is not allowlisted" % family)
    if not body.strip():
        raise ValueError("parameterized shaper arguments cannot be empty")

    parsed: dict[str, Union[float, int]] = {}
    for raw in body.split(","):
        argument = _ARGUMENT.fullmatch(raw)
        if argument is None:
            raise ValueError("only named decimal shaper arguments are supported")
        name = argument.group(1).lower()
        if name not in PARAMETERIZED_ARGUMENTS[family]:
            raise ValueError("unknown %s argument: %s" % (family, name))
        if name in parsed:
            raise ValueError("duplicate %s argument: %s" % (family, name))
        number = argument.group(2)
        if name == "n":
            numeric = float(number)
            if not numeric.is_integer():
                raise ValueError("MZV n must be an integer")
            parsed[name] = int(numeric)
        else:
            parsed[name] = float(number)

    if family == "mzv":
        if set(parsed) not in ({"n", "t"}, {"n", "tau"}):
            raise ValueError("parameterized MZV requires n and exactly one of t or tau")
        n = int(parsed["n"])
        if not 3 <= n <= MAX_EXECUTOR_PULSES:
            raise ValueError("parameterized MZV n must be between 3 and 10")
        spacing_name = "t" if "t" in parsed else "tau"
        spacing = float(parsed[spacing_name])
        spacing = float(_canonical_number(spacing))
        if not math.isfinite(spacing) or spacing <= 0.0:
            raise ValueError("parameterized MZV spacing must be finite and positive")
        if spacing_name == "t" and not spacing < 0.5 * (n - 1):
            raise ValueError("parameterized MZV t must be below (n - 1) / 2")
        if spacing_name == "tau":
            t = spacing * (n - 1.0) / (n + 2.0 * spacing - 2.0)
            if not 0.0 < t < 0.5 * (n - 1):
                raise ValueError("parameterized MZV tau violates upstream spacing constraints")
        canonical = "mzv(n=%d,%s=%s)" % (n, spacing_name, _canonical_number(spacing))
        return ShaperIdentifier(
            family,
            canonical,
            (("n", n), (spacing_name, spacing)),
        )
    raise ValueError("parameterized shaper is not implemented")


def canonical_shaper_identifier(value: str, *, allow_parameterized: bool = True) -> str:
    return parse_shaper_identifier(value, allow_parameterized=allow_parameterized).canonical
