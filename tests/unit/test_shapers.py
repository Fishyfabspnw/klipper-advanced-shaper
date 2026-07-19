import pytest

from klipper_advanced_shaper.klippy.adapter import ShaperSelection
from klipper_advanced_shaper.shapers import parse_shaper_identifier


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("ZVD", "zvd"),
        (" mzv ( n = 4 , t = .8 ) ", "mzv(n=4,t=0.800000)"),
        ("mzv(tau=1.25,n=10)", "mzv(n=10,tau=1.250000)"),
    ],
)
def test_allowlisted_shaper_identifiers_are_canonical(raw, canonical):
    assert parse_shaper_identifier(raw).canonical == canonical


@pytest.mark.parametrize(
    "raw",
    [
        "custom",
        "mzv()",
        "mzv(4,.8)",
        "mzv(n=4,0.8)",
        "mzv(n=4,t=.8,tau=1)",
        "mzv(n=4,t=.8,evil=1)",
        "mzv(n=4,n=5,t=.8)",
        "mzv(n=2,t=.2)",
        "mzv(n=11,t=.8)",
        "mzv(n=4,t=1.5)",
        "mzv(n=4,t=8e-1)",
        "mzv(n=4,t=-.8)",
        "mzv(n=4,t=.499999)",
        "mzv(n=4,tau=.499999)",
        "mzv(n=4,t=.0000001)",
        "mzv(n=4,t=1.4999999)",
        "ei(v_tol=.05)",
    ],
)
def test_arbitrary_mixed_or_unsafe_arguments_are_rejected(raw):
    with pytest.raises(ValueError):
        parse_shaper_identifier(raw)


def test_parameterized_forms_require_explicit_permission_when_requested():
    with pytest.raises(ValueError, match="experimental mode"):
        parse_shaper_identifier("mzv(n=4,t=.8)", allow_parameterized=False)


def test_runtime_selection_preserves_canonical_parameterized_identifier():
    selection = ShaperSelection(" MZV(t=.8,n=4) ", 72.25, "x", 0.04)
    assert selection.shaper_type == "mzv(n=4,t=0.800000)"
    assert selection.axis == "X"
    assert selection.parameterized is True


def test_family_specific_damping_limits_match_upstream():
    with pytest.raises(ValueError, match="0.20"):
        ShaperSelection("3hump_ei", 70.0, "X", 0.21)
