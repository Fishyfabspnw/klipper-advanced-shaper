# Experimental generalized MZV

This project can research a wider part of the input-shaper design space without
changing Klipper's motion planner. Current upstream Klipper accepts a
parameterized form of MZV such as `mzv(n=4,t=0.8)`. Klipper converts that name,
frequency, and damping ratio into a positive pulse sequence and sends it to the
existing generic input-shaper executor. The strict project limit is ten pulses,
but preflight also reads the installed executor capacity and bounds the search
to that smaller value when necessary.

This is best described as **generalized MZV optimization**, not a new Klipper
shaper family. The plugin does not patch, replace, or monkey-patch Klipper's
`shaper_defs`, `input_shaper`, or C kinematics code.

Pulse-generation parity is frozen against upstream Klipper commit
`7046bd00ef5c30dec6febc724f8d22967433c45c`. The runtime capability probe is
mandatory because older or vendor-modified Klipper builds may not accept
parameterized shaper names. A build without `get_shaper_cfg()` and
`init_shaper()` support, or without a provable executor capacity, abstains
safely.

## What the experimental optimizer does

The offline search varies pulse count, dimensionless pulse spacing, frequency,
and measured damping. It rejects negative-pulse designs, evaluates the
PSD-weighted residual energy over the measured modes and their damping
uncertainty, computes Klipper's 90/180-degree path-error proxy, and returns the
non-dominated trade-offs between:

- the 95th-percentile residual energy under damping uncertainty;
- sensitivity to that uncertainty; and
- the smoothing-derived acceleration estimate.

Measured modal damping is mandatory. A missing estimate causes abstention; the
experimental fitter never silently substitutes `0.1`.

The runtime command may explicitly request `PEAK_LOCK=1`. In that mode,
frequency is no longer searched: the generalized MZV is fixed to the detected
mode with the greatest measured PSD amplitude for that axis, while pulse count
and spacing remain optimized. The report records
`frequency_strategy: strongest_measured_peak`, the exact peak frequency, and
the selected canonical identifier. Peak locking is rejected outside
`PROFILE=experimental_mzv` and `PROFILE=adaptive_stock` before printer preflight
or motion.

Calling the optimizer directly still returns research-only candidates. The
normal calibration pipeline promotes those candidates only under
`PROFILE=experimental_mzv` or `PROFILE=adaptive_stock` when
`enable_experimental_generalized_mzv: True`. Its full-confidence default
requires three or more fitting/reference/candidate repeats. An explicit
lower-confidence fast protocol permits exactly two of each only with
`FAST_VALIDATION=1`, `VALIDATE=1`, and `HZ_PER_SEC=2`; one repeat is never
allowed. Both protocols require measured modal damping, per-capture QC, a 95%
attenuation confidence lower bound of at least 10%, no more than 5% cross-axis
regression, exact runtime readback, and successful rollback. A rejection is
never applyable or stageable.

`PROFILE=adaptive_stock` uses the same capability and validation protocol but
places the exact native ZV, MZV, ZVD, EI, 2HUMP_EI, and 3HUMP_EI candidates in
the same selection pool as generalized MZV. This lets each axis retain a native
winner when it is genuinely better under the selected metrics. It never emits
an arbitrary pulse vector or a project-specific shaper family.

The accepted identifier language is deliberately smaller than Klipper's
parser: `mzv(n=<3..10>,t=<safe decimal>)` or
`mzv(n=<3..10>,tau=<positive decimal>)`. Arguments must be named; unknown,
duplicate, positional, mixed `t`/`tau`, signed, exponent, NaN, and infinite
values are rejected. Reports, runtime apply, rollback snapshots, and staged
configuration use the same canonical identifier.

Current upstream also defines ZVD. ZVD is part of the exact-name allowlist, but
the default native selection profiles are unchanged. The parameter schema is
structured so a later, separately validated `ei(v_tol=...)` implementation can
be added without opening arbitrary shaper arguments; EI parameters are not
currently exposed.

## Non-inflating acceleration envelope

The experimental acceleration envelope is the minimum of the native-compatible
smoothing/path-error bound, conservative repeatability and model-uncertainty
bounds, and any available vibration-confidence, hardware, or print bound. A
vibration-confidence bound participates only when a held-out resonance test
recorded the actual acceleration. Normalized vibration attenuation is not
converted into an acceleration because that conversion would not be
dimensionally justified.

Every envelope is labeled, and reports keep three separate fields:

- `theoretical`: pulse-response and smoothing model only;
- `resonance_validated`: an explicit-acceleration held-out resonance test also
  limits the result; or
- `print_validated`: a supervised print-quality test also limits the result.

The formula never changes `[printer] max_accel`. A result can exceed a stock
shaper's estimate only when the optimized pulse design itself has less modeled
path error while still passing robust residual-vibration gates. The formula
cannot manufacture a higher number by relaxing Klipper's `0.12` smoothing
criterion.

Passing resonance validation does not by itself populate a
`resonance_validated` acceleration. That field requires a defensible recorded
test acceleration. Likewise, only a supervised print-quality test can populate
the print-validated field.

## Upstream implementation references

- [Klipper shaper definitions](https://github.com/Klipper3d/klipper/blob/master/klippy/extras/shaper_defs.py)
- [Klipper input-shaper integration](https://github.com/Klipper3d/klipper/blob/master/klippy/extras/input_shaper.py)
- [Klipper generic pulse executor](https://github.com/Klipper3d/klipper/blob/master/klippy/chelper/kin_shaper.c)
- [Klipper calibration and smoothing estimate](https://github.com/Klipper3d/klipper/blob/master/klippy/extras/shaper_calibrate.py)
