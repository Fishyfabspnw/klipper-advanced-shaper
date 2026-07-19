# Architecture and safety boundary

Klipper Advanced Shaper has three boundaries:

1. The Klippy controller owns commands, preflight checks, state transitions,
   capture coordination, cancellation, and restoration.
2. The analysis package receives captures and returns quality evidence and
   candidate scores. It does not directly control the printer.
3. The report/artifact layer serializes versioned results using atomic writes.

The controller's state progresses through `idle`, `preflight`, `baseline_capture`,
`analysis`, `temporary_validation`, and `review`. Error and cancellation paths
restore the original native input-shaper and velocity snapshot. A reviewed
result is not applied or persisted without a separate operator command.

## Non-goals

The project adds one Klippy extra loader and an optional macro include, but does
not modify Klipper's motion planner, `shaper_defs.py`, `input_shaper.py`,
`shaper_calibrate.py`, shaper impulse generator, C kinematics helper, MCU
firmware, or safety limits. It does not infer mechanically safe acceleration
from smoothing alone. It does not automatically modify heaters, fans, motor
currents, persistent velocity limits, or configuration files.

## Analysis contract

Capture quality must be checked before selection. The robust engine accounts for
sample timing, excitation-axis and cross-axis response, repeat consistency,
multiple modes, and uncertainty. Candidate selection returns either an
allowlisted shaper and frequency or an explicit abstention with reasons.

Native parity results and robust results remain distinguishable in artifacts.
Normal profiles preserve Klipper's ordinary autotune family set. ZVD is added
only to the explicit six-family `adaptive_stock` comparison because current
upstream defines it. The explicit
`experimental_mzv` profile searches only canonical generalized MZV. The
`adaptive_stock` profile compares exact-name ZV, MZV, ZVD, EI, 2HUMP_EI, and
3HUMP_EI candidates with generalized-MZV candidates. Both experimental profiles
require installed capability, validation, readback, and rollback gates.

## Klipper compatibility boundary

Klipper does not publish a stable third-party accelerometer capture ABI. The
capture adapter fingerprints the native `resonance_tester._run_test` signature,
uses Klipper's own bounded resonance motion and shaper candidate evaluator, and
refuses to run when the expected interface changes. All access to this private
surface is isolated in `klippy/capture.py`; the numerical engine and reports do
not import Klipper.

An omitted calibration `ACCEL_PER_HZ` remains omitted at that boundary, so
Klipper inherits the active `[resonance_tester]` value. An explicit value is
strictly parsed as an unsigned decimal from 20 through 350 mm/s^2/Hz. Signs,
exponents, whitespace, non-finite values, and trailing text fail closed. The
effective inherited value is subject to the same bounds.

After ordinary printer/activity preflight but before snapshot or motion, the
capture boundary reads the configured maximum sweep frequency, sweeping
acceleration, and current toolhead `max_accel`. Pulse acceleration
(`max_freq * accel_per_hz`) plus sweeping acceleration must not exceed 80% of
that motion limit. The calculation is retained in the result report. Missing,
invalid, or excessive values abort without a snapshot or capture. The capture
command object overrides only `ACCEL_PER_HZ`; all other resonance recipe
defaults continue to come from the running Klipper configuration. The same
resolved value is used for training, held-out reference, and candidate sweeps,
and the native recipe records the effective value in each capture.

An explicit `SCV` override is separate from the resonance command recipe. The
controller snapshots the original square-corner velocity first, sends a bounded
stock `SET_VELOCITY_LIMIT SQUARE_CORNER_VELOCITY=...` command, verifies exact
toolhead-status readback, uses that value for fitting, and restores and verifies
the original SCV on every exit path.

Sweep rate is an orthogonal strict override. `HZ_PER_SEC=CONFIG` inherits the
active `[resonance_tester]` value; explicit unsigned decimals are bounded to
0.1..2 Hz/s. The capture preflight resolves and validates the effective rate,
then the command boundary passes it unchanged to every capture. Reports retain
the resolved rate, its source, sweep count, and estimated physical sweep time.
The estimate explicitly excludes host analysis and artifact generation.

The default experimental validation protocol requires at least three training,
three held-out reference, and three candidate repeats. The explicit fast mode
is the sole exception: exactly two repeats per group, `VALIDATE=1`, and explicit
`HZ_PER_SEC=2`. It remains a held-out 95% attenuation-confidence test with QC,
cross-axis regression, exact readback, and rollback gates, but reports label it
as lower confidence. One-repeat experimental validation remains forbidden.

An optional `PEAK_LOCK=1` request is carried through the controller and analysis
boundary only for the two adaptive stock profiles. Analysis selects the highest-PSD
detected mode independently for each axis and restricts the generalized-MZV
frequency search to that one exact bin. This changes candidate generation, not
the validation or restoration protocol.

Training sweeps use Klipper's normal behavior of temporarily disabling input
shaping. Held-out reference and candidate sweeps retain their temporary native
shapers, so acceptance compares the proposal against the shaper active when the
session began. A candidate is never published when capture, analysis, artifact writing, or state
restoration fails.

Parameterized identifiers pass through one strict parser shared by analysis and
Klippy. Temporary application reads status back per axis before motion resumes;
rollback uses and verifies the same canonical snapshot. Unsupported installed
Klipper builds abstain instead of substituting native MZV. The capability record
includes the executor pulse capacity discovered from the installed Klipper
source; optimization never emits a candidate above that capacity or the
project-wide ten-pulse limit.

## Stock-compatible adaptive boundary

The project does not emit arbitrary pulse arrays or private shaper names and
does not install a custom executor.
`adaptive_stock` ranks only exact native families plus strictly parsed
generalized MZV. The installed `shaper_defs` implementation must reproduce the
expected generalized pulse sequence and expose a compatible executor capacity
before any adaptive motion. Every temporary result is applied through
`SET_INPUT_SHAPER`, read back from Klipper status, validated on held-out sweeps,
and rolled back exactly.
