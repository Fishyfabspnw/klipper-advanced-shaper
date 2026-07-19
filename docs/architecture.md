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

The production workflow is deliberately two-stage. Stage one belongs to normal
Klipper or Shake&Tune: calibrate, review, apply, and save an ordinary stock
shaper. Stage two snapshots the resulting live Klipper type, frequency, and
damping as the authoritative exact baseline, then treats parameterized MZV only
as an upgrade challenger. A parameterized candidate must clear the same
residual metric and profile limit as stock candidates and provide at least 5%
more theoretical smoothing acceleration than both the exact active baseline and
the best eligible stock candidate fitted from the same capture. Comparing with
the stronger of those two bounds implements both requirements. Failure returns
an explicit no-upgrade result before temporary candidate validation, apply, or
stage eligibility. `adaptive_stock` may retain an eligible stock winner.

## Klipper compatibility boundary

Klipper does not publish a stable third-party accelerometer capture ABI. The
capture adapter fingerprints the native `resonance_tester._run_test` signature,
uses Klipper's own bounded resonance motion and shaper candidate evaluator, and
refuses to run when the expected interface changes. All access to this private
surface is isolated in `klippy/capture.py`; the numerical engine and reports do
not import Klipper.

For `experimental_mzv` and `adaptive_stock` only, preflight also proves that
the installed native fitter explicitly accepts `max_vibrations`. The plugin
passes the finite fraction from that profile's `maximum_residual` (currently
`0.10`, or 10%, for both) to upstream fitting. This changes the frequency that
upstream fits within each native family before project-side ranking; it is not
the held-out requirement to demonstrate a 10% attenuation improvement. The
value is never normalized when absent: ordinary profiles omit the argument and
therefore retain their legacy fitting behavior. If an experimental profile
requires it and the installed fitter cannot prove support, preflight abstains
before snapshot or motion.

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
resolved value is used for training and, for experimental profiles, the
associated validation session; the native recipe records it in each resonance
capture.

An explicit `SCV` override is separate from the resonance command recipe. The
controller snapshots the original square-corner velocity first, sends a bounded
stock `SET_VELOCITY_LIMIT SQUARE_CORNER_VELOCITY=...` command, verifies exact
toolhead-status readback, uses that value for fitting, and restores and verifies
the original SCV on every exit path.

Sweep rate is an orthogonal strict override. `HZ_PER_SEC=CONFIG` inherits the
active `[resonance_tester]` value; explicit unsigned decimals are bounded to
0.1..2 Hz/s. The capture preflight resolves and validates the effective rate,
then the command boundary passes it unchanged to every resonance capture.
Reports retain the resolved rate and its source. Experimental transient
validation has separate bounded-motion metadata and makes no wall-time promise.

The default experimental protocol requires at least three unshaped training
`TEST_RESONANCES` sweeps and three held-out transient A/B pairs. Training uses
Klipper's normal behavior of disabling input shaping. Promotion evidence is
instead a readback-verified finite-reversal command followed by a raw,
post-command accelerometer ring-down window: the exact snapshot shaper is
captured as A, then the exact candidate as B, for the same pair ID before
advancing. Reports retain the ordered ledger and pair IDs, and the paired
bootstrap consumes those raw windows in that order. The explicit fast mode is
the sole exception: one training sweep and exactly two A/B pairs, `VALIDATE=1`,
and explicit `HZ_PER_SEC=2`. It remains a held-out 95% attenuation-confidence
test with QC, cross-axis regression, exact readback, and rollback gates, but is
labeled lower confidence. One-repeat experimental validation remains forbidden.

An optional `PEAK_LOCK=1` request is carried through the controller and analysis
boundary only for the two experimental profiles. Analysis selects the highest-PSD
detected mode independently for each axis and restricts the generalized-MZV
frequency search to that one exact bin. This changes candidate generation, not
the validation or restoration protocol.

Training sweeps use Klipper's normal behavior of temporarily disabling input
shaping. Experimental profiles never use a shaped
`TEST_RESONANCES INPUT_SHAPING=1` capture as ring-down validation, promotion
evidence, or acceleration evidence; their promotion gate is the finite-reversal
protocol. Ordinary profiles retain their existing native compatibility workflow.
A candidate is never published when capture, analysis, artifact writing, or
state restoration fails.

For `adaptive_stock`, training-time cross-axis ranking is candidate-specific.
Native candidates weight the measured cross-axis PSD by their exact upstream
frequency-response curves. Generalized MZV candidates use the same measured PSD
with `oscillator_response` and report a conservative 95th-percentile residual
over measured damping uncertainty. This modeled ranking signal never substitutes
for the mandatory measured held-out cross-axis non-regression gate.

Before any held-out transient motion, experimental profiles run a theory-only
spectral non-regression screen against the exact configured snapshot. It uses
installed-Klipper pulse definitions, measured unshaped along- and cross-axis
PSD, damping uncertainty, and the worst meaningful 5-Hz band. It is a
fail-closed preflight screen, not a measured validation or physical-acceleration
claim; the finite-reversal A/B ring-down gate still decides promotion.

The exact-band screen is independent of the 5% smoothing-uplift gate. It
compares installed-source pulse models for the exact baseline and challenger on
both along- and cross-axis training PSD, using measured damping uncertainty and
the worst meaningful 5-Hz band. Passing both theoretical gates still provides
no physical acceleration evidence.

Parameterized identifiers pass through one strict parser shared by analysis and
Klippy. Temporary application reads status back per axis before motion resumes.
Experimental validation also requires an enabled live Klippy axis, non-empty
input-shaper kinematics wrappers, and exact `n/A/T` agreement with installed
`shaper_defs.init_shaper`. These are Python-layer and C-acceptance checks, not a
C-struct executor readback. Rollback uses and verifies the same canonical
snapshot. Unsupported installed
Klipper builds abstain instead of substituting native MZV. The capability record
includes the executor pulse capacity discovered from the installed Klipper
source; optimization never emits a candidate above that capacity or the
project-wide ten-pulse limit.

Before any experimental sweep, compatibility preflight also requires the
installed ten-pulse executor's `p_ind` single-pass C source signatures and
verifies by AST that `InputShaperParams.update()` assigns the validated local
frequency to `self.shaper_freq`. These are source-feature proofs, not live
C-state readback; exact post-command status and Python pulse checks still run
before held-out motion.

The parser and capability foundation intentionally reserve allowlisted
parameter extensions for later work. In particular, upstream-style
`ei(v_tol=...)` is not runtime-exposed in this release; it cannot be passed as
an arbitrary shaper argument.

Generalized MZV is limited to `n=3..10` and exactly one finite spacing argument:
`t>=0.5` with strict `t < (n-1)/2`, or finite `tau>=0.5` whose conversion to
`t` must satisfy that same upstream bound. The executor and installed capability
limits may narrow the accepted set further.

## Stock-compatible adaptive boundary

The project does not emit arbitrary pulse arrays or private shaper names and
does not install a custom executor.
`adaptive_stock` ranks only exact native families plus strictly parsed
generalized MZV. The installed `shaper_defs` implementation must reproduce the
expected generalized pulse sequence and expose a compatible executor capacity
before any adaptive motion. Every temporary result is applied through
`SET_INPUT_SHAPER`, read back from Klipper status, validated with paired
finite-reversal post-command ring-down captures, and rolled back exactly.
