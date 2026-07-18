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

The project does not change Klipper's motion planner, shaper impulse generation,
kinematics, MCU firmware, or safety limits. It does not infer mechanically safe
acceleration from smoothing alone. It does not automatically modify heaters,
fans, motor currents, persistent velocity limits, or configuration files.

## Analysis contract

Capture quality must be checked before selection. The robust engine accounts for
sample timing, excitation-axis and cross-axis response, repeat consistency,
multiple modes, and uncertainty. Candidate selection returns either a supported
native shaper and frequency or an explicit abstention with reasons.

Native parity results and robust results remain distinguishable in artifacts.
Supported production candidates are ZV, MZV, EI, 2HUMP_EI, and 3HUMP_EI. ZVD
must remain experimental until runtime support and parity are verified.

## Klipper compatibility boundary

Klipper does not publish a stable third-party accelerometer capture ABI. The
capture adapter fingerprints the native `resonance_tester._run_test` signature,
uses Klipper's own bounded resonance motion and shaper candidate evaluator, and
refuses to run when the expected interface changes. All access to this private
surface is isolated in `klippy/capture.py`; the numerical engine and reports do
not import Klipper.

Training sweeps use Klipper's normal behavior of temporarily disabling input
shaping. Held-out reference and candidate sweeps retain their temporary native
shapers, so acceptance compares the proposal against the shaper active when the
session began. A
candidate is never published when capture, analysis, artifact writing, or state
restoration fails.
