# Matched benchmark protocol

A higher smoothing-derived acceleration estimate is not enough to establish a
better result. Compare the advanced recommendation and reference recommendation
under identical conditions and evaluate held-out captures.

## Controlled conditions

Record the printer and Klipper versions, accelerometer type and mounting, toolhead
mass state, probe location, belt state, square-corner velocity, excitation
settings, temperature, and fan state. Do not compare runs with different values
without a documented normalization method.

Use at least three fitting captures and three separate validation captures per
axis and candidate. Run validation as within-axis, readback-verified interleaved
reference/candidate pairs and retain pair IDs so time drift does not become an
untracked block-order confounder. Reject runs with
sensor clipping, timing dropout, inadequate Nyquist margin, excess noise,
Klipper timing faults, MCU faults, or failed state restoration.

## Acceptance metrics

For the initial target printer, the reference thresholds are:

| Axis | Reference | Estimated acceleration | Modeled vibration |
| --- | --- | ---: | ---: |
| X | MZV at 74.4 Hz | 16,150 mm/s² | 1.4% |
| Y | 2HUMP_EI at 76.4 Hz | 5,840 mm/s² | To be measured from raw capture |

For a separate target-printer benchmark, declare a benchmark pass only if all
of the following hold:

- Its acceleration estimate exceeds the axis reference under the same smoothing
  model and square-corner velocity.
- Held-out integrated resonant-band energy is at least 10% lower, with a 95%
  bootstrap confidence interval across three or more validation runs.
- Cross-axis energy does not regress by more than 5%.
- Every included capture passes quality checks and the calibration restores the
  printer state without errors.

These target-printer reference values are not plugin acceptance gates unless
they are configured through `minimum_max_accel_x` and `minimum_max_accel_y`.
An accepted calibration alone is not a benchmark pass.

Report theoretical shaping acceleration separately from acceleration actually
validated at a recorded resonance-test acceleration and from acceleration
validated by a supervised print. A configured global maximum, smoothing model,
or normalized attenuation fraction is not evidence that the mechanics can
safely sustain an acceleration. The calibration does not change
`[printer] max_accel`; acceleration changes require a separate operator decision and
separate printer/print validation.
