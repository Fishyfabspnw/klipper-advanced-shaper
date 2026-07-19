# Experimental generalized MZV research

This project can research a wider part of the input-shaper design space without
changing Klipper's motion planner. Current upstream Klipper accepts a
parameterized form of MZV such as `mzv(n=4,t=0.8)`. Klipper converts that name,
frequency, and damping ratio into a positive pulse sequence and sends at most
ten pulses to its existing generic input-shaper executor.

This is best described as **generalized MZV optimization**, not a new Klipper
shaper family. The plugin does not patch, replace, or monkey-patch Klipper's
`shaper_defs`, `input_shaper`, or C kinematics code.

Pulse-generation parity is frozen against upstream Klipper commit
`7046bd00ef5c30dec6febc724f8d22967433c45c`. The runtime capability probe is
still mandatory because older or vendor-modified Klipper builds may not accept
parameterized shaper names.

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

Research candidates are not accepted results and cannot be applied or staged.
A future executable gate must prove the running Klipper version can parse and
realize the exact parameterized name, complete held-out shaped validation, and
still require explicit user application.

## Non-inflating acceleration envelope

The experimental acceleration envelope is the minimum of the native-compatible
smoothing/path-error bound, conservative repeatability and model-uncertainty
bounds, and any available vibration-confidence, hardware, or print bound. A
vibration-confidence bound participates only when a held-out resonance test
recorded the actual acceleration. Normalized vibration attenuation is not
converted into an acceleration because that conversion would not be
dimensionally justified.

Every envelope is labeled:

- `theoretical`: pulse-response and smoothing model only;
- `resonance_validated`: an explicit-acceleration held-out resonance test also
  limits the result; or
- `print_validated`: a supervised print-quality test also limits the result.

The formula never changes `[printer] max_accel`. A result can exceed a stock
shaper's estimate only when the optimized pulse design itself has less modeled
path error while still passing robust residual-vibration gates. The formula
cannot manufacture a higher number by relaxing Klipper's `0.12` smoothing
criterion.

## Upstream implementation references

- [Klipper shaper definitions](https://github.com/Klipper3d/klipper/blob/master/klippy/extras/shaper_defs.py)
- [Klipper input-shaper integration](https://github.com/Klipper3d/klipper/blob/master/klippy/extras/input_shaper.py)
- [Klipper generic pulse executor](https://github.com/Klipper3d/klipper/blob/master/klippy/chelper/kin_shaper.c)
- [Klipper calibration and smoothing estimate](https://github.com/Klipper3d/klipper/blob/master/klippy/extras/shaper_calibrate.py)
