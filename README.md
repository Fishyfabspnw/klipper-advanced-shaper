# Klipper Advanced Shaper

Klipper Advanced Shaper is an experimental, fail-closed calibration and analysis
plugin for [Klipper](https://www.klipper3d.org/). It aims to find better measured
trade-offs between residual vibration, smoothing, repeatability, and usable
acceleration while continuing to use Klipper's native input-shaper execution.
It does **not** replace or modify Klipper's motion planner, kinematics, or MCU
code.

> **Alpha safety notice:** `0.1.0a1` has not been validated across printer
> architectures and must not be treated as proof that a printer can safely run
> at a reported acceleration. Keep clear of moving machinery, supervise tests,
> and use conservative mechanical limits.

## Design goals

- Compare candidates using held-out accelerometer captures, not only fitted data.
- Abstain when capture quality or statistical confidence is inadequate.
- Restore native shaper and velocity state after every calibration path.
- Require an explicit command before applying or staging a result.
- Keep raw accelerometer data private unless its owner deliberately publishes it.

The project makes no blanket claim to be “100% better” than Klipper. A result is
better only when a matched, repeatable benchmark demonstrates both a higher
smoothing-derived acceleration estimate and lower held-out residual vibration.
See [the benchmark protocol](docs/benchmarking.md).

An explicit, disabled-by-default `experimental_mzv` profile explores Klipper's
parameterized generalized-MZV design space. It becomes runtime-applicable only
after the installed Klipper proves exact support and mandatory held-out
validation passes. The standalone optimizer remains research-only, and neither
path changes printer acceleration.
See [experimental generalized MZV](docs/experimental-generalized-mzv.md).

The `adaptive_stock` profile compares native Klipper ZV, MZV, ZVD, EI,
2HUMP_EI, and 3HUMP_EI candidates with the same capability-proven generalized
MZV candidates. It may choose a different supported family for X and Y. Every
possible result is representable by stock `SET_INPUT_SHAPER`; arbitrary pulse
vectors and project-specific shaper names are not accepted. The profile keeps
mandatory held-out validation and never changes printer acceleration.

## Quick install

Requirements: a working Klipper host, Python 3.9–3.11, Git, a configured
`[resonance_tester]`, and a connected accelerometer supported by Klipper. Stop
any print and leave the printer idle before installing or restarting Klipper.

SSH into the Klipper host as the account that owns the Klipper installation,
then run:

```sh
cd ~
git clone --branch main \
  https://github.com/Fishyfabspnw/klipper-advanced-shaper.git
cd klipper-advanced-shaper
./scripts/install.sh
```

Add this section to `printer.cfg`:

```ini
[advanced_input_shaper]
# result_folder: ~/printer_data/config/AdvancedShaper_results
# keep_raw_data: True
# minimum_max_accel_x: 16150  # Optional target-printer acceptance gate
# minimum_max_accel_y: 5840   # Optional target-printer acceptance gate
# enable_experimental_generalized_mzv: False  # Explicit runtime opt-in

# The installer places this file in the printer config directory.
[include advanced_shaper_macros.cfg]
```

Restart the **host service** while the printer is idle:

```sh
sudo systemctl restart klipper
```

Klipper's G-code `RESTART` command is not enough after installing Python code.
Confirm that `ADV_SHAPER_STATUS` is recognized before attempting a calibration.

The installer does not restart Klipper or edit `printer.cfg`. It installs the
package into `~/klippy-env` and a small loader into `~/klipper/klippy/extras`.
Custom Klipper locations, updating, uninstalling, verification, and
troubleshooting are covered in the
**[complete installation guide](docs/installation.md)**.

Before calibration, the analysis interpreter boundary can be checked without
connecting to the MCU or commanding any printer motion:

```sh
~/klippy-env/bin/python -m klipper_advanced_shaper.worker_child \
  --diagnostic --memory-mb 1536 --cpu-seconds 30
```

Success prints JSON containing `"ok": true`, `"boundary":
"external-interpreter"`, and a completed NumPy sample count. This diagnostic
does not load Klippy, read accelerometer data, or issue G-code.

## Commands

```text
ADV_SHAPER_CALIBRATE AXIS=X|Y|ALL PROFILE=quality|balanced|performance|experimental_mzv|adaptive_stock REPEATS=3 VALIDATE=1 ACCEL_PER_HZ=CONFIG|20..150 HZ_PER_SEC=CONFIG|0.1..2 FAST_VALIDATION=0|1 PEAK_LOCK=0|1
ADV_SHAPER_STATUS
ADV_SHAPER_CANCEL
ADV_SHAPER_APPLY RESULT=<id>
ADV_SHAPER_STAGE RESULT=<id>
```

Run calibration only while the printer is idle, clear of obstructions, and
homed on every requested axis. Start with one axis and the conservative default
profile:

```text
ADV_SHAPER_CALIBRATE AXIS=X PROFILE=balanced REPEATS=3 VALIDATE=1
```

`ADV_SHAPER_UI_CALIBRATE` is the only macro shown in Mainsail's macro panel. It
is an ordinary parameterized macro so an operator can enter a numeric
`ACCEL_PER_HZ`; Mainsail's official Macro Prompt protocol provides buttons but
no free-text input control, so this workflow deliberately does not fake a
dropdown. The supplied macro does not use that prompt protocol and therefore
does not require Klipper's optional `[respond]` section. Supporting wrappers use
a leading underscore and remain hidden.
Klippy's low-level `ADV_SHAPER_STATUS`, `ADV_SHAPER_CANCEL`,
`ADV_SHAPER_APPLY`, and `ADV_SHAPER_STAGE` commands remain directly callable.

`ACCEL_PER_HZ` accepts `CONFIG` or an unsigned decimal from 20 through 150
mm/s^2/Hz inclusive. `CONFIG` inherits `[resonance_tester]` without overriding
it. Signs, exponent notation, leading-zero ambiguity, whitespace, non-finite
values, junk, and out-of-range numbers are rejected before preflight. The
effective inherited value must also be within 20..150.

`HZ_PER_SEC` independently accepts `CONFIG` or an unsigned decimal from 0.1
through 2 Hz/s inclusive, matching the installed upstream Klipper limit.
`CONFIG` inherits `[resonance_tester]`. Signs, exponent notation, ambiguous
leading zeroes, whitespace, non-finite values, junk, and out-of-range rates are
rejected before motion. The resolved rate is passed to every training,
held-out-reference, and candidate sweep and recorded in capture recipes and the
result report.

Before snapshot or motion, the plugin computes the maximum pulse excitation as
`max_freq * accel_per_hz`, adds Klipper's configured sweeping acceleration, and
requires the estimated peak to fit within 80% of the printer's current
`max_accel`. Missing or non-finite recipe/status fields fail closed. This margin
does not certify mechanical safety; supervise the test, keep clear of moving
machinery, and be ready to use `M112`.

Increasing `ACCEL_PER_HZ` may raise measured accelerometer response and PSD, but
it does not directly set the graph scale and does not guarantee a PSD above
`1e-5`. Excessive excitation can instead cause sensor clipping, skipped steps,
or hardware damage. A numeric value of 150 is accepted only when the dynamic
motion-budget preflight also passes.

The UI apply and stage actions default to the current accepted result ID. They
fail if no accepted result is ready. A rejected or failed attempt clears that
default, even if an older accepted result remains in process memory.

The Python controller independently enforces the mandatory validation and
repeat count for `experimental_mzv` and `adaptive_stock`; macro parameters cannot weaken them. The
calibration macro never invokes apply, stage, or `SAVE_CONFIG`.

`PEAK_LOCK=1` is available only to `experimental_mzv` and `adaptive_stock`. It fixes every
generalized-MZV design considered for an axis to that axis's strongest measured
PSD mode (the detected mode with the highest PSD amplitude), while still
optimizing the strict allowlisted `n` and `t` parameters. The exact target
frequency and strategy are recorded in the report. It does not skip capability
preflight, held-out reference/candidate sweeps, exact status readback, QC,
confidence, cross-axis regression, or rollback.

`APPLY` is runtime-only. `STAGE` prepares accepted native input-shaper values;
the operator must separately invoke Klipper's `SAVE_CONFIG`. Calibration never
automatically changes heater, fan, motor-current, or persistent acceleration
settings.

With `VALIDATE=1`, each axis uses three fitting sweeps, three independent
held-out sweeps using the shaper active at session start, and three sweeps using
the proposed shaper. To compare directly against a Shake&Tune result, apply that
reference result for the runtime before starting the session. The
candidate is accepted only when the 95% confidence interval demonstrates at
least 10% resonant-band attenuation. Every temporary setting is restored before
the result becomes reviewable.

The full-confidence validated workflow (`REPEATS=3 VALIDATE=1`) performs nine
full resonance sweeps per axis: three training, three held-out reference, and
three candidate sweeps. `AXIS=ALL` performs 18 total sweeps. `ACCEL_PER_HZ`
changes excitation intensity, not sweep duration; duration is primarily set by
the configured frequency range and `HZ_PER_SEC`.

For `experimental_mzv`, an explicit lower-confidence fast path is available:

```text
ADV_SHAPER_CALIBRATE AXIS=X PROFILE=experimental_mzv REPEATS=2 VALIDATE=1 ACCEL_PER_HZ=CONFIG HZ_PER_SEC=2 FAST_VALIDATION=1 PEAK_LOCK=1
```

The same bounded protocol can run the cross-family stock-compatible search:

```text
ADV_SHAPER_CALIBRATE AXIS=ALL PROFILE=adaptive_stock REPEATS=2 VALIDATE=1 ACCEL_PER_HZ=150 HZ_PER_SEC=2 FAST_VALIDATION=1 PEAK_LOCK=1
```

This performs two training, two held-out reference, and two candidate sweeps.
For a 5–135 Hz range, the six physical sweeps are approximately 6.5 minutes per
axis at 2 Hz/s. That estimate excludes movement between probe points, sensor
setup, host analysis, report rendering, and artifact I/O. The faster rate and
two-repeat confidence interval trade spectral and statistical confidence for
time; all QC, 95% attenuation CI, cross-axis regression, exact readback, and
rollback gates remain fail-closed. `FAST_VALIDATION=1` accepts neither one nor
three repeats, requires explicit `HZ_PER_SEC=2`, and never makes a rejected
result eligible for apply or stage. The default experimental path remains at
least three repeats.

Observed operational note: live attempt `9a822d6fdc4b` completed all 18 sweeps,
was rejected safely by validation, and reported restoration of the baseline.

For a supervised native-profile capture-path smoke test, an operator may use:

```text
ADV_SHAPER_CALIBRATE AXIS=X PROFILE=balanced REPEATS=1 VALIDATE=0 ACCEL_PER_HZ=30 HZ_PER_SEC=2
```

That single-sweep native test has no held-out comparison and must not be treated
as an accepted performance validation or automatically applied/staged. Both
adaptive profiles deliberately reject this shortcut: they always require
held-out validation and never permit one repeat. Their full-confidence
default requires `REPEATS>=3`; only the explicit fast protocol permits exactly
two repeats.

`PROFILE=experimental_mzv` and `PROFILE=adaptive_stock` additionally require the config opt-in shown above,
`VALIDATE=1`, and either the full-confidence or explicit fast repeat protocol.
Before any experimental sweep, the
plugin probes the installed `shaper_defs` implementation. Legacy or
vendor-modified builds without the exact parameterized parser abstain; there is
no silent compatibility fallback. After every temporary `SET_INPUT_SHAPER`, Klipper status
must read back the exact canonical axis, identifier, frequency, and damping
before validation can continue.

A candidate rejected by held-out validation is never available to `APPLY` or
`STAGE`. After the original printer state has been restored successfully, its
private report retains the validation metrics and, when `keep_raw_data` is
enabled, the training, reference, and candidate captures for diagnosis.

## Development

Python 3.9–3.11 is supported. Create a virtual environment, then run:

```sh
python -m pip install -e '.[test]'
python -m pytest
python -m build
python scripts/verify_public_tree.py
```

The numerical tests use generated signals only. Do not add real printer
captures, generated reports, printer configuration, login material, or local
automation state to this repository.

## References and relationship to other projects

The implementation is informed by Klipper's public documentation and source:

- [Measuring resonances](https://www.klipper3d.org/Measuring_Resonances.html)
- [Resonance compensation](https://www.klipper3d.org/Resonance_Compensation.html)
- [Klipper shaper calibration source](https://github.com/Klipper3d/klipper/blob/master/klippy/extras/shaper_calibrate.py)
- [Shake&Tune](https://github.com/Frix-x/klippain-shaketune)

This is an independent implementation, not a Shake&Tune fork. No Shake&Tune or
Klipper source is currently vendored. Any future adapted code must retain its
license notices and be recorded in `docs/third-party.md`.

## License

Copyright © 2026 Fishyfabspnw. Licensed under GPL-3.0-only. See [LICENSE](LICENSE).
