# Klipper Advanced Shaper

Klipper Advanced Shaper is an experimental, fail-closed calibration and analysis
plugin for [Klipper](https://www.klipper3d.org/). It aims to find better measured
trade-offs between residual vibration, smoothing, repeatability, and usable
acceleration while continuing to use Klipper's native input-shaper execution.
It does **not** modify Klipper's motion planner, `shaper_defs.py`,
`input_shaper.py`, `shaper_calibrate.py`, C kinematics helper, or MCU firmware.
Runtime selections are sent through stock Klipper's `SET_INPUT_SHAPER` command.

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

For a Shake&Tune-style command index, per-macro usage page, and
parameter/default/description tables, start with the
[Advanced Shaper documentation](docs/README.md).

The `adaptive_stock` profile compares native Klipper ZV, MZV, ZVD, EI,
2HUMP_EI, and 3HUMP_EI candidates with the same capability-proven generalized
MZV candidates. It may choose a different supported family for X and Y. Every
possible result is representable by stock `SET_INPUT_SHAPER`; arbitrary pulse
vectors and project-specific shaper names are not accepted. The profile keeps
mandatory held-out validation and never changes printer acceleration.

## Recommended two-stage workflow

1. Run Klipper's normal `SHAPER_CALIBRATE` or a normal Shake&Tune axis-shaper
   calibration. Review it, apply it, and use `SAVE_CONFIG` through that tool's
   documented workflow.
2. Restart as required and confirm the ordinary stock result is active. The
   shaper type, frequency, and damping in live Klipper status are the
   authoritative configured baseline for Advanced Shaper.
3. Run `PROFILE=experimental_mzv` as a challenger. A parameterized MZV must show
   at least 5% more **theoretical smoothing acceleration** than both that exact
   active baseline and the best eligible stock candidate fitted from the same
   training capture. All candidates use the same residual metric and gates.
4. The challenger must then pass the exact configured-baseline spectral band
   screen and mandatory paired held-out ring-down validation. If either
   theoretical gate rejects it, no candidate `SET_INPUT_SHAPER` validation
   motion occurs and the report says no upgrade. A later measured rejection
   also says no upgrade. Neither outcome becomes eligible for `APPLY` or
   `STAGE`.

`adaptive_stock` uses the same strict experimental safety boundary, but may
retain a stock candidate. Fast validation is exploratory; use the standard
protocol with at least three repeats for repeatable qualification.

## Quick install

Requirements: a working Klipper host, Python 3.9–3.11, Git, a configured
`[resonance_tester]`, and a connected accelerometer supported by Klipper. Stop
any print and leave the printer idle before installing or restarting Klipper.

Ordinary native capture uses Klipper's configured accelerometer. Experimental
finite-reversal promotion currently proves full-scale clipping limits only for
ADXL345, LIS2DW, and LIS3DH identities. Other sensors abstain before transient
motion instead of assuming a range.

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
package into `~/klippy-env`, adds the small `advanced_input_shaper.py` loader to
`~/klipper/klippy/extras`, and copies `advanced_shaper_macros.cfg` to
`~/printer_data/config`. It does not replace any stock Klipper module. Custom
Klipper locations, updating, uninstalling, verification, and troubleshooting
are covered in the
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
ADV_SHAPER_CALIBRATE AXIS=X|Y|ALL PROFILE=quality|balanced|performance|experimental_mzv|adaptive_stock REPEATS=3 VALIDATE=1 ACCEL_PER_HZ=CONFIG|20..350 HZ_PER_SEC=CONFIG|0.1..2 SCV=CONFIG|0.1..50 FAST_VALIDATION=0|1 PEAK_LOCK=0|1
ADV_SHAPER_STATUS
ADV_SHAPER_CANCEL
ADV_SHAPER_APPLY RESULT=<id>
ADV_SHAPER_STAGE RESULT=<id>
```

### Calibration parameters

The low-level command and the supplied Mainsail macro accept the same
calibration parameters:

| Parameter | Accepted values | Default | Meaning and restrictions |
| --- | --- | --- | --- |
| `AXIS` | `X`, `Y`, or `ALL` | `ALL` | Calibrates one axis or X followed by Y. Every requested axis must be homed. |
| `PROFILE` | `quality`, `balanced`, `performance`, `experimental_mzv`, or `adaptive_stock` | `balanced` | The first three retain the ordinary native analysis path. `experimental_mzv` treats generalized MZV only as a challenger to the active configured baseline and same-capture stock candidates. `adaptive_stock` may retain one of the six stock families. The last two require explicit opt-in and mandatory held-out validation. |
| `REPEATS` | Integer `1` through `20` | `3` | Unshaped `TEST_RESONANCES` training sweeps. Experimental profiles require at least three. Fast validation uses one training sweep and exactly two paired transient captures per condition. |
| `VALIDATE` | `0` or `1` | `1` | When `1`, runs the mandatory finite-reversal, raw ring-down A/B validation for experimental profiles. It is not another resonance sweep. A `0` run is not physical performance evidence. |
| `ACCEL_PER_HZ` | `CONFIG` or any unsigned decimal from `20` through `350` | `CONFIG` | Free numeric excitation control in mm/s^2/Hz—not presets. `CONFIG` inherits `[resonance_tester]`. The resolved value must pass the dynamic motion-budget check. |
| `HZ_PER_SEC` | `CONFIG` or any unsigned decimal from `0.1` through `2` | `CONFIG` | Sweep rate in Hz/s. It changes commanded sweep time, not excitation intensity. |
| `SCV` | `CONFIG` or any unsigned decimal from `0.1` through `50` | `CONFIG` | Temporary square-corner velocity in mm/s used by smoothing calculations. It is applied only after the exact printer snapshot, verified by readback, recorded in the report, and restored exactly. |
| `FAST_VALIDATION` | `0` or `1` | `0` | Lower-confidence mode for the two experimental profiles only. `1` requires exactly `REPEATS=2`, `VALIDATE=1`, and explicit `HZ_PER_SEC=2`; it runs one training sweep and two interleaved reference/candidate transient pairs. |
| `PEAK_LOCK` | `0` or `1` | `0` | Experimental profiles only. `1` fixes generalized-MZV frequency to the strongest measured PSD mode for that axis; it does not weaken any validation gate. |

Run calibration only while the printer is idle, clear of obstructions, and
homed on every requested axis. Start with one axis and the conservative default
profile:

```text
ADV_SHAPER_CALIBRATE AXIS=X PROFILE=balanced REPEATS=3 VALIDATE=1
```

`ADV_SHAPER_UI_CALIBRATE` is the only macro shown in Mainsail's macro panel.
Enter `ACCEL_PER_HZ` as any number from 20 through 350, or enter `CONFIG`; the
macro deliberately does not force a list of presets. Mainsail's standard macro
UI cannot attach a separate tooltip to each input. The macro description, its
start message, and the table above provide the parameter explanations. The
supplied macro does not use Macro Prompt and does not require Klipper's optional
`[respond]` section. Supporting wrappers use a leading underscore and remain
hidden.
Klippy's low-level `ADV_SHAPER_STATUS`, `ADV_SHAPER_CANCEL`,
`ADV_SHAPER_APPLY`, and `ADV_SHAPER_STAGE` commands remain directly callable.

The same free numeric control is available from the console, for example:

```text
ADV_SHAPER_UI_CALIBRATE AXIS=X PROFILE=adaptive_stock REPEATS=2 VALIDATE=1 ACCEL_PER_HZ=175 HZ_PER_SEC=2 SCV=15 FAST_VALIDATION=1
```

`ACCEL_PER_HZ` accepts `CONFIG` or an unsigned decimal from 20 through 350
mm/s^2/Hz inclusive. `CONFIG` inherits `[resonance_tester]` without overriding
it. Signs, exponent notation, leading-zero ambiguity, whitespace, non-finite
values, junk, and out-of-range numbers are rejected before preflight. The
effective inherited value must also be within 20..350.

`HZ_PER_SEC` independently accepts `CONFIG` or an unsigned decimal from 0.1
through 2 Hz/s inclusive, matching the installed upstream Klipper limit.
`CONFIG` inherits `[resonance_tester]`. Signs, exponent notation, ambiguous
leading zeroes, whitespace, non-finite values, junk, and out-of-range rates are
rejected before motion. The resolved rate is passed to every training sweep and
recorded in its capture recipe and the result report. Experimental held-out
validation uses finite reversals, not sweeps, so `HZ_PER_SEC` does not control
those transient captures.

Before snapshot or motion, the plugin computes the maximum pulse excitation as
`max_freq * accel_per_hz`, adds Klipper's configured sweeping acceleration, and
requires the estimated peak to fit within 80% of the printer's current
`max_accel`. Missing or non-finite recipe/status fields fail closed. This margin
does not certify mechanical safety; supervise the test, keep clear of moving
machinery, and be ready to use `M112`.

Increasing `ACCEL_PER_HZ` may raise measured accelerometer response and PSD, but
it does not directly set the graph scale and does not guarantee a PSD above
`1e-5`. Excessive excitation can instead cause sensor clipping, skipped steps,
or hardware damage. A numeric value of 350 is accepted only when the dynamic
motion-budget preflight also passes.

The hidden UI apply and stage wrappers default to the current accepted result
ID. The low-level commands remain explicit. Both fail if no accepted result is
ready. A rejected or failed attempt clears that default, even if an older
accepted result remains in process memory.

The Python controller independently enforces the mandatory validation and
repeat count for `experimental_mzv` and `adaptive_stock`; macro parameters
cannot weaken them. The calibration macro never invokes apply, stage, or
`SAVE_CONFIG`.

`PEAK_LOCK=1` is available only to `experimental_mzv` and `adaptive_stock`. It
fixes every generalized-MZV design considered for an axis to that axis's
strongest measured PSD mode (the detected mode with the highest PSD amplitude),
while still optimizing the strict allowlisted `n` and `t` parameters. The exact
target frequency and strategy are recorded in the report. It does not skip
capability preflight, held-out reference/candidate transient captures, exact status
readback, QC, confidence, cross-axis regression, or rollback.

`APPLY` is runtime-only. `STAGE` writes the accepted stock-Klipper shaper type,
frequency, and damping to Klipper's pending config state; the operator must
separately invoke `SAVE_CONFIG` to persist them. Neither command changes
`[printer] max_accel`. Calibration never automatically applies, stages, saves,
or changes heater, fan, motor-current, or persistent acceleration settings.

For experimental profiles with `REPEATS=3 VALIDATE=1`, each axis uses three unshaped fitting
`TEST_RESONANCES` sweeps followed by three interleaved A/B finite-reversal
ring-down pairs. Each pair captures the exact configured reference and the
temporary candidate with raw accelerometer windows after the command. The
candidate is accepted only when paired QC and window fairness, the modal
attenuation confidence interval, cross-axis regression, and measured total-band
and meaningful 5-Hz-band non-regression all pass; every temporary setting is
restored before a result becomes reviewable. `ACCEL_PER_HZ` changes excitation
intensity, not sweep duration.

The full-confidence experimental workflow has three training resonance sweeps
and six short transient validation captures per axis. `AXIS=ALL` repeats that
per-axis workflow. It must not be described as nine or eighteen resonance
sweeps: the transient captures are separate finite-reversal ring-down evidence.

For `experimental_mzv`, an explicit lower-confidence fast path is available:

```text
ADV_SHAPER_CALIBRATE AXIS=X PROFILE=experimental_mzv REPEATS=2 VALIDATE=1 ACCEL_PER_HZ=CONFIG HZ_PER_SEC=2 FAST_VALIDATION=1 PEAK_LOCK=1
```

The same bounded protocol can run the cross-family stock-compatible search:

```text
ADV_SHAPER_CALIBRATE AXIS=ALL PROFILE=adaptive_stock REPEATS=2 VALIDATE=1 ACCEL_PER_HZ=175 HZ_PER_SEC=2 SCV=15 FAST_VALIDATION=1 PEAK_LOCK=1
```

This performs one unshaped training resonance sweep, then two short,
interleaved reference/candidate finite-reversal ring-down pairs. The raw
accelerometer windows are the promotion evidence; they are not resonance sweeps.
The transient time depends on printer geometry, queued-motion timing, sensor,
and host, so no wall-time promise is made. The faster rate and two-repeat
confidence interval trade spectral and statistical confidence for time; all QC,
95% attenuation CI, cross-axis regression, exact readback, and rollback gates
remain fail-closed.
`FAST_VALIDATION=1` never reduces either held-out group below two captures,
accepts neither one nor three for `REPEATS`, requires explicit
`HZ_PER_SEC=2`, and never makes a rejected result eligible for apply or stage.
The default experimental path remains at least three repeats.

Validation-rejected attempts retain diagnostic artifacts only after successful
rollback and never become eligible for apply or stage.

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

`PROFILE=experimental_mzv` and `PROFILE=adaptive_stock` additionally require
the config opt-in shown above, `VALIDATE=1`, and either the full-confidence or
explicit fast repeat protocol. Before any experimental sweep, the plugin probes
the installed `shaper_defs` implementation and generic executor. Current stock
Klipper builds that provide the required allowlisted APIs work without core
patches. Experimental preflight also requires raw per-axis Klippy parameters;
a rounded status-only interface cannot support exact snapshot restoration.
Older, incompatible, or vendor-modified builds abstain; there is no silent
compatibility fallback. These two profiles also ask upstream Klipper's
native fitter for a profile-derived `max_vibrations` limit (currently 10%).
That is a per-family frequency-fitting constraint, not the separate held-out
10% attenuation-improvement gate; ordinary profiles omit it and retain legacy
native fitting. Installed Klipper must explicitly support the parameter or
preflight abstains before motion. After every temporary `SET_INPUT_SHAPER`,
Klipper status must read back the exact canonical axis, identifier, frequency,
and damping before validation can continue. Experimental validation additionally
checks the live Klippy axis state is enabled and that its `n/A/T` pulse arrays
match the installed `shaper_defs.init_shaper` result on active kinematics. This
is still not a readback of Klipper's private C executor structure.

Canonical generalized MZV accepts only `n=3..10` and exactly one of `t` or
`tau`. Both spacing forms must be finite and at least `0.5`. Direct `t` also
has the strict upstream upper bound `t < (n-1)/2`; `tau` is converted to `t`
and the converted value must satisfy the same upstream constraint. Unknown,
duplicate, positional, mixed, signed, exponent, and non-finite arguments fail
closed.

A candidate rejected by held-out validation is never available to `APPLY` or
`STAGE`. After the original printer state has been restored successfully, its
private report retains the validation metrics and, when `keep_raw_data` is
enabled, the training, reference, and candidate captures for diagnosis.

## Results and output files

Reports are private local files under:

```text
~/printer_data/config/AdvancedShaper_results/<attempt-id>/
```

The directory is created when an accepted report or a validation-rejected
diagnostic is written; installation alone does not create it. A run that fails
during preflight, capture, early analysis, artifact writing, or rollback may
correctly have no attempt directory. `ADV_SHAPER_STATUS` reports artifact paths
after they exist. Each completed report includes `report.html`, JSON and
manifest data, PNG/SVG graphs, candidate and validation CSV files when data is
available, and `captures.npz` when `keep_raw_data: True`. See
[calibration reports](docs/reports.md).

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
