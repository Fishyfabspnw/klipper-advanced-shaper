# Installation and maintenance

This guide installs the `0.1.0a1` development alpha on a Klipper host. The
software is experimental. Do not install, update, restart, or calibrate during
a print.

## Before you begin

You need:

- shell/SSH access to the Linux host running Klipper;
- Git and a working Klipper installation;
- Python 3.9–3.11 in Klipper's virtual environment;
- a working Klipper `[resonance_tester]` configuration; and
- a connected accelerometer that already works with Klipper's native
  `ACCELEROMETER_QUERY` and `SHAPER_CALIBRATE` workflow.

Ordinary native profiles use that Klipper-supported sensor normally. The
experimental finite-reversal promotion path currently has a strict full-scale
proof allowlist for ADXL345, LIS2DW, and LIS3DH identities. Any other sensor
abstains before transient motion rather than assuming a clipping limit.

Complete Klipper's official
[measuring resonances](https://www.klipper3d.org/Measuring_Resonances.html)
setup first. This plugin uses Klipper's configured resonance motion and sensor;
it does not configure or diagnose the accelerometer itself.

The examples assume the common paths `~/klipper` and `~/klippy-env`. Run the
commands as the account that owns those directories, not as root.

## Install

Clone the current alpha outside the Klipper source tree:

```sh
cd ~
git clone --branch main \
  https://github.com/Fishyfabspnw/klipper-advanced-shaper.git
cd klipper-advanced-shaper
./scripts/install.sh
```

The executable bit is stored in Git. If an archive or file transfer removed it,
restore it with:

```sh
chmod +x scripts/install.sh scripts/update.sh scripts/uninstall.sh
```

For nonstandard installations, pass absolute paths in the environment:

```sh
KLIPPER_DIR=/opt/klipper \
KLIPPER_VENV=/opt/klippy-env \
./scripts/install.sh
```

The installer:

1. checks that Klipper's `klippy/extras` directory and virtual-environment
   Python exist;
2. installs or upgrades this repository in Klipper's virtual environment;
3. preserves a different existing `advanced_input_shaper.py` loader as
   `advanced_input_shaper.py.previous`; and
4. installs the repository's small Klippy loader; and
5. installs `advanced_shaper_macros.cfg` in the printer config directory,
   preserving a different existing macro file as
   `advanced_shaper_macros.cfg.previous`.

It does not change `printer.cfg`, restart Klipper, flash an MCU, or modify
Klipper's motion planner, `shaper_defs.py`, `input_shaper.py`,
`shaper_calibrate.py`, C kinematics helper, or MCU firmware. The loader is a new
Klippy extra; it does not replace a stock Klipper module.

The default macro destination is
`~/printer_data/config/advanced_shaper_macros.cfg`. For a nonstandard printer
config directory, include `KLIPPER_CONFIG_DIR` with the other absolute paths:

```sh
KLIPPER_DIR=/opt/klipper \
KLIPPER_VENV=/opt/klippy-env \
KLIPPER_CONFIG_DIR=/opt/printer_data/config \
./scripts/install.sh
```

## Configure Klipper

Add the following to `printer.cfg`:

```ini
[advanced_input_shaper]
# result_folder: ~/printer_data/config/AdvancedShaper_results
# keep_raw_data: True
# minimum_max_accel_x: 16150
# minimum_max_accel_y: 5840
# analysis_timeout: 600
# worker_memory_mb: 1536
# worker_cpu_seconds: 300
# enable_experimental_generalized_mzv: False

[include advanced_shaper_macros.cfg]
```

Only the `[advanced_input_shaper]` section header is required for the low-level
commands. The include enables the supplied Mainsail-friendly macro. The
commented values show the defaults, except `minimum_max_accel_x` and
`minimum_max_accel_y`, which default to `0` (disabled). Those two options are
acceptance gates for a specific printer, not general safety limits. Do not copy
the example values unless they are justified for your machine.

Leave `enable_experimental_generalized_mzv` false for `quality`, `balanced`, and
`performance`. Set it to `True` only to opt into `experimental_mzv` or
`adaptive_stock`; those profiles still perform an installed-Klipper capability
probe and abstain before experimental motion on an unsupported build.
They also require the installed native fitter to expose upstream
`max_vibrations`; the plugin supplies the profile's finite fitting fraction
(currently 10%) before native-family selection. This is not the held-out 10%
attenuation-improvement gate. `quality`, `balanced`, and `performance` omit the
parameter and retain their legacy native fitting behavior.

With the printer idle, restart the actual Klipper service:

```sh
sudo systemctl restart klipper
```

Some images use a different service name. Find it with
`systemctl list-units --type=service | grep -i klipper`, then restart that unit.
Klipper's G-code `RESTART` command does not reload an updated installed Python
package and is not sufficient after installation or update.

## Verify the installation

Check the analysis boundary from the shell. This does not connect to the MCU,
move the printer, read the accelerometer, or issue G-code:

```sh
~/klippy-env/bin/python -m klipper_advanced_shaper.worker_child \
  --diagnostic --memory-mb 1536 --cpu-seconds 30
```

For a custom virtual environment, replace `~/klippy-env` with its path. Success
prints JSON with `"ok": true` and `"boundary": "external-interpreter"`.

In the Klipper console, run:

```text
ADV_SHAPER_STATUS
```

It should return a JSON status with `"state": "idle"`. If Klipper reports an
unknown command, see troubleshooting below.

Before the first calibration, use Klipper's native command to confirm the
sensor is still available:

```text
ACCELEROMETER_QUERY
```

## First calibration

Home the requested axis, remove obstructions, keep clear of moving machinery,
and make sure the printer is not printing. Begin with one axis:

```text
ADV_SHAPER_CALIBRATE AXIS=X PROFILE=balanced REPEATS=3 VALIDATE=1
```

The visible `ADV_SHAPER_UI_CALIBRATE` macro accepts these same parameters:

| Parameter | Values and behavior |
| --- | --- |
| `AXIS` | `X`, `Y`, or `ALL`; default `ALL`. |
| `PROFILE` | `quality`, `balanced`, `performance`, `experimental_mzv`, or `adaptive_stock`; default `balanced`. The last two require the config opt-in and validation. |
| `REPEATS` | Integer `1..20`; default `3`. Unshaped `TEST_RESONANCES` training sweeps; experimental full validation requires at least three, while fast validation uses one training sweep and two transient A/B pairs. |
| `VALIDATE` | `0` or `1`; default `1`. Enables finite-reversal raw ring-down validation and is mandatory for experimental profiles. |
| `ACCEL_PER_HZ` | `CONFIG` or any unsigned decimal `20..350` mm/s^2/Hz. It is a free numeric value, not a preset list. |
| `HZ_PER_SEC` | `CONFIG` or any unsigned decimal `0.1..2` Hz/s; default `CONFIG`. |
| `SCV` | `CONFIG` or any unsigned decimal `0.1..50` mm/s; default `CONFIG`. A numeric value is temporary, read back before capture, recorded, and restored to the exact snapshot value. |
| `FAST_VALIDATION` | `0` or `1`; default `0`. `1` is experimental-only, requires `REPEATS=2 VALIDATE=1 HZ_PER_SEC=2`, and runs one training sweep plus two interleaved reference/candidate transient pairs. |
| `PEAK_LOCK` | `0` or `1`; default `0`. Experimental-only; locks generalized MZV to the strongest measured axis peak. |

Mainsail's standard macro UI cannot show a different tooltip for each input.
The macro description and its start message summarize the controls; the table
above is the complete reference. Enter `ACCEL_PER_HZ` directly in the macro
parameters or console, for example:

```text
ADV_SHAPER_UI_CALIBRATE AXIS=X PROFILE=balanced ACCEL_PER_HZ=75
```

Before motion, the resolved `ACCEL_PER_HZ` must fit the plugin's 80% dynamic
motion budget. A value inside `20..350` can still be rejected for the current
frequency range and printer `max_accel`.

`VALIDATE=1` is the recommended fail-closed path. The plugin restores the
original shaper and velocity state before making an accepted result available.
Inspect the report and use the returned result ID explicitly:

```text
ADV_SHAPER_APPLY RESULT=<result-id>
ADV_SHAPER_STAGE RESULT=<result-id>
SAVE_CONFIG
```

`APPLY` changes only the current runtime. `STAGE` writes the accepted
stock-Klipper shaper type, frequency, and damping to Klipper's pending config
state, and the separate `SAVE_CONFIG` is what persists them. Calibration never
automatically applies, stages, or saves a result. Neither path changes
`[printer] max_accel`. Do not treat a reported smoothing-derived acceleration
as a mechanically safe machine limit.

Before an experimental run, establish the baseline with normal Klipper
`SHAPER_CALIBRATE` or Shake&Tune. Review, apply, and save the ordinary stock
result using that tool's workflow, restart if required, and verify its type,
frequency, and damping in live Klipper status. Those live values—not a previous
report—are the exact baseline used by the plugin.

`experimental_mzv` is a second-stage challenger. It needs at least 5% more
theoretical smoothing acceleration than both that baseline and the best
eligible stock candidate fitted from the same capture under common residual
gates. It must still pass the exact-band screen and paired transient validation.
If a theoretical gate rejects it, no candidate `SET_INPUT_SHAPER` validation
motion occurs and the report records no upgrade. A later measured rejection
also records no upgrade. Neither offers `APPLY` or `STAGE`; `adaptive_stock`
may retain stock.

For the stock-compatible adaptive search, enable the opt-in and use either the
full-confidence protocol:

```text
ADV_SHAPER_UI_CALIBRATE AXIS=X PROFILE=adaptive_stock REPEATS=3 VALIDATE=1 ACCEL_PER_HZ=CONFIG
```

or the explicitly lower-confidence fast protocol:

```text
ADV_SHAPER_UI_CALIBRATE AXIS=X PROFILE=adaptive_stock REPEATS=2 VALIDATE=1 ACCEL_PER_HZ=175 HZ_PER_SEC=2 SCV=15 FAST_VALIDATION=1
```

Fast validation performs one unshaped training resonance sweep and four short
transient captures per axis. Its duration depends on printer geometry, queued
motion, sensor, and host timing; it intentionally makes no wall-time promise.
Treat this fast protocol as exploratory. Repeatable qualification requires the
standard protocol with at least three training sweeps and three A/B pairs.

## Find the results

The default output location is:

```text
~/printer_data/config/AdvancedShaper_results/<attempt-id>/
```

The plugin creates it only when it writes an accepted report or a
validation-rejected diagnostic. Installation does not create the folder, and a
preflight, capture, early-analysis, artifact-write, or rollback failure may
leave no attempt directory. Run `ADV_SHAPER_STATUS` to see artifact paths once
they exist. See [calibration reports](reports.md) for the file list.

## Update

From an unmodified checkout:

```sh
cd ~/klipper-advanced-shaper
./scripts/update.sh
sudo systemctl restart klipper
```

The update script refuses a checkout with tracked local changes, performs a
fast-forward-only pull, and reruns the installer. This reinstalls both the
Python package and current loader/macro files. It never restarts Klipper or
edits `printer.cfg`.

If you deliberately maintain local changes, review and integrate upstream with
Git yourself, then rerun `./scripts/install.sh`.

Moonraker Update Manager integration is not supported in this alpha. Updating
the checkout alone is insufficient because the package must also be reinstalled
into Klipper's virtual environment before the service restarts. Use
`scripts/update.sh` so those steps happen in the required order.

## Uninstall or restore a previous loader

From the repository checkout, run:

```sh
cd ~/klipper-advanced-shaper
./scripts/uninstall.sh
sudo systemctl restart klipper
```

For custom paths, pass the same `KLIPPER_DIR` and `KLIPPER_VENV` values used for
installation. The uninstaller removes the Python package only after verifying
that the installed loader matches this checkout. If the installer preserved a
pre-existing loader, the uninstaller restores it. It refuses to overwrite or
remove a loader that has since changed.

Remove `[advanced_input_shaper]` and `[include advanced_shaper_macros.cfg]` from
`printer.cfg` before restarting. The uninstaller deliberately leaves the Git
checkout, installed macro file, any `.previous` macro backup, and private result
directory in place. Review, archive, restore, or delete those files manually as
appropriate. The script never deletes printer configuration or results.

## Troubleshooting

### Klipper or its Python environment was not found

Locate the active paths and rerun the script with `KLIPPER_DIR` and
`KLIPPER_VENV` set to absolute paths. On multi-instance installations, verify
which Klipper checkout and virtual environment the target service actually
uses.

### `ADV_SHAPER_STATUS` is an unknown command

Confirm that `[advanced_input_shaper]` is present in the active `printer.cfg`,
rerun the installer against the correct Klipper paths, and restart the host
service—not only the G-code runtime. Then inspect the Klipper log for the first
configuration/import error.

### NumPy or Matplotlib fails to install

Confirm the virtual environment uses Python 3.9–3.11 and that the host has
network access to its configured Python package index. Capture the complete pip
error before installing system build tools; supported platforms normally use
prebuilt packages, but availability varies by OS and CPU architecture.

### No connected resonance accelerometer

Return to Klipper's native resonance setup. `ACCELEROMETER_QUERY` and the native
resonance workflow must work first. Check the sensor section, `[resonance_tester]`,
wiring, MCU connection, and relevant Klipper log messages.

### Unsupported Klipper `resonance_tester` API

The plugin intentionally fails closed when Klipper's private resonance-capture
interface does not match the versions covered by this alpha. Record the Klipper
version, the exact error, and sanitized log context in a GitHub issue. Do not
upload `printer.cfg`, credentials, raw captures, network addresses, or private
reports.

### Update refuses local changes or a non-fast-forward pull

Run `git status` in the checkout. Preserve or commit intentional changes and
resolve branch history manually. The script will not discard local work or
rewrite history.

## Getting help safely

Open an issue at
<https://github.com/Fishyfabspnw/klipper-advanced-shaper/issues> with the plugin
version, Klipper version, host OS/CPU, installation paths (without usernames if
desired), the exact command, and the sanitized error. Follow [the security
policy](../SECURITY.md) for vulnerabilities. Never publish credentials,
`printer.cfg`, raw accelerometer captures, private reports, or network details.
