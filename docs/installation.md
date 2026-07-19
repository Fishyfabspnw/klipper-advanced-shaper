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
git clone --branch feature/initial-alpha \
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
4. installs the repository's small Klippy loader.

It does not change `printer.cfg`, restart Klipper, flash an MCU, or modify
Klipper's motion-planner source.

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
```

Only the section header is required. The commented values show the defaults,
except `minimum_max_accel_x` and `minimum_max_accel_y`, which default to `0`
(disabled). Those two options are acceptance gates for a specific printer, not
general safety limits. Do not copy the example values unless they are justified
for your machine.

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

`VALIDATE=1` is the recommended fail-closed path. The plugin restores the
original shaper and velocity state before making an accepted result available.
Inspect the report and use the returned result ID explicitly:

```text
ADV_SHAPER_APPLY RESULT=<result-id>
ADV_SHAPER_STAGE RESULT=<result-id>
SAVE_CONFIG
```

`APPLY` changes only the current runtime. `STAGE` prepares native input-shaper
values, and the separate `SAVE_CONFIG` is what persists them. Do not treat a
reported smoothing-derived acceleration as a mechanically safe machine limit.

## Update

From an unmodified checkout:

```sh
cd ~/klipper-advanced-shaper
./scripts/update.sh
sudo systemctl restart klipper
```

The update script refuses a checkout with tracked local changes, performs a
fast-forward-only pull, and reruns the installer. It never restarts Klipper.

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

Remove `[advanced_input_shaper]` from `printer.cfg` before restarting. The
uninstaller deliberately leaves the Git checkout and private result directory
in place. Delete or archive those separately only after reviewing their
contents.

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
