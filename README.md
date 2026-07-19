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

An offline, research-only optimizer also explores Klipper's parameterized
generalized-MZV design space and a conservative acceleration envelope. It is
not connected to `APPLY` or `STAGE`, and cannot change printer acceleration.
See [experimental generalized MZV](docs/experimental-generalized-mzv.md).

## Quick install

Requirements: a working Klipper host, Python 3.9–3.11, Git, a configured
`[resonance_tester]`, and a connected accelerometer supported by Klipper. Stop
any print and leave the printer idle before installing or restarting Klipper.

SSH into the Klipper host as the account that owns the Klipper installation,
then run:

```sh
cd ~
git clone --branch feature/initial-alpha \
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
ADV_SHAPER_CALIBRATE AXIS=X|Y|ALL PROFILE=quality|balanced|performance REPEATS=3 VALIDATE=1
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
