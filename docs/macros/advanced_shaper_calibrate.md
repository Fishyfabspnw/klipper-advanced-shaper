# Input shaper calibration

The `ADV_SHAPER_UI_CALIBRATE` macro measures axis resonance data, evaluates
only shapers executable by the installed stock Klipper build, and optionally
performs a matched held-out comparison before returning a reviewable result.
Calibration never automatically applies, stages, saves, or changes
`[printer] max_accel`.

## Usage

Before starting:

- finish Klipper's native accelerometer and `[resonance_tester]` setup;
- confirm `ACCELEROMETER_QUERY` and native resonance testing work;
- stop the print, home every requested axis, and clear the motion area;
- inspect belts, fasteners, frame, toolhead, and accelerometer mounting; and
- keep the printer supervised with `M112` available.

Run the visible Mainsail macro or enter the same command in the console:

```text
ADV_SHAPER_UI_CALIBRATE AXIS=X PROFILE=balanced REPEATS=3 VALIDATE=1 ACCEL_PER_HZ=CONFIG HZ_PER_SEC=CONFIG FAST_VALIDATION=0 PEAK_LOCK=0
```

Here are the available parameters:

| parameter | default value | description |
| --- | --- | --- |
| `AXIS` | `ALL` | Axis to test: `X`, `Y`, or `ALL`. Testing both axes performs each axis workflow separately. |
| `PROFILE` | `balanced` | Selection tradeoff: `quality`, `balanced`, `performance`, `experimental_mzv`, or `adaptive_stock`. The last two require the explicit config opt-in and mandatory validation. |
| `REPEATS` | `3` | Captures in each training/reference/candidate group. Experimental profiles require at least three, except the exact two-repeat fast protocol. |
| `VALIDATE` | `1` | `1` performs held-out reference and candidate captures; `0` skips them. Experimental profiles always require `1`. |
| `ACCEL_PER_HZ` | `CONFIG` | Free numeric excitation intensity: `CONFIG` or any unsigned decimal from `20` through `150` mm/s²/Hz. This is not a preset list. Higher is not automatically better and must pass the motion-budget check. |
| `HZ_PER_SEC` | `CONFIG` | Sweep rate: `CONFIG` or any unsigned decimal from `0.1` through `2` Hz/s. A faster rate shortens motion time but can reduce measurement confidence. |
| `FAST_VALIDATION` | `0` | `1` selects the lower-confidence experimental protocol and requires exactly `REPEATS=2 VALIDATE=1 HZ_PER_SEC=2`. It does not remove QC, confidence, readback, cross-axis, or rollback gates. |
| `PEAK_LOCK` | `0` | Experimental-only. `1` fixes generalized-MZV frequency to the strongest measured mode on that axis while still optimizing allowlisted pulse count and spacing. |

> **Mainsail note**
>
> Mainsail's standard macro parameter UI cannot attach a different tooltip to
> each input. The macro description and start message provide a short legend;
> the table above is the complete reference. `ACCEL_PER_HZ` remains a number
> you enter directly, like Shake&Tune's calibration parameter—not fixed
> presets.

## Choosing a profile

| profile | behavior |
| --- | --- |
| `quality` | Prefers lower residual vibration and retains the ordinary native candidate path. |
| `balanced` | Balances residual vibration, smoothing, repeatability, and cross-axis response on the ordinary native path. |
| `performance` | Gives more weight to the theoretical smoothing acceleration while retaining residual-vibration gates. |
| `experimental_mzv` | Searches only strict canonical `mzv(n=...,t=...)` or `mzv(n=...,tau=...)` candidates supported by the installed Klipper build. |
| `adaptive_stock` | Compares native ZV, MZV, ZVD, EI, 2HUMP_EI, and 3HUMP_EI with capability-proven generalized MZV. It may keep a native winner. |

The ordinary profiles do not silently add ZVD or parameterized shapers to
Klipper's default candidate set. The six-family override is used only by the
explicit `adaptive_stock` profile.

## Fast experimental example

```text
ADV_SHAPER_UI_CALIBRATE AXIS=X PROFILE=adaptive_stock REPEATS=2 VALIDATE=1 ACCEL_PER_HZ=150 HZ_PER_SEC=2 FAST_VALIDATION=1 PEAK_LOCK=1
```

For a configured 5–135 Hz range, this commands approximately 6.5 minutes of
resonance motion per axis. Probe movement, setup, status readback, analysis,
rendering, and artifact I/O add time, so it is not a promise that the complete
workflow finishes within seven minutes.

## Reading the outcome

Do not judge a candidate from the tallest PSD peak or theoretical acceleration
alone. Review:

- capture QC and repeatability;
- residual vibration and the 95% held-out attenuation confidence interval;
- cross-axis regression;
- smoothing/path-error tradeoff;
- exact applied-parameter readback and rollback status; and
- whether acceleration is labeled theoretical, resonance-validated, or
  print-validated.

Input shaping reduces ringing caused by measured resonances, but it does not
guarantee higher mechanically safe acceleration or good print quality. An
adaptive run can correctly keep the existing/native choice or reject every
candidate. See [reading reports](../reports.md) for the graph and artifact
reference.
