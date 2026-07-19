# Live evidence — 2026-07-19

This note records the evidence that motivated the stock-baseline-first
selection gate. It is not a claim that the printer, plugin, or any shaper has a
physical acceleration rating.

## Capture scope

- X attempt: `d2eef1fa191c`
- Y attempt: `13e5805bcbd6`
- Commanded recipe: `ACCEL_PER_HZ=175`, `HZ_PER_SEC=2`, `SCV=15`,
  `FAST_VALIDATION=1`, `PEAK_LOCK=1`
- Each attempt used one unshaped training sweep. Both stopped at the
  theoretical screen, before candidate `SET_INPUT_SHAPER` or held-out motion.
- The configured references were X `mzv` at 75.6 Hz / 0.038 damping and Y
  `mzv` at 50 Hz / 0.056 damping.

Because this was the exploratory fast protocol, these captures do not establish
repeatability. A production qualification still requires at least three
training repeats and mandatory paired held-out validation.

## Generalized-MZV replay

The stored spectra were replayed over 265 valid fixed-peak generalized-MZV
designs per axis (`n=3..10`, spacing step 0.025 in the production search
domain). Every design was compared with the exact configured reference in each
meaningful along- and cross-axis 5 Hz band over measured damping uncertainty.

| Axis | Exact configured reference | Best eligible stock | Originally selected generalized MZV | Exact screen result | Best screen-safe generalized MZV |
| --- | ---: | ---: | ---: | ---: | ---: |
| X | 16,678 mm/s² theoretical | ZV, 19,620 mm/s² | `mzv(n=10,t=0.650000)`, 16,692 mm/s² | 47.49× worst-band regression | `mzv(n=7,t=1.075000)`, 11,973 mm/s² |
| Y | 5,965 mm/s² theoretical | MZV, 6,963 mm/s² | `mzv(n=10,t=0.650000)`, 5,903 mm/s² | 28.65× worst-band regression | `mzv(n=8,t=1.250000)`, 3,663 mm/s² |

Zero generalized-MZV candidates on either axis both passed the 1.10
meaningful-band non-regression limit and increased theoretical smoothing
acceleration over the configured reference. Consequently, zero beat both the
configured reference and best eligible stock family.

## Parameterized-EI research replay

A separate research-only grid used upstream EI pulse equations with
`v_tol=0.02..0.10` in 0.005 increments and 25..150 Hz in 0.5 Hz increments.
Before exact band comparison, 295 X candidates and 29 Y candidates appeared to
beat the best eligible stock theoretical acceleration while meeting the common
0.10 residual limit. After the exact configured-reference 5 Hz screen, zero
remained on either axis.

Therefore `ei(v_tol=...)` remains research-only. These results do not prove a
future parameterized EI search can never work; they show that exposing the
current search at runtime would not be evidence-supported for these captures.

## Three-repeat production-gate run

After deploying the stock-baseline-first gate, the supervised printer was
tested again with `REPEATS=3`, `FAST_VALIDATION=0`, `ACCEL_PER_HZ=175`,
`HZ_PER_SEC=2`, `SCV=15`, and `PEAK_LOCK=1`. X attempt `aa4ab52d23cc` and Y
attempt `f1ac9236285c` both completed the three stock-baseline sweeps and
abstained before candidate `SET_INPUT_SHAPER` or held-out motion.

| Axis | Strongest generalized candidate | Exact configured reference | Best eligible stock | Result |
| --- | --- | ---: | ---: | --- |
| X | `mzv(n=10,t=0.550000)` at 72.455 Hz / 0.08696 damping, 17,471 mm/s^2 theoretical | MZV, 16,678 mm/s^2 | ZV, 19,491 mm/s^2 | +4.75% vs configured, -10.37% vs stronger stock comparator; no upgrade |
| Y | `mzv(n=3,t=0.550000)` at 48.828 Hz / 0.13942 damping, 6,153 mm/s^2 theoretical | MZV, 5,965 mm/s^2 | 3HUMP_EI, 4,425 mm/s^2 | +3.15% vs stronger configured comparator; no upgrade |

The installed Klipper capability proof passed for the 10-pulse single-pass
executor, parameterized initialization, the frequency-assignment fix, and the
`max_vibrations=0.10` native fitting call. Both attempts restored the exact
native X/Y identifiers, frequencies, damping ratios, velocity limits, and
square-corner velocity. The printer configuration hash remained unchanged.

## Decision

The plugin must not select a modified family merely because it is the best
member of a modified-only pool. It now requires a meaningful theoretical uplift
over both the exact active baseline and the best eligible stock candidate,
then screens exact installed-source pulse models before any candidate motion.
When no design passes, the correct result is **no upgrade**, with the normal
Klipper or Shake&Tune result left active.
