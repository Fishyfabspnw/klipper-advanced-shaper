# Calibration reports

Each attempt is written to a private directory beneath `result_folder`. The
default is:

```text
~/printer_data/config/AdvancedShaper_results/<attempt-id>/
```

On a typical Klipper Pi, `~` is the account running Klipper. Set
`result_folder` explicitly in `[advanced_input_shaper]` if a different location
is preferred. `ADV_SHAPER_STATUS` reports the artifact paths for the current
attempt.

The attempt directory contains:

- `report.html`: self-contained, offline technical report with decision status,
  validation gates, candidate comparison, method, and next action.
- `summary.png` and `summary.svg`: compact advanced-analysis overview.
- `input_shaper.png` and `input_shaper.svg`: Klipper-compatible frequency
  profile, using only measured PSD, response, and spectrogram data that are
  actually available.
- `candidates.csv`: aggregate native-candidate and Pareto metrics.
- `validation.csv`: aggregate held-out attenuation and cross-axis gate metrics,
  when validation data exists.
- `result.json` and `manifest.json`: exact versioned report and integrity hashes.
- `captures.npz`: private lossless sample arrays when `keep_raw_data: True`.

Raw accelerometer samples are never exported to CSV by default. Rejected
attempts are clearly labeled and are retained only for diagnosis; they cannot
be applied or staged.

Parameterized results retain the canonical shaper identifier, installed-runtime
capability proof, measured design damping and uncertainty samples, and held-out
QC/confidence/cross-axis evidence. Acceleration fields distinguish theoretical
smoothing, resonance-validated, and print-validated values; absent physical
evidence is recorded as unavailable rather than inferred from the model.
Every completed analysis also records the resolved excitation source and value,
maximum sweep frequency, sweeping acceleration, current printer acceleration
limit, 80% budget, and estimated peak used by the pre-motion safety check.
The validation protocol block labels full-confidence versus lower-confidence
fast validation, repeat and sweep counts, effective Hz/s, and estimated physical
motion time per axis. This timing deliberately excludes host analysis, report
rendering, and artifact I/O.
