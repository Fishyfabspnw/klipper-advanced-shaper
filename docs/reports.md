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
