# Calibration reports

Accepted results and validation-rejected diagnostics are written to private
directories beneath `result_folder`. The default is:

```text
~/printer_data/config/AdvancedShaper_results/<attempt-id>/
```

On a typical Klipper Pi, `~` is the home directory of the account running
Klipper. Set
`result_folder` explicitly in `[advanced_input_shaper]` if a different location
is preferred. `ADV_SHAPER_STATUS` reports the artifact paths for the current
attempt.

The results root is created on the first artifact write, not during
installation. A preflight, capture, early-analysis, artifact-write, or rollback
failure can therefore have an attempt ID/status but no attempt directory. This
is expected fail-closed behavior, not evidence that reports were redirected.

The attempt directory contains:

- `report.html`: self-contained, offline technical report with decision status,
  validation gates, candidate comparison, method, and next action.
- `summary.png` and `summary.svg`: compact advanced-analysis overview.
- `input_shaper.png` and `input_shaper.svg`: Klipper-compatible frequency
  profile, using only measured PSD, response, and spectrogram data that are
  actually available.
- `candidates.csv`: aggregate native-candidate and Pareto metrics.
- `validation.csv`: aggregate held-out attenuation and cross-axis gate metrics,
  exact reference/candidate identifiers, and the finite per-pair energy values
  used by the decision, when validation data exists.
- `result.json` and `manifest.json`: exact versioned report and integrity hashes.
- `captures.npz`: private lossless sample arrays when `keep_raw_data: True` and
  capture groups are available.

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
fast validation, repeat and sweep counts, effective Hz/s, estimated physical
motion time per axis, the A/B capture order, and stable pair IDs. Per-axis
validation rows include the paired-capture design and pair count. Candidate CSV
rows identify whether cross-axis ranking came from an upstream native response
curve or the generalized oscillator model. This timing deliberately excludes
host analysis, report rendering, and artifact I/O.
For experimental profiles, the same protocol and runtime-capability records
also preserve native-fitting `max_vibrations` provenance: its finite fraction,
percent, source (`selection_profile.maximum_residual`), and upstream parameter
name. It describes the upstream per-family frequency fit and is distinct from
the separately reported held-out 10% attenuation-improvement threshold.
The held-out chart normalizes each axis to its own reference mean of `1.0`, so a
high-energy axis cannot hide another axis. It shows every paired A/B observation,
the candidate's paired-bootstrap 95% interval, the 10% attenuation threshold,
the measured cross-axis change, and an explicit `PASS` or `REJECT`. This
normalization is display-only: `result.json` and `validation.csv` retain the raw
energy values with `acceleration_squared` units and exact pair IDs. Labels show
the configured reference and selected canonical shaper identifiers, frequencies,
and damping values.
The same block records the effective square-corner velocity and whether it came
from the printer snapshot or an explicit temporary `SCV` parameter. That value
is smoothing-model context, not a validated acceleration limit.
Neither a normalized validation plot nor its confidence interval establishes a
mechanically safe or print-validated acceleration. Theoretical smoothing,
resonance validation, and print validation remain separate evidence levels.
