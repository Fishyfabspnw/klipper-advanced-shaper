# Advanced Shaper documentation

Advanced Shaper measures stock-Klipper input-shaper choices and accepts a
candidate only when the requested validation protocol supports it. A clean
mechanical system and repeatable measurements matter more than chasing the
largest theoretical acceleration number. The plugin cannot correct loose
belts, frame motion, toolhead play, sensor problems, or extrusion limits.

## Commands and workflows

| Command or guide | Purpose |
| --- | --- |
| [`ADV_SHAPER_UI_CALIBRATE`](macros/advanced_shaper_calibrate.md) | Run the supervised calibration workflow with documented numeric parameters. |
| [`ADV_SHAPER_STATUS`](macros/result_workflow.md) | Show the active attempt, accepted result, artifact paths, and rollback errors. |
| [`ADV_SHAPER_CANCEL`](macros/result_workflow.md) | Request cancellation while preserving the mandatory rollback path. |
| [`ADV_SHAPER_APPLY`](macros/result_workflow.md) | Apply an accepted stock-Klipper result for the current runtime only. |
| [`ADV_SHAPER_STAGE`](macros/result_workflow.md) | Stage accepted shaper parameters; `SAVE_CONFIG` remains a separate operator action. |
| [Installation and maintenance](installation.md) | Install, configure, update, verify, and uninstall the plugin. |
| [Reading reports](reports.md) | Find output files and interpret candidates, validation, and acceleration evidence. |
| [Generalized MZV](experimental-generalized-mzv.md) | Understand the opt-in parameterized-MZV search and stock-Klipper capability gates. |
| [Benchmark protocol](benchmarking.md) | Compare results without overstating model-derived acceleration. |

The plugin does not modify Klipper's motion planner, shaper definitions,
input-shaper module, calibration module, C kinematics helper, or MCU firmware.
