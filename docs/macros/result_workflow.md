# Result review, apply, and stage workflow

Calibration restores the printer state before a result becomes reviewable. A
rejected or rollback-failed attempt cannot be applied or staged.

## Status

```text
ADV_SHAPER_STATUS
```

This reports the current state, attempt ID, accepted result ID, artifact paths,
validation protocol, cancellation state, and any error. Output files are
created only when an accepted report or validation-rejected diagnostic is
successfully written.

## Cancel

```text
ADV_SHAPER_CANCEL
```

Cancellation is cooperative. The controller still completes its rollback path
before the attempt exits.

## Apply for the current runtime

```text
ADV_SHAPER_APPLY RESULT=<result-id>
```

This sends the accepted canonical shaper type, frequency, and damping through
stock Klipper's `SET_INPUT_SHAPER` interface. It does not write `printer.cfg`,
run `SAVE_CONFIG`, or change `[printer] max_accel`.

## Stage for later saving

```text
ADV_SHAPER_STAGE RESULT=<result-id>
SAVE_CONFIG
```

`STAGE` writes accepted stock-Klipper input-shaper values to Klipper's pending
config state. Only the separate operator-issued `SAVE_CONFIG` persists them.
