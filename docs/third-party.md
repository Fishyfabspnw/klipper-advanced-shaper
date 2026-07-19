# Third-party notices

No third-party source files are vendored in this repository.

The generalized-MZV pulse construction, response scoring, and smoothing parity
code are independent Python implementations of behavior provided by GPLv3
Klipper. Generalized-MZV support was introduced upstream in commit
`e4c4a5b949f48f7bf1a77506ab3c3582f08aa9c8`; numerical parity is pinned against
the later stock snapshot `7046bd00ef5c30dec6febc724f8d22967433c45c`. This
project is also distributed under GPL-3.0-only.

The project interoperates with an installed stock Klipper through
`SET_INPUT_SHAPER`, status/config objects, and an isolated compatibility-checked
private resonance-capture API. It does not copy or replace Klipper's shaper,
motion-planner, kinematics, or MCU source. Klipper and Shake&Tune are research
references, as linked from the README. If source files are adapted in the future, add
its path, upstream project, exact revision, copyright notice, license, and
modification summary here before merging.
