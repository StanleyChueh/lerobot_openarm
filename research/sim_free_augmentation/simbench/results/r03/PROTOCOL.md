# R03 protocol — target area from source demos + partial visibility (pre-registered 2026-11-16, before any run)
User direction: the pipeline must be general/cross-task; the object randomization should come from the source demos
(+ URDF), and targets must not be rejected just because the object is not fully visible, as long as it is visibly in
the camera view. Method change #2 (of at most 4). Pipeline v1 recipe = R02 `joint_fallback` + this.

## Changes (robot- and task-agnostic; defaults keep v0 behaviour)
1. `object_targets.positions_xy_m: auto`: 3 cm lattice points (robot-frame aligned) within `max_shift_m` (recipe
   0.12 m) of any frozen source object position; the existing min/max shift + nearest-sources assignment is unchanged.
   Replaces the per-task arena copied from RoboTwin's env files (manual input removed). RoboTwin targets: cup 222,
   stack 176, handover 135 (hand-copied ranges: 108, 95, 63). OpenArm keeps its explicit grid.
2. `checks.visibility: partial`: in_view = visible fraction of the moved object's points (10 px margin) >=
   min(`min_visible_fraction` = 0.5, the same object's visible fraction at its source position). Default `full` = v0.

## Hypotheses (fresh dirs `s1e`, `r03_openarm_v1`)
- H1: each of the 3 RoboTwin tasks yields >= 10 object-position synthetic episodes that pass every gate (handover had 0).
- H2: selected object-position targets are visibly in view: every selected one has visible fraction >= its threshold
  (by construction) — reported with the distribution of visible fractions and decoded-frame inspection.
- H3 (OpenArm not worse): with the v1 recipe on the loose v03r2 config, every candidate passing in the R02 `cartesian`
  run is bit-identical; grid passes >= 154, targets >= 46, interaction >= 62, timing >= 11.
- Reported: per-task episodes, distinct positions, per-source counts, shift distances, top-down distribution map.
Negative results are kept.
