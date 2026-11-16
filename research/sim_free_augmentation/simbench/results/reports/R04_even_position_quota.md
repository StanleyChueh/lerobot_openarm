# R04 — even spread: exactly 100 synthetic episodes per dataset (fixes position-sensitive training sets)

Date 2026-11-16 · Opus 5.5 · local commits c7d5c01+ · protocol `results/r04/PROTOCOL.md` (pre-registered; Amendment 1 =
user's fixed N = 100) · first-party, pending review.

## Why
The user's real-robot evaluation of `ethanCSL/openarm_pringles_lerobot_real_and_sim_10_3cam_v03_loose` succeeded
~100 % at front-left, front-right and middle but sometimes failed at rear-left and rear-right. The published training
set was 6.1× denser at the front than at the rear (episodes within 4 cm of the eval starts: front ≈ 22–24, rear 4.0–4.5;
0 real demos near the rear). Causes: real demos sit mostly at the front; rear targets fail the gates more often;
handover location/timing episodes keep the can at the (front) real-demo positions.

## Rule (general, `simbench/tools/balanced_select.py N=100`)
Exactly 100 synthetic episodes per dataset, chosen one at a time: the 3 cm cell with the fewest episodes so far (real
demos count as already occupying their cell), ties by largest distance to everything already chosen (farthest point),
then within the cell the least-used source demo, then the lowest distortion score. All gates unchanged; 1.000× length.

## OpenArm v04 (pipeline v1 candidates: 239 passing; local, NOT published)
| region | v03r2 loose (138 syn + 10 real) | **v04 (100 syn + 10 real)** |
|---|---|---|
| front-right | 22.0 | 16.0 |
| front-left | 22.8 | 11.6 |
| middle | 24.2 | 13.2 |
| rear-right | 4.5 | **10.2** |
| rear-left | 4.0 | **9.7** |
| max / min | 6.1 | **1.7** (pre-registered goal ≤ 1.5: near miss) |
Datasets `artifacts/simbench_runs/r04_openarm/all_export/{dataset (100 ep / 25,803 frames), dataset_mixed (110 / 28,378)}`;
verify passes; 0 real episodes in synthetic-only; length 1.000; 18 synthetic episodes with the can at x < 17 cm (rear).
Families: 92 can position + 8 handover location (handover timing variants not selected). Rear frames
(`results/r04/openarm_v04_rear_frames.jpg`): clean; faint smudge at the old can position in some frames.

## RoboTwin (overwrote the 6 synthetic Hub repos; old release kept at tag `r03`)
| task | synthetic (r03 → r04) | density CV over the target lattice (r03 → r04) | p90/p10 |
|---|---|---|---|
| place_empty_cup | 128 → 100 | 0.67 → 0.54 | 9.0 → 6.0 |
| stack_blocks_two | 195 → 100 | 0.52 → 0.40 | 5.5 → 5.0 |
| handover_block | 155 → 100 | 0.95 → 0.68 | 24.0 → 12.0 |
handover keeps 94 block-position + 6 handover variants (r03: 92 + 63). Figures: `results/r04/density.png`,
`results/r04/distribution.png`; checks `results/r04/datasets_check.json`; Hub record `results/r04/hub_overwrite.json`.

## Limits / not checked
Uncovered cells remain far from any source (reported in the density maps, not filled). Even spread of training data
does not by itself prove better rear success: that needs the user's real-robot evaluation of v04 (same batch/steps as
v03r2). RoboTwin RGB quality unchanged (camera fit). Q = 2 variant results kept in `s1f/*/selection.json`.
