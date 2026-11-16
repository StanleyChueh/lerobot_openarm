# R03 — targets from the source demos + partial visibility; selection fix; RoboTwin datasets packaged

Date 2026-11-16 · Opus 5.5 · branch `research/sfagen-simbench` (local) · protocol `results/r03/PROTOCOL.md`,
`FREEZE.json` (frozen before the runs) · first-party, pending review.

## What changed (general, from source demos + URDF only)
1. **`object_targets.positions_xy_m: auto`.** The target area is a 3 cm lattice within the recipe's 12 cm limit of
   any source-demo object position (`target_plan.py:auto_targets`). This replaces the arena copied by hand from
   RoboTwin's env files. All arm groups of a task now share one target list.
2. **`checks.visibility: partial`.** The moved object must be visibly in view, not necessarily whole: visible
   fraction of its points ≥ min(0.5, the same object's visible fraction at its source position) (`checks.py`
   placement).
3. **Selection fix.** The ≤ 2-per-target selection now groups by target position, not by key name
   (`select_export.py`). On the old cup data it selects 61 episodes on 44 positions; the buggy version gave 60 on 41.

## OpenArm not worse (H3, holds)
Loose v03r2 config with the full v1 recipe (`artifacts/simbench_runs/r03_openarm_v1`):

| | v0 | R02 v1 | **R03 v1** |
|---|---|---|---|
| grid passes | 92/416 | 154/416 | **166/416** |
| grid targets | 39 | 46 | **52** |
| interaction location / timing | 60 · 11 | 62 · 11 | 62 · 11 |
| v0-passing candidates bit-identical | — | 163/163 | **163/163** |

OpenArm keeps its explicit grid. Only the visibility rule applies to it, and that rule can only add candidates.

## RoboTwin (H1, holds for all 3 tasks)
Run `artifacts/simbench_runs/s1e`:

| task | object position passing | location / timing | synthetic episodes | distinct positions | sources used |
|---|---|---|---|---|---|
| place_empty_cup | 145/499 (29 %) | n/a | **128** (was 60) | see `distribution.json` | 10/10 |
| stack_blocks_two | 315/500 (63 %) | n/a | **195** (was 99) | | 10/10 |
| handover_block | **181/500 (36 %)** (was 0/363) | 43/80 · 20/20 | **155** (92 position + 63) | | 10/10 |

**Datasets (local, LeRobot v3.0, 30 fps):** `artifacts/simbench_data/robotwin_sfagen_v1r3/<task>/{source,synthetic,mixed}`.

| task | source | synthetic | mixed |
|---|---|---|---|
| place_empty_cup | 10 ep / 1,741 frames | 128 / 22,191 | 138 / 23,932 |
| stack_blocks_two | 10 / 3,140 | 195 / 61,231 | 205 / 64,371 |
| handover_block | 10 / 2,835 | 155 / 43,567 | 165 / 46,402 |

Independent check (`results/r03/datasets_check.json`):
- every synthetic episode has length/source = 1.000;
- the synthetic sets contain 0 real episodes;
- the mixed sets contain exactly the 10 sources, flagged with `is_synthetic`;
- the mixed video-span check passes;
- `sfagen verify` passes for all 9 group exports.

Distribution figure: `results/r03/distribution.png` (+ `.json`). Videos: `artifacts/simbench_runs/s1e/*/*_export/compare_*.mp4`.

## RGB quality (inspected by eye; `results/r03/compare_frames.jpg`)
- **stack_blocks_two: good.** The block is moved and the arm is coherent; small fragments at the image edge.
- **place_empty_cup: mediocre.** The cup is moved correctly, but the arm fragments while carrying the cup, and
  edge-of-view cups show ghosting.
- **handover_block, block-position episodes: poor.** The 20 cm block re-renders distorted (box proxy) and the arm
  fragments. The location/timing episodes are better.
- **Known causes:** camera fit 16–35 px from truth (R01 P1), arm re-posing at the image edge (P5), and the box proxy
  used for a tall object.
- **The distortion score misses most of these.**

## Not checked
- Policy training.
- Simulator replay of the synthetic actions.
- Wrist-camera correctness (copied wrists).
