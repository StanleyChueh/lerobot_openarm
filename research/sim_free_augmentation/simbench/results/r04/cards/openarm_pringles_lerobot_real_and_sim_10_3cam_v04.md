---
license: apache-2.0
task_categories: [robotics]
tags: [LeRobot, openarm, synthetic, sfagen]
---
# OpenArm Pringles hand-over: 10 real + 100 synthetic demos, evenly spread over the table (sfagen v04)

Real + synthetic mixture (`is_synthetic` marks the synthetic episodes; real episodes come first).
110 episodes, 28,378 frames, 30 fps, LeRobot v3.0. Generated without a simulator from the
10 real demos of [`ethanCSL/openarm_pringles_lerobot_real_10_v00`](https://huggingface.co/datasets/ethanCSL/openarm_pringles_lerobot_real_10_v00) (rev cebdee98). Every synthetic episode has exactly its
source demo's duration (1.000x) and speed.

## What is new vs v03 rev 2 (`ethanCSL/openarm_pringles_lerobot_real_and_sim_10_3cam_v03_loose`)
A real-robot evaluation of the v03 model succeeded almost always at the front/middle can positions but sometimes failed at
the rear (close to the robot). Its training set was 6.1x denser at the front than at the
rear. v04 fixes the data side:
1. **Exactly 100 synthetic episodes**, chosen one at a time: the 3 cm cell of the table with the fewest episodes so far
   (real demos count as already occupying their cell), ties by largest distance to everything already chosen, then the
   least-used source demo, then the lowest distortion score.
2. **Pipeline v1** candidates: v0 Cartesian blend, with a joint-space fallback where it fails near folded/singular arm
   poses (rescues rear targets); partial-visibility rule (the can must be visibly in view).

Training episodes (real + synthetic) within 4 cm of the 40 measured evaluation starts, mean per region:

| region | v03 rev 2 loose (138 + 10 real) | **v04 (100 + 10 real)** |
|---|---|---|
| front-right | 22.0 | 16.0 |
| rear-right | 4.5 | 10.2 |
| rear-left | 4.0 | 9.7 |
| front-left | 22.8 | 11.6 |
| middle | 24.2 | 13.2 |

Densest / thinnest region: 6.1 -> **1.7** (pre-registered goal 1.5: near miss;
front-right stays somewhat denser). 18 synthetic episodes have the can in the rear (x < 17 cm).
Density map: `assets/density.png` (bottom row = OpenArm; left v03 rev 2, right v04).

## Contents
- families: object_workspace 92, interaction_location 8 (v03 rev 2 also had 60 hand-over location + 11 timing
  variants; most were not selected because they keep the can at the front real-demo positions)
- synthetic episodes per source demo: {'0': 11, '1': 8, '2': 11, '3': 12, '4': 8, '5': 7, '6': 7, '7': 11, '8': 13, '9': 12}
- per-episode provenance (source demo, variation, shift, label hashes): `meta/sfa_provenance.json`
- wrist cameras: copied from the source demo (can-relative geometry exact; background off by the shift); `wrist_view_valid`
- checks: `sfagen verify` PASS; every candidate passed the kinematic/geometric gates (joint limits, speed/acceleration vs the
  real demos, IK, collision, table, can in view). No physics, no hardware replay, no policy result yet.
- known image artifacts: a faint smudge at the old can position in some frames; ghost/duplicate-can artifacts are possible
  (the distortion score does not catch all of them).

## Load
`LeRobotDataset("ethanCSL/openarm_pringles_lerobot_real_and_sim_10_3cam_v04", revision="v3.0")`

## Code
sfagen, branch `research/sfagen-simbench`, commit `ab657f7a1a` (local; not yet public on GitHub). Selection:
`research/sim_free_augmentation/simbench/tools/balanced_select.py` (`N=100`); protocol `simbench/results/r04/PROTOCOL.md`.
