---
license: apache-2.0
task_categories:
- robotics
tags:
- LeRobot
- openarm
- data-augmentation
- synthetic
- bimanual
- handover
configs:
- config_name: default
  data_files: data/*/*.parquet
---

# openarm_pringles_lerobot_real_to_sim_10_3cam_v00

OpenArm bimanual hand-over ("Pick up the Pringles can with the right arm, hand it to the left arm"), three cameras (`body_cam`, `wrist_cam` = left wrist, `right_wrist_cam`), LeRobot v3.0, 30 fps.

**Every episode is synthetic.**
- **Count:** 96 episodes (24,645 frames).
- **Source:** generated without a simulator from the 10 real demonstrations of [`ethanCSL/openarm_pringles_lerobot_real_10_v00`](https://huggingface.co/datasets/ethanCSL/openarm_pringles_lerobot_real_10_v00) (revision `cebdee98`), all 10 used.
- **Variation families:** 29 object-position + 67 interaction-location (hand-over point).

The same episodes plus the 10 real demos: [`ethanCSL/openarm_pringles_lerobot_real_and_sim_10_3cam_v00`](https://huggingface.co/datasets/ethanCSL/openarm_pringles_lerobot_real_and_sim_10_3cam_v00).

## Revision history

| tag | content |
|---|---|
| `v04-65ep-two-pipelines` (`7179061`) | previous revision (task-specific pipeline) |
| current (`v3.0`) | **96 episodes from the single sfagen pipeline shared by both OpenArm tasks** |

## How it is made: one pipeline for every task

Generated with [`sfagen`](https://huggingface.co/datasets/ethanCSL/openarm_pringles_lerobot_real_to_sim_10_3cam_v00/blob/main/sfagen/README.md), the same code and settings that
generate the battery-insertion dataset. Only the task description differs: source
dataset, objects (the Pringles can (a 7.5 × 23 cm cylinder)) and the shift grid. Full method: `PIPELINE.md`; positioning against prior work: `NOVELTY.md`.

1. **Rig calibration (automatic).**
   - **Body camera:** render-and-compare against the robot's own silhouette; held-out IoU 0.307 → 0.361.
   - **Table height:** 0.200 m (median of known-size object silhouette fits (median IoU >= 0.5)).
2. **Reconstruction (per source demo, no labels).**
   - **Robot masks:** SAM2.1, prompted from the projected URDF.
   - **Objects:** segmented as what disappears when picked up. Known-size cylinders are posed by grasp-constrained render-and-compare.
   - **Time offset:** each demo's video/state offset is chosen photometrically.
3. **Two variation families** (kinematic edits of the recorded motion):
   - **Object position** (29 episodes): an object is moved ±3/±6 cm; the arm that grasps it follows (approach and grasp unchanged relative to the object) and rejoins the recording at the lift apex.
   - **Interaction location** (67 episodes, the hand-over point): the whole two-arm interaction is moved ±3/±5 cm; both hands and everything they hold move together, so every recorded contact relation is kept.
4. **Replay checks.** Every published episode passes all of them, on the commanded actions:
   - joint limits;
   - speed ≤ 1.25× and acceleration ≤ 1.5× the real demos;
   - arm–arm / arm–body mesh clearance and hand distance not below the source;
   - table clearance;
   - start pose;
   - untouched parts bit-identical;
   - IK exact at contact (≤ 1 mm / 0.5°);
   - object placement (in view, on the table, robot never closer to it than in the recording).
5. **Body camera: mesh-guided forward warping of real pixels** (no generative model). Every robot/object pixel moves in 3-D with its part, hidden surfaces are borrowed from other frames, and the background is relit. Frames where the scene equals the recording are the recorded frames.
6. **Wrist cameras.** Real recorded frames; `wrist_view_valid` and `wrist_view_pose_offset` say where they are exact.

## What the synthetic data adds (positions by forward kinematics at gripper events; the other real recordings were never used for generation)

| position (vs the other real recordings) | 10 sources | + previous revision | **+ this revision** |
|---|---:|---:|---:|
| can: within 2 cm (of 40) | 9 | 29 | **24** |
| can: area covered (cm²; other real recordings span 366) | 96 | 212 | **201** |
| handover: within 2 cm (of 40) | 33 | 38 | **40** |
| handover: area covered (cm²; other real recordings span 19) | 13 | 35 | **124** |

**Lower than the previous revision:** can coverage. This pipeline rejects object positions where the arm, on its way to the object, would come closer to the object than in the recording (robot-proximity rule); the previous revision had no such rule. The previous revision stays loadable by its tag.

## Image fidelity, measured against real ground truth

**Method.** Re-render real frame j+k from real frame j only (k = 4/8/15), compared with the real frame inside a region that
depends on no pipeline. Held-out episodes [0, 2, 3, 4, 6, 7, 9] (211 pairs), never used for any tuning decision; 95% CI by episode bootstrap.

| renderer | PSNR (dB) | SSIM |
|---|---:|---:|
| copy the source frame (no motion) | 14.51 | 0.549 |
| previous revision's renderer | 17.87 | 0.692 |
| **this revision (sfagen)** | **18.04** | **0.703** |

Difference from the previous renderer: +0.17 dB [-0.06, 0.40], better in 55% of pairs.

## Real vs synthetic, side by side

[highlights](https://huggingface.co/datasets/ethanCSL/openarm_pringles_lerobot_real_to_sim_10_3cam_v00/blob/main/comparison_videos/pringles_unified_highlights.mp4) ·
[all episodes, 2×](https://huggingface.co/datasets/ethanCSL/openarm_pringles_lerobot_real_to_sim_10_3cam_v00/blob/main/comparison_videos/pringles_unified_all_2x.mp4)
(left: real source; right: synthetic; the status bar says whether the frame is edited).

## Verification

- **Loading:** the dataset loads in LeRobot.
- **Labels:** state/action SHA-256 equal the plans.
- **Synthetic only:** 0 episodes equal a real episode.
- **Replay checks:** all pass.
- **Video:** every camera decodes with no black frames; the body camera matches the renders (39.38 dB after encoding).

## Limitations

- **Synthetic state.** `observation.state` is a planned kinematic candidate. Force, friction and contact are unchecked, and no episode has been replayed on hardware yet. Start with a ±3 cm episode, slowly.
- **Visible artifacts.** Arms re-posed close to the camera, or entering from the image edge, can show fragments. Watch the comparison videos.
- **Narrow variation.** Translations only (≤ 6 cm); no object rotation; the cable of the station (battery task) moves rigidly near the station.
- **Wrist cameras.** Not re-rendered.
- **No training evidence.** No policy has been trained on this revision yet.

## Episodes

| ep | source | variation | family | frames | shift (cm) | edited frames | max speed / real | max accel / real | min arm distance (mm) |
|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| 0 | 0 | `can_x+0_y-3` | object position | 273 | (+0, -3) | 127 | 0.97 | 0.90 | 117 |
| 1 | 0 | `interact_x+0_y+3_z+0` | interaction location | 273 | (+0, +3) | 147 | 1.03 | 0.90 | 27 |
| 2 | 0 | `interact_x+0_y+5_z+0` | interaction location | 273 | (+0, +5) | 147 | 1.07 | 0.90 | 27 |
| 3 | 0 | `interact_x+0_y-3_z+0` | interaction location | 273 | (+0, -3) | 147 | 1.00 | 0.90 | 27 |
| 4 | 0 | `interact_x+0_y-5_z+0` | interaction location | 273 | (+0, -5) | 147 | 1.03 | 0.90 | 28 |
| 5 | 0 | `interact_x+3_y+0_z+0` | interaction location | 273 | (+3, +0) | 147 | 1.06 | 0.90 | 28 |
| 6 | 0 | `interact_x+5_y+0_z+0` | interaction location | 273 | (+5, +0) | 147 | 1.12 | 0.90 | 28 |
| 7 | 0 | `interact_x-3_y+0_z+0` | interaction location | 273 | (-3, +0) | 147 | 0.88 | 0.90 | 27 |
| 8 | 0 | `interact_x-5_y+0_z+0` | interaction location | 273 | (-5, +0) | 147 | 0.86 | 0.91 | 27 |
| 9 | 1 | `can_x+0_y+3` | object position | 263 | (+0, +3) | 117 | 0.80 | 1.00 | 100 |
| 10 | 1 | `can_x+3_y+0` | object position | 263 | (+3, +0) | 117 | 0.80 | 1.00 | 102 |
| 11 | 1 | `can_x+6_y+0` | object position | 263 | (+6, +0) | 117 | 0.83 | 1.02 | 102 |
| 12 | 1 | `interact_x+0_y+3_z+0` | interaction location | 263 | (+0, +3) | 164 | 0.79 | 1.01 | 26 |
| 13 | 1 | `interact_x+0_y+5_z+0` | interaction location | 263 | (+0, +5) | 164 | 0.79 | 1.02 | 26 |
| 14 | 1 | `interact_x+0_y-3_z+0` | interaction location | 263 | (+0, -3) | 164 | 0.83 | 0.98 | 26 |
| 15 | 1 | `interact_x+0_y-5_z+0` | interaction location | 263 | (+0, -5) | 164 | 0.88 | 0.97 | 27 |
| 16 | 1 | `interact_x+3_y+0_z+0` | interaction location | 263 | (+3, +0) | 164 | 0.93 | 0.99 | 26 |
| 17 | 1 | `interact_x+5_y+0_z+0` | interaction location | 263 | (+5, +0) | 164 | 1.02 | 0.98 | 27 |
| 18 | 1 | `interact_x-3_y+0_z+0` | interaction location | 263 | (-3, +0) | 164 | 0.78 | 1.01 | 26 |
| 19 | 1 | `interact_x-5_y+0_z+0` | interaction location | 263 | (-5, +0) | 164 | 0.87 | 1.02 | 26 |
| 20 | 2 | `can_x+0_y+3` | object position | 280 | (+0, +3) | 137 | 1.00 | 0.82 | 89 |
| 21 | 2 | `can_x-3_y+0` | object position | 280 | (-3, +0) | 137 | 1.00 | 0.82 | 102 |
| 22 | 2 | `interact_x+0_y+3_z+0` | interaction location | 280 | (+0, +3) | 157 | 1.00 | 0.86 | 30 |
| 23 | 2 | `interact_x+0_y+5_z+0` | interaction location | 280 | (+0, +5) | 157 | 1.01 | 0.89 | 30 |
| 24 | 2 | `interact_x+3_y+0_z+0` | interaction location | 280 | (+3, +0) | 157 | 1.02 | 1.29 | 30 |
| 25 | 2 | `interact_x-3_y+0_z+0` | interaction location | 280 | (-3, +0) | 157 | 0.98 | 0.89 | 30 |
| 26 | 3 | `can_x+0_y+3` | object position | 250 | (+0, +3) | 124 | 1.00 | 1.00 | 87 |
| 27 | 3 | `can_x+0_y+6` | object position | 250 | (+0, +6) | 124 | 1.00 | 1.00 | 79 |
| 28 | 3 | `can_x+3_y+0` | object position | 250 | (+3, +0) | 124 | 1.00 | 1.00 | 87 |
| 29 | 3 | `can_x-3_y+0` | object position | 250 | (-3, +0) | 124 | 1.00 | 1.00 | 87 |
| 30 | 3 | `can_x-6_y+0` | object position | 250 | (-6, +0) | 124 | 1.00 | 1.02 | 88 |
| 31 | 3 | `interact_x+0_y+3_z+0` | interaction location | 250 | (+0, +3) | 148 | 1.03 | 1.05 | 34 |
| 32 | 3 | `interact_x+0_y+5_z+0` | interaction location | 250 | (+0, +5) | 148 | 1.06 | 1.08 | 34 |
| 33 | 3 | `interact_x+0_y-3_z+0` | interaction location | 250 | (+0, -3) | 148 | 1.02 | 1.00 | 34 |
| 34 | 3 | `interact_x+0_y-5_z+0` | interaction location | 250 | (+0, -5) | 148 | 1.07 | 1.00 | 34 |
| 35 | 3 | `interact_x+3_y+0_z+0` | interaction location | 250 | (+3, +0) | 148 | 1.08 | 1.01 | 34 |
| 36 | 3 | `interact_x+5_y+0_z+0` | interaction location | 250 | (+5, +0) | 148 | 1.15 | 1.10 | 34 |
| 37 | 3 | `interact_x-3_y+0_z+0` | interaction location | 250 | (-3, +0) | 148 | 0.96 | 1.00 | 34 |
| 38 | 3 | `interact_x-5_y+0_z+0` | interaction location | 250 | (-5, +0) | 148 | 0.97 | 1.00 | 34 |
| 39 | 4 | `can_x+3_y+0` | object position | 250 | (+3, +0) | 117 | 0.97 | 0.87 | 73 |
| 40 | 4 | `can_x-3_y+0` | object position | 250 | (-3, +0) | 117 | 0.94 | 0.87 | 75 |
| 41 | 4 | `interact_x+0_y+3_z+0` | interaction location | 250 | (+0, +3) | 149 | 0.86 | 0.87 | 40 |
| 42 | 4 | `interact_x+0_y+5_z+0` | interaction location | 250 | (+0, +5) | 149 | 0.83 | 0.87 | 41 |
| 43 | 4 | `interact_x+0_y-3_z+0` | interaction location | 250 | (+0, -3) | 149 | 1.00 | 0.87 | 40 |
| 44 | 4 | `interact_x+0_y-5_z+0` | interaction location | 250 | (+0, -5) | 149 | 1.03 | 0.87 | 40 |
| 45 | 4 | `interact_x+3_y+0_z+0` | interaction location | 250 | (+3, +0) | 149 | 0.92 | 0.87 | 40 |
| 46 | 4 | `interact_x+5_y+0_z+0` | interaction location | 250 | (+5, +0) | 149 | 1.04 | 0.87 | 40 |
| 47 | 4 | `interact_x-3_y+0_z+0` | interaction location | 250 | (-3, +0) | 149 | 0.96 | 0.87 | 40 |
| 48 | 4 | `interact_x-5_y+0_z+0` | interaction location | 250 | (-5, +0) | 149 | 0.99 | 0.89 | 40 |
| 49 | 5 | `can_x+0_y-3` | object position | 255 | (+0, -3) | 116 | 1.00 | 1.01 | 84 |
| 50 | 5 | `can_x+0_y-6` | object position | 255 | (+0, -6) | 116 | 1.00 | 1.02 | 85 |
| 51 | 5 | `can_x-3_y+0` | object position | 255 | (-3, +0) | 116 | 1.00 | 1.37 | 82 |
| 52 | 5 | `interact_x+0_y+3_z+0` | interaction location | 255 | (+0, +3) | 157 | 1.00 | 1.03 | 31 |
| 53 | 5 | `interact_x+0_y+5_z+0` | interaction location | 255 | (+0, +5) | 157 | 1.01 | 1.07 | 31 |
| 54 | 5 | `interact_x+0_y-3_z+0` | interaction location | 255 | (+0, -3) | 157 | 1.00 | 1.00 | 32 |
| 55 | 5 | `interact_x+0_y-5_z+0` | interaction location | 255 | (+0, -5) | 157 | 1.04 | 1.01 | 32 |
| 56 | 5 | `interact_x+3_y+0_z+0` | interaction location | 255 | (+3, +0) | 157 | 1.04 | 1.07 | 31 |
| 57 | 5 | `interact_x+5_y+0_z+0` | interaction location | 255 | (+5, +0) | 157 | 1.11 | 1.22 | 31 |
| 58 | 5 | `interact_x-3_y+0_z+0` | interaction location | 255 | (-3, +0) | 157 | 1.05 | 1.00 | 32 |
| 59 | 5 | `interact_x-5_y+0_z+0` | interaction location | 255 | (-5, +0) | 157 | 1.09 | 1.01 | 32 |
| 60 | 6 | `can_x+3_y+0` | object position | 241 | (+3, +0) | 150 | 1.00 | 0.99 | 74 |
| 61 | 6 | `interact_x+0_y+3_z+0` | interaction location | 241 | (+0, +3) | 147 | 1.05 | 1.04 | 43 |
| 62 | 6 | `interact_x+0_y+5_z+0` | interaction location | 241 | (+0, +5) | 147 | 1.08 | 1.07 | 43 |
| 63 | 6 | `interact_x+0_y-3_z+0` | interaction location | 241 | (+0, -3) | 147 | 0.95 | 0.97 | 43 |
| 64 | 6 | `interact_x+0_y-5_z+0` | interaction location | 241 | (+0, -5) | 147 | 0.93 | 0.97 | 43 |
| 65 | 6 | `interact_x+3_y+0_z+0` | interaction location | 241 | (+3, +0) | 147 | 1.07 | 1.01 | 43 |
| 66 | 6 | `interact_x+5_y+0_z+0` | interaction location | 241 | (+5, +0) | 147 | 1.13 | 1.02 | 43 |
| 67 | 6 | `interact_x-3_y+0_z+0` | interaction location | 241 | (-3, +0) | 147 | 0.98 | 0.97 | 43 |
| 68 | 6 | `interact_x-5_y+0_z+0` | interaction location | 241 | (-5, +0) | 147 | 1.08 | 0.97 | 43 |
| 69 | 7 | `can_x+0_y+3` | object position | 261 | (+0, +3) | 159 | 1.00 | 1.00 | 83 |
| 70 | 7 | `can_x+0_y-3` | object position | 261 | (+0, -3) | 159 | 1.00 | 1.00 | 83 |
| 71 | 7 | `can_x+3_y+0` | object position | 261 | (+3, +0) | 159 | 1.00 | 1.00 | 83 |
| 72 | 7 | `can_x-3_y+0` | object position | 261 | (-3, +0) | 159 | 1.00 | 1.00 | 83 |
| 73 | 7 | `can_x-6_y+0` | object position | 261 | (-6, +0) | 159 | 1.00 | 1.19 | 83 |
| 74 | 7 | `interact_x+0_y+3_z+0` | interaction location | 261 | (+0, +3) | 162 | 1.05 | 1.00 | 49 |
| 75 | 7 | `interact_x+0_y+5_z+0` | interaction location | 261 | (+0, +5) | 162 | 1.09 | 1.00 | 49 |
| 76 | 7 | `interact_x+0_y-3_z+0` | interaction location | 261 | (+0, -3) | 162 | 0.98 | 1.00 | 49 |
| 77 | 7 | `interact_x+3_y+0_z+0` | interaction location | 261 | (+3, +0) | 162 | 0.98 | 1.00 | 49 |
| 78 | 7 | `interact_x-3_y+0_z+0` | interaction location | 261 | (-3, +0) | 162 | 1.01 | 1.06 | 49 |
| 79 | 7 | `interact_x-5_y+0_z+0` | interaction location | 261 | (-5, +0) | 162 | 1.02 | 1.41 | 49 |
| 80 | 8 | `can_x+0_y+3` | object position | 249 | (+0, +3) | 143 | 1.00 | 1.00 | 87 |
| 81 | 8 | `can_x+0_y-3` | object position | 249 | (+0, -3) | 143 | 1.00 | 1.02 | 99 |
| 82 | 8 | `can_x+3_y+0` | object position | 249 | (+3, +0) | 143 | 1.00 | 1.21 | 98 |
| 83 | 8 | `interact_x+0_y+3_z+0` | interaction location | 249 | (+0, +3) | 154 | 0.91 | 1.00 | 40 |
| 84 | 8 | `interact_x+0_y+5_z+0` | interaction location | 249 | (+0, +5) | 154 | 0.91 | 1.00 | 40 |
| 85 | 8 | `interact_x+3_y+0_z+0` | interaction location | 249 | (+3, +0) | 154 | 0.98 | 1.38 | 40 |
| 86 | 8 | `interact_x-3_y+0_z+0` | interaction location | 249 | (-3, +0) | 154 | 1.11 | 1.09 | 40 |
| 87 | 9 | `can_x+0_y-3` | object position | 253 | (+0, -3) | 120 | 1.00 | 1.00 | 113 |
| 88 | 9 | `can_x+0_y-6` | object position | 253 | (+0, -6) | 120 | 1.00 | 1.00 | 129 |
| 89 | 9 | `can_x-3_y+0` | object position | 253 | (-3, +0) | 120 | 1.00 | 1.00 | 90 |
| 90 | 9 | `can_x-6_y+0` | object position | 253 | (-6, +0) | 120 | 1.00 | 1.33 | 91 |
| 91 | 9 | `interact_x+0_y+3_z+0` | interaction location | 253 | (+0, +3) | 156 | 1.07 | 1.01 | 44 |
| 92 | 9 | `interact_x+0_y+5_z+0` | interaction location | 253 | (+0, +5) | 156 | 1.11 | 1.01 | 44 |
| 93 | 9 | `interact_x+0_y-3_z+0` | interaction location | 253 | (+0, -3) | 156 | 1.21 | 1.29 | 44 |
| 94 | 9 | `interact_x+3_y+0_z+0` | interaction location | 253 | (+3, +0) | 156 | 0.98 | 1.01 | 44 |
| 95 | 9 | `interact_x+5_y+0_z+0` | interaction location | 253 | (+5, +0) | 156 | 0.97 | 1.24 | 44 |
