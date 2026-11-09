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
- insertion
configs:
- config_name: default
  data_files: data/*/*.parquet
---

# openarm_insertbattery_lerobot_real_to_sim_10_3cam_v00

OpenArm bimanual battery insertion ("Pick up the charging station with the left arm, and insert battery in it with the right arm"), three cameras (`body_cam`, `wrist_cam` = left wrist, `right_wrist_cam`), LeRobot v3.0, 30 fps.

**Every episode is synthetic.**
- **Count:** 114 episodes (84,690 frames).
- **Source:** generated without a simulator from the 10 real demonstrations of [`ethanCSL/openarm_insertbattery_lerobot_real_10_v02`](https://huggingface.co/datasets/ethanCSL/openarm_insertbattery_lerobot_real_10_v02) (revision `a1c29d77`), all 10 used.
- **Variation families:** 62 object-position + 52 interaction-location (insertion point).

The same episodes plus the 10 real demos: [`ethanCSL/openarm_insertbattery_lerobot_real_and_sim_10_3cam_v00`](https://huggingface.co/datasets/ethanCSL/openarm_insertbattery_lerobot_real_and_sim_10_3cam_v00).

## Revision history

| tag | content |
|---|---|
| `v01-65ep-v01` (`3a977b0`) | previous revision (task-specific pipeline) |
| current (`v3.0`) | **114 episodes from the single sfagen pipeline shared by both OpenArm tasks** |

## How it is made: one pipeline for every task

Generated with [`sfagen`](https://huggingface.co/datasets/ethanCSL/openarm_insertbattery_lerobot_real_to_sim_10_3cam_v00/blob/main/sfagen/README.md), the same code and settings that
generate the Pringles hand-over dataset. Only the task description differs: source
dataset, objects (the 18650 battery (a 1.85 × 6.5 cm cylinder) and the charging station (box, size from the image)) and the shift grid. Full method: `PIPELINE.md`; positioning against prior work: `NOVELTY.md`.

1. **Rig calibration (automatic).**
   - **Body camera:** render-and-compare against the robot's own silhouette; held-out IoU 0.248 → 0.299.
   - **Table height:** 0.254 m (lowest arm point over the source demos - 5 mm (object silhouette fits not confident)).
2. **Reconstruction (per source demo, no labels).**
   - **Robot masks:** SAM2.1, prompted from the projected URDF.
   - **Objects:** segmented as what disappears when picked up. Known-size cylinders are posed by grasp-constrained render-and-compare.
   - **Time offset:** each demo's video/state offset is chosen photometrically.
3. **Two variation families** (kinematic edits of the recorded motion):
   - **Object position** (62 episodes): an object is moved ±3/±6 cm; the arm that grasps it follows (approach and grasp unchanged relative to the object) and rejoins the recording at the lift apex.
   - **Interaction location** (52 episodes, the insertion point): the whole two-arm interaction is moved ±3/±5 cm; both hands and everything they hold move together, so every recorded contact relation is kept.
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
| battery: within 2 cm (of 45) | 22 | 41 | **41** |
| battery: area covered (cm²; other real recordings span 260) | 95 | 224 | **248** |
| station: within 2 cm (of 45) | 16 | 29 | **31** |
| station: area covered (cm²; other real recordings span 242) | 61 | 151 | **128** |
| insertion: within 2 cm (of 45) | 34 | 34 | **43** |
| insertion: area covered (cm²; other real recordings span 69) | 91 | 91 | **236** |

## Image fidelity, measured against real ground truth

**Method.** Re-render real frame j+k from real frame j only (k = 4/8/15), compared with the real frame inside a region that
depends on no pipeline. Held-out episodes [3, 5, 8, 9] (284 pairs), never used for any tuning decision; 95% CI by episode bootstrap.

| renderer | PSNR (dB) | SSIM |
|---|---:|---:|
| copy the source frame (no motion) | 16.28 | 0.587 |
| previous revision's renderer | 18.40 | 0.681 |
| **this revision (sfagen)** | **19.21** | **0.709** |

Difference from the previous renderer: +0.82 dB [0.64, 0.99], better in 74% of pairs.

## Real vs synthetic, side by side

[highlights](https://huggingface.co/datasets/ethanCSL/openarm_insertbattery_lerobot_real_to_sim_10_3cam_v00/blob/main/comparison_videos/battery_unified_highlights.mp4) ·
[all episodes, 2×](https://huggingface.co/datasets/ethanCSL/openarm_insertbattery_lerobot_real_to_sim_10_3cam_v00/blob/main/comparison_videos/battery_unified_all_2x.mp4)
(left: real source; right: synthetic; the status bar says whether the frame is edited).

## Verification

- **Loading:** the dataset loads in LeRobot.
- **Labels:** state/action SHA-256 equal the plans.
- **Synthetic only:** 0 episodes equal a real episode.
- **Replay checks:** all pass.
- **Video:** every camera decodes with no black frames; the body camera matches the renders (41.19 dB after encoding).

## Limitations

- **Synthetic state.** `observation.state` is a planned kinematic candidate. Force, friction and contact are unchecked, and no episode has been replayed on hardware yet. Start with a ±3 cm episode, slowly.
- **Visible artifacts.** Arms re-posed close to the camera, or entering from the image edge, can show fragments. Watch the comparison videos.
- **Narrow variation.** Translations only (≤ 6 cm); no object rotation; the cable of the station (battery task) moves rigidly near the station.
- **Wrist cameras.** Not re-rendered.
- **No training evidence.** No policy has been trained on this revision yet.

## Episodes

| ep | source | variation | family | frames | shift (cm) | edited frames | max speed / real | max accel / real | min arm distance (mm) |
|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| 0 | 0 | `battery_x+0_y-3` | object position | 768 | (+0, -3) | 372 | 1.00 | 1.00 | 46 |
| 1 | 0 | `battery_x+3_y+0` | object position | 768 | (+3, +0) | 372 | 1.00 | 1.00 | 30 |
| 2 | 0 | `battery_x+6_y+0` | object position | 768 | (+6, +0) | 372 | 1.00 | 1.17 | 46 |
| 3 | 0 | `interact_x+0_y+3_z+0` | interaction location | 768 | (+0, +3) | 353 | 1.05 | 1.01 | 0 |
| 4 | 0 | `interact_x+0_y-3_z+0` | interaction location | 768 | (+0, -3) | 353 | 1.00 | 1.00 | 0 |
| 5 | 0 | `interact_x+0_y-5_z+0` | interaction location | 768 | (+0, -5) | 353 | 1.00 | 1.21 | 0 |
| 6 | 0 | `interact_x+3_y+0_z+0` | interaction location | 768 | (+3, +0) | 353 | 1.03 | 1.01 | 0 |
| 7 | 0 | `interact_x-3_y+0_z+0` | interaction location | 768 | (-3, +0) | 353 | 1.00 | 1.00 | 0 |
| 8 | 0 | `station_x+3_y+0` | object position | 768 | (+3, +0) | 174 | 1.10 | 1.00 | 122 |
| 9 | 0 | `station_x-3_y+0` | object position | 768 | (-3, +0) | 174 | 1.00 | 1.00 | 124 |
| 10 | 0 | `station_x-6_y+0` | object position | 768 | (-6, +0) | 174 | 1.00 | 1.00 | 125 |
| 11 | 1 | `battery_x+0_y-3` | object position | 776 | (+0, -3) | 307 | 1.00 | 1.00 | 48 |
| 12 | 1 | `battery_x+0_y-6` | object position | 776 | (+0, -6) | 307 | 1.00 | 1.00 | 73 |
| 13 | 1 | `battery_x+3_y+0` | object position | 776 | (+3, +0) | 307 | 1.00 | 1.00 | 49 |
| 14 | 1 | `battery_x+6_y+0` | object position | 776 | (+6, +0) | 307 | 1.00 | 1.00 | 61 |
| 15 | 1 | `interact_x+0_y+3_z+0` | interaction location | 776 | (+0, +3) | 375 | 0.99 | 1.00 | 0 |
| 16 | 1 | `interact_x+0_y-3_z+0` | interaction location | 776 | (+0, -3) | 375 | 1.01 | 1.01 | 0 |
| 17 | 1 | `interact_x+3_y+0_z+0` | interaction location | 776 | (+3, +0) | 375 | 1.07 | 1.13 | 0 |
| 18 | 1 | `interact_x-3_y+0_z+0` | interaction location | 776 | (-3, +0) | 375 | 0.95 | 0.99 | 0 |
| 19 | 1 | `interact_x-5_y+0_z+0` | interaction location | 776 | (-5, +0) | 375 | 0.92 | 1.24 | 0 |
| 20 | 1 | `station_x+0_y+3` | object position | 776 | (+0, +3) | 163 | 1.00 | 1.00 | 120 |
| 21 | 1 | `station_x-3_y+0` | object position | 776 | (-3, +0) | 163 | 1.00 | 1.00 | 114 |
| 22 | 1 | `station_x-6_y+0` | object position | 776 | (-6, +0) | 163 | 1.00 | 1.00 | 117 |
| 23 | 2 | `battery_x+0_y-3` | object position | 785 | (+0, -3) | 358 | 1.00 | 1.00 | 30 |
| 24 | 2 | `battery_x+0_y-6` | object position | 785 | (+0, -6) | 358 | 1.00 | 1.00 | 31 |
| 25 | 2 | `battery_x+3_y+0` | object position | 785 | (+3, +0) | 358 | 1.00 | 1.02 | 30 |
| 26 | 2 | `battery_x-3_y+0` | object position | 785 | (-3, +0) | 358 | 1.00 | 1.00 | 28 |
| 27 | 2 | `interact_x+0_y+3_z+0` | interaction location | 785 | (+0, +3) | 393 | 1.16 | 1.03 | 0 |
| 28 | 2 | `interact_x+0_y-3_z+0` | interaction location | 785 | (+0, -3) | 393 | 1.02 | 1.27 | 0 |
| 29 | 2 | `interact_x+3_y+0_z+0` | interaction location | 785 | (+3, +0) | 393 | 1.01 | 1.01 | 0 |
| 30 | 2 | `interact_x-3_y+0_z+0` | interaction location | 785 | (-3, +0) | 393 | 1.04 | 0.99 | 0 |
| 31 | 2 | `interact_x-5_y+0_z+0` | interaction location | 785 | (-5, +0) | 393 | 1.06 | 1.42 | 0 |
| 32 | 2 | `station_x+3_y+0` | object position | 785 | (+3, +0) | 156 | 1.00 | 1.00 | 118 |
| 33 | 2 | `station_x-3_y+0` | object position | 785 | (-3, +0) | 156 | 1.00 | 1.00 | 122 |
| 34 | 3 | `battery_x+0_y-3` | object position | 718 | (+0, -3) | 351 | 1.00 | 1.00 | 37 |
| 35 | 3 | `battery_x+3_y+0` | object position | 718 | (+3, +0) | 351 | 1.00 | 1.00 | 30 |
| 36 | 3 | `battery_x+6_y+0` | object position | 718 | (+6, +0) | 351 | 1.00 | 1.00 | 39 |
| 37 | 3 | `interact_x+0_y+3_z+0` | interaction location | 718 | (+0, +3) | 320 | 1.01 | 1.01 | 0 |
| 38 | 3 | `interact_x+0_y+5_z+0` | interaction location | 718 | (+0, +5) | 320 | 1.00 | 1.41 | 0 |
| 39 | 3 | `interact_x+0_y-3_z+0` | interaction location | 718 | (+0, -3) | 320 | 1.00 | 1.09 | 0 |
| 40 | 3 | `interact_x+3_y+0_z+0` | interaction location | 718 | (+3, +0) | 320 | 0.98 | 1.01 | 0 |
| 41 | 3 | `interact_x-3_y+0_z+0` | interaction location | 718 | (-3, +0) | 320 | 1.08 | 1.05 | 0 |
| 42 | 3 | `interact_x-5_y+0_z+0` | interaction location | 718 | (-5, +0) | 320 | 1.14 | 1.44 | 0 |
| 43 | 3 | `station_x-3_y+0` | object position | 718 | (-3, +0) | 184 | 1.00 | 1.00 | 134 |
| 44 | 3 | `station_x-6_y+0` | object position | 718 | (-6, +0) | 184 | 1.00 | 1.00 | 134 |
| 45 | 4 | `battery_x+0_y-3` | object position | 717 | (+0, -3) | 355 | 0.87 | 0.95 | 36 |
| 46 | 4 | `battery_x+3_y+0` | object position | 717 | (+3, +0) | 355 | 0.87 | 0.95 | 45 |
| 47 | 4 | `battery_x+6_y+0` | object position | 717 | (+6, +0) | 355 | 0.87 | 0.95 | 45 |
| 48 | 4 | `interact_x+0_y+3_z+0` | interaction location | 717 | (+0, +3) | 347 | 0.83 | 0.99 | 0 |
| 49 | 4 | `interact_x+0_y-3_z+0` | interaction location | 717 | (+0, -3) | 347 | 0.92 | 1.36 | 0 |
| 50 | 4 | `interact_x+3_y+0_z+0` | interaction location | 717 | (+3, +0) | 347 | 0.90 | 1.17 | 0 |
| 51 | 4 | `interact_x-3_y+0_z+0` | interaction location | 717 | (-3, +0) | 347 | 0.85 | 0.95 | 0 |
| 52 | 4 | `station_x+0_y+3` | object position | 717 | (+0, +3) | 175 | 0.87 | 0.95 | 97 |
| 53 | 4 | `station_x+0_y+6` | object position | 717 | (+0, +6) | 175 | 0.87 | 0.95 | 106 |
| 54 | 4 | `station_x-3_y+0` | object position | 717 | (-3, +0) | 175 | 0.87 | 0.95 | 93 |
| 55 | 4 | `station_x-6_y+0` | object position | 717 | (-6, +0) | 175 | 0.87 | 0.95 | 97 |
| 56 | 5 | `battery_x+0_y-3` | object position | 715 | (+0, -3) | 367 | 0.99 | 0.78 | 28 |
| 57 | 5 | `battery_x+0_y-6` | object position | 715 | (+0, -6) | 367 | 1.00 | 0.78 | 29 |
| 58 | 5 | `battery_x+3_y+0` | object position | 715 | (+3, +0) | 367 | 0.96 | 0.81 | 28 |
| 59 | 5 | `battery_x+6_y+0` | object position | 715 | (+6, +0) | 367 | 0.96 | 0.86 | 29 |
| 60 | 5 | `battery_x-3_y+0` | object position | 715 | (-3, +0) | 367 | 1.01 | 0.78 | 27 |
| 61 | 5 | `interact_x+0_y+3_z+0` | interaction location | 715 | (+0, +3) | 278 | 0.98 | 0.83 | 0 |
| 62 | 5 | `interact_x+0_y+5_z+0` | interaction location | 715 | (+0, +5) | 278 | 0.98 | 1.33 | 0 |
| 63 | 5 | `interact_x+0_y-3_z+0` | interaction location | 715 | (+0, -3) | 278 | 0.98 | 0.89 | 0 |
| 64 | 5 | `interact_x+3_y+0_z+0` | interaction location | 715 | (+3, +0) | 278 | 0.98 | 0.86 | 0 |
| 65 | 5 | `interact_x-3_y+0_z+0` | interaction location | 715 | (-3, +0) | 278 | 0.98 | 0.83 | 0 |
| 66 | 5 | `interact_x-5_y+0_z+0` | interaction location | 715 | (-5, +0) | 278 | 0.98 | 1.28 | 0 |
| 67 | 5 | `station_x+0_y+3` | object position | 715 | (+0, +3) | 167 | 0.98 | 0.84 | 128 |
| 68 | 5 | `station_x+0_y+6` | object position | 715 | (+0, +6) | 167 | 0.98 | 0.84 | 134 |
| 69 | 5 | `station_x+3_y+0` | object position | 715 | (+3, +0) | 167 | 0.98 | 0.84 | 121 |
| 70 | 5 | `station_x+6_y+0` | object position | 715 | (+6, +0) | 167 | 0.98 | 0.84 | 119 |
| 71 | 6 | `battery_x+0_y+3` | object position | 687 | (+0, +3) | 321 | 1.07 | 1.00 | 53 |
| 72 | 6 | `battery_x+0_y-3` | object position | 687 | (+0, -3) | 321 | 1.00 | 1.00 | 58 |
| 73 | 6 | `battery_x+0_y-6` | object position | 687 | (+0, -6) | 321 | 1.00 | 1.00 | 58 |
| 74 | 6 | `battery_x-3_y+0` | object position | 687 | (-3, +0) | 321 | 1.00 | 1.00 | 50 |
| 75 | 6 | `battery_x-6_y+0` | object position | 687 | (-6, +0) | 321 | 1.00 | 1.00 | 33 |
| 76 | 6 | `interact_x+0_y+3_z+0` | interaction location | 687 | (+0, +3) | 290 | 1.19 | 1.10 | 0 |
| 77 | 6 | `interact_x+0_y-3_z+0` | interaction location | 687 | (+0, -3) | 290 | 1.10 | 1.29 | 0 |
| 78 | 6 | `interact_x+3_y+0_z+0` | interaction location | 687 | (+3, +0) | 290 | 1.16 | 1.11 | 0 |
| 79 | 6 | `interact_x-3_y+0_z+0` | interaction location | 687 | (-3, +0) | 290 | 1.08 | 1.01 | 0 |
| 80 | 7 | `battery_x+0_y-3` | object position | 815 | (+0, -3) | 352 | 1.00 | 1.00 | 22 |
| 81 | 7 | `battery_x+3_y+0` | object position | 815 | (+3, +0) | 352 | 1.00 | 1.00 | 40 |
| 82 | 7 | `battery_x+6_y+0` | object position | 815 | (+6, +0) | 352 | 1.00 | 1.01 | 41 |
| 83 | 7 | `interact_x+0_y+3_z+0` | interaction location | 815 | (+0, +3) | 398 | 1.00 | 1.00 | 0 |
| 84 | 7 | `interact_x+0_y+5_z+0` | interaction location | 815 | (+0, +5) | 398 | 1.00 | 1.29 | 0 |
| 85 | 7 | `interact_x+0_y-3_z+0` | interaction location | 815 | (+0, -3) | 398 | 1.05 | 1.14 | 0 |
| 86 | 7 | `interact_x+3_y+0_z+0` | interaction location | 815 | (+3, +0) | 398 | 1.04 | 1.33 | 0 |
| 87 | 7 | `interact_x-3_y+0_z+0` | interaction location | 815 | (-3, +0) | 398 | 1.00 | 1.00 | 0 |
| 88 | 7 | `interact_x-5_y+0_z+0` | interaction location | 815 | (-5, +0) | 398 | 1.00 | 1.28 | 0 |
| 89 | 8 | `battery_x+0_y+3` | object position | 700 | (+0, +3) | 305 | 1.00 | 1.00 | 28 |
| 90 | 8 | `battery_x+0_y-3` | object position | 700 | (+0, -3) | 305 | 1.00 | 1.00 | 61 |
| 91 | 8 | `battery_x+0_y-6` | object position | 700 | (+0, -6) | 305 | 1.00 | 1.00 | 61 |
| 92 | 8 | `battery_x+3_y+0` | object position | 700 | (+3, +0) | 305 | 1.00 | 1.00 | 60 |
| 93 | 8 | `battery_x+6_y+0` | object position | 700 | (+6, +0) | 305 | 1.00 | 1.00 | 60 |
| 94 | 8 | `battery_x-3_y+0` | object position | 700 | (-3, +0) | 305 | 1.00 | 1.00 | 49 |
| 95 | 8 | `interact_x+0_y+3_z+0` | interaction location | 700 | (+0, +3) | 322 | 1.00 | 1.17 | 0 |
| 96 | 8 | `interact_x+0_y-3_z+0` | interaction location | 700 | (+0, -3) | 322 | 1.00 | 1.39 | 0 |
| 97 | 8 | `interact_x+3_y+0_z+0` | interaction location | 700 | (+3, +0) | 322 | 1.00 | 1.17 | 0 |
| 98 | 8 | `interact_x-3_y+0_z+0` | interaction location | 700 | (-3, +0) | 322 | 1.00 | 1.10 | 0 |
| 99 | 8 | `interact_x-5_y+0_z+0` | interaction location | 700 | (-5, +0) | 322 | 1.02 | 1.49 | 0 |
| 100 | 8 | `station_x+3_y+0` | object position | 700 | (+3, +0) | 144 | 1.00 | 1.00 | 130 |
| 101 | 9 | `battery_x+0_y+3` | object position | 759 | (+0, +3) | 358 | 1.00 | 0.73 | 0 |
| 102 | 9 | `battery_x+0_y+6` | object position | 759 | (+0, +6) | 358 | 1.00 | 0.73 | 0 |
| 103 | 9 | `battery_x+0_y-3` | object position | 759 | (+0, -3) | 358 | 1.00 | 0.73 | 1 |
| 104 | 9 | `battery_x+0_y-6` | object position | 759 | (+0, -6) | 358 | 1.13 | 1.45 | 1 |
| 105 | 9 | `battery_x-3_y+0` | object position | 759 | (-3, +0) | 358 | 1.00 | 0.73 | 0 |
| 106 | 9 | `battery_x-6_y+0` | object position | 759 | (-6, +0) | 358 | 1.00 | 0.73 | 0 |
| 107 | 9 | `interact_x+0_y+3_z+0` | interaction location | 759 | (+0, +3) | 346 | 0.99 | 0.74 | 0 |
| 108 | 9 | `interact_x+0_y+5_z+0` | interaction location | 759 | (+0, +5) | 346 | 0.98 | 0.89 | 0 |
| 109 | 9 | `interact_x+0_y-3_z+0` | interaction location | 759 | (+0, -3) | 346 | 0.99 | 1.30 | 0 |
| 110 | 9 | `interact_x+3_y+0_z+0` | interaction location | 759 | (+3, +0) | 346 | 0.98 | 1.16 | 0 |
| 111 | 9 | `interact_x-3_y+0_z+0` | interaction location | 759 | (-3, +0) | 346 | 1.01 | 0.86 | 0 |
| 112 | 9 | `interact_x-5_y+0_z+0` | interaction location | 759 | (-5, +0) | 346 | 1.03 | 1.48 | 0 |
| 113 | 9 | `station_x+3_y+0` | object position | 759 | (+3, +0) | 153 | 1.00 | 0.73 | 116 |
