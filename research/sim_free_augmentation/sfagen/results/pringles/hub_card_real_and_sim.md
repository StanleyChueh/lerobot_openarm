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

# openarm_pringles_lerobot_real_and_sim_10_3cam_v00

**Real + synthetic** OpenArm bimanual hand-over ("Pick up the Pringles can with the right arm, hand it to the left arm"), three cameras, LeRobot v3.0, 30 fps.

| episodes | what |
|---|---|
| 0–9 | the 10 real demonstrations of [`ethanCSL/openarm_pringles_lerobot_real_10_v00`](https://huggingface.co/datasets/ethanCSL/openarm_pringles_lerobot_real_10_v00) @ `cebdee98` (original videos, not re-encoded) |
| 10–105 | the 96 synthetic episodes of [`ethanCSL/openarm_pringles_lerobot_real_to_sim_10_3cam_v00`](https://huggingface.co/datasets/ethanCSL/openarm_pringles_lerobot_real_to_sim_10_3cam_v00) (v3.0) |

**Telling them apart.**
- **Per frame:** `is_synthetic` is 0 for real and 1 for synthetic.
- **Per episode:** `meta/sfa_provenance.json` gives each episode's source, family, shift and replay checks.

These extra features (`is_synthetic`, `wrist_view_valid`, `wrist_view_pose_offset`) have no `observation.` prefix, so policies ignore them. Use them to filter or re-weight.

**Checks.**
- **Real episodes:** labels identical to the source (True).
- **Flag:** `is_synthetic` correct on every frame (True).

**Previous revision:** tag `v00-75ep-two-pipelines` (`13d7231`).

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

## Limitations

- **Synthetic state.** `observation.state` is a planned kinematic candidate. Force, friction and contact are unchecked, and no episode has been replayed on hardware yet. Start with a ±3 cm episode, slowly.
- **Visible artifacts.** Arms re-posed close to the camera, or entering from the image edge, can show fragments. Watch the comparison videos.
- **Narrow variation.** Translations only (≤ 6 cm); no object rotation; the cable of the station (battery task) moves rigidly near the station.
- **Wrist cameras.** Not re-rendered.
- **No training evidence.** No policy has been trained on this revision yet.
