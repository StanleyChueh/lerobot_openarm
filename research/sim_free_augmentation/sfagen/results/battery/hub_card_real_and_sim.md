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

# openarm_insertbattery_lerobot_real_and_sim_10_3cam_v00

**Real + synthetic** OpenArm bimanual battery insertion ("Pick up the charging station with the left arm, and insert battery in it with the right arm"), three cameras, LeRobot v3.0, 30 fps.

| episodes | what |
|---|---|
| 0–9 | the 10 real demonstrations of [`ethanCSL/openarm_insertbattery_lerobot_real_10_v02`](https://huggingface.co/datasets/ethanCSL/openarm_insertbattery_lerobot_real_10_v02) @ `a1c29d77` (original videos, not re-encoded) |
| 10–123 | the 114 synthetic episodes of [`ethanCSL/openarm_insertbattery_lerobot_real_to_sim_10_3cam_v00`](https://huggingface.co/datasets/ethanCSL/openarm_insertbattery_lerobot_real_to_sim_10_3cam_v00) (v3.0) |

**Telling them apart.**
- **Per frame:** `is_synthetic` is 0 for real and 1 for synthetic.
- **Per episode:** `meta/sfa_provenance.json` gives each episode's source, family, shift and replay checks.

These extra features (`is_synthetic`, `wrist_view_valid`, `wrist_view_pose_offset`) have no `observation.` prefix, so policies ignore them. Use them to filter or re-weight.

**Checks.**
- **Real episodes:** labels identical to the source (True).
- **Flag:** `is_synthetic` correct on every frame (True).

**Previous revision:** tag `v00-75ep-two-pipelines` (`53de718`).

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

## Limitations

- **Synthetic state.** `observation.state` is a planned kinematic candidate. Force, friction and contact are unchecked, and no episode has been replayed on hardware yet. Start with a ±3 cm episode, slowly.
- **Visible artifacts.** Arms re-posed close to the camera, or entering from the image edge, can show fragments. Watch the comparison videos.
- **Narrow variation.** Translations only (≤ 6 cm); no object rotation; the cable of the station (battery task) moves rigidly near the station.
- **Wrist cameras.** Not re-rendered.
- **No training evidence.** No policy has been trained on this revision yet.
