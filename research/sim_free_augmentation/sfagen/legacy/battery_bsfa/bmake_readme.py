import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
prov = json.loads((HERE / "dataset/meta/sfa_provenance.json").read_text()); eps = prov["episodes"]
ver = json.loads((HERE / "verification.json").read_text()); hm = json.loads((HERE / "analysis/battery_heatmap_metrics.json").read_text())
bb = json.loads((HERE / "analysis/benchmark_summary.json").read_text()); env = json.loads((HERE / "cache/envelope.json").read_text())
gen = {"n_candidates": 80, "n_pass": len(eps)}
rows = []
for e in eps:
    r = e["replay_checks"]; d = 100 * np.array(e["battery_shift_m"]); p = 100 * np.array(e["right_tcp_at_grasp_world_m"][:2])
    rows.append(f"| {e['export_episode_index']} | {e['source_real_episode']} | `{e['variation']}` | {e['n_frames']} | ({d[0]:+.0f}, {d[1]:+.0f}) | "
                f"({p[0]:.1f}, {p[1]:.1f}) | {e['schedule']['t_rejoin']} | {r['speed_ratio_to_real_max']:.2f} | {r['min_hand_distance_mm']:.0f} / {r['source_min_hand_distance_same_frames_mm']:.0f} | {100*e['wrist_valid_fraction']:.0f}% |")
H = hm["convex_hull_cm2"]; C = hm["other_real_to_nearest_training_cm"]; W2 = hm["other_real_within_2cm"]; R = hm["xy_range_cm"]
card = f"""---
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

Bimanual OpenArm battery insertion ("{prov['task']}"), three cameras (`body_cam`, `wrist_cam` = left wrist,
`right_wrist_cam`), LeRobot v3.0, 30 fps, `robot_type` `openarm_umeow`.

**Every episode is synthetic.**
- **Count:** {len(eps)} episodes ({ver['frames']:,} frames).
- **Source:** generated **without a physics simulator** from the 10 real demonstrations of
  [`ethanCSL/openarm_insertbattery_lerobot_real_10_v02`](https://huggingface.co/datasets/ethanCSL/openarm_insertbattery_lerobot_real_10_v02)
  (revision `{prov['source_revision'][:8]}`), all 10 used as sources.
- **Variation:** the battery's resting position on the table moves by ±3 or ±6 cm, and the right arm reaches, grasps and lifts it there.
- **Insertion:** the insertion into the station held by the left arm is the recorded, physically executed motion.

![battery positions](assets/battery_heatmap_topdown.png)

## Method

The pipeline is the one used for
[`openarm_pringles_lerobot_real_to_sim_10_3cam_v00`](https://huggingface.co/datasets/ethanCSL/openarm_pringles_lerobot_real_to_sim_10_3cam_v00).
It is a simulation-free real-to-real augmentation in the line of
[MimicGen](https://arxiv.org/abs/2310.17596) and [DemoGen](https://arxiv.org/abs/2502.16932) for trajectories, and
[Real2Edit2Real](https://arxiv.org/abs/2512.19402), [Phantom](https://arxiv.org/abs/2503.00779) and
[Masquerade](https://arxiv.org/abs/2508.09976) for geometry-grounded image editing. Every output pixel is a real
camera pixel; no generative model is used.

1. **Reconstruction, without manual annotation.**
   - **Robot geometry:** URDF meshes with the body-camera model, aligned on this rig.
   - **Background:** a clean plate, the temporal median of pixels that the dilated mesh render marks as free.
   - **Battery:** an 18650 cylinder. Its resting pose combines the right-hand pose at the grasp (kinematics) with the
     battery's bright blob in frame 0 (centroid and principal axis, back-projected to the grasp height).
   - **Attachment:** the battery is attached to the right hand from the grasp onward.
2. **Object-centric trajectory adaptation** (MimicGen/DemoGen-style segment transform):
   - **Reach and grasp:** the right-hand pose follows the battery's translation, blended in smoothly from motion onset
     to 15 frames before the grasp. The final approach and grasp are therefore the recorded motion *relative to the
     battery*.
   - **Rejoin:** the translation is blended out during the lift, and the trajectory rejoins the recording at the lift
     apex. Everything after that (moving to the station, insertion, pressing, return) is the recorded motion.
   - **Joints and actions:** damped-least-squares IK, with the joint offset temporally smoothed (σ = 3 frames) and
     re-verified by FK (≤ 1 mm, ≤ 0.5°). Actions keep the recorded command-to-state lead.
3. **Replay checks.** Every published episode passes all of these; {len(eps)} of {gen['n_candidates']} candidates did.
   - **Joint limits:** within the URDF limits extended to the demo's own recorded range.
   - **Speed and acceleration:** ≤ 1.25× / ≤ 1.5× the maximum over the 10 real demos.
   - **Arm clearance:** mesh distance (coal, URDF collision meshes) never closer than the source over the same frames (floored at 2 cm).
   - **Station clearance:** the right hand never closer to the station-holding left hand than in the source. The station itself is not modelled, so this is the proxy.
   - **Table:** table clearance holds.
   - **Untouched parts:** the left arm and both grippers are untouched, and the start pose is the recorded one.
   - **Battery placement:** on the table, in view, not under the robot, and ≥ 10 cm from the station's resting place.
4. **Body-camera rendering: mesh-guided forward warping** (`render_v4/mgw.py`).
   - **Masks:** robot masks come from the z-buffered mesh render plus background difference.
   - **Motion:** every pixel gets a 3-D point from the mesh depth and moves with its link's rigid motion, with per-pixel
     occlusion.
   - **Hidden surfaces:** surfaces hidden in the source frame are borrowed from nearby frames.
   - **Remaining holes:** these are filled from the same part's nearest pixels (part-aware fill, chosen on development
     episode 1).
   - **Copied frames:** frames after the rejoin are the recorded frames.
5. **Wrist cameras.** Every wrist frame is the real recorded wrist frame at the same time.
   - **`wrist_view_valid = 1`:** the whole scene (all joints and battery pose) equals the recording, which is the case after the rejoin.
   - **Before the rejoin:** the right wrist view is the recorded one, while the synthetic right wrist is offset by up to the battery shift; `wrist_view_pose_offset` gives the offset.

## Renderer benchmark on this task (real ground truth)

- **Method:** re-render real frame `j+k` from frame `j` (k = 4/8/15), compared with the real frame inside the robot+battery region.
- **Data:** held-out episodes {bb['episodes']}, {bb['n']} pairs.

| | copy source frame | mesh-guided warp (Pringles config) | + part-aware fill (used) |
|---|---:|---:|---:|
| PSNR (dB) | {bb['copy_psnr']:.2f} | {bb['pringles_cfg_psnr']:.2f} | **{bb['partfill_tm21_psnr']:.2f}** |
| SSIM | {bb['copy_ssim']:.3f} | {bb['pringles_cfg_ssim']:.3f} | **{bb['partfill_tm21_ssim']:.3f}** |

## Battery positions: 10 sources vs this dataset

Position = the right-hand position at the battery grasp. This is computed identically for every real and synthetic
episode from joint states, with no annotation.

- **Range:** x × y range {R['source'][0]} × {R['source'][1]} cm in the sources; {R['source+synthetic'][0]} × {R['source+synthetic'][1]} cm with the synthetic episodes.
- **Convex hull:** {H['source']:.0f} → {H['source+synthetic']:.0f} cm².
- **Coverage of the {hm['n']['other_real']} other real recordings** of
  [`openarm_insertbattery_lerobot_real_v02`](https://huggingface.co/datasets/ethanCSL/openarm_insertbattery_lerobot_real_v02):
  - median distance to the nearest training position: {C['source_only']['median']:.1f} → {C['source+synthetic']['median']:.1f} cm (p90 {C['source_only']['p90']:.1f} → {C['source+synthetic']['p90']:.1f} cm);
  - within 2 cm: {W2['source_only']} → {W2['source+synthetic']} of {W2['of']}.
- **Direction bias:** shifts toward the station are rejected by the clearance checks, so the coverage grows mostly away from the station.

![battery positions in the camera](assets/battery_heatmap_bodycam.png)

## Limitations

- **Synthetic state.** `observation.state` is a planned kinematic candidate, not a measurement. Force, friction and contact are unverified.
- **Not replayed.** No episode has been replayed on hardware. To replay, place the battery at the listed position (the first body-camera frame shows it) and use `lerobot-replay`. Start slow with a ±3 cm episode.
- **Proxies.** The station is not modelled, so clearance to it is a hand-distance proxy. The battery pose is an estimate (kinematics plus image blob).
- **Rendering artifacts.** They remain on the large right-forearm cover near the bottom image edge during the grasp, where the arm moves into view from outside the frame: smeared texture and occasional fragments.
- **Narrow variation.** Only the battery position varies. The station pick, insertion and placement are always the recorded ones, and episodes are variants of 10 real demos.
- **No training evidence.** No policy has been trained on this dataset yet.

## Episodes

| ep | source | variation | frames | battery shift (cm) | battery position (x, y) cm | rejoin frame | max speed / real max | min hand distance / source (mm) | exact wrist frames |
|---:|---:|---|---:|---|---|---:|---:|---|---:|
""" + "\n".join(rows) + "\n"
(HERE / "README.md").write_text(card); print("ok", len(rows))
