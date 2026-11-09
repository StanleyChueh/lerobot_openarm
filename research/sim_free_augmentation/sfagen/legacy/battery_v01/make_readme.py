import json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; T = HERE.parent / "target"
prov = json.loads((HERE / "dataset/meta/sfa_provenance.json").read_text()); eps = prov["episodes"]
ver = json.loads((HERE / "verification.json").read_text())
tm = json.loads((T / "figs/target_metrics.json").read_text())
cal = json.loads((T / "calib_battery.json").read_text())
b2 = pd.DataFrame(json.loads((T / "bbench2_heldout.json").read_text())); pa = pd.DataFrame(json.loads((T / "ablate_plate_bat_heldout.json").read_text()))
m = b2.merge(pa, on=["E", "j", "k"])
def ci(d, df):
    rng = np.random.default_rng(0); E = df.E.unique()
    bs = [np.mean(np.concatenate([d[df.E == e].to_numpy() for e in rng.choice(E, len(E))])) for _ in range(2000)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), (d > 0).mean()
c_cam = ci(m.fitted_psnr - m.inherited_psnr, m); c_pa = ci(m.pa10_psnr - m.pa0_psnr, m); c_all = ci(m.pa10_psnr - m.inherited_psnr, m)
nb = sum(e["family"] == "battery_position" for e in eps); ns = len(eps) - nb
st, ins, bg = tm["battery/station_pick"], tm["battery/insertion"], tm["battery/battery_grasp"]
rows = []
for e in eps:
    r = e["replay_checks"]; tw = e["targets_world_m"]
    obj = e.get("battery_initial_world_m") if e["family"] == "battery_position" else e.get("station_centre_initial_world_m")
    rows.append(f"| {e['export_episode_index']} | {e['source_real_episode']} | `{e['variation']}` | {e['family'].replace('_', ' ')} | {e['n_frames']} | "
                f"({100*e['shift_m'][0]:+.0f}, {100*e['shift_m'][1]:+.0f}) | ({100*obj[0]:.1f}, {100*obj[1]:.1f}) | "
                f"({100*tw['station_pick_left_tcp'][0]:.1f}, {100*tw['station_pick_left_tcp'][1]:.1f}) | {e['schedule']['t_rejoin']} | "
                f"{r['speed_ratio_to_real_max']:.2f} | {r['min_hand_distance_mm']:.0f} / {r['source_min_hand_distance_same_frames_mm']:.0f} | {100*e['wrist_valid_fraction']:.0f}% |")
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
`right_wrist_cam`), LeRobot v3.0, 30 fps.

**Every episode is synthetic.**
- **Count:** {len(eps)} episodes ({ver['frames']:,} frames).
- **Source:** generated **without a physics simulator** from the 10 real demonstrations of
  [`ethanCSL/openarm_insertbattery_lerobot_real_10_v02`](https://huggingface.co/datasets/ethanCSL/openarm_insertbattery_lerobot_real_10_v02)
  (revision `a1c29d77`); sources used: {', '.join(str(x) for x in ver['sources_used'])} (no candidate from source {', '.join(str(x) for x in sorted(set(range(10)) - set(ver['sources_used'])))} passed every check in this revision).
- **Variation families:**
  - **{nb} battery-position episodes:** the battery's resting position moves by ±3/±6 cm. The right arm's reach, grasp and lift follow it.
  - **{ns} station-position episodes (target-position variation):** the charging station's resting position moves by ±3/±6 cm. The left arm's reach, grasp and carry follow it.
- **Insertion:** in both families the insertion itself is the recorded, physically executed bimanual motion.

**Exact pipeline and method:** `PIPELINE.md` in this repository.

## Revision history

| tag / commit | content |
|---|---|
| `v00-36ep-inherited-cam` (`b41a6ef`) | 36 battery-position episodes; camera inherited from the Pringles rig; station never varied |
| current (`v3.0`) | **{len(eps)} episodes: {nb} battery-position + {ns} station-position; body camera re-calibrated on this rig; relit background** |

## 1. Target positions: what varies, measured

All positions come from forward kinematics of the joint states and gripper commands, with no annotation:
- **station pick point:** left hand at the left-gripper close;
- **insertion point:** right hand at the first right-gripper open;
- **battery position:** right hand at the battery grasp.

The same measurement is applied to the 45 other real recordings of
[`openarm_insertbattery_lerobot_real_v02`](https://huggingface.co/datasets/ethanCSL/openarm_insertbattery_lerobot_real_v02).

| | 10 sources | + previous revision (v00) | + this revision |
|---|---:|---:|---:|
| **station pick point**: convex hull (cm²) | {st['source']['hull_cm2']:.0f} | {st['synthetic_v00']['hull_cm2']:.0f} | **{st['synthetic_v01']['hull_cm2']:.0f}** |
| station pick: other real recordings within 2 cm (of {st['n_other_real']}) | {st['source']['other_real_within_2cm']} | {st['synthetic_v00']['other_real_within_2cm']} | **{st['synthetic_v01']['other_real_within_2cm']}** |
| station pick: median distance to nearest training position (cm) | {st['source']['other_real_nn_median_cm']:.2f} | {st['synthetic_v00']['other_real_nn_median_cm']:.2f} | **{st['synthetic_v01']['other_real_nn_median_cm']:.2f}** |
| **insertion point**: other real recordings within 2 cm | {ins['source']['other_real_within_2cm']} | {ins['synthetic_v00']['other_real_within_2cm']} | {ins['synthetic_v01']['other_real_within_2cm']} |
| **battery position**: other real recordings within 2 cm | {bg['source']['other_real_within_2cm']} | {bg['synthetic_v00']['other_real_within_2cm']} | {bg['synthetic_v01']['other_real_within_2cm']} |

**Station pick point.**
- **Before:** v00 did not move the station at all (0.0 mm from the source in every episode), although the real recordings spread it over {st['other_real_hull_cm2']:.0f} cm² and the 10 sources cover only {st['source']['hull_cm2']:.0f} cm² of that.
- **Now:** {st['synthetic_v01']['new_targets_gt_1cm_from_every_source']} synthetic episodes place it more than 1 cm from every source.
- **Remaining gap:** shifts toward −y (closer to the robot body) are mostly rejected by the station-clearance check, so the region y < 7 cm stays sparse.

**Insertion point.** It is deliberately unchanged, because the insertion is kept as the recorded motion. The 10 sources already cover {ins['source']['other_real_within_2cm']} of 45 real recordings within 2 cm.

![station pick points](assets/target_battery_station_pick.png)
![insertion points](assets/target_battery_insertion.png)
![battery positions](assets/target_battery_battery_grasp.png)

## 2. Trajectories (object-centric segment transform)

**One arm follows one moved object** (MimicGen, DemoGen):
- **Moved object:** the battery (right arm) or the station (left arm) is translated by d on the table.
- **Before the grasp:** that arm's TCP trajectory gets the offset w(t)·d. w ramps 0→1 (smoothstep) from motion onset to 15 frames before the grasp, so the approach and the grasp are the recorded motion *relative to the object*.
- **Through the grasp:** w = 1 until 10 frames after the grasp.
- **After the grasp:** w ramps back to 0, rejoining the recording before the bimanual phase.
  - Battery: the rejoin is the lift apex.
  - Station: it is the earlier of "left hand settles at the insertion pose" and "right hand within 20 cm of the left hand".

**Joints and actions:**
- Joints come from bounded damped-least-squares IK. The joint offset is Gaussian-smoothed (σ = 3 frames) and re-verified by FK (≤ 1 mm, ≤ 0.5°).
- Actions keep the recorded command-to-state lead: a′(t) = a(t) + Δq(t + k).

## 3. Replay checks (every published episode passes all of them)

- **Joint limits:** URDF limits extended to the demo's own recorded range.
- **Speed and acceleration:** ≤ 1.25× and ≤ 1.5× the maximum of the 10 real demos.
- **Arm-to-arm and arm-to-body clearance:** mesh distance (URDF collision meshes, coal) not below the source over the same frames (floored at 2 cm).
- **Station clearance:** the hands never closer to each other than in the source (−5 mm).
- **Table:** clearance holds.
- **Untouched parts:** the other arm is bit-identical to the recording; the start pose is the recorded one; frames after the rejoin are identical to the recording.
- **Placement:**
  - the moved object is on the table and in view;
  - battery and station stay clear of each other;
  - battery: ≥ 10 cm from the station;
  - station: in no frame before the final approach is the robot closer to the moved station than the recorded robot was to the recorded station (robot hull points within 1 cm of the box, count compared frame by frame).

The variation grid has 80 battery and 80 station candidates. {nb} and {ns} passed, respectively.

## 4. Body-camera rendering

**Mesh-guided forward warping.** This is the same algorithm as the Pringles dataset (`render_v4/mgw.py`):
- every real robot and object pixel gets a 3-D point from the z-buffered URDF / object-proxy render;
- it moves with its part's rigid motion, with per-pixel occlusion;
- newly visible surfaces are borrowed from other frames of the same recording;
- remaining holes are filled from the same part's pixels.

No generative model is used.

**New in this revision:**
- **Camera calibrated on this rig.** v00 used the body-camera model of the Pringles rig, which is 20–40 px off here, and that caused most of the arm fragmentation.
  - **Method:** render-and-compare (cf. [CtRNet](https://arxiv.org/abs/2302.14332), [RoboPose](https://arxiv.org/abs/2104.09359)).
  - **Fit:** IoU between the URDF silhouette and a camera-independent foreground (difference from the temporal median of all frames) on near-static frames, fitted on even episodes and evaluated on odd ones.
  - **Result:** held-out IoU {cal['heldout_iou']['inherited']:.3f} → {cal['heldout_iou']['fitted']:.3f}, better in {100*cal['heldout_frames_improved']:.0f}% of held-out frames.
- **Station as an image-segmented object.**
  - **Segmentation:** no CAD model or label is used. The station is the foreground in frame 0 that is not robot and not battery, connected to the left hand at its grasp.
  - **Geometry:** an oriented-box proxy gives it depth and occlusion; its pixels are always the real ones.
- **Relit background.** The clean plate is relit per frame (normalised-convolution log-ratio, σ = 10 px), which removes dark halos where a moved part used to be.

**Ground-truth benchmark.**
- **Method:** re-render real frame j+k from real frame j only (k = 4/8/15), scored on held-out episodes 3, 5, 8, 9 ({len(m)} pairs). The region is camera-independent (foreground of both frames), so neither camera decides where it is scored.

| renderer | PSNR (dB) | SSIM |
|---|---:|---:|
| copy the source frame (no motion) | {m.copy_psnr.mean():.2f} | {m.copy_ssim.mean():.3f} |
| v00 renderer (inherited camera) | {m.inherited_psnr.mean():.2f} | {m.inherited_ssim.mean():.3f} |
| + calibrated camera | {m.fitted_psnr.mean():.2f} | {m.fitted_ssim.mean():.3f} |
| **+ relit background (this revision)** | **{m.pa10_psnr.mean():.2f}** | **{m.pa10_ssim.mean():.3f}** |

Improvement over v00, with episode-bootstrap 95% CI:
- **Camera:** {c_cam[0]:+.2f} dB [{c_cam[1]:.2f}, {c_cam[2]:.2f}].
- **Relighting:** {c_pa[0]:+.2f} dB [{c_pa[1]:.2f}, {c_pa[2]:.2f}].
- **Total:** {c_all[0]:+.2f} dB [{c_all[1]:.2f}, {c_all[2]:.2f}], better in {100*c_all[3]:.0f}% of pairs.

Both settings were chosen on development episode 1 before the held-out run.

![examples](assets/examples.jpg)

*Top rows: real source. Bottom rows: synthetic. Episode 5 `bat_x-6` shows the remaining failure mode: gripper
fragments while the right arm grasps near the bottom image edge.*

## 5. Wrist cameras

Every wrist frame is the real recorded wrist frame at the same time index. `wrist_view_valid = 1` marks frames where
the whole scene (all joints, battery and station) equals the recording ({100*sum(e['wrist_valid_fraction']*e['n_frames'] for e in eps)/sum(e['n_frames'] for e in eps):.0f}% of frames exact).
`wrist_view_pose_offset` gives the per-frame offset of each synthetic wrist from the copied view.

## Limitations

- **Synthetic state.** `observation.state` is a planned kinematic candidate. Force, friction and contact are unverified; no episode has been replayed on hardware. Start with a ±3 cm episode, slowly.
- **Proxies.**
  - The station is a box proxy: depth and occlusion are approximate, and its cable moves rigidly with it near the station and stays put farther away.
  - Station clearance is checked on the box and on the hand distance.
- **Remaining artifacts.** Gripper fragments and smearing remain where an arm enters from the image border near the camera (right forearm during the battery grasp; left forearm during some station picks).
- **Narrow variation.** Translations only (±3/±6 cm). No object rotation, no change to the insertion itself; episodes are variants of 10 real demos.
- **No training evidence.** No policy has been trained on this revision yet.

## Episodes

| ep | source | variation | family | frames | shift (cm) | moved object start (x, y) cm | station pick (x, y) cm | rejoin frame | max speed / real max | min hand distance / source (mm) | exact wrist frames |
|---:|---:|---|---|---:|---|---|---|---:|---:|---|---:|
""" + "\n".join(rows) + "\n"
(HERE / "README.md").write_text(card); print("ok", len(rows))
