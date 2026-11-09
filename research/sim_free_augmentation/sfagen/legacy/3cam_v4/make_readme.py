import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
prov = json.loads((HERE / "dataset/meta/sfa_provenance.json").read_text())
ver = json.loads((HERE / "verification.json").read_text())
V3 = HERE.parent / "3cam_v3"
hm = json.loads((V3 / "analysis/can_heatmap_metrics.json").read_text())
cons = json.loads((V3 / "analysis/consistency_summary.json").read_text())
bh = json.loads((V3 / "analysis/benchmark_summary.json").read_text())
tm = json.loads((HERE.parent / "target/figs/target_metrics.json").read_text())["pringles/handover"]
sel4 = json.loads((HERE / "selection.json").read_text())
CAM = json.loads((HERE.parent / "target/abl_cam_pr_heldout.json").read_text())
tc = json.loads((HERE.parent / "render_v4/traj_check.json").read_text())
eps = prov["episodes"]
rows = []
for e in eps:
    r = e["replay_checks"]; c0 = 100 * np.array(e["can_initial_world_m"][:2]); s0 = 100 * np.array(e["source_can_initial_world_m"][:2])
    rows.append(f"| {e['export_episode_index']} | {e['source_real_episode']} | `{e['variation']}` | {e['n_frames']} | "
                f"({c0[0]:.1f}, {c0[1]:.1f}) | ({c0[0]-s0[0]:+.1f}, {c0[1]-s0[1]:+.1f}) | {e['handover_shift_mm']:.0f} | {r['speed_ratio_to_real_max']:.2f} | "
                f"{r['min_mesh_distance_mm']:.0f} | {100*np.mean(e['wrist_valid']):.0f}% | {e['wrist_offset_max']['right_trans_cm']} cm |")
nv = sum(1 for e in eps if e["variation"].startswith("init")); nh = sum(1 for e in eps if e["condition"] == "C"); nt = len(eps) - nv - nh
ho_eps = ", ".join(str(e["export_episode_index"]) for e in eps if e["condition"] == "C")
import pandas as _pd
_c = _pd.DataFrame(CAM); _d = (_c.fitted_psnr - _c.inherited_psnr).mean()
g = sel4["gate"]; rj = sel4["rejected_handover"]
R = hm["start_xy_range_cm"]; P = hm["pool10_49_start_to_nearest_training_start_cm"]; W2 = hm["pool10_49_starts_within_2cm_of_a_training_start"]
env = tc["envelope"]
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
- handover
configs:
- config_name: default
  data_files: data/*/*.parquet
---

# openarm_pringles_lerobot_real_to_sim_10_3cam_v00

Bimanual OpenArm hand-over task ("Pick up the Pringles can with the right arm, hand it to the left arm"), three cameras
(`body_cam`, `wrist_cam` = left wrist, `right_wrist_cam`).

**Every episode is synthetic.**
- **Count:** {len(eps)} episodes ({ver['frames']:,} frames).
- **Source:** generated **without a physics simulator** from the 10 real demonstrations of
  [`ethanCSL/openarm_pringles_lerobot_real_10_v00`](https://huggingface.co/datasets/ethanCSL/openarm_pringles_lerobot_real_10_v00)
  (revision `cebdee98`, source episodes {', '.join(str(x) for x in sorted({e['source_real_episode'] for e in eps}))}).
- **Variants:**
  - {nv} episodes put the can at a new initial position (±3/±6 cm in x or y);
  - {nt} episodes re-time the hand-over;
  - **{nh} episodes move the hand-over point itself** (the target position) by {min(e["handover_shift_mm"] for e in eps if e["condition"] == "C"):.0f}–{max(e["handover_shift_mm"] for e in eps if e["condition"] == "C"):.0f} mm (episodes {ho_eps}).
- **No real recordings:** no real recording is included as an episode.

![can positions](assets/can_heatmap_topdown.png)

## Revision history

| tag / commit | content |
|---|---|
| `v00-35ep-with-real` (`0b8d50b`) | 10 real + 25 synthetic; black wrist placeholders |
| `v01-25ep-black-wrist` (`15d337b`) | 25 synthetic; black wrist placeholders |
| `v02-20ep-v3render` (`f9692b3`) | 20 synthetic; wrist frames filled; old (v3) renderer |
| `v03-55ep-mgw` (`a66738c`) | 55 synthetic; mesh-guided renderer; can-position variants; hand-over point never varied |
| current (`v3.0`) | **{len(eps)} synthetic: the 55 above unchanged + {nh} hand-over-location episodes (target-position variation)** |

**Exact pipeline and method for this dataset and the battery dataset:** see `PIPELINE.md` in this repository.

## 1. Trajectories checked for real-robot replay

Every episode passes these offline checks on the **commanded action stream** (what `lerobot-replay` sends) and the
planned state (`render_v4/traj_check.py`):

- **Joint limits:** inside the URDF limits; 0 violations.
- **Speed:** the largest per-joint speed in any episode is ≤ 1.25× the largest of the 10 real demos. The same holds for acceleration at ≤ 1.5×. Five over-fast candidates were removed, including one (source 5 `timing_late`) that was in an earlier revision.
- **Collisions:** the minimum mesh distance between both arms and the body (URDF collision meshes, coal) must be ≥ 5 mm. Across all published episodes it is {min(e['replay_checks']['min_mesh_distance_mm'] for e in eps):.1f} mm at the closest, vs {env['real_min_mesh_distance_mm']:.1f} mm in the 10 real demos. (Earlier revisions of this card wrongly said the gate was the real demos' minimum.)
- **Table clearance:** the lowest arm point is never below the real demos' lowest point minus 1 cm.
- **Grasp:** the hand–can pose at the right grasp equals the source demo's (≤ 2 mm), because the can and hand move together.
- **Hand-over:** in the {len(eps)-nh} can-position and timing episodes, the receiving (left) arm follows the real recording exactly (≤ 1 mm), so the hand-over itself is a recorded, physically executed motion. In the {nh} hand-over-location episodes, **both arms and the can move together by the same rigid transform** through the whole transfer (see section 3b), so the hand–hand–can relation of the recorded hand-over is kept at every frame.
- **Start pose:** frame 0 equals a recorded pose (0.000 rad error).

**Replay.** Place the can at the episode's `can_initial_world_m` (also listed in the table below as world x, y in
cm). Equivalently, match the can to the first body-camera frame. Then replay the action stream, e.g.
`lerobot-replay --dataset.repo_id=ethanCSL/openarm_pringles_lerobot_real_to_sim_10_3cam_v00 --dataset.episode=N` with
your OpenArm robot config.

**These checks are kinematic and geometric only.** No force, friction or contact-stability check is possible without
a simulator or the robot, and **no episode has been replayed on hardware yet**. For a first replay, start slow, keep a
hand on the stop, and check an episode with a small shift (±3 cm) first.

## 2. Body-camera rendering: what changed and how much better it is

**Why the earlier revisions distorted the arms.**
- **Missed robot parts.** The old renderer cut the real arm pixels into per-link pieces using SAM2 masks, and those
  masks miss whole gripper bodies in carry and hand-over frames. Missed parts stay behind while the rest moves, which
  looks like a torn arm.
- **Flat warps.** Each piece was moved with a 2-D homography. That is only exact for flat objects, and it can leave a
  gripper where it was even though the labels say it moved.

**The new renderer (mesh-guided warping).** It follows recent simulation-free pipelines: Real2Edit2Real's
geometry-based editing of real pixels, and Phantom/Masquerade's rendering of the robot from its mesh with a known
camera.
- **Masks.** Robot masks are SAM2 ∪ the URDF mesh render, kept only where pixels differ from the clean background
  plate, so whole gripper bodies are included.
- **Motion.** Every real arm/can pixel gets a 3-D position from the z-buffered mesh render and moves with its link's
  exact rigid motion. Occlusions are resolved per pixel by depth.
- **Newly revealed surfaces.** These are borrowed from other frames of the same recording where they are visible.
- **No generated content.** No generative model and no simulator are used; all pixels are real.

**Ground-truth benchmark.**
- **Method:** re-render real frame `j+k` from real frame `j` only, then compare with the real frame `j+k`.
- **Held out:** 7 episodes never used for tuning, {bh['n']} pairs, k = 4/8/15 frames.

| | old renderer | new renderer |
|---|---:|---:|
| PSNR (dB, robot+can region) | {bh['v3_psnr']:.2f} | **{bh['mgw_psnr']:.2f}** (+{bh['d_psnr']:.2f}, 95% CI {bh['d_psnr_ci'][0]:.2f}–{bh['d_psnr_ci'][1]:.2f}) |
| SSIM | {bh['v3_ssim']:.3f} | **{bh['mgw_ssim']:.3f}** |
| pairs where the new one is better (PSNR / SSIM) | | {100*bh['win_psnr']:.0f}% / {100*bh['win_ssim']:.0f}% |
| hand-over phase PSNR | {bh['phase']['handover']['v3_psnr']:.2f} | **{bh['phase']['handover']['mgw_psnr']:.2f}** |

**Label–image agreement on this dataset's frames.** Agreement is the IoU of the moved arm/can in the image with
where the labels put it (every 3rd frame of all {cons["n_episodes"]} episodes): {cons['v3_placed']:.3f} with the old renderer, **{cons['mgw_placed']:.3f}** with the new
one. The new renderer is better in {100*cons['mgw_better_frac']:.0f}% of episodes.

**Trade-off.** Slightly more residue is left where a moved part used to be: {cons['mgw_ghost']:.3f} vs {cons['v3_ghost']:.3f} of that area still
differs from the background. These are leftover fragments the robot masks missed. They are small and stay at the
old position rather than moving with the arm.

**Remaining artifacts.**
- slight edge speckle on fast-moving thin parts;
- a faint ring or shadow where the can stood (the clean plate is imperfect there);
- some blur where newly revealed surfaces are inpainted;
- arms that leave the camera frame look cut at the image border, which is correct.

**Camera calibration.** We tried refitting the camera and joint offsets to the robot masks. It did not help
(held-out IoU +0.006; a joint-offset fit made it worse), because the masks themselves are the limit. The inherited
camera is kept (`render_v4/calib_result.json`).

## 3. Can positions: 10 sources vs this dataset

- **Initial can positions:** x × y range {R['source'][0]} × {R['source'][1]} cm in the sources, {R['source+synthetic_v3'][0]} × {R['source+synthetic_v3'][1]} cm with this dataset.
- **Convex hull area:** {hm['start_convex_hull_cm2']['source']:.0f} → {hm['start_convex_hull_cm2']['source+synthetic_v3']:.0f} cm². The previous 20-episode revision reached {hm['start_convex_hull_cm2']['source+prev_v02']:.0f}.
- **New positions:** {hm['distinct_start_positions_gt_1cm_from_any_source']['synthetic_v3']} of {len(eps)} synthetic starts are more than 1 cm from every source start (previous revision: {hm['distinct_start_positions_gt_1cm_from_any_source']['prev_v02']}).
- **Coverage of the other 40 real recordings** (pool 10–49, back-projected human lid clicks), measured as distance to the nearest training start:
  - median: sources only {P['source_only']['median']:.1f} cm; with the previous revision {P['source+prev_v02']['median']:.1f} cm; with this dataset {P['source+synthetic_v3']['median']:.1f} cm;
  - within 2 cm: {W2['source_only']}, {W2['source+prev_v02']} and {W2['source+synthetic_v3']} of {W2['of']} recordings respectively.
- **Hand-over positions are not varied.** The hand-over is always the recorded one.

![can positions in the camera](assets/can_heatmap_bodycam.png)

## 3b. Target position: the hand-over point

**Before this revision the hand-over point did not vary at all.**
- Every synthetic episode handed the can over exactly where its source did: 0.0 mm difference, the same 10 points.
- The hand-over point is measured as the right-hand position at the left-gripper close command (forward kinematics of the published joint states). It is computed the same way for the 40 other real recordings of `openarm_pringles_lerobot_real_v00`.

**What the real data shows.** Operators hand over in a small region: the 40 other recordings span {tm['other_real_hull_cm2']:.0f} cm², and the 10 sources already cover {tm['source']['other_real_within_2cm']} of 40 within 2 cm.

**What this revision adds:** {nh} hand-over-location episodes.
- **Generator.** These are the frozen v3.1 generator's grasp-relative SE(3) coordination (condition C, in the spirit of DexMimicGen's coordinated subtasks, [arXiv 2410.24185](https://arxiv.org/abs/2410.24185)).
- **Transform.** The hand-over is moved by a world translation of 3–5 cm, e.g. `pos_left_down` = (0, +4, −3) cm.
- **Giver.** The giving hand carries the can there.
- **Receiver.** The receiving hand is planned in the can's frame, so the recorded grasp-to-grasp relation is preserved at every frame of the transfer.
- **Afterwards.** Both arms blend back to the recording after the release.

**Gates.** Each episode passes the same replay checks. It also passes a render gate that was fixed before rendering:
- label–image agreement, per episode and inside the hand-over window, ≥ the 10th percentile of the 55 published episodes (episode {g['all_placed']:.3f}, window {g['handover_placed']:.3f});
- ghost residue ≤ their 90th percentile (episode {g['all_ghost']:.3f}, window {g['handover_ghost']:.3f}).

{len(rj)} candidates were rejected: {"; ".join(f"source {e['E']} `{e['key'].split('__')[0]}` ({', '.join(e['reject'])})" for e in rj)}.

| hand-over point | 10 sources | + previous revision | + this revision |
|---|---:|---:|---:|
| convex hull (cm²) | {tm['source']['hull_cm2']:.0f} | {tm['synthetic_v3']['hull_cm2']:.0f} | **{tm['synthetic_v4']['hull_cm2']:.0f}** |
| other real recordings within 2 cm (of 40) | {tm['source']['other_real_within_2cm']} | {tm['synthetic_v3']['other_real_within_2cm']} | **{tm['synthetic_v4']['other_real_within_2cm']}** |
| median distance of other real recordings to nearest training hand-over (cm) | {tm['source']['other_real_nn_median_cm']:.2f} | {tm['synthetic_v3']['other_real_nn_median_cm']:.2f} | **{tm['synthetic_v4']['other_real_nn_median_cm']:.2f}** |
| synthetic hand-over points > 1 cm from every source | – | {tm['synthetic_v3']['new_targets_gt_1cm_from_every_source']} | **{tm['synthetic_v4']['new_targets_gt_1cm_from_every_source']}** |

![hand-over points](assets/target_pringles_handover.png)

**Visual quality of these episodes is lower in the hand-over window.**
- **What looks wrong.** Both arms are re-posed there, and the right gripper near the image edge shows visible fragments in several episodes (example below; top rows real, bottom rows synthetic).
- **Not a calibration problem.** We refitted the body camera by render-and-compare; on 7 held-out episodes it changed PSNR by {_d:+.3f} dB, so the camera is kept.
- **Why it happens.** Re-posing the near-camera gripper by ~5 cm needs surfaces the camera never saw.
- **Filtering.** Use `condition == "C"` in `meta/sfa_provenance.json` (episodes {ho_eps}) to drop or down-weight them.

![hand-over-location examples](assets/handover_location_examples.jpg)

## Wrist cameras

No wrist view is synthesised. Every wrist frame is the real wrist frame of the source recording at the generator's
time map, with no black frames.
- **Exact frames.** `wrist_view_valid = 1` marks frames where the whole scene equals that source frame ({100*ver['wrist_exact_fraction']:.0f}%).
- **Pose offset.** `wrist_view_pose_offset` (left translation m, left rotation rad, right translation m, right rotation rad) gives how far each synthetic wrist is from the copied view.
- **Position variants.** The left wrist is exact. The right wrist is shifted by up to the can shift (3–6 cm, no rotation) before the grasp; the can and gripper look right, and the background is offset.

Neither feature has the `observation.` prefix, so LeRobot does not feed them to policies.

## Limitations

- **Synthetic state.** `observation.state` is a planned kinematic candidate, not a measurement.
- **Not independent samples.** Episodes are variants of {len({e['source_real_episode'] for e in eps})} real demonstrations. The hand-over is the recorded one except in the {nh} hand-over-location episodes, where it is rigidly moved, not re-demonstrated.
- **Hand-over-location artifacts.** Those {nh} episodes have visible right-gripper fragmentation during the transfer (section 3b).
- **No training evidence yet.** No model has been trained on this revision. The earlier v4B result (far grasp error −9.9% [6.1, 13.6] for real_10 + 25 synthetic, body camera only) used the previous renderer and a different episode set.
- **Not replayed.** No episode has been replayed on hardware.

## Episodes

| ep | source | variation | frames | can start (x, y) cm | can shift from source (cm) | hand-over shift (mm) | max speed / real max | min arm distance (mm) | exact wrist frames | max right-wrist offset |
|---:|---:|---|---:|---|---|---:|---:|---:|---:|---|
""" + "\n".join(rows) + "\n"
(HERE / "README.md").write_text(card)
print("ok", len(rows))
