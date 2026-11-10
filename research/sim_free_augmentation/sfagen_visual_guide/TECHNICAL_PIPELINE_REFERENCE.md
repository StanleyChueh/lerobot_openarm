# sfagen: technical pipeline reference

This document describes the exact algorithms, equations, code functions, coordinate frames and dependencies of the current pipeline.

- **Code.** `research/sim_free_augmentation/sfagen/sfagen/` at commit **50832b3**. The working copy that generated the datasets (`~/Stanley_ws/sfa_hf_export/sfagen/sfagen/`) is byte-identical (`diff -rq` is empty).
- **Links.** Links are relative to this folder. Line numbers refer to 50832b3.
- **Scope.** Everything here describes the published datasets:
  - `ethanCSL/openarm_pringles_lerobot_real_to_sim_10_3cam_v00` @ `30533e67`;
  - `ethanCSL/openarm_insertbattery_lerobot_real_to_sim_10_3cam_v00` @ `c3c571d0`.
- **Not here.** The historical v1/v2 method is described only in §9.

## 0. Dependencies and fixed settings

| item | value |
|---|---|
| Python env | `~/Stanley_ws/lerobot_openarm/.venv` (Python 3.12) |
| torch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| transformers (SAM2) | 5.5.4, `Sam2VideoModel` + `Sam2VideoProcessor` |
| SAM2 weights | `facebook/sam2.1-hiera-small`, HF revision `ee5bba1d82bb8749febdf90f45e84b687142ba03`, run in **bfloat16** |
| kinematics | pinocchio 3.7.0 (FK, Jacobians, `log6`); coal (via pinocchio) for mesh distances |
| images | OpenCV 4.13.0 (Telea inpainting, morphology), PyAV (decode) |
| robot model | `model/openarm_description.urdf` + meshes (visual and collision) |
| GPU used | RTX 4090 (24 GB) |
| simulator / physics / generative model | **none** |

**Sources.**

| task | source dataset @ revision | episodes |
|---|---|---|
| Pringles | `ethanCSL/openarm_pringles_lerobot_real_10_v00` @ `cebdee98` | 0–9 |
| Battery | `ethanCSL/openarm_insertbattery_lerobot_real_10_v02` @ `a1c29d77` | 0–9 |

The battery `a1c29d77` is the repo's `main`. Its `v3.0` tag points at `92476e0e`, so the pinned revision is **not** the tag LeRobot would load by default.

## 1. Coordinate frames, units, conventions

- **World** = the URDF root frame of `openarm_description.urdf`. Units are metres and radians; z is up and the table is a horizontal plane `z = z_table`.
- **Robot state/action.** 16-D, interleaved: indices 0,2,…,12 are the right-arm joints J1–J7, 1,3,…,13 the left-arm joints, 14/15 the right/left gripper raw values.
  - Gripper openness: `g = clip((raw − closed_raw)/(open_raw − closed_raw), 0, 1)`, with right `open_raw = −1.298`, left `1.2121`, `closed_raw = 0`.
  - Code: [`Robot.split`](../sfagen/sfagen/robot.py#L94).
- **TCP.** URDF frames `openarm_right_hand_tcp` / `openarm_left_hand_tcp`, written `T_s(t) ∈ SE(3)` (world ← TCP). Computed by `pin.framesForwardKinematics`, see [`Robot.fk`](../sfagen/sfagen/robot.py#L113).
  - The finger-closing axis is the world axis of the first prismatic finger joint ([`finger_axis`](../sfagen/sfagen/robot.py#L137)).
- **Camera (body_cam).** Pinhole, no distortion model. See [`camera.py`](../sfagen/sfagen/camera.py).
  - `R`: the columns are the camera axes expressed in world (x right, y down, z forward = OpenCV convention). `t`: the camera centre in world.
  - Projection: `x_c = Rᵀ (P − t)`, `u = f_x x_c/z_c + c_x`, `v = f_y y_c/z_c + c_y` ([`project`](../sfagen/sfagen/camera.py#L26)).
  - Back-projection onto a plane `z = z0`: ray `r = R K⁻¹ [u, v, 1]ᵀ`, `P = t + λ r` with `λ = (z0 − t_z)/r_z` ([`backproject_plane`](../sfagen/sfagen/camera.py#L30)).
  - Initial guess (config): `f = 607.22 px`, `c = (326.4, 238.7)`, 640×480, position `(0.056, −0.003, 0.831) m`, `q_wxyz = (0.1872, −0.6824, 0.6954, −0.1249)`. The config notes these are the IsaacLab mount values; that is a geometric prior only, and no simulator is run.
  - Calibration perturbation `p7 = (ω ∈ ℝ³ camera-frame rotation vector, δ ∈ ℝ³ world translation, log s)`: `R' = R·exp(ω)`, `t' = t + δ`, `f' = s·f` ([`perturbed`](../sfagen/sfagen/camera.py#L36)).
- **Time.** Video frame `j` shows the robot at **state time** `j + lag` (fractional; joints linearly interpolated, [`interp_rows`](../sfagen/sfagen/recon.py#L24)).
  - **Command lead `k`** = `argmin_k mean|A[t] − S[t+k]|`, k ≤ 6 ([`action_lead`](../sfagen/sfagen/plan.py#L31)). It is 2 frames for both representative pairs.
- **Object poses** are `world ← object` SE(3) matrices. The proxy local frame is:
  - vertical cylinder: base centre at the origin, axis +z;
  - horizontal cylinder: one end cap at the origin, axis local +z = horizontal;
  - box: centred in x/y, base at z = 0.

## 2. Once per rig

### 2.1 Camera calibration: render-and-compare
Code: [`calibrate.run`](../sfagen/sfagen/calibrate.py#L37).

**Frames.** Near-static frames are those with `max_joint |q̇| < 0.05 rad/s`. Fourteen are taken per episode; even-indexed source episodes are used for fitting and odd ones are held out.

**Foreground.** `FG_j = open3×3(max_c |F_j − median(F_{::3})| > 30)`, a temporal median that does not depend on the camera model.

**Robot silhouette.** The URDF visual meshes are area-sampled every 3 mm ([`mesh.Geometry`](../sfagen/sfagen/mesh.py#L16)) and z-buffered by a two-pass GPU point splat ([`mesh.splat`](../sfagen/sfagen/mesh.py#L36)). Only arm-owned geometries are scored.

**Objective.** The IoU is computed inside a 41×41-dilated region of the silhouette:

```
IoU_j(p7) = |FG_j ∩ ROI ∩ Sil_j(p7)| / |(FG_j ∩ ROI) ∪ Sil_j(p7)|
max_p7  mean_j∈fit IoU_j
```

**Optimizer.** Powell on `p7 / scale` (scale 0.02 for every component); `maxfev 900`, `xtol 1e-2`, `ftol 1e-5`. An optional coarse random search (±15°, ±15 cm, 400 samples) is for new rigs and was not used here.

**Adoption rule.** The fit is adopted if the held-out IoU improves by more than 0.005.

**Results** (`configs/work/<task>/calibration.json`):

| task | fit frames / held-out frames | held-out IoU (initial → fitted) | correction |
|---|---|---|---|
| Pringles | 23 / 42 | 0.307 → 0.361 | 3.18°, 32.5 mm, f ×1.020 |
| Battery | 70 / 70 | 0.248 → 0.299 | 1.59°, 22.9 mm, f ×1.004 |

The IoU is low in absolute terms because the foreground includes cables, shadows and objects. **It is not a metric accuracy.**

### 2.2 Table height
Code: [`Context.table_height`](../sfagen/sfagen/context.py), [`Recon._fit_table`/`_fit_cylinder`](../sfagen/sfagen/recon.py#L172). Results are in `table_height.json`.

- **Known-size cylinder available.** The cylinder is fitted jointly with z on frame 0 of every episode (see §3.4). If the median IoU is ≥ 0.5, `z_table` is the median z.
  - Pringles: z = 0.1997 m, per-episode IoU 0.58–0.84, spread 0.178–0.213 m.
- **Otherwise.** `z_table = min over source demos of the lowest arm point − 5 mm`.
  - Battery: IoU 0.10–0.21, so z = 0.2542 m.

## 3. Per-source-episode reconstruction ([`Recon.__init__`](../sfagen/sfagen/recon.py#L51))

Order of operations, all cached in `cache/recon_ep<E>.npz` and `cache/sam2_ep<E>.npz`:

1. **Default lag** −1.75.
2. **Silhouette lag scan** over −4…+1.5 (step 0.25). A new lag replaces the default only if it is better by more than 0.01 IoU. That never happened for the episodes inspected, because the scan curves are flat.
3. **Plate pass 1** (robot only).
4. **Table and objects.**
5. **Holders.**
6. **Plate pass 2** (robot and objects).
7. **SAM2 robot masks.** Computed **at the lag current at that time**.
8. **Photometric lag.** If it differs, the plate and masks are recomputed, but **SAM2 is not re-run**: its cache is reused.

### 3.1 Clean plate ([`_plate`](../sfagen/sfagen/recon.py#L121))
- **Per pixel:** the temporal median over frames where the pixel is outside the 15-px-dilated robot mesh silhouette and outside the 31-px-dilated object proxies (pass 2).
- **Never-free pixels** (Pringles ep0 3.6%, battery ep1 1.6%) are filled by Telea inpainting (radius 7).

### 3.2 Robot masks: SAM2 ([`segment.segment_arms`](../sfagen/sfagen/segment.py#L26), [`arm_prompt`](../sfagen/sfagen/segment.py#L12))
- **Camera.** body_cam only; wrist cameras are never segmented.
- **Prompt.** For each arm, a **mask prompt** = the union of the convex hulls of the projected visual geometries of that arm, excluding `link0..link3`, minus a 25-px-dilated neighbourhood of every object proxy.
  - It is issued every **15 frames** within independent **150-frame chunks**; one SAM2 session per chunk, propagated forward from the first prompt.
  - Prompts below 300 px are skipped.
  - Counts: Pringles ep0 38 prompts / 2 chunks; battery ep1 104 prompts / 6 chunks.
- **Overlap.** Pixels claimed by both arms go to the first arm (R).
- **After object masking.** Arm masks lose the pixels of the object masks (§3.6).
- **Lag used for the prompts.** −1.75 for both representative episodes, not the final photometric lag. A re-run at −1.75 reproduces the cache with IoU 1.000 on every frame (`diagnostics/run_seg_*_sam2rerun.json`).

### 3.3 Objects: "what disappears when picked up" ([`_segment_object`](../sfagen/sfagen/recon.py#L201), [`_after_ref`](../sfagen/sfagen/recon.py#L260))
1. **After-grasp reference.** `ref = per-pixel median over frames ≥ grasp + 20` (step 2) of robot-free pixels.
2. **Difference mask.** `seg = open3×3(|F_0 − ref| > 30) ∩ valid ∖ robot_sil(0)`, minus earlier objects. Keep the connected components of ≥ 40 px that touch a 45-px circle around the projected grasp TCP, and lie inside a 130-px circle.
3. **Unknown-size objects (station).** Back-project the segment to the plane `z_table + 0.75 h` and take the PCA axes. The extents are the 3rd–97th percentiles; the height comes from the grasp height, clipped to [0.02, 0.08] m.
   - Battery ep1 station: 0.126 × 0.062 × 0.063 m.
- **SAM2 is NOT used for objects.**

### 3.4 Known-size cylinders: grasp-constrained render-and-compare ([`_fit_cylinder`](../sfagen/sfagen/recon.py#L271))

Let `a` be the horizontal approach axis (TCP z with its component along `b` removed) and `b` the horizontal finger-closing axis. Then:

```
centre_xy = TCP_xy(t_grasp) + s·a + l·b,   s ∈ [−8, 8] cm, l ∈ [−4, 4] cm
```

- **Search.** Coarse 1 cm grid, then fine 2.5 / 5 mm, with z fixed to the rig table.
- **Score.** IoU of the projected proxy silhouette with `seg` inside 160 px of the grasp point.
- **Orientation.** A vertical cylinder (can) has a vertical axis. A horizontal cylinder (battery) has its axis horizontal and perpendicular to `b`.
- **Results.** Pringles ep0: IoU 0.777, centre 45.3 mm from the TCP. Battery ep1: IoU 0.093 (weak).

### 3.5 Holders and attachment ([`_finish_objects`](../sfagen/sfagen/recon.py#L317), [`obj_pose`](../sfagen/sfagen/recon.py#L340))
- **Segments.** `table [0, grasp+3)` → `hand A [grasp+3, open_A)` → either another hand B or `placed`.
  - It goes to hand B if, when A opens, B is closed and its TCP is within 25 cm of the object; B then holds the object with `X_B = T_B⁻¹(t) · T_obj(t)`.
  - Otherwise the object is `placed` at its world pose when A opens.
- **Attachment while held.** `X = T_A(t_g+3)⁻¹ · T_obj^table` is constant, and `T_obj(t) = T_A(t)·X`. This is a rigid-grasp assumption: no slip is modelled.
- **Battery note.** After the right gripper releases the battery inside the station (frame ≈ 434), the battery is "handed over" to the closed left hand, which holds the station. Physically the battery stays in the station; geometrically that is equivalent.

### 3.6 Object masks ([`_masks`](../sfagen/sfagen/recon.py#L367))
`obj_k(j) = (21-px-dilated projected proxy at its pose) ∩ (|F_j − plate| > 25) ∖ (mesh robot ∪ SAM2 robot)`.

### 3.7 Photometric video/state lag ([`_photometric_lag`](../sfagen/sfagen/recon.py#L149))
- **Pairs.** 16 high-motion pairs `(j, j+6)`, with `j` on a 7-frame grid offset from the benchmark grid.
- **Score.** For each candidate lag in −3…+0.5 (step 0.25), re-render real frame `j+6` from real frame `j` with the §5 renderer (`part_fill` off; borrowing excluded within ±5 frames of the target). Score the mean PSNR inside the camera-independent foreground of both frames. Choose the argmax.
- **Results.** Pringles ep0 −0.50 (PSNR range 1.58 dB, while the silhouette scan range is only 0.016 IoU). Battery ep1 −0.75 (1.22 dB vs 0.005).

## 4. New trajectories (kinematics only) ([`plan.py`](../sfagen/sfagen/plan.py))

Let `d ∈ ℝ³` be the world shift and `w(t) ∈ [0, 1]` the weight; `smoothstep(x) = x²(3 − 2x)`.

**F1, object position** ([`f1_schedule`/`f1_plan`](../sfagen/sfagen/plan.py#L113)). Only the arm that grasps the moved object is edited.
- **Motion onset** `t0` = first frame before `grasp − 15` where TCP speed > 2 cm/s.
- **Weight:** `w = smoothstep((t − t0)/(t_g − 15 − t0))` until `t_g − 15`; `1` on `[t_g − 15, t_g + 10]`; `1 − smoothstep((t − t_g − 10)/(t_r − t_g − 10))` until `t_r`; `0` after.
- **Rejoin** `t_r` = the lift apex, i.e. the argmax of TCP z on `[t_g + 30, min(t_g + 4 s, next_event − 10))`.
- The resting object is translated by `d`; held objects follow the new TCP with the recorded attachment.

**F2, interaction location** ([`f2_schedule`/`f2_plan`](../sfagen/sfagen/plan.py#L142)). All arms are edited.
- **Window:** the first to last frame with `min ‖TCP_a − TCP_b‖ < 0.20 m`. It is clipped to start after every table grasp + 10 + ramp and to end before any put-down − 10 − ramp. Ramps are 45 frames.
- **Weight:** `w = min(smoothstep((t − t_in)/45), 1 − smoothstep((t − t_end)/45))`.
- Every hand, and therefore every held object, moves by `w·d`, so relative hand–hand and hand–object poses are unchanged.

**Desired TCP** (orientation unchanged): `T*_s(t) = [R_s(t), p_s(t) + w(t)·d]`.

**IK** ([`ik`](../sfagen/sfagen/plan.py#L36)) is damped least squares in pinocchio's LOCAL frame, with the error `e = log6(T_s(q)⁻¹ T*)`:

```
q ← clip(q + Jᵀ (J Jᵀ + λ I₆)⁻¹ e,  lo, hi),   λ = 1e-3, ≤ 100 iterations
stop when ‖e_lin‖ < 1e-4 m and ‖e_ang‖ < 1e-3 rad
lo/hi = URDF limits ∪ the recorded joint values at that frame
```

**Pipeline per arm** ([`_retarget`](../sfagen/sfagen/plan.py#L49)):
1. **Warm start.** `q₀(t) = q_new(t−1) + (q_rec(t) − q_rec(t−1))`.
2. **Smoothing.** `Δq = GaussianFilter_σ=3(q_new − q_rec)·[w > 0]`.
3. **Contact refinement.** On frames with `w ≥ 0.999`, re-solve IK (30 iterations) starting from the smoothed solution.
4. **Re-verification.** Re-verify by FK and record `ik_err(t) = (‖Δp‖, ∠ΔR)`.

**Actions:** `a′(t) = a(t) + Δq(t + k)` with lead `k`, clipped to the URDF limits ∪ the recorded action range. Grippers and non-moved arms are bit-identical.

**Measured for the representative pairs.**

| pair | contact residual max | blending residual |
|---|---|---|
| Pringles ep0 `interact_x+5_y+0_z+0` | 0.094 mm / 0.032° | < 1 mm |
| Battery ep1 `battery_x+6_y+0` | 0.093 mm / 0.045° | — |

- **Ceiling.** Every passing candidate has a residual ≤ 0.1 mm, which is the IK stopping tolerance.
- **Reproducibility.** A re-plan reproduces the published arrays bit-for-bit.
- **Cost.** About 0.04–0.05 s per candidate (CPU).

## 5. Replay checks ([`Checker.check`](../sfagen/sfagen/checks.py#L68))

All checks are computed on the **commanded action stream** and the planned state. Defaults are in `checks.DEFAULTS`.

| check | rule |
|---|---|
| limits | all joints within URDF ∪ the recorded range of that demo (state and action) |
| speed / accel | per joint `max|Δa|·fps ≤ 1.25 ×` and `max|Δ²a|·fps² ≤ 1.5 ×` the max over all source demos |
| clearance | coal min distance between collision meshes of different arms and arm–base (link0/1 pairs skipped), sampled every 3 frames on edited frames, `≥ min(source on the same frames, 20 mm) − 1 mm` |
| hands | min TCP–TCP distance ≥ source − 5 mm |
| table | lowest arm point (excluding link0–2) ≥ lowest over all source demos − 1 cm |
| start / untouched / rejoin | start state within 0.01 rad; non-moved arms and grippers bit-identical; state identical after the rejoin |
| IK | ≤ 1 mm / 0.5° where `w ≥ 0.999`; ≤ 5 mm / 2° elsewhere |
| objects | proxy lowest point ≥ source − 5 mm. F1 placement: in view (10-px margin), plate HSV saturation/value at the new spot within 30 of a ring around the old spot, ≥ 1 cm from other objects, and no extra robot hull points within 1 cm of the moved object before `grasp − 15` |

**Dataset-wide pass rates and failure reasons** (`run_traj_*.json`):

| task | passing | most common failure reasons |
|---|---|---|
| Pringles | 96/160 | robot-near 31, accel 30, speed 16, hands 15, … |
| Battery | 114/240 | robot-near 53, hands 43, accel 41, speed 35, clearance 23, … |

**Gap in the clearance check.** For **88 battery candidates** the source's own minimum mesh distance on the edited frames is **0 mm**: the collision meshes touch during insertion. The clearance rule therefore cannot reject them.

**What the checks are not.** They are kinematic/geometric consistency checks on a URDF model. They are **not** a physics check (no dynamics, contact forces, friction or grasp stability) and **not** a hardware replay.

## 6. Body-camera rendering: mesh-guided forward warping ([`Renderer.render`](../sfagen/sfagen/render.py#L130))

Output frame `i` uses source frame `j = i` and the target state at time `i + lag`. Frames whose state and objects equal the recording are **copied** ([`render_stage.unchanged_rows`](../sfagen/sfagen/render_stage.py#L9)).

**Source state.** For the source frame, z-buffer the URDF points (1.5 mm spacing) plus the object proxy points ([`source_state`](../sfagen/sfagen/render.py#L66)), giving per-pixel part, depth and back-projected 3-D point `P`.
- **Robust masks** ([`robust_masks`](../sfagen/sfagen/render.py#L92)): the plate difference (> 28), assigned to the nearest category within 12 px; the mesh silhouette where the plate difference is > 12; then SAM2 arm and object masks override.

Per output frame:

1. **Moved parts.** The parts whose projection moves ≥ 0.5 px.
2. **Removal.** The pixels of moved parts (7-px dilated, never over static parts) are replaced by the **relit plate** ([`adapted_plate`](../sfagen/sfagen/render.py#L118)): `plate′ = (plate + 8)·exp(G_σ=10 * (w·L) / G_σ=10 * w) − 8`, where `L = clip(log(F_j + 8) − log(plate + 8), ±1)` is computed on background pixels only (`w`).
3. **Forward 3-D warp.** For each real pixel of a moved part: `P′ = M_target · M_source⁻¹ · P` (that part's rigid motion). Project, splat to 4 neighbours, and z-test against the front-most warped point and against static parts (5 mm margin). Close pinholes with a 5×5 normalised average.
4. **New surfaces.** Pixels inside the eroded (21×21) target silhouette that the warp did not reach are looked up as the same 3-D mesh point in other frames `j ± 3k`, k ≤ 15 ([`lookup`](../sfagen/sfagen/render.py#L106)). A lookup counts only if that point is visible there: depth within 15 mm, same label, same category in a 5-neighbourhood.
5. **Holes.** Remaining holes get Telea inpainting (radius 3). `part_fill` is **off** in the published config.
6. **Clean-up.** 3×3 median despeckle where `|img − median| > 40` inside the warped/inpainted region; 1-px Gaussian (σ 0.7) anti-aliasing on its boundary.

**Diagnostic replica.** The line-by-line copy in `scripts/diag_render.py` reproduces the published frames **pixel-identically** (6/6 frames).

**Benchmark** (existing; `sfagen/results`, `logs/xbench_*_final.json`): real-ground-truth pose transfer, i.e. re-render real frame `j + k` (k = 4/8/15) from real frame `j` with no borrowing within ±5 of the target.
- **Scoring region:** inside the foreground of both frames vs the temporal median, which is a pipeline-independent region.

| task | held-out episodes | pairs | sfagen | copy-source baseline | previous renderer | sfagen − previous |
|---|---|---|---|---|---|---|
| Pringles | 0, 2, 3, 4, 6, 7, 9 | 211 | 18.04 dB / SSIM 0.703 | 14.51 dB | 17.87 dB | +0.17 dB [−0.06, 0.41] |
| Battery | 3, 5, 8, 9 | 284 | 19.21 dB / 0.709 | 16.28 dB | 18.40 dB | +0.82 [0.64, 1.00] |

**Scope:** this measures re-rendering a *recorded* pose from another recorded frame. It does **not** measure fidelity at the new, unrecorded target poses.

## 7. Wrist cameras ([`build.run`](../sfagen/sfagen/build.py#L21))

`wrist_cam` (left) and `right_wrist_cam` are the **recorded real frames** at the same time index. They are not re-rendered.
- `wrist_view_valid[t] = 1` iff the whole scene at `t` equals the recording. The flag is shared by both wrists.
- `wrist_view_pose_offset = (‖Δp_L‖ m, ∠ΔR_L rad, ‖Δp_R‖, ∠ΔR_R)`: the synthetic vs recorded TCP pose difference.

## 8. Export and verification ([`build.py`](../sfagen/sfagen/build.py), [`verify.py`](../sfagen/sfagen/verify.py))

**Export.** LeRobot v3.0, `streaming_encoding=True`, AV1.

**`verify` checks:**
- the dataset loads;
- label SHA-256 equals the plan;
- 0 episodes are identical to a real episode;
- **total** decoded frames per camera;
- no black frames;
- body vs render PSNR.

**Gap:** `verify` does not check per-episode video spans. That is how the battery `wrist_cam` defect (episode 91 lost one frame) was reported only as 84,689 ≠ 84,690; see `diagnostics/battery_wrist_cam_frame_deficit.md`.

## 9. Historical method (v1 `e8abc26`, v2 `b9b7dfe`/`03ba314`): what differs

| component | v1/v2 (frozen; Pringles only, body_cam only, ep 0/4/8) | sfagen (current) |
|---|---|---|
| camera | inherited extrinsic "fitted in IsaacLab" (silhouette chamfer), ~20–40 px off; optional per-episode refinement | per-rig render-and-compare calibration with held-out frames (§2.1) |
| landmark / object track | **Gate 1 HSV lid tracker** (`contact_aware_augmentation/gate1/lid_tracker.py`), initialised from **one human-verified click** on frame 0; per-frame ok/uncertain/lost | no tracker, no click: object = "what disappears when picked up" + grasp-constrained proxy fit |
| object 3-D | lid point fitted in the hand frames from the lid track + hand motion; v2: joint closure-constrained fit across both hands | proxy pose at rest + rigid attachment to the grasping TCP |
| SAM2 | arms (mask prompts link4…fingers) **and can** (point prompts from the lid tracker) | arms only (mask prompts); objects by plate difference |
| optical flow / keypoints | **RAFT-large** (torchvision `C_T_SKHT_V2`, 12 updates) chained frame-to-frame with forward-backward check (2 px), used for **evaluation metrics only**: temporal warp error and keypoint following (whose targets come from the renderer's own homographies, i.e. circular) | **not used anywhere** (`grep -i raft sfagen/` is empty) |
| query points | 8 random pixels per hand/link7/finger region (projected URDF label map) and 16 on the can at `ref = right_close + 2/3 (left_close − right_close)` | — |
| trajectories | G / P / C hand-over generators (shared targets; phase profiles; grasp-relative SE(3) receiver; regrasp and timing variations) | F1 object position, F2 interaction location (both tasks) |
| renderer | **layered 2-D homography warping** of SAM2 layers, far-to-near compositing, feathered alpha, Telea plate | per-pixel **3-D forward warping** with z-buffer, hidden-surface borrowing, relit plate |
| outputs | local LeRobot prototypes (G 24, P 24, C 30 episodes) | 4 published Hub datasets (+ 9 source-budget datasets) |

The RAFT replay in `stage_02_tracking/` uses the frozen `sfa/track.py` unchanged.
