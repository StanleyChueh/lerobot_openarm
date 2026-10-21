# Milestone 2A: algorithmic wrist-camera calibration — result: **BLOCK**

**Date:** 2026-10-19 · **Scope:** what the user approved for M2A (bounded headless Isaac render, SAM 2.1 segmentation, fitting). No scene/cfg edits, no downloads, no robot, no commits.
**Decision:** wrist-camera extrinsics are **not identifiable** from `openarm_pringles_lerobot_real_v00` by robot-silhouette fitting. Two pre-registered PASS gates failed *before* any fit (mask quality, informative data), so no optimized transform was produced and **no camera patch is proposed**. Per the stop condition, a ChArUco capture protocol is given in §7. No manual GUI fallback was used.

## 1. Preserved M1 evidence
`../m1_evidence/` holds the M1 comparison images, scripts and json. `../m1_evidence/MANIFEST.sha256` lists the sha256 of the dataset parquet/video inputs and of every copied file, plus the scratchpad source path.

## 2. Frozen split and acceptance criteria (written before fitting)
- **`split.json`**, episode-level:
  - camera-cal **train** = real eps 11–34
  - camera-cal **val** = 35–49
  - **excluded** (`sim_v01_reallook_physfit` replay sources) = 0, 2, 3, 4, 5, 6, 9, 10
  - **reserve** = 1, 7, 8
  - Contact-fit membership is labelled: train 0–34 ⊇ cam-train; test 35–49 = cam-val. No episode contributes frames to two sets.
- **`acceptance_criteria.json`**: per camera, all of data, mask quality, held-out improvement, multi-start stability, bootstrap spread and identifiability must pass; otherwise BLOCK.

## 3. Inputs produced (bounded GPU run)
| Step | Command (abbrev.; full in §8) | Result | Cost |
|---|---|---|---|
| Frames | `tools/extract_split_frames.py --per_episode 12 --max_speed 0.25` | 235 frames per wrist camera (163 train / 72 val) | ~8 s CPU each |
| Render | `render_real_states.py --camera {wrist_cam,right_wrist_cam} --surface_points 600000` (task -v0, handover, cfg mount) | poses.npz + sim images | 228 s / 170 s wall, ~8 GB RSS, Isaac Sim 5.1.0 |
| Segmentation | `segment_robot_real.py --max_robot_box 0.8 --max_object_box 0.8` (GDINO-base + SAM 2.1-large, cached) | robot_masks.npz | 126 s / 109 s, ~2.5 GB RSS |
| Diagnostics | `tools/diagnose_wrist.py`, `tools/visibility_all_frames.py` | `evidence/diag_*.json`, `evidence/visibility_all_frames_*.json` | < 2 min CPU each |

## 4. Diagnosis

| Item | Finding | Status |
|---|---|---|
| **Intrinsics / principal point / distortion** | Live `rs-enumerate-devices -c` matches the values in the code. Left wrist (ASIC 254343065853 → `/dev/rs_wrist_left`): fx 605.44, fy 605.13, cx 322.85, cy 246.44. Right (254343060572): 607.79 / 607.61 / 322.58 / 248.52. Distortion is Inverse Brown-Conrady with all coefficients 0. Sim lens: fx = 640·24/aperture = 605.43 / 607.79, principal point forced to the centre (renderer limitation), so the real cy offset of 6.4–8.5 px (≈0.6–0.8°) can only be absorbed by the mount. | VERIFIED |
| **URDF hand-eye** | `v1_camera.urdf`: `openarm_{l,r}_camera_joint` = link7 + (0, 0, 0.06), rpy 0; ee_tcp ≡ link7. Camera-housing collision box centre is camera_link + (0.0657, 0, 0.03) → link7 (0.066, 0, 0.09). The cfg uses pos (0.06, 0.001, 0.11125) with a GUI-tuned rotation, identical for both arms. **The cfg comment says the position "is that +6 cm translation"; it is not.** The cfg position sits ~2 cm above the URDF housing centre and has no recorded source. The left camera visual mesh is mirrored in y (`scale 0.001 -0.001 0.001`) and the right is not, yet the cfg rotation is identical for both. | VERIFIED (files); true mount UNKNOWN |
| **Forward kinematics** | 43 of 235 frames per camera "settled" 0.010–0.087 rad from the real state (render log warnings; likely joint limits). A battery-task note reports a cm-level real-vs-URDF hand offset. Both bias any FK-based camera fit. | INFERRED (not quantified) |
| **Isaac transform composition** | The renderer composes parent ∘ cfg offset (ROS optical) as expected: the composed projection of the hand mesh matches the rendered depth to **3.2 mm** median (4 of 5 checked frames). **The `cam_pos_w` / `cam_quat_w_ros` saved in `poses.npz` are stale** (≈450 mm / 90–108° off the composed pose in every frame), so no tool should use them. `calibrate_camera.py` does not; this fit tool does not either. | VERIFIED |
| **Left/right mounting** | Same cfg offset on both ee_tcp frames. Real images differ between arms: left shows its brown foam fingertip at the left edge mid-height, right shows it at the right edge. In sim both fingertips sit at the bottom corners. Fingertip position cannot be used to infer rotation error, because the real foam pads extend the fingers by ~7 cm (unmodelled). | INFERRED |
| **Robot visibility (identifiability)** | Over **all** frames of the 39 split episodes (3,114 frames, every 3rd), the non-own robot covers ≥ 5 % of the image in **5** (left) / **30** (right) frames at the cfg mount; median 0 %. Over 12 random mounts (σ 10°, 3 cm), the median is 25 / 38 frames. When the other gripper is in view, it holds the can, which occludes it. The silhouette method has nothing to fit. | VERIFIED |
| **Segmentation quality** | On 16 val frames per camera (`evidence/mask_quality_val_*.jpg`), the "robot arm / robot gripper" masks are mostly the **black vacuum hose** behind the robot, plus the right camera's own foam fingertip. GDINO labels the black box as "pringles can". Across 137 / 121 frames with a real mask, **100 % of mask pixels (median) lie > 40 px from any sim robot pixel**, and IoU with the sim projection is 0. | VERIFIED → **gate fails** |

**Gates (from `acceptance_criteria.json`):**
- **mask_quality: FAIL** (both cameras).
- **data: FAIL in substance.** The counter's ≥ 0.5 % "informative" frames are background, not robot.
- **Remaining gates:** not evaluated. A fit to hose masks would give a well-converged but meaningless transform. I deliberately did not run one, so no misleading "after" numbers exist.

Other cues are also unusable without circularity:
- **Fingertips:** foam pads not modelled.
- **Pad edge:** pad height and position are HAND-set (M1 G3).
- **Room backdrop:** the -v1 backdrop was built *through* the current mount.
- **Can:** pose in hand is unknown.

The earlier data-only attempts are consistent with this: AX=XB disagreed by 17.7 vs 9.5 cm between identical cameras, and COLMAP SfM failed.

**Before-state evidence:** `evidence/before_cfg_val_*.jpg` shows held-out frames as real | sim at the cfg mount | blend. At rest, the sim shows the pad edge across the lower third where the real camera sees wall and floor. Either the mount pitch or the pad geometry is wrong; these data cannot tell which.

## 5. Proposed transforms / cfg patch
**None.** The current cfg wrist mounts stay as they are (still HAND, unvalidated). After §7 the patch takes this form, per camera, in `stack_ik_abs_visuomotor_env_cfg.py`:
```python
offset=CameraCfg.OffsetCfg(pos=(<x>, <y>, <z>), rot=(<w>, <x>, <y>, <z>), convention="ros")  # from charuco_handeye.json, parent openarm_{side}_ee_tcp
```

## 6. Remaining uncertainties
- True wrist mounts (both).
- Real-vs-URDF kinematic error (cm-level, unquantified).
- Fingertip pad geometry.
- Pad height and edge.
- Whether a single shared cfg rotation is even correct (the meshes are mirrored but the cfg is shared).

## 7. ChArUco capture protocol (≈ 30 min, operator-run; Claude not needed at runtime)
**Board:**
- ChArUco 7 × 5 squares, 40 mm squares, 30 mm markers, `DICT_5X5_100`, printed at 100 % on A3.
- Glue it flat to foam board.
- **Measure one square with calipers** and record it in a declared-input file.

**Fixed placement:**
- Lay the board flat on the pad about 0.35 m in front of the robot, near the hand-over area, and tape it down. It must not move for the whole capture.
- Check that the body camera also sees it: this lets the same data validate the body-cam fit and measure FK error.

**Poses, per arm** (teleop with Quest as usual):
- 20–25 **static** poses with the board fully in that arm's wrist view.
- Distance 0.25–0.6 m; tilt the view ≥ 30° about at least two axes across the set.
- Include roll variety.
- Hold each pose still ≥ 1 s. Gripper open and fixed.
- Static frames remove the state/video timestamp problem.

**Recording:** use the existing `lerobot-record` (3 cameras + joint states) to a new local dataset, e.g. `openarm_charuco_handeye_v00`. Run one episode per arm, then one extra episode with the board moved and rotated 90° (held out).

**Also measure (declared inputs, ~5 min, same session):**
- foam fingertip length and width (calipers)
- pad top height above the robot base mount
- pad near-edge distance
- Pringles can diameter, height and mass

**Analysis (to implement after approval; OpenCV + `urdf_fk.py` only):**
1. Per static frame: `cv2.aruco.CharucoDetector` → `solvePnP` with the factory intrinsics → T_cam←board.
2. FK T_base←link7 from the joint state.
3. Eye-in-hand AX=XB via `cv2.calibrateHandEye` (Tsai, Park, Horaud, Daniilidis) and `cv2.calibrateRobotWorldHandEye`. Fixed seed, no random restarts needed.
4. Validate on the held-out board placement: reprojection RMS of board corners predicted through FK ∘ X.
5. **PASS:** held-out reprojection RMS ≤ 2 px; methods agree within 5 mm / 0.5°; left and right each solved independently. The body-cam board pose vs the wrist-chain board pose gives the FK error.

## 8. Commands (fresh shell; GPU steps need `conda activate env_isaaclab` for Isaac)
```bash
R=~/.cache/huggingface/lerobot/ethanCSL/openarm_pringles_lerobot_real_v00; M=~/Stanley_ws/lerobot_openarm/research/task_aware_real2sim/m2a
L=~/Stanley_ws/IsaacLab/logs/real2sim/m2a; PY=~/Stanley_ws/lerobot_openarm/.venv/bin/python3
for c in wrist_cam right_wrist_cam; do
  $PY -I $M/tools/extract_split_frames.py $R $L/${c}_frames --camera $c --split $M/split.json --sets camera_cal_train camera_cal_val --per_episode 12 --max_speed 0.25
  (cd ~/Stanley_ws/IsaacLab && ./isaaclab.sh -p scripts/tools/sysid/render_real_states.py --enable_cameras --headless --camera $c \
      --frames $L/${c}_frames/frames.npz --out $L/${c}_render_cfg --surface_points 600000)
  (cd ~/Stanley_ws/IsaacLab && $PY scripts/tools/sysid/segment_robot_real.py $L/${c}_frames/frames.npz --max_robot_box 0.8 --max_object_box 0.8)
  $PY -I $M/tools/diagnose_wrist.py $L/${c}_render_cfg/poses.npz $L/${c}_frames/robot_masks.npz $L/${c}_frames/frames.npz \
      --intrinsics <fx>,<fx>,320,240 --out $M/evidence/diag_$c.json        # fx 605.43 (left) / 607.79 (right)
  $PY -I $M/tools/visibility_all_frames.py $L/${c}_render_cfg/poses.npz --camera $c --split $M/split.json --out $M/evidence/visibility_all_frames_$c.json
done
# fitter (implemented, NOT run because the gates failed): tools/fit_wrist_mount.py <poses> <masks> <frames> --camera <c> --intrinsics ... --seed 0
```

**Seeds:**
- frame selection deterministic
- surface sampling seed 0 (render script)
- visibility: point subsample seed 0, perturbation seed 1
- fitter default seed 0

**Hashes:** `MANIFEST.sha256`, covering repo SHAs, environment versions, URDF/cfg/script hashes, GDINO and SAM 2.1 safetensors, and every intermediate and output.

## 9. Summary
The algorithmic route is blocked by the data, not by tooling. The wrist cameras almost never see the robot except for their own foam-padded fingertips, and the segmenter labels the background hose as robot. The current wrist mounts therefore stay hand-set and unvalidated. A 30-minute static ChArUco capture with measured declared inputs makes the calibration a standard, reproducible OpenCV hand-eye problem with held-out validation.
