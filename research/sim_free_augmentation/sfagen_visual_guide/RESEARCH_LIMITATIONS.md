# Research limitations of the current pipeline (sfagen @ 50832b3)

This file lists confirmed weaknesses, quantities that were never measured, and an honest reading of the evidence.

- **Status:** documentation of the existing pipeline and published datasets.
- **Not changed:** no result, threshold or frozen study.
- **Not rewritten:** earlier negative results (v3/v3.1/v4B/v4B.1/SD10) stay as recorded in the Experiment Ledger.

## 1. Confirmed weaknesses (with evidence in this folder)

| # | weakness | evidence |
|---|---|---|
| 1 | **Wrist cameras are not synthetic.** Both wrist videos are the recorded real frames. In Pringles ep 6, 147 of 273 frames show an arm pose that the wrist image does not reflect (wrist moved by up to 5.0 cm); in battery ep 14, 307 of 776 frames (up to 6.0 cm). A VLA trained on these sees wrist images that contradict the state. | `stage_06_cameras/*`, `diagnostics/run_render_*.json` |
| 2 | **One shared validity flag.** `wrist_view_valid` is one "scene unchanged" flag for both wrists. Battery ep 14's left wrist did not move (offset 0 cm) but is flagged invalid for 307 frames. | `stage_06_cameras/battery_wrist_flags.png` |
| 3 | **A published defect in the battery wrist video.** Synthetic episode 91 (mixed episode 101) lost source wrist frame 139 in the streaming encoder. Its left-wrist image then runs one frame ahead of state/action, and its final frame comes from episode 92. `sfagen verify` only compares totals, so the defect surfaced only as 84,689 ≠ 84,690. | `diagnostics/battery_wrist_cam_frame_deficit.md` |
| 4 | **Rendering artifacts at large near-camera shifts.** Re-posed grippers near the camera or the image edge fragment into smeared ghosts with dark Telea fill: up to 6,282 inpainted pixels per frame in battery and 3,760 in Pringles; 95th percentile 1,496 / 533. | `stage_05_rendering/*_failure_frames.jpg`, STAGE_TIMINGS row S4c |
| 5 | **Renderer fidelity is only measured at recorded poses.** The held-out benchmark (Pringles 18.04 dB, battery 19.21 dB) re-renders a recorded frame from another recorded frame. No real image exists at the new target positions, so fidelity there is **unmeasured**. | TECHNICAL §6 |
| 6 | **Segmentation chunk restarts.** SAM2 runs in independent 150-frame chunks, and the masks jump at the boundaries (battery left arm: frame-to-frame IoU 0.40 at frame 150; area 60k → 25k px). Masks also leak (battery frame 125) and lose fingers (frames 452/549). | `stage_01_segmentation/battery_segmentation_stability.png`, `_failures.jpg` |
| 7 | **SAM2 prompts use the pre-photometric time offset** (−1.75 instead of the final −0.5 / −0.75 frames). The cached masks were then reused. The effect on render quality was not isolated. | `diagnostics/run_seg_*_sam2rerun.json` (exact reproduction at −1.75) |
| 8 | **Coarse object proxies.** The battery proxy matches its image weakly (fit IoU 0.09; residual median 0.16). The station is a PCA box. The battery table height is a fallback rule, not a fit (fit IoU 0.10–0.21). Grasps are assumed rigid: no slip, no in-hand motion. | `stage_03_reconstruction/battery_object_lag_table.png` |
| 9 | **Camera calibration only from silhouettes.** The held-out IoU rises 0.31 → 0.36 and 0.25 → 0.30, but there is no metric ground truth (no checkerboard, no external measurement). No lens-distortion model. | `stage_03_reconstruction/*_calibration_before_after.jpg` |
| 10 | **Clearance check is vacuous where the source meshes touch.** For 88 battery candidates the source's own minimum mesh distance on the edited frames is 0 mm, so the "not below the source" rule cannot reject them. Collision checks use URDF collision meshes, not the real cables or objects. | `diagnostics/run_traj_battery.json` |
| 11 | **Kinematic plausibility ≠ physical feasibility.** The checks cover limits, speed/acceleration envelopes, mesh distances and IK residuals on a model. There is **no dynamics, no contact force, no grasp stability, no friction and no hardware replay**. Insertion success at a shifted location is assumed from the recorded relative motion. | TECHNICAL §5 |
| 12 | **Narrow variation.** Translations only (±3/±5/±6 cm) and two families. No rotations, no new grasp styles, no timing or speed changes, no new objects. Coverage gains are kinematic coverage at gripper events. | NOVELTY.md, coverage.json |
| 13 | **Battery source revision is not the `v3.0` tag.** sfagen read `real_10_v02` @ `a1c29d77` (`main`), while the repo's `v3.0` tag points at `92476e0e`. Anyone reloading the source with LeRobot's default revision gets a different revision; whether the content differs was not checked. | TECHNICAL §0 |
| 14 | **The recon docstring is out of date.** Its first line says "no SAM2", but the pipeline does use SAM2 for arm masks (`robot_masks: sam2`). | `sfagen/sfagen/recon.py` line 1 vs `_masks` |

## 2. Quantities that are NOT measured (do not claim)

- **Segmentation accuracy** (mIoU / Dice / boundary F). There are no manual masks; only agreement with the projected robot model and frame-to-frame stability are reported.
- **Tracking accuracy** of RAFT keypoints (historical) or the HSV lid tracker. The Gate 1 and Gate 1R manual annotations are `PENDING`.
- **Metric camera calibration error** (pixels on independent landmarks, or millimetres).
- **Object pose accuracy** (no ground-truth object poses).
- **True camera exposure time / synchronisation.** The photometric offset is the best re-rendering offset, not a measured exposure time.
- **Image fidelity at new target poses**, and perceptual realism (no user study, no learned realism score).
- **Physical executability** of synthetic trajectories (no hardware replay) and force/contact validity.
- **VLA benefit.** No policy trained on the published sfagen datasets has been evaluated. Only 5-step loading smoke tests exist.
- **Real-robot success** of any policy trained on this data.
- **Cross-robot generality.** The code is robot-agnostic by design but has run on OpenArm only.

## 3. What is completely implemented, partially implemented, weakest

**Completely implemented and reproducible** (verified in this task):
- the published datasets equal the saved candidates;
- re-planning gives bit-identical arrays;
- re-checking gives identical values;
- the instrumented renderer reproduces the published frames pixel-for-pixel;
- SAM2 re-runs reproduce the cache exactly.

This covers both tasks: automatic segmentation prompts, render-and-compare calibration, object proxies, F1/F2 trajectory editing with IK, replay checks, 3-D forward-warping renderer, export and verification.

**Partially implemented:**
- the three-camera dataset: wrist views are reused, not rendered;
- verification: no per-episode span check, so it missed the battery defect;
- the clearance check, which is vacuous where the source meshes touch;
- object models: coarse proxies and a rigid-grasp assumption.

**Weakest evidence:**
1. **Wrist-camera consistency and VLA usefulness.** These are not measured at all, and the wrist images are known to contradict the state on ~40–55% of frames in the two examples.
2. **Image fidelity at the new positions.** This is the core claim of "video generation" and is only indirectly supported by a recorded-pose benchmark.

## 4. What differs from the historical v2 pipeline

| | v2 (frozen) | sfagen (current) |
|---|---|---|
| object / landmark | HSV lid tracker initialised by one human click | "what disappears when picked up" + grasp-constrained proxy fit; no human input |
| RAFT | used for evaluation metrics only (circular keypoint following) | not used |
| camera | inherited extrinsic, ~20–40 px off | per-rig render-and-compare calibration |
| renderer | 2-D homography layers | per-pixel 3-D forward warping with hidden-surface borrowing |
| scope | Pringles only, body cam only, 3 episodes, local datasets | 2 tasks, 3-camera LeRobot datasets on the Hub (wrist views reused) |
| variations | G/P/C hand-over families (incl. regrasp/timing) | F1 object position, F2 interaction location |

See TECHNICAL §9 for the full table.

## 5. What the original five-block proposal still requires

The slide reads: Real demonstrations → Segmentation → Keypoints → Video generation → VLA dataset.

1. **Segmentation.** Done automatically. To claim accuracy, a small hand-labelled mask set is still needed.
2. **Keypoints.** Replaced by geometry: FK, a calibrated camera and object proxies. If image keypoints are wanted in the paper, they need independent manual landmarks to validate calibration and tracking (the v4A landmark prerequisites are still open).
3. **Video generation.** The body camera is done. **Wrist cameras are not generated.** A synthetic wrist-view renderer, or an honest single-camera dataset, is needed before claiming three-camera data. Fidelity at the new positions needs an evaluation design (e.g. re-recorded real demos at the shifted positions).
4. **VLA dataset.** Built and published. **No learning result:** a preregistered policy study (real-only vs real + synthetic, held-out positions, real robot) is still required, along with a hardware replay of a sample of synthetic trajectories.

## 6. Honest scientific interpretation

sfagen is a reproducible, simulation-free engineering pipeline that turns a few real demonstrations into kinematically checked variants with re-rendered body-camera video.
- **What the evidence shows:**
  - internal consistency (bit-exact reproduction);
  - better recorded-pose re-rendering than copying frames or than the previous renderers;
  - wider kinematic coverage of object and interaction positions.
- **What the evidence does not show:** that the images are faithful at the new positions, that the trajectories are executable on hardware, or that a VLA trained on the data performs better. Earlier learning screens on older generators (v3, v4B, v4B.1) were negative or mixed, and they remain the only learning evidence in the project.
