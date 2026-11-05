# v3.2 — simulation-free three-camera generation (body_cam + wrist_cam + right_wrist_cam)

* **Inputs:** real 3-camera Pringles demonstrations (`real_10`, all 10 episodes) and the v3 candidates (C, exported set).
* **Not used:** Isaac, MuJoCo, physics, robot motion, pushes.
* **Methods:** kinematics, projection, RAFT flow, triangulation, image warping and inpainting.
* **Rule respected:** no generated episode contains an unchanged source wrist video under a modified trajectory. This is enforced as a hard validation gate.

## Summary

1. **Calibration (no board):**
   * **Lags:** the wrist cameras' video-to-state lags are measured without depending on the mount: right −1.5 frames, left 0.0 (body cam −1.75 on ep0).
   * **Mount rotation:** the nominal (GUI-tuned) rotations are consistent with the image rotations, with a 1.2° held-out residual. A rotation-only hand-eye fit does not beat them.
   * **Mount translation:** not identifiable from video plus FK (fits land 18–23 cm from nominal).
   * **Can-silhouette fit:** projection agreement improves on held-out episodes (IoU 0.23 → 0.37 left, 0.06 → 0.26 right). The right-camera fit is constrained, because a free fit (61°, 20 cm) contradicted the independent image rotations (residual 1.25 → 3.1°).
   * **Bottom line:** the mounts are *consistency* calibrations, not verified physical extrinsics.
2. **Generation works on all 10 source episodes**, 47 full episodes (37 variants plus 10 identity controls), both wrist views each.
   * **Identity:** reproduces the source wrist frames exactly (regression-tested).
   * **Three-camera dataset:** 37 episodes and 9,398 frames, validated. Schema, exact state/action, boundaries, per-camera video segments, decoded PSNR ≥ 36.9 dB, and zero unchanged wrist frames under a moved camera.
   * **Sync on decoded video:** right −1.5 (expected −1.5), left −0.5 (expected 0.0).
3. **Evaluation (Task D) — honest mixed result:**
   * **Object motion:** the can shift seen in the generated wrist views follows the commanded shift better than unchanged frames would. Right wrist, new positions: 71 vs 119 px error; left 8–9 vs 12–21 px.
   * **Grasp evidence:** preserved; during holds 66–69% of right-wrist pixels are the attached can, with an identical red fraction.
   * **Viewpoint:** the generated frames are **not** closer to the real frame recorded from the nearest camera pose than the unchanged frames are (50% win rate, median NCC gain ≈ 0). Viewpoint realism is not demonstrated.
   * **Temporal:** about 5–7% more flicker than the source (flow-warp error 3.29 vs 3.09 right, 1.96 vs 1.89 left).
   * **Novelty:** modest (mean |Δ| of 3–8 levels; 8–20% of pixels changed by more than 10 levels), because the variations move the camera only 1–6 cm.

## C1. Inspection

* **Mounting:** each wrist camera sits on its own link7 (= IsaacLab ee_tcp). Nominal offset (0.05342, 0.001, 0.12199) m, quaternion (0.68057, −0.19188, −0.19188, 0.68057), identical for both arms and GUI-tuned (`stack_ik_abs_visuomotor_env_cfg.py`).
* **Intrinsics:** measured, from `rs-enumerate-devices` (earlier audit): left 605.44/605.13/322.85/246.44, right 607.79/607.61/322.58/248.52, zero distortion.
* **Content:** mostly static room (walls, window, black hose, plates, table edge). The other arm is rarely visible.
  * The can is visible during approaches.
  * While the observing hand holds the can, it fills about half the view and moves with the camera.
* **Timestamps:** all three cameras share the dataset frame index. Their effective lags to the state differ, as measured in C2.
* **Cross-camera object evidence:** the can is visible in the body cam throughout. In the right wrist it is visible during the giver's approach and hold; in the left wrist during the receiver's approach and hold.
* **Occlusion and disocclusion:** moving a wrist camera reveals scene regions the source frame never saw, mostly at image borders. Close to the can, the can occludes the table.

## C2. Calibration and diagnostics (`outputs/calibration/wrist_calibration.json`, `figures/wrist_calibration.png`)

| | right_wrist_cam | wrist_cam (left) |
|---|---|---|
| lag (rotation-magnitude scan, mount-independent) | **−1.5 frames** | **0.0 frames** |
| image vs FK rotation-magnitude correlation | 0.67 | 0.77 |
| hand-eye rotation residual, held-out eps 5–9: nominal / rotation fit | 1.25° / 1.20° (not adopted: < 0.1° gain) | 1.23° / 1.23° (not adopted) |
| translation fit (scale-free AX = XB) | 18.5 cm from nominal (not trusted) | 22.9 cm (not trusted) |
| can-silhouette IoU, held-out: nominal → fitted | 0.06 → 0.26 (constrained: ≤ 10°, ≤ 8 cm, both at the bound) | 0.23 → 0.37 |
| hand-eye residual with the fitted mount | 1.16° (free fit: 3.11° → rejected) | 1.25° |
| adopted mount | constrained can fit | can fit |

**Adoption rule:** held-out can IoU must rise by at least 0.05, and the held-out hand-eye residual may worsen by at most 0.25°.

The fitted mounts absorb body-cam can-reconstruction and FK errors. A ChArUco capture, protocol in `research/task_aware_real2sim/m2a/WRIST_CALIBRATION_M2A.md` §7, remains the way to get true extrinsics.

## C3. Generator (`sfa32/render_wrist.py`)

**Per source episode and camera:**
* camera poses T_j = FK_link7(q(j + lag)) · X;
* per-frame depth from RAFT flow to frame j ± 3, triangulated along the known relative camera pose (parallax < 0.4° → "far" at 2.5 m);
* camera-attached pixels: near-zero flow under predicted rotation flow;
* can mask: convex hull of the red HSV components. The lid colour range was dropped because it matched the beige wall.

**Per generated frame i:**
* the new camera pose follows the candidate trajectory, T'_i = FK(q_new(i + lag)) · X;
* the source frame is j = round(τ(i + lag) − lag);
* static scene: z-buffered forward splat;
* holes: filled from the two real source frames with the nearest camera poses (multiview), then Telea inpainting at half resolution;
* can not held by this hand: rigid homography from the source to the candidate can pose;
* can held by this hand (gripper closed in both source and candidate): copied with the finger pads.

Every pixel is labelled with its provenance:

| label | meaning |
|---|---|
| reprojected | known surface, same frame |
| attached | camera-attached copy |
| can warp | rigid can warp |
| multiview | known surface from another real frame |
| inpainted | hallucinated surface |
| unknown | inpainted more than 25 px from any known pixel |

**Pixel provenance on frames whose camera moved** (`figures/wrist_provenance.png`, `wrist_eval_summary.json`):

| set | reprojected | attached | can warp | multiview | inpainted | unknown |
|---|---|---|---|---|---|---|
| new initial position, right wrist (27) | 46.2% | 35.8% | 9.2% | 5.2% | 2.2% | 1.4% |
| new initial position, left wrist (27) | 84.7% | 1.7% | 11.9% | 1.3% | 0.4% | 0.0% |
| original position, right wrist (10) | 30.1% | 58.2% | 8.3% | 1.5% | 1.3% | 0.5% |
| original position, left wrist (10) | 56.0% | 27.8% | 9.6% | 3.6% | 2.1% | 0.8% |

Known (reprojected, attached, multiview) surfaces dominate. Hallucinated (inpainted) and unknown pixels together are 0.4–3.6%, concentrated at image borders revealed by the camera moving.

## C4. Outputs

* **Short clips** (`outputs/clips/ep{0,3,8}/*_{approach,handover}_3cam.mp4`, 32 frames each):
  * row 1: original body / left wrist / right wrist;
  * row 2: generated views;
  * row 3: provenance maps.
* **Full episodes:** `outputs/episodes/ep{0,3,5,8}/*_full_3cam.mp4`.
* **Object-tracking overlay:** `outputs/clips/ep3_init_x-3__C__s1.0_right_wrist_object_overlay.mp4`, showing the commanded can outline (green), the detected can hull (red) and provenance.
* **Camera pose plot:** `figures/wrist_camera_poses_ep0_init_x-6__C__s1.0.png`. The right-wrist camera translates up to 6 cm during the relocated reach, with about 0° extra rotation; the left camera moves up to 1.8 cm.
* **Synchronization:** `dataset_export/sync_replay_threecam.json`. Per-camera lag curves re-measured on the decoded dataset match the manifest lags within 0.5 frame.

## C5. Three-camera LeRobot export (`outputs/dataset_export/sfa_v32_C_threecam_*`)

| check | result |
|---|---|
| episodes / frames | 37 / 9,398 (10 source episodes; 93 v3 C_all candidates not in the representative set) |
| schema (3 cameras, 16-D names, fps 30), boundaries, exact state/action | ✓ |
| per-camera video segment = n_frames / fps | ✓ |
| decoded PSNR min / median: body / wrist / right wrist | 37.9 / 40.7, 37.9 / 41.8, 36.9 / 40.8 dB |
| unchanged wrist frame under a camera move of more than 1 px at 2 m | 0 |
| decoded-video sync: right / left wrist lag | −1.5 / −0.5 (expected −1.5 / 0.0) |

**Row convention:** camera c's image in row i depicts state time i + lag_c. The lags are recorded per episode in the manifest.

**Failure found and fixed:** the first export used LeRobot streaming encoding, which **dropped one `wrist_cam` frame** in episode 19. That episode's last row then decoded from the next episode.
* A video-segment-length check (`scripts/check_segments.py`) is now a hard gate.
* The export uses non-streaming encoding.
* All v2/v3/v3.1 datasets were re-checked and are unaffected.

## D. Research assessment

| question | evidence | verdict |
|---|---|---|
| Do the new wrist views reflect the generated robot motion? | Camera poses follow the generated trajectory by construction, and decoded-video sync matches. But in the nearest-real-viewpoint test, generated frames beat unchanged frames in only 50% (new positions, right wrist) and 33–49% (original positions) of tested frames. | **Geometric motion: yes by construction. Photometric viewpoint realism: not demonstrated.** |
| Is the relocated object consistent? | Detected can shift vs commanded shift: right wrist 71 vs 119 px error for unchanged frames; left 9 vs 12 px. During the object-relative final approach, the can keeps its demonstrated place in the view, as it should. | **Partially:** better than unchanged views, still calibration-limited. |
| Is grasp-related evidence preserved? | Hold frames: 66–69% attached can pixels (right), red fraction identical to the source. | **Yes, by construction:** the hand–can relation is the source's. |
| Temporal and cross-camera consistency? | Flow-warp error +5–7% vs source. Can visibility agreement with the commanded pose is identical to the unchanged control (0.55 right, 0.89–0.91 left), limited by calibration. | **Slight temporal degradation; no measurable cross-camera gain.** |
| Genuinely different observations? | Mean \|Δ\| 3–8 levels, 8–20% of pixels changed by more than 10 levels, on 99–154 changed-pose frames per candidate. | **Modestly different:** small camera translations (1–6 cm). |

**Known vs hallucinated vs unknown:**
* **Known:** reprojected, attached and multiview-filled pixels come from real wrist images.
* **Inferred:** the can-warp pixels are real can pixels under an assumed rigid motion.
* **Hallucinated or unknown:** only 0.4–3.6% of changed-frame pixels are inpainted or unknown.

### Failure analysis

* **Calibration is the bottleneck.**
  * The right-wrist mount fit hits its constraint bounds, and the projected can is still 50–100 px off at close range.
  * The body-cam can reconstruction (2–39 mm closure floor) propagates into the wrist views.
* **Depth from two-view triangulation** is noisy for low-parallax frames (set to 2.5 m). Near objects other than the can (the other arm, table edges) are treated as static.
* **Can tearing at close range:** on 6 cm relocations, the can's non-red top and lid edge can tear when this hand approaches it closely.
* **No refinement model:** the optional pretrained image/video refinement was not used, and no learned generator touched the pixels.
* **Scope:** 37 variants are exported, not all 136 C candidates (a representative set per episode, chosen to bound compute).

### Recommendation

1. **Calibrate properly first:** run the ChArUco wrist-calibration capture (operator-run, about 30 min) before using generated wrist views for training. All Task D failures trace to mount and can-pose uncertainty.
2. **Train body-cam-first:** given Task A (the policy uses vision, and synthetic handovers hurt), the next training screen should stay body-cam-only and balanced (initial-position-only vs handover-only synthetic data, fixed real fraction).
3. **Only then test wrist views:** add the three-camera arm (`sfa_v32_C_threecam`) as a separate factor once its calibration passes a held-out landmark test.

## Runtime

| stage | time |
|---|---|
| hand-eye diagnostics, 10 episodes | 88 s |
| can-silhouette mount fit | 283 s |
| per-episode wrist source geometry (RAFT flows + triangulation, both cameras) | 9–52 s |
| wrist rendering | median 35 s per candidate (2 cameras × 250–280 frames, 12 CPU workers in parallel) |
| 10 episodes × 5 candidates | about 11 min wall-clock |
| three-camera export (non-streaming) + validation | 267 s + 10 s |

## Commands (from `research/sim_free_augmentation/v3_2`)

```bash
PY="env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python"
$PY scripts/calib_wrist.py && $PY scripts/calib_wrist_can.py && $PY scripts/crosscheck_mounts.py && $PY scripts/calib_wrist_can_constrained.py
$PY scripts/gen_wrist.py --episodes 0 3 8 --clip            # short 32-frame clips
$PY scripts/gen_wrist.py --episodes 0 1 2 3 4 5 6 7 8 9      # full episodes, representative set
$PY scripts/eval_wrist.py --episodes 0 1 2 3 4 5 6 7 8 9 --max_per_episode 4 && $PY scripts/temporal_wrist.py
$PY scripts/export32.py && $PY scripts/sync_replay.py && $PY scripts/make_threecam_videos.py && $PY scripts/fig_threecam.py
$PY -m unittest discover -s tests -v
```
