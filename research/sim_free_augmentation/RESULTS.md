# Simulation-free, contact-aware hand-over augmentation: working prototype (one real episode)

* **Date:** 2026-11-05.
* **Data:** `ethanCSL/openarm_pringles_lerobot_real_10_v00` @ `cebdee98`, **episode 0**, `body_cam`.
* **Compute:** 1× RTX 4090 for SAM2 and RAFT, CPU for everything else. End-to-end runtime **136 s**.
* **Code:** `sfa/`, `scripts/run_pipeline.py`, `configs/ep0.json`.
* **Outputs:** `outputs/ep0/`.

**What did not run:**
* no Isaac Sim, IsaacLab, MuJoCo or any physics engine;
* no robot, hardware or CAN;
* no training;
* no Hub upload. Model weights were *downloaded*: SAM2.1-small and RAFT-large.

Everything below was produced by the committed code in one clean run (`outputs/logs/ep0_run.log`).

## 1. What the prototype does

| step | implementation | result on episode 0 |
|---|---|---|
| formats | LeRobot v3 parquet + AV1 video. 16-D interleaved `RJ1,LJ1,…,RJ8,LJ8` (motor rad); grippers raw (R open −1.298, L open 1.2121, closed 0); 3 cameras 480×640 at 30 fps | 273 frames. Command events: right close 97, left close 176, right open 181. Action leads state by **2 frames** (mean error 7.5 mrad) |
| segmentation | **SAM2.1-hiera-small** video (transformers). Arm prompts = projected URDF link4…finger hulls as **mask prompts** every 15 frames; can = lid point + red-body point | masks on all 273 frames for both arms and the can. Box+point prompts were tried first and **failed**: the left-arm mask locked onto the bright window |
| keypoint tracking | **RAFT-large** flow chaining with a forward–backward test, 64 semantic points (both hands, link7, fingers, can) | 95 % of point-frames visible in the original clip |
| motion recovery | Robot: FK (pinocchio, URDF). Video↔state lag re-estimated. Can: lid centre fitted as a fixed point in the **holding hand's** TCP frame (structure from known hand motion); the can axis from mask IoU | lag **−1.75 frames**. Giver fit **5.7 px** RMS (66 frames), receiver fit **2.4 px** (62 frames). Can-axis IoU 0.84 / 0.62. The standing can uses its lid ray at the recovered height, because the grasp pose reprojects 20.5 px off (the can slides during the grasp). No 3D is claimed where no hand holds the can |
| new trajectory | DLS IK (URDF limits − 0.02 rad), warm-started on the recorded posture. Pass point moved by **δ = (0, −4.5, +4) cm** | IK error ≤ 0.1 mm / 0.04° |
| candidate actions | `a_new[t] = a_rec[t] + (q_new − q_rec)[t+2]` on the arm joints; recorded gripper commands | both candidates meet the wrapper contracts: request–state gap 0.064 rad (≤ 0.5), jump 0.038 rad (≤ 0.25), 0 limit violations. Saved as `candidate_*.npz` and `candidate_actions_*.csv` |
| video synthesis | Background plate (masked temporal median, Telea inpaint for the 4.1 % never-free pixels). Per-link and can **homography warps** from projected 3D model points (original → new pose); far-to-near compositing with feathered masks. **No generative model** | 2 generated clips, 273 frames each, 22–27 s per clip |
| export | `clips/*_side_by_side.mp4`: original \| keypoint control (target skeletons, TCPs, can) \| generated. `comparison_original_geometry_contact.mp4`: 4-panel. Stills: `still_t*.jpg` | H.264 MP4s (15 MB total) |

## 2. The two generators (Stage 2 contact model: `sfa/plan.py`)

* **Geometry-only (G):**
  * each arm gets its own 6 cm TCP offset (same size, independent direction), applied by a time bump centred on the
    hand-over and blind to the contact phases;
  * the can is rendered with whichever hand held it in the recording.
* **Contact-constrained (K):** one offset, shared by both hands and switched on and off by the contact phases:
  * the giver offset starts only after the lift;
  * the receiver offset ramps in over its approach;
  * both hands are identical during dual contact.

  The model:
  * **object attachment:** can = T_holder · X_holder;
  * **dual-arm transfer:** the receiver's take pose is its recorded relative grasp applied to the can's new pose;
  * **phase transitions:** the can stays still before the grasp, and the receiver closes before the giver opens
    (5-frame overlap kept);
  * **joint limits;**
  * **arm–arm capsule clearance;**
  * **joint speed:** no new excess over the demo.

  A failing check shrinks δ (× 0.7, ≤ 5 tries). The checks are offline kinematics, not physics.

| check | geometry-only | contact-constrained |
|---|---|---|
| IK / joint limits / speed / clearance | pass (clearance 14.9 cm) | pass (clearance 11.0 cm; source 11.3) |
| receiver relative to the can at its close vs the recording | **10.1 cm off** | 0.01 cm |
| can continuous (max step vs source 4.3 cm) | **9.7 cm jump** | 4.3 cm |
| giver grasp offset at close | 1.2 cm | 0.0 cm |
| contact model `all_ok` | **False** | **True** (first try, scale 1.0) |

## 3. Stage 3 evidence (`outputs/ep0/results.json`)

All image numbers are automatic measurements on the clips, not ground truth.

| metric | original | geometry-only | contact-constrained |
|---|---|---|---|
| **Keypoint following**, hand-over window (frames 97–221): tracked point vs commanded position, median / p90 px | — | 2.0 / 35.8 | 2.4 / 33.3 |
| … commanded displacement of those points, median / max px | — | 61 / 148 | 44 / 93 |
| … can points only, median / p90 px | — | 2.2 / 16.4 | 2.7 / 10.4 |
| **Temporal consistency:** flow-warp error, whole clip / hand-over window (8-bit) | 1.98 / 2.23 | 2.01 / 2.15 | 2.08 / 2.35 |
| static-background flicker (8-bit) | 2.03 | 2.03 | 2.03 |
| **Robot/action alignment:** prompt-free foreground IoU with the commanded pose / with the original pose (median, 42 hand-over frames) | 0.48 / 0.24–0.35 (recorded vs new) | 0.51 / 0.18 | 0.46 / 0.31 |
| … frames where the commanded pose explains the image better | 88–95 % (recorded) | 100 % | 88 % |
| **Grasp attachment in the clip:** lid re-tracked, rigid fit to the new giver / receiver TCP, RMS px | 5.7 / 2.4 | 5.7 / **no fit** (receiver hold: 0 trackable frames) | 6.0 / 2.9 |
| lid tracker frames lost | 0 | **90** | 17 |
| receiver fingers to lid during the transfer (tracked in the clip), median px | 73 | **241** | 101 |
| **Candidate action feasibility** (wrapper contracts, URDF limits) | pass | pass | pass |
| contact model (§2) | — | **fails** (attachment, continuity) | passes |
| render time per clip | — | 27.2 s | 22.8 s |

**Reading.**
* **Both clips move the robot where the new actions say.**
  * Keypoint following: median 2–2.4 px for points commanded 44–61 px away.
  * Pose alignment: the commanded pose explains the image better than the original pose in 88–100 % of frames, with
    absolute IoU ≈ the original clip's own 0.48.
  * Temporal consistency: the warp error stays at the original video's level.
* **Only the contact-constrained clip keeps the can consistent with the hands.**
  * Its lid is rigidly attached to the new giver and receiver TCPs at almost the recorded residual (6.0 / 2.9 px vs
    5.7 / 2.4).
  * In the geometry-only clip:
    * the receiver closes about 10 cm from the can;
    * the can then jumps between hands (the tracker loses it for 90 frames);
    * during the transfer the receiver fingers sit 241 px from the lid, against 73 px in the recording.
* **The candidate joint actions are kinematically feasible in both cases.** Kinematic feasibility alone does not
  catch the contact failure; the contact model does.
* **One unexplained number:** the contact-constrained transfer distance (101 px vs 73 px) is larger than the
  recording's. Perspective, tracking error and the shared 3D offset are all plausible causes; it is not diagnosed.

## 4. Observed failure modes

Inspected on the stills and clips.

1. **Ghost residue:** robot parts SAM2 misses (dark finger tips, some cables) stay at the original position, so fingers
   appear doubled (e.g. t = 160–178).
2. **Dark smears:** 4.1 % of pixels are never free of the robot (arm bases, the right-arm corridor). The plate there is
   Telea inpainting, which shows as dark blur where an arm moved away.
3. **Link seams:** one homography per link approximates out-of-plane rotation. Small seams appear at link boundaries,
   and the R hand carries the largest residual (keypoint p90 64–88 px; partly also tracking drift on dark, textureless
   parts).
4. **Geometry-only teleport:** the can jumps between hands, and a red sliver of the old can remains where the can mask
   was imperfect.
5. **Generated-clip tracking:** the HSV lid tracker loses the can more often on generated frames (K 17, G 90, original 0).
6. **Absolute calibration:** the inherited body_cam model is 20–40 px off in places, and the projected-silhouette IoU
   is only ≈ 0.48 even on the original. Warps use only the *relative* projected motion, which is far less sensitive.
   Absolute 3D positions should not be trusted beyond that.
7. **Wrist cameras not generated:** they move with the hands and have no calibration, and re-rendering them needs
   novel-view synthesis. The candidate episode therefore has a consistent `body_cam` only; it is not yet a complete
   3-camera LeRobot episode.

## 5. What was tried and replaced

| component | first choice | outcome | replacement |
|---|---|---|---|
| keypoint tracker | CoTracker3 (weights downloaded) | code fetch from GitHub too slow (~25 KB/s tarball). Running the individually fetched files was **denied by the session's permission policy** (code from an external source); not worked around, files deleted | torchvision RAFT flow chaining |
| segmentation model | SAM2.1-hiera-large | download ~0.4 MB/s (≈ 40 min) | SAM2.1-hiera-small |
| arm prompts | box + points | left mask = bright window | projected-silhouette mask prompts |
| standing-can position | attachment pose at the grasp | 20.5 px off the image (the can slides during the grasp) | lid ray at the recovered height |
| video model | — | no video-diffusion weights fetched (several GB at 0.4 MB/s) | geometric warping + plate inpainting |

## 6. Limitations (what this does NOT show)

* **Real-world executability or force validity:** not shown. The actions satisfy joint-space contracts and an offline
  kinematic contact model only. No dynamics, friction or grasp stability is checked, and no measured force exists.
* **Scale:** one episode and one offset. No diversity sweep, no other episodes, no policy-training test.
* **Photo-realism under large edits:** limited by the plate and the per-link warps, with no learned refinement.
* **Automatic labels only:** the SAM2 masks, lid tracks, keypoint tracks and camera model are automatic or inherited,
  not human ground truth.

## 7. Next improvements

Ordered by expected payoff:
1. **Self-calibrate body_cam from this data:** bundle adjustment over the lid point and the orange wrist tags under
   known FK motion. This should reduce the 20–40 px absolute error, and with it the seams and the mask mismatch.
2. **Masks:** SAM2-large with more prompts, and cable handling.
3. **Texture-mapped mesh re-rendering:** project real pixels onto the URDF meshes, then re-render, instead of per-link
   homographies.
4. **Learned refinement:** a video inpainting / refinement model for disocclusions and seams, conditioned on the
   keypoint-control video. Requires downloading weights.
5. **CoTracker3:** if the user permits running its code, for more robust tracks.
6. **Wider generation:** batch over episodes and a δ / timing / approach sweep, with the contact model as the
   accept/reject gate.
7. **Wrist cameras:** novel-view synthesis.
8. **Only then:** a policy-level test of whether the generated clips help.

## 8. Reproduce

```bash
cd ~/Stanley_ws/lerobot_openarm
E="env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1"
$E .venv/bin/python research/sim_free_augmentation/scripts/run_pipeline.py --config research/sim_free_augmentation/configs/ep0.json
$E CUDA_VISIBLE_DEVICES= .venv/bin/python -m unittest discover -s research/sim_free_augmentation/tests -v   # 11 tests
```

Needs:
* the local HF cache of the dataset;
* `facebook/sam2.1-hiera-small`;
* torchvision `raft_large_C_T_SKHT_V2`;
* the Gate 1 lid track `research/contact_aware_augmentation/outputs/gate1/tracks/lid_track_ep0.npz`.

`outputs/ep0/cache/` (SAM2 masks, raw frames; 209 MB) is not committed and is regenerated by the run.
