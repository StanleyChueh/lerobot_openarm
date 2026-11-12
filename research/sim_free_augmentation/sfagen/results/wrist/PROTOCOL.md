# B: wrist-camera rendering, frozen protocol (task-agnostic, Pringles + battery)

**Frozen before any wrist rendering**, on 2026-11-12. Its sha256 is recorded in every result file.

## Question
Can sfagen re-render the two wrist cameras for a changed scene better than copying the source wrist frame? Copying is
what ships today, and those frames are flagged invalid.

**Changed scene** means three things:
- the camera moves with the new arm trajectory;
- objects move;
- the other arm moves.

The renderer may use only the robot URDF, joint states, the body-camera object reconstruction and the source wrist
video. It may not use task-specific colours or labels.

## Data
- **Pringles:** `ethanCSL/openarm_pringles_lerobot_real_10_v00@cebdee98`, episodes 0–9
  (`configs/openarm_pringles_handover.yaml`, cached reconstructions).
- **Battery:** `ethanCSL/openarm_insertbattery_lerobot_real_10_v02@a1c29d77`, episodes 0–9
  (`configs/openarm_battery_insertion.yaml`, cached reconstructions).
- **Cameras:**
  - `observation.images.wrist_cam`, on the left arm (L);
  - `observation.images.right_wrist_cam`, on the right arm (R).
- **Splits:**
  - **dev:** Pringles episodes 0–4, for design, depth-method choice and calibration refinement.
  - **held-out:** Pringles episodes 5–9.
  - **battery:** episodes 0–9, **held-out only**, never used for design. This is the generality test.

## Camera model (initial, before refinement)
- Camera pose = FK(`openarm_{side}_link7`) · X, at state time = video frame + lag.
- **X (mount):** the v3_2 adopted mounts (`v3_2/outputs/calibration/wrist_calibration.json`, can-silhouette fits).
- **Lag:** R −1.5, L 0.0 frames (v3_2, mount-independent).
- **Intrinsics:** the measured RealSense values (v3_2 `wristcam.py`).

## Tests
**T1, same-episode novel view.**
- Frames: j on the grid 10, 20, …, T−20, with k = j + d for d ∈ {4, 8, 15}.
- Keep only pairs whose camera moved ≥ 1 cm or ≥ 2°.
- Render source frame j with the full real scene at frame k (both arms, grippers, objects), never borrowing from source
  frames within ±5 of k. Compare with real frame k.

**T2, cross-episode, generation-like.**
- Pairs: ordered pairs (A, B) within one split.
- Phase times:
  - t_g = the first gripper close of the object-0 grasping arm;
  - t_l = the next close of the other arm after t_g, or T−1 if there is none;
  - phases start = 5, approach = 0.6 t_g, grasp = t_g, lift = t_g + 0.5 (t_l − t_g), with offsets −2, 0, +2;
  - video frame = state time − wrist lag.
- Render A's frame with B's real scene at the aligned frame, and compare with B's real frame.
- Keep only frames where the A-to-B camera rotation is ≤ 5°: generation keeps hand orientation, so larger rotations are
  outside its use.
- Bin by camera translation: [0, 3), [3, 6), [6, 11), [11, 30] cm.

**Score.**
- PSNR over the whole image.
- Secondary: per region, using geometry labels at the ground-truth frame (own hand plus held object, other objects,
  other arm, world).
- The `copy` baseline is the source frame unchanged.
- Pair-level (T2) and episode-level (T1) means.
- 95 % CIs: episode-cluster bootstrap, 2000 resamples, seed 0.

## Design choices (dev only; held-out and battery are rendered once, after the choices are frozen)
- **Background depth:** RAFT triangulation along the known camera motion, or Depth Anything V2 metric depth
  scale-aligned to the table plane and URDF depth. Take the higher dev T1 mean PSNR.
- **Calibration refinement:** search mount translation ±5 cm, rotation ±3° and lag ±1 frame on dev T1.
  - Adopt if the dev T1 mean gain is ≥ 0.3 dB **and** the held-out T1 change is ≥ −0.1 dB. The held-out check is the
    only held-out use before the final run.

## Gate (per task and camera, on held-out Pringles and on battery)
- **Pass** if:
  - the T1 render − copy CI lower bound is > 0; **and**
  - the T2 render − copy CI lower bound is > 0 in every translation bin below 11 cm that has ≥ 3 pairs.
- **On pass:** generated wrist frames come from the renderer. `wrist_view_valid = 1` where the frame is an exact copy,
  or rendered with ≤ 5 % inpainted pixels; 0 otherwise.
- **On fail:** that camera keeps today's copy plus the invalid flag, and the failure is reported.

## Not claimed
- PSNR is a proxy, not policy usefulness.
- The mounts are consistency calibrations, not a ChArUco extrinsic.
- Pringles episodes are reused development data.
