# Battery `wrist_cam` frame deficit (84,689 vs 84,690): cause

**Status:** cause located and confirmed. The published files are unchanged; this task authorizes no dataset modification.

## Observation

`configs/work/openarm_battery_insertion/verification.json` reports 84,690 decoded frames for `body_cam` and `right_wrist_cam`, but 84,689 for `wrist_cam` (the left-wrist camera).

## What was checked (2026-11-10, read-only)

1. **Decoded frame count per file.**
   - `wrist_cam/chunk-000/file-000.mp4` has 47,900 frames, as expected.
   - `file-001.mp4` has 36,789 frames against 36,790 expected.
   - ffprobe (`-count_frames`), PyAV demux/decode and torchcodec all agree: the file holds 36,789 coded frames. No decoder drops the frame; it is absent.
2. **Episode metadata** (`meta/episodes`, `videos/observation.images.wrist_cam/{from,to}_timestamp`). Every episode's timestamp span equals its length, except **synthetic episode 91**: length 700 frames, but a wrist_cam span of 699 frames. All other episodes and all other cameras are consistent, and episodes are contiguous inside each file.
3. **Location of the missing frame.** Episode 91 is source real episode 8, `battery_x+0_y-6` (object-position family). Its 699 encoded wrist frames were aligned to the 700 real source wrist frames, testing every single-skip hypothesis.
   - Best explanation: **source wrist frame 139 is absent** (total cost 1171, vs 2277 for "last frame missing" and 1944 for "first frame missing").
   - Median absolute frame difference under that alignment: 1.34 before the skip and 1.80 after it, vs 5.40 / 2.84 under the wrong hypothesis.
4. **Effect when loaded with LeRobot** (`LeRobotDataset(...)[i]`).
   - Episode 91, frames 139…698: the `wrist_cam` image is the real wrist frame of the *next* time step, one frame (33 ms) ahead of `observation.state` / `action`.
   - **Episode 91, frame 699** (global index 68,522): LeRobot returns the **first wrist frame of episode 92**, a different synthetic episode built from source 8, frame 0. Its mean absolute difference to that frame is 1.28, vs 15.38 to the correct source frame.
   - No error is raised.
   - `body_cam` and `right_wrist_cam` are unaffected.
5. **Hub copies.** The Hub `v3.0` files have the same SHA-256 as the local files (LFS sha256 `27809dec…`):
   - `ethanCSL/openarm_insertbattery_lerobot_real_to_sim_10_3cam_v00` @ `c3c571d0`;
   - `ethanCSL/openarm_insertbattery_lerobot_real_and_sim_10_3cam_v00`, where the same file is reused as `wrist_cam/chunk-000/file-003.mp4`. The affected episode there is **101** (= 10 real + 91).
6. **Other sfagen datasets.** The same metadata check finds no mismatch in:
   - the Pringles 96-episode synthetic and mixed datasets;
   - the nine source-budget datasets `openarm_pringles_lerobot_real_to_sim_{1..9}_3cam_v00`.

## Cause

One wrist frame (source frame 139 of episode 91) was **not written by the streaming AV1 encoder** during `LeRobotDataset.add_frame` / `save_episode` with `streaming_encoding=True`.
- The encoder's per-episode frame counter then recorded 699 frames for that episode.
- All later episodes were nevertheless contiguous, so only episode 91 is affected.

The same class of fault occurred once more:
- **Where:** the source-budget builds of 2026-11-10, datasets N = 5, 6 and 7.
- **What was lost:** the last wrist frame of the last episode.
- **Effect of rebuilding:** the rebuilt datasets were complete, so the fault is non-deterministic.

**Mechanism inside the encoder: UNKNOWN.** The thread/queue code of LeRobot's streaming encoder was not instrumented. The frame loss is a fact; why the frame was lost is not established.

## Why the existing verification missed the meaning

`sfagen verify` compares only the total decoded count per camera. It flagged 84,689 ≠ 84,690, but did not check per-episode spans. The difference was reported as a number, not as a defect.

**Recommended (not done here):**
1. Add a per-episode `span == length` check to `verify`.
2. Rebuild the battery datasets (synthetic and mixed) with a new tag. The previous tag should be kept.

The training impact is one frame of cross-episode leakage plus a one-frame left-wrist misalignment in a single episode. It is small but real.
