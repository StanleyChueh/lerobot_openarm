# R07: does the pipeline move to a new task? OpenArm wipe-board, 10 synthetic episodes (2026-11-18, Claude Opus 5.5)

First-party, pending review. Local only: nothing published, pushed or trained. Code uncommitted on `main-pipeline`
(base `fbe3130` + R05 + R06). Results and images: `../r07/`. Run dir: `artifacts/simbench_runs/wipe/`.

## Answer
- **Yes, end to end, with a small config change, after two general fixes found on this dataset.** 10 synthetic
  episodes from `ethanCSL/openarm_wipeboard_lerobot_real_v00` (rev `9d4b709`, 10 demos of 26–36 s: left arm passes an
  eraser to the right arm, which wipes a board). Export passes every check: loads in LeRobot, labels match, no copy of
  a real episode, all replay checks pass, video spans aligned, body codec 42.6 dB.
- **Image quality: the wiping part is the real video (unchanged); the moved part is mixed.** Shifts 5.3–11.9 cm.
  At 5–8 cm the arm and eraser look right; at 10–12 cm the left arm is smeared or partly missing during the approach
  (ep3 f96, ep9 f73), and a faint ghost stays where the eraser was. Sheets: `../r07/source_vs_synthetic_*.jpg`.
- **Time: 63 min of compute for 10 demos → 10 synthetic episodes** (2 RTX 4090), plus about 10 min of config work.

## What had to change
| Item | Change | Kind |
|---|---|---|
| `source` | repo id, revision, local path (same camera names) | per task, 5 lines |
| `objects` | `eraser: {grasped_by: L, grasp_index: 0, proxy: {type: box}}` (Pringles: can, right arm, cylinder) | per task, 3 lines |
| `variations` | object targets `auto` from the source demos (R03 rule) instead of Pringles' hand-written grid | general |
| **New flag** `reconstruction.photometric_align` | the body camera has **auto exposure** (background 37 → 52 grey levels when the arms move); every "changed pixels" step now first matches gain and offset per channel to the background (`recon.match_exposure`; also in calibration and renderer) | general fix, default off |
| **New flag** `reconstruction.object_footprint: sam2` | the object size came from "what disappears after the pickup", but here the board also changes (shadows of the wiping arm, ink wiped, cables, board moved) → eraser box 20 × 16 cm. Now: the largest SAM2 segment at the grasp point that mostly disappears → 15 × 5.4 cm on both checked demos | general fix, default off |
| Not changed, reused from Pringles | robot block, URDF, camera initial guess, wrist-camera lags, all thresholds | same robot |

Known errors left in: table height by the fallback rule is 0.219 m vs 0.200 m from the Pringles can fit on the same
table (≈ 2 cm); the eraser height falls back to the 8 cm cap (real ≈ 4 cm). Both affect the proxy, not the arms.
Checks: 69 / 69 unit tests; `guard_openarm.sh r07_flags_off` bit-identical (28 / 28, 4 / 4).

## Time, measured (attempt 2, 23:24 → 00:27; GPUs otherwise idle)
| Stage | Wall time | Notes |
|---|---|---|
| camera fit (`calibrate`) | 5.7 min | mostly decoding 10 full AV1 videos for the background median |
| table height | < 0.1 min | fallback rule |
| reconstruction of 10 demos (`source_positions.py`, 2 GPUs) | **33 min** | ≈ 6.3 GPU-min per 33-s demo |
| plan + replay check, 505 candidates | 5.5 min | 218 pass (43 %), 52 cells of 3 cm |
| preselect 12 + render 12 | 6 min | 27.5 s per 1000-frame episode |
| export (`build`) | 8 min | H.264 4:4:4 encoding of 3 cameras × 9313 frames |
| verify + comparison video | 3.3 min | optional |
| **total compute** | **63 min** | |
| human / agent setup | ≈ 10 min config; this time also ≈ 40 min to find and fix the two problems (attempt 1 discarded) | first time only |

Marginal cost per extra synthetic episode on this task: ≈ 27.5 s render + ≈ 48 s export ≈ **1.3 min**.

## Where the time goes (cProfile of one reconstruction, `../r07/recon_profile_ep2.txt`, 311 s)
| Part | Time | Share |
|---|---|---|
| **camera–joint time-offset search** (`_photometric_lag`, 240 trial renders) | **205 s** | **66 %** |
| background plates (3 passes) | 19 s | 6 % |
| SAM2 arm + object masks | ≈ 30 s | 10 % |
| change masks + exposure matching | 25 s | 8 % |

The offset is searched again for every demo, but it is a property of the rig: −0.25 to −1.25 frames (median −0.5) on
all 10 wipe demos and the same spread on the 10 Pringles demos.

## How it compares (published numbers)
| Method | Output | Speed |
|---|---|---|
| DemoGen (ICLR/RSS 2025) | point clouds, no RGB | ≈ 0.01 s per trajectory |
| Real2Render2Real (CoRL 2025) | RGB from Gaussian splats; needs an object scan | 51 demos / min on one RTX 4090; ≈ 10 min human setup |
| Real2Edit2Real (CVPR 2026) | RGB by video diffusion | generation is their stated bottleneck at low GPU counts; no per-episode number in the text |
| **sfagen (this run)** | RGB from the real video, no scan, no simulator | ≈ 1.3 min per extra episode; ≈ 1 h fixed cost per task |

## Speed-up plan (ordered by measured gain; none implemented yet)
1. Time offset once per rig (median over 2–3 demos), then a short per-demo check: −66 % of reconstruction
   (≈ 33 → 12 min). Needs a check that image quality does not drop (pose-transfer PSNR).
2. More reconstructions per GPU: peak GPU memory 0.6–3 GiB of 24 GiB and the work is single-core Python; 4 jobs
   per GPU → 10 demos in ≈ 10 min with step 1.
3. Export: the wrist videos are copied unchanged, yet re-encoded; copy their streams and encode the body camera in
   parallel per episode: 8 → ≈ 2–3 min.
4. Calibration: decode only the sampled frames instead of whole videos: 5.7 → ≈ 1 min.
5. Later: batch the point splatting over frames (the inner loop of rendering and checks).
Expected: 63 → ≈ 15–20 min fixed per task; ≈ 0.6 min per extra episode.

## Not checked
- Policy success with these 10 episodes; the hand-over and wiping are the source's own (not varied).
- Gold-set mask accuracy on this task (no gold labels).
- A different embodiment (SO-101 still not run); OpenArm wrist cameras were covered here, unlike in Pringles.
