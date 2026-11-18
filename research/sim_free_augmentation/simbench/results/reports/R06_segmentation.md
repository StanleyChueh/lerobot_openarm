# R06: arm and object masks scored against a hand-checked gold set (2026-11-17, Claude Fable 5.2; camera step by Claude Opus 5.5)

Status: **RoboTwin 17 / 18 gold frames with the pipeline's own camera; OpenArm 0 / 12 (robot model, not camera). Not done.** First-party, pending independent review and your approval of the gold set.

## Update (Opus 5.5, same day): the camera step — read this first
| | Baseline v1 | R06 masks, fitted camera | **R06 masks + refined camera (H)** | R06 masks, true camera |
|---|---|---|---|---|
| Camera error vs truth, median px: cup / stack / hand-over | 25.8 / 15.8 / 35.3 | same | **3.7 / 3.0 / 3.6** | 0 |
| Gold frames passing: cup / stack / hand-over | 0/6, 0/6, 0/6 | 4/6, 4/6, 2/6 | **6/6, 5/6, 6/6** (hand-over after 4 rounds; 5/6 after 2) | 6/6, 6/6, 6/6 |
| Gold frames passing: OpenArm | 0/12 | 0/12 | **0/12** | – |
| Pose-transfer PSNR vs baseline: cup / stack | – | – | **+1.4 dB [0.9, 1.9] / +1.7 dB [1.2, 2.3]** | +1.8 / +1.9 |
| Pose-transfer PSNR vs baseline: OpenArm (172 pairs) | – | +0.36 [0.29, 0.44] | +0.25 [0.14, 0.36] | – |

**New: `sfagen calibrate CONFIG --refine`** (`sfagen/calibrate.py:refine`, `contour_icp`, `contour_pairs`, `contour_frames`).
Rounds of: reconstruct `calibration.refine_episodes` with the current camera (R06 masks) → pair the outer contour of the
projected URDF with the nearest arm-mask edge (pixels next to an object mask or the image border are skipped) → solve
the pose by PnP from the current pose (intrinsics fixed) with a threshold shrinking 40 → 3 px → adopt if the mean
contour distance on held-out episodes drops. Then re-make the masks with the new camera and repeat (2 rounds; hand-over
got 2 more). About 20–30 s per ICP; the reconstructions dominate (about 10 min per episode with 4 jobs on 2 GPUs).
The camera was refined only on episodes **outside** the gold set (cup right-arm group 0,1,2,4; stack group RL 1,3,4,5;
hand-over 2,3,4,6; OpenArm 1,2,4,6) and copied to the gold-set work dirs.

| Rig | Truth error per round (median px; p95) |
|---|---|
| cup | 25.8 → 3.8 → 3.7 (p95 43.9 → 8.4) |
| stack | 15.8 → 7.0 → 3.0 (p95 → 5.9) |
| hand-over | 41.3 → 25.4 → 6.4 → 2.8 → 3.6 (p95 → 7.9; rotation 0.9 → 0.6°, centre 3.0 → 2.5 cm in the last round) |

**What did not work first (kept as negative results):**
- Powell on mask IoU from the fitted camera: stack 15.6 → 8.1 px, cup 26 → 28, hand-over 41 → 36.
- Blurred coarse-to-fine IoU with random starts: cup 33 / 40 / 19 / 47 px, hand-over 35 / 17 / 10 px. Differential
  evolution: cup 17 px, hand-over 41 px. Silhouette IoU is flat along rotation-plus-translation directions.
- Contour ICP before closing the point-splat holes: the true camera itself scored 17–21 px of contour distance
  (holes inside the projected arm made false contours). After the fix: 5.7 / 7.1 / 12.3 px.
- Random restarts of the ICP (±8°, ±10 cm): all worse; the lowest contour distance always picked the best one.

**Open failures with the refined camera:**
- stack e7 f67 (grasp): gripper recall 0.03. The arm's SAM2 track holds 0.8 of the gripper until frame 56, drops to
  0.4 from the prompt at frame 60 while holding the block; the fingers end up owned by nothing. A tracking failure at
  contact, not a camera error.
- hand-over: none after 4 rounds (after 2 rounds, e0 f30 failed: arm precision 0.46 on a 637-px arm sliver at the image top; kept in `artifacts/simbench_runs/r06/logs/gold_H_handover_after2rounds.json`).
- **OpenArm 0/12.** Two causes seen on the overlays: (1) the wrist cameras on both arms are not in the URDF, so the
  masks leave them out while they move with the arm (they would stay behind as ghosts); left-gripper recall down to
  0.38–0.44 on demo 5; (2) the right gripper projects 20–30 px off while the left arm lines up, which a camera cannot
  cause: joint zero offsets or a time offset. The refinement changes the OpenArm contour distance only 15.06 → 15.01 px
  and the image quality −0.11 dB [−0.20, −0.01] versus the fitted camera with the same masks.

**Rule proposed for the next pre-registered run (set after seeing OpenArm; not applied to these results):** adopt a
refined camera only if the held-out contour distance drops by more than 1 px (× W / 640). It would keep all RoboTwin
rounds and reject OpenArm.

Other checks: unit tests 68 / 68 (new: contour ICP on two synthetic robots, error 12 → 1.8 px at 320 px width; the
threshold of that test was set after its first run); `guard_openarm.sh r06b_flags_off` bit-identical (28 / 28
candidates, 4 / 4 body renders). Files: `../r06/gold_H_{cup,stack,handover,openarm}.{json,jpg}`,
`../r06/bench_H_refined_camera.json`, logs `artifacts/simbench_runs/r06/logs/H_*`.

---
(Below: Fable's R06 report before the camera step, unchanged.)

Status at that point: **not done**. First-party, pending independent review and your approval of the gold set.
Code is uncommitted on `main-pipeline` (base `fbe3130`, on top of the uncommitted R05 changes). Protocol with three
dated amendments: `../r06/PROTOCOL.md`. Run dir: `artifacts/simbench_runs/r06/`.
No dataset rebuilt or published, nothing pushed, no training, no robot, no simulator.

## Answer in five lines
1. **With the true camera, every RoboTwin development frame now passes the gate: 18 of 18** (was 0 of 18 with the R05
   rules). Arm recall and precision, gripper recall and object IoU are all at or above 0.97 on the scored pixels.
2. **With the camera the pipeline fits itself, 10 of 30 frames pass** (was 0 of 30): RoboTwin 10 of 18, OpenArm 0 of 12.
   The object masks pass on all OpenArm frames; the OpenArm arms do not.
3. **The gripper does not need its own model or mask.** What it needed: the object as its own SAM2 track that competes
   with the arm per pixel, and arm prompts whose edges come from the image instead of filled convex hulls.
4. **Image quality:** with the new masks OpenArm pose-transfer PSNR rises by 0.36 dB (95 % CI 0.29 to 0.44, better in
   83 % of 172 pairs), provided the renderer still carries leftover changed pixels (shadow) with their owner.
   Using the exact masks alone lowers PSNR (RoboTwin cup −0.7 dB, stack −1.5 dB).
5. **Not reached:** the camera fit (RoboTwin still 16–35 px off), the OpenArm arm gate, held-out tasks and a new robot.

## What I built
| Item | File | What it does |
|---|---|---|
| Gold set, 30 development frames | `../r06/gold/` (`frames.json`, `clicks/*.json`, `*.npz`), sheet `../r06/gold_sheet_dev.jpg` | Frames picked from the recorded gripper state only. Labels by SAM2.1-large with my clicks, colour picks and polygons, checked on zooms |
| `gold_make.py`, `gold_consensus.py`, `gold_check.py` | `simbench/tools/` | build the set; drop RoboTwin pixels where my label and the simulator's true silhouette disagree; score a work dir per frame |
| `reconstruction.object_masks: sam2` | `sfagen/segment.py`, `sfagen/recon.py:_masks` | each object is a SAM2 track in the same session as the arms; a pixel goes to the track with the highest logit |
| `reconstruction.object_prompt: select` | `sfagen/segment.py:pick_object_mask` | at a clean view, SAM2 proposes three masks at the object's innermost point; the one that best matches the proxy is the prompt |
| `reconstruction.arm_prompt: select` | `sfagen/segment.py:arm_prompt_select` | the same rule per robot link, with the link's projected hull as the reference |
| `reconstruction.prompt_object_cut: none` | `sfagen/segment.py:arm_prompt` | no object cut in the arm prompt (the tracks compete instead) |
| `render.trust_masks: true` | `sfagen/render.py:robust_masks` | renderer uses the masks as they are (for scoring; not recommended for rendering, see H4) |

All flags default to off. With them off: guard bit-identical (28 / 28 candidates, 4 / 4 body renders), 67 / 67 unit tests.

## H1 Gold results (gate: every frame; arm recall and precision ≥ 0.95, gripper recall ≥ 0.95, object IoU ≥ 0.90)
Worst / median over the frames of each rig. "R06 rules" = the five flags above.

| Condition | Frames passing | Rig | Arm recall | Arm precision | Gripper recall | Object IoU |
|---|---|---|---|---|---|---|
| Baseline v1, fitted camera | **0 / 30** | OpenArm | 0.66 / 0.93 | 0.76 / 0.91 | 0.35 / 0.74 | 0.75 / 0.91 |
| | | cup | 0.57 / 0.84 | 0.83 / 0.95 | 0.13 / 0.71 | 0.38 / 0.44 |
| | | stack | 0.95 / 0.99 | 0.60 / 0.82 | 0.55 / 0.96 | 0.04 / 0.44 |
| | | hand-over | 0.00 / 0.16 | 0.00 / 0.68 | 0.00 / 0.00 | 0.14 / 0.20 |
| R05 rules, fitted camera | **0 / 30** | (all four rigs) | within 0.05 of the baseline rows, except cup gripper median 0.78 | | | |
| **R06 rules, fitted camera** | **10 / 30** | OpenArm (0 / 12) | 0.78 / 0.92 | 0.81 / 0.95 | 0.27 / 0.98 | 0.98 / 1.00 |
| | | cup (4 / 6) | 0.72 / 0.99 | 0.99 / 1.00 | 0.38 / 0.99 | 1.00 / 1.00 |
| | | stack (4 / 6) | 0.97 / 1.00 | 1.00 / 1.00 | 0.52 / 1.00 | 0.98 / 1.00 |
| | | hand-over (2 / 6) | 0.53 / 0.99 | 0.88 / 1.00 | 0.00 / 1.00 | 0.67 / 1.00 |
| R05 rules, TRUE camera | **0 / 18** | cup / stack / hand-over | 1.00 / 0.99 / 0.84 worst | 0.94 / 0.64 / 0.82 worst | 1.00 / 0.91 / 0.57 worst | 0.43 / 0.03 / 0.30 worst |
| **R06 rules, TRUE camera** | **18 / 18** | cup / stack / hand-over | 1.00 / 0.97 / 1.00 worst | 1.00 / 1.00 / 1.00 worst | 1.00 / 0.98 / 1.00 worst | 1.00 / 0.99 / 1.00 worst |

Files: `../r06/gold_{A,B,F,PB,PF}.json` (per frame, with the raw SAM2 and projection columns) and overlays
`../r06/gold_{A,B,F,PB,PF}.jpg` (masks filled, gold edges drawn, failed checks printed on each tile).

## H2 Where the errors came from (step 1 of the plan)
| Cause | Evidence | Fixed by |
|---|---|---|
| Object mask = "changed pixels inside the grown proxy" in the reconstruction, plus a renderer step that adds every changed pixel near a projected shape (and `object_mask_grow: 60`) | object IoU 0.03–0.64 even with the true camera; the blob covered the gripper fingers in hand-over (gripper recall 0.57) | object as its own SAM2 track + per-pixel competition |
| The R05 depth cut and repair remove gripper pixels when the proxy's depth order is wrong | true camera, hand-over: SAM2 gripper recall 0.02 on one frame | both off once objects are tracks: gripper recall 1.00 |
| SAM2 follows a mask prompt. Filled convex hulls made it include the table between links | true camera, stack: arm precision 0.57–0.85 | prompts built from SAM2's own candidates per link: precision 1.00 |
| One point on a cup selects only the inside of the cup | cup IoU 0.57–0.72 with a point prompt | candidate chosen by agreement with the proxy: IoU 1.00 |
| Camera fit | fitted-camera run: the 8 failing RoboTwin frames are the grasp frames of cup, one stack frame and hand-over episode 0 | **not fixed** |
| OpenArm: projected links sit visibly off the real links (about 15–40 px at the grippers, one episode, by eye: `../r06/openarm_projection_ep0.jpg`), black arm on a black table | OpenArm arm recall median 0.92, worst 0.78; right gripper 0.27 on one grasp frame | **not fixed** |

## H3 Camera: two planned ideas tested, neither is a fix yet
- **Refining the camera against SAM2 masks that were prompted from the projection does not work.** Those masks agree
  more with the fitted (wrong) camera than with the true one: stack 0.846 vs 0.799, hand-over 0.576 vs 0.326, cup
  0.750 vs 0.752 (mean IoU of projected hulls with the mask; n = 57, 100, 107 arm-frames). The loop would confirm itself.
- **Masks from point prompts are less biased but incomplete:** the true camera scores higher on cup (0.516 vs 0.469,
  83 % of 18) and hand-over (0.544 vs 0.462, 77 % of 30), not on stack (0.414 vs 0.404, 45 % of 47); white links are
  missed. Not built into the pipeline.
- No change to `calibrate.py` in R06. The camera sub-gate (≤ 3 px) was not attempted beyond these two tests.

## H4 Pose-transfer benchmark (re-render real frame j+k from j, PSNR against the real frame)
| Comparison | Pairs | Before | After | Difference (95 % CI) |
|---|---|---|---|---|
| OpenArm, fitted camera: baseline → R06 masks, renderer attaches leftovers | 172 | 18.71 dB | 19.07 dB | **+0.36 (+0.29, +0.44)**, better in 83 % |
| OpenArm, fitted camera: baseline → R06 masks used exactly | 172 | 18.71 | 18.59 | −0.12 (−0.23, −0.01) |
| cup, true camera: R05 → R06 masks, attach | 34 | 18.03 | 18.10 | +0.07 (−0.03, +0.20) |
| stack, true camera: R05 → R06 masks, attach | 59 | 17.66 | 17.59 | −0.07 (−0.23, +0.07) |
| hand-over, true camera: R05 → R06 masks, attach | 54 | 19.70 | 19.75 | +0.05 (−0.29, +0.37) |
| cup / stack / hand-over, true camera: R05 → R06 masks used exactly | 34 / 59 / 54 | | | −0.71 (−1.30, −0.17) / −1.54 (−2.17, −0.91) / −0.22 (−0.68, +0.23) |

Reading: the exact masks decide who owns a pixel. The shadow is neither robot nor object, and when it is left behind
the image gets worse. So the recommended setting is the R06 masks with `render.trust_masks` **off**: leftover changed
pixels still travel with the nearest owner, as before. Files: `../r06/bench_*.json`.

## Rungs that failed and stay off
| Tried | Result |
|---|---|
| Object prompt = proxy silhouette as a mask | shadow inside the mask, IoU 0.40–0.89 (true camera) |
| Object prompt = point + box | cup passes, blocks drop to 0.38–0.75 |
| Object prompt = five spread points | worse than one point (cup 0.40–0.56) |
| R05 `prompt_object_cut: depth` and `mask_repair` together with object tracks | hand-over gripper recall 0.28–0.69 (true camera) |
| `render.trust_masks: true` for rendering | PSNR loss, table above |

## What "done" still needs
1. **Your check of the gold sheet** (`../r06/gold_sheet_dev.jpg`, about 10 minutes). Magenta zones are pixels I could not
   assign on OpenArm (black gripper body on the black table, the far finger behind the can, loose cable). Tell me
   which are robot.
2. **Camera.** Still the blocker for RoboTwin (8 of 18 frames) and probably for OpenArm. Open options: a global search
   on a faster batched objective; a robot mask that does not depend on our projection (RobotSeg, needs your OK for
   the weights download) as the target; a tracked-point fit (CalibAll style, needs a point tracker).
3. **OpenArm arms.** 0 of 12 pass. After the camera, the remaining risk is black on black, where SAM2 (small and
   large) leaks and where my gold is weakest.
4. Held-out split (battery insertion, three RoboTwin tasks) and the SO-101 dataset: **not run**, the code is not frozen.

## Limits of this result
- One rater (the model) made the gold set; you have not approved it yet. OpenArm frames use a 5-px edge band (10 px on
  the three blurred "far" frames) and ignore zones, so an edge error below about 5 px on dark edges cannot be seen.
- RoboTwin arms are scored only on pixels where my label and the simulator's true silhouette agree (amendment 3,
  made after the first scoring, because my labels missed white links). The object gate does not use the simulator.
- 30 frames from 10 episodes. Rare failures are not covered; `mask_check.py` over all frames was not re-run for R06.
- The 18 / 18 result uses the true simulator camera. It shows what the masks can do, not what the pipeline does today.
- SAM2 runs once more per keyframe (image model) with the new prompts; run time was not measured.
- No effect on policy success was measured.
