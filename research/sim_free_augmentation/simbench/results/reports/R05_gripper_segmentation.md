# R05: robust arm / object segmentation – what was fixed, what was not (2026-11-16/17, Claude Fable 5.2)

Status: first-party, pending independent review. Code is **uncommitted** on `main-pipeline` (base `fbe3130`).
Protocol (frozen before any run): `../r05/PROTOCOL.md`. Run dir: `artifacts/simbench_runs/r05/` (logs, scripts, caches).
No dataset was rebuilt or published, nothing was pushed, no training, no robot use.

## Answer in four lines
1. **The gripper was missing from the arm mask because of our own prompt rule, and that is fixed.** With the prompt
   cut by depth, SAM2 covers the gripper while it is near or holding the object: 0.27–0.54 → 0.82–0.91 of gripper
   pixels on the stack task (true-camera reference); frames with an almost empty gripper mask go from 17–43 % to 0 %.
2. **This did not make the rendered images visibly better at large shifts.** On OpenArm at 11–12 cm and on all
   RoboTwin comparison episodes, the before / after frames show the same fragments.
3. **RoboTwin is blocked by the camera fit, not by segmentation.** The fitted camera is 16–45 px from truth.
   Swapping in the true camera raises pose-transfer PSNR by 1.7, 2.0 and 5.0 dB (cup, stack, hand-over); changing the
   mask rules moves it by at most 0.2 dB. The optimiser stops in a local optimum: the true camera scores higher on
   our own objective than the fitted one.
4. **Two of my planned changes failed their tests and stay off:** fixed focal length, and the geometry repair when the
   camera is not accurate.

## What changed in the code (all flags default off; OpenArm guard bit-identical; 66 / 66 unit tests)
| Flag | File | What it does | Verdict |
|---|---|---|---|
| `reconstruction.prompt_object_cut: depth` | `sfagen/segment.py:arm_prompt`, `sfagen/recon.py:scene_labels` | the arm prompt loses only pixels where an object proxy is in front of the arm, instead of a 25-px zone around every object | **keep** (see H1) |
| `reconstruction.mask_repair: 0.6` | `sfagen/recon.py:_repair` | trims arm-mask pixels > 25 px from the projected arm, removes pixels behind an object proxy, re-adds links covered < 60 % | **not accepted yet**: helps only with an accurate camera |
| `calibration.fit_focal: false` | `sfagen/calibrate.py:run` | keeps the given focal length | **rejected** (H2) |
| – | `simbench/tools/mask_check.py` | the measurement: arm and gripper coverage against the true-camera silhouette | new tool |
| – | `sfagen/tests/test_mask_repair.py` | 4 tests on 3 synthetic URDFs (rotating jaw, no gripper block, two arms with fingers), 640 and 256 px | pass |

Not done, by decision before the runs: a shadow rule for the object mask (it would also delete dark or grey objects).

## H1 Masks (RoboTwin: reference = URDF silhouette through the true camera; all pilot episodes)
Gripper coverage of the final arm mask, median, near / holding. A = baseline, B = prompt cut by depth + repair, same
(fitted) camera and same reconstruction.

| Task group (arm-frames near / holding) | A | B raw SAM2 | B final | Gripper < 20 % covered, A → B | Mask px > 10 px outside the arm, A → B |
|---|---|---|---|---|---|
| cup, left arm (30 / 45) | 0.46 / 0.48 | 0.80 / 0.81 | 0.73 / 0.74 | 17 / 20 % → 0 / 0 % | 0.1 % → 3.9 % |
| cup, right arm (41 / 55) | 0.54 / 0.30 | 0.88 / 0.54 | 0.85 / 0.66 | 32 / 40 % → 0 / 5 % | 0.9 % → 1.0 % |
| stack LL (37 / 46) | 0.30 / 0.28 | 0.88 / 0.86 | 0.87 / 0.86 | 30 / 35 % → 0 / 0 % | 8.5 % → 10.3 % |
| stack LR (36 / 45) | 0.45 / 0.48 | 0.88 / 0.80 | 0.82 / 0.82 | 3 / 13 % → 0 / 0 % | 6.8 % → 9.6 % |
| stack RL (86 / 120) | 0.30 / 0.35 | 0.89 / 0.80 | 0.86 / 0.84 | 19 / 30 % → 0 / 0 % | 8.0 % → 8.7 % |
| stack RR (18 / 23) | 0.27 / 0.27 | 0.91 / 0.88 | 0.86 / 0.88 | 17 / 43 % → 0 / 0 % | 14.2 % → 14.4 % |
| hand-over (159 / 337) | 0.00 / 0.00 | 0.45 / 0.02 | 0.36 / 0.01 | 81 / 92 % → 47 / 64 % | 0.0 % → 0.8 % |
| OpenArm Pringles, demos 0,3,5,8 (41 / 107), reference = fitted camera, not truth | 0.57 / 0.57 | 0.68 / 0.69 | 0.73 / 0.70 | 0 / 6 % → 0 / 0 % | 0.6 % → 0.6 % |

- Target ">= 0.8 near and holding" is met on the four stack groups, not on cup (0.66–0.85) and not on hand-over.
- The gain comes from the prompt fix (column "B raw SAM2"). The repair step lowers coverage slightly where the fitted
  camera is off (cup left: 0.80 → 0.73) and lets more background into the mask (0.1 % → 3.9 %).
- Files: `../r05/mask_{A,B}_{robotwin,openarm}.json`, overlays `../r05/mask_{A,B}_robotwin.jpg`.

## H2 Camera with fixed focal length – rejected
| Rig | Baseline fit (median px from truth) | `fit_focal: false` |
|---|---|---|
| place_empty_cup | 25.8 | 35.0 |
| stack_blocks_two | 15.8 | 13.1 |
| handover_block | 35.3 | 45.5 |

Diagnosis (`../r05/calib_multistart_handover.log`, `artifacts/simbench_runs/r05/logs/calib_diag.py`): on our own
objective the **true** camera scores higher than the fitted one on cup (0.491 vs 0.429) and hand-over (0.584 vs
0.500), and the same on stack (0.628 vs 0.633). So the objective is usable and the search is the weak part. A
multi-start trial on hand-over (6 starts, focal fixed) chose a solution 14.7 px from truth instead of 35.3 px; the
objective there (0.558) is still below the truth value, so better solutions exist.

## H3 Images – not supported
Same candidates, baseline masks (A) vs R05 masks (B): `../r05/compare_robotwin_A_vs_B.jpg` (9 episodes),
`../r05/compare_openarm_A_vs_B.jpg` (6 episodes, shifts 11.3–11.8 cm), zoom `../r05/compare_openarm_zoom.jpg`.
- RoboTwin cup and hand-over: fragments in both; B is not better (hand-over episode 1 looks worse).
- OpenArm at 11–12 cm: the gripper is smeared and split in both. In demo 8 the repair pulled a piece of blue tape
  from the background into the arm.
- Single rater (me), not blinded. Pixel changes A → B are small: 165–562 px per edited frame on cup / stack,
  612–1,961 px on OpenArm.

## H4 No regression – holds
- OpenArm held-out pose transfer (episodes 0,2,3,4,6,7,9; 172 pairs): B − A = −0.001 dB, 95 % CI [−0.018, +0.017]
  (`../r05/bench_openarm_A_vs_B.json`). No loss, and no gain on this metric.
- Flags off: `guard_openarm.sh r05_flags_off` passes (28 / 28 candidates, 4 / 4 body renders bit-identical).

## H5 Generality
- Unit tests on three synthetic robots pass (design uses only URDF link ownership; no names; pixel constants scale
  with image width).
- **Held-out RoboTwin tasks were not run.** The protocol runs them once on frozen code, and the code is not frozen:
  the repair is not accepted and the camera problem is open.

## Privileged ablation: the same pipeline with the TRUE camera (evaluation only, never sfagen data)
Configs `simbench/configs/r05/*_PA.yaml` (old mask rules) and `*_PB.yaml` (R05 rules); true head camera and table
height 0.74 m; 2 episodes per task. This answers "what limits RoboTwin image quality: the masks or the camera?".

**Pose-transfer benchmark** (re-render real frame j+k from j, PSNR against the real frame;
`../r05/bench_robotwin_camera_vs_masks.json`):
| Task (pairs) | Copy source frame | Fitted camera, old masks | Fitted camera, R05 masks | True camera, old masks | True camera, R05 masks |
|---|---|---|---|---|---|
| place_empty_cup (34) | 12.98 dB | 16.34 | 16.39 | 18.02 | 18.03 |
| stack_blocks_two (59) | 13.54 | 15.66 | 15.63 | 17.66 | 17.66 |
| handover_block (54) | 14.63 | 14.76 | 14.93 | 19.75 | 19.70 |

- **Camera: +1.7, +2.0 and +5.0 dB** (true better than fitted in 75–85 % of pairs). On hand-over the fitted-camera
  render is barely better than copying the source frame.
- **Masks: −0.04 to +0.17 dB.** On this metric the mask rules do not matter.
- Only 2 episodes per task, so no confidence intervals; the camera effect is large and has the same sign on all three.

**Masks with the true camera** (`../r05/mask_P{A,B}_robotwin.json`), gripper coverage near / holding:
| Task | Old rules | R05 rules | Arm IoU, old → R05 | Links repaired |
|---|---|---|---|---|
| cup, left arm | 0.38 / 0.50 | 0.94 / 0.96 | 0.69 → 0.84 | 36 |
| stack LL | 0.50 / 0.37 | 0.91 / 0.88 | 0.66 → 0.75 | 193 |
| hand-over | 0.19 / 0.40 | 0.28 / 0.46 | 0.68 → 0.71 | 654 |

- With an accurate camera the R05 rules reach the 0.8 target on cup and stack, and the repair no longer hurts
  (background in the mask 0.2 % and 5.7 %).
- Even with the true camera, the **old** prompt rule leaves the gripper at 0.19–0.50: the prompt rule was a real bug.
- Hand-over stays at 0.28 / 0.46. Part of that is the reference: it counts gripper pixels hidden behind the 20 cm
  block as "should be covered". Without true object masks I cannot separate real occlusion from missed pixels.
- With the true camera the estimated video lag is 0 to −0.25 frames; with the fitted camera some episodes came out
  at −3.0 frames, so the camera error also corrupts the lag estimate.

## What this means for "the gripper must be segmented no matter what"
The mask can now be made right, but three other things decide the image, in this order:
1. **Camera fit** (RoboTwin). Until it is within a few pixels, prompts, depth order and repair all use a shifted
   projection.
2. **The warp itself at large shifts.** When the arm is re-posed by 10 cm or more, the gripper is seen from another
   side; moving 2-D pixels cannot show surfaces the source frame never saw. A complete mask does not fix this.
3. **The object mask** still takes shadows and, with a shifted camera, arm pixels (hand-over: one large "object" blob).

## Recommendation (one decision each)
1. **Turn on `prompt_object_cut: depth`** in the next pipeline version. It is the only change with a clear measured
   gain and no measured loss. It changes OpenArm output, so it needs your approval.
2. **Fix the camera search next** (multi-start with selection by the objective, then check against truth on the
   three rigs). This is the blocker for RoboTwin, and for every new robot.
3. **Keep `mask_repair` off** until the camera is accurate; re-test it then.
4. **For large shifts, decide between a shift cap and a different renderer for the gripper.** I have no fix for that
   inside segmentation.

## Not checked
- Held-out tasks and other embodiments (LIBERO Franka, SO-101).
- Whether better masks change policy success.
- OpenArm accuracy against a true reference (there is none; a hand-checked gold set was not built).
- Prompt fix without the repair, rendered (B always had both on).
