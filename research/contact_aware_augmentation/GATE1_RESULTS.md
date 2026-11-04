# Gate 1 pilot results: corrected CPU-only geometry verifier (Pringles real_10 eps 8, 0, 4)

**Verdict: PARTIAL.**

* The pre-registered phase-aware verifier **fails**: it false-rejects 2 of 3 real episodes.
* A post-hoc relative variant (C_rel) rejects 0/3 real episodes. It detects timing, wrong-pairing, hand-phase and
  slip/transfer errors in every or nearly every case, and non-rigid displacements partly. It **cannot** detect a
  displacement that is rigid in the hand frame, nor a constant image offset. Being post-hoc, it is exploratory until it
  is confirmed on the sealed episodes.
* Human ground truth does not exist yet. The 8 px robot-reprojection and tracker-accuracy criteria are therefore
  **PENDING**. The annotation interface and instructions are prepared.

Date 2026-11-04. Base commit `3cce98e` (Gate 0); the delivery commit is in `git log`, and script SHA-256s are in
[GATE1_METRICS.json](GATE1_METRICS.json). Companion files:
* [GATE1_FAILURE_ANALYSIS.md](GATE1_FAILURE_ANALYSIS.md)
* [GATE1_METRICS.json](GATE1_METRICS.json)
* `outputs/gate1/`
* `gate1/` (code)

**Scope kept.** CPU only. No VLA training, no diffusion or force generation, no hardware or CAN, no Hub access, nothing
pushed. No force or contact-force validation is claimed. No automatic label is used as ground truth.

## 0. Corrections adopted from Codex's Gate-0 review

| review point | how this pilot handles it |
|---|---|
| M2 (image-space TCP-offset drift) is blind to constant offsets and false-alarms on valid rotations | replaced by an explicit rigid model `u(t) = π(T_hand(q_meas(t+lag))·p)`, with p a 3D lid point in the hand frame. Both counterexamples are reproduced (§6): M2 gives 32–94 px on a perfectly rigid synthetic point; the new model gives 0.000 px |
| historical 49/47 tie and "pass is the bottleneck" cited as motivation | superseded by the matched 53/34 result; reconciled in §8 |
| leakage: thresholds fitted and tested on the same episodes | leave-one-episode-out (LOEO). Lag, reference attachment points and every threshold are frozen on the 2 development episodes, then the held-out episode and its corruptions are scored. Episodes 1, 2, 3, 5, 6, 7, 9 are **sealed**: never loaded, decoded or scored (asserted in code) |
| no window-level bootstrap precision | episode-level exact counts only (k/3 per variant); no confidence intervals are claimed |
| track perturbations only test score algebra | cross-episode pairs are real mismatched video/joint inputs through the tracker. Temporal offsets equal an end-to-end video shift for this causal, time-invariant tracker. Displacements are 3D, perspective-correct deltas on top of the real tracker output |
| need event/time-only and generic-distance baselines | E (event/time only) and Bp (generic robot–object proximity) are included |
| measured state for the robot, commands for phases | robot pose comes from `observation.state`; phases come from command (action) gripper events, named as such |
| constant offset without a boundary cue | `offset_px20_const` is a constant 20 px offset, ramped in over [grasp, grasp+10) before the scored window starts |
| human-assisted vs autonomous tracking | the tracker is causal and autonomous. It uses only the human-verified frame-0 click for initialization: no recovery from later clicks, no FK, no commands |

## 1. Data and split

* **Dataset:** `ethanCSL/openarm_pringles_lerobot_real_10_v00` @ `cebdee98`, body_cam only.
* **Pilot episodes:** 0, 4, 8, with command-defined events (frames):

| ep | right close | left close | right open | length |
|---|---|---|---|---|
| 0 | 97 | 176 | 181 | 273 |
| 4 | 87 | 163 | 168 | 250 |
| 8 | 83 | 160 | 164 | 249 |

* **Leakage caveat:** the body_cam extrinsic was fitted earlier on pool imagery that includes these episodes. This is
  therefore held-out *annotation and measurement* validation on one rig and session, not calibration generalization.

## 2. Tracking (task 3)

`gate1/lid_tracker.py` tracks the lid centre: HSV lid mask → connected component nearest a constant-velocity
prediction → ellipse centre. Every frame gets a status:
* `ok`;
* `uncertain`: border contact, area jump, low solidity, centre disagreement or jump;
* `lost`.

Only `ok` frames are ever scored. The tracker is CPU-only and uses OpenCV core (CSRT, SAM 2 and CoTracker are not
available offline here).

| ep | frames | ok | uncertain | main reason |
|---|---|---|---|---|
| 0 | 273 | 251 | 22 | lid cut by the image top at the end |
| 4 | 250 | 202 | 48 | fast transport after release (jump/low solidity), then border |
| 8 | 249 | 213 | 36 | border at the end |

Visual review of `outputs/gate1/figures/verifier_overlay_ep*.jpg` (by the model, **not** human ground truth) shows the
track on the lid in every `ok` frame inspected. Tracker accuracy is unmeasured until human clicks exist (G2).

## 3. Human annotation (task 4): prepared, not performed

* **Tool:** `gate1/annotation/annotate_gate1.py` (OpenCV GUI, blind, resumable).
* **Instructions:** `gate1/annotation/ANNOTATION_INSTRUCTIONS.md`.
* **Manifest:** `outputs/gate1/annotation/annotation_manifest.csv`. It has 137 frames: key frames, dense grasp/lift
  and take-over windows, and 18 seeded repeats.
* **Keypoints:** 5 per frame (lid centre, both orange wrist tags, both jaw midpoints).
* **Visibility codes:** visible / uncertain / occluded. **Uncertain and occluded points are excluded from ground truth.**
  With two annotators, points disagreeing by more than 6 px are excluded.
* **Evaluator:** `gate1/annotation/evaluate_annotations.py`. It currently returns **PENDING** (exit 3) because no
  annotation file exists. Its rules were fixed before any annotation exists.
* **Not exercised:** the GUI's interactive loop was not run in this session. Its state machine, manifest and ground-truth
  rule are unit-tested.

## 4. Verifier and comparison (task 7)

**Video-to-state lag** (nuisance, frozen per fold on the dev episodes; fractional, via interpolated q): −1.5, −1.5 and
−1.75 frames. The image shows the joint state of about 50 ms earlier, consistent across folds. This is the first
data-based estimate of the camera-age gap (Gate 0 U3), and it is model-dependent.

**Methods** (score = max over components of value/threshold; flagged when > 1):

| | method | components |
|---|---|---|
| A | video only | object-track acceleration outliers, pre-grasp stillness |
| Bp | generic proximity | 95th-percentile distance from the object to the nearest projected TCP |
| B | robot geometry, phase-free | two-hand rigid mixture with free per-frame assignment, plus absolute attachment |
| C | **phase-aware geometry (pre-registered)** | giver/receiver rigid residuals, absolute attachment vs dev, take-over continuity, release (object stops following the giver), command phase logic |
| E | event/time only | object motion onset vs giver close command, phase logic |
| C_rel | *post-hoc* | C without absolute attachment and take-over continuity |
| B_rel, C_rel_take | *post-hoc* | B without absolute attachment; C_rel with take-over |

**Results** (nominal camera, LOEO, 3 real originals and 81 corruptions). "Detected" counts a corruption only if its
own paired original was **not** flagged; raw counts are in the JSON.

| method | real false rejections | AUC (all) | detected | temporal (18) | cross-episode (6) | phase (9) | slip (9) | world displacement (18) | hand-rigid displacement (18) | const 20 px (3) |
|---|---|---|---|---|---|---|---|---|---|---|
| A video only | 1/3 (ep 8) | 0.51 | 0/81 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Bp proximity | 0/3 | 0.70 | 6/81 | 2 | 3 | 0 | 0 | 0 | 1 | 0 |
| B geometry | 2/3 (eps 4, 8) | 0.81 | 19/81 | 6 | 2 | 1 | 2 | 3 | 4 | 1 |
| **C phase-aware (pre-reg.)** | **2/3 (eps 0, 8)** | 0.85 | 20/81 | 6 | 2 | 3 | 3 | 3 | 3 | 0 |
| E event/time | 1/3 (ep 4) | 0.49 | 7/81 | 4 | 1 | 2 | 0 | 0 | 0 | 0 |
| *B_rel (post-hoc)* | 1/3 (ep 8) | 0.76 | 18/81 | 12 | 4 | 0 | 0 | 2 | 0 | 0 |
| ***C_rel (post-hoc)*** | **0/3** | 0.84 | **51/81** | **18** | **6** | **9** | **8** | **10** | **0** | **0** |
| *C_rel_take (post-hoc)* | 1/3 (ep 8) | 0.82 | 33/81 | 12 | 4 | 6 | 5 | 6 | 0 | 0 |

**Per-error detection rate, C_rel** (k of 3 held-out episodes):

| error | 1 cm | 2 cm | 4 cm |
|---|---|---|---|
| world-frame displacement, x / y | 1 / 1 | 2 / 1 | 3 / 2 |
| hand-frame displacement (rigid), jaw axis / approach axis | 0 / 0 | 0 / 0 | 0 / 0 |
| slip during hand-over (receiver slides down) | — | 2 | 3 |

| error | detected |
|---|---|
| temporal offset 3 / 6 / 10 frames, lag and lead | 3 / 3 / 3 each direction |
| transfer failure (object stays with the giver) | 3 |
| wrong hand phase: roles swapped / giver releases 10 frames before receiver closes / receiver window 15 frames early | 3 / 3 / 3, each also detected with the command phase-logic rule switched off (geometry only) |
| constant 20 px image offset (Codex counterexample 1) | 0 |
| cross-episode video/joint pairs | 6 / 6 |

**A vs B vs C:**
* **Video only (A) detects nothing beyond its own false alarm.** It also serves as the artifact check (§7).
* **Robot geometry (B, B_rel) adds detection of temporal and cross-episode errors.**
* **Phase awareness (C_rel vs B_rel)** adds the hand-phase errors (9 vs 0), slip/transfer (8 vs 0) and more
  world displacements (10 vs 2), at the same or lower false-rejection count.
* The phase gain is a gain over a phase-free *mixture* model on 3 episodes. It is not evidence about generated data.

## 5. Calibration sensitivity (task 8)

The body_cam model was perturbed by rotation (0.5 / 1 / 2° about each camera axis), translation (5 / 10 / 20 mm along
each axis) and focal length (±3 %), in two modes:
* **consistent:** dev and test both use the perturbed camera; the lag is kept;
* **drift:** thresholds were frozen with the nominal camera, test uses the perturbed one.

| perturbation | C_rel real false rejections (consistent / drift) | C_rel detected (of 81) |
|---|---|---|
| rotation ≤ 1°, any axis | 0 / 0 | 49–55 |
| rotation 2° about x or z | 0 / 0 | 49–55 |
| rotation 2° about y | 0 / **2** | 46 / 14 |
| translation x or z ≤ 10 mm | 0 / 0 | 50–54 |
| translation z 20 mm | 0 / 1 | 47 / 34 |
| translation y 5 / 10 / 20 mm | 1 / 0, 1 / 1, 1 / 2 | 31–49 |
| focal −3 % / +3 % | 1 / 1, 0 / 0 | 31–32 / 50–54 |

* Every false rejection under perturbation comes from the receiver-hold residual, whose threshold is about 4 px.
* The **pre-registered C** keeps its 2/3 nominal false rejections under every perturbation, rising to 3/3 in three
  cases (x 20 mm drift, y 10 mm consistent, y 20 mm drift).
* The relative residual is insensitive to small extrinsic errors because the fitted point absorbs constant errors; the
  real-data giver residual changes from 6.0 to at most 8.1 px.
* So the verifier tolerates about 1° and 10 mm of camera error on the giver hold. The receiver check needs a better
  camera-y position and focal length, or a looser, pre-registered threshold.

## 6. Reconciling the 19 px scatter with the 8 px threshold (task 5)

| quantity | median | p90 | what it measures |
|---|---|---|---|
| projected TCP at the right-close command vs human can click at frame 0, raw (43 non-sealed pool episodes) | 88.2 px | 107.2 px | different material point **and** different time |
| … after a constant pixel bias (LOO) | 18.3 | 33.8 | the Gate-0 "19 px" |
| … after a 3D world-frame offset (LOO) | 16.9 | 29.1 | |
| … after a 3D hand-frame offset (LOO) | 17.5 | 36.9 | |
| **same material point, same time:** lid track vs rigid model, giver hold (eps 0/4/8) | **6.0 / 4.9 / 7.4 px RMS** | | tracker noise + pose-varying FK/calibration error + residual lag |
| same, receiver hold | 2.7 / 1.3 / 2.7 px RMS | | |
| 1 cm at the lid during the giver hold | 13.6 px (p10–p90 8.3–16.8) | | scale of the errors to detect |

**Reconciliation:**
* The 19 px does not shrink under physically motivated 3D offset models (16.9–18.3 px). It is dominated by
  non-calibration effects: object motion between frame 0 and the grasp, grasp-height variation, and the lid-vs-body
  click point. It is **not** a calibration error and not a bound on one.
* The 8 px criterion (G1) concerns robot keypoints against FK at the same instant. It remains **untested (PENDING)**.
* The same-point, same-time lid residual (1.3–7.4 px RMS) is consistent with a dynamic error budget below 8 px. However,
  constant camera or FK errors are absorbed into the fitted point and stay unvalidated. That residual is also why 1 cm
  displacements (about 14 px) are near the detection floor, while 4 cm displacements are not.

**M2 counterexamples** (`m2_counterexamples` in the JSON):
* Adding a constant 20 px to the real track leaves M2 unchanged (blind).
* A perfectly rigid synthetic point gives M2 = 31.7–93.8 px (false alarm) and a rigid residual of 0.000 px.
* For the new model, the 20 px offset raises the giver residual by only 0.1–1.2 px. The fitted point moves by
  12.7–15.0 mm instead. A constant offset is indistinguishable from a different grasp point unless an absolute reference
  exists (FAILURE_ANALYSIS §1).

## 7. Injection artifacts (task 9)

* **Before the fix:** the video-only score A "detected" 16/18 hand-frame displacements (AUC 0.93). The cause was a step:
  the offset switched from the giver's frame to the receiver's at release.
* **After the fix:** the offset is continuous across the transfer.
* **Final state:**
  * A detects 0/81 beyond its own false alarm.
  * Per-feature AUCs of track-only artifact features (n_ok, speed p95/max, acceleration p95/max, high-frequency ratio),
    valid originals vs each injected corruption family, lie at 0.33–0.74 (n = 3 valid, so coarse). Cross-episode pairs
    reach 0.28, as expected for genuinely different videos.
  * The median paired relative change of any feature is ≤ 6.6 %, except for cross-episode pairs, which are genuinely
    different videos.
  * Every displacement and slip variant keeps its original's `ok` mask exactly (18/18, 18/18, 3/3, 9/9).
  * Temporal shifts change `n_ok` by a median of 0.6 % (end-of-track frames become unevaluable; 9/18 keep the exact
    count).
* Displacements ramp in over [grasp, grasp+10), before the scored window. The const-offset variant has no step inside any
  scored window.

## 8. Matched 53/100 vs 34/100 and historical 49/100 vs 47/100 (task 10)

Source: committed records only (`outputs/gate1/eval_reconciliation.json`, `gate1/reconcile_eval.py`). No policy was run.

| model | historical (unpaired, random unlogged poses) | matched (paired, fixed 100 poses, 330 steps) | historical per-region rates on the matched pose mix | stratified MH, historical vs matched |
|---|---|---|---|---|
| manual-10 | 49/100 (x < 0.203: 25/43; 0.203–0.27: 16/46; ≥ 0.27: 8/11) | 53/100 (21/35, 20/44, 12/21) | expected 50.9 ± 4.9 | p = 0.74 |
| generated-500 | 47/100 (6/37, 30/47, 11/16) | 34/100 (1/35, 22/44, 11/21) | expected 48.2 ± 4.5 (observed 34, z = −2.2) | **p = 0.034**, OR 2.2 |

* **Manual:** 49 → 53 is noise plus pose mix.
* **Generated:** 47 → 34 is **not** explained by the pose mix; the drop is spread over the near-base and middle bands.
  Candidate causes are not isolated:
  * a single unlogged historical run (its own repeat estimate was 45 %);
  * inference latency 209 → about 130 ms with RTC;
  * exact rest-height placement;
  * the horizon and success-window definition.
  **UNRESOLVED.** Resolving it needs repeated runs of the same checkpoint under both protocols.
* **Pairing is not what changed the verdict.** An unpaired Fisher test on the matched counts gives p = 0.010, versus 0.89
  historically.
* **Consequence for this research line:**
  * Both protocols agree that the 500 generated demos do not help, and that generated-policy failures concentrate where
    generation never started.
  * They **disagree on the failure stage.** Matched simulator ground truth puts it at the **right-hand grasp/lift** (can
    never above 2 cm in 32 vs 50 trials); pass failures are only 5 vs 11. The historical "pass bottleneck" was a
    detector artifact.
  * The Gate-0 motivation ("generation fails at the hand-over contact") is withdrawn. Errata were added to
    GATE0_AUDIT.md and GATE1_EXPERIMENT_PLAN.md.

## 9. Criteria

| criterion | status |
|---|---|
| pre-registered C: ≤ 1/3 false rejections and ≥ 2/3 detection per corruption type | **FAIL** (2/3 false rejections) |
| post-hoc C_rel: 0/3 false rejections; per type ≥ 2/3 for temporal, cross-episode, phase, slip 4 cm, transfer failure, world 4 cm (x) | met, **exploratory** (chosen after seeing results) |
| detection of rigid hand-frame displacement and constant image offset | **FAIL**: 0/18 and 0/3 (not identifiable without an absolute reference) |
| injections not trivially distinguishable | PASS after the continuity fix (video-only 0/81; injected-family artifact AUCs 0.33–0.74 at n = 3) |
| calibration tolerance | PARTIAL: robust to ≤ 1° and ≤ 10 mm (x/z); camera-y translation ≥ 5 mm, focal −3 % or 2° about y can cause 1–2 false rejections |
| G1 robot reprojection ≤ 8 px median / ≤ 15 px p90 (held out) | **PENDING** (human annotation) |
| G2 tracker vs human ≤ 6 px | **PENDING** |

**PARTIAL**, because what exists supports only a narrow feasibility statement. On these three episodes, a relative rigid
model with command-phase structure detects timing, pairing, hand-phase and transfer errors in recorded video with no
false rejection. That statement rests on a post-hoc rule, no human ground truth and n = 3.

**Next steps (not run; no approval implied):**
1. Collect the human annotations (about 1 h). Evaluate G1 and G2, and re-score on human lid points.
2. Pre-register C_rel, its floors and the lag rule.
3. Run once on the 7 sealed episodes, with Codex's thresholds (≤ 1/7 false rejections, ≥ 6/7 per corruption type).
4. Absolute attachment needs a measured reference (can geometry, pad geometry, or a grasp-point model), not inter-episode
   statistics.

## 10. Reproduction

```bash
cd ~/Stanley_ws/lerobot_openarm
R="env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 .venv/bin/python -W ignore::DeprecationWarning"
$R research/contact_aware_augmentation/gate1/run_gate1.py --out research/contact_aware_augmentation/outputs/gate1   # ~25 s, 12 workers
$R research/contact_aware_augmentation/gate1/reconcile_eval.py --out research/contact_aware_augmentation/outputs/gate1/eval_reconciliation.json
$R research/contact_aware_augmentation/gate1/annotation/annotate_gate1.py manifest --out research/contact_aware_augmentation/outputs/gate1/annotation/annotation_manifest.csv
$R research/contact_aware_augmentation/gate1/annotation/evaluate_annotations.py --annotations "research/contact_aware_augmentation/outputs/gate1/annotation/annotations_*.csv" \
   --gate1 research/contact_aware_augmentation/outputs/gate1 --out research/contact_aware_augmentation/outputs/gate1/annotation_eval.json   # exit 3 = PENDING
python3 research/contact_aware_augmentation/gate1/make_gate1_metrics.py
$R -m unittest discover -s research/contact_aware_augmentation/gate1/tests -t research/contact_aware_augmentation/gate1/tests -v
```

**Tests** (logs in `outputs/gate1/test_output/`), all OK:
* Gate 1: 17
* Gate 0: 11
* dataset diagnostics: 39

**Outputs** in `outputs/gate1/`:
* `metrics_full.json`
* `samples.jsonl` (every scored sample with components)
* `eval_reconciliation.json`
* `annotation_eval.json`
* `tracks/lid_track_ep*.npz`
* `annotation/`
* `figures/`: verifier overlays, score by family, lag estimation
