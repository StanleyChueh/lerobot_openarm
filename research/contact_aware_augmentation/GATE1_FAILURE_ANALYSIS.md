# Gate 1 failure analysis

Companion to [GATE1_RESULTS.md](GATE1_RESULTS.md). Every number comes from `outputs/gate1/samples.jsonl` or
`metrics_full.json`.

Each failure is classified as one of:
* **calibration:** the camera or FK model;
* **visibility/tracking:** the object signal;
* **score model:** the verifier's assumptions;
* **identifiability:** no method can separate the case with this information;
* **implementation:** a bug, now fixed.

## 1. Pre-registered absolute attachment false-rejects real episodes (identifiability + score model)

* **Symptom:** C false-rejects eps 0 and 8, and B false-rejects eps 4 and 8, through `abs_giver_mm`, `abs_recv_mm` and
  `B_abs_mm`.
* **Cause:** the lid point in the hand frame differs between real episodes because the operator grasped the can at
  different places.
  * Dev reference points (mean of the 2 dev episodes) move by up to 18 mm along the hand x axis between folds
    (giver x 131 / 138 / 149 mm).
  * Held-out distances to them are 14–33 mm.
  * The threshold rule `1.5 × max(dev distance to the dev mean)` compares each dev episode with a mean that includes
    itself. For n = 2 that is half the inter-episode distance, so the rule underestimates real spread by about 2×.
    This is a design error in the pre-registration, recorded here rather than silently corrected.
  * Even a leave-one-out version would only re-measure one inter-episode distance.
* **Consequence:**
  * A displacement that is rigid in the hand frame is physically the same as a different grasp point.
  * Relative consistency cannot see it (0/18). A constant image offset is likewise absorbed (0/3): the giver residual
    rises by only 0.1–1.2 px while the fitted point moves 12.7–15.0 mm.
  * These cases are **not identifiable** from monocular video plus FK without an *absolute* reference. That means
    measured can and foam-pad geometry, defining where the object must sit between the jaws, or many more episodes to
    model grasp variability.
  * For generated data this matters most: "object rendered 2 cm off the grasp" is exactly this class.

## 2. Take-over continuity false-rejects ep 8 (score model)

* **Symptom:** at the take-over, the giver's and receiver's rigid predictions of the lid disagree by 45.6 px in ep 8
  (threshold 30.2 from eps 0 and 4, at 16.1 and 21.6 px).
* **Evidence:** `figures/verifier_overlay_ep8.jpg`, frames 155–179. The red (giver) and blue (receiver) circles straddle
  the lid.
* **Interpretation:** the rigid-attachment assumption breaks exactly at the contact-rich step. The can shifts or rotates
  in the hands while both grip it, so "one fixed point in each hand frame" does not hold across the overlap. This is a
  real-data phenomenon, not tracker failure (the lid track is `ok` there).
* **Consequence:**
  * Continuity must be modelled with a tolerance learned from more than two episodes, or the take-over must be scored
    with a different model.
  * C_rel drops it, and that choice is post-hoc.

## 3. Small world-frame displacements are near the noise floor (calibration + visibility)

* **Numbers, C_rel:** 1 cm detected in 2/6, 2 cm in 3/6, 4 cm in 5/6. At the lid, 1 cm ≈ 13.6 px (8.3–16.8).
* **Real giver residual floor:** 4.9–7.4 px RMS, with thresholds 8.2–9.0 px.
* **Misses concentrate in ep 4.** Its corrupted giver residual reaches only 7.4 px even at 4 cm (y), because ep 4's carry
  has little hand rotation. A world-constant offset is then almost hand-rigid, and therefore almost unobservable.
* **Consequence:** detectability depends on the motion. The verifier should report per-episode *observability* (e.g.
  the rotation span of the hold) and refuse to certify low-observability holds rather than pass them.

## 4. Slip during the hand-over: one 2 cm miss (visibility)

* **The miss:** ep 4's 2 cm slip is not detected (receiver residual 3.4 vs threshold 4.0). The tracker keeps only 16
  `ok` frames of ep 4's receiver hold (fast transport, then the border), so the window barely spans the 15-frame slip
  ramp.
* **The rest:** 4 cm slips (3/3) and transfer failure (3/3) are detected.
* **Consequence:** receiver-hold coverage is the limiting factor. The camera framing (the lid leaves the image at the end)
  and tracker recovery after fast motion are the fixes. Recovery must stay causal, without human-click re-initialization.

## 5. Calibration sensitivity concentrates in the receiver threshold (calibration + score model)

* **Where it fails:** C_rel's false rejections under perturbation come only from `res_recv`. That threshold is about
  4 px, because the dev receiver residuals are 1.3–2.7 px, with the floor at 3 px.
* **Triggers:**
  * camera-y translation of 5–20 mm (1 false rejection consistent; up to 2 drift);
  * focal length −3 % (1);
  * 2° about y in drift mode (2).
* **Robustness:** the giver residual changes little (≤ 8.1 px) under all perturbations. The rigid point absorbs constant
  error.
* **Consequence:** either measure the camera (ChArUco, Gate-0 N3) or pre-register a receiver floor justified by a
  calibration-error budget. Do not tune it on these episodes.

## 6. Video-only and event-only scores (expected, and useful as controls)

* **A (video only)** detects nothing beyond its own false alarm on ep 8 (acceleration ratio 19.0 vs threshold 13.2: ep 8
  has a fast, near-camera lift). Most corruption families leave the object track statistically unchanged by
  construction:
  * temporal shift;
  * cross-episode pairs (they are real videos);
  * phase errors (video untouched);
  * smooth displacements.
  A's failure to detect is therefore the artifact control, not a weakness of the comparison.
* **E (event/time only)** false-rejects ep 4: its can starts moving 16 frames after the close command, vs −1 and −2 in
  the dev episodes. That is real behavioural variation. E is not a viable verifier, confirming Codex's expectation that
  timing alone does not identify these errors.

## 7. Implementation defects found and fixed during the pilot

| defect | how found | fix | effect |
|---|---|---|---|
| hand-frame displacement switched frames abruptly at the giver's release → step in the track | video-only score A detected 16/18 (AUC 0.93), which should be impossible | offset continuous across the transfer | A: 0/18 beyond original; artifact AUCs ≈ 0.5 |
| displacement ramp started before the grasp, where no 3D lid estimate exists → step at the grasp | code review before the first run | ramp over [grasp, grasp+10) | — |
| the two focal perturbations (0.97, 1.03) shared one tag and were pooled | 6 rows per calibration cell instead of 3 | signed tags | separate rows |
| synthetic unit-test hand motion too slow to make 6-frame shifts and world offsets observable | 2 test failures | richer synthetic rotation and depth change | 17/17 pass |

## 8. Threats to validity that remain

1. **n = 3 episodes, one session, one rig.** Detection counts are k/3 per variant. No precision is claimed.
2. **Post-hoc variants.** C_rel, B_rel and C_rel_take were defined after the pre-registered rules failed. Their numbers are
   exploratory.
3. **Tracker tuned on the pilot.** The HSV band came from these episodes' frame-0 clicks; the flag thresholds were fixed
   a priori. The sealed episodes are the first unbiased test.
4. **No human ground truth.** Tracker accuracy, robot-keypoint reprojection (G1) and click uncertainty are unknown. A
   tracker bias that is constant within a hold would be absorbed into the fitted point and stay invisible here.
5. **Camera model leakage.** The body_cam extrinsic was fitted on pool imagery that includes these episodes.
6. **Track-level injection.** Displacement, slip and transfer failure were injected into the tracked coordinates, not the
   pixels. Cross-episode pairs and temporal shifts are the only end-to-end video inputs.
7. **Recorded-data rigidity.** The verifier passes real episodes under a rigid-grasp model. It says nothing about
   physical feasibility, force closure or contact forces. No force or contact validation is claimed.
8. **Lag as a nuisance.** The −1.5 to −1.75 frame video-to-state lag is estimated, not measured. A per-sample lag fit would
   absorb temporal corruptions, so it is frozen per fold.
