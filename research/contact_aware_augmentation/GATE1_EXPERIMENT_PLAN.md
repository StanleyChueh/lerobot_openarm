# Gate 1 plan: can we *verify* action- and contact-consistency of generated data on this rig?

**Status: plan only. Nothing in this file has been run.**
Inputs: [GATE0_AUDIT.md](GATE0_AUDIT.md), [CONTACT_OBSERVABILITY.md](CONTACT_OBSERVABILITY.md).

Gate 1 is CPU-only and read-only: no robot, no CAN, no GPU training, no Hub writes. Its only human input is about
25 minutes of image clicking.

**Why Gate 1 is a verifier, not a generator.** Every generation hypothesis below (geometry-, phase- or
force-conditioned) needs a way to reject generated episodes whose video disagrees with their actions or with
contact physics. The prior repo study showed that 500 generated demos tied 10 manual ones. They added no variation
at the contact step, and nothing measured that before training
(`research/generated_data_diagnosis/ROOT_CAUSE_RANKING.md` #2). If such a check cannot be built and shown
sensitive on real data, no augmentation claim on this rig is verifiable. That makes it the decisive first
experiment.

## 1. Task 5: 2D segmentation / keypoint-tracking proof of concept (3 Pringles episodes)

**Episodes:** real_10 eps **8, 0, 4**. All three have zero quality flags. They span the grasp workspace:

| ep | role | FK grasp proxy (x, y) m | can click (u, v) px | grasp / take-over / release frame | overlap |
|---|---|---|---|---|---|
| 8 | near base, nearest camera (largest projection error seen) | (0.126, 0.076) | (322, 353) | 84 / 162 / 164 | 2 |
| 0 | mid | (0.240, 0.027) | (372, 205) | 98 / 178 / 181 | 3 |
| 4 | far | (0.318, 0.013) | (389, 130) | 88 / 165 / 168 | 3 |

Stress cases held back for later: ep 2 (`timing_outlier`, widest y) and ep 5 (`action_spike`).

**Camera:** body_cam only. It is the only camera with a fitted extrinsic; the wrist extrinsics are blocked
(M2A). The frames are already cached locally.

### Steps

1. **Human keypoints (ground truth), about 25 min.** Annotate 8 key frames per episode = 24 frames: start,
   grasp −20, grasp, grasp +30, mid-carry, take-over, release, end. Up to 6 points per frame:
   * K1/K2: right and left wrist-disc centre (visible orange-tagged flange);
   * K3/K4: right and left jaw midpoint (between the foam pads);
   * K5: can lid centre;
   * K6: can label centre.

   Mark each point `visible / occluded`. Store it as a CSV next to `annotations/real_v00.csv`, with the same
   columns plus `keypoint` and `visibility`. A small OpenCV click tool is to be written; it reads frames through
   `tools/dataset_diagnostics.loader`.

2. **FK–camera consistency fit (CPU).** For each robot keypoint, fit a constant 3D offset in its URDF link frame
   (link7 for the discs, hand_tcp for the jaw midpoints). That is 3 parameters per keypoint; it absorbs the
   unmeasured foam-pad length (GATE0_AUDIT U7). Optionally refine the 6-DoF body_cam pose.
   * **Leave-one-episode-out:** fit on 2 episodes, report the reprojection error on the third.
   * Report it first with the camera pose fixed to the Isaac-fitted value, then with refinement. This separates
     calibration error from FK error.

3. **Object tracking (CPU).** The can is initialized from the human click at frame 0 and tracked through all
   ~250 frames.
   * **T-a:** OpenCV CSRT box tracker plus pyramidal-LK points on the lid.
   * **T-b:** HSV red-body mask plus convex hull. Known to be contaminated by orange robot tags (M1 §4c); use it only
     inside a gate around the T-a box.
   * Lost-track recovery at the take-over by re-initializing from the nearest annotated key frame. This is counted
     and reported, never hidden.
   * Optional, **needs separate approval:** SAM 2.1 video-predictor inference. It is cached locally (M2A) and runs as
     inference only, no training, but it uses the GPU. CPU tiny-model inference on 750 frames is the fallback.
     Earlier GDINO auto-masks labelled the vacuum hose as "robot" (M2A §4), so only object prompts from clicks are
     allowed: no text-prompted robot masks.

4. **Metrics** (§2): computed on the 3 real episodes and on corrupted copies (§5).

Deliverables:
* `gate1/annotations_keypoints.csv`
* `gate1/fk_fit.json` (LOEO errors)
* `gate1/tracks/*.npz`
* `gate1/metrics.json`
* `gate1/corruption_auc.json`
* overlay videos of the 3 episodes
* unit tests for every metric on synthetic tracks

## 2. Task 6: geometric and grasp-consistency metrics (FK + tracked object)

**Notation:**
* `π` = body_cam projection;
* `FK_k(q_t)` = pose of link k;
* `o_t` = tracked can keypoint (px);
* `z_t` = FK depth of the grasping TCP along the optical axis.

Pixel-to-metric conversion at the TCP depth is `Δx ≈ z_t·Δu / f`. At z ≈ 0.5–0.6 m, 1 cm ≈ 10–12 px.

**Phases** come from the action stream (GATE0_AUDIT §5). **"Blocked hold"** means the gripper is commanded
closed and the jaw stall exceeds the empty threshold.

| id | metric | definition | real-data reference |
|---|---|---|---|
| M1 | robot reprojection error | `‖π(FK_k(q_t)·p_k) − u_{k,t}‖` over visible robot keypoints (median, p90) | LOEO fit, §1.2 |
| M2 | grasp rigidity drift | during a blocked hold, `d_t = (o_t − π(TCP_t)) − (o_{t0} − π(TCP_{t0}))`, with t0 = grasp + 10 frames; report `max_t ‖d_t‖·z_t/f` in mm | expect ≤ few mm (can slips are rare; `hold_slip` flag 0 in real_10) |
| M3 | pre-contact object stillness | `‖o_t − o_0‖` for t < grasp − 20, in mm | ≈ 0: the can is untouched until the approach; the ~2 cm push happens in the last frames before the grasp (REUSED, sim) |
| M4 | contact-onset alignment | Δt between object-motion onset (‖ȯ‖ > threshold, from the track) and the gripper-close command; separately, the first frame where the projected jaw region overlaps the object box | real distribution from 3 eps (later: the full pool) |
| M5 | bimanual hand-over coherence | at take-over: (a) the object keypoint lies inside both projected jaw regions; (b) the FK TCP distance lies in the real range 0.114–0.151 m; (c) after release, the object follows the **left** TCP (M2 with the left TCP) | Gate-0 FK geometry |
| M6 | jaw-width consistency | measured stall openness while holding vs object width (Pringles R 0.40–0.43, L 0.34–0.38); a generator that changes object size must change the stall consistently | Gate-0 events |
| M7 (battery, FK only) | insertion relative pose | right-TCP-in-left-TCP pose at release over the bay; Mahalanobis distance to the real distribution (mean (2.4, 3.2, −4.3) cm, std (0.8, 1.2, 0.6) cm, angle 168–180°, 54 eps) | Gate-0 FK geometry |

**Episode-level score:** a 1 s window passes if M2 ≤ τ2 and M4 ≤ τ4 and M5 holds (at take-over windows). The
thresholds τ are set from real windows only (99th percentile) **before** any corrupted data are scored.

**Ablation required:** the same scores without FK, i.e. the object track alone (smoothness, stillness). If FK adds
nothing, the metric is a video-quality metric, not an action-consistency metric.

## 3. Task 7: calibration and hardware measurements essential for NEXT

Needed **before** a generator is evaluated. Ordered by cost; items marked ⚙ need the robot and an explicit
approval.

| # | measurement | why it is essential | cost |
|---|---|---|---|
| N1 | body_cam keypoint validation (Gate 1 §1.2, from existing frames) | without it, M1–M5 have an unknown bias that is as large as the effects being measured: the current offset to can clicks is 90 px (median), 18.6 px after removing a constant bias | 25 min of clicks |
| N2 | declared physical dimensions: can ⌀/height/mass, foam pad length/width, charger 3D size/mass, AA cell, pad height | converts px to mm on the object (not only at TCP depth); payload model for CONTACT_OBSERVABILITY §2.2 | 15 min, tape, calipers, scale; no robot motion |
| N3 ⚙ | ChArUco static capture (M2A §7 / URDF_WRIST_CHECK §5: 5–25 static poses per arm, board visible in body_cam) | validates the body_cam extrinsic independently of FK; gives the first wrist-camera hand-eye; measures FK error (F ≤ 2 cm) | ~30 min, operator teleop |
| N4 ⚙ | camera-to-joint time sync (LED or flash in view, toggled at logged host times, during a normal recording) | M4 resolution: frame age is unmeasured (HARDWARE_AUDIT §5, sync error 3) | 5 min at the next recording |
| N5 | record gains, `gripper_squeeze_tau`, code SHA and `--robot.telemetry_dir` with **every** new dataset; ask the operator whether battery v02 had a sidecar | residuals become interpretable; future datasets become force-auditable | process only |
| N6 ⚙ | Gate-2 passive telemetry bench → D2/D3/D4 → D5/GT1–GT3 (CONTACT_OBSERVABILITY §5) | **required only for the force-conditioned branch** | GATE2_HARDWARE_READINESS: ~20 min bench, then sessions |

## 4. Task 8: geometry-only vs phase-conditioned vs force-conditioned generation, against prior work

Citation status:
* "verified" = title, venue and arXiv id checked by web search on 2026-11-04;
* "repo" = cited in this repository, not re-checked;
* "memory" = from background knowledge, not re-checked (check before citing).

| hypothesis | what is generated / conditioned on | closest prior work | what this rig's data supports | prior evidence in this repo | novelty risk |
|---|---|---|---|---|---|
| **H-G geometry-only:** object/scene transforms + re-targeted arm motion, or appearance-only video edits | object pose; action trajectory transformed with the object | MimicGen (Mandlekar et al., CoRL 2023, memory); DexMimicGen (arXiv 2410.24185, verified); DemoGen (arXiv 2502.16932, RSS 2025, verified); Real2Render2Real (arXiv 2505.09601, CoRL 2025, verified: kinematic, collisions off); ROSIE / GenAug (RSS 2023, memory) | yes: FK, phases, start clicks, body_cam model | **measured failure mode:** generated_500 tied manual (47 vs 49 / 100); it gained only where its starts overlapped the evaluation, and the hand-over contact step was neither varied nor improved | **very high**; incremental on this rig |
| **H-P phase-conditioned:** generation segmented at contact phases, with per-phase constraints (rigid attachment while holding, bimanual relative pose at contact, phase timing) and a verifier that rejects violations in the *generated real-camera video* | phase labels from gripper events; per-phase geometric constraints | DexMimicGen (per-arm subtasks + coordination constraints for hand-over: phase segmentation itself is **not novel**); DreamGen (arXiv 2505.12705, CoRL 2025, verified: video world model + IDM pseudo-actions, no contact check); "Seeing Realism from Simulation" (arXiv 2605.02757, ICML 2026, verified title/abstract only) | yes: phases are clean in both tasks; M1–M7 computable; contact-phase *verification* is what Gate 1 tests | the failing step of both policies was the contact phase (ROOT_CAUSE_RANKING #2) | medium: the open gap appears to be *verified* contact-phase consistency of generated video for bimanual hand-over and insertion on a low-cost arm; a literature search found no direct hit, but it was not exhaustive |
| **H-F force-conditioned:** generation additionally conditioned on, or producing, a contact/effort channel | measured effort residual or contact label; or a generated force profile | TA-VLA (arXiv 2509.07962, CoRL 2025, verified: torque into a VLA decoder + torque prediction, real torque sensing); Dreaming the Sound of Contact (arXiv 2609.19137, verified: generated audio → force profile → force control on Franka; also used as a data engine; **closest collision**); NeuralActuator (arXiv 2607.11734, repo); force-aware world/action models FAWAM (2606.08555), ForceVLA2 (2603.15169) (verified titles only) | **no:** zero force/torque/telemetry in either dataset; telemetry never validated on hardware; battery right jaws close fully, so geometry is ambiguous exactly where force matters | none | high novelty *if* sensorless effort on a low-cost dual arm is shown to carry contact information; **untestable until N6** |

**Predictions to falsify later:**
* H-G improves approach-phase coverage but not contact-step success.
* H-P > H-G at a matched budget on the take-over / press step.
* H-F > H-P only on battery's press, where jaw position cannot even confirm the grasp.

All three comparisons need policy training: GPU, not part of Gate 0/1.

## 5. Task 9: GO/NO-GO criteria and the minimum decisive experiment

### Minimum decisive experiment: the corruption-sensitivity test (CPU, about 2 days)

On the 3 real episodes, synthesize the failure modes a generator produces, then ask whether M1–M5 separate them
from the real windows. All thresholds are fixed from real data first.

| corruption | construction (on the joint stream or the object track; video untouched) | magnitude levels |
|---|---|---|
| C1 temporal desync | shift the object track vs joints | ±3, ±6, ±10 frames |
| C2 object off-gripper | offset the object track during the hold, converted from mm at TCP depth | 1, 2, 3 cm |
| C3 grasp failure | freeze the object track at the grasp while the TCP moves | — |
| C4 cross-episode | joints of ep i with the video track of ep j | 6 ordered pairs |
| C5 phase shift | shift the gripper events vs the video | ±5, ±10 frames |

### Gate-1 criteria

**GO**, all required:

| id | criterion | threshold | rationale |
|---|---|---|---|
| G1 | LOEO robot reprojection (M1), body_cam | median ≤ 8 px, p90 ≤ 15 px | 2 cm ≈ 21 px at grasp depth; the error budget must be < ½ of the effect |
| G2 | tracker vs human clicks on visible key frames | median ≤ 6 px; lost-track rate during carry ≤ 10 % of frames; recovery at take-over counted | M2/M5 need a stable object signal |
| G3 | real episodes pass their own consistency | M2 drift ≤ 10 mm median and M3 ≤ 5 mm in all 3 episodes | the metric must not reject real data |
| G4 | sensitivity | window-level AUC ≥ 0.90 (bootstrap 95 % CI lower bound ≥ 0.80) for C2 at 2 cm and C1 at 6 frames; C3 and C4 detected in 100 % of episodes | it must detect errors at the scale of generative artefacts |
| G5 | FK adds information | the AUC of the FK-free ablation is lower by ≥ 0.10 on C1/C2 | otherwise the "action-consistency" framing is unjustified |

**NO-GO / redirect:**
* **G1 fails after the offset fit:** stop. N3 (ChArUco) is required before any further metric work.
* **G4 fails at 2–3 cm:** generated-video action consistency cannot be verified on this rig at the needed
  resolution. Then:
  * drop the "action-consistent video augmentation" claim;
  * redirect to FK-only (state-space) augmentation;
  * or redirect to the force/telemetry branch.
* **G5 fails:** reframe as a video-quality filter. That is not a contact-aware contribution.

**Force-conditioned branch:**
* NO-GO on historical data, unconditionally.
* GO to plan data collection only after the Gate-2 bench is GO, S1 is resolved, and D2–D4 show a residual noise
  floor below the candidate press signature: CONTACT_OBSERVABILITY §2.3 shows 0.031 → 0.039 rad at unchanged
  speed; a separable signal must stand out from the free-motion residual distribution at matched pose and speed.

### Effort and order

1. N2 (15 min, no robot) and the Gate-1 annotation (25 min), in parallel.
2. Gate-1 CPU implementation and analysis: about 2 days.
3. Decision.
4. If GO, request approval for N3/N4/N5 at the next recording session, and separately for the Gate-2 bench (N6).

**Stop here.** Generator training and policy comparisons (§4 predictions) are out of scope until Gate 1 is GO and
GPU work is approved.
