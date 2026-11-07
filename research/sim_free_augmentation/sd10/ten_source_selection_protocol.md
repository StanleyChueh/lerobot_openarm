# SFA-SD10: ten-source selection protocol (source-diversity curation)

**Status: PROPOSED study — source curation only.**
* Frozen **before** any per-episode position or interaction feature was computed.
* Before writing it, I had looked at: the dataset schemas, the STEP 1 integrity and duplication results, the first four rows of the existing click CSV, and the gripper command/state traces of pool episodes 0, 1, 2, 13, 27 and 44 (to define the measured-grasp gates).
* No VLA training, augmentation, generation, robot operation or VLA score is involved.

**Dates and authorship:**
* Frozen 2026-11-07 by Claude Code on branch `research/contact-aware-augmentation`, base `96b464d`.
* Amendments must be appended, dated and justified. Earlier text is never edited.

## 1. Scope and separation from earlier protocols

**Source pool:** `ethanCSL/openarm_pringles_lerobot_real_v00` @ `2c1bc860347c44fc0fbae7567469d36ee49baa8d`, episodes 0–49.

**Reference set ("original ten"):** `ethanCSL/openarm_pringles_lerobot_real_10_v00` @ `cebdee98115ab6827a4ec2655335c536b5dd085b`. STEP 1 showed that its episodes 0–9 are the same captures as pool episodes 0–9:
* states and actions are bit-identical (the unique match);
* frame alignment is identical;
* the video is separately encoded.
See `FULL50_EPISODE_AUDIT.md` §1.

**This is a new, separately named protocol (SD10).** It does not modify or reuse as its own:
* the v4A lock (`v4a/locked_pairs.json`, its manifests and exclusions);
* v3, v4B or v4B.1 data, schedules, results or evaluation pools.

**Eligibility:**
* **v4A locked records 10, 17, 24, 31, 38 and 45 are ineligible.**
  * They are audited from the body camera only (a v4A renderer input).
  * Their wrist streams are not decoded; only packet and PTS integrity are checked.
* All other pool episodes are eligible if they pass every quality gate, including the original ten.

**Leakage statement:**
* Recordings 10–49 informed earlier development (v3 to v4B.1 evaluation).
* Any evaluation on recordings 10–49 is development evaluation, never an untouched independent test.
* Selected episodes become training sources, so they must be excluded from any evaluation pool of the future SD10 study.

## 2. Quality gates (quality first; all must pass)

| gate | definition |
|---|---|
| Q1 integrity | STEP 1 tabular checks pass. All three streams have a packet count equal to the episode length and regular PTS. Body pixels, and wrist pixels for non-locked records, decode with no black or white frames. Longest run of identical consecutive frames ≤ 3. |
| Q2 command structure | Exactly one right close, one left close and one right reopen, at frames `rc < lc ≤ ro`, defined by commanded openness crossing 0.5. The left gripper never reopens. |
| Q3 measured right grasp | Measured right openness (state, not command) has a median in [0.15, 0.80] over [rc+10, lc), with at least 10 frames in that window. The fingers are blocked by an object rather than closed empty (≈0). Measured right openness > 0.80 at the last frame (released). |
| Q4 measured left hold | Measured left openness has a median in [0.15, 0.80] over [lc+15, T), and p95 − p5 ≤ 0.10 there: a stable blocked hold to the end, with no slip to empty closure. |
| Q5 visual transfer | Manual review of keyframes: body camera, plus both wrist cameras for non-locked records. Frames: start, rc, rc+20, lc, ro+10, last. Label SUCCESS / FAIL / UNCERTAIN. SUCCESS requires all of: the right gripper lifts the can off the table; after the right releases, the can is between the left fingers; the can is not on the table or dropped at the last frame. Only SUCCESS passes. Annotator: Claude Code (model), recorded per episode. |
| Q6 localisable can | A SAM2.1 mask at frame 0 (body camera) prompted by the existing human click contains the click. Its area is within [0.25×, 4×] of the pool median. The overlay is accepted on visual QA. |

**Gripper evidence is measured joint state, not commands.** Physical contact force is **not** claimed. "Blocked closure plus visual transfer" is evidence of a successful handover in the recording, not force validation.

**Soft quality score `q`** (tie-breaking only; it never overrides a gate). It is the mean of rank-normalised values over eligible episodes of:
1. body-camera can-crop sharpness at frame 0;
2. right-wrist sharpness at rc;
3. left-wrist sharpness at lc;
4. the negative count of identical consecutive frames, summed over the three streams.

## 3. Features

**Position (primary):**
* P = (u, v): the centroid of the frame-0 SAM2.1 can mask in body-camera pixels (640×480).
* The human click (`research/dataset_diagnostics/annotations/real_v00.csv`, `verified=1`) is the prompt only. Its lid-or-body reference point varies, so it is not used as the feature.

**Metric 3D can position: NOT ESTIMATED.**
* The body-camera extrinsic is an inherited IsaacLab silhouette fit, unvalidated on held-out keypoints (`sfa/robot.py`). v4A refused it as training-only calibration.
* Image coordinates are a position proxy only.

**Interaction (secondary):**
* I = (xL, yL, zL, Δt):
  * the left TCP position at lc (forward kinematics of measured state, `model/openarm_description.urdf`, robot base frame);
  * Δt = (lc − rc)/30 s, the time from right grasp to left receive.
* Recorded but not used in the distance:
  * the right TCP pose at rc (the grasp-location proxy, mostly determined by P);
  * the right TCP at lc;
  * the inter-TCP distance at lc;
  * the measured dual-hold overlap;
  * ro − lc;
  * episode length.

**Normalisation:** each feature is divided by its population standard deviation over the eligible set.

**Distance:**

  `d(a, b) = sqrt( Δu² + Δv² + 0.25·(ΔxL² + ΔyL² + ΔzL² + ΔΔt²) )`

The interaction block contributes about 1/3 of the expected squared distance and position about 2/3 (position first).

## 4. Selection rule (deterministic, k = 10)

1. **Stage 1 — exact minimax coverage (k-center).** R* = min over 10-subsets S ⊆ eligible of max over e ∈ eligible of min over s ∈ S of d(e, s). Solved exactly as a set-cover MILP over the sorted candidate radii (scipy `milp`, HiGHS).
2. **Stage 2 — representativeness at that coverage.** Among subsets with radius ≤ R* + 1e-9, minimise Σ over e of min over s ∈ S of d(e, s) − 1e-4·Σ over s ∈ S of q_s. This is a p-median with a radius constraint. It places centres inside dense regions rather than only on the hull, which avoids selecting only extreme outliers.
3. **Stage 3 — ties.** Re-solve with a no-good cut excluding the returned subset. If an alternative reaches the same stage-2 objective within 1e-9, choose the lexicographically smallest sorted index tuple.
4. If fewer than 10 episodes are eligible, report INFEASIBLE. Do not relax gates.

**Controls and diagnostics** (none of them influences the selection):
* **Random-ten control:** `numpy.random.default_rng(20261107).choice(eligible, 10, replace=False)`, sorted.
* **Null distribution:** 10,000 random tens from `default_rng(20261108)`.
* **Greedy farthest-point** in the same metric, starting from the eligible medoid (diagnostic only).

## 5. Comparison metrics (original ten vs selected ten vs random ten)

**Coverage targets:** computed in body-camera pixels for two target sets: (a) all 50 pool can positions; (b) the 6 v4A locked positions.

**Coverage metrics:**
* covering radius (max nearest-source distance);
* mean nearest-source distance;
* fraction within 20 px and within 40 px of a source;
* source convex-hull area, and its fraction of the pool hull;
* minimum pairwise source spacing;
* the number of selected episodes on the eligible hull.

**Interaction and quality:**
* range and standard deviation of each interaction feature;
* mean `q`;
* gate pass status.

The null-distribution percentile is reported for each coverage metric.

**Interpretation limits:**
* These metrics are computed on the same pool the selection used, so they describe coverage, not generalisation.
* **No downstream VLA score is used, or may be used, to choose or amend the selection.**

## 6. Disclosure for any future use

* "10 selected from 50 collected": this supports a 10-input-demonstration augmentation claim, not a 10-demonstration acquisition cost.
* A fair future comparison would be a 2×2: {original ten, SD10} × {real-only, matched augmentation}, plus random ten, same budgets and seeds. Results should be evaluated on prospectively recorded, locked episodes (a separate authorisation), not on recordings 10–49.
* Diversity-based demonstration selection is prior art (e.g. data-scaling and data-selection work); it is not a claimed novelty.
