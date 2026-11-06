# v4B experiment specification: does position-only augmentation improve far-position grasp prediction?

**Status:** FROZEN before any v4B training run. It is committed together with the code, configs and pre-training audits. Any later change is a dated amendment at the end of this file; earlier text is not edited.

**Roadmap position:** v4B is the learning-usefulness workstream. The other workstreams remain open and are not touched here (section 10):
* v4A: three-camera geometry;
* v4C: interaction-aware augmentation vs strong G+;
* v4D: force-conditioned generation.

**Scope restrictions:**
* Body camera only. Three-camera training stays blocked by the v3.2 visual-validity defects.
* Offline open-loop action prediction only. No simulator, no robot, no CAN, no pushes.

## 1. Question and hypotheses

**Question:** does simulation-free initial-object-position augmentation (v3.1 corrected G+) improve SmolVLA grasp-action prediction on real recordings whose initial Pringles position is underrepresented by the 10 source demonstrations?

| id | hypothesis | tested by |
|---|---|---|
| A | Position-only augmentation improves far-position grasp prediction. | R+Position vs R, primary metric (section 6.1) |
| B | Handover-only augmentation degrades handover prediction. | R+Handover vs R, handover error, all 40 recordings |
| C | Dataset composition explains part of the earlier null/negative v3 result. | Descriptive only (section 6.4) |
| D | Image artifacts could explain a remaining gap. | Diagnostic only (section 6.5); no artifact-free control exists, see section 3.4 |

### Historical baselines (not modified; different scopes, never pooled)

* **v3 primary ten-recording screen:** R 0.3341456, R+G+ 0.3285117, R+C 0.3268559.
* **v3.1 40-recording re-evaluation:** R 0.38107, R+G+ 0.38727, R+C 0.38814.
  * No aggregate augmentation benefit.
  * C is not better than G+.

## 2. Evaluation data status

* **Held-out pool:** `ethanCSL/openarm_pringles_lerobot_real_v00@2c1bc860347c44fc0fbae7567469d36ee49baa8d`, recordings 10–49.
* **Development data, not a test set.** Recordings 10–49 have already been used to evaluate v3 and v3.1. They are not an untouched publication test set.
* **Any positive v4B result is a development screen.** It must be confirmed on new real recordings, collected and locked before inspection.
* **Leakage check (B0):**
  * Every training episode (10 real, 71 synthetic) is at least 0.0916 rad (L∞, first 60 frames) away from every pool 10–49 recording.
  * No duplicate state hashes.

## 3. Data (Task B0 audit, run before this spec was frozen)

### 3.1 Sources and frozen inputs (read-only)

* **Real:** `sfa_v31_R`, the 10 real_10 demonstrations from `ethanCSL/openarm_pringles_lerobot_real_10_v00@cebdee98`. Episode i is source i, and state equals the source exactly.
* **Synthetic:** `sfa_v31_Gp_matched_evidence_complete`, 92 corrected-v3.1 G+ episodes.
  * These are body-camera layered-warp renders with exported 16-D state/action.
  * The export was made at commit `50a5079`.
* **Frozen file hashes:** the sha256 of every file of both datasets, plus the manifests, is in `configs/frozen_inputs_sha256.json`. `tests/test_v4b.py` re-checks them.
* **Generator:** v3.1 G+. Bounded smoothstep tails give an identical frame-0 state. Hard construction/chronology gates apply. Residual-model uncertainty is flagged. Only evidence-complete episodes are used.

### 3.2 Measured augmentation classes (`scripts/audit_families.py` → `outputs/v4b/family_audit.{json,md}`)

**Classification is measured from the candidate arrays,** against the identity candidate (which equals the recorded source, max |diff| 0). Family names are not trusted.

**Window definitions:**
* *init*: standing-can displacement at frame 0;
* *approach*: max arm-joint change on [0, rc−5);
* *handover window*: [lc−5, ro+5), covering arm-joint change, can-pose change and gripper-event shift.

**Rules:**
* *position_only*: init ≥ 5 mm and handover window unchanged (≤ 1 mrad, ≤ 1 mm, no event shift).
* *handover_only*: init ≤ 1 mm, approach unchanged, handover window changed.
* *both*: init ≥ 5 mm and handover window changed.
* *other*: anything else.

**Accepted v3.1 G+ candidates, by v3 family:**

| v3 family | position_only | handover_only | both | other |
|---|---|---|---|---|
| A_original_geometry | 0 | 69 | 0 | 17 (`approach_above`/`approach_side`: these also change pre-grasp motion) |
| B_original_contact | 0 | 43 | 0 | 0 |
| C_new_initial_position | 68 | 0 | 0 | 0 |
| D_new_position_plus_handover | 0 | 0 | 23 | 0 |

**Frozen evidence-complete dataset, by source:**

| class | ep0 | ep1 | ep2 | ep3 | ep4 | ep5 | ep6 | ep7 | ep8 | ep9 | total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| position_only | 5 | 5 | 8 | 8 | 0 | 3 | 0 | 5 | 6 | 6 | 46 |
| handover_only | 4 | 4 | 3 | 3 | 1 | 4 | 0 | 4 | 2 | 2 | 27 |
| both | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 6 |
| other | 2 | 1 | 0 | 2 | 0 | 2 | 0 | 2 | 2 | 2 | 13 |

**Documented limitations:**
* No evidence-complete position-only candidates exist for sources 4 and 6.
* Handover-only has only 27 episodes.
* The joint-change class (*both*, 6 episodes over 6 sources) is too small for a budget-matched arm.
  * **R+Both is therefore defined as a matched mixture of the two single-factor pools** (section 4).
  * It is not the v3 family-D episodes.
  * No episodes were fabricated to fill gaps.

### 3.3 Validity audit of every sampleable synthetic episode (`scripts/audit_dataset.py` → `outputs/v4b/dataset_audit.json`)

This covers 46 position-only and 25 handover-only episodes.

| check | position_only (46) | handover_only (25) |
|---|---|---|
| frozen LeRobot rows == manifest sha256 == candidate arrays | 46 | 25 |
| v3.1 hard label gates, IK, targets, verifier | 46 | 25 |
| exported-action gripper chronology (rc < lc < ro), same number of transitions as source | 46 / 46 | 25 / 25 |
| residual-uncertain flag | 0 | 0 |
| receiver alignment "ok" (not UNKNOWN) | 46 | 25 |
| frozen v3.1 image QA | 46 | 25 |
| state differs from source (not a copy) | 46 | 25 |
| pre-grasp object–image alignment (see below) | median 2.7 px; max 10.5 px; 1 episode > 10 px | 0 px |
| pre-grasp body frames identical to the source | 0 % | 100 % (by construction: same pre-grasp motion) |

**How the alignment row is computed:** it is the red-can centroid minus the projected candidate can centre, relative to the same quantity on the real source. All x−6 cm relocations have 5.4–10.5 px residuals (likely perspective change of the visible can face plus render error).

**Visual inspection** (`contact_sheet_position.jpg`, `contact_sheet_handover.jpg`, `artifact_zoom.jpg`) shows known artifacts in position-only renders:
* a faint dark ghost ring/shadow at the can's original location;
* a dark seam across the can lid near grasp, e.g. ep2 `init_x-3`, `init_y-6` at t≈98;
* reduced label texture on the relocated can.

These are present in all position arms and are not repaired in v4B.

### 3.4 Artifact control is unavailable

The v3.1 identity "re-render" is an exact copy of the source video (PSNR 138 dB, i.e. identical). It therefore cannot serve as a same-trajectory render-artifact control. Building such a control requires renderer changes (v4A), so hypothesis D is diagnostic only.

## 4. Conditions and frozen selection (`sfa4b/mixture.py`, `configs/v4b.json`)

* **Shared synthetic sources:** S = {0, 1, 2, 3, 5, 7, 8, 9}. These are the sources that have both position-only and handover-only episodes.
* **Episodes per source:** k_E = min(#position_only_E, #handover_only_E) = {0:4, 1:4, 2:3, 3:3, 5:3, 7:4, 8:2, 9:2}, i.e. 25 episodes.
* **Which episodes:** chosen by a seeded permutation (seed 20261106) of the key-sorted list per source and class. The choice does not depend on outcomes.

| arm | synthetic pool (distinct episodes per source = k_E unless noted) |
|---|---|
| R | none |
| R+Position (`RpPos`, **primary**) | first k_E position-only |
| R+Handover (`RpHO`) | first k_E handover-only |
| R+Both (`RpBoth`) | ⌈k_E/2⌉ position + ⌊k_E/2⌋ handover (alternating which family gets the odd one by source order) |
| R+PositionAll (`RpPosAll`, **secondary**, unmatched distinct-episode budget) | all 46 position-only |

**Sampling (common random numbers across arms):**
* **Batch composition:** each batch has 16 rows.
  * Augmented arms: exactly 8 real and 8 synthetic.
  * R: 16 real.
* **Synthetic rows:** source uniform over S, then frame uniform within that source's pool for the arm.
* **Real rows:** source drawn from arm-specific weights, then frame uniform within the source.
  * R: weights 1/10 each.
  * Augmented arms: w_E = (0.1 − 0.5·s_E)/0.5, i.e. 0.075 for E ∈ S and 0.2 for sources 4 and 6.
  * Result: **every source episode receives exactly 10 % of all training rows in every arm** (equal source exposure; tested).
* **Shared draws:** for a given (seed, step), all augmented arms draw identical real rows and identical synthetic uniforms.

**Held identical across arms:**

| setting | value |
|---|---|
| pretrained checkpoint | `lerobot/smolvla_base` (snapshot 5e8d12a6; sha256 recorded) |
| trainable parts | SmolVLA defaults (vision encoder frozen, expert-only + state projection) |
| optimizer | AdamW, lr 1e-4, betas (0.9, 0.95), wd 1e-10, grad-clip 10 |
| scheduler | cosine with 1000 warmup steps (as in v3, so the whole screen is inside warmup) |
| batch | 16 |
| horizon | chunk 50 / n_action_steps 50 |
| preprocessing | no image augmentation; `body_cam → camera1` |
| updates | 1000 |
| seeds | 0, 1, 2 (`--seed` and sampler seed; paired across arms) |
| normalisation | the **real-only** statistics of `sfa_v31_R`, for every arm |

**Normalisation differs from v3:** v3's augmented arms used mixture statistics. This is a deliberate v4B correction (tested: saved normaliser == real-only stats).

**Training loop:** the official `lerobot.scripts.lerobot_train.train`, with only the dataset view (`TwoPart`) and the batch sampler substituted (`scripts/train_v4b.py`).

**Runs:** 5 arms × 3 seeds = 15. The v3 checkpoints are untouched and are evaluated only as references.

## 5. Evaluation (`scripts/eval_v4b.py`)

* **Frozen v3.1 evaluator components:**
  * pool revision as above, recordings 10–49 (40);
  * every 5th frame, 50-step chunks, 1,524 chunks;
  * one N(0,1) noise tensor per (episode, frame): seed 20261105 + 1000·e + t;
  * bf16 autocast;
  * each checkpoint's own processors;
  * fixed language string.
* **Checkpoints evaluated:** all 15 v4B checkpoints, plus the 9 v3 checkpoints as references (`v3_R`, `v3_RpGp`, `v3_RpC`).
* **Image conditions** (states, targets, language and noise held fixed):

| condition | definition |
|---|---|
| original | recorded body_cam frame |
| blank | mid-grey 128 |
| wrong_episode | same relative time in the held-out recording whose initial lid is farthest (v3.1 partner; real image, out-of-distribution image/state pairing) |
| object_only | red-can hulls within 150 px of the lid track + lid disk (1.15 × equivalent radius), dilated 5 px, **minus** gripper pixels → grey |
| background | the object mask translated (nearest offset ≥ 100 px) to a spot clear of object, robot links and grippers; same pixel count. Equal-area disk fallback in 151/1524 frames; 3 frames have an empty object mask |
| gripper_only | FK silhouettes of both hands and four fingers through the nominal body camera, dilated 15 px |

* **Mask files:** `outputs/v4b/masks/` holds the masks; `mask_check.jpg` holds overlays.
* **Mean mask area:**
  * object 4.0 %, background 3.6 %, gripper 18.6 % of the image.
  * The gripper mask is large because of the dilation and the inherited body-camera offset.

## 6. Metrics and decision rules (`scripts/score_v4b.py`; frozen)

### Common definitions

* **Per-step error:** |prediction − recorded action| over the 14 arm joints, divided by the per-joint action std of real_10 episodes 0–9, averaged over joints.
* **Phase labels:** each chunk step k gets the phase of its own target frame t+k:
  * approach [0, rc−5);
  * grasp [rc−5, rc+15);
  * carry [rc+15, lc−5);
  * handover [lc−5, ro+5);
  * release [ro+5, T).
  * Events come from the held-out gripper commands. These are command-derived proxies, not measured contact.
* **Per-recording value:** the mean over that recording's steps in the phase.
* **Phase-balanced:** the mean of a recording's phase means.
* **Arm value:** seeds are averaged per recording first, then recordings are averaged.
* **Near/far bins** (v3.1 rule):
  * a recording is *far* if its clicked initial lid is more than 40 px from every source initial click, otherwise *near*;
  * this gives 15 far and 25 near recordings.
  * *far_covered_RpPos*: a far recording within 20 px of a projected RpPos generated initial position.
* **Uncertainty:** paired recording-level bootstrap (10,000 resamples, seed 20261106) of the relative change 1 − mean(arm)/mean(R).
  * Per-seed paired values are reported alongside.
  * The bootstrap does not model retraining variance.

### 6.1 Primary (hypothesis A)

* **Metric:** far-position grasp error, R+Position vs R.
* **Screening PASS requires all three:**
  1. relative improvement ≥ **5 %**;
  2. 95 % CI lower bound **> 0**;
  3. all-40 handover error degradation **≤ 2 %** (point estimate).

The same rule is applied and reported for RpHO, RpBoth and RpPosAll. These are secondary; only RpPos tests hypothesis A.

**Stability statement:** the primary result is called **unstable** if the three per-seed paired far-grasp changes do not share the sign of the pooled estimate. An unstable result is reported as "not established", whatever its CI.

### 6.2 Secondary

* near-position grasp error;
* approach, carry, handover and release errors;
* phase-balanced error (all 40, far, primary-10);
* gripper transition timing (first close/open transition within the first 25 chunk steps, |predicted − recorded| frames);
* far-covered vs far-uncovered grasp error.

### 6.3 Hypothesis B

* **Established** if R+Handover's all-40 handover error is higher than R's and the 95 % CI of the relative change lies entirely on the worse side.
* Otherwise **not established**.

### 6.4 Hypothesis C (descriptive; no causal claim)

Composition is examined by contrasting two groups on all 40 recordings:
* the v4B arms (50 % real, equal source exposure, real-only normalisation);
* the v3 references (`v3_RpGp`: 91.7 % synthetic rows, mixture normalisation).

This contrast is confounded with generator version (v3 vs v3.1 G+), sampler and normalisation. It can motivate, but not establish, a composition explanation. RpPos vs RpHO at equal budget is the cleaner within-v4B composition contrast.

### 6.5 Hypothesis D and visual dependence (B4; diagnostic)

* **What is reported:**
  * the relative error increase of each condition vs the model's own original images (approach, grasp, handover, phase-balanced; all 40 and far);
  * the prediction change;
  * the *object − background* grasp contrast with CI.
* **Interpretation rule:**
  * Visual dependence and synthetic-data usefulness are separate quantities.
  * A model that depends more on the object region is not evidence that its training images were valid.
  * Artifacts are not claimed as a cause, because no artifact control exists (section 3.4).
* **No invalid RGB/action training pairs are constructed.**

### Not claimed under any outcome

* real-robot success;
* closed-loop performance;
* physical/force validity;
* few-shot (< 10 demos) benefit;
* publication-level generalisation.

### Small-sample warning

* The far set has 15 recordings, and only those covered by generation can plausibly benefit.
* Seeds share data.
* If the result is within noise it is reported as such.

## 7. Smoke tests (completed before freezing)

| test | result |
|---|---|
| dataset readback | `audit_dataset.py` all exact |
| model init and 20-step training (R, RpPos, RpBoth) | rc 0 |
| saved normaliser == `sfa_v31_R` stats | max diff 2e-8 |
| sampled batches | 8/8 real/synthetic, correct pools |
| six-condition evaluation readback on 4 recordings, and scorer end-to-end on the smoke predictions | done; these numbers are discarded |
| unit tests | `tests/test_v4b.py` 5/5 |

## 8. Saved artifacts

**Committed:**
* configs, `frozen_inputs_sha256.json`;
* audits (`family_audit`, `dataset_audit`, contact sheets, `mask_check`);
* `train_runs.json`: git SHA, seeds, wall time, GPU usage, losses;
* per-run `*_mixture.json`: schedule sha256, pools, weights;
* training logs;
* `checkpoint_hashes.json`, `scores.json`, `summary.md`, RESULTS.md.

**Local only (sha256 recorded):**
* model checkpoints (`outputs/v4b/train/`);
* full prediction archives (`outputs/v4b/predictions/`);
* masks.

## 9. Codex coordination

* **Codex is auditing this protocol independently.** Codex review files are never modified by this workstream.
* **Before GPU training:** check whether a Codex v4B audit file has appeared in the repository root or `artifacts/`. If it reports a critical leakage or fairness problem, fix it before training and record the fix as a dated amendment below.
* **After results:** no metric or threshold changes. Additional analyses are labelled exploratory.

## 10. Preserved roadmap (not executed here)

* **v4A:**
  * Fix the ep6/frame91 wrong object copy and the ep0/frame85 texture tearing.
  * Shared continuous object/attachment clock; explicit object–camera SE(3); shared cylinder/lid geometry; depth confidence and full pixel provenance.
  * Calibration and synchronisation validation; optional Video Depth Anything and VGGT evaluation; locked 24-view real-camera replay.
  * Three-camera training stays blocked until this passes.
* **v4C:** strong G+ vs interaction-aware C on receiver grasp position/orientation, transfer timing and object motion, with matched targets, budgets and visuals. G+ is never weakened.
* **v4D:** Battery insertion with independently validated external-force estimates. These must change candidate trajectories (FACTR-2/NEXT, OpenArm telemetry audits). No invented torque; no robot/CAN without separate authorization.

## 11. Commands (from `research/sim_free_augmentation/v4`)

```bash
PY="env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python"
$PY scripts/audit_families.py && $PY scripts/audit_dataset.py
CUDA_VISIBLE_DEVICES= $PY -m unittest discover -s tests -v
$PY scripts/run_v4b.py --lanes 4
$PY scripts/eval_v4b.py --train_dir outputs/v4b/train
$PY scripts/eval_v4b.py --train_dir ../v3/outputs/train --prefix v3_
$PY scripts/score_v4b.py
```

## Amendments

(none)
