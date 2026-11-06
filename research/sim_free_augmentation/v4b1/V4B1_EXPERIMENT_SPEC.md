# v4B.1 specification: explaining the v4B grasp gain and its handover regression

**Status:** FROZEN before any v4B.1 training run. It is committed with the code, configs, tests and readiness evidence. Changes after the Codex pre-training audit are appended as dated amendments; earlier text is not edited.

**What this continues:**
* v4B (protocol `0d7007c`, execution `805822e`, wording `34ec1cf`).
* Codex reviews `CODEX_SIM_FREE_V4B_PRETRAIN_AUDIT.md` and `CODEX_SIM_FREE_V4B_REVIEW.md` (local, untracked).

**v4B results preserved unchanged:**
* Primary R+Position **FAIL**: far grasp +13.243 %, handover degradation 3.810 %.
* Secondary R+Both **PASS**: far grasp +9.855 %, handover degradation 0.957 % (point estimate), overall +1.887 %.

**Scope:**
* Offline, open-loop, body camera only.
* Evaluation recordings 10–49 are reused **development data**.
* No simulator, robot, CAN or pushes.

**Corrections carried from the Codex audit:**
* Effective LR schedule is **33 warmup / 1,000 decay steps**. The installed LeRobot auto-scales the configured 1,000/30,000. v4.B "inside warmup" statement was wrong.
* Source exposure is 10 % **in expectation**; realised counts were 9.51–10.48 %.
* v4B's "handover-only" pool is better described as *fixed-initial-position, later-motion* G+. 17/25 of its episodes also change the grasp window.
* 151/1,524 background masks are unequal-area disk fallbacks.
* Blank-image far-grasp gains persist (Position 9.99 %, Both 11.83 %).
* No longer training is motivated by the warmup error.

## 1. Hypotheses

| id | question | null (what we expect if false) |
|---|---|---|
| **H1** | Does generated body-camera RGB add learning value beyond the synthetic state/action trajectories it accompanies? | R+Both trained with synthetic images *missing* reaches the same far-grasp error as R+Both with generated images. |
| **H2** | Does synthetic action supervision outside the grasp phase (carry, handover, release targets) cause R+Position's handover regression? | Excluding those synthetic targets from the loss leaves the handover regression unchanged. Alternatively, it changes it only as much as an equal-size random reduction of synthetic supervision. |

## 2. Data, selection, schedules: identical to v4B (read-only)

* **Real data:** `sfa_v31_R`, real_10 `ethanCSL/openarm_pringles_lerobot_real_10_v00@cebdee98115ab6827a4ec2655335c536b5dd085b`.
* **Synthetic data:** `sfa_v31_Gp_matched_evidence_complete` (v3.1 G+, local, sha256-pinned in `v4/configs/frozen_inputs_sha256.json`).
* **Selection:** frozen v4B selection (`v4/sfa4b/mixture.py`).
  * RpBoth: 25 episodes (13 position-only + 12 fixed-initial-position later-motion).
  * RpPos: 25 position-only.
  * Sources {0,1,2,3,5,7,8,9}.
* **Sampler:** the v4B `StratifiedBatches` of the base arm, unchanged.
  * 16 rows per batch: 8 real + 8 synthetic.
  * Real-source weights 0.075 / 0.2.
  * Common random numbers.
  * **Every new arm draws bit-identical batches to the frozen v4B base-arm run of the same seed.** This is tested against the stored v4B `schedule_sha256` values.
* **Training settings, as v4B:**
  * pretrained `lerobot/smolvla_base` 5e8d12a6 (`7cd549ac…01eb`);
  * AdamW lr 1e-4, betas (0.9, 0.95), wd 1e-10, clip 10;
  * effective cosine schedule with 33 warmup / 1,000 decay steps;
  * batch 16, chunk and horizon 50, 1,000 updates, seeds 0/1/2;
  * real-only normalisation; no image augmentation; `body_cam → camera1`;
  * the official `lerobot_train.train` loop;
  * installed LeRobot 0.6.2 from `huggingface/lerobot@3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e` (PEP 610 metadata).

## 3. Conditions

**New runs** (3 paired seeds each, 15 runs; `configs/v4b1.json`):

| arm | base schedule | synthetic RGB | synthetic target loss | role |
|---|---|---|---|---|
| `RpBoth_NoSynthRGB` | RpBoth | **missing** | all targets | **H1 treatment** |
| `RpBoth_rerun` | RpBoth | generated | all targets | H1 replication control: same code path, measures run-to-run nondeterminism |
| `RpPos_PhaseMask` | RpPos | generated | **only approach/grasp targets** | **H2 treatment** |
| `RpPos_RandMask` | RpPos | generated | **random subset, same expected size as PhaseMask** | H2 matched synthetic-budget control |
| `RpPos_rerun` | RpPos | generated | all targets | H2 replication control |

**Frozen references (reused, not retrained; hashes verified):** v4B `R`, `RpBoth`, `RpPos`, seeds 0–2.

### 3.1 Missing synthetic RGB (H1)

SmolVLA's native missing-camera representation is used:
* **Pixels:** synthetic rows' `observation.images.camera1` pixels are set to the empty-camera fill (0 in [0,1]). `prepare_images()` maps this to −1, exactly as it fills absent cameras.
* **Attention mask:** `observation.images.camera1_padding_mask = False` removes those image tokens from attention.

**What is unchanged:**
* real rows, which keep their real RGB;
* tensor shapes, action targets and the policy architecture.

**Verified** (`outputs/readiness.json`):
* A synthetic row's training loss (fixed noise and time) is **exactly invariant** to its pixels when masked: max |Δ| 0.0.
* It *does* change with pixels when unmasked: 0.0103.
* Real rows' losses are unaffected: 0.0.

No source RGB is substituted for changed geometry.

**What H1 can establish:** whether generated images on synthetic rows add value *at this budget and with this renderer*, beyond the same state/action supervision trained without visual input.

**What it cannot establish:**
1. Whether *perfect* generated images would help.
2. Whether gains come from geometrically correct pixels versus any image-consistent cue.
3. Whether the image-free pathway transfers differently in a longer or larger run.

The missing-image rows also train an image-free mode that real evaluation rows never use. This is the declared counterfactual, not a perfect one.

### 3.2 Phase-restricted synthetic targets (H2)

**Mask rule:**
* For a synthetic row at dataset frame f, target step k (target frame f+k, k = 0..49) is removed from the loss iff f+k ≥ `right_close + 15`.
* That is the v3.1/v4B per-target-step grasp→carry boundary.
* `right_close` comes from that synthetic episode's own exported gripper commands.

**Implementation:**
* The mask is OR-ed into `action_is_pad`, so all 16 channels share it.
* Real rows' `action_is_pad` is never modified (tested; also logged every step).

**Gripper checks (tested on every RpPos episode):**
* The right-gripper close event lies in the supervised region.
* Left close and right open lie in the excluded region.

**Loss normalisation:** SmolVLA's loss is `sum(valid elements) / count(valid elements)` over the batch (verified numerically). Masking therefore lowers the synthetic share of the loss and raises the per-element weight of real targets.

**Budget under the frozen RpPos schedules** (`outputs/budget.json`):

| quantity | value |
|---|---|
| synthetic target steps kept by the phase mask | **33.66 %** (pooled; per seed 34.4 / 32.9 / 33.7 %) |
| synthetic rows fully masked | 59–61 % (rows starting after grasp-end) |
| synthetic share of valid loss elements | about 50 % → about 27 % |

**Matched budget control (RandMask):**
* Every synthetic target step is dropped independently with p_drop = **0.6634**, so 33.5–33.7 % are kept.
* The draw is a fixed hash of (seed, row index).
* It matches the amount of synthetic supervision while keeping its phase composition.

**What H2 can establish:** whether *which* synthetic targets are supervised, rather than how many, drives the handover regression, under these schedules.

**What it cannot establish:**
* **Input-side effects are not removed.** Fully masked synthetic rows still occupy batch slots (no gradient), and partially masked rows keep their changed input states.
* **No physical contact mechanism.**

## 4. Evaluation (unchanged v4B evaluator)

* **Evaluator:** `v4/scripts/eval_v4b.py`.
  * pool `ethanCSL/openarm_pringles_lerobot_real_v00@2c1bc860347c44fc0fbae7567469d36ee49baa8d`, recordings 10–49;
  * 1,524 chunks with frozen per-(episode, frame) noise;
  * bf16; each checkpoint's own processors.
* **Image conditions:** original, blank, wrong_episode, object_only, background, gripper_only. Masks are the cached v4B masks, read-only.
* **Evaluation images are always real RGB.** Masks are used only in training. No evaluation data or descendant enters training (v4B lineage audit; unchanged).
* **Frozen reference archives:** the v4B archives of R/RpBoth/RpPos are reused (sha256 in `v4/outputs/v4b/predictions_manifest.json`).

## 5. Metrics

As v4B:
* per-target-step phases from command events;
* arm-joint error normalised by real-training action std;
* per recording, seed-averaged, then averaged over recordings.

**Sets:** far (15), near (25), all 40.

**Metrics reported:**
* far-position grasp, near grasp;
* approach, carry, handover, release;
* phase-balanced;
* gripper transition timing.

**Contrast definition:** relative change = 1 − mean(x)/mean(ref) (positive = x better).

**Uncertainty:**
* paired recording bootstrap, 10,000 resamples, seed 20261107;
* per-seed paired values alongside.
* Conditional on the fitted seeds; not a model of retraining variance.

## 6. Primary contrasts (frozen)

* **H1:** `RpBoth` (frozen) vs `RpBoth_NoSynthRGB` on **far-position grasp error**.
* **H2:** `RpPos_PhaseMask` vs `RpPos` (frozen) on **all-40 handover error**, with far-grasp preservation vs `R`.

**Secondary contrasts:**
* NoSynthRGB vs R; RpBoth vs R;
* PhaseMask vs R; RandMask vs RpPos; PhaseMask vs RandMask; RandMask vs R; RpPos vs R;
* reruns vs frozen (noise floor);
* treatment vs rerun (sensitivity).

**Inference-time image conditions** compare two models under the *same* evaluation image condition. This is diagnostic and kept separate from the training intervention.

## 7. Decision rules (frozen; `scripts/score_v4b1.py::decisions`)

### H1

**GENERATED_RGB_USEFUL** requires all of:
* the incremental far-grasp improvement of RpBoth over NoSynthRGB is ≥ **3 %**;
* its 95 % CI lower bound is > 0;
* all three per-seed differences are > 0;
* it exceeds the replication noise |RpBoth_rerun vs RpBoth|.

**GENERATED_RGB_HARMFUL** if the improvement is ≤ −3 % and the CI upper bound is < 0.

**INCREMENTAL_VALUE_NOT_ESTABLISHED** otherwise.

**Also reported:**
* whether NoSynthRGB alone passes the v4B far-grasp part vs R (≥ 5 %, CI > 0): "kinematic supervision alone gives the screen gain";
* the retained-gain fraction (R − NoRGB)/(R − Both) with bootstrap CI.

### H2

**Conditions:**
* **P1:** PhaseMask's handover error is lower than RpPos's (CI lower > 0), AND PhaseMask's handover degradation vs R is ≤ 2 % (point estimate).
* **P2:** PhaseMask keeps far grasp vs R ≥ 5 % with CI lower > 0.
* **P3:** PhaseMask's handover error is lower than RandMask's (CI lower > 0).

**Verdicts:**
* **PHASE_SPECIFIC_INTERFERENCE_SUPPORTED** if P1 and P3.
* **HANDOVER_REDUCED_BUT_EXPLAINED_BY_SYNTHETIC_BUDGET** if P1 and not P3, and RandMask also improves handover vs RpPos with CI lower > 0.
* **HANDOVER_REDUCED_PHASE_SPECIFICITY_NOT_ESTABLISHED** if P1 only.
* **PHASE_RESTRICTION_DOES_NOT_REDUCE_HANDOVER_REGRESSION** otherwise.

P2 is reported beside the verdict.

**v4B screening rule:** the frozen rule (≥ 5 % far grasp, CI > 0, ≤ 2 % handover degradation, all vs R) is reported for every new arm as a screen. It does not replace the H1/H2 verdicts. A new PASS does not overturn the v4B primary FAIL.

### Interpretation guards

* **Causal language:** used only for a verdict reached by the frozen rule, and only within this training intervention.
* **Replication noise:** if a primary effect is smaller than its replication noise (rerun vs frozen), it is reported as indistinguishable.
* **Additional analyses:** labelled exploratory.

## 8. Potential confounders (declared)

| confounder | mitigation |
|---|---|
| GPU nondeterminism between frozen and new runs | rerun arms on the same code path; treatment vs rerun sensitivity |
| missing-image rows train an image-free mode | declared limitation (§3.1); not fixable without changing the counterfactual |
| masking shifts loss weight to real targets | RandMask matched-budget control; per-step budget logs |
| fully masked synthetic rows keep their batch slots | identical in a matched way only in expectation; disclosed |
| changed input states in partially masked rows | disclosed (input-side effects not removed) |
| fixed-initial-position later-motion pool is not a pure handover factor | H1 uses RpBoth only as a whole; no handover-factor claim |
| 15 far / 2 generation-covered recordings; reused development pool | development-only conclusions; fresh-data confirmation remains required |
| render artifacts (ghosting, seam, alignment) | H1 measures the *existing* renderer's value; no artifact-cause claim |

## 9. Execution gate

**Pre-training checks** (`V4B1_IMPLEMENTATION_READY.md`):
* data readback;
* sample-pair (schedule) equality;
* action-target identity;
* image-intervention (unit and model-level) checks;
* loss-mask boundary and gripper tests;
* loss-normalisation check;
* model-initialisation and 20-step smoke training of all arms;
* evaluation readback.

**Codex pre-training audit:** obtained before the full runs. Critical findings are fixed and recorded as amendments; non-critical ones are documented.

**Ordering:** `scripts/run_v4b1.py` writes pre-launch provenance (`outputs/prelaunch.json`) before any worker starts.

**Bounded runs:** 15 × 1,000 updates (about 16 min on 2 × RTX 4090). No longer training. No v4A/v4C/v4D training.

## 10. Commands (from `research/sim_free_augmentation/v4b1`)

```bash
PY="env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python"
$PY scripts/budget.py
CUDA_VISIBLE_DEVICES= $PY -m unittest discover -s tests -v
$PY scripts/readiness.py --smoke_dir /tmp/v4b1_smoke
$PY scripts/run_v4b1.py --lanes 4
$PY ../v4/scripts/eval_v4b.py --train_dir outputs/train --pred_dir outputs/predictions
$PY scripts/score_v4b1.py && $PY scripts/figures_v4b1.py
```

## Amendments

(none)
