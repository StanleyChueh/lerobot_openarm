# v4B results: initial-position augmentation and SmolVLA grasp prediction

**Status:** Executed — Pending Independent Codex Review.

* **Protocol:** `V4B_EXPERIMENT_SPEC.md`, frozen and committed at `0d7007c` before any training. No amendments.
* **Scope:** offline, open-loop, body camera only. No simulator, no robot, no CAN, no pushes.
* **Evaluation data:** pool 10–49 is **development data**. It has been used before and is not a publication test set.

## Summary

1. **Primary screen (R+Position vs R): FAIL under the frozen rule.**
   * Far-position grasp error improves by **+13.2 % [95 % CI +7.0, +20.9]**. All three paired seeds agree (+13.4 / +13.1 / +13.2 %).
   * Handover error degrades by **3.8 % [2.8, 4.9]**, which exceeds the frozen 2 % guardrail.
2. **R+Both (secondary arm) passes the same rule.**
   * Far grasp **+9.9 % [+6.1, +13.6]**; seeds +10.9 / +9.6 / +9.1 %.
   * Handover −1.0 % [−2.5, +0.6], within the guardrail.
   * Overall phase-balanced error improves **+1.9 % [+1.1, +2.7]**.
   * This is the first augmented condition in this project whose 40-recording aggregate CI excludes zero on the better side. R+Both is a secondary arm, so this is a hypothesis for confirmation, not the primary result.
3. **Hypothesis B (handover augmentation hurts handover) is not established.**
   * R+Handover: handover −0.9 % [−3.7, +1.9]; seeds +1.1 / +2.8 / −1.1 % degradation.
   * R+Handover gives no grasp benefit: +1.7 % [−0.2, +3.3].
4. **Composition (descriptive):** the old v3 R+G+ models have about the same far-grasp gain (+13.0 %) but lose 10.2 % on handover. The 50 %-real, equal-exposure v4B arms lose 0.9–4.1 %. This contrast is confounded (section 4).
5. **The grasp gain is not specific to generation-covered positions.**
   * Far-uncovered recordings (13): +10.7 %.
   * Near recordings (25): +6.4 %.
   * Only **2** far recordings are within 20 px of an RpPos generated position, too few for a coverage claim.

## 1. Executed runs

| item | value |
|---|---|
| git HEAD at launch | `0d7007c` (frozen protocol; v4 tree clean) |
| runs | 5 arms × seeds {0,1,2} × 1000 updates = 15, all rc 0 |
| wall time | 972 s total (4 lanes on 2× RTX 4090); about 105–270 s per run |
| GPU usage | peak 13.2 / 11.4 GB per GPU, 3.4 GB logged per run; mean utilisation 98 / 88 % |
| evaluation | 15 v4B + 9 v3 reference checkpoints × 6 image conditions × 1,524 chunks; 583 s + 842 s |
| final training loss (s0/s1/s2) | R .069/.069/.074; RpPos .083/.081/.089; RpHO .095/.096/.100; RpBoth .089/.086/.095; RpPosAll .085/.084/.089 |
| pretrained | `lerobot/smolvla_base` 5e8d12a6, model sha256 `7cd549ac…01eb` |
| normaliser | max difference from the `sfa_v31_R` real-only statistics over all 15 checkpoints: 2.8e-8 |
| training data | `sfa_v31_R` + `sfa_v31_Gp_matched_evidence_complete` (frozen; sha256 in `configs/frozen_inputs_sha256.json`). Pool `2c1bc860…`, real_10 `cebdee98…` |
| pipeline check | the v4B R arm reproduces the v3 R checkpoints: far grasp +0.0 % [−0.7, +0.9]; phase-balanced +0.3 % |

**Recorded artifacts:**
* `outputs/v4b/train_runs.json` (git, seeds, wall times, losses, GPU);
* `train/*_mixture.json` (per-run pools, real-source weights, schedule sha256);
* `logs/`;
* `predictions_manifest.json` and `predictions/checkpoint_hashes.json`. Checkpoints (19 GB) and predictions (310 MB) stay local.

## 2. Original-image results (seed-averaged; normalised per-step arm-action error; lower is better)

| arm | far grasp (15) | near grasp (25) | approach | grasp | carry | handover | release | phase-balanced | gripper timing (frames) |
|---|---|---|---|---|---|---|---|---|---|
| R | 0.4110 | 0.3106 | 0.2821 | 0.3483 | 0.4654 | 0.3713 | 0.4448 | 0.3824 | 5.08 |
| R+Position | **0.3565** | 0.2907 | 0.2671 | 0.3154 | 0.4753 | 0.3855 | 0.4511 | 0.3789 | 5.13 |
| R+Handover | 0.4039 | 0.3097 | 0.2806 | 0.3450 | 0.4746 | 0.3747 | 0.4439 | 0.3838 | 5.25 |
| R+Both | 0.3705 | 0.2862 | 0.2667 | 0.3178 | 0.4713 | 0.3749 | 0.4451 | **0.3752** | 4.91 |
| R+PositionAll (secondary) | 0.3543 | 0.2870 | 0.2674 | 0.3123 | 0.4582 | 0.3866 | 0.4511 | 0.3751 | 5.06 |
| v3 R (reference) | 0.4108 | 0.3095 | 0.2815 | 0.3475 | 0.4629 | 0.3695 | 0.4439 | 0.3811 | 5.07 |
| v3 R+G+ (reference) | 0.3576 | 0.2952 | 0.2749 | 0.3186 | 0.4789 | 0.4091 | 0.4549 | 0.3873 | 4.96 |
| v3 R+C (reference) | 0.3551 | 0.2934 | 0.2794 | 0.3166 | 0.4813 | 0.4134 | 0.4501 | 0.3881 | 4.93 |

Approach through phase-balanced columns are over all 40 recordings.

### Relative change vs R (positive = better), paired recording bootstrap 95 % CI

| arm | far grasp | near grasp | approach | carry | handover | release | phase-balanced | per-seed far grasp | handover per-seed (degradation) | frozen PASS |
|---|---|---|---|---|---|---|---|---|---|---|
| R+Position | **+13.2 % [+7.0, +20.9]** | +6.4 % [+4.0, +8.8] | +5.3 % [+3.0, +7.8] | −2.1 % [−3.6, −0.7] | **−3.8 % [−4.9, −2.8]** | −1.4 % [−2.0, −0.9] | +0.9 % [−0.2, +2.1] | +13.4 / +13.1 / +13.2 | 4.2 / 4.5 / 2.7 % | **No** (handover) |
| R+Handover | +1.7 % [−0.2, +3.3] | +0.3 % [−0.8, +1.3] | +0.5 % [+0.1, +1.0] | −2.0 % [−2.9, −1.1] | −0.9 % [−3.7, +1.9] | +0.2 % [−0.9, +1.3] | −0.4 % [−1.1, +0.4] | +2.3 / +1.7 / +1.2 | 1.1 / 2.8 / −1.1 % | No |
| R+Both | +9.9 % [+6.1, +13.6] | +7.9 % [+5.6, +10.1] | +5.4 % [+3.3, +7.8] | −1.3 % [−2.3, −0.3] | −1.0 % [−2.5, +0.6] | −0.1 % [−0.6, +0.4] | **+1.9 % [+1.1, +2.7]** | +10.9 / +9.6 / +9.1 | 1.1 / 1.5 / 0.3 % | **Yes** (secondary) |
| R+PositionAll | +13.8 % [+7.6, +20.8] | +7.6 % [+4.6, +10.5] | +5.2 % [+3.0, +7.5] | +1.5 % [−0.4, +3.4] | −4.1 % [−5.1, −3.1] | −1.4 % [−2.0, −0.9] | +1.9 % [+0.7, +3.2] | +13.9 / +14.5 / +13.0 | 4.6 / 4.7 / 3.1 % | No (handover) |
| v3 R+G+ | +13.0 % [+6.7, +20.3] | +5.0 % [+1.4, +8.2] | +2.6 % [+0.2, +5.0] | −2.9 % [−5.3, −0.8] | −10.2 % [−13.7, −6.7] | −2.3 % [−3.8, −0.8] | −1.3 % [−2.9, +0.4] | – | – | No |
| v3 R+C | +13.6 % [+7.0, +21.5] | +5.5 % [+2.2, +8.7] | +1.0 % [−1.2, +3.4] | −3.4 % [−6.1, −1.0] | −11.3 % [−14.6, −8.1] | −1.2 % [−2.8, +0.3] | −1.5 % [−3.2, +0.2] | – | – | No |

**Other bins:**
* Far-covered grasp (2 recordings: pool 13, 47): R+Position +27.7 % [+9.0, +41.6]. Only two recordings.
* Far-uncovered grasp (13 recordings): R+Position +10.7 % [+5.6, +17.1].

**Gripper timing:** changes are within ±0.2 frames for every arm (R 5.08 frames).

**Primary-10 subset, phase-balanced (the v3 scope):** R+Position +1.3 % [−1.2, +4.5]; R+Both +1.0 % [−0.3, +2.3]. Both CIs include zero.

## 3. Visual dependence (B4; relative error increase vs the model's own original images, all 40 recordings)

| arm | blank grasp | wrong-episode grasp | object-only grasp | equal-area background grasp | gripper-only grasp | object − background (grasp) | object-only approach |
|---|---|---|---|---|---|---|---|
| R | +22.4 % | +34.4 % | +9.0 % [+1.9, +17.1] | −0.8 % [−3.6, +2.3] | −0.8 % | +12.7 % [+4.1, +22.9] | +14.6 % |
| R+Position | +25.1 % | +50.8 % | +14.8 % [+6.7, +23.8] | +0.1 % | −1.8 % | +16.9 % [+7.6, +27.2] | +9.7 % |
| R+Handover | +21.9 % | +34.7 % | +12.0 % | +0.7 % | −0.1 % | +14.7 % [+5.9, +24.5] | +15.0 % |
| R+Both | +20.0 % | +49.9 % | +10.8 % | −0.8 % | −2.1 % | +13.3 % [+5.5, +22.1] | +9.5 % |
| R+PositionAll | +29.1 % | +54.5 % | +17.9 % | +0.3 % | −1.2 % | +19.3 % [+9.7, +29.9] | +11.8 % |

Full CIs, the handover phase and the v3 references are in `outputs/v4b/summary.md`.

**What the diagnostics show:**
* **Every model uses object pixels at approach and grasp.** Occluding the can (object-only) raises grasp error by 9–18 %. Occluding an equal-area background patch has no effect (CIs include 0). This corrects the v3.1 caveat that its occlusion mask also removed gripper and context pixels.
* **Position-augmented models depend more on image–state consistency at grasp.** Wrong-episode grasp: +50–55 % vs +34 % for R and R+Handover. The object − background contrast is also higher, though its CIs overlap R's.
* **Gripper-only occlusion does not raise error (−2 % to +0.6 %).** Either the gripper is inferred from state, or the FK mask is misplaced; it uses the nominal body camera, dilated 15 px, with a known 20–40 px inherited offset. This is unresolved.
* **Handover-phase image dependence is small** for all arms.

**Separation from usefulness:** these numbers measure dependence, not data validity. Greater object reliance does not show that the synthetic images are correct.

## 4. Interpretation of the four questions (no causal claims beyond the design)

**A. Does position augmentation improve grasp prediction?**
* **Yes for open-loop grasp-action error on development data.**
  * The gain is large, seed-consistent and its CI excludes zero.
  * It appears in all position-containing arms (+9.9 to +13.8 % far grasp; +6.4 to +7.9 % near grasp; about +5 % approach).
  * R+Handover does not show it. The gain therefore tracks the position-only data, not "more synthetic data".
* **But the frozen primary screen fails,** because of a 3.8 % handover degradation.
* **"On underrepresented positions" is only partly supported.**
  * Far gains exceed near gains (13.2 vs 6.4 %).
  * But far recordings *not* covered by generated positions also improve (+10.7 %), and only 2 far recordings are covered.
  * The data cannot separate "coverage of new positions" from a general grasp-phase regularisation effect.

**B. Does handover augmentation hurt downstream learning?**
* **Not established at a 50 % real, equal-exposure budget.**
  * Handover-only data neither helped nor clearly hurt handover (−0.9 % [−3.7, +1.9]).
  * It gave no grasp benefit.
* **The handover loss in R+Position is not caused by handover-window images or labels.**
  * In position-only episodes, 95 % of handover-window frames are exact source copies (exploratory, `exploratory_render_vs_source_by_phase.json`), and states and actions there equal the source.
  * Its cause is unknown. Candidates:
    1. carry-phase G+ return tails, whose 50-step chunks extend into the handover;
    2. interference between learning new reach/grasp behaviour and shared handover prediction;
    3. the 8-source synthetic pool.
* **Mixing handover-only data in (R+Both) is associated with a smaller handover loss** (−1.0 %). This is an observation from a secondary arm, not a mechanism.

**C. Does dataset composition explain part of the earlier failure?**
* **Descriptively, likely.**
  * Old v3 R+G+ (91.7 % synthetic rows, mixture normalisation, v3 generator, uniform frame sampling) had the same far-grasp gain (+13.0 %) as v4B R+Position.
  * But its handover loss was 10.2 % and its aggregate was −1.3 %.
  * The matched 50 %-real arms lose ≤ 4.1 % on handover, and R+Both improves the aggregate (+1.9 %).
* **The contrast is confounded** by generator version (v3 vs v3.1 G+), normalisation, sampler and family mix. A composition-only ablation (same v3.1 pool, real fraction 8 % vs 50 %) is needed to isolate it.

**D. Could image artifacts explain a remaining gap?**
* **Not testable with available controls.**
* **Known artifacts are present only in pre-handover frames:**
  * ghost rings at the old can location;
  * a lid seam near grasp;
  * reduced label texture;
  * x−6 cm alignment residuals of 5–10 px.
* The identity render is an exact source copy, so a same-trajectory artifact control does not exist (v4A dependency).
* Artifacts may limit the grasp gain, but nothing here measures that.

## 5. Negative findings and limitations

* **Primary rule failed** (handover guardrail). R+PositionAll also fails it.
* **R+Handover:** no benefit; hypothesis B not established.
* **Primary-10 subset (the v3 scope):** no arm's phase-balanced CI excludes zero.
* **Few covered recordings:** 2 far-covered recordings, so no coverage-specific conclusion.
* **Data limitations:**
  * Sources 4 and 6 have no evidence-complete position-only data.
  * There are only 27 handover-only episodes. The joint-change class (6) was too small for its own arm.
  * R+Both is a mixture arm, not joint-change episodes.
* **Statistical limitations:**
  * Uncertainty is recording-level and does not model retraining variance beyond 3 seeds.
  * Seeds share data.
  * Recordings 10–49 are reused development data.
  * 1000 updates fall entirely inside the 1000-step LR warmup (same as v3). Behaviour at convergence is unknown.
* **Scope limitations:**
  * Phases are command-derived proxies.
  * The B4 masks use HSV/FK heuristics; the gripper mask has calibration offsets.
  * Open-loop error is not robot task success. Physical/force validity is unknown.

## 6. Evidence categories

| category | status |
|---|---|
| Engineering feasibility | Matched, reproducible 15-run screen with exact exposure control, real-only normalisation and v3 R reproduction. |
| Dataset validity | Structural and label validity pass for all 71 sampled episodes. Pre-grasp visual artifacts are documented, not repaired. Physical validity unknown. |
| VLA learning (open loop, development data) | Grasp-phase improvement from position-only augmentation (robust, seed-consistent). The frozen primary screen fails on the handover guardrail. The R+Both secondary arm passes. |
| Real-robot performance | Not tested. |
| Scientific novelty | None claimed. Object-position augmentation for grasp generalisation is established (DemoGen, R2RGen, Real2Render2Real). The real-ratio/mixture effect overlaps EMMA-style data-mix work. |

## 7. Smallest next experiment

**Confirm R+Both vs R on fresh data, with one design change.**

Confirmation data:
* Collect ≥ 15 new real recordings at predeclared far initial positions inside the generated range.
* Operator-run, with separate authorization. Lock them before inspection.
* Re-use the 15 frozen v4B checkpoints. No retraining is needed for the confirmation.

Same frozen rule, primary arm R+Both:
* ≥ 5 % far grasp improvement;
* CI excluding zero;
* ≤ 2 % handover degradation.

Development-only add-on (no new data):
* A composition-only ablation: the same RpPos pool at 8 % vs 50 % real.
* A carry-tail ablation: position-only episodes with the synthetic rows restricted to frames before the right-gripper close.

These isolate the cause of the handover loss.

## Reproduction (from `research/sim_free_augmentation/v4`)

See `V4B_EXPERIMENT_SPEC.md` §11. Scores: `outputs/v4b/scores.json`, `outputs/v4b/summary.md`.
