# v3.1 — why the generated data helps SmolVLA so little (Task A) and the v3 confound corrections (Task B)

**What this is:** offline evaluation and regeneration only.

**Not done:** retraining, simulation, physics, robot operation, Hub pushes.

**Preserved:** v3 code, config and results are untouched; this is guarded by a test.

**Commits:**
* `68eedca`: v3 protocol;
* `dad1b52`: Task A protocol declared before inference;
* `0d5086a`: Task B protocol frozen before the regeneration;
* the final results commit.

## Summary

* **The v3 policies do use vision.**
  * Every arm meets the pre-declared CLEAR rule. Blank and wrong-episode images each raise error by at least 10%, and the recording-level 95% CIs exclude 0.
  * Removing image tokens doubles the error.
* **What the image is used for:**
  * Vision matters in approach and grasp. It hardly matters in carry.
  * Occluding only the can produces about half of the blank-image effect in grasp, so object evidence is used, but much of the visual signal is scene and arm context.
* **Why augmentation barely helps: phase-dependent gains and losses that cancel.**
  * Compared with real-only training, R+G+ and R+C lower grasp-phase error by about 9% (0.348 → 0.317–0.319).
  * They raise handover error by about 11% (0.370 → 0.409–0.413) and carry error by 3–4%.
  * Under the per-step metric with training-only normalisation, the net effect is −0.3% to −1.9%, i.e. slightly worse.
  * Augmented models also lean more on the image: removing image tokens costs +101–107% against +76% for R.
* **Where generation added coverage, the sign flips:**
  * On held-out recordings near the source positions augmentation hurts (−3.3/−3.5%).
  * On far recordings it helps slightly (+1.0/+0.8%).
  * On the 4 far recordings that generation actually covers it helps by +1.3/+1.7%.
  * Too few recordings to claim anything.
* **Task B corrections:**
  * G+ and C now start from the identical recorded state: frame-0 difference 0.0 rad (v3: 0.102 rad); at most 0.003 rad at the first exported row.
  * Apparent lead is now a diagnostic. Of the 66 v3 lead-only rejections, 45 are accepted and residual-consistent, 1 is accepted but flagged residual-uncertain, 1 is rejected for joint speed, and 19 were not needed.
  * 109 matched pairs pass every export check; 92 of them are evidence-complete.

---

## Task A — vision-dependence experiment (`configs/vision_ablation.json`, committed before inference)

### Setup

* **Checkpoints:** the nine existing v3 checkpoints (R, R+G+, R+C × seeds 0–2); sha256 in `outputs/vision_ablation/checkpoint_hashes.json`.
* **Held-out data:** 40 external real recordings, pool 10–49 at revision `2c1bc860`. The primary set is pool 10–19, as in v3. Every 5th frame gives 1,524 chunks of 50 steps.
* **Held fixed:**
  * state, language, action targets and preprocessing (each checkpoint's own processors);
  * the diffusion noise: one stored N(0,1) tensor per (episode, frame);
  * bf16 autocast.
* **Image conditions:**
  * original;
  * blank (mid-grey);
  * wrong episode: the held-out recording with the farthest initial lid, at the same relative time;
  * can occluded: the Gate 1 HSV lid track plus red pixels, filled grey;
  * state-only: image tokens masked out of attention.
* **Metrics:**
  * per-step phases, with each chunk step labelled by its own target frame: approach / grasp / carry / handover / release;
  * training-only normalisation (real source episodes 0–9);
  * recording-level bootstrap;
  * near/far bins by distance from the 10 source initial lids (≤ 40 px).
* **Predictions:** all saved. Pool 10–19 is committed in `outputs/predictions_primary/`; the full set stays local with sha256 in `predictions_manifest.json`.
* **Runtime:** 374 s (2 GPUs, batch 32). Batched vs single-sample predictions differ by at most 0.004 rad (bf16), identically for every condition.

### Results (`outputs/vision_ablation/summary.md`, `scores.json`, `vision_ablation.png`)

| arm | condition | phase-balanced error, 10 primary | Δ vs original [95% CI] | all 40 recordings | gripper timing (frames) |
|---|---|---|---|---|---|
| R | original | 0.3556 | — | 0.3811 | 5.07 |
| R | blank | 0.3958 | **+11.3% [+7.3, +14.9]** | +10.6% | 5.87 |
| R | wrong episode | 0.4009 | **+12.7% [+5.9, +20.2]** | +12.2% | 5.67 |
| R | can occluded | 0.3848 | +8.2% [+4.0, +12.7] | +4.0% | 5.16 |
| R | state only | 0.6248 | +75.7% [+62.3, +89.4] | +57.1% | 7.81 |
| R+G+ | original | 0.3566 | — | 0.3873 | 4.96 |
| R+G+ | blank / wrong / occluded / state-only | | +11.9 / +16.4 / +6.4 / +106.5% | +10.6 / +14.7 / +2.4 / +87.3% | |
| R+C | original | 0.3584 | — | 0.3881 | 4.93 |
| R+C | blank / wrong / occluded / state-only | | +12.3 / +16.1 / +7.6 / +101.5% | +10.3 / +14.5 / +3.0 / +82.8% | |

Image dependence is **CLEAR for all three arms** under the pre-declared rule.

The OOD caveat from the task applies: blank and state-only are out-of-distribution. The conclusion rests on the agreement of all four interventions, including the in-distribution wrong-episode images.

**Per-step phase error, original images, 40 recordings:**

| arm | approach | grasp | carry | handover | release |
|---|---|---|---|---|---|
| R | 0.282 | 0.348 | 0.463 | 0.370 | 0.444 |
| R+G+ | 0.275 | **0.319** | 0.479 | **0.409** | 0.455 |
| R+C | 0.279 | **0.317** | 0.481 | **0.413** | 0.450 |

**Near / far initial positions, original images:**

| set | n recordings | R | R+G+ | R+C |
|---|---|---|---|---|
| primary (pool 10–19) | 10 | 0.3556 | −0.3% | −0.8% |
| all | 40 | 0.3811 | −1.6% | −1.9% |
| near source (≤ 40 px) | 25 | 0.3741 | −3.3% | −3.5% |
| far (> 40 px) | 15 | 0.3927 | +1.0% | +0.8% |
| far, covered by v3 generated positions (≤ 20 px) | 4 | 0.4279 | +1.3% | +1.7% |

Positive means better than R.

The v3 start-frame metric reproduces from these predictions (R 0.3347, R+G+ 0.3314, R+C 0.3323 with training-only normalisation): the earlier ~2% "gain" was a phase-mixed average.

### Diagnosis: why the benefit is limited

1. **Synthetic data dominates training.** With uniform row sampling the mixtures were 91.7% synthetic. That synthetic data contains displaced, rotated, regrasped and re-timed **handovers** (families A, B, D), so the model's handover behaviour is pulled away from the real handover distribution that held-out recordings follow. Handover error rises 11%.
2. **The gains sit where generation adds coverage.** New initial positions improve grasp error by about 9%, and far and generation-covered recordings improve slightly. But those are a minority of held-out frames.
3. **The model is not blind:** removing the image costs 76–107%. The limiting factor is the content and mix of the synthetic data, not a VLA that ignores RGB.

**Next experiment** (no large training; matches the v3 review's proposal):
* equal-budget, source-balanced arms: R vs R + initial-position-only G+ vs R + handover-only G+;
* equal synthetic frame counts and a fixed real fraction, e.g. 50% real rows;
* the same 40-recording evaluation with near/far bins and per-step phases.

The prediction to test: initial-position-only data keeps the grasp gain without the handover loss.

---

## Task B — corrections to v3 (frozen in `sfa31/protocol.py` before the regeneration; runtime replacements, v3 files untouched)

| item | implementation | evidence |
|---|---|---|
| 1. identical initial robot states | G+ tails are compact smoothstep ramps from an identity anchor at max(0, t_first − 60), and back to identity by t_last + 60. They replace v3's Gaussian tails. | frame-0 G+ − C difference **0.0 rad** over 218 common-scale pairs (v3: 0.102 rad over 207); first exported row ≤ 0.003 rad; `test_gplus_and_c_start_from_the_recorded_state`, `test_v3_gplus_did_move_the_first_state` |
| 2. identical targets, scales and sources | same families, waypoints, tolerances and repair scales; `configs/v3_1.json` records the v3 config sha256 | `test_targets_unchanged` |
| 3. construction vs apparent lead | **hard gates:** exact construction, unsupported rows, both grippers' command chronology and latency, finite values, hash. **Diagnostic only:** per-joint apparent lead. **Physical uncertainty:** a source-fitted per-joint residual model `r ≈ α + β q̇ + γ q̈`; the candidate is RESIDUAL_UNCERTAIN if the discrepancy is above 2× the source RMS (and above 2 mrad). | hard gates still catch source-action substitution, single-joint delays and command-only gripper delays (`test_hard_gate_still_catches_the_v2_false_negatives`; 26/26 per-episode corruption suite on 9 episodes again); `test_apparent_lead_is_diagnostic_not_a_gate` |
| 4. the 66 v3 lead-only rejections | 45 accepted and residual-consistent (lead flag only); 1 accepted but RESIDUAL_UNCERTAIN; 1 rejected (joint speed, new G+ path); 19 not re-proposed (accepted at a larger scale) | `outputs/taskB_analysis.json` |
| 5. receiver window UNKNOWN | kept as UNKNOWN in every exported episode's manifest (G+ 17, C 20 of the export-valid candidates). A separate **evidence-complete** subset requires receiver alignment "ok" and no residual uncertainty. | `outputs/dataset_export/*` |
| 6. v3 preserved | v3 code, config and results unchanged | `test_v3_tree_unchanged` |

### Paired sensitivity, v3 → v3.1 (10 episodes, 25 variations × 2 conditions)

| | v3 | v3.1 |
|---|---|---|
| accepted at own scale, G+ / C | 213 / 216 | 218 / 225 |
| at full scale, G+ / C | 167 / 176 | 174 / 196 |
| common-scale pairs | 207 | 218 |
| matched pairs passing every export check | 113 | 109 |
| evidence-complete matched pairs | — | 92 |
| C_all export | 130 | 136 |

* **Full-scale acceptance:** the corrected rule recovers more full-scale C candidates (+20) than G+ ones (+7). v3's lead heuristic was biased against C.
* **Matched pairs fall** (113 → 109) even though more candidates are accepted. G+'s new endpoint-preserving paths change some renders, and image QA then rejects them.

**Datasets** (local LeRobot v3, body cam only, all validated: schema, boundaries, exact arrays, decoded PSNR ≥ 37.8 dB, and per-camera video-segment lengths):

| dataset | episodes |
|---|---|
| R | 10 |
| G+ matched / C matched | 109 / 109 |
| G+ / C matched, evidence-complete | 92 / 92 |
| C_all | 136 |

* **Leakage:** none against pool 10–19.
* **Render cache:** entries are bound to the candidate hash and the renderer sha256.
* **Runtime:** regeneration 2,901 s (2 lanes × 9 CPU workers); export 820 s.

## Limitations

* **Open loop only:** one task, offline, open-loop. Seeds share recordings.
* **OOD interventions:** blank and state-only images are out-of-distribution; occlusion depends on an HSV lid and can mask.
* **Unbalanced near/far bins:** 25 near vs 15 far recordings; only 4 are generation-covered.
* **Residual model:** a linear-in-velocity/acceleration tracking model, not robot dynamics.

## Commands (from `research/sim_free_augmentation/v3_1`)

```bash
PY="env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python"
$PY scripts/vision_ablation.py && $PY scripts/score_ablation.py && $PY scripts/fig_ablation.py
$PY scripts/run_all31.py && $PY scripts/export31.py && $PY scripts/analyse_taskB.py
CUDA_VISIBLE_DEVICES= $PY -m unittest discover -s tests -v
```
