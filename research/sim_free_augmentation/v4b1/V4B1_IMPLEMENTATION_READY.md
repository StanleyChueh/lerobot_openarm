# v4B.1 implementation ready: execution-gate evidence

**Status:** implemented, tested and smoke-run. **The full bounded training has not started.** It waits for Codex's pre-training audit of this commit (see `V4B1_EXPERIMENT_SPEC.md` §9).

**Code:**
* `sfa4b1/interventions.py`: dataset-view flags and the post-preprocessing applier.
* `scripts/train_v4b1.py`: the frozen v4B trainer path plus the intervention.
* `scripts/run_v4b1.py`: orchestrator; writes pre-launch provenance before any worker starts.
* `scripts/budget.py`: loss-budget calculation and the matched `p_drop`.
* `scripts/readiness.py`: GPU-side gate checks.
* `scripts/score_v4b1.py`: frozen decisions.
* `scripts/figures_v4b1.py`.
* `tests/test_v4b1.py`.

**Config:** `configs/v4b1.json` (arms, `p_drop` = 0.6634, scoring and thresholds).

## Gate checks

| # | check | evidence | result |
|---|---|---|---|
| 1 | Data readback / frozen inputs | `test_frozen_inputs_and_v4_untouched`: all 17 frozen v4B dataset/manifest sha256 match; `v3`, `v3_1`, `v4` trees have no modifications | PASS |
| 1b | Frozen reference checkpoints | `readiness.json`: model.safetensors of v4B R/RpBoth/RpPos × seeds 0–2 match the v4B evaluation manifest (9/9) | PASS |
| 2 | Sample-pair equality | `test_schedules_equal_frozen_v4b`: every new arm × seed reproduces the stored v4B `schedule_sha256` of its base arm (1,000 batches; 15/15) | PASS |
| 3 | Action-target identity | `test_action_targets_identical_and_aligned`: for real and synthetic rows of every arm, `action`, `observation.state`, `action_is_pad` and image tensors equal the v4B view. Synthetic targets equal parquet actions f..f+49; padding exactly beyond the episode end | PASS |
| 4a | Image intervention (unit) | `test_applier_image_and_mask_effects`: real images unchanged; synthetic pixels = empty-camera fill; `camera1_padding_mask = ~is_synthetic`; shape (B,3,480,640) unchanged; non-RGB arms untouched | PASS |
| 4b | Image intervention (model) | `readiness.json`: SmolVLA training loss (fixed noise/time, per sample) of masked synthetic rows is invariant to their pixels (max \|Δ\| **0.0**). Unmasked rows change with pixels (**0.0103**). Real rows unaffected (**0.0**) | PASS |
| 5a | Loss-mask boundaries | `test_phase_mask_boundaries_and_grippers`: for all 25 RpPos episodes, the boundary is at right_close + 15 from the episode's own commands. Each tested step is excluded iff its v3.1 per-target-step phase is carry/handover/release | PASS |
| 5b | Gripper channels | same test: right close lies inside supervised targets; left close and right open lie in excluded targets; the mask covers all 16 channels together | PASS |
| 5c | Real targets untouched | unit tests plus a per-step training log (`real_pad_ever_changed = false` in all smoke runs) | PASS |
| 5d | Random mask | `test_random_mask_rate_and_real_untouched`: drop rate within 1 % of `p_drop`; deterministic; never on real rows | PASS |
| 5e | Loss normalisation | `readiness.json`: policy loss with a target mask equals sum(valid)/count(valid) computed by hand (\|Δ\| 0.0) | PASS |
| 6 | Model init + 20-step smoke | all 5 arms rc 0 and a checkpoint was written. Logged synthetic share of valid loss elements: NoSynthRGB / reruns 0.50; PhaseMask 0.27 (4.45 of 8 synthetic rows fully masked per batch); RandMask 0.25 | PASS |
| 7 | Evaluation readback | the v4B evaluator ran on the 5 smoke checkpoints (40 recordings × 6 conditions, 233 s); `score_v4b1.py` and `figures_v4b1.py` ran end-to-end. Smoke numbers are discarded | PASS |
| 8 | Unit tests | `tests/test_v4b1.py` 6/6; `v4/tests/test_v4b.py` unaffected | PASS |

## Budget (H2) under the frozen RpPos schedules (`outputs/budget.json`)

| quantity | PhaseMask | RandMask |
|---|---|---|
| synthetic target steps kept, seed 0 / 1 / 2 | 34.4 / 32.9 / 33.7 % | 33.5 / 33.6 / 33.7 % |
| synthetic rows fully masked | 59–61 % | ≈ 0 % |

**Budget confound:** the synthetic share of the loss halves (about 50 % → about 25–27 %). That is why the RandMask control exists.

## Provenance corrections applied (from the Codex v4B audit)

* **Pre-launch provenance:** written before workers start (`run_v4b1.py`) and per run (`*_run.json`, `prelaunch` block).
* **LeRobot revision:** recorded from PEP 610 metadata: `huggingface/lerobot@3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e`, version 0.6.2.
  * This differs from the `530b253f…` cited in the Codex review. That revision is probably a separate checkout; the installed package metadata is used here.
* **Effective schedule:** documented as 33 warmup / 1,000 decay steps.
* **Exposure:** stated as exact in expectation only.
* **Terminology:** the "handover-only" pool is renamed *fixed-initial-position later-motion* in v4B.1 text.

## Remaining non-critical limitations (documented, not blocking)

* The missing-image counterfactual trains an image-free mode on synthetic rows.
* Fully masked rows still occupy batch slots.
* Input-side effects of partially masked rows remain.
* The evaluation pool is reused development data.
* The cached v4B masks include 151 unequal-area background fallbacks (diagnostic only).
