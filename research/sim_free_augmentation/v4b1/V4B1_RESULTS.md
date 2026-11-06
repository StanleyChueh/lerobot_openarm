# v4B.1 results: H1 (missing synthetic RGB) and H2 (phase-placed synthetic targets)

**Status: Executed — Pending Independent Codex Review.** Date 2026-11-06 (Asia/Taipei).

**Scope:** open-loop action-prediction error on development recordings 10–49 of the pool at `2c1bc860`. These recordings have been used repeatedly, so they are development data, not a test set. Nothing here measures robot success.

## 1. Provenance

| item | value |
|---|---|
| Code/protocol launched | HEAD `d29013e1fe3cb303510d600cfaeb86dbe465c5b9` (A1 code `e4149556`). Independent Codex pre-training re-audit: **GO** (`CODEX_SIM_FREE_V4B1_PRETRAIN_REAUDIT.md`, local) |
| Pre-launch guard | `outputs/prelaunch.json`: clean v4b1 tree, no untracked code. 38/38 content checks: 17 frozen inputs, 3 base-weight snapshots, 9 frozen R/RpPos/RpBoth model and normaliser pairs, 6 mask manifests, 3 exact matched-control checks |
| Runtime | LeRobot 0.6.2, PEP 610 revision `3f2c29ef`; SmolVLA `smolvla_base` snapshot 5e8d12a6; 2× RTX 4090 |
| Training | 15 runs (5 arms × seeds 0/1/2 × 1,000 updates); 4 lanes; 716 s wall; about 175 s per run; max memory 8.9 GiB per GPU. Same LR auto-scaling as v4B (warmup 1000 → 33, decay 30000 → 1000) |
| Evaluation | `v4/scripts/eval_v4b.py`, unchanged: 40 recordings, 1,524 chunks, 6 image conditions, frozen noise; 592 s |
| Scoring | `scripts/score_v4b1.py`, final mode. All A1 contract checks pass (completeness, frozen-reference hashes, evaluation contract, run verification) |
| Post-training integrity | `outputs/posttrain_content_checks.json`: 38/38 checks pass again after training. The tracked v4 and v3_1 trees are unchanged |

### Training verification (all 15 runs PASS)

`outputs/run_verification.json` (`scripts/verify_runs.py`) checks each run against the frozen schedule:
* trainer step = 1,000 and preprocessing calls = 1,000;
* the row indices the trainer actually consumed equal the frozen v4B schedule;
* the synthetic flags and the applied supervision masks equal the expected masks bit for bit;
* real rows are fully supervised, padding is preserved, and real padding is never changed;
* the schedule SHA-256 equals the frozen value, and the normaliser equals the frozen real-only normaliser `aaf92ed3…`;
* the tree was clean at launch.

`outputs/optimizer_verification.json` (`scripts/verify_optimizer_state.py`, added post-run for re-audit non-blocking item 3) inspects the **saved** training state directly:
* all 155 AdamW per-parameter `step` counters = 1,000 (the same count as the frozen reference);
* scheduler `last_epoch` = 1,000, `_step_count` = 1,001, final LR 2.5e-6;
* optimizer param groups and scheduler state are identical to the frozen v4B run of the same base arm and seed;
* the only checkpoint is `001000`, with `last` pointing to it, and `resume=False`;
* the training config has no differences from the frozen base in steps, batch size, seed, optimizer, scheduler, save frequency or workers;
* the evaluated model hash equals a fresh rehash.

No run resumed and no frozen checkpoint was overwritten.

**Realised supervision budgets** (`outputs/train/*_budget.json`, mean valid target steps per batch):

| | real | synthetic | synthetic share | fully masked synthetic rows per batch |
|---|---|---|---|---|
| unmasked arms | 362.3 / 362.4 / 361.9 | 362.8 / 362.6 / 361.9 | 50.0% | 0 |
| PhaseMask = MatchedRandMask | 362.3 / 362.4 / 361.9 | 124.7 / 119.3 / 122.1 | 24.5 / 23.7 / 24.2% | 4.74 / 4.84 / 4.81 |

The two H2 arms are identical per batch, as registered (Amendment A1).

### Replication finding: reruns are bit-identical

`RpPos_rerun_s{0,1,2}` and `RpBoth_rerun_s{0,1,2}` produce **model.safetensors files with the same SHA-256** as the frozen v4B `RpPos_s*` and `RpBoth_s*`. Their predictions differ by exactly 0.0 under every image condition.

Two consequences follow:
1. The v4B.1 training path (dataset view, Applier, sampler) reproduces v4B exactly. GPU nondeterminism (spec §8) is therefore **not** a confounder of any treatment-vs-frozen contrast.
2. The registered replication-noise guard is satisfied trivially (noise = 0). It does **not** measure seed-to-seed or retraining variability; it was never meant to. That uncertainty is visible only through the three paired per-seed values and the recording bootstrap, which is conditional on the fitted seeds.

## 2. Results

Error is open-loop and normalised per target. "Rel." is the relative change in error, where + means the first model is better. 95% CIs are paired recording-bootstrap intervals (10,000 resamples); per-seed values are paired. Full tables are in `summary.md` and `results.json`.

### Seed-averaged error, original images

| arm | far grasp | near grasp | handover (40) | carry | phase-balanced | gripper timing (frames) |
|---|---|---|---|---|---|---|
| R (frozen) | 0.4110 | 0.3106 | 0.3713 | 0.4654 | 0.3824 | 5.08 |
| RpPos (frozen) | 0.3565 | 0.2907 | 0.3855 | 0.4753 | 0.3789 | 5.13 |
| RpBoth (frozen) | 0.3705 | 0.2862 | 0.3749 | 0.4713 | 0.3752 | 4.91 |
| RpBoth_NoSynthRGB | 0.3812 | 0.3000 | 0.3784 | 0.4812 | 0.3822 | 5.04 |
| RpPos_PhaseMask | 0.3553 | 0.2875 | 0.3979 | 0.4739 | 0.3794 | 5.21 |
| RpPos_MatchedRandMask | 0.3614 | 0.2924 | 0.3974 | 0.4745 | 0.3809 | 5.29 |
| RpBoth_rerun / RpPos_rerun | identical to frozen RpBoth / RpPos | | | | | |

### Key paired contrasts

| contrast | far grasp | near grasp | handover | phase-balanced | per-seed far grasp | per-seed handover | timing Δ (frames) |
|---|---|---|---|---|---|---|---|
| **H1** RpBoth vs NoSynthRGB | +2.8% [−1.2, +7.3] | **+4.6% [+2.7, +6.7]** | +0.9% [−0.2, +2.0] | **+1.8% [+1.1, +2.6]** | +1.0 / +3.2 / +4.1 | +0.6 / +2.2 / −0.0 | −0.14 |
| NoSynthRGB vs R | **+7.2% [+3.4, +10.9]** | +3.4% [−0.1, +6.5] | **−1.9% [−3.1, −0.8]** | +0.0% [−1.0, +1.1] | +10.0 / +6.6 / +5.2 | −1.7 / −3.7 / −0.3 | −0.04 |
| **H2** PhaseMask vs RpPos | +0.3% [−2.0, +2.0] | +1.1% [+0.3, +1.8] | **−3.2% [−4.9, −1.6]** | −0.1% [−0.9, +0.6] | −0.2 / +1.1 / +0.1 | −3.4 / −3.7 / −2.5 | +0.08 |
| MatchedRandMask vs RpPos | −1.4% [−3.2, +0.0] | −0.6% [−1.4, +0.1] | **−3.1% [−4.8, −1.4]** | −0.5% [−1.3, +0.2] | −1.8 / −0.5 / −1.8 | −3.2 / −3.6 / −2.4 | +0.17 |
| **H2-D** PhaseMask vs MatchedRandMask | **+1.7% [+0.9, +2.4]** | **+1.7% [+1.3, +2.1]** | −0.1% [−0.2, −0.0] | +0.4% [+0.2, +0.5] | +1.6 / +1.6 / +1.8 | −0.2 / −0.1 / −0.1 | −0.09 |
| PhaseMask vs R | +13.5% [+8.3, +19.7] | +7.5% [+5.0, +9.9] | **−7.1% [−8.6, −5.7]** | +0.8% [−0.2, +1.7] | +13.3 / +14.1 / +13.3 | −7.8 / −8.4 / −5.3 | +0.12 |
| RpPos vs R (frozen v4B) | +13.2% [+7.1, +20.8] | +6.4% | −3.8% [−4.9, −2.8] | +0.9% | +13.4 / +13.1 / +13.2 | −4.2 / −4.5 / −2.7 | +0.04 |
| RpBoth vs R (frozen v4B) | +9.9% [+6.2, +13.7] | +7.9% | −1.0% [−2.5, +0.6] | +1.9% [+1.1, +2.6] | +10.9 / +9.6 / +9.1 | −1.1 / −1.5 / −0.3 | −0.18 |

The frozen v4B point estimates are reproduced exactly (+13.24% / −3.81%; +9.86% / −0.96% / +1.89%). The CI endpoints differ by up to 0.2 points from the v4B report because these bootstrap tables use different resampling streams.

### v4B screening rule applied to the new arms (screen only; does not change the v4B verdicts)

| arm | far grasp vs R | handover degradation | screen |
|---|---|---|---|
| RpBoth_NoSynthRGB | +7.2% [+3.5, +11.0] | 1.9% [+0.8, +3.1] | PASS (point estimate under 2%; CI permits up to 3.1%) |
| RpPos_PhaseMask | +13.5% [+8.3, +19.5] | 7.1% [+5.8, +8.6] | FAIL |
| RpPos_MatchedRandMask | +12.1% [+6.7, +18.6] | 7.0% [+5.6, +8.5] | FAIL |

## 3. Frozen decisions (`results.json → decisions`)

### H1: `INCREMENTAL_VALUE_NOT_ESTABLISHED`

Generated RGB vs missing synthetic camera, far grasp: **+2.8% [−1.2, +7.3]**. All three seeds are positive (+1.0 / +3.2 / +4.1), but the effect is below the 3% threshold and the CI includes 0.

Reported alongside, as registered:
* NoSynthRGB alone passes the far-grasp part of the screen vs R (+7.2%, CI > 0).
* Retained-gain fraction (R − NoRGB)/(R − Both): **0.73 [0.38, 1.14]**.

Not part of the frozen H1 rule, reported as secondary evidence: generated RGB improves near grasp **+4.6% [+2.7, +6.7]**, approach +4.5% [+2.9, +6.1], carry +2.0% [+0.6, +3.5] and phase-balanced error **+1.8% [+1.1, +2.6]**. Handover is +0.9% [−0.2, +2.0].

### H2: `PHASE_MASK_INCREASES_HANDOVER_ERROR`; Q2 = `NOT_SUPPORTED_BY_THIS_SCREEN`

| A1 component | result |
|---|---|
| A: handover improvement vs RpPos | **worsened**: −3.2% [−4.9, −1.6], all seeds negative. MatchedRandMask is worsened too: −3.1% [−4.8, −1.4] |
| B: residual handover degradation vs R (2% guardrail) | **7.1% [5.8, 8.6]: fails.** Worse than the v4B RpPos FAIL (3.8%) |
| C: far-grasp preservation | vs R +13.5% [+8.3, +19.5]: **P2 holds**. vs RpPos +0.3% [−2.0, +2.0]: not resolved |
| D: phase-specific increment vs matched random (handover) | −0.12% [−0.21, −0.03], seeds −0.18 / −0.10 / −0.09. A very small difference, **not in the hypothesised direction** |
| E: statistical resolution | rerun noise = 0 (bit-identical). CI widths: A 3.4 points, D 0.18 points |
| original P1 / P2 / P3 | false / true / false |

Under the original §7 wording, the outcome is "phase restriction does not reduce handover regression". The A1 scorer reports the stronger, CI-supported statement that both masks **increase** handover error.

## 4. Failure cases and negatives (all retained)

* H1 did not meet its registered threshold. Generated RGB's incremental far-grasp value is not established.
* About three quarters of the R+Both far-grasp gain is retained without any synthetic images (CI up to 114%).
* Without synthetic RGB, handover is significantly worse than R (−1.9%). With generated RGB, handover is −1.0% vs R with a CI that includes 0.
* H2's premise failed. Removing synthetic carry, handover and release targets did not reduce R+Position's handover regression; it roughly doubled it (degradation 3.8% → 7.1% vs R). The matched random control did the same.
* PhaseMask is not better than matched random on handover. Its only advantage over matched random is in far and near grasp (+1.7%), the phase it keeps supervising.
* Gripper timing changes are below 0.3 frames for every contrast.
* Under inference-time image conditions (summary table, "Inference-time image conditions"), RpBoth is much more harmed than NoSynthRGB by wrong-episode images: −15.8% far grasp, −24.0% grasp. It gains relative to NoSynthRGB under background or gripper occlusion (+5.2% / +5.5% far grasp), and has no advantage under blank images (−1.4%). This is consistent with the synthetic-RGB model relying more on image content. It is diagnostic only.

## 5. Artefacts

**Committed:**
* `results.json`, `summary.md`, `figures/v4b1_contrasts.png`;
* `outputs/prelaunch.json`, `outputs/train_runs.json`, `outputs/run_verification.json`, `outputs/optimizer_verification.json`, `outputs/posttrain_content_checks.json`;
* `outputs/predictions_manifest.json`: SHA-256 of the 15 prediction archives, `targets.npz` and `eval_meta.json`; the evaluated checkpoint hashes; and the hashes of the recorded mask logs;
* `outputs/train/*_{run,budget}.json` and `outputs/train/*_masks.npz` (applied masks and consumed indices per step);
* `outputs/logs/` (training and evaluation logs).

**Kept local, gitignored** (the existing artefact location used for v4B): checkpoints `outputs/train/<run>/` (19 GB) and prediction archives `outputs/predictions/` (194 MB).

Interpretation: `V4B1_CAUSAL_INTERPRETATION.md`.

## 6. Reproduction (from `research/sim_free_augmentation/v4b1`)

```bash
PY="env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 ../../../.venv/bin/python"
$PY scripts/run_v4b1.py --lanes 4                     # skips runs whose final checkpoint exists
CUDA_VISIBLE_DEVICES= $PY scripts/verify_runs.py
$PY ../v4/scripts/eval_v4b.py --train_dir outputs/train --pred_dir outputs/predictions
CUDA_VISIBLE_DEVICES= $PY scripts/verify_optimizer_state.py
CUDA_VISIBLE_DEVICES= $PY scripts/score_v4b1.py
$PY scripts/figures_v4b1.py
```
