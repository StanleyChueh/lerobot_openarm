# SD10-L1 results: real-only SmolVLA source-set screen (Original10 / SD10 / Random10)

**Date:** 2026-11-08. **Status: Executed — Pending Independent Codex Review.** Exploratory, open-loop, reused
development data. No robot operation, physics simulation, synthetic data or external push.

**Registered decision: FAIL (primary screen not passed; far-grasp interval inconclusive).**
**Recommendation: STOP SD10 and RETURN TO v4A.** SD10 does not improve over the fixed Random10 control, so the
pre-registered condition for a source-set × augmentation experiment ("improves over both controls") is not met.

Report scores only within this study. Do **not** compare them numerically with frozen v4B/v4B.1 results: the training
sources, evaluation recordings (21 vs 40), bins and normalizers all differ.

## 1. Provenance and conformance

| Item | Value |
|---|---|
| Contract | `SD10_LEARNING_PROTOCOL.md` + `sd10/learning_v1/`, seal SHA-256 `7c74ef44bd658136535e3ad65991d53278c5f957624388f929ca929c54197183` (verified before launch, at every fit and at scoring) |
| Pre-training decision | Codex **GO** (protocol header; Notion "SD10-L1 preregistration", 2026-11-07) |
| Git | base `59b2bfe`; frozen bundle tracked unchanged at `ac35752`; execution code committed **before launch** at `703692b` (launch HEAD, recorded in `outputs/prelaunch.json`) |
| Data | pool `ethanCSL/openarm_pringles_lerobot_real_v00@2c1bc860…` for every arm (Original10 = pool 0–9; the separately encoded real_10 RGB is not used) |
| Model | `lerobot/smolvla_base@5e8d12a6…`, SmolVLM2-500M `@7b375e1b…`, LeRobot source `530b253f` (clean; imported from source, not the venv wheel) |
| Fits | 3 arms × seeds 0/1/2 × 1,000 AdamW updates, batch 16, sequential on one RTX 4090 in the frozen order, ≈125 s each |
| Eval | 834 locked inputs from 21 recordings; every RGB frame re-decoded and matched to the locked hash/PTS/time base |

Verified for all nine runs (`outputs/run_verification.json`, `ALL_PASS: true`):
- 1,000 consumed optimizer updates, and Adam `step == 1000` on every trainable tensor of the saved optimizer state.
- Every consumed batch equals the frozen schedule row, with 1,600 draws per source and zero discarded prefetched batches.
- The logged LR sequence equals `learning_rates.npy`, which is bit-equal (max abs diff 0) to the pinned LeRobot cosine scheduler (33 warmup / 1,000 decay).
- Saved normalizer and unnormalizer tensors equal the frozen arm statistics.
- Losses are finite. Model and optimizer hashes match the terminal records.

Prediction archives are float32 `(834,50,16)` with exactly the locked item order and are finite. Repeated inference was
bit-identical in a smoke test.

**Disclosed implementation facts (not deviations from the contract text, but material to interpretation):**
1. **Parameter storage dtype.** Training used no autocast/AMP, with TF32 and cuDNN benchmarking off. However, the
   pinned SmolVLA checkpoint stores the VLM and most expert matrices in **bfloat16** (96.6 M of 99.9 M trainable
   parameters; projections are float32). This is identical to `smolvla_base` and the historical v4B checkpoints, and
   was not altered.
2. **Determinism.** Deterministic algorithms were enabled in warn-only mode with `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
   No nondeterminism warning was emitted. The only captured warning was Python's fork-in-threaded-process
   DeprecationWarning.
3. **Training loop.** A single-process transcription of `lerobot_train.update_policy`, without accelerate. The
   registered rate is set before each update.

## 2. Absolute errors (14 arm joints, rad; mean over 3 seeds; recording-bootstrap 95% CI)

| Endpoint | Original10 | SD10 | Random10 |
|---|---|---|---|
| **Far grasp (6 recordings, primary)** | 0.0769 [0.0502, 0.1219] | 0.0691 [0.0372, 0.1185] | 0.0725 [0.0386, 0.1220] |
| Near grasp (15) | 0.0494 [0.0439, 0.0548] | 0.0434 [0.0354, 0.0529] | **0.0375** [0.0332, 0.0427] |
| Handover (21) | 0.0494 [0.0430, 0.0568] | **0.0402** [0.0355, 0.0453] | 0.0417 [0.0378, 0.0463] |
| Phase-balanced (21) | 0.0547 [0.0499, 0.0596] | 0.0439 [0.0396, 0.0494] | 0.0439 [0.0391, 0.0502] |
| Overall time-weighted (21) | 0.0536 [0.0497, 0.0574] | 0.0425 [0.0389, 0.0467] | 0.0428 [0.0390, 0.0477] |
| Handover gripper R (openness MAE) | 0.346 [0.310, 0.380] | 0.350 [0.307, 0.393] | 0.350 [0.308, 0.395] |
| Handover gripper L (openness MAE) | 0.333 [0.286, 0.382] | 0.269 [0.234, 0.306] | 0.257 [0.224, 0.291] |

**Per-seed values (seed 0 / 1 / 2).**

| Arm | Far grasp | Handover | Overall |
|---|---|---|---|
| Original10 | 0.0768 / 0.0770 / 0.0769 | 0.0499 / 0.0489 / 0.0495 | 0.0536 / 0.0531 / 0.0542 |
| SD10 | 0.0693 / 0.0691 / 0.0689 | 0.0406 / 0.0397 / 0.0403 | 0.0424 / 0.0426 / 0.0424 |
| Random10 | 0.0723 / 0.0727 / 0.0725 | 0.0422 / 0.0417 / 0.0413 | 0.0428 / 0.0428 / 0.0429 |

Seed variation is tiny compared with recording variation, so the crossed seed+recording intervals almost equal the
recording-only intervals. Uncertainty is dominated by **which recordings** are evaluated, especially among the six
far recordings.

## 3. Registered contrasts (relative improvement = 1 − mean(arm)/mean(ref); positive = lower error)

### SD10 vs Original10 (primary)

| Endpoint | Point | 95% CI (recording) | Crossed CI | Seeds |
|---|---|---|---|---|
| **Far grasp** | **+10.2%** | **[−9.4, +33.2]** | [−9.4, +33.2] | +9.8 / +10.3 / +10.4 |
| Near grasp | +12.2% | [−12.8, +30.2] | | +12.0 / +12.3 / +12.4 |
| Handover (21) | +18.6% | [+9.6, +26.6] | | +18.6 / +18.8 / +18.6 |
| Phase-balanced (21) | +19.6% | [+11.1, +27.4] | | |
| Overall (21) | +20.8% | [+13.1, +28.2] | | |
| Gripper L handover MAE (increase) | −0.064 abs | [−0.121, −0.008] | | |
| Gripper R handover MAE (increase) | +0.004 abs | [−0.054, +0.065] | | |

Criterion 2:
- Far-grasp improvement ≥5%: **pass** (10.2%).
- CI lower bound >0: **FAIL** (−9.4%).
- All three seeds nonnegative: **pass**.
- Handover harm ≤2%: **pass** (−18.6%, an improvement).
- Phase-balanced harm ≤2%: **pass**.
- Gripper guard (increase >0.02): **not flagged** on either side.

**Verdict FAIL.** The interval spans both no benefit and material benefit, so the primary result is
**inconclusive, not evidence of no effect**. The crossed (seed-robustness) CI lower bound is also ≤0.

Utility: the handover harm CI upper bound is −9.6%, so harm is confidently no more than 2% (an offline screening
statement, not demonstrated noninferiority).

### SD10 vs Random10 (hierarchical secondary — **not tested** because the primary failed; reported descriptively)

| Endpoint | Point | 95% CI |
|---|---|---|
| Far grasp | +4.7% | [−8.2, +15.4] |
| Near grasp | **−15.6%** | [−34.0, −0.0] |
| Handover (21) | +3.6% | [−6.9, +12.9] |
| Phase-balanced (21) | −0.2% | [−5.3, +4.4] |
| Overall (21) | +0.8% | [−3.3, +4.8] |

**Advantage over this random draw is not established.** On the aggregate endpoints SD10 and Random10 are
indistinguishable, and Random10 is better on near grasp.

### Random10 vs Original10 (descriptive)

Overall +20.1% [+12.4, +27.2], phase-balanced +19.7%, handover +15.6% [+3.1, +26.0], near grasp +24.1%
[+12.1, +33.8], far grasp +5.7% [−19.0, +31.3].

**The broad ~20% improvement over Original10 is shared by an uncurated random draw from the same pool.**

## 4. Error versus object position (`figures/l1_error_vs_position.png`)

- **Far-bin per-recording grasp MAE** (Original10 / SD10 / Random10, seed mean):

  | Recording | Original10 | SD10 | Random10 |
  |---|---|---|---|
  | 12 | 0.186 | 0.182 | 0.191 |
  | 20 | 0.054 | 0.030 | 0.032 |
  | 22 | 0.070 | 0.072 | 0.056 |
  | 23 | 0.051 | 0.033 | 0.036 |
  | 30 | 0.044 | 0.067 | 0.081 |
  | 40 | 0.057 | 0.032 | 0.039 |

  SD10 is better on 4 of 6 far recordings and worse on 30. Recording 12 has about 3× the error of the other far
  recordings in every arm and dominates the far mean and its interval. This is descriptive only; no post-hoc
  exclusion is applied.
- **No arm shows a detectable dependence of grasp error on distance to its own nearest source.** Spearman ρ over the
  21 recordings:
  - Original10: ρ = 0.18 (p = 0.44);
  - SD10: ρ = −0.07 (p = 0.78);
  - Random10: ρ = 0.27 (p = 0.23).
- The SD10 − Original10 grasp difference is unrelated to distance from Original10 sources (ρ = −0.06, p = 0.81).
- The SD10/Original10 overall-error ratio improves for **later** evaluation recordings (ρ = −0.57 vs recording index,
  p = 0.007). This is consistent with a recording-order/pace effect rather than a spatial-coverage effect.

## 5. Training-source pace and phase exposure (`outputs/exposure_pace.json`, `figures/l1_exposure_pace.png`)

Realized consumed exposure equals the frozen plan exactly. Each fit has 800,000 valid targets, and each source
1,600 draws / 80,000 targets.

| | Original10 | SD10 | Random10 |
|---|---|---|---|
| Native rows (mean length) | 2,575 (257.5) | 2,415 (241.5) | 2,424 (242.4) |
| Eligible / unique input starts | 2,085 | 1,925 | 1,934 |
| Start → right close (s) | 2.92 [2.73–3.37] | 2.81 [2.13–3.43] | 2.76 [2.17–3.20] |
| Right close → left close (s) | 2.64 [2.47–2.90] | 2.41 [1.77–2.83] | 2.40 [2.00–3.00] |
| Left close → right open (s) | 0.22 | 0.16 | 0.18 |
| Right open → end (s) | 2.80 | 2.67 | 2.73 |

Seed-mean target-phase exposure relative to Original10:

| Phase | SD10 | Random10 |
|---|---|---|
| Approach | +1.9% | −1.1% |
| Grasp | +10.5% | +9.1% |
| Carry | −4.9% | −5.5% |
| Handover | −2.5% | +1.1% |
| Release | +0.2% | +3.5% |

The 21 evaluation recordings average 241 frames. That matches the pace of SD10 and Random10 (241–242), not
Original10 (257.5): the original ten are the earliest and slowest recordings.

## 6. Interpretation

1. **The registered primary screen fails.** The SD10 far-grasp gain (+10.2%) is consistent across seeds but not
   established across recordings. With six far recordings, one of them an outlier, this screen cannot resolve it.
2. **The curation is not what helps.** Replacing the original ten with any ten from the 10–49 era improves
   open-loop error by about 20%, and the fixed Random10 draw gets the same aggregate gain as SD10. SD10's extra
   spatial coverage shows no measurable benefit over Random10, and no arm's error tracks distance to its own sources.
3. **The most plausible driver is distributional match** between the training sources and the evaluation recordings
   (recording order, operator pace, action and phase distribution, and fitted normalizers), not object-position
   diversity. This is a hypothesis: the screen cannot separate these factors, and recording order is not a verified
   session ID. **No causal position claim is made.**
4. Per the registered hierarchy and the requested rule: **STOP SD10 and RETURN TO v4A.** No source-set ×
   augmentation experiment is justified by this evidence.
5. **A methodological flag for earlier work (hypothesis, not a finding about v4B).** Earlier studies trained on
   pool 0–9 and evaluated on recordings 10–49. In this screen, swapping those sources for later-era recordings
   lowered open-loop error by ~20%, which suggests that source/evaluation mismatch in recording era or pace can be
   large. This screen does not show that mismatch alone accounts for it. Future evaluation designs should match or stratify recording era and pace before
   attributing gains to augmentation. Frozen v4B/v4B.1 results are unchanged and are not re-scored here.

## 7. Limitations

- **Evaluation data.** The evaluation set is reused development data. All 21 recordings were exposed during earlier
  development, and initial positions informed source selection. This is not an untouched or publication test.
- **Far bin.** It has only six recordings, one of which dominates.
- **Seeds.** The intervals condition on three seeds. There is no estimate of generalization across operators or
  sessions, and within-session dependence is possible.
- **Scope of evidence.** Open-loop action prediction only. There is no closed-loop robot success measure and no
  force or contact validation.
- **Event timing.** Phase boundaries are command proxies; body-camera/state alignment is a dataset convention.
- **Precision.** Expert weights are stored in bf16 (see §1). Bitwise reproducibility on other hardware is not
  guaranteed.
- **Selection context.** SD10 is ten selected from 50 collected. Source curation is not novel.

## 8. Reproduce

```bash
sha256sum -c research/sim_free_augmentation/sd10/learning_v1/locked_artifacts.sha256
cd research/sim_free_augmentation/sd10/l1_exec
PY="env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../../.venv/bin/python -B"
$PY build_frame_cache.py                      # decode + verify 834 locked RGB hashes (CPU)
CUDA_VISIBLE_DEVICES=0 $PY run_all.py         # pre-launch checks, 9 fits, eval, scoring
$PY verify_l1.py; $PY exposure_l1.py; $PY figures_l1.py
$PY score_l1.py                               # rescoring from committed prediction archives only
```

The committed `outputs/predictions/*.npz` (float32) allow rescoring without retraining. Checkpoints and optimizer
states (≈1.3 GB per run) are local only; their SHA-256 values are in `outputs/run_verification.json` and each
`outputs/train/*/terminal.json`.

## 9. Artifacts

| Path | Contents |
|---|---|
| `outputs/results.json` | completeness, absolute errors, all contrasts, screens, per-record scores |
| `outputs/per_record_scores.csv` | per-record scores |
| `outputs/per_joint_errors.json` | 16-channel errors |
| `outputs/predictions/*.npz`, `*.json` | prediction archives and their identities |
| `outputs/run_verification.json`, `outputs/prelaunch.json`, `outputs/train_runs.json` | run verification and launch records |
| `outputs/train/*/{prelaunch,terminal}.json`, `train_log.json`, `training_state/schedule_state.json` | per-run records |
| `outputs/exposure_pace.json`, `outputs/frame_cache_report.json` | exposure/pace analysis and frame-cache report |
| `figures/l1_endpoints.png`, `figures/l1_error_vs_position.png`, `figures/l1_exposure_pace.png` | figures |
