# Training comparison

Sources: both checkpoints' `train_config.json`, `config.json` and normalizer files (small files only; model weights
not downloaded or loaded), checkpoint revisions pinned:

| | manual-data policy | generated-data policy | (generated_400 policy, for reference) |
|---|---|---|---|
| repo @ revision | `ethanCSL/openarm_pringles_lerobot_sim_smolvla_10_v00` @ `9ce7ce48` (2026-10-21) | `ethanCSL/openarm_pringles_lerobot_sim_10_generated_500_smolvla_v00` @ `733c6856` (2026-10-24) | `…_generated_400_smolvla_v00` @ `60661c0d` |
| dataset | sim_10_v00 (10 eps, 2,364 frames), `revision: null` → v3.0 tag = `b6d39dae` | generated_500 (500 eps, 134,418 frames) → `860741a2` | generated_400 (first 400 of generated_500) |

## 1. What differs and what does not (`diff` of the two train_config.json)

Only `dataset.repo_id`, `batch_size` (8 → 32), `steps` (20k → 40k), output/job names. Everything else is identical:

| setting | both |
|---|---|
| base | `lerobot/smolvla_base` (no revision pinned), VLM `SmolVLM2-500M-Video-Instruct` |
| trainable | `freeze_vision_encoder=True`, `train_expert_only=True`, `train_state_proj=True` |
| chunk | `chunk_size=50`, `n_action_steps=50`, `n_obs_steps=1`, flow steps 10 |
| normalization | STATE/ACTION MEAN_STD, VISUAL identity; each normalizer = its own dataset's stats (5e-8) |
| images | `resize_imgs_with_padding [512, 512]`, rename_map right_wrist→camera1, wrist→camera2, body→camera3, `empty_cameras 0` |
| RGB transforms | enabled: brightness/contrast 0.8–1.2, saturation 0.5–1.5, hue ±0.05, sharpness 0.5–1.5, affine ±5°/±5 %, up to 6, random order |
| optimizer | AdamW lr 1e-4, betas (0.9, 0.95), wd 1e-10, clip 10 |
| scheduler | cosine with warm-up, 1,000 warm-up / 30,000 decay → 2.5e-6, seed 1000 |
| precision | fp32, `use_amp=False` |

The fork's scheduler auto-scales when `steps < num_decay_steps` (`optim/schedulers.py`): the 20k run warms up
666 steps and decays over all 20k; the 40k run decays over 30k and then trains 10k steps at the floor lr 2.5e-6.
Both end fully decayed. **CONFIRMED.**

## 2. Exposure

| | manual | generated |
|---|---|---|
| samples seen | 20k × 8 = 160k | 40k × 32 = 1.28M (8×) |
| passes over frames | 67.7 | 9.5 |
| passes over distinct trajectory templates (≈10 vs ≈5.6) | ~16k samples / template | ~229k samples / template |
| distinct camera views / looks | 1 (the evaluation's) | ≈ one per episode, none equal to the evaluation's |

The 67.7-vs-9.5 frame-pass ratio overstates under-training: 500 generated episodes are ≈ 5.6 templates with small
can offsets (473/500 near-duplicates), so per *behaviour* the generated model saw far more samples than the
manual model. What it saw less of is the evaluation's visual condition — never.

**Undertraining of the generated model: UNKNOWN, not supported.** No loss curves (wandb disabled; no
`outputs/train` on this machine) and no intermediate checkpoint are available. The 40k run saved a step-20k
checkpoint (`save_freq=20000`) on the training machine; it was not pushed. If it still exists, it is the cheapest
budget ablation (F).

## 3. Evidence that bears on optimization

* The generated policy reproduces its data's timing and action lead in closed loop (ACTION_SEMANTICS §3, §5) —
  the behaviour was learned.
* Its successes and failures are position-structured exactly along its data's coverage (EVALUATION_COMPARISON §4)
  — a coverage signature, not a uniform under-fitting signature.
* generated_400 (identical recipe, 400 episodes) scored an estimated 61/100 vs 47/100 for generated_500 on
  different random can positions (Fisher p ≈ 0.06): more generated episodes did not monotonically help, and
  checkpoint-to-checkpoint variance is large. (The 61 is a detector estimate from the tick log, validated at
  99–100/100 agreement on the recorded runs; EVALUATION_COMPARISON §1.)

## 4. Fairness verdict

| claim | status |
|---|---|
| the two runs share base, transforms, chunking, freezing, rename map, normalization rule, schedule shape | CONFIRMED |
| batch and steps differ (8×20k vs 32×40k) | CONFIRMED |
| "the generated model is undertrained (9.5 vs 67.7 passes)" | UNKNOWN — not supported by the closed-loop evidence; no curves to test it |
| "the comparison is budget-matched" | DISPROVEN — 8× samples; also untested seeds (one seed each) |
