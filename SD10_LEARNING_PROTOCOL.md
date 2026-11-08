# SD10-L1: preregistered real-only SmolVLA source-set screen

**Date:** 2026-11-07 (Asia/Taipei). **Decision: GO for this bounded experimental design. No training has been run or authorized by this document-writing task.** Execution must conform to the locked artifacts and pass the launch checks below. This is a new SD10 study; v4A, v4B, v4B.1 and the frozen source selection are unchanged.

**Status:** preregistered, exploratory, not an untouched test or a confirmation study. Git base verified locally: `59b2bfe697c7400e201c988d3f05123b9018dcac`, branch `research/contact-aware-augmentation`. New protocol files are local/uncommitted; no push or remote availability is implied. The [independent selection review](CODEX_PRINGLES_DIVERSE10_SELECTION_REVIEW.md) and current Notion SD10 continuation informed this protocol. Historical selection-review MODIFY is preserved; this protocol addresses its prospective split/exposure requirements.

## 1. Question and bounded scope

Does training on the fixed SD10 source set improve open-loop real-recording action prediction over the original ten, at the same source and optimization budget? Does it improve over the already fixed Random10 draw?

Exactly **three conditions × three paired seeds × 1,000 optimizer updates = nine fresh fits**. Real observations and real recorded actions only. No synthetic demonstrations, renderer, physics engine, phase-loss masking, image intervention, image augmentation, policy rollout or robot evaluation. No historical trained checkpoint substitutes for a new arm.

The estimand is the effect of the **source-set package under this training recipe**, including its action/pace distribution and per-arm training normalization. It is not the independent causal effect of initial object position. Neither real-only learning nor source curation is a new augmentation mechanism.

## 2. Frozen captures, revisions and exclusions

| Condition | Pool episode IDs | Native rows | Eligible full-chunk input rows |
|---|---|---:|---:|
| Original10 | `0,1,2,3,4,5,6,7,8,9` | 2,575 | 2,085 |
| SD10 | `1,6,7,11,13,14,27,34,42,47` | 2,415 | 1,925 |
| Random10 | `5,6,7,8,18,25,29,32,43,44` | 2,424 | 1,934 |

**Every arm uses the same encoding from** `ethanCSL/openarm_pringles_lerobot_real_v00@2c1bc860347c44fc0fbae7567469d36ee49baa8d`. Original10 names the original captures, read from this pool encoding. Do not mix in the separately encoded RGB of `ethanCSL/openarm_pringles_lerobot_real_10_v00@cebdee98115ab6827a4ec2655335c536b5dd085b`. Its episodes 0–9 are the unique bit-identical state/action aliases of pool 0–9, verified here and used only for lineage confirmation.

Canonical capture IDs are `pringles-real-v00-capture-XX`; dataset-name aliases do not make independent captures. No source quality filtering, substitutions, reselection, new random-ten draw or alternate seeds after the freeze. Selection IDs and source quality judgments remain as previously audited.

Common evaluation, in this exact order:

`[12,15,16,19,20,21,22,23,26,28,30,33,35,36,37,39,40,41,46,48,49]`.

It excludes the **union** of all three training-source sets, every alias/descendant of those captures, and v4A records `[10,17,24,31,38,45]`. All three arms must use precisely this common evaluation. No synthetic descendant is eligible anywhere in this real-only screen. Capture overlap, not merely episode-name overlap, invalidates the study.

Recordings 10–49 were reused in earlier development. Seven SD10 sources were formerly evaluation recordings; they are training sources only in this separately named study. Initial pool positions also informed source selection. The 21 recordings are **reused exploratory development data**, not publication-test data. This study must not be numerically spliced into the old 40-recording scores. The source audit's unfavorable close-coverage diagnostics remain evidence and must not trigger reselection.

## 3. Identical initialization, inputs and optimization

Use `lerobot/smolvla_base@5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`, loaded from its pinned local snapshot. Backbone/tokenizer: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct@7b375e1b73b11138ff12fe22c8f2822d8fe03467`, also a pinned local path. All checkpoint/config/tokenizer file hashes are in `source_and_runtime_manifest.json`; mutable Hub names are insufficient. Rebuild processors using the new arm's frozen normalization; do not inherit the base model's six-dimensional statistics.

LeRobot implementation base: `530b253f262d6af92ea3b91d62abcae0295af59f` (`update_dev_v2`, clean at freeze). Source hashes and installed package versions are recorded. SmolVLA settings:

- One observation: raw native body RGB and 16-D measured state at row `t`.
- Explicit input features: `observation.state:[16]` and **only** `observation.images.camera1:[3,480,640]`, renamed from `body_cam`. No wrist inputs, retained camera2/3 features or missing-camera treatment.
- PyAV RGB uint8 → CHW float32 /255 → the pinned policy's aspect-preserving 512×512 resize/zero letterbox and conversion to `[-1,1]`. VISUAL normalization is IDENTITY. No random crop, color change, blur, geometric/image augmentation or ImageNet-stat substitution.
- Fixed task: “Pick up the Pringles can with the right arm, hand it to the left arm”. Apply the same task-newline/tokenizer processing, maximum length 48, right padding to maximum length.
- Chunk 50; `n_action_steps=50`; internal state/action width 32; supervised/output feature width **16**. Native interleaved channels `RJ1,LJ1,…,RJ8,LJ8`; indices 14/15 are the right/left gripper commands.
- Freeze vision encoder; `train_expert_only=true`, `train_state_proj=true`; no Aloha remapping/delta actions, PEFT, EMA or compilation. Full configuration overrides are frozen in `training_config.json`.

| Setting | Fixed value |
|---|---|
| Paired training seeds | `0,1,2` |
| Optimizer updates | Exactly 1,000 per fit; gradient accumulation 1 |
| Batch | 16, single process/single GPU |
| Optimizer | AdamW, lr `1e-4`, betas `(0.9,0.95)`, epsilon `1e-8`, weight decay `1e-10`, global gradient clip 10 |
| LR schedule | Explicit effective cosine warmup **33** updates and decay 1,000, floor `2.5e-6`; exact update-indexed rates in `learning_rates.npy` |
| Precision | Training float32/no autocast or AMP; evaluation bf16 autocast with float32 saved predictions |
| Data loading | Four workers, prefetch 2; frozen batch sampler, no additional shuffle |
| Selection/checkpoint | Terminal update 1,000 only; save final model, optimizer, scheduler, RNG and consumed-schedule position |

The LR schedule matches the effective short-run schedule of the existing LeRobot implementation; **it is not 1,000 updates entirely inside warmup**. Scheduler rates are applied before each corresponding optimizer update, with scheduler stepping afterward.

Run sequentially on the same GPU model/device and software environment. Disable TF32 and cuDNN benchmarking; seed Python/NumPy/Torch/CUDA before initialization and log determinism settings/warnings. Paired seeds/common draws do not guarantee bitwise reproducibility on different hardware. Fixed arm order: seed 0 Original10→SD10→Random10; seed 1 SD10→Random10→Original10; seed 2 Random10→Original10→SD10. No environment simulator or online evaluation is configured. Uploads and W&B are disabled.

## 4. Matched sampling, action construction and exposure

Three frozen `sampling_seed*.npz` files define **all 1,000×16 draws** per arm. Each has exactly **1,600 input draws per source**, 16,000 per fit. Common source IDs occupy common slots across conditions; shared captures therefore receive identical starts at identical draw positions. Other slots use paired time quantiles across different source captures. Slot allocation and RNG PCG64/SeedSequence rules are fixed in `protocol_tools.py`.

For episode length `T`, eligible starts are **`0,…,T−50` inclusive**. Each slot's 1,600 stratified uniform quantiles map by `floor(q*(T−49))`. This is source-balanced full-chunk sampling, not uniform sampling of all dataset rows. The last 49 rows are not input anchors but remain valid target rows. Shorter demonstrations receive more reuse per eligible input frame; that is documented rather than called matched epochs.

An input at row `t` has state/RGB from that native row and targets **`action[t:t+50]`**, offsets 0–49. Do not silently shift targets to `t+1`. Preserve the recorded row convention; no fitted camera/action retiming. Standard SmolVLA flow-matching MSE supervises every real target timestep and all 16 real channels. Internal dimensions 16–31 are excluded from loss by the output-feature width. No phase restriction or special gripper reweighting.

All chunks are full length: **800 valid target timesteps / 12,800 scalar loss entries per batch**; **800,000 / 12,800,000 per fit**; 80,000 target timesteps per source. Set `action_is_pad` false for all 50 steps. Denominators and source coefficients match exactly across conditions. No cross-record chunks or repeated endpoint padding.

`training_exposure.json` freezes every run's input/target phase counts, per-source counts, unique starts and per-native-target reuse multiplicities. **Phase counts are not matched.** Changes in recording pace, phase duration, command timing and action distribution are part of this source-set intervention. The same total loss budget is not proof of equal phase-specific optimization exposure. Execution must report actual consumed counts, including discarded prefetched batches separately from optimizer-consumed batches.

For example, the frozen seed-0 plan has the following target-timestep counts (not executed training results):

| Arm | Approach | Grasp | Carry | Handover | Release |
|---|---:|---:|---:|---:|---:|
| Original10 | 222,828 | 76,960 | 227,207 | 63,344 | 209,661 |
| SD10 | 227,144 | 85,037 | 216,013 | 61,784 | 210,022 |
| Random10 | 220,412 | 83,959 | 214,688 | 64,014 | 216,927 |

SD10 therefore receives about 10.5% more grasp targets than Original10 despite the equal total budget. This is predeclared exposure composition, not an isolated diversity mechanism; all three seed plans are saved.

## 5. Training-only normalization

`training_normalization.json` freezes separate state/action means and population standard deviations for each arm. Fit on **all native rows of that arm's ten real training captures only**, each row equally weighted, float64 accumulation, standard-deviation floor `1e-6`, stored float32 tensors. This row weighting is explicit; it is not the source-balanced sampler's weighting. All seeds of an arm use identical statistics.

Use MEAN_STD with processor epsilon `1e-8`, no clipping/quantile normalization. Follow the pinned normalizer's forward `(x−mean)/(std+eps)` and inverse `x*std+mean` convention. No evaluation rows, other-arm-only sources, synthetic data, inherited full-pool `meta/stats.json`, or old exported-dataset statistics may fit these processors. Verify saved normalization tensors against the frozen values and retain their hashes. Pixel normalization is fixed and does not fit evaluation images.

Per-arm numeric normalizers differ legitimately. This is a **common normalization policy, not identical normalization tensors**, and it is another reason not to attribute a learning contrast solely to position diversity. Evaluation compares unnormalized predictions in common physical coordinate units, never arm-specific normalized losses.

## 6. Frozen evaluation frames, timing, bins and phases

The canonical [evaluation manifest](research/sim_free_augmentation/sd10/learning_v1/locked_evaluation_manifest.json) fixes **834 inputs**, ordered by ascending episode then native frame. Starts are stride five through the last possible full chunk, with `T−50` always included. Thus every physical target row is represented without padding. `evaluation_arrays.npz` fixes ordered items, measured input states, raw 16-D action targets, target-level phase IDs and inference noise. Each manifest entry binds its RGB hash, actual MP4 PTS/time base, row timestamp, state/target/noise hashes and target interval.

No evaluation image ablations: **original body RGB only**, identical observations/targets/noise for all nine models. Use `predict_action_chunk`, reset state between independent batches, inference batch 32, ten flow steps. A stored `(50,32)` float32 Gaussian noise tensor per input uses NumPy PCG64 seed `20261105+1000*episode+frame`; it is shared across all arms and fitted seeds. Use each checkpoint's own verified pre/post processors and save unnormalized `(834,50,16)` predictions as **float32**, not float16.

The manifest distinguishes native camera PTS from row time. Same-index RGB/state/action is a fixed dataset convention, **not independently measured simultaneous exposure**. The earlier body lag estimate is a proxy with unknown physical alignment; it is not used to tune shifts or recalibrate on these 21 recordings. This screen cannot validate physical contact synchronization.

**Position bins are common across conditions**, using the fixed A1 lid-center feature from `outputs/lid_centers.json`: Euclidean native-body-pixel distance to the nearest **Original10** initial lid center; near ≤40 px, far >40 px. Initial RGB position metadata defines bins, not policy scores or action errors. Per-arm distances are saved for description only; do not change group membership by arm.

- Far (6): `[12,20,22,23,30,40]`.
- Near (15): `[15,16,19,21,26,28,33,35,36,37,39,41,46,48,49]`.

Command proxies: `rc` right-close, `lc` left-close, `ro` right-reopen at recorded commanded openness crossing 0.5. For **each individual target frame `u=t+k`**, use these half-open intervals:

| Phase | Native target frames |
|---|---|
| Approach | `[0,rc−5)` |
| Grasp | `[rc−5,rc+15)` |
| Carry | `[rc+15,lc−5)` |
| Handover | `[lc−5,ro+5)` |
| Release | `[ro+5,T)` |

Commands define reproducible phase proxies, not measured contact labels. No chunk-start-only labeling. Exactly `rc−5` is grasp, `rc+15` carry, `lc−5` handover and `ro+5` release. Per-record phase vectors and every chunk's target-phase vector are frozen.

## 7. Metrics, confidence intervals and practical criteria

**Primary metric:** far-recording grasp-phase **14-arm-joint raw MAE**, radians, SD10 versus Original10. Exclude the two gripper channels from this arm metric, but score/report them separately. For forecast `(t,k)`, compute the mean absolute error over native channels 0–13 at target `u=t+k`. Where multiple anchors forecast the same target time, average their **absolute errors**, not their predicted actions, before phase averaging. Each physical target time therefore has equal weight within its phase. Average within recording, then equally over the six far recordings and three fitted seeds.

Secondary metrics: near grasp, each phase over all 21, handover over all 21, phase-balanced error (equal five phases per recording), overall time-weighted error (equal native target times per recording), and right/left handover gripper MAE in openness units. Gripper error is `abs(pred−target)/abs(open_raw)` with R open `−1.298`, L open `1.2121`, **without clipping**; a grossly mis-scaled command must not look correct merely through saturation. Save per-joint errors and per-record/seed scores as well as aggregates.

For a comparison A against B, relative improvement is `1−mean(error_A)/mean(error_B)`, positive better. Report absolute errors, absolute differences and each seed's paired contrast. Handover/overall relative harm is its negative. Do not average per-record percentage ratios instead of this registered ratio of means.

If a reference error is exactly zero, report absolute differences and mark the corresponding relative criterion undefined/inconclusive; do not divide by zero or invent an epsilon-based percentage.

**Uncertainty:** 10,000 paired recording bootstrap draws, PCG64 seed 20261107; exact all21/near/far index matrices are frozen in `bootstrap_indices.npz`. First average the three fitted seeds within each recording, then resample **recording IDs**, keeping conditions paired, and use percentile 2.5/97.5%. Frames/chunks are not independent observations. Primary CIs condition on three fitted seeds and these reused recordings. Also report the frozen crossed paired-seed/recording bootstrap sensitivity (resample three seed indices and recording indices, preserve pairings), individual-seed contrasts and the small-six-far limitation. Do not describe three seeds as broad training-population certainty. If intervals are wide, call it inconclusive; do not add recordings or weaken thresholds after results.

The recording bootstrap does not estimate generalization across new operators or sessions. Actual independent session identifiers are unavailable; potential within-session dependence remains a limitation of both registered intervals.

**Fixed decision hierarchy:**

1. **Completeness first:** all nine terminal checkpoints, 1,000 verified optimizer updates each, all 834 prediction items per run, required targets/normalizers/configuration/exposure identities and finite arrays. Otherwise **INCOMPLETE**, with no scientific screen verdict. No historical R substitution or missing-seed averaging.
2. **Positive source-set screen versus Original10:** far-grasp relative improvement ≥**5%**, paired-recording 95% CI lower bound >0, and nonnegative far-grasp improvement in **all three** paired seeds. Handover harm and phase-balanced harm over all 21 must each be ≤**2%** in point estimate. Either gripper's handover openness MAE increase >**0.02 absolute** flags **MODIFY**, even if the arm-error criteria pass.
3. **Utility uncertainty:** always report handover/overall/gripper CIs. The point-estimate guardrail is an offline screening rule, **not demonstrated noninferiority**. If the handover CI still allows >2% harm, label utility uncertain; a confident no-more-than-2%-harm statement requires the harm CI upper bound ≤2%. “Robust to seed uncertainty” additionally requires the crossed-bootstrap improvement CI lower >0.
4. **Hierarchical selection-specific secondary:** only after criterion 2 passes, test SD10 versus the fixed Random10. Require far-grasp improvement ≥**3%**, CI lower >0, all three seed contrasts nonnegative, and the same point utility/gripper guardrails against Random10. Otherwise report the actual contrast and say **advantage over this random draw not established**. This single random subset does not estimate all possible random-ten selections.
5. Other endpoints/comparisons are descriptive; their intervals are unadjusted. No endpoint replacement or post-hoc favorable subgroup verdict. If the primary CI upper bound is below 5%, the intended practical effect is unsupported within the conditional screen; if the CI spans material benefit and no benefit, it is inconclusive. Failure to reject zero is never equivalence. Preserve beneficial grasp plus harmful handover as a mixed/failed utility result.

Thresholds are frozen prospectively for this new screen. Earlier v4B/v4B.1 thresholds/results remain unchanged. `protocol_tools.py` is the reference metric/CI definition; future scorer implementation must match it.

## 8. Confounding and reporting rules

Every result must state: selected ten from **50 collected**, not ten collected; exploratory reused development pool; real-only, open-loop conditional action prediction; three paired seeds; no robot success or force/contact validation.

**Do not claim a causal position-coverage effect**, even if the far-grasp endpoint passes. Source initial positions, recording order/operator pace, action/phase distributions, frame reuse and fitted normalizers co-vary. Recording order is not a verified session identifier. Report source/event-duration distributions and realized phase/exposure counts alongside effects. A pace/session-matched design or other independently justified control is needed to isolate position. This screen does not compare generated RGB, contact constraints, augmentation mechanisms or three-camera validity. Keep v4A/C/D open and preserve all historical negatives.

## 9. Launch checks, stopping and frozen artifacts

**No executable training wrapper is supplied or invoked by this preregistration.** Future implementation must verify, before its first optimizer update:

- Seal/input/code/checkpoint/tokenizer hashes; approved dataset revisions and source aliases; zero source/evaluation capture intersections.
- Exact one-camera/16-D features, no hidden extra cameras, all augmentation disabled, normalization tensors from the registered arm only, 50 correctly ordered real targets and zero padding.
- Exact registered schedules, same slot/quantile arrays, full-run source counts and per-batch valid-target denominators; no accidental second sampler/shuffle. Phase exposure differences must be reported rather than silently equalized.
- The frozen LR array, trainable-parameter configuration, matched hardware/precision, all required seed/configuration values, fresh initialization and no evaluation-driven selection.
- A new output namespace separate from historical experiments; actual model/processor/optimizer hashes, Adam step counters and schedule-consumption records saved for audit. Hash the new adapter/scorer implementation before launching; it may implement this contract but may not tune it after seeing outcomes.

A subsequent implementation/preregistration commit may descend from `59b2bfe`; record its actual SHA. The verifier allows descendants while enforcing the frozen dependency hashes. It does not assume this local base is the final training-code commit.

Run all nine fits to update 1,000 without intermediate evaluation/early stopping. Training loss may be monitored for nonfinite values or infrastructure failure only. Stop on integrity violations/nonfinite loss; record failure and leave the study incomplete. Infrastructure-only resumes require identical checkpoint, optimizer, RNG, LR and sampler position, without duplicated/skipped updates. If exact resume is impossible, restart the same registered arm/seed from the same base and disclose the failed attempt; no seed/configuration replacement. Do not extend training or add arms after unfavorable results.

The seal is `research/sim_free_augmentation/sd10/learning_v1/locked_artifacts.sha256`, verified from the repository root. It binds this protocol, evaluation manifest/arrays, source/runtime identities, training configuration/normalization/exposure, schedules, LR rates, bootstrap indices and model-free reference/freezing/verification scripts. Its own SHA is recorded in the Notion handoff; there is no self-hash cycle. Changes require a dated, separately named amendment **before training**, retaining the original files and reason. Do not overwrite a freeze.

```bash
# Read-only verification; no policy loading or training:
sha256sum -c research/sim_free_augmentation/sd10/learning_v1/locked_artifacts.sha256
env -u PYTHONPATH -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES='' \
  HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
  .venv/bin/python -B research/sim_free_augmentation/sd10/learning_v1/verify_preregistration.py
```

`freeze_inputs.py` documents reproducible construction and refuses to overwrite generated artifacts. Rebuild only in a **new** directory for semantic comparison; NPZ compressed-container hashes and freeze timestamps need not be byte-identical across rebuilds, but array contents and semantic definitions must match. Raw native images remain in the pinned dataset; only hashes/indices and small numeric evaluation arrays are in the lock bundle.

**GO conclusion:** this is a defensible bounded exploratory learning screen once implementation conforms. The prior prospective split problem is resolved by a common frozen 21-recording evaluation and exact exposure accounting. Learning efficacy, quality gains, causal spatial mechanisms and scientific novelty remain unestablished until appropriate evidence exists. This task created preregistration artifacts only; it did not train, generate synthetic demonstrations, operate hardware, simulate physics, change historical files or push externally.
