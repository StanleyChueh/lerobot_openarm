# SmolVLA 6D/16D state audit (Phase A)

Date: 2026-10-24. Read-only audit. No training, no hardware, no change to LeRobot or the checkpoints.

**Verdict: PASS. The "6D state" is stale metadata. At run time the 16‑D OpenArm state is normalized
and zero‑padded to 32 dimensions, and all 16 dimensions reach `state_proj`, receive gradients and were
trained. Nothing is truncated, dropped or normalized with the wrong statistics.**
The tool work (Phases B–H) is therefore unblocked.

Evidence: `research/dataset_diagnostics/audit_smolvla_state.py` (rerunnable) →
`research/dataset_diagnostics/smolvla_state_audit.json`.

## 1. Repository state (recorded, no branch switched)

| repo | branch | HEAD | working tree |
|---|---|---|---|
| `~/Stanley_ws/lerobot_openarm` | `official-lerobot-quest` | `249096010cdd0c1a3135fbfe73e6f24d4676a09b` | clean at start |
| `~/Stanley_ws/lerobot_experiment/lerobot` (fork) | `update_dev_v2` | `f10cbf97a0577fd4652dc55aff43eaf3c607a2df` | clean; `origin/update_dev_v2` is one commit ahead (`530b253f2`, README only) |

Environments: `lerobot_openarm/.venv-v2` imports the fork (editable,
`lerobot_experiment/lerobot/src/lerobot`). `lerobot_openarm/.venv` is lerobot 0.6.2 (3f2c29ef), the
older eval env.

## 2. Training provenance

| checkpoint (Hub SHA) | dataset (Hub SHA, `main` = `v3.0` tag) | episodes arg | batch | steps | pushed |
|---|---|---|---|---|---|
| `real_smolvla_20_v00` (`cd6907a4`) | `openarm_pringles_lerobot_real_20_v00` (`dad0defb`) | `null` (all) | **16** | 20 000 | 2026‑10‑20 12:05 UTC |
| `real_smolvla_10_v00` (`8593bf25`) | `openarm_pringles_lerobot_real_10_v00` (`cebdee98`) | `null` (all) | **8** | 20 000 | 2026‑10‑21 13:11 UTC |
| `sim_smolvla_10_v00` (`9ce7ce48`) | `openarm_pringles_lerobot_sim_10_v00` (`b6d39dae`) | `null` (all) | **8** | 20 000 | 2026‑10‑21 13:12 UTC |

- **Training episodes.** All three used every episode of their dataset. Each normalizer's `count` equals
  the dataset's `total_frames` (5080 / 2575 / 2364), and `episode_index.max` = 19 / 9 / 9. The
  normalizer mean and std equal the dataset's `meta/stats.json` exactly (max |Δ| = 0.0). The
  datasets' `v3.0` tag equals `main` (no stale tag; compare memory note *Hub v3.0 tag pitfall*).
- **Episode IDs.** The training sets are datasets `{real_20: 0–19, real_10: 0–9, sim_10: 0–9}`. Which
  episodes of the original collection these subsets came from is **not recorded** in the dataset or
  checkpoint metadata, and no local script references them. That mapping is UNKNOWN. It does not
  affect the analysis, which runs on the training datasets themselves.
- **Source version.** `train_config.json` contains `streaming_native_http_subranges`. That field
  exists only in the fork, introduced by `ca69a2068` (2026‑10‑07), not in 3f2c29ef. So training used
  the update_dev_v2 lineage. The exact SHA is not recorded. Between `ca69a2068` and
  `f10cbf97a`, the SmolVLA, factory, normalizer and `pad_vector` code changed only in
  `ae2b57f12`, which added `validate_chunk_input_features` (serving-side only; it does not touch
  the training or inference math). Exact SHA: **UNKNOWN, but the training-path code is identical
  across the candidate window.**
- **Confound for Phase H.** real_20 used batch 16, the others batch 8, all for 20k steps. real_20
  therefore saw 2× as many samples (320k vs 160k). The 20‑vs‑10 comparison mixes dataset size with
  sample exposure.
- All three: `pretrained_path=lerobot/smolvla_base`, seed 1000, `image_transforms.enable=true`,
  identical `rename_map` (right_wrist_cam→camera1, wrist_cam→camera2, body_cam→camera3).
  real_10 and sim_10 `train_config.json` are identical except for repo ids and job name.

## 3. Declared vs. actual shapes

| item | real_20 | real_10 | sim_10 |
|---|---|---|---|
| `config.input_features["observation.state"]` | **[6]** | **[6]** | **[6]** |
| `config.output_features["action"]` | [16] | [16] | [16] |
| dataset `observation.state` / `action` | [16] / [16] | [16] / [16] | [16] / [16] |
| normalizer `observation.state.mean` / `action.mean` | [16] / [16] | [16] / [16] | [16] / [16] |
| `max_state_dim` / `max_action_dim` | 32 / 32 | 32 / 32 | 32 / 32 |
| `model.state_proj.weight` | [960, 32] | [960, 32] | [960, 32] |
| `policy_preprocessor.json` normalizer `features.observation.state.shape` | [6] | [6] | [6] |

The `[6]` and `camera1/2/3 [3,256,256]` values are copied from `lerobot/smolvla_base`'s config. All
three cached base revisions declare `state [6]`, `action [6]`.

## 4. Where the `[6]` comes from (source trace, fork `f10cbf97a`)

1. `src/lerobot/policies/factory.py:303-305`:
   ```python
   cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
   if not cfg.input_features:
       cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}
   ```
   Output features are always overwritten from the dataset, giving action [16]. Input features are
   kept when the pretrained config already has them, so state stays at [6]. This explains why the
   two are inconsistent.
2. The normalizer is built with `stats=dataset.meta.stats` (`processor/normalize_processor.py:534`).
   Its `features` dict selects only *which keys* and *which norm mode* apply
   (`_normalize_observation`, `:281-288`). `_apply_transform` broadcasts the 16‑D stats tensors. The
   declared shape is never used to slice.
3. `modeling_smolvla.py:448-452` `prepare_state`: takes the last step if 3‑D, then
   `pad_vector(state, max_state_dim=32)` (`policies/common/vla_utils.py:116`). With
   `truncate=False` and 16 < 32, it zero‑pads. It never truncates.
4. `modeling_smolvla.py:547` `state_proj = nn.Linear(max_state_dim=32, 960)`; used at `:658` in
   `embed_prefix`. The state becomes one prefix token.
5. Actions: `prepare_action` pads 16→32. The output is unpadded with `actions[:, :, :action_feature.shape[0]]`
   (`:220`), using the correct [16] output feature.
6. The declared `[6]` is consumed in only one place: the serving validator
   `validate_chunk_input_features` (`modeling_smolvla.py:345-371`, fork `lerobot-policy-server`). It
   **refuses** the checkpoint. It cannot silently misuse it. This is already handled by
   `remote_inference/fix_checkpoint_features.py`.
7. Image metadata `[3,256,256]` is likewise unused. `resize_imgs_with_padding=[512,512]` resizes the
   480×640 frames at run time.

## 5. Runtime verification (CPU, real checkpoint + dataset frame 120)

Run on real_20 and sim_10 (real_10 has the identical config, checked statically):

| check | real_20 | sim_10 |
|---|---|---|
| raw `observation.state` | [16] | [16] |
| after checkpoint preprocessor | [1, 16] | [1, 16] |
| equals `(x − mean)/(std + 1e‑8)` with checkpoint stats | PASS | PASS |
| `state_proj` input | [1, 32] | [1, 32] |
| input[:16] == normalized state | PASS | PASS |
| input[16:32] all zero | PASS | PASS |
| `state_proj` output | [1, 960] | [1, 960] |
| `predict_action_chunk` | [1, 50, 16] | [1, 50, 16] |
| post‑processed action | [1, 16] | [1, 16] |
| `state_proj.weight.requires_grad` | True | True |
| grad norm, columns 0–15 / 16–31 | 7.13 / 0.0 | 5.11 / 0.0 |
| nonzero grad columns | 16 | 16 |
| trainable params (vision encoder) | 99.9 M (0) | 99.9 M (0) |

The sim_10 first predicted step tracks the dataset action at that frame within ~0.02 rad (sanity check only,
not an evaluation).

**Optimizer inclusion.** `use_policy_training_preset=true`, so `optim/factory.py:37` takes
`SmolVLAPolicy.get_optim_params()` (= `self.parameters()`, `modeling_smolvla.py:194`). AdamW
therefore receives `state_proj`. It requires grad because `train_state_proj=true`
(`modeling_smolvla.py:581-583`).

**Trained-weight evidence.** This is the strongest evidence. Comparing each checkpoint's
`state_proj.weight` with `smolvla_base`, all three cached base revisions have identical `state_proj`:

| | max abs Δ, columns 0–5 | columns 6–15 | columns 16–31 |
|---|---|---|---|
| real_20 | 0.028–0.040 | 0.028–0.047 | **0.0** |
| real_10 | 0.027–0.037 | 0.030–0.047 | **0.0** |
| sim_10 | 0.027–0.042 | 0.026–0.036 | **0.0** |

Columns 6–15 moved as much as columns 0–5, so dimensions 7–16 of the state were learned, not
ignored. Columns 16–31 stayed exactly at the base values, consistent with zero padding.

## 6. Ordering, units, gripper, real/sim/inference compatibility

- State and action names are identical in all three datasets, in the interleaved order
  `RJ1, LJ1, …, RJ8, LJ8` (index 14 = right gripper RJ8, 15 = left gripper LJ8). `robot_type` is
  `openarm_umeow` in all three. Arm joints are in rad. Grippers are in raw motor units:
  real state RJ8 ∈ [−1.26, −0.51], LJ8 ∈ [0.40, 1.19]; sim state RJ8 ∈ [−1.30, −0.66],
  LJ8 ∈ [0.53, 1.21]. Gripper actions are binary in all three ({−1.298, 0} and {0, 1.212}).
  Sim was collected through `lerobot-record` with the same schema. The older Isaac converter
  differences (memory *sim-vs-lerobot-schema-gap*) do **not** apply to these datasets.
- Each checkpoint's pre‑ and post‑processor stats are identical and equal its own dataset stats.
  **Real/sim normalization differs.** The sim_10 checkpoint normalizes with sim statistics, which
  differ strongly in some joints: LJ5 std 0.45 sim vs 0.13 real, LJ2 mean −0.27 vs −0.09, RJ4 mean
  1.80 vs 1.43. Deploying sim_10 on the real robot feeds real states through sim statistics.
  This is correct as specified (it is the checkpoint's own normalizer), but it is a real/sim
  distribution shift and should be reported as such, not as a processing defect. Co‑training
  needs merged stats (LeRobot recomputes them on `merge`).
- Inference: the plugin robot client and policy server build `observation.state` from the robot's
  `observation_features` in the robot's own key order. That order is the same `openarm_umeow`
  interleaved order that wrote the datasets. Memory *update-dev-v2-eval-compat* records that the
  SmolVLA `sample_actions` and normalizer code are unchanged between the eval venv and the fork.

## 7. Conclusions

| question | result |
|---|---|
| Stale metadata or execution error? | **Stale metadata only** (`factory.py:304`). |
| Dimensions truncated / dropped? | **No.** PASS: 16 used, 16 zero‑padded. |
| State normalization correct? | **PASS.** Dataset stats, 16‑D, mean/std as configured. |
| `state_proj` trainable, in optimizer, gets gradient? | **PASS** on all three counts. |
| 6‑D pretrained config adapted correctly to 16‑D? | **PASS** at run time. The config file mis‑declares it. |
| Ordering/units consistent real vs sim? | **PASS** (names, order, units). Distributions differ (WARN). |
| Defect requiring LeRobot changes? | **None.** No architecture or normalization file was modified. |

Remaining cosmetic or operational issue (WARN, not a defect): checkpoint `config.json` and
`policy_preprocessor.json` declare `state [6]` and `camera* [3,256,256]`. This breaks strict
validators (fork `lerobot-policy-server`) and misleads readers. Workaround: `fix_checkpoint_features.py`.
Upstream fix option, **not applied** because it touches LeRobot core: set
`input_features` from the dataset when the pretrained shapes disagree with the dataset. The new analyzer
reports this mismatch automatically (`--policy-checkpoint`).
