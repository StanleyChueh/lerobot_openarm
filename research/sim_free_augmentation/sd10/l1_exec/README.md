# SD10-L1 execution package (implements the frozen contract; does not alter it)

Contract: `SD10_LEARNING_PROTOCOL.md` + sealed bundle `../learning_v1/` (seal SHA-256 `7c74ef44…7183`, independent
Codex pre-training decision **GO**). This directory only implements that contract and writes a new output namespace
`outputs/` (gitignored; small results force-added). Committed **before** the first optimizer update.

| File | Role |
|---|---|
| `l1lib.py` | paths, seal check, determinism, policy/processor construction, frozen batch sampler, source dataset |
| `build_frame_cache.py` | PyAV rgb24 decode of the 23 source captures + the 834 locked eval inputs; every eval frame checked against locked RGB hash/PTS/time base |
| `train_l1.py` | one fit (arm × seed), 1,000 updates |
| `eval_l1.py` | 834-input predictions per terminal checkpoint |
| `score_l1.py` | completeness gate, `protocol_tools.recording_scores` / `paired_relative`, decision hierarchy |
| `run_all.py` | pre-launch checks, nine fits sequentially on one GPU in the frozen order, then eval and scoring |

## Implementation choices (declared before launch)

- **LeRobot source:** imported from the registered tree `../../../../lerobot/src` at `530b253f` (clean). The venv
  wheel (`3f2c29ef`) is not used; the launcher asserts the import path.
- **Training loop:** a direct transcription of `lerobot_train.update_policy` without accelerate (single process):
  `loss, _ = policy(batch)` → `backward` → `clip_grad_norm_(all params, 10)` → `AdamW.step` → `zero_grad`.
  Optimizer over `requires_grad` parameters, lr/betas/eps/weight-decay from `training_config.json`. The registered rate
  `learning_rates.npy[i]` is written to the param group **before** update *i*; the launcher asserts it equals the
  pinned `CosineDecayWithWarmupSchedulerConfig(33, 1000, 1e-4, 2.5e-6)` sequence exactly.
- **Policy/processors:** `PreTrainedConfig.from_pretrained(smolvla_base@5e8d12a6)` with every `policy_overrides`
  field applied, `SmolVLAPolicy.from_pretrained(..., config=cfg)`, processors rebuilt by
  `make_smolvla_pre_post_processors(cfg, dataset_stats=<frozen arm stats>)`. Normalizer/unnormalizer tensors are
  asserted bit-equal to `training_normalization.json` at train and eval time; eps 1e-8.
- **Precision:** no autocast/AMP in training; TF32 off (matmul + cuDNN), cuDNN benchmark off, deterministic
  algorithms in warn-only mode with `CUBLAS_WORKSPACE_CONFIG=:4096:8`. **Parameter storage dtype is the pinned
  checkpoint's:** SmolVLA stores the VLM and most expert matrices in **bfloat16** (96.6 M of 99.9 M trainable
  parameters; projections in float32), identical to `smolvla_base` and the historical v4B checkpoints. This is
  disclosed, not altered.
- **Data:** batch *i* is exactly row *i* of `sampling_seed<seed>.npz` (`<arm>_episode`, `<arm>_frame`); every
  consumed batch is asserted equal to the frozen row. Inputs: native body RGB (uint8 → float32/255, renamed
  `observation.images.camera1`) and state at row *t*; targets `action[t:t+50]`; `action_is_pad` all False. Four
  workers, prefetch 2, no shuffle, no augmentation, no wrist cameras, no synthetic data.
- **Evaluation:** batch 32, bf16 autocast, stored per-input noise, `policy.reset()` per batch, saved processors,
  float32 `(834,50,16)` archives. Repeated inference on a smoke checkpoint was bit-identical.
- **Terminal state:** `pretrained_model/` (weights, config, processors), `training_state/` (optimizer state, RNG
  state, schedule position, Adam step counters), `terminal.json` (hashes, consumed-source counts, warnings).
