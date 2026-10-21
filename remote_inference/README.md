# Async (server/client) evaluation on update_dev_v2 (`.venv-v2`)

The fork's lerobot (`update_dev_v2`, installed editable in `.venv-v2`) removed `lerobot.async_inference`, so
`python -m lerobot_robot_openarm_umeow.policy_server` / `.robot_client` only run in `.venv` (lerobot 3f2c29ef).
In `.venv-v2` the same job is done by lerobot's own **remote inference**: `lerobot-policy-server` on the GPU
and `lerobot-rollout --inference.type=remote` on the robot. It is needed for the policies only the fork has
(LaWAM, DM05, FineART-VLA, FLUX 3) and works for the others too. lerobot's guide:
`~/Stanley_ws/lerobot_experiment/lerobot/docs/source/remote_inference.mdx`.

What our old server patched is built in here: server-side guided RTC, the checkpoint's camera names
(`--rename_map` on the client), no forced image resize for SmolVLA, and request timing by remaining playback
(`refill_seconds`) instead of the "observation too similar" check.

```bash
cd ~/Stanley_ws/lerobot_openarm && source .venv-v2/bin/activate   # never `uv sync` / `uv run` here
```

## 1. Checkpoint (once per model)

A SmolVLA fine-tuned from `lerobot/smolvla_base` keeps the base model's input features in its config
(`observation.state` [6]); the server checks them against the robot schema and refuses it. Make a corrected
local copy (metadata only; weights and stats unchanged):

```bash
python remote_inference/fix_checkpoint_features.py ethanCSL/openarm_pringles_lerobot_real_smolvla_20_v00 \
  outputs/remote_checkpoints/openarm_pringles_lerobot_real_smolvla_20_v00
```

Policies trained from the dataset's own features (GR00T, LaWAM, ...) do not need this; the script prints
"input features already match the dataset" for them.

## 2. Server (terminal 1)

```bash
lerobot-policy-server --config_path=remote_inference/openarm_pringles_smolvla.yaml
```

Wait for `Policy server ready: deployment=openarm ...`. The YAML holds the checkpoint path, guided RTC
(execution horizon 10, guidance weight 10, EXP -- the old server's flags) and the robot schema (16 joints in
rad, three 480x640 cameras under the checkpoint's names). Another checkpoint with the same schema:
`--model.repo_or_path=...`. A chunk-only policy: `--execution.supported_modes='[chunk]' --execution.rtc.enabled=false`.

## 3. Robot (terminal 2)

```bash
lerobot-rollout --strategy.type=episodic \
  --inference.type=remote --inference.endpoint=tcp/127.0.0.1:7447 --inference.deployment=openarm \
  --inference.mode=rtc_guided --inference.semantics=openarm-umeow-rad-v1 \
  --rename_map='{"observation.images.right_wrist_cam": "observation.images.camera1", "observation.images.wrist_cam": "observation.images.camera2", "observation.images.body_cam": "observation.images.camera3"}' \
  --robot.type=openarm_umeow --robot.right_port=can0 --robot.left_port=can1 \
  --robot.cameras="{body_cam: {type: opencv, index_or_path: /dev/rs_body, width: 640, height: 480, fps: 30}, wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_left, width: 640, height: 480, fps: 30}, right_wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_right, width: 640, height: 480, fps: 30}}" \
  --fps=30 --task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --dataset.repo_id=ethanCSL/rollout_smolvla_pringles_remote_v00 --dataset.push_to_hub=false \
  --dataset.single_task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --dataset.num_episodes=100 --dataset.episode_time_s=12 --dataset.reset_time_s=5
```

`--inference.mode` must be one the server serves (`rtc_guided` here; `chunk` for a chunk-only server). The
`--rename_map` is the one in the checkpoint's `train_config.json`; drop it for a policy trained on the robot's
own camera names. The episodic strategy records each episode into the local dataset (name starts with
`rollout_`), with the slow return to the start pose between episodes. For one open-ended run instead:
`--strategy.type=base --duration=60` and no `--dataset.*`.

Old robot_client flags have no 1:1 equivalent: `--chunk_size_threshold` / `--aggregate_fn_name` are replaced by
`--inference.refill_seconds` (default 0.5 s of remaining playback) and, for plain chunks,
`--inference.chunk_merge` / `--inference.blend_*` (see lerobot's guide).

While no action is available (first chunk, starvation, `/vqa`) the arm holds its last commanded target
(`OpenArmUmeow.supports_position_hold`). A server that is lost is not reconnected: restart the rollout.
