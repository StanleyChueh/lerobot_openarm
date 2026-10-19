# This is the LeRobot-compatible VLA Training Pipeline

![Real and sim data collection, training and evaluation pipeline](docs/pipeline.svg)

Collect data on the real robot (Steps 0-5) or in Isaac Sim ([Sim Steps 1-6](#isaac-sim---robottypeopenarm_isaac)),
train, then evaluate on the real arm or in sim.

```
Quest app (UDP :5006) -> openarm_quest teleop (mink IK) -> lerobot-record (30 Hz) -> openarm_umeow robot -> CAN
```

> ⚠️ Do **not** use the official `--robot.type=openarm_follower` / `lerobot-calibrate` on this robot: its
> calibration writes a new zero into the motors and breaks `calibration.json`.

### Step 0. One-time setup

```bash
cd ~/Stanley_ws/lerobot_openarm
git checkout official-lerobot-quest
uv sync                      # installs lerobot, both plugins and the IK libraries
```

Link to Quest3 Pro domain

```
sudo ip addr add 10.100.1.240/24 dev wlp7s0
```

Keep `~/Stanley_ws/dora-openarm-data-collection` next to this repo (the IK model is read from it).

### Step 1. CAN-FD links up (once per boot)

```bash
cd ~/Stanley_ws/openarm_can/setup && sudo ./my_arm
```

### Step 2. Teleoperate the real arm (no recording)

```bash
cd ~/Stanley_ws/lerobot_openarm
source .venv/bin/activate
unset PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
lerobot-teleoperate \
  --robot.type=openarm_umeow \
  --robot.right_port=can0 --robot.left_port=can1 \
  --teleop.type=openarm_quest \
  --fps=30
```

Type `YES` (the arms go home), press **X** in the headset to start, **Ctrl-C** to stop.

### Step 3. Record a dataset

```bash
cd ~/Stanley_ws/lerobot_openarm
source .venv/bin/activate
unset PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
lerobot-record \
  --robot.type=openarm_umeow --robot.right_port=can0 --robot.left_port=can1 \
  --robot.cameras="{body_cam: {type: opencv, index_or_path: /dev/rs_body, width: 640, height: 480, fps: 30}, wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_left, width: 640, height: 480, fps: 30}, right_wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_right, width: 640, height: 480, fps: 30}}" \
  --teleop.type=openarm_quest \
  --dataset.repo_id=ethanCSL/openarm_pringles_lerobot_real_v00 --dataset.no_stamp=true \
  --dataset.single_task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --dataset.num_episodes=50 --dataset.fps=30 \
  --dataset.episode_time_s=120 --dataset.reset_time_s=15 \
  --dataset.streaming_encoding=true --dataset.encoder_threads=2 \
  --display_data=true
```

Run the same command again to continue the local dataset (only a local copy is continued, never the Hub's). `--dataset.num_episodes` is the total to reach: with 21 episodes recorded, `--dataset.num_episodes=40` records 19 more. `--dataset.push_to_hub=false` keeps it local.

### Quest controls

| Quest | state | what happens |
|---|---|---|
| **1st X** | `HELD` | start recording (`LIVE`) |
| **2nd X** | `LIVE` | return home slowly, then save the episode |
| **Y** | `LIVE` / `PAUSED` / `HELD` (episode open) | discard the episode and return home |
| **Triggers** | `LIVE` | close the grippers |

Check the recorded data:

```bash
cd ~/Stanley_ws/lerobot_openarm
source .venv/bin/activate
unset PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
lerobot-dataset-viz --repo-id ethanCSL/openarm_pringles_lerobot_real_v00 --episode-index 0
```

### Step 4. Train

Training runs on the training machine with the lerobot fork
[`StanleyChueh/lerobot`](https://github.com/StanleyChueh/lerobot), branch `update_dev` (commit `1e5f2049e`);
this machine runs collection and evaluation with official lerobot `3f2c29ef`.
On a 16 GB GPU, SmolVLA needs `--batch_size=32 --steps=40000` (batch 64 runs out of memory).
`--dataset.image_transforms.enable=true` turns on image augmentation.

**SmolVLA**:

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=ethanCSL/openarm_pringles_lerobot_real_v00 \
  --rename_map='{"observation.images.right_wrist_cam": "observation.images.camera1", "observation.images.wrist_cam": "observation.images.camera2", "observation.images.body_cam": "observation.images.camera3"}' \
  --policy.repo_id=ethanCSL/openarm_pringles_lerobot_real_smolvla_v00 \
  --batch_size=64 --steps=20000 --policy.device=cuda \
  --output_dir=outputs/train/openarm_pringles_lerobot_real_smolvla_v00 \
  --job_name=openarm_pringles_lerobot_real_smolvla_v00 \
  --wandb.enable=true --dataset.image_transforms.enable=true --dataset.image_transforms.random_order=true --dataset.image_transforms.max_num_transforms=6
```

**GR00T N1.7** (does not fit a 16 GB GPU):

```bash
lerobot-train --policy.type=groot --policy.base_model_path=nvidia/GR00T-N1.7-3B \
  --policy.chunk_size=16 --policy.n_action_steps=16 \
  --policy.use_relative_actions=true --policy.relative_exclude_joints='["LJ8", "RJ8"]' \
  --dataset.repo_id=ethanCSL/openarm_pringles_lerobot_real_v00 \
  --policy.repo_id=ethanCSL/openarm_pringles_lerobot_real_gr00t_v00 \
  --batch_size=64 --steps=20000 --policy.device=cuda \
  --output_dir=outputs/train/openarm_pringles_lerobot_real_gr00t_v00 --job_name=openarm_pringles_lerobot_real_gr00t_v00 \
  --wandb.enable=true --dataset.image_transforms.enable=true --dataset.image_transforms.random_order=true --dataset.image_transforms.max_num_transforms=6
```

**pi0.5**:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True lerobot-train \
  --policy.path=lerobot/pi05_base \
  --dataset.repo_id=ethanCSL/openarm_pringles_lerobot_real_v00 \
  --rename_map='{"observation.images.body_cam": "observation.images.base_0_rgb", "observation.images.wrist_cam": "observation.images.left_wrist_0_rgb", "observation.images.right_wrist_cam": "observation.images.right_wrist_0_rgb"}' \
  --policy.dtype=bfloat16 --policy.gradient_checkpointing=true \
  --policy.repo_id=ethanCSL/openarm_pringles_lerobot_real_pi05_v00 \
  --batch_size=32 --steps=30000 --policy.device=cuda \
  --output_dir=outputs/train/openarm_pringles_lerobot_real_pi05_v00 --job_name=openarm_pringles_lerobot_real_pi05_v00 \
  --wandb.enable=true --dataset.image_transforms.enable=true --dataset.image_transforms.random_order=true --dataset.image_transforms.max_num_transforms=6
```

### Step 5. Evaluate on the real robot: normal, RTC, async, async + RTC

| mode | tool |
|---|---|
| normal | `lerobot-rollout` |
| RTC | `lerobot-rollout --inference.type=rtc` |
| async | `policy_server` + `robot_client` |
| async + RTC | `policy_server --rtc=true` + `robot_client` |

Models: SmolVLA `ethanCSL/openarm_pringles_lerobot_real_smolvla_v00`, GR00T N1.7
`ethanCSL/openarm_pringles_lerobot_real_gr00t_v00`, pi0.5 `ethanCSL/openarm_pringles_lerobot_real_pi05_v00`.

During an episode: **Enter** ends it, **Left arrow** reruns it, **Esc** stops. `--display_data=false` turns
off the rerun window.

#### Normal

`--dataset.repo_id` must start with `rollout_`.

```bash
# SmolVLA
lerobot-rollout --strategy.type=episodic \
  --policy.path=ethanCSL/openarm_pringles_lerobot_real_smolvla_v00 \
  --robot.type=openarm_umeow --robot.right_port=can0 --robot.left_port=can1 \
  --robot.cameras="{body_cam: {type: opencv, index_or_path: /dev/rs_body, width: 640, height: 480, fps: 30}, wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_left, width: 640, height: 480, fps: 30}, right_wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_right, width: 640, height: 480, fps: 30}}" \
  --rename_map='{"observation.images.right_wrist_cam": "observation.images.camera1", "observation.images.wrist_cam": "observation.images.camera2", "observation.images.body_cam": "observation.images.camera3"}' \
  --task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --dataset.repo_id=ethanCSL/rollout_openarm_pringles_lerobot_real_smolvla_v00 --dataset.push_to_hub=false \
  --dataset.single_task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --dataset.num_episodes=100 --dataset.episode_time_s=15 --dataset.reset_time_s=5 \
  --display_data=true

# GR00T N1.7
lerobot-rollout --strategy.type=episodic \
  --policy.path=ethanCSL/openarm_pringles_lerobot_real_gr00t_v00 \
  --robot.type=openarm_umeow --robot.right_port=can0 --robot.left_port=can1 \
  --robot.cameras="{body_cam: {type: opencv, index_or_path: /dev/rs_body, width: 640, height: 480, fps: 30}, wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_left, width: 640, height: 480, fps: 30}, right_wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_right, width: 640, height: 480, fps: 30}}" \
  --task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --dataset.repo_id=ethanCSL/rollout_openarm_pringles_lerobot_real_gr00t_v00 --dataset.push_to_hub=false \
  --dataset.single_task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --dataset.num_episodes=100 --dataset.episode_time_s=15 --dataset.reset_time_s=5 \
  --display_data=true

# pi0.5
lerobot-rollout --strategy.type=episodic \
  --policy.path=ethanCSL/openarm_pringles_lerobot_real_pi05_v00 \
  --robot.type=openarm_umeow --robot.right_port=can0 --robot.left_port=can1 \
  --robot.cameras="{body_cam: {type: opencv, index_or_path: /dev/rs_body, width: 640, height: 480, fps: 30}, wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_left, width: 640, height: 480, fps: 30}, right_wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_right, width: 640, height: 480, fps: 30}}" \
  --rename_map='{"observation.images.body_cam": "observation.images.base_0_rgb", "observation.images.wrist_cam": "observation.images.left_wrist_0_rgb", "observation.images.right_wrist_cam": "observation.images.right_wrist_0_rgb"}' \
  --task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --dataset.repo_id=ethanCSL/rollout_openarm_pringles_lerobot_real_pi05_v00 --dataset.push_to_hub=false \
  --dataset.single_task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --dataset.num_episodes=100 --dataset.episode_time_s=15 --dataset.reset_time_s=5 \
  --display_data=true
```

#### RTC

pi0.5 on the real arm: use async + RTC instead (here it gets no actions through).

```bash
# SmolVLA
lerobot-rollout --strategy.type=episodic \
  --inference.type=rtc --inference.rtc.mode=guided \
  --inference.rtc.execution_horizon=10 --inference.rtc.max_guidance_weight=10.0 \
  --policy.path=ethanCSL/openarm_pringles_lerobot_real_smolvla_v00 \
  --robot.type=openarm_umeow --robot.right_port=can0 --robot.left_port=can1 \
  --robot.cameras="{body_cam: {type: opencv, index_or_path: /dev/rs_body, width: 640, height: 480, fps: 30}, wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_left, width: 640, height: 480, fps: 30}, right_wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_right, width: 640, height: 480, fps: 30}}" \
  --rename_map='{"observation.images.right_wrist_cam": "observation.images.camera1", "observation.images.wrist_cam": "observation.images.camera2", "observation.images.body_cam": "observation.images.camera3"}' \
  --task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --dataset.repo_id=ethanCSL/rollout_smolvla_rtc_pringles_v00 --dataset.push_to_hub=false \
  --dataset.single_task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --dataset.num_episodes=10 --dataset.episode_time_s=60 --dataset.reset_time_s=60 \
  --display_data=true

# GR00T N1.7
lerobot-rollout --strategy.type=episodic \
  --inference.type=rtc --inference.rtc.mode=guided \
  --inference.rtc.execution_horizon=8 --inference.rtc.max_guidance_weight=10.0 \
  --inference.queue_threshold=8 \
  --policy.path=ethanCSL/openarm_pringles_lerobot_real_gr00t_v00 \
  --robot.type=openarm_umeow --robot.right_port=can0 --robot.left_port=can1 \
  --robot.cameras="{body_cam: {type: opencv, index_or_path: /dev/rs_body, width: 640, height: 480, fps: 30}, wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_left, width: 640, height: 480, fps: 30}, right_wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_right, width: 640, height: 480, fps: 30}}" \
  --task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --dataset.repo_id=ethanCSL/rollout_groot_rtc_pringles_v00 --dataset.push_to_hub=false \
  --dataset.single_task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --dataset.num_episodes=10 --dataset.episode_time_s=60 --dataset.reset_time_s=60 \
  --display_data=true

# pi0.5
lerobot-rollout --strategy.type=episodic \
  --inference.type=rtc --inference.rtc.mode=guided \
  --inference.rtc.execution_horizon=20 --inference.rtc.max_guidance_weight=10.0 \
  --policy.path=ethanCSL/openarm_pringles_lerobot_real_pi05_v00 \
  --robot.type=openarm_umeow --robot.right_port=can0 --robot.left_port=can1 \
  --robot.cameras="{body_cam: {type: opencv, index_or_path: /dev/rs_body, width: 640, height: 480, fps: 30}, wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_left, width: 640, height: 480, fps: 30}, right_wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_right, width: 640, height: 480, fps: 30}}" \
  --rename_map='{"observation.images.body_cam": "observation.images.base_0_rgb", "observation.images.wrist_cam": "observation.images.left_wrist_0_rgb", "observation.images.right_wrist_cam": "observation.images.right_wrist_0_rgb"}' \
  --task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --dataset.repo_id=ethanCSL/rollout_pi05_rtc_pringles_v00 --dataset.push_to_hub=false \
  --dataset.single_task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --dataset.num_episodes=10 --dataset.episode_time_s=60 --dataset.reset_time_s=60 \
  --display_data=true
```

#### Async

Terminal 1, the policy server (keep it running across episodes and evaluations):

```bash
python -m lerobot_robot_openarm_umeow.policy_server --host=127.0.0.1 --port=8080
```

Terminal 2, the robot client:

```bash
# SmolVLA
python -m lerobot_robot_openarm_umeow.robot_client \
  --server_address=127.0.0.1:8080 \
  --robot.type=openarm_umeow --robot.right_port=can0 --robot.left_port=can1 \
  --robot.cameras="{body_cam: {type: opencv, index_or_path: /dev/rs_body, width: 640, height: 480, fps: 30}, wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_left, width: 640, height: 480, fps: 30}, right_wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_right, width: 640, height: 480, fps: 30}}" \
  --task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --policy_type=smolvla --pretrained_name_or_path=ethanCSL/openarm_pringles_lerobot_real_smolvla_v00 \
  --policy_device=cuda --actions_per_chunk=50 --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average --fps=30 \
  --num_episodes=10 --episode_time_s=60 --reset_time_s=60

# GR00T N1.7
python -m lerobot_robot_openarm_umeow.robot_client \
  --server_address=127.0.0.1:8080 \
  --robot.type=openarm_umeow --robot.right_port=can0 --robot.left_port=can1 \
  --robot.cameras="{body_cam: {type: opencv, index_or_path: /dev/rs_body, width: 640, height: 480, fps: 30}, wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_left, width: 640, height: 480, fps: 30}, right_wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_right, width: 640, height: 480, fps: 30}}" \
  --task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --policy_type=groot --pretrained_name_or_path=ethanCSL/openarm_pringles_lerobot_real_gr00t_v00 \
  --policy_device=cuda --actions_per_chunk=16 --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average --fps=30 \
  --num_episodes=10 --episode_time_s=60 --reset_time_s=60

# pi0.5
python -m lerobot_robot_openarm_umeow.robot_client \
  --server_address=127.0.0.1:8080 \
  --robot.type=openarm_umeow --robot.right_port=can0 --robot.left_port=can1 \
  --robot.cameras="{body_cam: {type: opencv, index_or_path: /dev/rs_body, width: 640, height: 480, fps: 30}, wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_left, width: 640, height: 480, fps: 30}, right_wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_right, width: 640, height: 480, fps: 30}}" \
  --task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --policy_type=pi05 --pretrained_name_or_path=ethanCSL/openarm_pringles_lerobot_real_pi05_v00 \
  --policy_device=cuda --actions_per_chunk=50 --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average --fps=30 \
  --num_episodes=10 --episode_time_s=60 --reset_time_s=60
```

Use our `lerobot_robot_openarm_umeow.policy_server` / `robot_client`, not lerobot's own.

#### Async + RTC

Terminal 1:

```bash
# SmolVLA
python -m lerobot_robot_openarm_umeow.policy_server --host=127.0.0.1 --port=8080 \
  --rtc=true --rtc_execution_horizon=10 --rtc_max_guidance_weight=10.0 --rtc_prefix_attention_schedule=EXP

# GR00T N1.7
python -m lerobot_robot_openarm_umeow.policy_server --host=127.0.0.1 --port=8080 \
  --rtc=true --rtc_execution_horizon=8 --rtc_max_guidance_weight=10.0 --rtc_prefix_attention_schedule=EXP

# pi0.5
python -m lerobot_robot_openarm_umeow.policy_server --host=127.0.0.1 --port=8080 \
  --rtc=true --rtc_execution_horizon=20 --rtc_max_guidance_weight=10.0 --rtc_prefix_attention_schedule=EXP
```

Terminal 2:

```bash
# SmolVLA
python -m lerobot_robot_openarm_umeow.robot_client \
  --server_address=127.0.0.1:8080 \
  --robot.type=openarm_umeow --robot.right_port=can0 --robot.left_port=can1 \
  --robot.cameras="{body_cam: {type: opencv, index_or_path: /dev/rs_body, width: 640, height: 480, fps: 30}, wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_left, width: 640, height: 480, fps: 30}, right_wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_right, width: 640, height: 480, fps: 30}}" \
  --task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --policy_type=smolvla --pretrained_name_or_path=ethanCSL/openarm_pringles_lerobot_real_smolvla_v00 \
  --policy_device=cuda --actions_per_chunk=50 --chunk_size_threshold=0.5 \
  --aggregate_fn_name=latest_only --fps=30 \
  --num_episodes=10 --episode_time_s=60 --reset_time_s=60

# GR00T N1.7
python -m lerobot_robot_openarm_umeow.robot_client \
  --server_address=127.0.0.1:8080 \
  --robot.type=openarm_umeow --robot.right_port=can0 --robot.left_port=can1 \
  --robot.cameras="{body_cam: {type: opencv, index_or_path: /dev/rs_body, width: 640, height: 480, fps: 30}, wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_left, width: 640, height: 480, fps: 30}, right_wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_right, width: 640, height: 480, fps: 30}}" \
  --task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --policy_type=groot --pretrained_name_or_path=ethanCSL/openarm_pringles_lerobot_real_gr00t_v00 \
  --policy_device=cuda --actions_per_chunk=16 --chunk_size_threshold=0.5 \
  --aggregate_fn_name=latest_only --fps=30 \
  --num_episodes=10 --episode_time_s=60 --reset_time_s=60

# pi0.5
python -m lerobot_robot_openarm_umeow.robot_client \
  --server_address=127.0.0.1:8080 \
  --robot.type=openarm_umeow --robot.right_port=can0 --robot.left_port=can1 \
  --robot.cameras="{body_cam: {type: opencv, index_or_path: /dev/rs_body, width: 640, height: 480, fps: 30}, wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_left, width: 640, height: 480, fps: 30}, right_wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_right, width: 640, height: 480, fps: 30}}" \
  --task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --policy_type=pi05 --pretrained_name_or_path=ethanCSL/openarm_pringles_lerobot_real_pi05_v00 \
  --policy_device=cuda --actions_per_chunk=50 --chunk_size_threshold=0.5 \
  --aggregate_fn_name=latest_only --fps=30 \
  --num_episodes=10 --episode_time_s=60 --reset_time_s=60
```

#### Timing report

Every evaluation saves a timing report to `outputs/eval_logs/<date>_<mode>_<policy>/` (table + `report.png`).
Compare runs:

```bash
python -m lerobot_robot_openarm_umeow.eval_report              # latest run of each mode and policy
python -m lerobot_robot_openarm_umeow.eval_report outputs/eval_logs/RUN_A outputs/eval_logs/RUN_B
```

## Isaac Sim (`--robot.type=openarm_isaac`)

Same commands as the real arm, with `--robot.type=openarm_isaac` in place of
`--robot.type=openarm_umeow --robot.right_port=can0 --robot.left_port=can1 --robot.cameras=...`.
No CAN step, no `YES`.

The simulator bridge and the renderer are shared across tasks. Pick a task cfg for the movable objects,
contacts, reset rules and success condition, and a `profile.json` folder for the dataset-fitted room,
robot appearance and camera response. A profile lists its source dataset and compatible task IDs; the
server rejects a mismatched combination. The two profiles currently shipped with IsaacLab are:

| Real dataset | Simulator task | Profile folder |
| --- | --- | --- |
| `ethanCSL/openarm_pringles_lerobot_real_v00` | `Isaac-PickUp-RedCube-OpenArm-IK-Abs-v1` | `source/isaaclab_assets/data/real2sim/pringles_v00` |
| `ethanCSL/openarm_insertbattery_lerobot_real_v01` | `Isaac-InsertBattery-OpenArm-IK-Abs-v0` | `source/isaaclab_assets/data/real2sim/battery_v01` |

The task cfgs already select those profile folders. For a new task, build its object geometry and reset
logic, fit a profile from that task's real frames, and set `real_look_dir` on the task cfg. The shared
`lerobot_sim_server.py` and `openarm_real_look.py` do not need a new rendering implementation. An explicit
`--real_look /path/to/profile` works for a task with no built-in profile. A profile cannot supply a
new collision shape or success condition; those must be validated separately for each interaction.
The task-by-task process and current real-to-sim research are summarized in
[docs/real2sim2real_tasks.md](docs/real2sim2real_tasks.md).

### Pringles Step 1. Start the simulator (terminal 1, keep it running)

```bash
cd ~/Stanley_ws/IsaacLab
conda activate env_isaaclab
./isaaclab.sh -p scripts/tools/lerobot_sim_server.py --enable_cameras --headless \
  --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v1 --task_mode handover \
  --domain_randomization none \
  --mimic_hdf5 logs/demos/pringles_sim_src_v01.hdf5
```

Wait for `[lerobot_sim_server] ready on 127.0.0.1:5710`. Leave out `--mimic_hdf5` for teleoperation and
evaluation. The `-v1` task applies the fitted real look (robot colours, photo backdrop, lighting and camera
sensor response) and fitted contact physics from the Pringles real data. The old `-v0` command rendered the
plain sim scene; `-v0 --real_look source/isaaclab_assets/data/real2sim/pringles_v00` selects the same
profile explicitly. Use `--domain_randomization none` while
recording Mimic **source** demos: the annotation stage replays their states in the unrandomized task. See
`~/Stanley_ws/IsaacLab/scripts/tools/sysid/README.md` for the look calibration and its measured limits.

### Pringles Step 2. Teleoperate in sim (no recording)

```bash
lerobot-teleoperate \
  --robot.type=openarm_isaac \
  --teleop.type=openarm_quest \
  --fps=30 --display_data=true
```

### Pringles Step 3. Record sim demos (lerobot dataset + Mimic source demos)

```bash
lerobot-record \
  --robot.type=openarm_isaac \
  --robot.expected_task=Isaac-PickUp-RedCube-OpenArm-IK-Abs-v1 \
  --robot.expected_task_mode=handover \
  --robot.expected_domain_randomization=none \
  --robot.expected_control_hz=30 \
  --robot.expected_real_dataset=ethanCSL/openarm_pringles_lerobot_real_v00 \
  --teleop.type=openarm_quest \
  --dataset.repo_id=ethanCSL/openarm_pringles_lerobot_sim_v01 --dataset.no_stamp=true \
  --dataset.single_task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --dataset.num_episodes=20 --dataset.fps=30 \
  --dataset.episode_time_s=120 --dataset.reset_time_s=1 \
  --dataset.streaming_encoding=true --dataset.encoder_threads=2 \
  --display_data=true
```

End each demo with the can in the LEFT hand, then press X.
The `expected_*` options stop the recording at connection if an older sim server is still running with a
different task or randomization profile. Use a new dataset repo ID for the real-look run; an existing
`..._sim_v00` dataset may contain plain `-v0` frames, and mixing the two appearances under one ID makes
the source of a training result ambiguous.
Watch the server's `steps/s` line during actual recording. If it reports slower than real time, the sim
still advances 1/30 s per command, but the Quest user's command is sampled more slowly in wall time;
try `--display_data=false` and reduce concurrent GPU work before collecting the full set. Check that
the recorded grasp and handover durations remain close to the real demonstrations. On this workstation,
the headless battery server with all three 640 × 480 cameras measured about 21–23 steps/s during replay;
live Quest timing should be measured again before collecting teleoperated training episodes.

### Battery insertion example

The battery task uses the **same** robot bridge, cameras and real-look renderer. Start its server from
`~/Stanley_ws/IsaacLab`:

```bash
./isaaclab.sh -p scripts/tools/lerobot_sim_server.py --enable_cameras --headless \
  --task Isaac-InsertBattery-OpenArm-IK-Abs-v0 --domain_randomization none
```

Then inspect the scene with the same Quest bridge from this repo:

```bash
lerobot-teleoperate \
  --robot.type=openarm_isaac \
  --robot.expected_task=Isaac-InsertBattery-OpenArm-IK-Abs-v0 \
  --robot.expected_control_hz=30 \
  --robot.expected_real_dataset=ethanCSL/openarm_insertbattery_lerobot_real_v01 \
  --teleop.type=openarm_quest \
  --fps=30 --display_data=true
```

The leftmost charger bay now has an open compound collider and a success check requiring both objects
to be lifted and the battery to stay seated for 15 steps. Its slot dimensions were estimated from the
real image, not measured from CAD. The task is not yet validated for insertion data: image-aligned
replays of eight real episodes lifted the charger in seven and the battery in only one under the best
bounded contact fit; none has passed insertion. Do not label a battery sim dataset successful until
physical replay or teleoperated episodes pass the success check. Its `profile.json` contains no
Pringles contact fit. Do not apply the Pringles Mimic instructions below to this task.

### Physically validated replay dataset

For Pringles, three real action trajectories have passed the simulator's stageful right-to-left
handover check. The local LeRobot dataset is
`~/Stanley_ws/lerobot_openarm/datasets/ethanCSL/openarm_pringles_lerobot_sim_replay_v01`
(three episodes, 778 frames at 30 Hz). `validation.json` identifies source real episodes 0, 4 and 5,
their start poses, and final simulated object poses. The three 640 × 480 video streams and 16 motor
action/state values match the real dataset schema. This is a replay dataset, not evidence that a
policy trained on it will transfer to the real robot.

To reproduce a new local dataset, start the Pringles `-v1` server above without `--mimic_hdf5`, then
run from this repo with its Python environment:

```bash
python collect_validated_sim_replay.py --task pringles --episodes 0,4,5 \
  --starts artifacts/pringles_real_refs/can_starts_all.json \
  --repo_id ethanCSL/openarm_pringles_lerobot_sim_replay_v01 \
  --root datasets/ethanCSL/openarm_pringles_lerobot_sim_replay_v01
python compare_sim_real_schema.py \
  --real ethanCSL/openarm_pringles_lerobot_real_v00 \
  --sim datasets/ethanCSL/openarm_pringles_lerobot_sim_replay_v01
```

Choose a fresh `--root` for each run. The collector discards failed episodes and records only
simulated observations paired with the replayed real motor commands. For battery insertion, use
`--task battery`, the battery task server, and the starts in
`artifacts/battery_real_refs/object_starts.json` only after a candidate has passed the slot success
check. The charger geometry and finger pads need measurement and further contact fitting.

### Pringles Step 4. Generate a synthetic dataset with Isaac Lab Mimic

Stop the sim server first (Ctrl-C), then:

```bash
cd ~/Stanley_ws/IsaacLab && conda activate env_isaaclab

# 4a. Annotate
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-Mimic-v0 \
  --task_mode handover --auto --from_states \
  --enable_cameras --headless \
  --input_file logs/demos/pringles_sim_src_v01.hdf5 \
  --output_file logs/demos/pringles_sim_src_v01_annotated.hdf5

# 4b. Generate (with domain randomization)
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-Mimic-v0 --task_mode handover \
  --input_file logs/demos/pringles_sim_src_v01_annotated.hdf5 \
  --output_file logs/demos/pringles_sim_mimic_v01.hdf5 \
  --generation_num_trials 500 --num_envs 4 --enable_cameras --headless \
  --enable_domain_randomization --domain_randomization_profile visual
```

Convert to a lerobot dataset (lerobot terminal):

```bash
python mimic_to_lerobot.py \
  --hdf5 ~/Stanley_ws/IsaacLab/logs/demos/pringles_sim_mimic_v01.hdf5 \
  --repo_id ethanCSL/openarm_pringles_lerobot_mimic_v01 --push_to_hub
```

### Pringles Step 5. Train: synthetic only, or co-train sim + real

Before merging, check the real and sim metadata (this downloads only `meta/info.json` if needed):

```bash
python compare_sim_real_schema.py \
  --real ethanCSL/openarm_pringles_lerobot_real_v00 \
  --sim ethanCSL/openarm_pringles_lerobot_sim_v01
python compare_sim_real_schema.py \
  --real ethanCSL/openarm_pringles_lerobot_real_v00 \
  --sim ethanCSL/openarm_pringles_lerobot_mimic_v01
```

The check requires matching LeRobot version, `fps`, `robot_type`, `action` and `observation.state` names/order, and all
three `observation.images.*` shapes. Metadata cannot verify physical units; confirm the action/state
conventions and calibration separately. Run
`python analyze_dataset_quality.py --root <real-dataset> <sim-dataset>` to
compare grasp timing, gripper holds and tracking. Keep a real episode holdout for evaluation and compare
real-only, real + real-look teleop, and real + Mimic training at the same total update count. A larger
synthetic set can overwhelm the real images; sample or weight it so real frames remain prominent.

For vision, use IsaacLab's `scripts/tools/sysid/eval_realism.py` on paired real frames and sim renders at
the same recorded states (body and both wrist cameras). It reports paired DINOv2 feature distance and a
real-vs-sim classifier AUC. The shipped real look improved those distances, but the reported classifier
AUC is still 1.0, so visual transfer must be checked with real rollouts. Inspect grasp contact and
handover success as well: the fitted contact parameters improved held-out replay tracking in IsaacLab's
sysid report, but were not measured physical material constants. If you change camera mounts, lighting,
can geometry or finger pads, rerun the paired vision and replay checks before collecting more data.

The Mimic `visual` profile above is a separate robustness experiment on the plain Mimic task. It does
not inherit the `-v1` appearance or fitted contact physics: its random light and pad appearance can
conflict with the photo backdrop. Keep its dataset separate until a held-out real rollout shows that
adding it helps.

```bash
# Optional co-training experiment after separate real-only and real-look baselines
lerobot-edit-dataset --operation.type=merge \
  --operation.repo_ids='["ethanCSL/openarm_pringles_lerobot_real_v00", "ethanCSL/openarm_pringles_lerobot_mimic_v01", "ethanCSL/openarm_pringles_lerobot_sim_v01"]' \
  --new_repo_id=ethanCSL/openarm_pringles_lerobot_real_mimic_v01 --push_to_hub=true

# SmolVLA
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=ethanCSL/openarm_pringles_lerobot_real_mimic_v01 \
  --rename_map='{"observation.images.right_wrist_cam": "observation.images.camera1", "observation.images.wrist_cam": "observation.images.camera2", "observation.images.body_cam": "observation.images.camera3"}' \
  --policy.repo_id=ethanCSL/openarm_pringles_lerobot_real_mimic_smolvla_v01 \
  --batch_size=64 --steps=20000 --policy.device=cuda \
  --output_dir=outputs/train/openarm_pringles_lerobot_real_mimic_smolvla_v01 \
  --job_name=openarm_pringles_lerobot_real_mimic_smolvla_v01 \
  --wandb.enable=true

# GR00T N1.7
lerobot-train --policy.type=groot --policy.base_model_path=nvidia/GR00T-N1.7-3B \
  --policy.chunk_size=16 --policy.n_action_steps=16 \
  --policy.use_relative_actions=true --policy.relative_exclude_joints='["LJ8", "RJ8"]' \
  --dataset.repo_id=ethanCSL/openarm_pringles_lerobot_real_mimic_v01 \
  --policy.repo_id=ethanCSL/openarm_pringles_lerobot_real_mimic_gr00t_v01 \
  --batch_size=64 --steps=20000 --policy.device=cuda \
  --output_dir=outputs/train/openarm_pringles_lerobot_real_mimic_gr00t_v01 --job_name=openarm_pringles_lerobot_real_mimic_gr00t_v01 \
  --wandb.enable=true

# pi0.5
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True lerobot-train \
  --policy.path=lerobot/pi05_base \
  --dataset.repo_id=ethanCSL/openarm_pringles_lerobot_real_mimic_v01 \
  --rename_map='{"observation.images.body_cam": "observation.images.base_0_rgb", "observation.images.wrist_cam": "observation.images.left_wrist_0_rgb", "observation.images.right_wrist_cam": "observation.images.right_wrist_0_rgb"}' \
  --policy.dtype=bfloat16 --policy.gradient_checkpointing=true \
  --policy.repo_id=ethanCSL/openarm_pringles_lerobot_real_mimic_pi05_v01 \
  --batch_size=32 --steps=30000 --policy.device=cuda \
  --output_dir=outputs/train/openarm_pringles_lerobot_real_mimic_pi05_v01 --job_name=openarm_pringles_lerobot_real_mimic_pi05_v01 \
  --wandb.enable=true
```

### Pringles Step 6. Evaluate in sim: normal, RTC, async, async + RTC

The simulator from Pringles Step 1 (without `--mimic_hdf5`) in terminal 1, then:

**Normal**:

```bash
# SmolVLA
lerobot-rollout --strategy.type=episodic \
  --policy.path=ethanCSL/openarm_pringles_lerobot_real_smolvla_v00 \
  --robot.type=openarm_isaac \
  --rename_map='{"observation.images.right_wrist_cam": "observation.images.camera1", "observation.images.wrist_cam": "observation.images.camera2", "observation.images.body_cam": "observation.images.camera3"}' \
  --task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --dataset.repo_id=ethanCSL/rollout_smolvla_pringles_sim_v00 --dataset.push_to_hub=false \
  --dataset.single_task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --dataset.num_episodes=10 --dataset.episode_time_s=60 --dataset.reset_time_s=1 \
  --display_data=true

# GR00T N1.7
lerobot-rollout --strategy.type=episodic \
  --policy.path=ethanCSL/openarm_pringles_lerobot_real_gr00t_v00 \
  --robot.type=openarm_isaac \
  --task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --dataset.repo_id=ethanCSL/rollout_groot_pringles_sim_v00 --dataset.push_to_hub=false \
  --dataset.single_task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --dataset.num_episodes=10 --dataset.episode_time_s=60 --dataset.reset_time_s=1 \
  --display_data=true

# pi0.5
lerobot-rollout --strategy.type=episodic \
  --policy.path=ethanCSL/openarm_pringles_lerobot_real_pi05_v00 \
  --robot.type=openarm_isaac \
  --rename_map='{"observation.images.body_cam": "observation.images.base_0_rgb", "observation.images.wrist_cam": "observation.images.left_wrist_0_rgb", "observation.images.right_wrist_cam": "observation.images.right_wrist_0_rgb"}' \
  --task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --dataset.repo_id=ethanCSL/rollout_pi05_pringles_sim_v00 --dataset.push_to_hub=false \
  --dataset.single_task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --dataset.num_episodes=10 --dataset.episode_time_s=60 --dataset.reset_time_s=1 \
  --display_data=true
```

**RTC**:

```bash
# SmolVLA
lerobot-rollout --strategy.type=episodic \
  --inference.type=rtc --inference.rtc.mode=guided \
  --inference.rtc.execution_horizon=10 --inference.rtc.max_guidance_weight=10.0 \
  --policy.path=ethanCSL/openarm_pringles_lerobot_real_smolvla_v00 \
  --robot.type=openarm_isaac \
  --rename_map='{"observation.images.right_wrist_cam": "observation.images.camera1", "observation.images.wrist_cam": "observation.images.camera2", "observation.images.body_cam": "observation.images.camera3"}' \
  --task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --dataset.repo_id=ethanCSL/rollout_smolvla_rtc_pringles_sim_v00 --dataset.push_to_hub=false \
  --dataset.single_task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --dataset.num_episodes=10 --dataset.episode_time_s=60 --dataset.reset_time_s=1 \
  --display_data=true

# GR00T N1.7
lerobot-rollout --strategy.type=episodic \
  --inference.type=rtc --inference.rtc.mode=guided \
  --inference.rtc.execution_horizon=8 --inference.rtc.max_guidance_weight=10.0 \
  --inference.queue_threshold=8 \
  --policy.path=ethanCSL/openarm_pringles_lerobot_real_gr00t_v00 \
  --robot.type=openarm_isaac \
  --task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --dataset.repo_id=ethanCSL/rollout_groot_rtc_pringles_sim_v00 --dataset.push_to_hub=false \
  --dataset.single_task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --dataset.num_episodes=10 --dataset.episode_time_s=60 --dataset.reset_time_s=1 \
  --display_data=true

# pi0.5
lerobot-rollout --strategy.type=episodic \
  --inference.type=rtc --inference.rtc.mode=guided \
  --inference.rtc.execution_horizon=20 --inference.rtc.max_guidance_weight=10.0 \
  --policy.path=ethanCSL/openarm_pringles_lerobot_real_pi05_v00 \
  --robot.type=openarm_isaac \
  --rename_map='{"observation.images.body_cam": "observation.images.base_0_rgb", "observation.images.wrist_cam": "observation.images.left_wrist_0_rgb", "observation.images.right_wrist_cam": "observation.images.right_wrist_0_rgb"}' \
  --task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --dataset.repo_id=ethanCSL/rollout_pi05_rtc_pringles_sim_v00 --dataset.push_to_hub=false \
  --dataset.single_task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --dataset.num_episodes=10 --dataset.episode_time_s=60 --dataset.reset_time_s=1 \
  --display_data=true
```

**Async**: terminal 2, the policy server:

```bash
python -m lerobot_robot_openarm_umeow.policy_server --host=127.0.0.1 --port=8080
```

Terminal 3, the robot client:

```bash
# SmolVLA
python -m lerobot_robot_openarm_umeow.robot_client \
  --server_address=127.0.0.1:8080 \
  --robot.type=openarm_isaac \
  --task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --policy_type=smolvla --pretrained_name_or_path=ethanCSL/openarm_pringles_lerobot_real_smolvla_v00 \
  --policy_device=cuda --actions_per_chunk=50 --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average --fps=30 \
  --num_episodes=10 --episode_time_s=60 --reset_time_s=1

# GR00T N1.7
python -m lerobot_robot_openarm_umeow.robot_client \
  --server_address=127.0.0.1:8080 \
  --robot.type=openarm_isaac \
  --task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --policy_type=groot --pretrained_name_or_path=ethanCSL/openarm_pringles_lerobot_real_gr00t_v00 \
  --policy_device=cuda --actions_per_chunk=16 --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average --fps=30 \
  --num_episodes=10 --episode_time_s=60 --reset_time_s=1

# pi0.5
python -m lerobot_robot_openarm_umeow.robot_client \
  --server_address=127.0.0.1:8080 \
  --robot.type=openarm_isaac \
  --task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --policy_type=pi05 --pretrained_name_or_path=ethanCSL/openarm_pringles_lerobot_real_pi05_v00 \
  --policy_device=cuda --actions_per_chunk=50 --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average --fps=30 \
  --num_episodes=10 --episode_time_s=60 --reset_time_s=1
```

**Async + RTC**: terminal 2, the policy server:

```bash
# SmolVLA
python -m lerobot_robot_openarm_umeow.policy_server --host=127.0.0.1 --port=8080 \
  --rtc=true --rtc_execution_horizon=10 --rtc_max_guidance_weight=10.0 --rtc_prefix_attention_schedule=EXP

# GR00T N1.7
python -m lerobot_robot_openarm_umeow.policy_server --host=127.0.0.1 --port=8080 \
  --rtc=true --rtc_execution_horizon=8 --rtc_max_guidance_weight=10.0 --rtc_prefix_attention_schedule=EXP

# pi0.5
python -m lerobot_robot_openarm_umeow.policy_server --host=127.0.0.1 --port=8080 \
  --rtc=true --rtc_execution_horizon=20 --rtc_max_guidance_weight=10.0 --rtc_prefix_attention_schedule=EXP
```

Terminal 3, the robot client:

```bash
# SmolVLA
python -m lerobot_robot_openarm_umeow.robot_client \
  --server_address=127.0.0.1:8080 \
  --robot.type=openarm_isaac \
  --task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --policy_type=smolvla --pretrained_name_or_path=ethanCSL/openarm_pringles_lerobot_real_smolvla_v00 \
  --policy_device=cuda --actions_per_chunk=50 --chunk_size_threshold=0.5 \
  --aggregate_fn_name=latest_only --fps=30 \
  --num_episodes=10 --episode_time_s=60 --reset_time_s=1

# GR00T N1.7
python -m lerobot_robot_openarm_umeow.robot_client \
  --server_address=127.0.0.1:8080 \
  --robot.type=openarm_isaac \
  --task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --policy_type=groot --pretrained_name_or_path=ethanCSL/openarm_pringles_lerobot_real_gr00t_v00 \
  --policy_device=cuda --actions_per_chunk=16 --chunk_size_threshold=0.5 \
  --aggregate_fn_name=latest_only --fps=30 \
  --num_episodes=10 --episode_time_s=60 --reset_time_s=1

# pi0.5
python -m lerobot_robot_openarm_umeow.robot_client \
  --server_address=127.0.0.1:8080 \
  --robot.type=openarm_isaac \
  --task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --policy_type=pi05 --pretrained_name_or_path=ethanCSL/openarm_pringles_lerobot_real_pi05_v00 \
  --policy_device=cuda --actions_per_chunk=50 --chunk_size_threshold=0.5 \
  --aggregate_fn_name=latest_only --fps=30 \
  --num_episodes=10 --episode_time_s=60 --reset_time_s=1
```
