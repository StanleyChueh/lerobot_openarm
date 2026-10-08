# This is the LeRobot-comparible VLA Training Pipeline

## Official LeRobot pipeline (Meta Quest)

```
Quest app (UDP :5006) -> openarm_quest teleop (mink IK) -> lerobot-record (30 Hz) -> openarm_umeow robot -> CAN
```

Two lerobot plugins in [`plugins/`](plugins/README.md) make this work:

| CLI flag | what it is |
|---|---|
| `--robot.type=openarm_umeow` | Our follower (`robots/umeow_openarm_follower`: gravity feed-forward, CAN fixes) with a safe start/stop, a jump guard and the gripper squeeze. Uses `calibration.json`; never re-zeroes the motors. |
| `--teleop.type=openarm_quest` | The Quest controllers -> the dora pipeline's pose mapping, smoothing and IK -> joint targets for the robot, with the reference captured on X, a slow return home and safety pauses. |

> ⚠️ Do **not** use the official `--robot.type=openarm_follower` / `lerobot-calibrate` on this robot: its
> calibration writes a new zero into the motors and breaks `calibration.json`.

### Step 0. One-time setup

```bash
cd ~/Stanley_ws/lerobot_openarm
git checkout official-lerobot-quest
uv sync                      # installs lerobot, both plugins and the IK libraries
hf auth login                # only if you push datasets / models to the Hub
```

The IK model is read from the dora checkout next to this one
(`~/Stanley_ws/dora-openarm-data-collection/.../scenes/v1_camera/scene.xml`), so keep that repo there.

### Step 1. Every session: prepare the terminal

```bash
# 1. CAN-FD links up (once per boot)
cd ~/Stanley_ws/openarm_can/setup && sudo ./my_arm

# 2. A terminal for lerobot, with ROS 2 kept out of it
cd ~/Stanley_ws/lerobot_openarm
source .venv/bin/activate
unset PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
```

### Step 2. Teleoperate the real arm (no recording)

Drive the real OpenArm with the Quest through the official `lerobot-teleoperate`. Use it to practise the
task and to check tracking before you record.

```bash
lerobot-teleoperate \
  --robot.type=openarm_umeow \
  --robot.right_port=can0 --robot.left_port=can1 \
  --teleop.type=openarm_quest \
  --fps=30
```

What happens:

1. The terminal shows each joint's current vs. home position. **Type `YES`**, and both arms move slowly
   (0.3 rad/s) to the home pose with the grippers open.
2. The arms hold home (`HELD`) until you press **X** in the headset. See [Quest controls](#quest-controls-and-safety).
3. **Ctrl-C** stops: the arms ramp back to where they started, then the motors are disabled.

Notes:

- `--fps=30` matches recording. The default of 60 is more than the follower's CAN reads keep up with.
- To see the cameras while teleoperating, add the `--robot.cameras=...` from Step 4 plus `--display_data=true`.
- Every 5 s the terminal prints an `[openarm_umeow]` status line; its `bound X%` should stay near 0%.
- Keep a hand near the power / e-stop for the first run.

### Step 3. Record a dataset

```bash
cd ~/Stanley_ws/lerobot_openarm && source .venv/bin/activate
unset PYTHONPATH; export LD_LIBRARY_PATH=/usr/local/cuda/lib64

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

Useful flags:

- Continuing a dataset is automatic: run the same command again (same `--dataset.repo_id`, with
  `--dataset.no_stamp=true`). If `~/.cache/huggingface/lerobot/<repo_id>` already has episodes, the plugin adds
  `--resume=true --dataset.root=...` itself and prints how many there are. `--dataset.num_episodes` then counts the
  episodes to ADD, not the total. `--resume=false` turns this off.
- `--dataset.push_to_hub=false`: keep the dataset local only. It is saved under
  `~/.cache/huggingface/lerobot/<repo_id>` either way.

### Quest controls and safety

| Quest | state | what happens |
|---|---|---|
| **1st X** | `HELD` | **Start driving** (`LIVE`). Your current hand poses *and* the headset's pose are captured at this press; the arms follow your hands *relative to that moment* only. |
| **2nd X** | `LIVE` | **Return home slowly** (`RETURNING`, every joint <= 0.3 rad/s), still recording, then **save** the episode on arrival. The grippers open after the save. |
| **X** | `PAUSED` | **Return home slowly while still recording** (grippers kept, so a held object stays held); the arms then wait at home with the episode open. |
| **X** | `HELD` (episode open) | **Continue** the same episode from home (hands re-anchored, no jump). |
| **Y** | `LIVE` / `PAUSED` / `HELD` (episode open) | **Discard** the episode at once and **return home slowly**; grippers open on arrival. |
| **Triggers** | `LIVE` | close the grippers. |
| **A** / **B** | any | no function. |

A controller position that jumps faster than 4 m/s or 900 deg/s is never followed. One such jump is a
tracking snap (the Quest re-finding a controller it lost sight of, e.g. held low or behind the can, since the
headset hangs at the neck): it is ignored, the arm holds still and follows on from there (terminal: "a tracking
snap -- IGNORED"). Keep the controllers in front of the headset to avoid them.

Every return home (2nd X, Y, the time limit, recovery) is verified on the MEASURED joints: it ends only
when each arm joint is within 0.1 rad of home (3 s to settle). If one is not, the terminal and rerun say
`NOT at the reset pose: RJ7.pos is +0.50 rad from home`, and X is refused until it is -- no episode starts
from a wrong pose. A motor that trips its protection (OVERCURRENT from twisting a wrist against its stop or
cable, OVERTEMP, ...) switches itself off and goes limp: `MOTOR FAULT` names it, the arm holds, and the
hold stays until the motor answers again (Ctrl+C and restart; saved episodes are safe). The 5 s stats line
also shows the observation read time, which grows when a motor stops answering.

`PAUSED` stops the arms where they are when something is too fast: repeated jumps (3 within 2 s), a joint
more than 1 rad behind its target (joint speed cap: 2 rad/s), or the IK jumping (e.g. an arm stretched out).
The terminal and the rerun panel say which joint or limit and why. If `episode_time_s` runs out meanwhile,
X recovers home and the episode is saved on arrival.

**Rerun window** (`--display_data=true`): top-left the **status** panel (episode, phase, Quest buttons and
tracking); top-right the **network** panel and a latency plot -- ping to the Quest (median of 3 per second,
plus Wi-Fi wake-up spikes), how late the Quest's packets arrive vs the best this session (the lag your
teleop actually feels), and this PC's Wi-Fi (SSID, band, signal, link speed). A line climbing in the plot
means latency is building up; ✅ < 50 ms, ⚠️ < 150 ms, ❌ above.

Check the recorded data:

```bash
lerobot-dataset-viz --repo-id ethanCSL/openarm_plate_wiping_quest_v00 --episode-index 0
```

### Step 5. Train (official command)

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=ethanCSL/openarm_pringles_lerobot_real_v00 \
  --rename_map='{"observation.images.right_wrist_cam": "observation.images.camera1", "observation.images.wrist_cam": "observation.images.camera2", "observation.images.body_cam": "observation.images.camera3"}' \
  --policy.repo_id=ethanCSL/smolvla_pringles_lerobot_real_v00 \
  --batch_size=64 --steps=20000 --policy.device=cuda \
  --output_dir=outputs/train/smolvla_pringles_lerobot_real_v00 \
  --job_name=smolvla_pringles_lerobot_real_v00 \
  --wandb.enable=true
```

- `--rename_map` is required: `smolvla_base` names its three cameras `camera1/2/3`, our datasets name them
  `right_wrist_cam / wrist_cam / body_cam`. This is the same mapping your earlier SmolVLA checkpoints used.
- `--policy.repo_id` is where the model is uploaded; to keep it local, use `--policy.push_to_hub=false` instead.
- Train on datasets recorded with **this** pipeline. Checkpoints trained on the Isaac-mirror datasets are not
  interchangeable: those store sim-unit joints (grippers 0-0.044) in a different key order.

**GR00T N1.7**, the same recipe as the earlier `openarm_pringles_gr00t_real_v00` checkpoint (does **not**
fit the 16 GB GPU here, even at batch 1 -- train on a larger GPU):

```bash
lerobot-train --policy.type=groot --policy.base_model_path=nvidia/GR00T-N1.7-3B \
  --policy.chunk_size=16 --policy.n_action_steps=16 \
  --policy.use_relative_actions=true --policy.relative_exclude_joints='["LJ8", "RJ8"]' \
  --dataset.repo_id=ethanCSL/openarm_pringles_lerobot_real_v00 \
  --policy.repo_id=ethanCSL/groot_pringles_lerobot_real_v00 \
  --batch_size=64 --steps=20000 --policy.device=cuda \
  --output_dir=outputs/train/groot_pringles_lerobot_real_v00 --job_name=groot_pringles_lerobot_real_v00
```

### Step 6. Evaluate on the real robot: normal, async, RTC

Four ways to run a trained policy, for both SmolVLA and GR00T N1.7: normal, async and RTC are official lerobot
tools; async + RTC is the combination the RTC docs recommend ("use both together"), which lerobot's policy
server does not implement, so our server wrapper adds it. Prepare the terminal as in Step 1 (the Quest is not
needed).

| mode | what it does | tool |
|---|---|---|
| **normal** | predict a chunk, execute it, predict the next; the robot waits during each inference | `lerobot-rollout` |
| **async** ([docs](https://huggingface.co/docs/lerobot/main/en/async)) | a policy server computes the next chunk while the robot client is still executing the current one, and overlapping chunks are aggregated; no waiting, inference can run on another machine | `policy_server` + `robot_client` |
| **RTC** ([docs](https://huggingface.co/docs/lerobot/main/en/rtc)) | inference in a background thread, and each new chunk is *guided* to continue smoothly from the actions already being executed | `lerobot-rollout --inference.type=rtc` |
| **async + RTC** | async's server/client split, with every chunk RTC-guided to continue the client's unexecuted actions | `policy_server --rtc=true` + `robot_client` |

#### Normal

```bash
# SmolVLA
lerobot-rollout --strategy.type=base \
  --policy.path=outputs/train/smolvla_pringles_lerobot_real_v00/checkpoints/last/pretrained_model \
  --robot.type=openarm_umeow --robot.right_port=can0 --robot.left_port=can1 \
  --robot.cameras="{body_cam: {type: opencv, index_or_path: /dev/rs_body, width: 640, height: 480, fps: 30}, wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_left, width: 640, height: 480, fps: 30}, right_wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_right, width: 640, height: 480, fps: 30}}" \
  --rename_map='{"observation.images.right_wrist_cam": "observation.images.camera1", "observation.images.wrist_cam": "observation.images.camera2", "observation.images.body_cam": "observation.images.camera3"}' \
  --task="Pick up the Pringles can with the right arm, hand it to the left arm" --duration=60 --display_data=true

# GR00T N1.7 (no --rename_map: GR00T keeps the dataset's camera names)
lerobot-rollout --strategy.type=base \
  --policy.path=outputs/train/groot_pringles_lerobot_real_v00/checkpoints/last/pretrained_model \
  --robot.type=openarm_umeow --robot.right_port=can0 --robot.left_port=can1 \
  --robot.cameras="{body_cam: {type: opencv, index_or_path: /dev/rs_body, width: 640, height: 480, fps: 30}, wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_left, width: 640, height: 480, fps: 30}, right_wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_right, width: 640, height: 480, fps: 30}}" \
  --task="Pick up the Pringles can with the right arm, hand it to the left arm" --duration=60 --display_data=true
```

#### Async (policy server + robot client)

Terminal 1, the policy server (same machine: `127.0.0.1`; another GPU machine: its IP, and `--host=0.0.0.0`):

```bash
cd ~/Stanley_ws/lerobot_openarm
source .venv/bin/activate
unset PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
python -m lerobot_robot_openarm_umeow.policy_server --host=127.0.0.1 --port=8080
```

Terminal 2, the robot client:

```bash
# SmolVLA
cd ~/Stanley_ws/lerobot_openarm
source .venv/bin/activate
unset PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
python -m lerobot.async_inference.robot_client \
  --server_address=127.0.0.1:8080 \
  --robot.type=openarm_umeow --robot.right_port=can0 --robot.left_port=can1 \
  --robot.cameras="{body_cam: {type: opencv, index_or_path: /dev/rs_body, width: 640, height: 480, fps: 30}, wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_left, width: 640, height: 480, fps: 30}, right_wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_right, width: 640, height: 480, fps: 30}}" \
  --task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --policy_type=smolvla --pretrained_name_or_path=outputs/train/smolvla_pringles_lerobot_real_v00/checkpoints/last/pretrained_model \
  --policy_device=cuda --actions_per_chunk=50 --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average --fps=30

# GR00T N1.7: --policy_type=groot --pretrained_name_or_path=outputs/train/groot_pringles_lerobot_real_v00/checkpoints/last/pretrained_model --actions_per_chunk=16
```

- Use **`lerobot_robot_openarm_umeow.policy_server`**, not `lerobot.async_inference.policy_server`: it is
  lerobot's server with the fixes our checkpoints need -- it keeps the checkpoint's camera rename map
  (lerobot's client cannot send one, and the server would otherwise drop it), decodes GR00T's relative
  actions a whole chunk at a time (lerobot's server does one step at a time, which GR00T rejects), and
  loads GR00T in bf16 (`--policy_dtype=auto|bf16|fp32`). Same flags otherwise.
- It also fixes lerobot's "too similar observation" filter for this robot: the server skips an observation
  whose joint vector is within 1.0 of the last one it ran, meant as ~1 degree for lerobot's degree-based arms;
  ours report radians (1.0 = ~57 degrees), so lerobot's server only predicted once the client's queue had
  emptied -- async collapsed to synchronous. Now `--obs_similarity_atol=0.0175` (1 degree in radians).
- And it passes the camera images at their own resolution. lerobot's server resizes them to the policy's
  image feature shape, which for a SmolVLA fine-tuned from smolvla_base is the base model's 256x256: our
  640x480 frames were squashed (aspect ratio lost), unlike in training and in `lerobot-rollout`, where the
  policy gets the full frame and letterboxes it itself. On a real checkpoint that changed actions by up to 0.04 rad.
- The client is lerobot's own. `--actions_per_chunk` <= the policy's chunk size (SmolVLA 50, our GR00T 16);
  `--chunk_size_threshold` 0.5-0.6 is the docs' recommendation; add `--debug_visualize_queue_size=true` to
  plot the action queue when tuning.
- The arm goes home first (type `YES`); Ctrl-C stops the client, and the arm returns to its rest pose.

#### RTC (Real-Time Chunking)

```bash
# SmolVLA
lerobot-rollout --strategy.type=base \
  --inference.type=rtc --inference.rtc.mode=guided \
  --inference.rtc.execution_horizon=10 --inference.rtc.max_guidance_weight=10.0 \
  --policy.path=outputs/train/smolvla_pringles_lerobot_real_v00/checkpoints/last/pretrained_model \
  --robot.type=openarm_umeow --robot.right_port=can0 --robot.left_port=can1 \
  --robot.cameras="{body_cam: {type: opencv, index_or_path: /dev/rs_body, width: 640, height: 480, fps: 30}, wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_left, width: 640, height: 480, fps: 30}, right_wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_right, width: 640, height: 480, fps: 30}}" \
  --rename_map='{"observation.images.right_wrist_cam": "observation.images.camera1", "observation.images.wrist_cam": "observation.images.camera2", "observation.images.body_cam": "observation.images.camera3"}' \
  --task="Pick up the Pringles can with the right arm, hand it to the left arm" --duration=60 --display_data=true

# GR00T N1.7: same with --policy.path=outputs/train/groot_pringles_lerobot_real_v00/checkpoints/last/pretrained_model, --inference.rtc.execution_horizon=8, no --rename_map
```

- `execution_horizon`: actions of the previous chunk the new one is guided to match; 8-12, and below the
  chunk size. `max_guidance_weight` 10.0 is the docs' value for 10-step flow matching.

#### Async + RTC

The async setup above, with RTC done by the server: for each new observation, the server takes the actions
of its last chunk that the client has not executed yet (it knows their timesteps), re-anchors them to the new
state for relative-action policies (GR00T), and predicts the next chunk RTC-guided to continue them, with the
inference delay estimated from the measured latency -- the same lerobot helpers `lerobot-rollout`'s RTC uses.

Terminal 1:

```bash
# SmolVLA
python -m lerobot_robot_openarm_umeow.policy_server --host=127.0.0.1 --port=8080 \
  --rtc=true --rtc_execution_horizon=10 --rtc_max_guidance_weight=10.0 --rtc_prefix_attention_schedule=EXP

# GR00T N1.7: the same with --rtc_execution_horizon=8
```

Terminal 2, as for async but with `--aggregate_fn_name=latest_only` (RTC has already blended the overlap;
averaging it again would undo that):

```bash
# SmolVLA
python -m lerobot.async_inference.robot_client \
  --server_address=127.0.0.1:8080 \
  --robot.type=openarm_umeow --robot.right_port=can0 --robot.left_port=can1 \
  --robot.cameras="{body_cam: {type: opencv, index_or_path: /dev/rs_body, width: 640, height: 480, fps: 30}, wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_left, width: 640, height: 480, fps: 30}, right_wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_right, width: 640, height: 480, fps: 30}}" \
  --task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --policy_type=smolvla --pretrained_name_or_path=outputs/train/smolvla_pringles_lerobot_real_v00/checkpoints/last/pretrained_model \
  --policy_device=cuda --actions_per_chunk=50 --chunk_size_threshold=0.5 \
  --aggregate_fn_name=latest_only --fps=30

# GR00T N1.7: --policy_type=groot --pretrained_name_or_path=outputs/train/groot_pringles_lerobot_real_v00/checkpoints/last/pretrained_model --actions_per_chunk=16
```

- The server log shows each chunk: `RTC chunk @ timestep N: guided by K unexecuted actions, inference_delay D`.
  `K` > 0 means RTC is active; `D` is the estimated latency in steps (measured on the RTX 5080: about 2 for
  GR00T, 4-5 for SmolVLA; it rises for a while after a slow inference).
- RTC needs observations mid-chunk: keep `--chunk_size_threshold` around 0.5. Near 0 the client only asks
  when its queue is empty, and there is nothing left to continue.

#### Recording evaluation episodes

Normal and RTC also work with `--strategy.type=episodic` instead of `base`, which records each evaluation
episode (to watch or score later):

```bash
  --strategy.type=episodic \
  --dataset.repo_id=ethanCSL/rollout_smolvla_pringles_v00 --dataset.no_stamp=true \
  --dataset.single_task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --dataset.num_episodes=10 --dataset.episode_time_s=60 --dataset.reset_time_s=60
```

During an episode **Right arrow** ends it (saved), **Left arrow** discards it, **Esc** stops. Then the arm
returns **slowly** (<= 0.3 rad/s) to the start pose and the reset phase gives you `reset_time_s` to reset
the scene; **Right arrow** ends the reset early. Episodic dataset names must start with `rollout_`.

#### Notes for all modes

- The arm goes to the home pose first (type `YES`; `--robot.assume_yes=true` skips it), grippers open. On exit
  it returns to that pose, then to where it was before connecting, then the motors are disabled.
- `--rename_map` is required for SmolVLA in `lerobot-rollout` (it checks the cameras before loading the policy).
- `--task` and `--robot.cameras` must match the recording **exactly** (same text, same camera names).
- GR00T uses relative actions: lerobot's normal mode decodes them one step at a time, which GR00T rejects,
  so the robot plugin decodes each chunk whole instead (same result, still synchronous). GR00T is also loaded
  in bf16 (`--robot.policy_dtype=auto`, the default): in fp32 it ran out of memory on the 16 GB GPU.
- The robot's safety guard applies to every mode: an action that jumps or strays from the measured joints
  triggers a `SAFETY HOLD` (the arm stops) instead of being executed.
- Checkpoints trained on the old Isaac-mirror datasets cannot drive this robot (different joint order and
  units): train on data recorded with Step 4.
