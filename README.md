# Run in Real-world

Two ways to collect data and evaluate on the real OpenArm:

- **[Official LeRobot pipeline with the Meta Quest](#official-lerobot-pipeline-meta-quest)** (this branch,
  `official-lerobot-quest`): `lerobot-teleoperate` / `lerobot-record` / `lerobot-train` / `lerobot-rollout`, exactly as on Koch /
  SO-100. No Isaac Sim, ROS 2 or dora in the loop.
- **[Legacy pipeline](#legacy-pipeline-isaac-sim-mirror--custom-deploy-scripts)**: Isaac Sim mirror
  collection (`record_demos_openarm.py --real_arm`) and the custom `deploy_smolvla_*.py` scripts.

---

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

The `unset` / `export` lines are required in every new terminal: the ROS 2 Humble environment loads
ROS's pinocchio instead of the venv's, and the robot plugin then fails with
`Could not import third-party plugin: lerobot_robot_openarm_umeow`.

3. **Stop the dora dataflow** if it is running (`dora run dataflow-vr-mujoco-ros2.yaml`): only one
   program can receive the Quest's packets on UDP port 5006.
4. (Optional) close rerun viewers left from earlier sessions: `pkill -f 'rerun --port='`. If one is still
   open, a new window opens anyway (on another port) and the terminal says so.
5. Start the Quest app as usual. It keeps sending to this PC's port 5006; nothing changes on the headset.

### Step 2. Dry run: check the VR mapping (nothing moves)

```bash
python -m lerobot_teleoperator_openarm_quest.preview --viewer
```

A MuJoCo window shows the pose the IK would send. The terminal prints:

```
[LIVE] quest  72.0 Hz (last 5 ms) | IK   460 Hz (0 failed) | biggest arm step between frames 0.012 rad
  LJ1..8 (motor rad): ...
  RJ1..8 (motor rad): ...
```

- `quest ~70 Hz`: the headset packets arrive. `none yet`: check the Quest app's target IP/port and that dora is stopped.
- Press **X**: `HELD` -> `LIVE`, then move your hands and watch the arms follow in the window.
- If the pose looks wrong here, the problem is the VR mapping or the IK, not the robot.

Ctrl-C to quit.

### Step 3. Teleoperate the real arm (no recording)

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

### Step 4. Record a dataset

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

What happens:

1. A rerun window opens (`--display_data=true`), then the robot prints each joint's current vs. target
   position and asks you to **type `YES`**. It moves both arms slowly (0.3 rad/s) to the home pose with the
   grippers open. Add `--robot.assume_yes=true` to skip the prompt. The cameras and joint plots appear in
   rerun once recording starts.

   The **status** panel at the top of the rerun window shows, live:

   ```
   ⏳ WAITING for X (not recording yet) · episode 3 of 50 · 4 s   (🔴 RECORDING / 🟡 RESETTING / 💾 SAVING ...)
   saved in the dataset: 3 of 50
   Meta Quest: ✅ connected (192.168.x.x, last packet 12 ms ago)          <- or ❌ NO PACKETS / STOPPED
   headset ✅ OK · right controller ✅ OK (+0.25, +1.10, +0.30) · left controller ❌ LOST (...)
   buttons pressed: X · triggers R 0.60 L 0.00 · grips R 1.00 L 0.00 · sticks R (+0.50, -0.20) L (...)
   Quest: ⏸ HELD at home, grippers open -- press X to start
   teleop says: X ignored: the left controller is not tracked -- wake it / bring it into view.
   episode 3 will be SAVED when the reset ends (Y now = discard instead)     <- during the reset
   ⛔ ROBOT SAFETY HOLD: ...                                                  <- only if it fires
   last: episode 2 SAVED (3 in the dataset)
   ```

   If the arms do not respond, look here first: no packets, a ❌ controller, or the "teleop says" line tells
   you why.
2. Each episode **waits for your X** (`⏳ WAITING`): nothing is recorded until you press X, so episodes
   never start with you getting ready. X is ignored while the robot is still moving to home at startup.
3. **X** starts recording and driving; do the task; then:
   - **X** again: the arms return home slowly (<= 0.3 rad/s) **while still recording**, and the episode is
     **saved when they arrive**, so it always ends at the home pose. The grippers stay as they were until
     the episode has ended, then open.
   - **Y**: the episode is **discarded** at once, and the arms return home (not recorded).
   - `episode_time_s` running out acts like the 2nd X: return home, then save. It is a cap, not a cut.
4. The reset phase (`reset_time_s`, not recorded) follows; reset the scene. X is ignored until the next
   episode shows `⏳ WAITING for X`.
5. Keyboard: **Right arrow** = end and save at once (no recorded return), **Left arrow** = discard,
   **Esc** = stop the whole session.
6. When done (or on Esc), the arms ramp back to where they started, then the motors are disabled.

Useful flags:

- `--dataset.no_stamp=true`: keep the dataset name exactly as typed. Without it lerobot appends the start
  time (`..._v00_20261007_191607`), a new dataset per run, and Step 5's `--dataset.repo_id` must use that name.
- `--resume=true`: add episodes to an existing dataset (same `--dataset.repo_id`, so use `no_stamp`).
- `--dataset.push_to_hub=false`: keep the dataset local only. It is saved under
  `~/.cache/huggingface/lerobot/<repo_id>` either way.
- Every 5 s the robot prints `[openarm_umeow] 30 Hz commands | step limit 1 rad/s bound 0% of them ...`,
  which should stay near **0%**. Higher means the recorded actions are running ahead of the arm.

### Quest controls and safety

| Quest | state | what happens |
|---|---|---|
| **X** | `HELD` | **Start driving** (`LIVE`). Your current hand poses *and* the headset's pose are captured at this press; the arms follow your hands *relative to that moment* only. |
| **X** | `LIVE` | **Return home slowly** (`RETURNING`, every joint <= 0.3 rad/s), still recording, then **save** the episode on arrival. The grippers open after the save. |
| **Y** | `LIVE` / `PAUSED` | **Discard** the episode at once and **return home slowly**; grippers open on arrival. |
| **X** | `RETURNING` / `PAUSED` | ignored. From `PAUSED`, only Y. |
| **X** | `HELD`, during the reset | ignored: wait for `WAITING for X`. |
| **Triggers** | `LIVE` | close the grippers. |
| **A** / **B** | any | no function. |

The arms never jump to the controllers: on every X the hand poses are re-captured, and nothing from an
earlier press or episode is reused. The headset can hang and swing at your neck: its pose is only read on X.
(Before this, a 5 degree swing of the headset moved the arm targets 3.4 cm with the hands still.)

**`PAUSED`**: the teleop stops the arms where they are and prints the reason when something looks wrong:

- `controller pose jumped N cm / M deg between two packets (tracking glitch)`: no hand moves that fast.
- `IK solution jumped ...` / `IK target ran ... ahead ...`: the arm would have to jump or race.
- `the robot refused a command (...)`: the robot's own guard below fired.

Then **Y** discards the episode and returns home. A controller that loses
tracking (asleep, out of view) makes its arm hold; when it is seen again it is re-anchored where it
reappears. X is refused while a controller is untracked.

**`SAFETY HOLD`** (robot side, also for `lerobot-rollout`): the robot refuses any command whose arm joints
jump more than `--robot.max_command_jump` (0.25 rad) in one tick or sit more than
`--robot.max_tracking_error` (0.5 rad) from the measured joints, and holds the arm until commands come back
near the held pose. It never executes more than `--robot.max_joint_speed` (1 rad/s) per joint.

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
python -m lerobot_robot_openarm_umeow.policy_server --host=127.0.0.1 --port=8080
```

Terminal 2, the robot client:

```bash
# SmolVLA
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

### Troubleshooting

| symptom | fix |
|---|---|
| `Could not import third-party plugin: lerobot_robot_openarm_umeow` / pinocchio `undefined symbol` | `unset PYTHONPATH; export LD_LIBRARY_PATH=/usr/local/cuda/lib64` (Step 1) |
| `invalid choice: 'openarm_umeow'` | run `uv sync` on this branch |
| preview shows `quest 0 Hz (last none yet)` | dora still running, or the Quest app sends to another IP/port |
| `REFUSED: ... would have to travel ... rad` at start | the arm is too far from home: move it closer by hand, or check `calibration.json` |
| `The two arms' CAN cables are SWAPPED` | swap the CAN cables, or swap `--robot.right_port` / `--robot.left_port` |
| `No module named 'pynput'` | `uv sync` (lerobot-record's keyboard controls need it). The Quest's X / Y do not: they set lerobot-record's episode flags directly |
| `--display_data=true` but no rerun window, or it shows old data | an old rerun viewer (e.g. from `mirror_bridge.py`'s collection viewer) still holds port 9876 and receives the data instead. Close it, or `pkill -f 'rerun --port=9876'`, then start again |
| `Could not load libtorchcodec` traceback at start | `uv sync` on this branch: torchcodec is pinned to 0.11 to match torch 2.11 (0.10 could not load) |
| `another rerun viewer already holds port 9876` | an old viewer is still open; a new window was opened anyway. Close the old one: `pkill -f 'rerun --port=9876'` |
| the arms do not respond to the Quest | read the rerun status panel: `NO PACKETS` (app not sending to this PC / dora still running), a ❌ controller or headset, or the "teleop says" line (e.g. why X was ignored) |
| an episode is missing after Ctrl-C | an episode is only saved after its reset phase; end it with X (save) first, then Esc or Ctrl-C |

### Tests without hardware

```bash
python plugins/tests/test_quest_safety.py               # fake Quest: anchoring, headset swing, glitches, slow return, grippers
python plugins/tests/test_robot_guard.py                # robot safety hold: jumps, tracking error, speed clamp
python plugins/tests/test_rerun_status.py /tmp/mock_rr  # rerun status panel through a 2-episode mocked lerobot-record
python plugins/tests/test_record_gate.py /tmp/mock_gate # episodes start on X; X ends the reset; timer end returns home
bash plugins/tests/eval_chain/run_chain.sh /tmp/chain   # record -> lerobot-train SmolVLA (30 steps) -> evaluate: normal, episodic, RTC, async, async+RTC (GPU, ~5 min)
# GR00T N1.7 on the mocked robot with any GR00T checkpoint (an old one only proves it loads and runs):
python plugins/tests/eval_chain/chain_rollout.py base <groot checkpoint> - plugins/tests/eval_chain                    # normal
python plugins/tests/eval_chain/chain_rollout.py base <groot checkpoint> - plugins/tests/eval_chain --inference.type=rtc --inference.rtc.execution_horizon=8   # RTC
python plugins/tests/eval_chain/chain_async.py groot <groot checkpoint> 16 plugins/tests/eval_chain                     # async
CHAIN_SERVER_ARGS="--rtc=true --rtc_execution_horizon=8" CHAIN_CLIENT_ARGS="--aggregate_fn_name=latest_only" \
  python plugins/tests/eval_chain/chain_async.py groot <groot checkpoint> 16 plugins/tests/eval_chain                   # async + RTC
python plugins/tests/test_record_mock.py /tmp/mock_ds   # official lerobot-record end to end, CAN mocked out
python plugins/tests/test_teleoperate_mock.py           # official lerobot-teleoperate end to end, CAN mocked out
```

More detail on the plugins: [plugins/README.md](plugins/README.md).

---

## Legacy pipeline (Isaac Sim mirror + custom deploy scripts)

### Activate CAN-FD

```
cd ~/Stanley_ws/openarm_can/setup
```

```
sudo ./my_arm 
```

### Model evaluation on real robot

```
cd ~/Stanley_ws/lerobot_openarm
uv sync
source .venv/bin/activate
env -u PYTHONPATH LD_LIBRARY_PATH=/usr/local/cuda/lib64 python deploy_smolvla_pickup_jointspace.py     --checkpoint ethanCSL/openarm_visuomotor_VR_pringles_V14_background_30hz     --body-cam-index rs_body --wrist-cam-index rs_wrist_left --right-wrist-cam-index rs_wrist_right     --calibration calibration.json     --inference-hz 30 --max-joint-speed 1.5 --max-episode-seconds 600 
```

Deploy in async evaluation

```
cd ~/Stanley_ws/lerobot_openarm
uv sync
source .venv/bin/activate
env -u PYTHONPATH LD_LIBRARY_PATH=/usr/local/cuda/lib64 python deploy_smolvla_async.py     --checkpoint ethanCSL/openarm_visuomotor_VR_pringles_V14_background_30hz    --body-cam-index rs_body --wrist-cam-index rs_wrist_left --right-wrist-cam-index rs_wrist_right     --calibration calibration.json     --control-hz 30 --max-joint-speed 1.5     --actions-per-chunk 50 --chunk-size-threshold 0.8     --max-episode-seconds 25 --max-episodes 20
```
