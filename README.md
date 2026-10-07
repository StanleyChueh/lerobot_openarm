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
| `--teleop.type=openarm_quest` | The Quest controllers -> the same pose mapping, smoothing, X/Y anchoring and IK as the dora pipeline -> joint targets for the robot. |

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
4. Start the Quest app as usual. It keeps sending to this PC's port 5006; nothing changes on the headset.

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
  --robot.max_joint_speed=1.0 \
  --teleop.type=openarm_quest \
  --fps=30
```

What happens:

1. The terminal shows each joint's current vs. home position. **Type `YES`**, and both arms move slowly
   (0.3 rad/s) to the home pose.
2. The arms hold home until you press **X** in the headset. X anchors your current hand pose onto the home
   pose, so there is no jump, and from then on the arms follow your hands.
3. **Triggers** close the grippers. **X** again or **Y** sends the arms home and holds them; press **X** to
   drive again.
4. **Ctrl-C** stops: the arms ramp back to where they started, then the motors are disabled.

Notes:

- `--robot.max_joint_speed=1.0` is a cautious jump guard for the first session. Once tracking looks right,
  raise it to the default 2.0 (or drop the flag).
- `--fps=30` matches recording. The default of 60 is more than the follower's CAN reads keep up with.
- To see the cameras while teleoperating, add the `--robot.cameras=...` from Step 4 plus `--display_data=true`.
- Every 5 s the terminal prints an `[openarm_umeow]` status line; its `bound X%` should stay near 0%.
- Keep a hand near the power / e-stop for the first run.

### Step 4. Record a dataset

```bash
lerobot-record \
  --robot.type=openarm_umeow --robot.right_port=can0 --robot.left_port=can1 \
  --robot.cameras="{body_cam: {type: opencv, index_or_path: /dev/rs_body, width: 640, height: 480, fps: 30}, wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_left, width: 640, height: 480, fps: 30}, right_wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_right, width: 640, height: 480, fps: 30}}" \
  --teleop.type=openarm_quest \
  --dataset.repo_id=ethanCSL/openarm_plate_wiping_quest_v00 \
  --dataset.single_task="Pick up the plate and then wipe it" \
  --dataset.num_episodes=50 --dataset.fps=30 \
  --dataset.episode_time_s=60 --dataset.reset_time_s=20 \
  --dataset.streaming_encoding=true --dataset.encoder_threads=2 \
  --display_data=true
```

What happens:

1. The robot prints each joint's current vs. target position and asks you to **type `YES`**, then moves both
   arms slowly (0.3 rad/s) to the home pose. Add `--robot.assume_yes=true` to skip the prompt.
2. Recording of episode 0 starts. The arms hold the home pose until you press **X**.
3. Each episode, in the headset:

   | Quest button | action |
   |---|---|
   | **X** (1st press) | anchor your current hand pose onto the home pose and start driving the arms |
   | **X** (2nd press) or **Y** | send the arms back to home and hold there |
   | **Triggers** | close the grippers |
   | **A** | end this episode and **save** it (same as the Right arrow key) |
   | **B** | **discard and re-record** this episode (same as the Left arrow key) |

   On the keyboard: **Right arrow** = save, **Left arrow** = re-record, **Esc** = stop the whole session.
4. After each episode comes a `reset_time_s` reset phase (not recorded). The teleop stays live, so press
   **Y** to send the arms home and reset the scene, then **X** when the next episode starts.
5. When done (or on Esc), the arms ramp back to where they started, then the motors are disabled.

Useful flags:

- `--resume=true`: add episodes to an existing dataset (same `--dataset.repo_id`).
- `--dataset.push_to_hub=false`: keep the dataset local only. It is saved under
  `~/.cache/huggingface/lerobot/<repo_id>` either way.
- `--robot.max_joint_speed=2.0` (rad/s): the jump guard. Every 5 s the robot prints
  `[openarm_umeow] 30 Hz commands | step limit 2 rad/s bound 0% of them ...`, which should stay near **0%**.
  Higher means the recorded actions are running ahead of the arm.

Check the recorded data:

```bash
lerobot-dataset-viz --repo-id ethanCSL/openarm_plate_wiping_quest_v00 --episode-index 0
```

### Step 5. Train (official command, unchanged)

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id=ethanCSL/smolvla_plate_wiping_quest_v00 \
  --dataset.repo_id=ethanCSL/openarm_plate_wiping_quest_v00 \
  --batch_size=64 --steps=20000 --policy.device=cuda \
  --output_dir=outputs/train/smolvla_plate_wiping_quest_v00 \
  --job_name=smolvla_plate_wiping_quest_v00 \
  --wandb.enable=true
```

`--policy.repo_id` is where the model is uploaded. To keep it local instead, replace that line with
`--policy.push_to_hub=false`.

### Step 6. Evaluate on the real robot

Prepare the terminal as in Step 1 (the Quest is not needed). Then:

```bash
lerobot-rollout \
  --strategy.type=base \
  --policy.path=outputs/train/smolvla_plate_wiping_quest_v00/checkpoints/last/pretrained_model \
  --robot.type=openarm_umeow --robot.right_port=can0 --robot.left_port=can1 \
  --robot.cameras="{body_cam: {type: opencv, index_or_path: /dev/rs_body, width: 640, height: 480, fps: 30}, wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_left, width: 640, height: 480, fps: 30}, right_wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_right, width: 640, height: 480, fps: 30}}" \
  --task="Pick up the plate and then wipe it" \
  --duration=60
```

- `--policy.path` also takes a Hub id, e.g. `ethanCSL/smolvla_plate_wiping_quest_v00`.
- `--task` and `--robot.cameras` must match the recording **exactly** (same text, same camera names).
- The arm goes to the home pose first (type `YES`), then the policy runs for `--duration` seconds.
- Slow inference: add `--inference.type=rtc --inference.rtc.execution_horizon=10`.
- To record the evaluation runs as a dataset: `--strategy.type=episodic --dataset.repo_id=ethanCSL/eval_...`
  (see `lerobot-rollout --help`).

### Troubleshooting

| symptom | fix |
|---|---|
| `Could not import third-party plugin: lerobot_robot_openarm_umeow` / pinocchio `undefined symbol` | `unset PYTHONPATH; export LD_LIBRARY_PATH=/usr/local/cuda/lib64` (Step 1) |
| `invalid choice: 'openarm_umeow'` | run `uv sync` on this branch |
| preview shows `quest 0 Hz (last none yet)` | dora still running, or the Quest app sends to another IP/port |
| `REFUSED: ... would have to travel ... rad` at start | the arm is too far from home: move it closer by hand, or check `calibration.json` |
| `The two arms' CAN cables are SWAPPED` | swap the CAN cables, or swap `--robot.right_port` / `--robot.left_port` |
| A/B buttons do nothing | use the keyboard arrows; A/B simulate key presses and need a desktop (X11) session |
| `torchcodec ... cannot be loaded` warnings | harmless: lerobot falls back to pyav for video |

### Tests without hardware

```bash
python plugins/tests/test_quest_teleop.py               # fake Quest -> teleop: hold, anchor, tracking, triggers
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
