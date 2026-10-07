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
4. Close any rerun viewer left over from an earlier session (`pkill -f 'rerun --port=9876'`), so
   `--display_data=true` opens a fresh window for this one.
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
  --teleop.type=openarm_quest --teleop.episode_buttons=false \
  --fps=30
```

What happens:

1. The terminal shows each joint's current vs. home position. **Type `YES`**, and both arms move slowly
   (0.3 rad/s) to the home pose with the grippers open.
2. The arms hold home (`HELD`) until you press **X** in the headset. See [Quest controls](#quest-controls-and-safety).
3. **Ctrl-C** stops: the arms ramp back to where they started, then the motors are disabled.

Notes:

- `--teleop.episode_buttons=false`: nothing records here, so X/Y must not press arrow keys into your desktop.
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
   🔴 RECORDING · episode 3 of 50 · 12 s          (or 🟡 RESETTING / 💾 SAVING / ⚫ STOPPING ...)
   saved in the dataset: 3 of 50
   Quest: ▶ LIVE -- arms follow the controllers. X = save, Y = discard
   episode 3 will be SAVED when the reset ends (Y now = discard instead)     <- during the reset
   ⛔ ROBOT SAFETY HOLD: ...                                                  <- only if it fires
   last: episode 2 SAVED (3 in the dataset)
   ```

   The "will be SAVED / DISCARDED" line follows the Quest's X / Y; a keyboard arrow press is not shown there.
2. Recording of episode 0 starts. The arms hold the home pose (`HELD`) until you press **X**.
3. Each episode: **X** to start driving, do the task, then **X** again to **save** it or **Y** to **discard**
   it. Either way the arms return home slowly and the grippers open (see
   [Quest controls](#quest-controls-and-safety)).
4. The reset phase (`reset_time_s`, not recorded) follows. The arms finish returning home; reset the scene.
   A saved episode is written at the end of the reset phase; pressing **Y** during the reset phase
   discards it after all. Keep `reset_time_s` longer than the return (a return of up to ~1.5 rad takes ~5 s).
5. Keyboard equivalents: **Right arrow** = save, **Left arrow** = discard, **Esc** = stop the whole session.
   `episode_time_s` is only a cap: episodes normally end on your 2nd X / Y.
6. When done (or on Esc), the arms ramp back to where they started, then the motors are disabled.

Useful flags:

- `--resume=true`: add episodes to an existing dataset (same `--dataset.repo_id`).
- `--dataset.push_to_hub=false`: keep the dataset local only. It is saved under
  `~/.cache/huggingface/lerobot/<repo_id>` either way.
- Every 5 s the robot prints `[openarm_umeow] 30 Hz commands | step limit 1 rad/s bound 0% of them ...`,
  which should stay near **0%**. Higher means the recorded actions are running ahead of the arm.

### Quest controls and safety

| Quest | state | what happens |
|---|---|---|
| **X** | `HELD` | **Start driving** (`LIVE`). Your current hand poses *and* the headset's pose are captured at this press; the arms follow your hands *relative to that moment* only. |
| **X** | `LIVE` | **Save** the episode (recording) and **return home slowly** (`RETURNING`, every joint <= 0.3 rad/s). On arrival the grippers **open** and the arms wait (`HELD`). |
| **Y** | `LIVE` / `PAUSED` | **Discard** the episode (recording) and **return home slowly**, grippers open on arrival. |
| **X** | `RETURNING` | ignored: wait for `HELD`. |
| **X** | `PAUSED` | **Resume** driving from where the arms stopped (re-anchors, no jump). |
| **Triggers** | `LIVE` | close the grippers. |
| **A** / **B** | any | save / discard the episode without moving the arms. |

The arms never jump to the controllers: on every X the hand poses are re-captured, and nothing from an
earlier press or episode is reused. The headset can hang and swing at your neck: its pose is only read on X.
(Before this, a 5 degree swing of the headset moved the arm targets 3.4 cm with the hands still.)

**`PAUSED`**: the teleop stops the arms where they are and prints the reason when something looks wrong:

- `controller pose jumped N cm / M deg between two packets (tracking glitch)`: no hand moves that fast.
- `IK solution jumped ...` / `IK target ran ... ahead ...`: the arm would have to jump or race.
- `the robot refused a command (...)`: the robot's own guard below fired.

Then **X** resumes from there, **Y** returns home (and discards the episode). A controller that loses
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
| A/B buttons do nothing / `episode_buttons disabled (No module named 'pynput')` | `uv sync` (installs `pynput`). A/B simulate arrow-key presses, so they need the X11 desktop session; the keyboard arrows (or `n` / `r` / `q` in the terminal) always work |
| `--display_data=true` but no rerun window, or it shows old data | an old rerun viewer (e.g. from `mirror_bridge.py`'s collection viewer) still holds port 9876 and receives the data instead. Close it, or `pkill -f 'rerun --port=9876'`, then start again |
| `torchcodec ... cannot be loaded` warnings | harmless: lerobot falls back to pyav for video |

### Tests without hardware

```bash
python plugins/tests/test_quest_safety.py               # fake Quest: anchoring, headset swing, glitches, slow return, grippers
python plugins/tests/test_robot_guard.py                # robot safety hold: jumps, tracking error, speed clamp
python plugins/tests/test_rerun_status.py /tmp/mock_rr  # rerun status panel through a 2-episode mocked lerobot-record
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
