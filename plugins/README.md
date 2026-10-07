# Official lerobot CLIs on this OpenArm, teleoperated with the Meta Quest

Two lerobot plugins, so data collection, training and evaluation all run through the official
`lerobot-record`, `lerobot-train` and `lerobot-rollout`, the same way they run on Koch / SO-100:

| plugin | type | what it is |
|---|---|---|
| `lerobot_robot_openarm_umeow` | `--robot.type=openarm_umeow` | `robots/umeow_openarm_follower` (gravity feed-forward, CAN-read fixes) with the official one-argument `send_action`, a per-tick step limit, a safe start/stop and the gripper squeeze. Uses `calibration.json`; never re-zeroes the motors. |
| `lerobot_teleoperator_openarm_quest` | `--teleop.type=openarm_quest` | The Quest app's UDP packets -> the same pose mapping, smoothing, X/Y anchoring and mink IK as the dora pipeline -> joint targets in motor radians. |

```
Quest app (UDP :5006) -> openarm_quest (IK @ ~500 Hz) -> lerobot-record (30 Hz) -> openarm_umeow -> CAN
```

No Isaac Sim, ROS 2, dora or MuJoCo rendering in the loop. `action` is the joint target sent to the arm
and `observation.state` the measured joints, in the same units and key order.

Both are installed (editable) by `uv sync`; they are declared in the top-level `pyproject.toml`.

## Environment

Run every command from `~/Stanley_ws/lerobot_openarm` with ROS 2 kept out of the process. Its
`PYTHONPATH` / `LD_LIBRARY_PATH` load ROS Humble's pinocchio/eigenpy instead of the venv's, and the
robot plugin fails to import (lerobot then reports `Could not import third-party plugin`):

```bash
cd ~/Stanley_ws/lerobot_openarm && uv sync && source .venv/bin/activate
alias lr='env -u PYTHONPATH LD_LIBRARY_PATH=/usr/local/cuda/lib64'
```

Bring the CAN-FD links up first (`openarm_can/setup: sudo ./my_arm`), and stop the dora dataflow:
only one process can listen on the Quest's UDP port.

## 0. Dry run (nothing moves)

```bash
lr python -m lerobot_teleoperator_openarm_quest.preview --viewer
```

Prints HELD/LIVE, the Quest packet rate, the IK rate and the motor targets, and draws the IK solution
in MuJoCo. Press X and move: if the pose looks wrong here, it is the VR mapping / IK, not the robot.

## 1. Record

```bash
lr lerobot-record \
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

- On connect the arm ramps at 0.3 rad/s to the IK model's `home` keyframe (type `YES` first; add
  `--robot.assume_yes=true` to skip). On exit it ramps back to where it started before de-energising.
- Each episode: **X** anchors your hands onto the home pose and starts driving. **X** again or **Y**
  resets to home and holds. Triggers drive the grippers.
- **A** = end the episode and save it (Right arrow), **B** = re-record it (Left arrow), **Esc** = stop.
  The arrows also work on the keyboard. Without a display, type `n` / `r` / `q` + Enter in the terminal.
- The reset phase between episodes keeps the teleop live: press Y to send the arm home.
- `--resume=true` continues a dataset; `--dataset.push_to_hub=false` keeps it local.
- Every 5 s the robot prints how often the step limit (`--robot.max_joint_speed`, 2.0 rad/s) cut a
  command. It should stay near 0%. If it does not, recorded actions are running ahead of the arm.

## 2. Train (unchanged)

```bash
lr lerobot-train --policy.path=lerobot/smolvla_base \
  --policy.repo_id=ethanCSL/smolvla_plate_wiping_quest_v00 \
  --dataset.repo_id=ethanCSL/openarm_plate_wiping_quest_v00 \
  --batch_size=64 --steps=20000 --policy.device=cuda \
  --output_dir=outputs/train/smolvla_plate_quest --job_name=smolvla_plate_quest
```

## 3. Evaluate

```bash
lr lerobot-rollout --strategy.type=base \
  --policy.path=outputs/train/smolvla_plate_quest/checkpoints/last/pretrained_model \
  --robot.type=openarm_umeow --robot.right_port=can0 --robot.left_port=can1 \
  --robot.cameras="<same as recording>" \
  --task="Pick up the plate and then wipe it" --duration=60
```

For a slow VLA, add `--inference.type=rtc --inference.rtc.execution_horizon=10`. To record the
evaluation episodes, use `--strategy.type=episodic` with `--dataset.repo_id=...` (see
`lerobot-rollout --help`).

## Keep in step with the dora pipeline

`quest_input.py` copies dora-openarm-vr's frame constants and filter, and `OpenArmQuestConfig.ik_args`
copies the ik node's arguments in `dataflow-vr-mujoco-ros2.yaml`. The IK model is read in place from
the dora checkout (`--teleop.ik_xml` / `--robot.ik_xml` to point elsewhere). If you retune one side,
retune the other.
