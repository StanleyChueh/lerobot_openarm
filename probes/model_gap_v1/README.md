# Model-gap probe v1 (OpenArm, right arm)

These 9 short episodes compare the 3D robot model with the real robot.
Each one starts at the rest pose and follows a real demo's own right-arm path at half speed. It then **holds still for 20 s**, and comes back the same way. The left arm stays at rest the whole time.

At each hold, measure two numbers. The model's predictions for the same joint angles are in `predictions.json`. Please measure before you look at them, so the readings stay unbiased.

## Before you start
- Clear the table: no can, no lid, nothing in the workspace.
- Keep a hand on the e-stop. The driver's usual protections are on: the start-pose ramp, the step limit, the safety guard, and the return to rest on exit.
- Each path is a recorded demo path that did not touch the body on the real robot. Episodes 6–8 also turn one joint 5° further *away* from the body.

## Run (from the repo root, one episode at a time)
```bash
python probes/model_gap_v1/replay_log.py \
  --robot.type=openarm_umeow --robot.right_port=can0 --robot.left_port=can1 \
  --dataset.repo_id=local/openarm_model_gap_probe_v1 --dataset.root=probes/model_gap_v1/dataset \
  --dataset.episode=0
```
This takes the same arguments as `lerobot-replay` and drives the robot the same way. It also saves the measured joint angles to `outputs/model_gap_probe/ep<k>_<time>.csv`; please send me those files.

If `replay_log.py` fails to import, plain `lerobot-replay` with the same arguments also works. It only skips the log.

## What to measure during each hold
- **gap**: the shortest gap, in mm, from any part of the right gripper (palm/rail or fingers) to the body column (the 58 × 58 mm post). Write 0 if they touch. Also note which part is closest.
- **height**: the gap, in mm, from the lowest point of the right gripper (usually a fingertip) down to the table top.

| ep | pose | hold (s from start) | gap (mm) | closest part | height (mm) |
|---|---|---|---|---|---|
| 0 | demo 8, frame 64 | 6–26 | | | |
| 1 | demo 8, frame 70 | 7–27 | | | |
| 2 | demo 5, frame 196 | 15–35 | | | |
| 3 | demo 7, frame 202 | 16–36 | | | |
| 4 | demo 2, frame 70 | 7–27 | | | |
| 5 | demo 9, frame 192 | 15–35 | | | |
| 6 | ep 0 pose, right joint 1 turned +5° | 10–30 | | | |
| 7 | ep 0 pose, right joint 2 turned +5° | 10–30 | | | |
| 8 | ep 0 pose, right joint 3 turned +5° | 10–30 | | | |

A photo of each hold, taken from the side and from above, helps too.

## How the results are used
At every hold, the model's numbers are recomputed at the measured joint angles from the CSV. The difference between real and model is then fitted:
- the same offset at every pose points to the arm's mount position;
- an offset that changes when one joint turns points to that joint's zero;
- an offset that grows as the arm stretches out points to sag under gravity.

Built by `research/sim_free_augmentation/simbench/tools/r09_probe_build.py` (branch `r08-general-clearance`) from `ethanCSL/openarm_pringles_lerobot_real_10_v00` at revision cebdee98. The model is the unmodified OpenArm URDF.
