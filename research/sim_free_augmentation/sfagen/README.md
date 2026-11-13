# Current branch: v03 (latest), v02b, v02

v03 (even arena coverage with strict/loose distortion caps): [RESULTS.md](results/v03/RESULTS.md),
[PROTOCOL.md](results/v03/PROTOCOL.md). 8 datasets `ethanCSL/openarm_pringles_lerobot_*_v03_{strict,loose}`.

v02: see [the v02 visual report](results/v02/index.html) and [frozen protocol](results/v02/PROTOCOL.md). Generation accepted 104/189 attempts and 18/35 grid targets; rear coverage remains incomplete. Both local exports per condition pass verification/quality checks; visible RGB artifacts and incomplete rear coverage remain.

# sfagen: simulation-free real-to-real demonstration augmentation

## Workspace-wide initial-position mode (2026-11-10)

Use [WORKSPACE_F1W.md](WORKSPACE_F1W.md) and the new `*_workspace.yaml` configurations for broad initial-object variation. [WORKSPACE_F1W_RESULTS.md](WORKSPACE_F1W_RESULTS.md) records the 1–5-source checks and local export tests. Legacy dense configurations and public datasets are preserved. The body camera is rendered; copied wrist views require validity masking and must not be treated as generated multiview RGB.


**What it does.** One pipeline turns a handful of real LeRobot demonstrations into many new, replay-checked
demonstrations, for any task and any robot that has a URDF. It uses no simulator, no generative model and no manual
labels.

**What changes between tasks.** Only a short task description changes between tasks. The robot, camera, renderer,
checks and evaluation stay identical. The two OpenArm datasets
(`openarm_pringles_lerobot_real_to_sim_10_3cam_v00`, `openarm_insertbattery_lerobot_real_to_sim_10_3cam_v00`) are
produced with exactly the same code and settings.

```
real demos ─► calibrate (once per rig) ─► table (once per rig) ─► generate (reconstruct + plan + replay checks)
           ─► render (body camera) ─► build (LeRobot v3.0: synthetic-only and/or real+synthetic) ─► bench (ground truth)
```

## Install / environment

`sfagen` needs the following Python packages:
- pinocchio
- coal (through pinocchio)
- torch (CUDA)
- opencv
- av
- numpy, scipy, pandas
- pyyaml
- huggingface_hub
- lerobot (for `build`)

On this machine, run from this tracked directory (not from the old untracked copy in `~/Stanley_ws/sfa_hf_export/sfagen`):
```bash
cd <repo>/research/sim_free_augmentation/sfagen
PY="env -u PYTHONPATH -u LD_LIBRARY_PATH $HOME/Stanley_ws/lerobot_openarm/.venv/bin/python"
$PY -B -m unittest discover -s tests          # CPU tests, about 10 s
```

### Data directories

Code, configs and tests are in git. Generated data is not: `configs/work/` (about 210 GB) and `logs/` stay in the old
data location and are linked here (both are git-ignored):
```bash
ln -s ~/Stanley_ws/sfa_hf_export/sfagen/configs/work configs/work
ln -s ~/Stanley_ws/sfa_hf_export/sfagen/logs logs
```
The tools import `sfagen` from this directory and the published baselines from `legacy/`. `tools/xbench.py <task>
--imports-only` lists every research file it loads and stops if one is not tracked. `tools/push_unified.py` refuses an
empty or incomplete export before any Hub write and uploads a `provenance.json` (git commit, uncommitted files, file
hashes, `uv.lock` hash, SAM2 revision). `results/hub_provenance.json` maps each published dataset to the commit of its code.

## Commands

```bash
$PY -m sfagen calibrate configs/openarm_battery_insertion.yaml            # body camera, render-and-compare
$PY -m sfagen table     configs/openarm_battery_insertion.yaml            # table height of the rig
$PY -m sfagen generate  configs/openarm_battery_insertion.yaml            # all source episodes (or --episodes 0,1)
$PY -m sfagen render    configs/openarm_battery_insertion.yaml
$PY -m sfagen build     configs/openarm_battery_insertion.yaml \
      --repo-id   ethanCSL/openarm_insertbattery_lerobot_real_to_sim_10_3cam_v00 \
      --mixed-repo-id ethanCSL/openarm_insertbattery_lerobot_real_and_sim_10_3cam_v00
$PY -m sfagen bench     configs/openarm_battery_insertion.yaml --episodes 3,5,8,9 --compare camera
# or everything after calibration/table in one go:
$PY -m sfagen run       configs/openarm_battery_insertion.yaml --repo-id <id> [--mixed-repo-id <id>]
```

**Outputs.** Everything goes to `configs/work/<config name>/`:

| Path | Contents |
|---|---|
| `calibration.json` | fitted camera |
| `table_height.json` | rig table height |
| `cache/` | per-episode reconstructions |
| `candidates/ep*/` | planned trajectories + `checks.json` |
| `body/` | rendered body-camera frames |
| `dataset/`, `dataset_mixed/` | LeRobot datasets |
| `profile.jsonl` | wall time and peak GPU memory of every stage |

Upload a finished dataset with `huggingface_hub` (see `../push_rev.py` for revision-safe updates of an existing repo).

## Use it on your own task (same robot)

Copy `configs/openarm_battery_insertion.yaml` and change only these three blocks:

```yaml
source:                      # your real LeRobot dataset
  repo_id: you/your_task_real
  revision: <commit>         # pin it
  root: null                 # null = download from the Hub
  body_camera: observation.images.body_cam
  wrist_cameras: [observation.images.wrist_cam, observation.images.right_wrist_cam]
  wrist_camera_arms: {observation.images.wrist_cam: L, observation.images.right_wrist_cam: R}
  episodes: all
objects:                     # every object an arm picks up from the table
  mug:   {grasped_by: R, grasp_index: 0, proxy: {type: cylinder, radius: 0.04, length: 0.10, axis: vertical}}
  tray:  {grasped_by: L, grasp_index: 0, proxy: {type: box}}        # unknown size: box from segmentation
variations:
  object_position:      {objects: [mug, tray], shifts_cm: [[3,0],[6,0],[-3,0],[-6,0],[0,3],[0,6],[0,-3],[0,-6]]}
  interaction_location: {shifts_cm: [[3,0],[-3,0],[0,3],[0,-3]], d_interact_m: 0.20, ramp_frames: 45}   # 2+ arms only
```

Then run `calibrate` (only if the camera moved) → `table` → `generate` → `render` → `build`.

## Use it on a different robot

Replace the `robot:` block and the `camera:` initial guess:

```yaml
robot:
  urdf: /path/to/robot.urdf
  package_dirs: [/path/to/meshes]      # where the URDF's mesh files live (STL natively; OBJ/DAE need `trimesh`)
  arms:
    arm:                               # any name; one or more arms
      joints: [joint1, joint2, joint3, joint4, joint5, joint6]   # URDF joint names...
      state_indices: [0, 1, 2, 3, 4, 5]                          # ...and where they are in observation.state/action
      tcp_frame: gripper_tcp                                     # a URDF frame between the fingertips
      gripper: {state_index: 6, open_raw: 1.0, closed_raw: 0.0, finger_joints: [finger_joint], finger_open: 0.04, close_threshold: 0.5}
camera:
  intrinsics: {fx: 600, fy: 600, cx: 320, cy: 240, width: 640, height: 480}
  extrinsics: {position: [x, y, z], quat_wxyz: [w, x, y, z]}    # rough guess: within ~15 cm / 15 deg
```

Then:
1. **Camera.** Run `$PY -m sfagen calibrate cfg.yaml --search`. The coarse random search, ±15° / ±15 cm around your guess, plus Powell refinement, finds the camera from the robot's own silhouette.
2. **Check the fit.** Check `calibration.json` ("adopted", held-out IoU). Then look at the overlay produced by `bench`, or render one episode.
3. **Continue.** Run `table`, then `generate`, `render`, `build`.

**Requirements on the data** (the method's assumptions):
- **Camera:** one static third-person camera (`body_camera`), plus a URDF of the robot. Wrist cameras are kept as real frames and flagged, not re-rendered.
- **Joints:** joint positions in `observation.state`, and joint commands in `action` (same layout).
- **Objects:** rigid objects that rest on a surface, are visible in frame 0, and move only while grasped.
- **Grippers:** parallel grippers whose open/close commands mark grasps and releases.
- **Shifts:** small translations (a few cm) relative to the camera distance. No object rotations, no new viewpoints.

**Not supported:**
- deformable objects;
- moving third-person cameras;
- mobile bases;
- contact-rich edits other than moving a whole multi-arm interaction rigidly.

## What is identical for every task (the method)

| stage | rule (code) |
|---|---|
| camera | render-and-compare IoU fit on near-static frames, fit / held-out episode split (`calibrate.py`) |
| table | median of known-size object silhouette fits if median IoU ≥ 0.5, else lowest arm point − 5 mm (`cli.py estimate_table`) |
| lag | default −1.75 frames, changed only if the silhouette agreement scan improves by > 0.01 (`recon.py`) |
| objects | segmented as what disappears when picked up; cylinders: grasp-constrained render-and-compare; boxes: PCA of the segment (`recon.py`) |
| holders | table → grasping hand → hand-over to a closed hand / released (`recon.py`) |
| F1 object position | grasping arm follows the moved object; rejoins the recording at the lift apex (`plan.py`) |
| F2 interaction location | all arms and held objects move rigidly during the multi-arm interaction window (`plan.py`) |
| replay checks | limits, speed ≤ 1.25×, accel ≤ 1.5×, clearance, hand distance, table, start pose, untouched parts, IK (≤ 1 mm / 0.5° at contact, ≤ 5 mm / 2° while blending), object placement (`checks.py`) |
| rendering | mesh-guided forward warping of real pixels: tm_erode 21, part fill, despeckle, plate relighting σ = 10 px (`render.py`) |
| evaluation | real-ground-truth pose-transfer benchmark, camera-independent region, episode-bootstrap CI (`bench.py`) |
