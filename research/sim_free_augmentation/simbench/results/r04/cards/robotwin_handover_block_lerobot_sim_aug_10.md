---
license: other
license_name: robotwin-upstream
license_link: https://github.com/RoboTwin-Platform/RoboTwin/blob/main/LICENSE
task_categories: [robotics]
tags: [LeRobot, robotwin, aloha-agilex, sim-free-augmentation, sfagen]
configs:
- config_name: default
  data_files: data/*/*.parquet
---
# RoboTwin 2.0 `handover_block` (sfagen synthetic only)

LeRobot v3.0 dataset, 100 episodes / 28317 frames at 30 fps, robot `aloha-agilex`
(14-D joint state/action, head camera `cam_high` + 2 wrist cameras, 640x480).

**Source demos.** The first 10 `demo_clean` episodes of `handover_block` in
[`lerobot/robotwin_unified`](https://huggingface.co/datasets/lerobot/robotwin_unified) at revision `1287871839fa`
(global episode indices 4400-4409), split with LeRobot's `split_dataset`
(videos re-encoded once, AV1 CRF 30).

**Synthetic episodes.** Made by *sfagen* (simulation-free augmentation; no physics simulator, no simulator state used)
from the 10 source demos and the robot URDF only: the object (or the hand-over) is moved, the arm trajectory is
retargeted by IK (pipeline v1: Cartesian blend, joint-space fallback near singular poses), and the head-camera
image is re-rendered from the source frames. Every synthetic episode has exactly its source demo's length (1.000x).
- families: object_workspace 94, interaction_timing 3, interaction_location 3
- shift from its source demo (object position, or hand-over location for interaction episodes): median 7.0 cm, max 12.0 cm
- synthetic episodes per source demo: {'0': 9, '1': 10, '2': 9, '3': 11, '4': 9, '5': 10, '6': 9, '7': 11, '8': 12, '9': 10}
- target positions: 3 cm lattice within 12 cm of the source object positions, kept when the object is visibly in the
  head camera (>= 50 % of its points, or as visible as in its source demo)
- **even spread (R04):** exactly 100 synthetic episodes, chosen round by round over 3 cm cells (the emptiest cell
  first, ties by distance to everything already chosen; real demos count as occupied) so no area of the table is
  over-represented. Training-density evenness (episodes within 4 cm over the target lattice): CV 0.95
  -> 0.68 vs the first release (r03, 155 synthetic episodes; still reachable at tag `r03`).
- all candidates passed sfagen's kinematic/visual gates (speed/acceleration vs the source demos, IK, joint limits,
  collision, table, object placement); `sfagen verify` passed.
- contains NO source episodes.
- per-episode provenance (source demo, family, shift): `meta/sfa_provenance_merged.json`.

**Image-quality warning (handover_block).** poor for the block-position episodes (the 20 cm block re-renders distorted, arm fragments); the handover location/timing episodes look better. These are research data from a pipeline under development (the camera
calibration is still 16-35 px off); inspect the videos before training on them.

**Load** (`lerobot` >= 0.4): `LeRobotDataset("ethanCSL/robotwin_handover_block_lerobot_sim_aug_10", revision="v3.0")`. Task string: "Pick up the red block using the left arm, give it to the right arm, and place it on the blue pad." (see `meta/tasks.parquet`).

**Provenance.** sfagen code commit `379bd2b91f` (branch `research/sfagen-simbench`; local, not yet public). Upstream data:
RoboTwin 2.0 (MIT-licensed code, https://github.com/RoboTwin-Platform/RoboTwin); the upstream LeRobot dataset declares no
license, so the RoboTwin terms apply. Please cite RoboTwin 2.0 when using these data.
