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
# RoboTwin 2.0 `place_empty_cup` (sfagen synthetic + 10 source demos)

LeRobot v3.0 dataset, 138 episodes / 23932 frames at 30 fps, robot `aloha-agilex`
(14-D joint state/action, head camera `cam_high` + 2 wrist cameras, 640x480).

**Source demos.** The first 10 `demo_clean` episodes of `place_empty_cup` in
[`lerobot/robotwin_unified`](https://huggingface.co/datasets/lerobot/robotwin_unified) at revision `1287871839fa`
(global episode indices 15950-15959), split with LeRobot's `split_dataset`
(videos re-encoded once, AV1 CRF 30).

**Synthetic episodes.** Made by *sfagen* (simulation-free augmentation; no physics simulator, no simulator state used)
from the 10 source demos and the robot URDF only: the object (or the hand-over) is moved, the arm trajectory is
retargeted by IK (pipeline v1: Cartesian blend, joint-space fallback near singular poses), and the head-camera
image is re-rendered from the source frames. Every synthetic episode has exactly its source demo's length (1.000x).
- families: object_workspace 128
- shift from its source demo (object position, or hand-over location for interaction episodes): median 5.8 cm, max 11.9 cm
- synthetic episodes per source demo: {'0': 12, '1': 15, '2': 3, '3': 26, '4': 2, '5': 6, '6': 14, '7': 15, '8': 18, '9': 17}
- target positions: 3 cm lattice within 12 cm of the source object positions, kept when the object is visibly in the
  head camera (>= 50 % of its points, or as visible as in its source demo)
- all candidates passed sfagen's kinematic/visual gates (speed/acceleration vs the source demos, IK, joint limits,
  collision, table, object placement); `sfagen verify` passed.
- contains the 10 source demos as well; per-frame flag `is_synthetic` (0 = source, 1 = synthetic).
- per-episode provenance (source demo, family, shift): `meta/sfa_provenance_merged.json`.

**Image-quality warning (place_empty_cup).** mediocre: the cup is moved correctly, but the arm shows black fragments while carrying the cup and cups near the image edge show ghosting. These are research data from a pipeline under development (the camera
calibration is still 16-35 px off); inspect the videos before training on them.

**Load** (`lerobot` >= 0.4): `LeRobotDataset("ethanCSL/robotwin_place_empty_cup_lerobot_source_and_sim_aug_10", revision="v3.0")`. Task string: "Slide the cup for holding liquids onto the round coaster with light streaks with the arm." (see `meta/tasks.parquet`).

**Provenance.** sfagen code commit `cc53c07542` (branch `research/sfagen-simbench`; local, not yet public). Upstream data:
RoboTwin 2.0 (MIT-licensed code, https://github.com/RoboTwin-Platform/RoboTwin); the upstream LeRobot dataset declares no
license, so the RoboTwin terms apply. Please cite RoboTwin 2.0 when using these data.
