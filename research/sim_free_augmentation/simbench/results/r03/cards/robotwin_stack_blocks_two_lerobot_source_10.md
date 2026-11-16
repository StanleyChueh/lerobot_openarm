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
# RoboTwin 2.0 `stack_blocks_two` (10 source demos)

LeRobot v3.0 dataset, 10 episodes / 3140 frames at 30 fps, robot `aloha-agilex`
(14-D joint state/action, head camera `cam_high` + 2 wrist cameras, 640x480).

**Source demos.** The first 10 `demo_clean` episodes of `stack_blocks_two` in
[`lerobot/robotwin_unified`](https://huggingface.co/datasets/lerobot/robotwin_unified) at revision `1287871839fa`
(global episode indices 24750-24759), split with LeRobot's `split_dataset`
(videos re-encoded once, AV1 CRF 30).

This repo contains only those 10 source demos (no synthetic data). They are the inputs of the companion `sim_aug` datasets.

**Load** (`lerobot` >= 0.4): `LeRobotDataset("ethanCSL/robotwin_stack_blocks_two_lerobot_source_10", revision="v3.0")`. Task string: "Set red block at the center, then stack green block over red block after placing it alongside." (see `meta/tasks.parquet`).

**Provenance.** sfagen code commit `951d725a26` (branch `research/sfagen-simbench`; local, not yet public). Upstream data:
RoboTwin 2.0 (MIT-licensed code, https://github.com/RoboTwin-Platform/RoboTwin); the upstream LeRobot dataset declares no
license, so the RoboTwin terms apply. Please cite RoboTwin 2.0 when using these data.
