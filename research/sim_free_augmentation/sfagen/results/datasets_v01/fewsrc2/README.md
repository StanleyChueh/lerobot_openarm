---
license: apache-2.0
task_categories: [robotics]
tags: [LeRobot, openarm, synthetic, simulation-free, few-shot]
configs:
- config_name: default
  data_files: data/*/*.parquet
---
# OpenArm Pringles hand-over: synthetic-only, from 2 real demos (sfagen v01 candidate)

**51 synthetic episodes / 15934 frames, 3 cameras, 30 fps, LeRobot v3.0.** These are generated from the real demos
[3, 8] of [`ethanCSL/openarm_pringles_lerobot_real_10_v00@cebdee98`](https://huggingface.co/datasets/ethanCSL/openarm_pringles_lerobot_real_10_v00).
The dataset contains **no real episodes**. No simulator and no learned generator were used.
**Candidate for a SmolVLA few-source screen; not yet validated by policy training or on the robot.**

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("ethanCSL/openarm_pringles_lerobot_real_to_sim_2_3cam_v01", revision="v3.0")
# mask wrist cameras where batch["wrist_view_valid"] == 0 (order: wrist_cam, right_wrist_cam)
```

## How it was made (sfagen, code commit `92f602004232`, uploaded under `sfagen/`)
- **Demos:** chosen by a rule frozen before any evaluation: k-center over the recorded can positions. The order is
  3, 8, 5, 1, …, so N=1 → {3}, N=2 → {3, 8}, N=4 → {1, 3, 5, 8}.
- **Variation (F1W):** the can starts elsewhere on the table, **shifted at most 11 cm**. That is the measured
  body-camera limit r* (study R1). The grasping arm's reach is translated rigidly by IK, and the time is retimed to fit
  the speed and acceleration envelope of **these 2 demos only**. Every variant that passes that envelope is kept.
- **Body camera:** mesh-guided forward warping of the real pixels, with a clean plate and real-pixel borrowing. This is
  the published renderer, unchanged: the can-fix candidates did not pass their registered test (study A).
- **Wrist cameras:** copied: the time-remapped real source frame. The registered wrist gate failed for Pringles, so the frames are not re-rendered, and `wrist_view_valid` = 0 wherever the scene changed.

## Quality gate (recomputed from this export; `quality_report.json`)
| Check | Result |
|---|---|
| finite, joint limits, start state = source, grippers bit-identical to source | pass |
| max speed / acceleration ratio vs the 2-demo envelope (limits 1.25 / 1.5) | 1.08 / 1.49 |
| max jerk ratio (reported) | 2.16 |
| lowest arm point vs source (table) | 0.2750 m vs 0.2750 m |
| labels = generator labels (SHA-256), no episode equals a real one | pass |
| video spans, decoding, body decoded vs rendered | pass, 39.58 dB |
| body pixels inpainted / borrowed (mean) | 0.015 % / 0.076 % |
| body flicker vs source (median ratio) | 1.01 |
| can shift (median / max) | 3.7 / 10.9 cm |
| episode length (median synthetic / source) | 282 / 250 frames |
| LeRobot load + 50-step action chunk | pass |
| can-position coverage of 40 unseen real recordings (within 2 cm; proxy) | 24 / 40 |

**Wrist cameras** (wrist gate on held-out Pringles demos 5–9 and on battery; T1 render − copy, 95 % CI):

| Camera | Gate | In this dataset | `wrist_view_valid` = 1 |
|---|---|---|---|
| `observation.images.right_wrist_cam` | battery: pass (+1.64 dB [+1.35, +1.93]); pringles_heldout: FAIL (+1.01 dB [+0.67, +1.31]) | copied, invalid where the scene changed | 0.36 |
| `observation.images.wrist_cam` | battery: FAIL (+1.47 dB [+1.09, +1.80]); pringles_heldout: FAIL (+0.82 dB [+0.74, +0.90]) | copied, invalid where the scene changed | 0.36 |

## Limits
- Labels are nominal kinematic labels, not measured execution. No physics, force, or hardware replay was done.
- Body-camera fidelity: about 16 dB PSNR against real frames when the can is moved across episodes. The dominant error
  is the reconstructed object pose (about 1–2 cm).
- Coverage is an FK proxy on reused development recordings.
- Wrist mounts are photometric consistency calibrations, not ChArUco extrinsics.

Full study, protocols and reproduction commands: [GitHub `research/sfagen-wrist-render`](https://github.com/StanleyChueh/lerobot_openarm/tree/research/sfagen-wrist-render).
