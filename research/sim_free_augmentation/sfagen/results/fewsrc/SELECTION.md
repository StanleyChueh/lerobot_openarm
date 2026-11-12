# C1: source-only demo selection rule (frozen before any coverage number)

**Date:** 2026-11-12.

**Rule:** greedy farthest-point (k-center) over the 10 demos' recorded can positions.
- **Position:** FK of the right TCP at its first gripper close, the same proxy the coverage tool uses.
- **Start:** the demo nearest the centroid.
- **Input:** only the source demos (`ethanCSL/openarm_pringles_lerobot_real_10_v00@cebdee98`). The evaluation pool is never
  used.

**Output:** `selection.json`, written by `tools/select_sources.py configs/openarm_pringles_fewsrc.yaml`.
- Order: 3, 8, 5, 1, 9, 0, 2, 7, 6, 4.

| N | Demos |
|---|---|
| 1 | {3} |
| 2 | {3, 8} |
| 4 | {1, 3, 5, 8} |
| 10 | all |

These subsets define the v01 candidate datasets. Coverage (C3) is computed only after this file is committed.
