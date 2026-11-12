# Literature check for the current weaknesses (2026-11-12)

This is the standing rule for this branch: check recent work before designing a fix. Each entry records what the paper
shows and what it changed here.

| Weakness | Paper | What it shows | Effect on sfagen |
|---|---|---|---|
| Few source demos | **Real2Edit2Real** (arXiv 2512.19402, CVPR 2026) | 1–5 real demos → 200 generated demos, trained on generated data only; beats 50 real demos (78.8–81.3 % vs 61.3 %) on 4 real tasks. Metric VGGT depth, arm depth re-rendered from the URDF, depth-conditioned video model for head and both wrists. Ablation: depth quality drives output quality. | Few-source generation is a realistic target. Wrist-view depth needs care (here: three options compared on dev T1). Their video model, trained on 7K episodes, is not reproducible here. |
| Wrist views | **Augmentron** (arXiv 2611.09145) | Wrist views from FK + calibrated mount + a calibrated table plane + room cubemap, with real robot and object pixels composited back. | Supports the geometric wrist renderer: the table plane from the body-camera reconstruction, real pixels kept. |
| Wrist views | **WristWorld** (arXiv 2510.07313) | VGGT-based 4D world model generates wrist views from third-person views (CALVIN +3.8 %). | Generative and needs training. Kept as the fallback if the geometric gate fails. |
| Wrist calibration | **Bi-JCR** (arXiv 2505.24819) | Markerless hand-eye calibration of two wrist cameras with DUSt3R + FK, with metric scale. | Fallback for B2. It needs `dust3r` (not installed); our photometric refinement came first. |
| Moved-object appearance | **ConCent** (arXiv 2606.30268) | Projects object shadows on the ground plane as an augmentation. | Shadow transfer listed as an A2 candidate. Not built, because A1 showed pose misalignment dominates. |
| Moved-object appearance | **RoboSplat** (arXiv 2504.13175), **NeRF-Aug** (arXiv 2411.02482) | Replace objects with 3DGS/NeRF models built from dense captures. | We have no dense capture. A2's per-point colour atlas is the lightweight equivalent. A finds the remaining error is object *pose*, which dense models would also fix. |
| Generation without simulation | **DemoGen**; **R2RGen** (arXiv 2510.08547) | Point-cloud editing from 1 demo. Real2Edit2Real notes this does not fit multi-view RGB policies. | sfagen stays RGB-native and multi-camera. |

**Searches:**
- wrist-view generation for manipulation;
- simulation-free demo generation with object relocation;
- markerless eye-in-hand calibration;
- shadow and appearance harmonisation for relocated objects.

**Not read in full:** the papers' own limitation sections, except where quoted.
