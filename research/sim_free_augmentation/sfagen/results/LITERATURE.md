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

# Update 2026-11-14 (SIM_BENCHMARK_PLAN S0, Opus 5.5)

Papers read for the benchmark plan (`SIM_BENCHMARK_PLAN.md` §9) and a code-release check made on 2026-11-14
(project pages and the GitHub API).

| Paper | Needs beyond one RGB camera + URDF | Code released? (checked 2026-11-14) | Effect on sfagen |
|---|---|---|---|
| **DemoGen** (arXiv 2502.16932) | depth; point-cloud policy | (not re-checked) | 3-D policies only; not a baseline for 2-D VLAs |
| **R2RGen** (arXiv 2510.08547, RSS 2026) | depth; scanned object templates | **No.** `github.com/r2rgen/R2RGen` holds only `README.md` + `demo.ipynb` (19 KB, last push 2026-04-29); page says "Code (coming soon)" | compare on input requirements only |
| **Real2Edit2Real** (arXiv 2512.19402, CVPR 2026) | 3 cameras; VGGT; video model tuned on ~7K episodes | **Yes.** `github.com/Real2Edit2Real/Real2Edit2Real`, Apache-2.0, last push 2026-08-04 (`editing/`, `videogen/`, `vggt/`, `configs/`); also a Hugging Face page | **Changes §9:** "none of the real-to-real pages links code" is no longer true. A runnable generative baseline exists; whether it runs on RoboTwin (3 cameras: yes) within 24 GB is an open question for the user |
| **R2RDreamer** (arXiv 2606.17040) | depth; tuned image-to-video model | No (project page anonymous, "ICRA 2027 Submission", no code link) | input-requirement comparison |
| **RoboSplat** (2504.13175), **Real2Render2Real** (2505.09601) | dense multi-view capture / object scan + IsaacLab | (not re-checked) | not post-hoc on teleop data |
| **MimicGen** (2310.17596) / **DexMimicGen** (2410.24185) | simulator + digital twin | yes (known) | basis of the R10+M reference arm |
| **Real2Sem2Real** (arXiv 2610.28880, new) | one human RGB video; Gaussian-splat tracking; IsaacSim / MuJoCo rendered robot demos | No ("Paper and code coming soon") | uses a simulator to render; different input class |

Searches: "R2RGen github code release", "Real2Edit2Real github", "R2RDreamer 2606.17040 code", "Real2Sem2Real 2610.28880";
then each project page and `gh api repos/<owner>/<repo>/contents`.

# Update 2026-11-15 (R01 Problem 1: near-singular start poses; Opus 5.5)
Question: how do demo-generation methods retarget the free-space transit vs the contact segments?
| Method | Contact / skill segment | Free-space transit | Source |
|---|---|---|---|
| MimicGen (CoRL 2023) | object-frame segment replayed in Cartesian EE space | linear EE interpolation between segments (scene-unaware; collisions = generation failures) | Isaac AutoData docs "Generation Algorithms" |
| SkillMimicGen / SkillGen (CoRL 2024, arXiv 2410.18907) | skill segment replayed relative to the object | collision-free motion planning to the segment start (held object attached); planning failure = attempt failure | arXiv 2410.18907; isaac-sim.github.io/AutoData skillgen |
| DemoGen (RSS 2025, arXiv 2502.16932) | skill segment transformed with the object (MimicGen strategy) | motion segments re-planned (TAMP-style) | arXiv 2502.16932 |
Consequence for sfagen: enforcing the shifted Cartesian path on transit frames is not required by any of these; they only
keep the contact segment exact. sfagen "pipeline v1" therefore keeps exact IK only on full-shift (contact/hold) frames and
blends the transit in joint space (no planner; collisions are judged by the unchanged mesh-clearance gate). This is
standard practice, not a novelty claim. Joint-space transit avoids Cartesian-path singularities by construction
(RoboTwin start pose all-zero, near-singular; R01).
