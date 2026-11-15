# R02 — pipeline v1: joint-space fallback for near-singular robots (fixes R01 Problem 1); RoboTwin S1 completed

Dates 2026-11-15/16 · executor Opus 5.5 · branch `research/sfagen-simbench` (local commits only) · first-party result,
pending independent review. Protocol, amendments and freezes: `results/r02/PROTOCOL.md`, `FREEZE*.json`.

## In plain words
- **Before.** RoboTwin produced 0 usable synthetic episodes. sfagen shifted the object starting at frame 0 and solved
  the exact gripper pose on every frame. RoboTwin's arm starts folded at a near-singular pose, so the wrist had to
  twist in a way no demo ever does, and the speed check rejected every candidate.
- **Pipeline v1.** Every candidate is first made exactly as before (v0). Only if v0 fails a check is the motion re-made
  with a **joint-space blend**:
  - the gripper pose is enforced exactly only near the grasp (from 15 frames before it);
  - the free-space travel before and after is a smooth joint-space transition.
- **Robot-agnostic.** It needs nothing but the URDF; there is no threshold and nothing to tune per robot or task.
- **OpenArm.** Every v0 OpenArm episode is bit-identical, so the published data cannot get worse. The fallback adds
  OpenArm episodes where v0 failed.

## OpenArm: not worse (user requirement), measured
Loose v03r2 config, current code, generate only (`artifacts/simbench_runs/r02_openarm_{cartesian,fallback}`):

| | v0 (`cartesian`) | **v1 (`joint_fallback`)** |
|---|---|---|
| grid passes | 92/416 (= published v03r2) | **154/416** |
| grid targets with ≥ 1 pass | 39 | **46** |
| interaction location / timing | 60/80 · 11/20 | **62/80 · 11/20** |
| v0-passing candidates bit-identical in v1 | — | **163/163** |
| body renders, ep 5, vs published v03r2 renders | — | 9/12 byte-identical; 3 interaction episodes differ by 1–17 px of 78 M, and a fresh v0-config render differs by the same amount (GPU render nondeterminism, not v1) |
| distortion of grid episodes (median / p90 / max) | 0.868 / 2.249 / 3.759 (n = 92) | added episodes: 0.847 / 2.364 / 3.721 (n = 62); loose threshold 4.578 |

Guard with the default setting: 28/28 candidates and 4/4 bodies bit-identical (`guard_blendspace_a1`). 61/61 tests
pass (58 old + 3 new in `tests/test_blend_space.py`).

## RoboTwin (aloha-agilex): episodes generated and exported to LeRobot
Run `artifacts/simbench_runs/s1d` (pipeline v1, all gates unchanged, 1.000× length, 10 demo_clean sources per task).

| task | object position (grid) | interaction location | timing | selected → LeRobot episodes | frames |
|---|---|---|---|---|---|
| place_empty_cup | **71/339 (21 %)**, 41 targets | n/a (hands never meet) | n/a | 60 | 10,401 |
| stack_blocks_two | **145/261 (56 %)**, 59 targets | n/a | n/a | 99 | 31,036 |
| handover_block | **0/363** (120 targets leave the camera view; speed 156) | 43/80 | 20/20 | 63 | 17,849 |
| (v0 on cup, R01) | 0/339 | | | 0 | |

**LeRobot datasets** (v3.0, 30 fps, synthetic only, verified):
`artifacts/simbench_data/robotwin_sfagen_v1/{place_empty_cup,stack_blocks_two,handover_block}`. Each has
`meta/sfa_provenance_merged.json` (source episode, shift, family, group).
- Independent check (`results/r02/robotwin_v1_datasets.json`): length/source = 1.000 for all 222 episodes; 0 real
  episodes; samples load.
- `sfagen verify` passes for all 7 group exports. Total size 1.4 GB.

**Side-by-side videos**: `artifacts/simbench_runs/s1d/<task>/<group>_export/compare_eps_*_1x.mp4` (6 videos).
Frame sheets: `results/r02/cup_L_compare_frames.jpg`, `results/r02/stack_handover_compare_frames.jpg`.

## Quality: what the frames show (inspected by eye)
- **stack_blocks_two:** mostly clean. The block is moved and grasped convincingly; occasional arm fragments where the
  arm enters from the image edge.
- **place_empty_cup:** small shifts are plausible. There is a white "ghost" patch at the old cup area, and parts of the
  gripper are erased. At 9 cm shifts, black arm fragments float near the image edge.
- **handover_block** (interaction variants): white holes and arm fragments around the re-posed arms.
- **The distortion score misses these.** Every selected RoboTwin episode scores under OpenArm's strict threshold
  (medians 0.35–0.46, max 1.96; `distortion_robotwin_selected.json`), yet the fragments are visible. This is the
  known limitation from the v03 visual audit.
- **Likely causes:** the camera fit is still 16–35 px from truth (R01 Problem 2), the renderer re-posing arms at the
  image edge (P5), and table shadows.
- **Verdict:** trajectories and labels are valid. The RGB is **not yet training-quality**; treat these datasets as a
  pipeline demonstration, not as benchmark training data.

## Problems and options
1. **RGB fragments on RoboTwin** (T5/T9; blocks a fair benchmark). Options:
   - (a) fix the camera fit first (R01 P1: fit on SAM2 arm masks), then re-render. General; about 1 day.
   - (b) renderer variants judged against the simulator oracle frames (plan P5). General; costly.
   - (c) add a fragment detector to the quality gates (connected dark components outside the rendered arm mask).
     General, but it is a new gate, so pre-register it first.

   Recommend (a) and then measure again; the camera is the cheapest root cause.
2. **Handover object position 0 %.** The env's randomisation range extends past the head camera's view, so shifted
   blocks leave the image (in-view gate). Options:
   - (a) restrict targets to the part of the arena inside the view (projection of the arena through the fitted
     camera). General; small.
   - (b) accept and report.

   Recommend (a), because the in-view part can be computed from the camera alone.
3. **Tooling bug fixed:** `s1_task.sh` read configs before rewriting them for the new run dir.
   - Effect: the `s1c` run was invalid (`s1c/INVALID.txt`). Its source-position recon also **added** cache files under
     `s1/handover_block/L/cache` and `s1/stack_blocks_two/*/cache`. Nothing was overwritten; this was a write into an
     older run dir.
   - Fix: the driver now writes configs first and asserts the rig files are present.

## Pre-registration record (negative results kept)
- **Registered variant (joint blend everywhere): H1 failed.** Cup 8/339. The cause was a velocity kink at the blend
  edge, plus a wrist-yaw offset that is genuinely needed at 10 cm from the object.
- **Amendment 1** (Hermite edges; exact segment from `t_grasp − 15`, the F1 convention):
  - cup 71/339, so H1 passes;
  - OpenArm 111/416 but 38 targets instead of 39: one target lost. 43 OpenArm candidates passed only with v0; the
    joint blend failed them on the object check (hand nearer the can than in the demo).
- **Amendment 2** (`joint_fallback`, written before running):
  - H2-A2 holds exactly: 163/163 bit-identical, 154/416, 46 targets;
  - H1-A2 holds: cup 21 %;
  - H3-A2 holds on total yield: handover 14 %, stack 40 %, though handover's object-position family is 0 %.

## What the adaptation to a new robot took (M4)
- **Per robot (aloha-agilex):**
  - URDF with an added TCP frame (`make_aloha_agilex_urdf.py`) and DAE→STL meshes (separate venv);
  - the arm joint names / state indices / gripper mapping (the robot block in `make_configs.py`, about 10 lines);
  - camera intrinsics and a rough extrinsic guess.
- **Per task:** `objects` (proxy box size, grasping arm per episode group) and the arena from the env's randomisation
  range.
- **Recipe knobs changed:** only `plan.blend_space: joint_fallback`, identical for every robot and task.
- **Still manual:** episode groups by grasping arm (A13).
- **GPU time:** calibrate about 30 min, table about 2 h, recon about 3.5 min/episode, render 14–25 s per episode, all
  for 3 tasks on 2 GPUs.

## Not checked
- Policy training on these datasets.
- Simulator replay of the synthetic actions (scoring the actions in RoboTwin, plan M1).
- Wrist-camera correctness (copied wrists).
- Whether the OpenArm v1-added episodes help a policy.
