# sfagen v02 / v02b / v03: arena augmentation for the OpenArm Pringles hand-over

> **Research branch.** Robot setup, teleoperation, recording, training and evaluation commands are on
> [`official-lerobot-quest`](https://github.com/StanleyChueh/lerobot_openarm/tree/official-lerobot-quest#branch-map).

## v03: even arena coverage with a pre-registered distortion cap (revision 2, 2026-11-14, latest)

> **Revision 2 (Hub tag `v3.0`).** Every synthetic episode is exactly as long as its source demo (8.0–9.3 s), and the
> robot moves at the source speed. Revision 1 slowed some reaches down instead (up to 3.56× the source length, ~33 s)
> and is kept at the Hub tag `v03-rev1` as a negative result. The changes are Amendments 4–5 in the protocol.

**Goal:** cover the arena evenly (corners, edges, middle) instead of staying near the 10 real demos. Reject only
episodes whose body-camera distortion exceeds a threshold that was frozen before rendering.

**How the data are made** (code under `research/sim_free_augmentation/sfagen`):
1. Targets: a 3 cm grid of can start positions over x 9–36 cm, y −9 to 15 cm (90 cells).
2. Planning: for each cell, the F1W planner (`sfagen/workspace_plan.py`) re-plans the reach of every real demo
   within 12 cm:
   - no retiming: frame count == source;
   - the shift is spread over a wide blend, from the previous gripper event to just before the next one;
   - rigid near-object reach and vertical transit;
   - action-aware IK bounds;
   - kinematic, speed, view and table gates (`sfagen/checks.py`), plus `duration` (== source frames) and
     `speed_similarity` (moved-arm TCP peak and mean speed 0.8–1.25× the source).
3. Rendering: the body camera is re-rendered by forward-warping the source video (`sfagen/render.py`). The left wrist
   is copied from the source; the right wrist is either copied or rendered (`_wristrender`, an ablation).
4. Distortion score (`sfagen/distortion.py`): per frame, (missing + spilled robot/can silhouette pixels) / silhouette;
   the episode score is the 95th percentile over frames.
   - **strict** ≤ 2.289, the worst v02 episode with shift ≤ 4 cm;
   - **loose** ≤ 4.578, twice strict.
   - Each cell keeps its 2 lowest-score candidates.
5. The v02 hand-over location/timing variants are added; the result is exported as LeRobot v3.0. The 10 real demos
   are added only in the separate `real_and_sim` repos.

**Results** ([RESULTS.md](research/sim_free_augmentation/sfagen/results/v03/RESULTS.md),
[protocol + amendments](research/sim_free_augmentation/sfagen/results/v03/PROTOCOL.md)):
- **Episodes:**
  - strict 130 synthetic (71 hand-over + 59 grid, 35 cells);
  - loose 138 (71 + 67, 39 cells).
- **Duration:** length / source = 1.000 for every synthetic episode. TCP speed vs source: peak 0.86–1.16×, mean
  0.92–1.15×.
- **Coverage of the 40 eval starts** (worst nearest can start):
  - loose 2.70 cm (35/40 within 2 cm);
  - strict 6.63 cm (28/40; rear-right uncovered);
  - v00 3.91 cm (28/40).
- **Checks:** all 8 exports pass verify, quality (including duration and speed similarity) and an independent export
  check.
- **Negative findings:**
  - without slow-down, only 92/416 grid candidates pass (rev 1: 189);
  - strict cannot cover the rear-right;
  - 4 hand-over timing variants are excluded for being too slow;
  - 13 episodes (11 strict) carry small visual flags (notched can top, ghost sliver, dark lid arc).
- No policy, physics or hardware result yet.

![v03 coverage](research/sim_free_augmentation/sfagen/results/v03r2/coverage.png)

| Hugging Face dataset | contents |
|---|---|
| [`ethanCSL/openarm_pringles_lerobot_real_to_sim_10_3cam_v03_strict`](https://huggingface.co/datasets/ethanCSL/openarm_pringles_lerobot_real_to_sim_10_3cam_v03_strict) | 130 synthetic, copied wrists |
| [`ethanCSL/openarm_pringles_lerobot_real_and_sim_10_3cam_v03_strict`](https://huggingface.co/datasets/ethanCSL/openarm_pringles_lerobot_real_and_sim_10_3cam_v03_strict) | 10 real + 130 synthetic |
| [`ethanCSL/openarm_pringles_lerobot_real_to_sim_10_3cam_wristrender_v03_strict`](https://huggingface.co/datasets/ethanCSL/openarm_pringles_lerobot_real_to_sim_10_3cam_wristrender_v03_strict) | 130 synthetic, rendered right wrist |
| [`ethanCSL/openarm_pringles_lerobot_real_and_sim_10_3cam_wristrender_v03_strict`](https://huggingface.co/datasets/ethanCSL/openarm_pringles_lerobot_real_and_sim_10_3cam_wristrender_v03_strict) | 10 real + 130, rendered right wrist |
| [`ethanCSL/openarm_pringles_lerobot_real_to_sim_10_3cam_v03_loose`](https://huggingface.co/datasets/ethanCSL/openarm_pringles_lerobot_real_to_sim_10_3cam_v03_loose) | 138 synthetic, copied wrists |
| [`ethanCSL/openarm_pringles_lerobot_real_and_sim_10_3cam_v03_loose`](https://huggingface.co/datasets/ethanCSL/openarm_pringles_lerobot_real_and_sim_10_3cam_v03_loose) | 10 real + 138 synthetic |
| [`ethanCSL/openarm_pringles_lerobot_real_to_sim_10_3cam_wristrender_v03_loose`](https://huggingface.co/datasets/ethanCSL/openarm_pringles_lerobot_real_to_sim_10_3cam_wristrender_v03_loose) | 138 synthetic, rendered right wrist |
| [`ethanCSL/openarm_pringles_lerobot_real_and_sim_10_3cam_wristrender_v03_loose`](https://huggingface.co/datasets/ethanCSL/openarm_pringles_lerobot_real_and_sim_10_3cam_wristrender_v03_loose) | 10 real + 138, rendered right wrist |

Reproduce v03 revision 2 (configs `configs/openarm_pringles_v03r2*.yaml`, written by
`results/v03/run_scripts/make_configs.py`; same environment as below). The full driver sequence is in
`results/v03/run_scripts/` (`REV=2` selects revision 2). The core steps are:
```bash
cd research/sim_free_augmentation/sfagen
P="env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 /home/stanley/Stanley_ws/lerobot_openarm/.venv/bin/python -B"
$P -m sfagen generate configs/openarm_pringles_v03r2_gen.yaml         # grid candidates + checks
$P -m sfagen render configs/openarm_pringles_v03r2.yaml               # body renders + distortion diagnostics
$P -m sfagen render configs/openarm_pringles_v03r2_wristrender.yaml    # rendered right wrist
# score/threshold/selection (results/v03/thresholds.json, selection.json) then per export, e.g. strict:
$P -m sfagen build configs/openarm_pringles_v03r2_strict.yaml --repo-id <id> --mixed-repo-id <mixed id>
$P -m sfagen verify configs/openarm_pringles_v03r2_strict.yaml
$P tools/quality_report.py configs/openarm_pringles_v03r2_strict.yaml results/v03r2/quality_strict_copied
```
Selection, merge, export, timing and review helpers: `results/v03/run_scripts/` (`v03_steps.py`, `merge_v03.py`,
`make_exports.py`, `build_all.sh`, `interaction_timing.py`, `review_clips.py`). Revision-2 outputs are in `results/v03r2/`;
revision 1 is reproduced with `REV=1` and the `openarm_pringles_v03*.yaml` configs.

## v02: fixed-duration arena augmentation

This research branch implements the approved Step 2 batch from `research/sfagen-wrist-render@77e52d4`.
It adds absolute can targets and v00-style ±10-frame handover timing, keeping sfagen F1/F2 geometry and the body renderer.

**Current result:** 104/189 candidates passed generation: 29/89 positions, 60/80 handover locations, 15/20 timing variants.
Only **18/35 target positions** survived. This batch does not achieve full rear-arena coverage and is not a validated
policy improvement. Both matched synthetic-only and mixed exports pass local verification and the frozen quality gates (53 CPU tests). The existing wrist fidelity limitation and visible image artifacts remain. No training, robot operation or publication.

[Visual evidence](research/sim_free_augmentation/sfagen/results/v02/index.html) ·
[Frozen protocol](research/sim_free_augmentation/sfagen/results/v02/PROTOCOL.md) ·
[Exact input/code hashes](research/sim_free_augmentation/sfagen/results/v02/FREEZE.json) ·
[Step 1 diagnosis](research/sim_free_augmentation/sfagen/results/eval_v00/index.html)

![Accepted versus rejected positions](research/sim_free_augmentation/sfagen/results/v02/accepted_coverage.png)

![Timing and duration checks](research/sim_free_augmentation/sfagen/results/v02/duration_gate.png)

## Reproduce locally

Use the existing main-checkout `.venv`, clear `PYTHONPATH` and `LD_LIBRARY_PATH`, and set `HF_HUB_OFFLINE=1`.
Source revision is `cebdee98115ab6827a4ec2655335c536b5dd085b`; cached inputs and exact paths are frozen in FREEZE.json.
The local configs point at separate output directories; choose fresh work directories for a new export. Never overwrite
an existing dataset. The current protocol is frozen: a new experiment requires a new protocol, not changing its gates.

```bash
cd research/sim_free_augmentation/sfagen
TASK_PY=/home/stanley/Stanley_ws/lerobot_openarm/.venv/bin/python
env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 "$TASK_PY" -B -m unittest discover -s tests -q
env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 "$TASK_PY" -B -m sfagen generate configs/openarm_pringles_v02.yaml
env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 "$TASK_PY" -B -m sfagen render configs/openarm_pringles_v02.yaml
env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 "$TASK_PY" -B -m sfagen render configs/openarm_pringles_v02_wristrender.yaml
# Build each condition only after every passing candidate has current raw renders.
env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 "$TASK_PY" -B -m sfagen build configs/openarm_pringles_v02.yaml --repo-id local/v02-copied --mixed-repo-id local/v02-copied-mixed
env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 "$TASK_PY" -B -m sfagen build configs/openarm_pringles_v02_wristrender.yaml --repo-id local/v02-right-wrist --mixed-repo-id local/v02-right-wrist-mixed
# Verify and quality-check both conditions:
env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 "$TASK_PY" -B -m sfagen verify configs/openarm_pringles_v02.yaml
env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 "$TASK_PY" -B tools/quality_report.py configs/openarm_pringles_v02.yaml results/v02/quality_copied
env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 "$TASK_PY" -B -m sfagen verify configs/openarm_pringles_v02_wristrender.yaml
env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 "$TASK_PY" -B tools/quality_report.py configs/openarm_pringles_v02_wristrender.yaml results/v02/quality_right_wrist
env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 "$TASK_PY" -B tools/v02_report.py
```

The matching condition uses the same states/actions/body images, and only re-renders the right wrist. Copied wrists
carry validity flags. The existing Pringles wrist fidelity gate failed; codec quality does not fix that limitation.
Labels and mesh checks are kinematic proxies, not contact-force or physical robot validation.
