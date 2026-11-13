# LIBERO design study: scoping (2026-11-13, Claude). Nothing installed or run.

## Why simulation
On the real robot, each design question (wrist copy vs render, distortion cap, coverage pattern, number of sources)
costs a training run plus about 45 robot trials, and the answer is noisy (v00: 10/20 → 15/20, p = 0.19). In simulation:
- **The counterfactual image exists.** The true render of each synthetic trajectory is available as an oracle, so the
  cost of our body-camera distortion and of copied wrists can be measured exactly.
- **Trials are cheap**, ≥ 50 per condition.
- **Position-shift benchmarks already show VLAs collapsing:** [LIBERO-PRO](https://arxiv.org/abs/2510.03827),
  [LIBERO-Plus](https://arxiv.org/abs/2510.13626) (CVPR 2026).

Simulation picks the design; the real robot confirms the winners. Insights may not transfer fully, because sim images
are cleaner and the robot is a single-arm Panda.

## What exists locally
- `lerobot 0.6.2` in the project venv ships `lerobot/envs/libero.py`:
  - `LiberoEnv`, with cameras `agentview_image` and `robot0_eye_in_hand_image`, i.e. a third-person view and a wrist
    camera, matching our body + wrist setup;
  - suites `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`.
- It also ships `robotwin.py` (RoboTwin 2.0, dual-arm, 50 tasks). This is a closer match to the OpenArm handover;
  propose it as a second stage.
- Not installed: `libero`, `robosuite`. MuJoCo is present. Hardware: 2 × RTX 4090 (24 GB).

## Proposed setup (each item needs approval as marked)
1. **Install (approval: software install).** A separate venv `~/Stanley_ws/libero_venv` with `lerobot[libero]` at the
   same lerobot version. Don't touch the project venv.
2. **Tasks:** 3–4 `libero_spatial` / `libero_object` pick-and-place tasks whose target object start varies. Positions
   follow the LIBERO-PRO / LIBERO-Plus perturbations, plus a uniform grid with corners, matching the real-robot claim.
3. **Sources:** 1, 3 or 10 of LIBERO's human demos per task act as the "real" source; the rest are held out.
4. **Porting sfagen** (engineering, about 2–4 days):
   - **Robot:** a Panda URDF/robot config (7 joints + gripper) from robosuite's model.
   - **Camera:** exact intrinsics/extrinsics from MuJoCo; no calibration needed.
   - **Actions:** LIBERO uses OSC end-effector delta actions, not joint targets. Relabel as Δa(t) = original delta +
     the change of the shifted TCP path (w(t)·d), and verify by kinematic replay error. This is the largest porting
     item.
   - **Depth and masks:** two variants.
     - "real-like": estimated depth + our segmentation path, so the artifacts are comparable to the real rig;
     - "best-case": simulator ground-truth depth and masks, which separates renderer error from perception error.
   - **Wrist:** copy, render (`sfagen/wrist.py`) or drop.
5. **Oracle (approval: rule exception).** Render the *same* synthetic joint trajectory in MuJoCo by setting the
   object's initial pose and replaying the trajectory kinematically (qpos set, no physics stepping).
   - It is used only as an upper-bound condition, never as the proposed method's data.
   - It touches the standing rule "no physics-engine-generated demonstrations", so it needs the user's explicit OK.
6. **Training and evaluation (approval: GPU training + physics simulation).**
   - SmolVLA with the same recipe as on the real robot.
   - Evaluation runs in the simulator with physics, which the rules require approval for.

## Factors (first screen, one task family)
| Factor | Levels |
|---|---|
| Wrist view | copy / sfagen render / dropped / oracle |
| Body view | sfagen render / oracle |
| Distortion cap | strict / loose / none (the same rule as v03) |
| Coverage | near-source only (v00-like) / even grid (v03-like) |
| Real sources | 1 / 3 / 10 |

- The full factorial (4 × 2 × 3 × 2 × 3 = 144) is too many. Run a screening design instead:
  - the main-effect contrasts against one baseline (wrist copy, sfagen body, strict, even grid, 10 sources);
  - the two oracle conditions;
  - real-only.
- That's about 12 trainings.
- Readout: success at held-out perturbed positions, ≥ 50 trials per condition and position set, with fixed seeds and
  Wilson CIs.

## Cost estimate (rough; to re-measure on the first run)
- Training: about 3–4 GPU-h per SmolVLA fine-tune (assumed 20k steps, batch 32, on a 4090). 12 runs ≈ 40–50 GPU-h,
  about 1 day on 2 GPUs.
- Evaluation: about 0.5–1 h per condition with vectorised environments.
- Engineering: 2–4 days for the port, 1 day for checks (kinematic replay of relabelled deltas, oracle-vs-render
  image comparison).

## Decisions needed from the user
1. Install LIBERO in a separate venv: yes / no.
2. Allow the kinematic MuJoCo oracle render (upper bound only): yes / no.
3. Approve the GPU training and simulator evaluation budget (~50 GPU-h).
4. LIBERO first, RoboTwin (bimanual) second: confirm.
