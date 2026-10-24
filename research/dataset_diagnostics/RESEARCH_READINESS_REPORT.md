# Research readiness report: OpenArm Pringles hand-over (real_10 / real_20 / sim_10)

Date: 2026-10-24. Produced read-only with `tools/dataset_diagnostics` (lerobot_openarm `249096010`, uncommitted
tool files) on the exact training datasets. No training, no hardware, no Hub pushes.
Full output: `research/dataset_diagnostics/pringles_v00/dataset_report.html` (+ CSV/JSON/PNG).
State-path audit: `SMOLVLA_STATE_AUDIT.md`. Verdict PASS: the 6‑D declaration is stale metadata, and all 16 state
dimensions are used and trained.

## 1. What was analysed (provenance recovered, not assumed)

| set | Hub SHA | episodes used | origin (per-episode SHA-1 of the state array) |
|---|---|---|---|
| real_10 | `cebdee98` | all 10 | = real_v00 eps 0–9 |
| real_20 | `dad0defb` (local copy, stats equal the checkpoint's normalizer exactly) | all 20 | = real_v00 eps 0–19 |
| sim_10 | `b6d39dae` | all 10 | = sim_20_v00 eps 0–9, Quest Pro teleop in Isaac; `demo_i` of `IsaacLab/logs/demos/openarm_pringles_lerobot_sim_10_v00.hdf5` = episode i (joint trajectories identical) |
| real_v00 (pool) | local | 50 | 30 real episodes (20–49) never used for training |

"All episodes" is backed by `train_config.dataset.episodes = null` and normalizer `count` = `total_frames`.
**Confound:** real_20 trained at batch 16, real_10 and sim_10 at batch 8 (all 20k steps), so real_20 saw 2× as many samples.

## 2. Results

| | real_10 | real_20 | sim_10 |
|---|---|---|---|
| **Technical** | WARN: stale `state [6]` only | WARN: same | WARN: same |
| **Coverage** | UNKNOWN (no metric positions); image space provisional | UNKNOWN; image space provisional | WARN (recorded poses) |
| **Quality** | WARN: 0 severe, 1 timing, 1 spike | WARN: 1 severe (ep 16, release before left close), 3 spikes, 4 length outliers | WARN: 0 severe, 1 slip, 1 timing, 1 spike |
| **Real/sim** | WARN (vs sim_10) | WARN (vs sim_10) | WARN |

**Technical/training compatibility.** Schema, names and order (`RJ1,LJ1,…,RJ8,LJ8`), 3 cameras at 480×640, 30 fps,
timestamps, frame continuity and stats.json all PASS for all three. Each checkpoint's normalizer equals its own
dataset's statistics. Every episode is ≥ the 50-step chunk size.

**Spatial coverage.**
- *sim_10 (source A, metric, robot base frame).* Starts span x 0.134–0.332 m and only **y −0.009…+0.041 m**.
  75% of the sim evaluation distribution (the truncated Gaussian `_RECORD_SPAWN_GAUSSIAN` the recording env samples;
  leave-one-out range 72–75%) lies within 4 cm of a demo; for a uniform target over the spawn box it is 55%. There are no near-duplicates, but the
  kernel effective n is 2.8 at 4 cm. Recommended next sim starts, all on the empty y < 0 side:
  (0.21, −0.01), (0.25, −0.03), (0.17, −0.03), (0.25, 0.07), (0.21, −0.05) m (`sim_10/recommended_positions.csv`).
- *real (no recorded pose, no validated camera transform).* **Metric coverage is UNKNOWN.** Two partial measurements exist:
  - Colour-blob start positions in body_cam (unverified). Every proposal falls on the can, checked by eye by the
    model; the lid/body reference point varies (5/50 on the body). Spread: real_10 u 293–405 px; real_20 u 267–449;
    pool u 248–462.
  - Grasp-position proxy: FK of the right TCP at the first commanded close. In sim it matches the can position at
    the grasp frame (median 5 mm, max 21 mm). It is a *grasp* position: the can is pushed ~2 cm before the grasp.
    Spread: real_10 x 0.126–0.334, y 0.006–0.093; real_20 x 0.126–0.403, **y −0.037–0.112**; sim_10 y 0.004–0.046.
    Five of real_20's ten extra episodes grasp > 3.5 cm from every real_10 grasp. Kernel effective n at 4 cm:
    real_10 3.4, real_20 4.8, sim_10 3.1.

**Real/sim compatibility (sim_10 vs real).** Schema PASS, start pose within 0.017 rad, action leads state by 2
frames in both, per-frame joint speed ratio 1.13–1.19 (the old 20 Hz-labelled-30 fps problem is absent). WARN:
- **The gripper state while holding differs.** The right gripper stalls at openness 0.76 in sim vs 0.43 on the
  real robot (left: 0.46 vs 0.35). `state_proj` sees this every frame of the hold.
- Six arm joints shift by more than 1 pooled std or have a std ratio outside 0.5–2 (LJ2, RJ4, RJ5, LJ5, RJ7, LJ7).
  The sim operator used different arm postures.
- The both-hold overlap at the hand-over is shorter in sim (median 1 vs 3 frames).

## 3. Assessment

**1. Is spatial coverage actually deficient?** For sim, yes, measurably: the y < 0 half of the evaluation
distribution has no demos. For real, it is *plausible but not shown*. real_10 occupies a narrower band than real_20
(image u range 112 vs 182 px; grasp y range 8.7 vs 14.9 cm), and half of real_20's extra episodes add new grasp
locations. But the 20‑vs‑10 gap (≈100% vs 2/10) also contains 2× demos and 2× sample exposure. Without the
evaluation-region positions and per-trial outcomes, coverage cannot be singled out as the cause.

**2. Is a balanced 10-real recollection justified?** Not as the first step. The 50-episode real_v00 pool
already spans a wider area (u 248–462 px, grasp y −7.6…+11.9 cm). A balanced 10-episode subset can be *selected*
from it at zero collection cost once start positions are human-verified. Avoid pool episodes 28–47: 13 of them
release the right gripper at or before the left close (`real_v00_pool/readiness.json`), a late-session operator
habit. Recollect only for evaluation regions the pool does not reach.

**3. Are new simulated demonstrations needed?** A few targeted ones (the five positions above) are cheap and
close a measured gap. However, the larger measured sim→real difference is the gripper hold state (0.76 vs 0.43
openness) and arm posture, not position. Fix the sim gripper stall level, or record it as a known gap, before
reading co-training results as a coverage effect.

**4. Fair comparison of original 10, balanced 10 and real+sim co-training.**
- Use the same batch size and steps for every arm. real_20 must be retrained at batch 8, or the others at 16.
  Use ≥3 seeds per arm.
- Use one evaluation protocol: the 5 fixed regions × ≥10 trials each, with the same can pose per region, in a
  randomized trial order. Report per-region success with Wilson intervals; at 10 trials per arm, 2/10 vs 5/10 is
  not distinguishable.
- Arms: (a) original real_10, (b) ≥3 *random* 10-episode subsets of the clean pool (separates "different
  episodes" from "balanced"), (c) coverage-balanced 10, (d) real_10 + sim_10 merged with `lerobot-edit-dataset`
  (merged normalizer stats), (e) real_20 as reference.
- Log per-trial start positions and outcomes so failures can be mapped back onto the coverage map.

**5. Standard engineering vs possible research.** Standard: schema, timing and stats checks, checkpoint
compatibility, occupancy/KDE heatmaps, nearest-demo distance, greedy farthest-point recommendations, and
homography validation by held-out error. Possible research contributions, only if they predict or improve
success:
- (i) Coverage measured in a policy-relevant space: the grasp-configuration proxy, calibrated against sim ground
  truth, links object space to the joint space the policy acts in.
- (ii) Evaluation-distribution-weighted coverage with leave-one-out fragility as a *pre-training* predictor of
  per-region success.
- (iii) Using sim ground truth to validate a real-camera position estimator (the detector reproduces sim poses to
  ~1 mm held-out; the real camera still needs its own correspondences).
- (iv) Closing the loop from evaluation failures to collection targets.

A heatmap alone is not a contribution.

**6. Experiment that would show coverage-guided collection beats the alternatives.** Use a fixed budget of K=10
episodes, all strategies drawing from the same candidate pool. That removes operator-skill drift, which the pool
shows exists. Strategies:
- random (≥5 draws);
- grid-uniform (nearest pool episode to each cell centre);
- diversity-only (farthest-point in trajectory space, e.g. `analyze_dataset_quality` nn_dist features);
- coverage-guided (maximize evaluation-weighted coverage of verified start positions, quality flags excluded).

Train with matched compute and 3 seeds. Evaluate on the 5 regions plus held-out random positions. The claim holds
only if coverage-guided wins on per-region success *and* the pre-training coverage score rank-correlates with
success across all subsets. Run it first in Isaac, where 100-episode randomized evaluation is cheap and start
poses are recorded: a pool of sim teleop or generated demos. Then confirm the winning pair on the real robot with
the fewest runs.

## 4. Missing inputs (exact)

1. Human-verified real start positions: `python -m tools.dataset_diagnostics annotate --root ~/.cache/huggingface/lerobot/ethanCSL/openarm_pringles_lerobot_real_v00 --label real_v00_pool --camera observation.images.body_cam --propose tools/dataset_diagnostics/configs/pringles_can_detect_real.json --interactive --output research/dataset_diagnostics/annotations/real_v00.csv`
   (50 clicks). real_10/real_20 reuse it via `annotations:research/dataset_diagnostics/annotations/real_v00.csv#real_v00_pool`
   in their manifest entries (their episodes are pool eps 0–19 with the same indices).
2. The 5 evaluation regions as body_cam pixels (or metric positions). Put them in `pringles_body_cam_image.json`
   as `eval_distribution: {"type": "points", ...}` and `regions`.
3. The reachable boundary for the real can (image polygon or metric box). Recommendations stay off until
   `boundary_measured` is true.
4. For metric real coverage: ≥ 6 tape-measured can positions in the robot base frame, with their body_cam pixels
   → `fit-transform` (refused unless the held-out error ≤ 2 cm).
5. Per-trial logs (start region, outcome) of the real evaluations (real_10 2/10, sim_10 1/10) and of the Isaac
   randomized runs (52/100, 15/72).
