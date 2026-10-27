# Matched evaluation protocol: same 100 can poses, 330 simulated steps

Question: *when evaluation positions and simulated time are controlled, does training on 500 generated
demonstrations improve task success or stable hand-over compared with 10 manual demonstrations?*

Priorities 1–2 of `../NEXT_EXPERIMENT_PROTOCOL.md`, A0 vs B0 only. No training, no generation, no Hub push.

## 1. What was run, exactly

| | value |
|---|---|
| Model A (manual-10) | `ethanCSL/openarm_pringles_lerobot_sim_smolvla_10_v00` @ `9ce7ce4811de11adcc4f407eb5a93586d94a6aee`, loaded from the local HF cache snapshot of that revision (not the 10-real checkpoint) |
| Model B (generated-500) | `ethanCSL/openarm_pringles_lerobot_sim_10_generated_500_smolvla_v00` @ `733c6856a507a25875ee2a75b3458756bc60547b`, same |
| poses | `../protocol/eval_poses_100.json`, SHA-256 `2b95ecf0dc6e4253023277d23bd32bfe2dd3b2587afede68066a1e6d3ce3b3c1`, trial i = pose i, both models, same order |
| horizon | `--episode_steps=330` policy-controlled simulation steps (11.0 s of sim time at 30 Hz); reset settle, return home and reset wait excluded |
| lerobot_openarm | `149bd6e1` (official-lerobot-quest) + the uncommitted matched-eval edits (IMPLEMENTATION_AUDIT §3); worktree hash in each `run_meta.json` |
| IsaacLab | `e684cb65` (openarm-dev) + the uncommitted `lerobot_sim_server.py` reporting edit. `e684cb65` vs the historical runs' `4aa5e026`: only asset auto-download hooks (no task/physics change; the server ran with `HF_HUB_OFFLINE=1`, local assets) |
| lerobot fork | `f10cbf97` (update_dev_v2), clean; eval venv `.venv-v2` (py3.12) |
| GPU | RTX 5080 16 GB, both servers and the client on one machine |

Settings (identical for both models; identical to the historical runs except where marked):

| | |
|---|---|
| simulator | `lerobot_sim_server.py --enable_cameras --headless --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v1 --task_mode handover --domain_randomization none` (lockstep, 30 Hz, 3 settle steps) |
| policy server | `policy_server --rtc=true --rtc_execution_horizon=10 --rtc_max_guidance_weight=10.0 --rtc_prefix_attention_schedule=EXP` |
| client | `--policy_type=smolvla --policy_device=cuda --actions_per_chunk=50 --chunk_size_threshold=0.5 --aggregate_fn_name=latest_only --fps=30 --reset_time_s=1` |
| changed vs historical | `--episode_steps=330` instead of `--episode_time_s=15`; fixed poses; `--display_data=false` (historical: rerun on; no keyboard listener: DISPLAY unset); local-only recording (`--push_to_hub=false`) |

The server and client are restarted between nothing: one simulator process and one policy server served the smoke
tests and both full runs (the policy server reloads the policy on each client's `SendPolicyInstructions`).

## 2. Pose file validation (tests/test_matched_eval.py)

* 100 poses, trial order 0..99, region counts equal the file's own `region_counts`.
* Every pose inside the task's recording spawn range (x ∈ [0.12, 0.40], y ∈ [−0.08, 0.08]) and outside every arm and
  hand keep-out box — constants parsed from `openarm_task_modes.py` at the commit used (`_CAN_GROWTH` = 0 for the
  0.0293 m can radius).
* Regions: x < 0.203: 35, 0.203–0.27: 44, x ≥ 0.27: 21 (none at x ≥ 0.27 with y ≥ 0.03: hand keep-out). The file's
  own note: its x-proportions are approximate (KS p = 0.03 against the 200 estimated historical starts).
* Applied in the simulator: verified per trial from the server's ground truth (§3).

## 3. Per-trial checks (all recorded in `trials.jsonl`, verified in `comparison.json → checks`)

* requested pose = `poses[i]` for both models; actual can pose after the reset within 2 mm (x-y), 4 mm (z vs the task's
  rest height 0.3801 m), upright, hand-over stage 0 — else the run stops (`PoseNotApplied`).
* exactly 330 policy steps, contiguous simulator step counter 1..330; zero scene resets during the trial.
* metrics from those 330 steps only, before the return home (`success_before_return`); `success_incl_return` — what
  the old wall-clock mode would have reported — kept separately.
* incomplete trials (key, stall ≥ 60 s without a simulation step, error) are labelled incomplete, never success or
  failure. None occurred (see RESULTS_SUMMARY).

## 4. Metrics (simulator ground truth; `matched_eval.trial_metrics`)

1. **any-step success** — the task's `handover_success` true at any of the 330 steps (the historical metric, minus the
   return phase).
2. **stable hand-over** — a pass happened (stage ≥ 2), then ≥ 30 consecutive steps with the left hand confirmed holding
   (`receiver_grip_confirmed`: on the can, aperture in band), the right hand not holding (`object_grasped_by`), and
   the can above rest height + 5 mm. Recomputed independently of the task's counter; must equal its stage-3 step
   (`stable_matches_stage`). Within the 330 policy steps, metric 2 ≡ metric 1 by the task's construction
   (`handover_success` = stage 3 ∧ grip counter ≥ 30); the recomputation is a consistency check, not a new criterion.
   Cross-checked against the diagnosis's gripper-only detector (`detector_agreement`).
3. **holding at the end** — at step 330: left confirmed holding, right not, can above rest + 5 mm. Also at 240/270/300.

Stage outcome per trial: `no_right_grasp`, `right_grasp_no_pass`, `pass_not_secured`, `stable_then_lost`,
`stable_held_to_end`.

## 5. Statistics (`analyze.py`)

Wilson 95 % intervals; exact McNemar (binomial on discordant poses) with the paired difference and a Wald CI;
per-region counts with exact McNemar (descriptive, small n); policy × position interaction by conditional logistic
regression for matched pairs (= logistic regression of "generated wins" on standardized x, y among discordant poses;
likelihood-ratio test of the two position terms).

## 6. Reproduce

```bash
# terminal 1: simulator
cd ~/Stanley_ws/IsaacLab && source ~/miniforge3/etc/profile.d/conda.sh && conda activate env_isaaclab
HF_HUB_OFFLINE=1 ./isaaclab.sh -p scripts/tools/lerobot_sim_server.py --enable_cameras --headless \
  --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v1 --task_mode handover --domain_randomization none

# terminal 2: RTC policy server
cd ~/Stanley_ws/lerobot_openarm
env -u PYTHONPATH LD_LIBRARY_PATH=/usr/local/cuda/lib64 .venv-v2/bin/python -m lerobot_robot_openarm_umeow.policy_server \
  --host=127.0.0.1 --port=8080 --rtc=true --rtc_execution_horizon=10 --rtc_max_guidance_weight=10.0 \
  --rtc_prefix_attention_schedule=EXP

# terminal 3: the two evaluations (run_eval.sh = robot_client with the settings above; add --resume=true to continue)
M=research/generated_data_diagnosis/matched_evaluation
$M/run_eval.sh manual_10     100 $M/results_manual_10
$M/run_eval.sh generated_500 100 $M/results_generated_500

# analysis (CPU)
env -u PYTHONPATH -u LD_LIBRARY_PATH .venv-v2/bin/python $M/analyze.py
# unit tests (CPU)
env -u PYTHONPATH -u LD_LIBRARY_PATH .venv-v2/bin/python -m pytest -q $M/tests
```

Outputs: `results_<model>/` (run_meta.json, trials.jsonl, steps/, eval_logs/), local recordings in
`outputs/matched_eval/eval_matched_matched_evaluation_results_<model>/` (not pushed), `comparison.json`, `figures/`.
