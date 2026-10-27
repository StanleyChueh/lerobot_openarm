# Implementation audit: how a sim evaluation episode runs, before and after the matched mode

Read from the code (no assumptions); file:line refers to the commits in `EVALUATION_PROTOCOL.md` §1.

## 1. Call order of the existing time-based evaluation (`--episode_time_s=15`), unchanged

`robot_client.py run_episodes()` with `--robot.type=openarm_isaac`:

| # | when | what happens | where |
|---|---|---|---|
| 1 | client construction | `RobotClient.__init__` → `robot.connect()` → `OpenArmIsaac._hw_connect` → sim `reset(joints=keyframe)` → `env.reset()`: **the can pose is sampled** by the `randomize_object` reset event (`reset_object_free`, Gaussian x~N(0.235,0.07), y~N(0.01,0.04), truncated, keep-out boxes); `reset_handover_stage` clears the stage machine; 3 settle steps | `openarm_isaac.py _hw_connect`; `lerobot_sim_server.py reset`; `openarm_task_modes.py reset_object_free`, `:2373` |
| 2 | connect | approach ramp to the start keyframe (already there) | `openarm_umeow.py _connect` |
| 3 | `client.start()` | gRPC handshake, policy loaded on the server | vendored `robot_client.py start` |
| 4 | each episode | `client.new_episode()`: action queue emptied, `latest_action=-1`, chunk size reset, `stub.Ready` → server `_reset_server`: observation queue, predicted timesteps, **RTC previous chunk and latencies cleared** | `robot_client.py EpisodeClient.new_episode`; `policy_server.py _reset_server` |
| 5 | episode | control loop until **15 s of wall clock** (`watch()` thread); each executed action = `robot.send_action` → one lockstep `step` → `episode_success |= success` | `robot_client.py _run_episode`; `openarm_isaac.py _hw_send_action` |
| 6 | after the loop | `return_to_initial_position` → patched `slow_return`: ramp home (**more simulation steps; `episode_success` keeps OR-ing**), 1 s settle, then `reset_scene(report=True)`: the episode's result = `episode_success` **including the return** is appended; **then the next scene reset samples the next can pose** | `openarm_umeow.py _slow_down_rollout_returns` |
| 7 | recording | `recorder.save(success=new[-1])` — the value from step 6 | `robot_client.py` |
| 8 | reset phase | `reset_time_s` of wall clock, nothing stepped | `_reset_phase` |

Consequences (all CONFIRMED by reading the code):

* Episode 1's can pose is drawn at connect; episode k≥2's at the end of episode k−1's return. Neither is seeded nor logged.
* The success label is an OR over policy steps **and** return steps.
* The simulated length is whatever 15 s of wall clock allowed (270–321 steps in the historical runs).
* The policy server's RTC history and the client queue ARE reset between episodes (step 4) — no leakage in the
  existing code. The receiver thread of an ended episode is fenced off (`EpisodeClient._stale`).
* `lerobot_sim_server.reset(object_xy=...)` existed (for replay tools) but placed the can at a hard-coded `CAN_REST_Z`
  = 0.383 m, 3 mm above the task's own resting height (`_object_rest_z` = 0.380 m), with identity orientation (= the
  task's `CAN_ROT`). Nothing in the OpenArm evaluation path called it.
* The server did not report the can pose or the stage machine; only the boolean `success`.

## 2. Call order of the matched mode (`--episode_steps=330 --poses=... --results_dir=...`)

| # | when | what happens |
|---|---|---|
| 1–3 | as above | connect (a random reset — superseded before any policy step), handshake |
| — | before the first trial | `run_meta.json` written (git SHAs/dirty state of the 3 repos, checkpoint identity, pose-file SHA-256, argv, simulator `hello`) |
| 4 | each trial i | `robot.auto_scene_reset = False` (set once); **one** `reset_scene(object_xy=poses[i])` → server `reset(object_xy, object_at_rest=True)`: `env.reset()` (random draw + stage machine cleared), then the can is written to (x_i, y_i, task rest z), upright, zero velocity; robot joints/targets restored; 3 settle steps; the reply's ground-truth can pose is checked (`check_reset`: ≤ 2 mm in x-y, ≤ 4 mm in z, upright, stage 0) — **PoseNotApplied stops the run** |
| 5 | | `client.new_episode()` (queue, timesteps, RTC history cleared, as before); `robot.step_log = []`; `client.begin_budget(330)` |
| 6 | episode | lerobot's control loop; after every executed action the step-budget client checks `robot.sim_steps − start`; at 330 it sets `shutdown_event` inside the same tick, so no 331st action is sent and no further observation is sent. Wall clock is not a limit; a watchdog marks the trial **incomplete** if no simulation step happens for 60 s. An exception marks it incomplete and stops the run. |
| 7 | right after the loop | per-step log detached; **metrics computed from the 330 policy steps only**; `success_before_return` captured |
| 8 | return | the slow return runs (simulation steps) but **does not reset the scene** (`auto_scene_reset=False`); `success_incl_return` (what the old mode would have reported) is recorded separately for comparison |
| 9 | bookkeeping | `trials.jsonl` line + `steps/trial_XXX_<attempt>.json`; the local-only recorder saves a complete trial as an episode and discards an incomplete one |
| 10 | reset phase | `reset_time_s` (1 s), then trial i+1 at step 4 |

**Re-home (added during the run, 18:4x).** Generated-500 trial 30 ended with the left hand still holding the can;
on the return home the can was dropped and blocked the right arm (`RJ6` 0.79 rad from the start pose). The next
reset kept the robot where it was, the can landed against the arm, and `check_reset` refused the pose (6.9 mm off,
tilted) — the run stopped, as designed. Fix: if the measured arm is more than `HOME_TOL_RAD` (0.1 rad) from the
start pose at the reset, the same reset also writes the start joints (zero velocity) and the safety guard's last
command is synced to it (`prepare_return`); recorded per trial as `placed.rehomed` / `home_error_rad_before`. A
robot that reached home is untouched, so the change is inert for every trial that ran before it (none of the 100
manual trials and none of generated trials 0–30 had a failed return — checked in the logs); generated trials 31–99
ran with it (`run_meta_1.json`).

A Left-arrow rerun repeats the same pose (new attempt number). `--resume=true` runs only the trials without a
complete attempt; an existing results folder is never overwritten.

## 3. Files changed

| repo | file | change |
|---|---|---|
| IsaacLab | `scripts/tools/lerobot_sim_server.py` | every observation reply gains `handover` (read-only): can pose, `_object_rest_z`, put-down margin, stage, receiver-grip counter, latches, right/left grasp by `object_grasped_by`, left `receiver_grip_confirmed` — the success term's own functions and thresholds (`success_term.params`). `reset` accepts `object_at_rest=True` (can at the task's resting height instead of `CAN_REST_Z`). No task, physics, randomization or success-term change; older clients ignore the extra key. |
| lerobot_openarm | `.../matched_eval.py` (new) | pose loading + hash, reset verification, metrics, results folder, run metadata, step-budget client |
| | `.../robot_client.py` | flags `--episode_steps`, `--poses`, `--results_dir`, `--stall_timeout_s`, `--push_to_hub`; `_run_matched`; step-mode watchdog; time-based path unchanged |
| | `.../openarm_isaac.py` | `sim_steps`, `scene_resets`, `last_reset`, optional `step_log`, `auto_scene_reset`; `reset_scene(object_xy=...)` with verification |
| | `.../openarm_umeow.py` | the slow return resets the scene only if `robot.auto_scene_reset` (default True: unchanged behaviour) |
| | `.../eval_recording.py` | `--push_to_hub=false`: local-only recording, no Hub contact, never pushed (default unchanged) |

Not changed: `policy_server.py` (its `_reset_server` already clears RTC state), `eval_metrics.py` (its per-episode
ticks/chunks give the per-trial latency/RTC numbers), the Isaac Lab task, physics, checkpoints, datasets.
