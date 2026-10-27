# Results summary — matched evaluation (2026-10-27)

**Question:** with positions and simulated time controlled, do 500 generated demos beat 10 manual demos?
**Answer: no.** The generated-data policy succeeds less often (34 vs 53/100, paired McNemar p = 0.005) and never better
in any position region; it only holds the can longer (33 vs 0 still held at step 330).

## 1. Identical poses? — yes

Both models ran trials 0–99 on `protocol/eval_poses_100.json` (SHA-256 `2b95ecf0…`) in the same order; requested pose
identical for every trial; actual can pose after the reset within 1.66 mm of the request (simulator ground truth),
A vs B ≤ 2.65 mm. No scene reset inside a trial; no re-homing needed.

## 2. Exactly 330 steps? — yes, 200/200

Every trial complete: 330 policy-controlled simulation steps (contiguous step counter), 11.0 s simulated, 14.7–18.2 s
wall clock. No incomplete trial in the final data. One run stop: generated trial 31's pose was refused (can 6.9 mm
off, tilted, because the arm had not returned home after trial 30); fixed (re-home on reset, inert when the arm is home)
and resumed — trial 31 then placed at 1.5 mm.

## 3. The three metrics

| | manual-10 | generated-500 | paired difference (B − A), McNemar |
|---|---|---|---|
| any-step success (original) | 53 [43–62] | 34 [25–44] | −19 [−31, −7], p = 0.005 |
| stable hand-over | 53 [43–62] | 34 [25–44] | −19 [−31, −7], p = 0.005 |
| holding at step 330 | 0 [0–4] | 33 [25–43] | +33 [+24, +42], p = 2e−10 |
| holding at step 240 / 270 / 300 | 34 / 11 / 0 | 34 / 33 / 33 | |

Wilson 95 % intervals. Success during the return home: 0 in both runs. Metric 2 equals metric 1 by the task's
construction (verified 200/200).

## 4. By position

| | manual | generated | |
|---|---|---|---|
| x < 0.203 | 21/35 | **1/35** | 20 poses only manual, 0 only generated |
| 0.203–0.27 | 20/44 | 22/44 | p = 0.77 |
| x ≥ 0.27 | 12/21 | 11/21 | |
| y < −0.01 | 3/31 | 7/31 | descriptive |
| −0.01–0.03 | 36/47 | 19/47 | p = 0.0009 |
| y ≥ 0.03 | 14/22 | 8/22 | descriptive |

Policy × position (conditional logistic, 43 discordant poses): LR p = 0.008. Near the base the generated policy closes
its right hand without lifting the can (29/34 failures there never lift it).

## 5. Coverage hypothesis

* "Generated model fails where generation never started (x < 0.203)": **supported, stronger** (1/35 vs 21/35).
* "Generated model better at 0.203–0.27": **not reproduced** (22 vs 20 of 44).
* "Manual model weak where its demos have no start (y < −0.01)": supported (3/31), small n.
* The extra generated data helped nowhere in the matched comparison; its cost near the base is not offset.

## 6. Did generated data improve stable hand-over? — no

Stable hand-overs 34 vs 53; pass failures 11 vs 5; the generated model is ~40 steps slower at every stage. It does keep
the can to the end of the 330 steps (33 vs 0) — the manual policy opens the left hand at step 233–290, where its
demonstrations end. Correction to the earlier diagnosis: the main failure is the right-hand grasp/lift (can never
raised > 2 cm in 32 manual and 50 generated trials), not the pass; the old gripper-only detector counted brief
right-hand blips as grasps.

## 7. Files

Modified: IsaacLab `scripts/tools/lerobot_sim_server.py`; lerobot_openarm `plugins/.../robot_client.py`,
`openarm_isaac.py`, `openarm_umeow.py`, `eval_recording.py`, `README.md`; new `plugins/.../matched_eval.py`.
New here: `tests/test_matched_eval.py` (24 tests, pass), `run_eval.sh`, `analyze.py`, `comparison.json`, `figures/` (9),
`results_manual_10/`, `results_generated_500/`, `smoke/`. Local recordings (not pushed): `outputs/matched_eval/`.
Details: IMPLEMENTATION_AUDIT.md, EVALUATION_PROTOCOL.md (§6: reproduce), MODEL_COMPARISON.md.

## 8. Next experiment

**NO-GO** for generated-only arms (C1 cleaned / C2 coverage-resampled from the existing generation): they cannot add
starts at x < 0.203, where the generated policy fails.
**GO** for E1/E2 (sim_10 merged with the cleaned generated data, manual up-weighted), 3 training seeds per arm,
evaluated with this matched protocol, with holding-at-330 reported next to any-step success. When regeneration is
allowed: generate with the evaluation's spawn rule (including x < 0.203).
