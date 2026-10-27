# Next-experiment protocol (prepared, NOT executed)

Nothing here has been run: no training, no Isaac rollouts, no regeneration, no pushes.

## 0. Evaluation protocol (fixes ROOT_CAUSE #3/#4; needed before any arm is compared)

| requirement | how | status |
|---|---|---|
| same 100 can poses, same order, for every policy | `protocol/eval_poses_100.json` (seed 2027, the evaluation's own spawn rule, stratified over 3×3 x/y regions) | file ready |
| the sim places the can at the listed pose | sim server already accepts `reset(object_xy=…)` (`lerobot_sim_server.py:219`); `OpenArmIsaac.reset_scene` must pass it (≈10 lines: a `--robot.reset_poses=<json>` option) | change needed |
| fixed SIMULATED budget, not wall-clock | count executed actions: `robot_client.py --episode_steps=330` (11.0 s sim; ≥ the longest demo, 311) instead of `--episode_time_s` | change needed |
| success excluding the return phase | snapshot `episode_success` before `return_to_initial_position` | change needed |
| per-step logging | write the server's per-step `success` and the can pose (`objects`, via the task's `report_objects`) into `meta/openarm_eval.json`: first-success step, success at steps 240/270/300/330, can pose at start and end | change needed |
| metrics | (1) original any-step; (2) **stable**: pass completed and the left hand still holds 30 steps later and at step 330; (3) stage reached (gripper detector, validated); (4) per-region success, Wilson 95 % CI | analysis ready (`scripts/eval_*.py`) |
| tests | paired by pose: McNemar per arm pair; logistic success ~ arm × x + arm × y; report per region | ready |
| reproducibility | checkpoint repo@SHA, dataset SHAs, server/client flags, the pose file's SHA-256, git SHAs of the three repos, in every result folder | |

Keep everything else identical to the existing runs: `Isaac-PickUp-RedCube-OpenArm-IK-Abs-v1 --task_mode handover
--domain_randomization none`; policy server `--rtc=true --rtc_execution_horizon=10 --rtc_max_guidance_weight=10.0
--rtc_prefix_attention_schedule=EXP`; client `--actions_per_chunk=50 --chunk_size_threshold=0.5
--aggregate_fn_name=latest_only --fps=30`; `--record_repo_id=<user>/eval_<…>` for every run.

## 1. Arms

| arm | data | training | why | priority |
|---|---|---|---|---|
| **A0 / B0 / B400** | existing checkpoints: manual `…sim_smolvla_10_v00@9ce7ce48`, generated `…generated_500_smolvla_v00@733c6856`, `…generated_400_smolvla_v00@60661c0d` | none | replicate the policy × position interaction on identical poses; first look at stable success and the 330-step horizon | **1 (GPU eval only, ~30 min each)** |
| **C1** clean generated | generated_500 minus the 108 `check_mimic_demos.py` flags (incl. #91) = 392 eps | recipe of B | effect of the can-jolt / windup artefacts | 2 |
| **C2** coverage-matched | C1 importance-resampled to the evaluation's start density (drop the 28 % inside the evaluation keep-out) | recipe of B | does spending data where the evaluation is matter (within the generated range)? | 3 |
| D corrected actions | — | — | **not justified: no action-label defect (ACTION_SEMANTICS_AUDIT)** | dropped |
| **E1** manual + clean generated | merge sim_10 + C1 (`lerobot-edit-dataset --operation.type=merge`; both `openarm_umeow`, same features) | recipe of B | restores the near-base sources and the evaluation look | 2 |
| **E2** E1 with manual up-weighted | sim_10 episodes repeated so they are ~25 % of frames | recipe of B | 10 vs 392 episodes makes the manual data 2.4 % of frames in E1 | 3 |
| **F1** budget | generated_500 at 20k × 8 (the manual budget) | | is 9.5 passes "under-trained"? | 3 |
| **F2** budget | sim_10 at 40k × 32 (the generated budget) | | budget-matched manual baseline | 3 |
| **F3** | generated run's step-20k checkpoint, **if it still exists on the training machine** | none | free budget point | 2 (if available) |

Seeds: 3 training seeds for every trained arm (seed 1000 was the only one used so far). 100 poses × 3 seeds per
arm. Same lerobot fork commit for every run (`f10cbf97` or record the one used).

Data-editing caveat: generated_500's body_cam video for episodes 87 and 224 is one frame short; stock
`lerobot-edit-dataset delete_episodes` fails on it (memory note `generated-500-subsets`). Patch or drop those two
episodes when building C/E.

## 2. Predictions (decide what each outcome means before running)

| if … | then |
|---|---|
| A0 ≫ B0 at x < 0.203 and B0 > A0 at 0.203–0.27 on identical poses | coverage (cause #1) replicated without position confounding |
| E1/E2 ≥ max(A0, B0) in every region | the generated data is useful where it covers; the near-base gap was the loss |
| C1 ≈ B0 | the 22 % flagged episodes are not what limits it |
| no arm lowers the "left lost it within 1 s" failure (~50 %) | the pass is the ceiling (cause #2); generation must vary the pass, not the approach |
| stable success (2) ≪ any-step success (1) for A0 but not B0 at 330 steps | horizon artefact (cause #4) confirmed under controlled timing |
| F1 ≈ B0 | under-training is not the explanation |

## 3. Later, once regeneration is allowed (not now)

Generate with the evaluation's spawn rule (or a coverage-targeted one), a fixed fraction of episodes with
`--domain_randomization none`, and the hand-over segment varied (it is replayed untransformed today). Those are the
three levers this diagnosis implicates; each is a separately testable arm.
