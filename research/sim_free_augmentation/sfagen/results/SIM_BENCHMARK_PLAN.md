# SIM_BENCHMARK_PLAN: optimized dataset augmentation on LeRobot benchmarks

**Status: Approved by the user 2026-11-14 (session prompt: "plan by Fable 5.2, approved by me"; authorized scope = §4 "Authorized by approving this plan"; not authorized: training beyond 5-step smoke tests, Hub or GitHub pushes, robot/CAN, method changes, deleting or overwriting existing run directories). Planned by Claude Fable 5.2, executed by Opus 5.5, audited by Codex. Supersedes `results/libero_scoping/PLAN.md`.**
Code references are relative to `research/sim_free_augmentation/sfagen/` at `49957aa`.
**[V]** = verified today (repo, installed packages, Hub, official docs). **[A]** = assumption; Opus confirms it in
the step that first uses it and records the result.

## 1. Goal and what a top venue will ask for
**Thesis.** sfagen is a minimal-input, robot-agnostic augmentation tool for LeRobot datasets: one RGB camera, joint
trajectories, the URDF and ~10 demos → a larger, replay-checked dataset with new object and interaction positions. No
depth, multi-view capture, CAD, simulator or generative model.

| Reviewer question | Evidence this plan produces | Where |
|---|---|---|
| Does it help strong policies? | Few-shot success with and without sfagen for ACT, SmolVLA, π0.5, GR00T N1.7 on public protocols | §6 |
| Is it general? | Same code on Aloha-AgileX (RoboTwin), Franka (LIBERO), OpenArm (real), + one more | §3, §5 |
| Is it real? | OpenArm real-robot results with the same four policies (you run these) | §8 S9 |
| Why not a generative or 3-D method? | Input-requirement table + oracle and reference arms | §9, §6 |
| Where does it break? | Measured failure map against simulator truth | §7 |
| Can I use it? | Setup cost log, open code, Hub datasets | §5, M4 |

**Venues [V].** ICRA 2027 closed 2026-09-15. RSS 2027: mandatory 6-page stage 1 on **2026-12-04** (method complete,
experiments may be preliminary), full paper 2027-04-16 by invitation. IROS 2027: 2027-03-01 (IEEE RAS calendar).
CoRL 2027: not announced; CoRL 2026 was 2026-05-28 **[A: late May 2027]**.
**Recommendation:** CoRL 2027 is the target. RSS stage 1 is decided on 2026-11-23 from the first report (§8, D1).

## 2. Verified starting point
| Fact | Status |
|---|---|
| Current code: worktree `artifacts/sfagen_v02_worktree`, branch `research/sfagen-v02` @ `49957aa`; main checkout is `docs/sfagen-visual-walkthrough` @ `efda1cb` without the v03 modules | [V] |
| Only OpenArm has been run; Pringles v00 and battery v00 configs differ only in `source`, `objects`, `variations` | [V] |
| Policy evidence: 10/20 → 15/20, p = 0.19, user-reported, unequal batch size | [V] |
| No training process or GPU load on this machine now (the brief said one was running) | [V] |
| Installed `lerobot 0.6.2`: env types `aloha, libero, libero_plus, metaworld, robotwin, robomme, robocasa, vlabench`; policy dirs `act, smolvla, pi05, groot` | [V] |
| No simulator package is importable; `mujoco 3.11.0`, `gymnasium 1.3.0`, `pin 3.7.0`, `coal 3.0.1` are | [V] |
| `HANDOFF.md` (2026-11-09) is stale; current state = `results/v03/RESULTS.md` + `CURRENT_TASK.md` | [V] |

## 3. Benchmarks (all LeRobot-native)
| | **RoboTwin 2.0** (first) | **LIBERO** (second) | **LIBERO-plus** (evaluation) |
|---|---|---|---|
| Robot | Aloha-AgileX, 2 arms, 14-D | Franka Panda | same as LIBERO |
| Hub dataset [V] | `lerobot/robotwin_unified`: v3.0, 30 fps, state = action = 14-D joints (`left_waist … left_gripper, right_waist … right_gripper`), 3 cameras 480×640 (`cam_high`, `cam_left_wrist`, `cam_right_wrist`), 27,500 episodes | `lerobot/libero`: v3.0, 10 fps, state 8-D (eef pos, axis-angle, gripper), action 7-D end-effector delta, 2 cameras 256×256 (`image`, `image2`), 1,693 episodes, 40 tasks | `lerobot/libero_plus`: 20 fps, 8-D / 7-D, `front` / `wrist` 256×256, 14,347 episodes |
| Fit to sfagen | **Native**: same layout as OpenArm (joint actions, 30 fps, 640×480, body + 2 wrist cameras). Config only, plus the finger model **[A]** | Needs joint positions (not in the Hub dataset), an end-effector action adapter, fps- and pixel-relative constants | Not a data source; its *objects layout*, *robot initial states* and *camera viewpoints* variants are the robustness test |
| Simulator install [V] | SAPIEN + CuRobo, in its own Docker image behind `lerobot-env-server` (`docker/Dockerfile.sim`, stage `robotwin_native`, ~20 min build) | `pip install -e ".[libero]"`, `MUJOCO_GL=egl` | clone `sylvestf/LIBERO-plus`, `pip install --no-deps -e .`, uninstall `hf-libero`; **replaces** vanilla LIBERO → its own venv; assets from `Sylvest/LIBERO-plus` |
| Stock evaluation [V] | `lerobot-eval --env.type=robotwin --env.task=<t> --eval.n_episodes=100`; Easy (`demo_clean`) and Hard (`demo_randomized`) | `lerobot-eval --env.type=libero --env.task=libero_spatial,… --eval.n_episodes=10` (400 episodes) | `lerobot-eval --env.type=libero_plus --env.task=<suite> --env.task_ids='[…]' --eval.n_episodes=1` |
| Setup check checkpoints [V] | `lerobot/smolvla_robotwin`, `SidneyXie/pi05_robotwin` | `lerobot/smolvla_libero`, `lerobot/pi05_libero_finetuned_v044` (97.5 % average in the LeRobot docs) | `lerobot/smolvla_libero_plus` |

Why RoboTwin first: sfagen can run on its Hub episodes **today with a robot config and no simulator**, so the first
problem report costs days, not weeks. A third embodiment for the generality claim is added at the freeze (§8 S8):
`gym-aloha` (ViperX, joint actions, `lerobot[aloha]`) – one day, config only.

### 3.1 Tasks (several per benchmark; exact IDs fixed in S1 / S3 before any generation)
| Category | RoboTwin dev | RoboTwin held-out | LIBERO dev | LIBERO held-out |
|---|---|---|---|---|
| Pick-and-place | `place_empty_cup` | `move_can_pot` | spatial, task 0 | spatial + object, one each |
| Bimanual hand-over | `handover_block` | `handover_mic` | – | – |
| Tight / precise | `stamp_seal` | `hanging_mug` | – | – |
| Press / repeat | `click_bell` | `press_stapler`, `shake_bottle` | goal "turn on the stove" | – |
| Articulated | `turn_switch` | `open_microwave` | goal "open the middle drawer of the cabinet" | goal, one more |
| Multi-stage | `stack_blocks_two` | `stack_bowls_two` | `libero_10`, task 0 | `libero_10`, one more |

Skip `open_laptop` (broken upstream [V]). Development tasks may be used to tune fixes; held-out tasks are run once
after the freeze, with the old and the new pipeline side by side. Press and articulated tasks are expected to be
**not expressible** by today's task config (no grasped-and-lifted object) **[A]**; that is a result, not a reason to
drop them.

## 4. Rules for the executor (binding)
- sfagen never sees simulator truth: its own calibration, segmentation, table and lag estimates, as on real data.
- Physics is allowed only for replay scoring, policy evaluation and labelled reference arms; never in sfagen data.
- Task-agnostic rules only. Per task, only `objects` and the arena / shift grid change. Report every manual step.
- Synthetic episodes: 1.000× source length, similar speed. Never retime to pass a gate.
- Pre-register each experiment and each fix (protocol + hash) before evaluating; keep negative results; never loosen a
  gate after seeing data.
- Frozen studies stay untouched. Synthetic-only datasets never contain source episodes.
- One step per fresh session; update `CURRENT_TASK.md` and `log.md` after every result; long logs go to files.
- Graph-first lookups (`graphify query`), project venv for sfagen:
  `env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ~/Stanley_ws/lerobot_openarm/.venv/bin/python -B`.

**Authorized by approving this plan:** a new worktree and branch `research/sfagen-simbench` from `49957aa` with local
commits (no push); separate venvs / the RoboTwin Docker image (project venv untouched); downloading benchmark data
(≤ 30 GB); sfagen generation and rendering on ≤ 10 source episodes × ≤ 12 tasks per benchmark; simulator replay
scoring; `lerobot-eval` of published checkpoints as a setup check; 5-step training smoke tests; the enabling code
changes E1–E8 (§5.2).
**Not authorized (ask each time):** any training beyond smoke tests, Hub pushes, GitHub pushes, robot / CAN, method
changes (§7), deleting or overwriting existing run directories.

## 5. Cross-embodiment audit of the current code [V]
### 5.1 Assumptions found (every module in `sfagen/` read in full)
| # | Where | Assumption | Hits | Class | Change |
|---|---|---|---|---|---|
| A1 | `plan.py:31-33,94-97`; `checks.py:66-68,113`; `robot.py:150-159`; `data.py:37` | `action` = joint positions, same layout as `observation.state` | LIBERO | **code** | action adapter `joint_position` \| `ee_delta` \| `ee_absolute`, plus `joint_state_key`; exact, because our edits are pure translations |
| A2 | `plan.py:137,139,152,162,174-175`; `recon.py:140,157,207,321`; `workspace_plan.py:123-148`; `render.py:19`; `segment.py:26`; `checks.py:36-37`; `handover_timing.py:106`; `plan.py:81,31` | time constants in frames (15, 10, 30, 45, σ = 3, lead ≤ 6) tuned at 30 fps | LIBERO (20 fps) | **code** | constants in seconds × fps, rounding to today's values at 30 fps |
| A3 | `recon.py:183,213-216,282`; `render.py:108-115`; `distortion.py:16`; `checks.py:213,216`; `segment.py:41`; `calibrate.py:28` | pixel constants (160 px ROI, 45 / 130 px seeds, 21 px erode, 2000 px) for 640×480 | LIBERO at 256 px | **auto** | scale by width / 640; until then export LIBERO at 640×480 |
| A4 | `robot.py:101,110` | every finger joint = `finger_open × openness`, closed = 0, same sign | Piper, ViperX **[A]** | **code** | per-finger open / closed values |
| A5 | `build.py:31-32` | wrist features of shape `(n_wrist,)`; `(0,)` when there is no wrist camera – suspected export failure **[A]** | gym-aloha | **code** | omit wrist features when there is none |
| A6 | `data.py:29` | one task string per dataset (`tasks.parquet` row 0); `robotwin_unified` has 23,559 task strings | RoboTwin, LIBERO | **code** | task per episode (until then: split a 10-episode subset with `lerobot-edit-dataset`) |
| A7 | `recon.py:180,202` | every object has a grasp-close event; none → `IndexError` | press, articulated | **code** | clear error now; real support only via §7 |
| A8 | `generate.py:30`; `plan.py:164`; `handover_timing.py:115` | interaction families need ≥ 2 arms; one arm gets object-position only, place targets never move | LIBERO | by design | report (T8) |
| A9 | configs `sam2_exclude_links`, `skip_collision_links`, `table_skip_links` (`segment.py:16`, `checks.py:44,95`) | OpenArm link names, substring match | all | **config** → auto | derive from the source demos |
| A10 | `robot.py:51-55`; `context.py:14,17` | one URDF with all arms; robot block inlined in every task config | all | **config** | `robot_config:` include |
| A11 | `camera.py:14-34`; `calibrate.py:18,44-52` | pinhole static camera, world z up, guess within ±15 cm / 15°, near-static frames (`qd < 0.05`) exist | scripted demos **[A]** | **config** | see problem P1 |
| A12 | `recon.py:300-301,334,222`; `plan.py:137` | object within ±8 / ±4 cm of the TCP; hand-over within 0.25 m; box height 2–8 cm; onset 0.02 m/s | small grippers | **auto** | keep as defaults; scale later if measured |
| A13 | `robot.py:157`; `objects.*.grasped_by / grasp_index` | grasp = command crossing a fixed openness | all | **config** → auto | hysteresis; nearest TCP at disappearance |
| A14 | `target_plan.py:16` | grid targets need source positions frozen by a hand-run script in untracked `artifacts/` | all | **code** (tool) | `sfagen grid` writes them after reconstruction |
| A15 | `context.py:48`; `cli.py:72`; `recon.py:185`; `checks.py:214-227` | one horizontal table; "on table" = same plate colour | textured / raised surfaces | by design | see problem P7 |

Already generic [V]: arm count and names, state indices, TCP frame, camera names, SAM2 prompts (from URDF hulls),
table height, lag, thresholds as ratios of the source demos.

### 5.2 Enabling changes E1–E8 (pre-authorized; they must not change OpenArm output)
E1 = A1, E2 = A2, E3 = A4, E4 = A5, E5 = A6, E6 = A7 (error message), E7 = A10, E8 = A14.
**Guard after every change:** `unittest discover -s tests` (56 tests) passes, and one Pringles v03r2 source
regenerates bit-identical candidate files (`candidate_sha256` in `generation_manifest.json`) and body frames.

## 6. Measurements
**Pipeline v0** = `49957aa` + E1–E8, one recipe for every task: the `reconstruction`, `render`, `checks`, `plan`
blocks of `configs/openarm_pringles_v03r2_loose.yaml` unchanged; object targets on a 3 cm grid, sources ≤ 12 cm, F1W
wide blend, `time_scales: [1.0]`, `fixed_duration`, `speed_similarity: [0.8, 1.25]`; interaction shifts ±3 / ±5 cm for
≥ 2 arms; ≤ 2 lowest-distortion candidates per target; copied wrists; no distortion cap (the 2.289 / 4.578 values are
OpenArm rig numbers).

| | What | Needs | Decision rule (fixed now) |
|---|---|---|---|
| **M0** Runs at all | calibration held-out IoU, overlays, yield with first failed gate, `verify`, side-by-side videos | data only | any stage failing → Problem Report |
| **M1** Validity | replay generated actions in the simulator from the true start state + shift `d`; success per phase, family and shift bin; controls: source replay (ceiling), source actions on the shifted scene (floor); also replay gate-**rejected** candidates → gate precision / recall | simulator | P(success \| pass) < 0.8 × ceiling on ≥ 2 expressible dev tasks → validity is the first fix |
| **M2** Fidelity | sfagen frame vs **oracle** (kinematic simulator render of the same joints, object at `p_src(t) + d` before the grasp and `+ w(t)·d` while held): PSNR / SSIM per region × phase × shift; object-mask IoU; ghost rate; perception error (camera px, object cm, table mm, lag); copied wrist vs oracle wrist | simulator | ρ(distortion score, oracle error) < 0.5 → score not used for gating; object error > 1 cm or reprojection > 5 px → perception first |
| **M3** Policy | §6.1 | training (you) | §6.1 |
| **M4** Ease of use | hours to first verified export; config lines changed; manual steps; knobs changed outside `objects` / arena (target 0); failures needing code, with `file:line`; GPU time (`profile.jsonl`) | log | > 1 day or > 60 lines per robot + task → Problem Report |

Truth firewall for M1 / M2: the exporter writes `public/` (dataset, URDF, intrinsics, an extrinsic guess = truth + a
seeded offset ≤ 10 cm / 10°, nominal object sizes) and `truth/` (states, poses, masks, success). `truth/` is
`chmod 000` while sfagen runs, and a test fails if any path in a `generation_manifest.json` points into it.

### 6.1 Policy study (Opus prepares; you train)
| Item | Setting |
|---|---|
| Policies | ACT (per task), SmolVLA, π0.5, GR00T N1.7 (multi-task per benchmark), all from their **base** weights, not from benchmark-tuned checkpoints |
| Data conditions | **R10** (10 demos / task); **R10+S** (sfagen); **R10+2D** (standard image shift / crop / colour augmentation); **R50** (the benchmark's 50 demos: "how many real demos is sfagen worth?"). Reference arms on SmolVLA only: **R10+O** (oracle images), **R10+M** (MimicGen-style: same candidates executed with physics, successes only) |
| Evaluation | RoboTwin: stock Easy protocol, 100 episodes / task (Hard reported as secondary). LIBERO: stock 4 suites × 10; LIBERO-plus: *objects layout* and *robot initial states* variants of the same tasks (from the fork's `task_classification.json`), 1 episode / variant |
| Equal exposure | same steps and batch size in every condition (the v00 comparison used batch 16 vs 8) |
| Order | Tier 1: SmolVLA + ACT on both benchmarks. Tier 2: π0.5 + GR00T on the same data. Tier 3: K ∈ {5, 20} curve for SmolVLA |
| 24 GB notes [V, `HANDOFF.md` §5] | GR00T trains only with `--policy.lora_rank=16 --policy.tune_diffusion_model=false`; π0.5 full fine-tune runs out of memory → LoRA / frozen backbone, settings fixed by the 5-step smoke test |
| Primary outcome | pooled success, (R10+S) − R10, paired by task and seed, 95 % CI; secondary: gap closed (S − R10) / (O − R10); success vs distance from the evaluation start to the nearest training position |
| Rules | O − R10 < 10 points → position augmentation does not help on that benchmark (report; do not chase). O − R10 ≥ 10 and S − R10 < ⅓ of it → rendering is the bottleneck. M − O ≥ 10 → invalid trajectories are |

Sfagen output is exported back into each benchmark's **native schema** before training (LIBERO: 8-D state by FK,
7-D action, 256×256 `image` / `image2`), so stock `lerobot-train` / `lerobot-eval` commands work unchanged.

## 7. Problem reports and the fix loop
After every step Opus writes `simbench/results/reports/RNN_<topic>.md` and stops for your decision when a rule fires.

**Report template:** symptom (one line) · evidence (images, numbers, denominators) · scope (tasks / robots / shift
sizes) · cause, with the `file:line` · 2–3 options, each with generality, cost and risk · recommendation · what was
not checked.

**Anticipated problems and the options Opus should weigh** (hypotheses **[A]** until measured):
| # | Problem | Likely on | Options (general ones first) |
|---|---|---|---|
| P1 | Camera fit poor: arms enter from the image edge, few near-static frames in scripted demos, small robot in view | RoboTwin head camera, LIBERO agentview | (a) fit on SAM2 arm masks instead of frame-minus-median; (b) fps-relative static-frame rule; (c) accept dataset-provided extrinsics when they exist (`calibration: given`) |
| P2 | Low resolution (256 px) | LIBERO Hub data | (a) pixel constants scaled by width (A3); (b) sim-only: export at 640×480 |
| P3 | Object segmentation fails in clutter, on textured tables or for objects that are pushed / pressed, not lifted | LIBERO, RoboTwin Hard | (a) SAM2 object mask prompted at the projected TCP at the grasp; (b) contact-anchored objects (also fixes A7) |
| P4 | Wrong support plane (raised surfaces, varied table height) | LIBERO cabinets, RoboTwin Hard | per-object support height from the grasp TCP |
| P5 | Arm fragments when re-posed near the camera | RoboTwin | renderer variants judged against the oracle (mask-driven warping, erode scaled to link size) |
| P6 | Copied wrist views show the wrong background | both | measure vs oracle wrist; re-render every moved wrist by one fixed rule only if M3 says ≥ 10 points |
| P7 | Yield low from the plate-colour "on table" rule on textured tables | LIBERO wood tables | geometric rule: table plane ∩ view ∩ support region |
| P8 | Yield low from speed gates because scripted demos have no idle time | RoboTwin | none that retimes; report coverage honestly, try longer event-free blends only |
| P9 | Ghost / split object | all | mask growth, object atlas; judged by the M2 ghost rate |
| P10 | Rotation never varied | all | yaw about the object axis, after an M2 probe |

**Two kinds of change.** Enabling changes (E1–E8) keep OpenArm output bit-identical: Opus makes them and logs them.
**Method changes** alter outputs: literature check (logged in `results/LITERATURE.md`) → report with options → your
choice → pre-registered hypothesis, metric and threshold → implement → evaluate on development tasks only → accept or
reject (rejected fixes stay in the report) → regression on OpenArm and every earlier benchmark task. At most 4 method
changes before the freeze. No per-task components.

**Failure classes** used in every report: T1 not expressible · T2 tight contact · T3 repeated / counted · T4 rotation ·
T5 near-camera re-posing · T6 wrist views · T7 ghost object · T8 one arm vs two · T9 perception · T10 gate validity.

## 8. Steps for Opus 5.5 (one per session; dates assume the CoRL target)
| Step | Dates | Do | Output | Stop and ask if |
|---|---|---|---|---|
| **S0** Workspace | Nov 16–17 | worktree `~/Stanley_ws/lerobot_openarm_simbench`, branch `research/sfagen-simbench`; `research/sim_free_augmentation/simbench/{configs,tools,results/reports}`; guard script (§5.2) from `artifacts/sfagen_v03r2_run`; add today's new papers to `results/LITERATURE.md` | guard hashes; 56 / 56 tests | guard cannot be reproduced |
| **S1** RoboTwin, data only | Nov 18–22 | fetch `meta/` of `lerobot/robotwin_unified`, map episodes to task names **[A]**, split 10 episodes for `handover_block`, `place_empty_cup`, `stack_blocks_two` (`lerobot-edit-dataset`); clone RoboTwin for the `aloha-agilex` URDF and camera config **[A]**; write `_aloha_agilex_robot.yaml`; E3, E7 if needed; run `calibrate --search → table → generate → render → build → verify → video`; M0 + M4 | **R01** with overlays, yield table, contact sheets, videos | calibration not adopted; yield < 10 % ; any stage needs a method change |
| **D1** | Nov 23 | you decide: RSS stage 1 (write Nov 24 – Dec 4 from R01 + the real 10/20 → 15/20 result) or CoRL only | – | – |
| **S2** RoboTwin simulator | Nov 24–30 | build the Docker image; setup check: `lerobot-eval` of `lerobot/smolvla_robotwin` on 2 tasks, 20 episodes; exporter with seeds and truth; firewall test; M1 + M2 on the 3 pilot tasks, then the other 3 dev tasks | **R02** validity + fidelity + first failure map | setup check far below the published level; M1 / M2 rule fires |
| **S3** LIBERO | Dec 1–13 | venv with `lerobot[libero]`; setup check with `lerobot/smolvla_libero` on `libero_spatial`; export 10 human demos / task by replaying the original demonstrations from their stored states, recording joint positions, 640×480 **[A]**; E1, E2; Franka robot config; pipeline + M0, M1, M2, M4 on 5 dev tasks; native-schema exporter | **R03** | demos cannot be replayed; adapter error > 1 mm in replay |
| **S4** LIBERO-plus | Dec 14–18 | second venv; assets; smoke test with `lerobot/smolvla_libero_plus`; select *objects layout* / *robot initial states* variant IDs for the dev tasks; record fork revision | variant list + commands | install breaks vanilla LIBERO venv |
| **S5** Training pack, tier 1 | Dec 19–23 | build R10, R10+S, R10+2D, R50 (+ O, M) datasets locally in native schema; exact `lerobot-train` and `lerobot-eval` commands; 5-step smoke test per policy; GPU-hour estimate from the smoke tests | `simbench/TRAINING_PACK.md` | a policy cannot train on 24 GB with any documented setting |
| **G2** | by Jan 17 | you train tier 1; Opus aggregates (`simbench/tools/aggregate.py`), applies the §6.1 rules | **R04** policy results | O − R10 < 10 on both benchmarks → strategy review with you |
| **S6** Fix loop | Dec 24 – Feb 21 | ranked problems → method changes as in §7; rebuild R10+S after each accepted fix; tier-2 pack | **R05…** one per fix | every method change (by rule) |
| **S7** Freeze + audit | Feb 22–28 | SHA + protocol hash; Codex audit of all reports | freeze record | audit finds a blocking issue |
| **S8** Held-out, once | Mar 1–21 | old and new pipeline on held-out tasks of both benchmarks; third embodiment: `gym-aloha` TransferCube + Insertion, config only, Codex as second operator from the README | **R-final** tables | – (results stand) |
| **S9** Real robot (yours) | Mar – Apr | OpenArm with the frozen pipeline and the four policies; not planned here | – | – |
| **S10** Paper | Apr – May | figures from the reports; limitations from the failure map | draft | – |

**Compute (estimates [A], re-measured in S1 and S5):** sfagen ≈ 1–3 GPU-h per task (`HANDOFF.md` §6) → ~80 GPU-h
for 24 tasks × ~1.5 passes; simulator scoring is CPU-bound. Training (yours): tier 1 ≈ 20 SmolVLA / ACT-set runs,
tier 2 ≈ 16 π0.5 / GR00T runs; hours per run come from the S5 smoke tests.

## 9. Positioning (papers read today via their arXiv HTML)
| Work | Needs beyond one RGB camera + URDF | Robots shown | Post-hoc on an existing teleop dataset |
|---|---|---|---|
| DemoGen 2502.16932 | depth; point-cloud policy | Franka (gripper, Allegro), Galaxea R1 | yes, 3-D policies only |
| R2RGen 2510.08547 | depth; scanned object templates | UR5 mobile, dual Piper | yes, 3-D policies only |
| Real2Edit2Real 2512.19402 | 3 cameras; video model tuned ~60 h on 8 H100 | Agibot G1 | yes, with that model |
| R2RDreamer 2606.17040 | depth; tuned image-to-video model | one arm | yes |
| RoboSplat 2504.13175 | dense multi-view capture | FR3 → UR5e | no |
| Real2Render2Real 2505.09601 | object scan, human video, IsaacLab renderer | YuMi | no |
| MimicGen 2310.17596 / DexMimicGen 2410.24185 | simulator + digital twin | Panda, Sawyer, IIWA, UR5e; GR-1 | no |
| Real2Sem2Real 2610.28880 (new) | human video; simulator-rendered data | xArm7, Franka | no |

Only RoboSplat, Real2Render2Real and MimicGen claim or show cross-embodiment use; R2RDreamer and Real2Edit2Real do not.
None reports results with a public LeRobot benchmark protocol, and none of the six real-to-real pages links code.
**Our angle:** minimal input; post-hoc on LeRobot datasets with 2-D VLA policies; the same code on four embodiments;
gains on public protocols; a truth-scored failure map; replay gates validated against physics.
**Baselines we can actually run:** R10+2D, the oracle and MimicGen-style references, R50. Competitors are compared on
input requirements. Not confirmed today: the venue labels in `NOVELTY.md`, and whether any competitor released code
since (Opus checks in S0).

## 10. Risks
| Risk | Mitigation |
|---|---|
| Position augmentation gives little on stock LIBERO starts (small reset range) | LIBERO-plus layout variants and RoboTwin Easy carry the claim; rule in §6.1 stops chasing |
| 12 cm cap covers little of RoboTwin's randomisation range | S2 reports the fraction of evaluation starts within reach of 10 sources before any training |
| RoboTwin Hub episodes carry no seed, so they cannot be replayed **[A]** | S1 uses them for M0 only; S2 regenerates sources with seeds |
| 10 demos give ~0 % for some policies | R50 and the reference arms show the ceiling; K-curve in tier 3 |
| v0 is "current + enabling changes" | bit-identical OpenArm guard |
| Simulator images are cleaner than real | same pose-transfer benchmark (`sfagen bench`) in sim and real; real-robot results in S9 |
| Held-out contamination | held-out tasks only after the SHA freeze |
| Training turnaround is yours | tier order; Opus continues the fix loop meanwhile |

## 11. Decisions for you
1. **D1 (Nov 23):** RSS 2027 stage 1 as well, or CoRL only?
2. Confirm the oracle and MimicGen-style arms as reference-only exceptions to "no physics-engine generated
   demonstrations".
3. Training: you run every full training (assumed). Should Opus also run tier 1 on this machine when the GPUs are idle?
4. Third embodiment at the freeze: `gym-aloha` (recommended) or Meta-World (Sawyer)?
5. Scope of authorization in §4: strike anything you do not want pre-approved.

## Sources (accessed 2026-11-14)
LeRobot docs: [LIBERO](https://huggingface.co/docs/lerobot/libero) ·
[LIBERO-plus](https://huggingface.co/docs/lerobot/libero_plus) · [RoboTwin 2.0](https://huggingface.co/docs/lerobot/robotwin).
Venues: [RSS 2027 call](https://roboticsconference.org/information/cfp/) · [ICRA 2027](https://2027.ieee-icra.org/contribute/) ·
[IEEE RAS dates](https://www.ieee-ras.org/events/category/deadlines/) · [CoRL](https://www.corl.org/).
Papers: arXiv [2502.16932](https://arxiv.org/abs/2502.16932), [2510.08547](https://arxiv.org/abs/2510.08547),
[2512.19402](https://arxiv.org/abs/2512.19402), [2606.17040](https://arxiv.org/abs/2606.17040),
[2504.13175](https://arxiv.org/abs/2504.13175), [2505.09601](https://arxiv.org/abs/2505.09601),
[2310.17596](https://arxiv.org/abs/2310.17596), [2410.24185](https://arxiv.org/abs/2410.24185),
[2610.28880](https://arxiv.org/abs/2610.28880), [2606.19586](https://arxiv.org/abs/2606.19586),
[2510.13626](https://arxiv.org/abs/2510.13626). Hub: `lerobot/robotwin_unified`, `lerobot/libero`,
`lerobot/libero_plus`, `lerobot/smolvla_libero`, `lerobot/pi05_libero_finetuned_v044`, `lerobot/smolvla_libero_plus`,
`lerobot/smolvla_robotwin`, `SidneyXie/pi05_robotwin`.
