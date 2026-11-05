# Baseline generator audit: what the existing Pringles hand-over generator varies, and what it replays

Date 2026-11-05. CPU only, read-only.

* **Repositories:**
  * lerobot_openarm at `2779069` (branch `research/contact-aware-augmentation`);
  * IsaacLab at `4aa5e026` (`openarm-dev`, tree clean).
* **What was run:** no simulation, GPU, network, hardware or training.
* **Evidence:** code citations below, plus `outputs/dry_run/dry_run_summary.json` and `candidates.jsonl` (`scripts/run_dry_run.py`).
* **Relation to Gate 1R:** that line is closed and frozen. This audit does not use or modify the image-based verifier
  (test `test_gate1r_verifier_frozen`).

Labels: **CONFIRMED** (code read and/or measured), **MEASURED HERE** (new numbers from this audit), **UNKNOWN**.

## 1. Summary

| question | answer | status |
|---|---|---|
| What does the generator change? | The can's start xy, and the right arm's approach-and-grasp, by one translation `T = can_new − can_src` (rotation discarded). Timing changes through inserted bridges (right +44 steps, left +36 steps at 30 Hz) | CONFIRMED |
| What does it replay? | Everything after the right grasp (presentation, pass point, release, withdraw) and the **entire left arm** (wait, approach, grasp, carry), in world coordinates, untransformed. The release order and overlap are imposed by a runtime clamp | CONFIRMED |
| Do the sources themselves contain hand-over diversity? | Very little. Across the 10 sim demos the can starts spread 6.4 cm (x std), but the pass point spreads only 1.0 / 0.8 / 2.3 cm (x/y/z std). corr(start x, pass x) = 0.14: the operator hands over at nearly one place, wherever the can started | MEASURED HERE |
| Consequence | Geometry-only generation cannot produce new hand-overs. It copies one of ≈ 10 nearly identical passes. A plans for the 24 pilot poses: receiver-path novelty median **0.13** (1 = typical source-to-source distance), **0/24** outside the source hand-over envelope. Earlier audit: 473/500 near-duplicates, 5.6 effective templates | CONFIRMED + MEASURED HERE |
| Does the simulator expose what is needed to judge a hand-over? | Object pose, both grasp frames, joints, joint targets and the latched hand-over stage are recorded. Contact, force, arm–arm collision and the per-step can pose for the sim_10 sources are not available (§4) | CONFIRMED, with gaps |

## 2. Data actually available on this machine

| item | content | use here |
|---|---|---|
| `ethanCSL/openarm_pringles_lerobot_sim_10_v00` @ `b6d39dae` (LeRobot cache) | The 10 manual sim demos that the failed `generated_500` set was made from. 30 Hz, current env (`-v1`, real look). 16-D joint state and action. **No object pose** | Pilot sources. Kinematics via FK |
| `research/generated_data_diagnosis/data/` | Simulator truth extracted earlier from the server HDF5: initial can xy (`sim20_start_poses.json`), per-episode events and right-grasp offset (`episode_events.csv`) | Joined to the sim_10 sources |
| `IsaacLab/logs/demos/pickup_pringles_V9_annotated.hdf5` (10 demos), `pickup_pringles_stanley_annotated.hdf5` (30 demos) | Full per-step simulator truth: can pose, both EEF poses, joints, actions, Mimic subtask signals | Metric-code validation and attachment geometry only |
| `IsaacLab/logs/demos/openarm_pringles_lerobot_sim_10_v00{,_annotated,_generated}.hdf5` | Server-recorded originals with per-step can pose (cited by DATA_GENERATION_AUDIT §1) | **NOT on this machine.** Blocker B1 (§6) |

**The local GT files are older than the current code.** They are 20 Hz `-v0` recordings from 2026-09-08 and 09-16. The
following changes came after them:

* the 30 Hz switch (`356c21e8`, 10-02);
* the foam-pad fingertips (`8d9f4d07` / `b9d17d95`, 10-19/21);
* the identity-rotation `get_object_poses` override (`98bdbe2d`, 10-21).

The code warns that sources annotated before that override must be re-annotated. So these files are **not** used as
generation sources. They serve only as known-good simulator recordings for validating metric code, and for the typical
can-in-hand geometry.

## 3. How the current generator works (IsaacLab `4aa5e026`)

### 3.1 Structure (`openarm_task_modes._handover_subtask_configs`, `openarm_pickup_ik_abs_mimic_env*.py`)

| element | setting | effect |
|---|---|---|
| right arm (giver), subtask 0 | `object_ref="can"`, `nearest_neighbor_object`, `nn_k=1`, term signal `grasp_right` | the **only** transformed segment: source approach-and-grasp re-anchored by `T_now · T_src⁻¹` on the can pose |
| right arm, subtask 1 | `object_ref=None` | present, release, withdraw replayed **untransformed**, at the source's pass point |
| left arm (receiver) | one subtask, `object_ref=None`, `interp = LEFT_INTERP` | wait, approach, grasp, carry replayed **untransformed** for the whole episode |
| task constraints | none | the arms stay in step only because both replay one source |
| `get_object_poses` | rotation forced to identity | `T` is a pure translation |
| `generation_select_src_per_subtask` | False | one source per episode, chosen at the first subtask of the first arm (the right) |
| bridges | `SEAM_INTERP = scale_steps(15)` = 22 steps, smoothstep easing; `LEFT_INTERP = 2·22 − 8` = 36 | right delayed 44 steps (entry bridge + grasp seam), left 36: the right runs 8 steps later relative to the left than in the source |
| `waypoint_smoothing_window` | `scale_odd(5)` = 7 | moving average of positions, endpoints pinned |
| `action_noise` | 0 | no per-step noise |
| `_hold_giving_hand_until_taken` | clamps the right jaws shut once `presented`, until the latched `handover` + `scale_steps(5)` = 8 steps | release order and overlap are a runtime rule, not data. The earlier audit measured 10–17-frame overlaps vs 1–10 in the sources |
| 16-D joint sources | `action_to_target_eef_pose` returns the measured EEF pose | source "targets" are the reached poses, ≈ FK of the recorded state (§3.4) |
| success | `generation_success_at_final_state=True`; hand-over stage machine with a 1 s firm receiver grip | judged at the end of the waypoints |
| spawn during generation | uniform x ∈ [0.2029, 0.4217], \|y\| ≤ 0.0769 (`_HANDOVER_X/Y_RANGE` = source region ± margin) | differs from the evaluation's truncated Gaussian (earlier audit: 34 % of evaluation spawns uncovered) |

### 3.2 What is varied

* **The can start:** sampled per episode.
* **The right approach:** shifted by `|T|`. The code docstring reports a median of 4.2 cm and up to 9.5 cm at nn_k=1.
* **The 22-step seam at the grasp:** drags the just-grasped can back onto the source trajectory, so the pass is the
  source's.
* **Global timing:** shifted by the bridges.

### 3.3 What is replayed

In world coordinates, unchanged:
* the pass point;
* the receiver approach path, grasp pose and carry;
* the giver's presentation and withdrawal.

The release overlap is the clamp's 8 steps plus the 8-step relative drift. The receiver therefore always closes where
that source's receiver closed, whatever the can's start. **A new can position does not produce a new hand-over.**

**Measured here** (A plans re-implemented offline, 24 pilot poses; `test_A_replays_the_handover`,
`test_A_translates_only_the_right_approach`):
* receiver-path nearest-source RMS / source-to-source median = **0.13** (path nn scale 2.0 cm);
* left-arm targets equal the smoothed source to 1e-9;
* the right target after the seam equals the source within 1 cm;
* 0/24 A plans have a pass ≥ 3 cm from every source pass, or a receiver grasp outside the source range.

**Sources are themselves near-identical at the pass** (FK of the recorded joints, nominal attachment §3.5):

| | x | y | z |
|---|---|---|---|
| can start, std (range) | 6.4 cm (0.134–0.332) | 1.6 cm (−0.009–0.041) | — |
| pass point, std (range) | **1.0 cm** (0.168–0.197) | **0.8 cm** (−0.011–0.012) | 2.3 cm (0.509–0.580) |
| receiver − giver grasp along the can | 6.1–12.8 cm | relative azimuth 176–233° | |

The prior audit's conclusion ("500 episodes ≈ 5.6 trajectory templates × a can offset") follows from this. Varying
which source is used cannot vary the hand-over much, because the 10 sources hand over at almost the same place, in
almost the same way.

### 3.4 Kinematic consistency (measured here)

* **FK:** URDF `openarm_*_hand` + 2.4 mm along z reproduces the recorded sim grasp frames (FrameTransformer on
  `*_ee_tcp` + 0.1025 m). Over the 40 GT demos, error < 0.001 mm and rotation identical (`test_fk_matches_simulator_grasp_frames`).
* **Joint order of the IsaacLab recordings:** `L1,R1,…,L7,R7,Lf1,Lf2,Rf1,Rf2`.
* **Action order:** `[0:7]` left arm, `[7]` left finger (m), `[8:15]` right arm, `[15]` right finger.
* **LeRobot conventions:**
  * interleaved `RJ1,LJ1,…,RJ8,LJ8`;
  * sim joint = motor rad (calibration sign +1, offset 0);
  * gripper raw: right open −1.298, left open 1.2121, closed 0, mapped linearly to the 0–0.044 m finger;
  * action leads state by 2 frames (ACTION_SEMANTICS_AUDIT).

### 3.5 Hand-over geometry in simulator truth (40 local GT demos, measured here)

* **Giver attachment** (can pose in the right grasp frame, 15 steps after `grasp_right`):
  * position (4.9, −0.3, 4.7) cm, std (1.35, 0.29, 0.68) cm;
  * the can axis lies along the grasp frame x (0.985);
  * settle-to-take drift: median 5.3 / 7.2 mm, max 14 / 21 mm.
* **Receiver attachment** after the take: (−5.0, −0.2, 4.6) cm, std (1.3, 0.2, 0.8) cm.
* **Placement on the can:** the receiver grasps 5–6 cm above the can centre, the giver 4–5 cm below.
* **Jaw-capture prior:** the can centre between the open jaws, |y| ≤ 1.5 cm, depth 0–7.5 cm, on the body. It holds
  for 39/40 giver and 40/40 receiver attachments in GT.
* **The plan-time prediction is not reliable for the sim_10 sources:**
  * with the nominal attachment, the receiver's own source grasp satisfies the capture test in only **6/10** of the
    (GT-successful) sources;
  * the sim_10 copies lack the can pose, and the attachment varies by 1.35 cm across demos.
  * So no plan-time contact prediction can be trusted for A or B (§3.6, §4).

### 3.6 Offline checks of the A plans (24 pilot poses)

Kinematics: IK 24/24; clearance 24/24; phase order 24/24; jump > 0.25 rad 0/24. But **11/24** exceed the real-robot step
cap of 2.5 rad/s. This is inherited: the manual demos themselves peak at 1.86–2.59 rad/s.

The plan-time contact predictions:

| check | A plans passing |
|---|---|
| receiver closes after the giver arrives | 11/24 |
| receiver in the env's grasp cylinder (predicted) | 17/24 |
| jaw capture (predicted) | 8/24 |

These predictions are **uncalibrated**: they reject 4/10 of the known-good sources. They must not be read as A failure
rates; only simulation can measure those.

## 4. Simulator capabilities for judging a hand-over

| need | available? | where / note |
|---|---|---|
| object pose per step | **yes** | `states/rigid_object/can/root_pose` (post-step). Do **not** use `obs/datagen_info/object_pose`, whose rotation is forced to identity |
| both grasp-frame poses | **yes** | `obs/eef_pos/quat`, `obs/right_eef_*`, `datagen_info/eef_pose/*` (pre-step) |
| joints, joint targets, finger positions | **yes** | `states/articulation/robot/joint_position`. In generation files also `lerobot/joint_pos(_target)` |
| hand-over stage, success | **yes** | `_handover_tick`: right holds → passed to left → 1 s firm receiver grip with the giver off. The offline replica (`gf/sim_metrics.py`) reproduces the recorded success flag on **40/40** GT demos |
| stable retention, drop, slip, jolts | **derivable** | from can pose + grasp frames (`gf/sim_metrics.py`) |
| contact between pads and can | **NO** | Only a debug `ContactSensor` on the right arm's left finger, filtered to the can. It is not recorded, and the left hand has none. Dual contact cannot be observed directly |
| contact / grasp force | **NO** | no force or joint-effort logging. Even if added, it would be simulator force, not a validated external torque |
| arm–arm collision | **NO** | robot self-collision is OFF by default (`OPENARM_SELF_COLLISIONS=0`: it blocked real-replay joints), so the arms can interpenetrate. Only an offline capsule check is possible (`gf/kin.py`, link radius 4.5 cm). Known-good demos: min clearance 8.4 cm |
| robot–can, robot–table, can–table collision | yes (physics) | |
| set the can start per rollout | yes | `lerobot_sim_server.reset(object_xy=…)` writes the can root pose; a runner can do the same |
| headless without cameras | not built in | the cfg names three cameras in sensors, observation terms and DR terms. A runner must strip all three from its cfg copy (no edit of IsaacLab files) |
| per-step can pose for the sim_10 sources | **not on this machine** | the server HDF5. Blocker B1 |

**Missing capabilities, in priority order:**
1. per-step source object pose for sim_10 (copy the server HDF5);
2. a pilot runner (§5 of the plan; deliberately not written before review);
3. contact sensors on all four pad links filtered to the can, recorded per step: dual-contact ground truth, and the
   sim-side analogue of where force would later enter;
4. an arm–arm collision signal (self-collision on, for the pilot only, or a recorded offline capsule check);
5. joint-effort logging (simulator quantity, for future comparison with validated telemetry only).

## 5. Implications for the new experiment

* **A** is a faithful offline re-implementation of the baseline. Zero-parameter B equals A exactly (`test_zero_parameter_B_equals_A`).
* Any hand-over diversity must be created at the pass:
  * where the hand-over happens;
  * how and when the receiver approaches and grasps;
  * when the giver lets go.
  This needs the two arms to stay consistent with **one** can. That is what the existing generator avoided by
  replaying (its docstring: "any difference between the two arms' transforms is the error at the hand-over").
* **Re-anchoring needs a prediction made close to the pass.** The documented failure of the earlier ten-segment
  structure was re-anchoring the receiver on the can at `presented`, 14 cm away from the pass point. Condition C
  instead predicts the can **at the pass** from the measured giver attachment (§3 of the plan).
* **Contact cannot be predicted offline for these sources** (6/10 self-capture). Success and novelty must be measured
  on simulator rollouts, from the simulator's can pose, not from plans and not from images.
