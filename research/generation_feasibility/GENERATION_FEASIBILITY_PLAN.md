# Generation feasibility: can we generate genuinely new, executable bimanual hand-overs?

* **Date and version:** 2026-11-05, spec `genfeas-0.1-dryrun` (`gf/config.py`).
* **Status:** design + dry run complete. **No simulation has been run.** This document stops here for Codex review.
* **Relation to earlier work:** a new experiment, not a continuation of the Gate 1 / 1R verifier. That line stays closed
  and frozen (test `test_gate1r_verifier_frozen`). Gate 2 stays NO-GO under that protocol.
* **Inputs:** [BASELINE_GENERATOR_AUDIT.md](BASELINE_GENERATOR_AUDIT.md),
  `research/generated_data_diagnosis/` (earlier audit of generated_500), IsaacLab `4aa5e026`.

## 1. Question and what counts as an answer

**Question:** starting from the 10 manual sim demos (`ethanCSL/openarm_pringles_lerobot_sim_10_v00`), can we produce
simulated hand-overs that meet both conditions?

* **(a) They succeed stably,** judged on simulator ground truth.
* **(b) They differ from every source in the hand-over itself:** where the can is passed, and where and how the
  receiver grasps it. A new can start or a shifted approach alone does not count.

**Novel, at the hand-over, means:** an executed rollout whose descriptor (`gf/descriptors.py`) satisfies at least one of:
* the pass point (can centre when both hands hold it) is ≥ 3 cm from **every** source pass. The source-to-source
  nearest-neighbour distance is 1.0 cm;
* the receiver grasp lies outside the source range:
  * axial separation from the giver's grasp beyond [6.1, 12.8] cm by ≥ 1 cm; or
  * relative azimuth beyond [176°, 233°] by ≥ 10°.

**A novelty claim needs executed rollouts.** Planned descriptors only show what the generator asked for.

## 2. Conditions (same sources, same 24 can starts, same source per start)

| | name | what it changes | runtime rules (applied by the runner, using sim state) |
|---|---|---|---|
| **A** | geometry-only (baseline) | Offline re-implementation of the current Mimic hand-over config (audit §3.1): nearest source (nn_k=1); right approach translated by `can_new − can_src`; everything else replayed; bridges 22 / 22 / 36 steps, smoothstep; smoothing 7 | env's giver clamp (hold until taken + 8 steps) |
| **B** | contact-phase-conditioned | A, plus per-candidate parameters θ at the hand-over (ranges in §2.1). **Open loop:** the receiver's new grasp is the source receiver pose moved by θ, relative to the *source-predicted* can; timing as sampled | env clamp + sampled extra giver hold |
| **C** | contact-constrained | B's θ, plus constraint enforcement (§2.2) | env clamp + extra hold; **re-anchor** of the receiver at the pass; **receiver close gate** |

### 2.1 Parameters θ (shared by B and C for each pose)

`gf/config.py: B_RANGES`, sampled with `rng(4242 + i)`:

| parameter | range | applies to |
|---|---|---|
| pass point translation, x / y / z | ±7 / ±6 / −4…+5 cm | giver from lift to pass and back over its withdraw; receiver from approach through carry (one rigid translation for both, so the hands still meet) |
| receiver approach via-point | 0–6 cm, random direction ⟂ approach | zero at both ends |
| receiver grasp on the can | ±4 cm along the axis, ±45° about it | relative to the source's receiver grasp |
| receiver approach speed | ×0.8–1.25 | time-warp of the approach |
| receiver dwell before closing | 0–12 steps | |
| giver release delay | 0–15 steps beyond the 8-step overlap | the giver holds position, closed |

### 2.2 What C enforces (`gf/plans.py: plan_phase(cond="C")`, `gf/checks.py: make_C`)

1. **Object attachment / jaw capture.**
   * The receiver grasp is built in the predicted can's frame. It keeps the source receiver's orientation, rotated
     about the can axis by θ, with the can centre placed at the GT-measured capture point of the jaws (audit §3.5).
   * It stays on the can body (|axial| ≤ 7.5 cm) and ≥ 4.5 cm from the giver's grasp along the axis.
   * The approach ends in a 6 cm straight line along the grasp frame's approach axis.
2. **Synchronisation, both ways.**
   * The receiver closes only after the giver has reached the pass (plan time; receiver dwell added if needed).
   * The giver holds still at the pass until the receiver has closed plus the overlap. Otherwise the replayed withdraw
     drags the clamped can away from a late receiver: this was found in the dry run and holds for B as well.
3. **Runtime re-anchor.**
   * At `anchor` (15 steps after the right grasp) the runner measures the giver attachment
     `X = T_R⁻¹ · T_can` from sim state.
   * It predicts the can **at the planned take** as `target_R[take] · X`.
   * It applies one rigid correction to all remaining receiver targets, blended in over the approach.
   * Only the post-anchor drift (median 5–7 mm) remains as receiver error. The earlier failed variant re-anchored on
     the can at `presented`, 14 cm from the pass.
4. **Receiver close gate:** close only when the simulator's `hand_on_object` test holds. Wait ≤ 15 steps, then close
   anyway and log it.
5. **Robot limits.**
   * IK inside the URDF limits − 0.02 rad, elbow ≥ 0.10 rad; IK error ≤ 5 mm / 3°.
   * Joint step ≤ 2.5 rad/s, the real-robot wrapper cap, repaired by retiming both arms together.
   * Target jump ≤ 0.25 rad.
6. **Collision (offline, because sim self-collision is off):**
   * arm–arm link capsules ≥ 3 cm;
   * hand–hand ≥ 1.5 cm outside the dual-contact window;
   * finger tips and the carried can ≥ 1 cm above the pad outside the pick.
7. **Shrink or reject.** A violation shrinks the spatial parts of θ to 60 %, then 35 %. A candidate that still fails is
   **rejected**, not repaired by replaying the source. Rejected candidates count as failures in S1.

## 3. What the dry run established (`scripts/run_dry_run.py`, 21 s CPU)

Outputs:
* `outputs/dry_run/dry_run_summary.json`;
* `candidates.jsonl` (every check, descriptor and novelty value);
* `pilot_manifest.json` (all 84 candidates with parameters, marks, runtime rules and plan SHA-256);
* `plans/*.npz`;
* `planned_handover_diversity.png`.

| (24 pilot plans per condition) | A | B | C |
|---|---|---|---|
| accepted for execution | 24 | 24 | **20** (4 rejected: IK or clearance at every scale) |
| IK feasible (≤ 5 mm, 3°) | 24 | 12 | 20 of 20 accepted |
| > 2.5 rad/s joint step (real-robot cap) | 11 (inherited from the sources) | 17 | **0** |
| receiver in jaw capture under the plan's own prediction | 8 | 8 | 24 |
| receiver closes after the giver arrives (plan time) | 11 | 17 | 24 |
| **planned** hand-over outside the source envelope | **0** | 19 | **11 / 20** (pass ≥ 3 cm: 11; azimuth: 2; axial: 0) |
| median planned novelty ratio (pass / path), 1 = source-to-source | 0.97 / 0.13 | 5.1 / 2.5 | 3.5 / 1.9 |
| capture under sampled giver-attachment error (model, §3 caveat) | 0.43 | 0.14 | 1.00 |

Smoke stage (4 disjoint poses): A 4, B 4, C 3 accepted.

**Reading the table:**
* **A** is what the audit says: it re-executes the source hand-over (0/24 novel, path ratio 0.13).
* **B** asks for new hand-overs but is kinematically infeasible in half of its plans (12/24 IK). Under the plan's own
  prediction its receiver usually misses the jaw capture.
* **C** produces 20 kinematically valid, synchronised plans. 11 of them request a new pass point. Its grasp-on-can
  variation stays almost entirely **inside** the source range (2 azimuth, 0 axial): the can body (20 cm) and the
  two-hand separation leave little room, and the sources already span 6–13 cm and 176–233°.
* **Grasp diversity is the weak axis.** Pass-location diversity is the main thing C can add.

**Limits of the dry run (none of these is a success estimate):**
* The contact predictions are uncalibrated. With the nominal attachment, the plan-time capture test passes only 6/10
  of the known-good sources (audit §3.5).
* The attachment-error model predicts 0.43 for A, whose real success rate in the current env is unknown. It is
  recorded only to motivate C's re-anchor; no decision uses it.
* No physics, contact or friction was simulated. Only simulation can show whether a shifted pass point or a re-anchored
  receiver actually holds the can.

**Metric code validated on recorded sim truth** (40 local GT demos, 20 Hz, older env):
* the offline replica of the env's success rule matches the recorded flag on **40/40**;
* stable success holds on **40/40**;
* deliberately corrupted copies are rejected: a dropped can, jaws shut on air, a giver that never releases
  (unit tests). This tests the code, not generated data.

**Design revisions made during the dry run** (all before any simulation, recorded here; plan/check code only):
1. **Stable success:** the first definition rejected 10/40 known successes, because the operator lowers the held can
   to within 5 mm of the pad. It now requires a drop (neither hand holds ≥ 3 steps) or a lost receiver grip; lowering
   is logged only.
2. **Slip criterion:** known-good demos already show slip > 1.5 cm in 11/40, so S3 is relative to the P0 replay baseline.
3. **Azimuth wraparound:** the relative azimuth now lives in [0°, 360°). The sources sit near 180°, and the old ±180°
   wrap made the envelope cover everything.
4. **Wider θ ranges:** with the first ranges (±4 cm, ±25°) only 6/21 C plans left the envelope, so the pilot could not
   have met S2 even at 100 % success.
5. **C timing fix:** C now references the can where the giver will actually be at the planned take, and adds both
   synchronisation rules.
6. **Nominal attachment:** the sim_10 attachment estimate from the right-close offset was 2–4 cm off. It is replaced by
   the GT mean.

## 4. Pilot metrics (simulator ground truth only; `gf/sim_metrics.py`)

| metric | definition |
|---|---|
| **stable hand-over success** | env success at the final state (stage machine + 1 s firm receiver grip, giver off), **and** firm receiver grip from the giver's release to the end with no loss > 2 steps (≥ 30 steps), **and** no drop (neither hand holds for ≥ 3 steps after the lift), **and** the can never below the pad − 3 cm |
| receiver grasp / hand-over diversity | executed descriptor: pass point; receiver grasp axial separation and relative azimuth on the can; receiver path; timing. Novelty ratio to the 10 sources and the envelope test (§1) |
| slip | max drift of the can in the holding hand's frame: giver (grasp + 10 → pass) and receiver (release + 10 → end) |
| post-transfer retention | steps of firm receiver grip after the giver's release |
| joint-command validity | max request–state gap (> 0.5 rad) and request jump (> 0.25 rad) on the executed joint targets; URDF limit excursions |
| collision / constraint violations | offline arm–arm capsule clearance on the **measured** joints (sim self-collision off); can below the pad; the receiver close gate's timeout count (C) |
| difference from source | per family: nearest-source distance and ratio; DTW-free path RMS after arc-length resampling |
| jolts (reported) | can speed > 1 m/s, tilt > 30° |

## 5. Pilot protocol (prepared, NOT executed)

### 5.1 Stages

| stage | rollouts | purpose | gate |
|---|---|---|---|
| **P0a: source replay** | 10: each sim_10 source re-executed through the runner at its own can start (A with T = 0) | the runner, env and timing reproduce known successes in the current env; sets the source slip baseline | ≥ 8/10 stable, else **stop and report: env/runner broken** |
| **P0b: smoke** | 4 disjoint poses × A, B, C (11 accepted) | the runtime rules work (re-anchor, close gate, extra hold); measure wall time per rollout | A ≥ 3/4 stable, else stop. Engineering fixes are allowed here **only to the runner**, logged, before P1. No change to θ ranges, thresholds or criteria |
| **P1: pilot** | 24 × A + 24 × B + 20 × C (accepted) = 68 | the experiment | evaluated **once** (§6) |

### 5.2 Runner specification

`scripts/run_pilot_isaac.py` is to be written only after review; its absence is a preflight failure.
1. **Environment:** `Isaac-PickUp-RedCube-OpenArm-IK-Abs-Mimic-v1`, `task_mode=handover`, `OPENARM_CONTROL_HZ=30`,
   no domain randomization, `num_envs=1`, headless.
   * Cameras are removed from a **runtime cfg copy**: sensors, observation terms and DR terms.
   * IsaacLab files are not edited.
   * Contact sensors on all four pad links, filtered to the can, are **added in the cfg copy** and recorded (simulator
     force, labelled as such).
2. **Per candidate:**
   * reset;
   * write the can root pose to (x, y, 0.380) as `lerobot_sim_server.reset(object_xy)` does;
   * the robot at its default pose (all sources start at rest).
3. **Each step:**
   * build per-arm target poses from the plan;
   * apply the runtime rules (C re-anchor at `anchor`, close gate at `L_close`, extra giver hold);
   * convert with the env's own `target_eef_pose_to_action`, so the env's giver clamp applies identically to A, B, C;
   * step.
4. **Record** the Mimic HDF5 layout, so `gf/sim_metrics.py` reads it unchanged:
   * can pose, joints, joint targets, both grasp frames, gripper commands;
   * stage-machine signals and contact forces;
   * ≤ 420 steps (14 s; the longest plan is 341).
5. **Per-rollout JSON:** `rollout_metrics` + `executed_descriptor`, plus the plan SHA-256 from the manifest.
6. **Runner tests before P0:**
   * an offline unit test that the plan→action conversion equals Mimic's delta-pose math;
   * P0a itself is the integration test.

### 5.3 Resources and safety

| item | estimate |
|---|---|
| rollouts | 10 + 11 + 68 = **89** (≤ 341 steps each) |
| wall time | **Upper bound only.** The single timing reference is the generated_500 file: ≥ 500 episodes with 3 cameras on 4 envs, written ≤ 8.7 h after the code fix, i.e. ≤ 63 s per episode. Without cameras: ≤ 89 × 63 s ≈ 1.6 h. P0b measures the real figure |
| compute budget cap | **3 GPU-hours, 1 GPU, 1 process.** Simulation only: no training, no rendering of datasets |
| disk | ≈ 1–2 MB per rollout without images (< 0.5 GB) |
| what is NOT done | no VLA or diffusion training; no robot, CAN or hardware; no Hugging Face upload; no production or safety code change; no edit of IsaacLab sources |
| prerequisites (`scripts/pilot_preflight.py`, currently **REFUSED**, exit 2) | B1 server HDF5 with per-step can pose for the sim_10 sources; B2 the runner; a reviewer-written `outputs/PILOT_APPROVED.json` carrying the spec SHA-256 |

## 6. Success / failure criteria (pre-registered; `gf/config.py: CRITERIA`)

| id | criterion |
|---|---|
| P0 | P0a ≥ 8/10 stable source replays; P0b A ≥ 3/4. Failure = the setup is invalid, not a result about generation |
| **S1 executable** | C stable successes **≥ 15/24** (rejected candidates count as failures), and not more than 3 below A |
| **S2 novel** | ≥ 50 % of C's stable successes are novel at the hand-over (§1, **executed** descriptors), **and** C's median pass or path novelty ratio ≥ 0.5 while A's ≤ 0.2 |
| **S3 quality** | in C's stable successes: retention ≥ 30 steps in ≥ 90 %; slip > 1.5 cm rate ≤ P0a rate + 15 points; 0 joint-target contract violations; 0 offline arm–arm clearance violations |
| S4 (reported) | C − B stable success, paired by pose (McNemar), and C − A |

**Interpretation, fixed before the run:**

| outcome | meaning |
|---|---|
| S1 ∧ S2 ∧ S3 | **GO** for a scaled generation experiment: new poses, then a policy-level test against matched geometry-only, filtering and resampling controls. Still **no** evidence of policy benefit, real-robot transfer or force awareness |
| S1 ∧ ¬S2 | the method only reproduces existing hand-overs (in the tested ranges) → NO-GO for "new hand-overs" |
| ¬S1, with B and C failing mostly at the take | geometry plus kinematics cannot produce executable new passes with these sources → NO-GO for this generator; the bottleneck is contact (where force would enter, §7) |
| ¬S1 with A also < 15/24 | the baseline itself is unreliable in the current env; report, no comparison claim |

**Statistics.** n = 24 per arm, Wilson 95 % interval about ±18 points at 70 %. This is a feasibility pilot, not a powered
comparison. A–B–C differences below about 25 points are not interpreted.

## 7. Where validated external torque would enter (future; nothing here is force-aware)

There are no force measurements: no historical dataset has torque, and the `openarm_telemetry/1` sidecar has never run
on hardware. Simulator contact forces, if recorded in the pilot, are simulator quantities and do not validate anything
about the real robot. The generation mechanism has five specific points where a **validated** external-torque signal
could replace a geometric or timing rule:

| hook in this generator | today (geometry / timing) | with validated torque |
|---|---|---|
| C receiver close gate (§2.2-4) | sim `hand_on_object` cylinder test | close on measured contact onset at the receiver: external torque on the receiver's wrist / gripper motor exceeding its no-contact band |
| giver release (env clamp, 8 steps + θ delay) | fixed step count after the latched take | release when load transfer is measured: the giver's external torque attributable to the can (0.205 kg, ≈ 2 N) falls and the receiver's rises. This replaces the fixed overlap that produced the 10–17-frame squeeze in generated_500 |
| C re-anchor (§2.2-3) | sim GT can pose → attachment X | real robots have no GT can pose. A torque-consistent load estimate (mass and lever arm from the giver's wrist torque) constrains where the can sits in the giver's hand |
| slip | GT drift of the can in the hand frame | a sudden change in the load torque while the jaw is unchanged |
| candidate filtering / θ sampling | kinematic constraints only | reject or down-weight candidates whose simulated contact-force profile (dual-contact squeeze, take impulse) falls outside the profile measured on validated real hand-overs. Condition θ (dwell, release delay) on force targets instead of step counts |

**Preconditions for any of these:**
* a hardware-validated telemetry stream;
* a torque-to-contact calibration with measured noise, which must resolve ≈ 0.6 N·m (2 N at 0.3 m) on the relevant joints;
* a sim-to-real force correspondence.

None exists. Until then: no force-aware claim.

## 8. Stop rules

1. P0 failure → stop, report the setup problem. The pilot is not run.
2. P1 is run **once** on the frozen 24 poses. No θ-range, threshold or criterion change after P1 has started.
3. Any change after the run → a new spec version, new poses (not trials 0–23 or 96–99), a new pilot.
4. If S1, S2 or S3 fails for C: **NO-GO** for scaling this generator; no retuning on the same poses.
5. Budget: stop at 3 GPU-hours regardless of progress. Report what finished.

## 9. Implementation and test plan

**Implemented (CPU):** `research/generation_feasibility/`

| file | role |
|---|---|
| `gf/config.py` | the spec: constants mirrored from IsaacLab with citations, θ ranges, criteria, stop rule |
| `gf/sources.py` | read-only loaders (sim_10 LeRobot + diagnosis joins; local GT HDF5) |
| `gf/kin.py` | pinocchio FK/IK on `model/openarm_description.urdf`, capsule clearance |
| `gf/plans.py` | segmentation; generators A, B, C |
| `gf/checks.py` | offline checks, retiming, C acceptance (shrink or reject), attachment model |
| `gf/descriptors.py` | hand-over descriptors, novelty, envelope |
| `gf/sim_metrics.py` | GT metrics: stage-machine replica, stable success, slip, retention, validity, clearance, executed descriptor |
| `scripts/run_dry_run.py` | dry run → `outputs/dry_run/` |
| `scripts/pilot_preflight.py` | prerequisite and approval gate; never launches simulation |
| `tests/test_generation_feasibility.py` | 16 tests (log in `outputs/test_output/`) |

**Tests** (`env -u PYTHONPATH -u LD_LIBRARY_PATH .venv/bin/python -m unittest discover -s research/generation_feasibility/tests -v`):

| area | tests |
|---|---|
| kinematics | FK vs recorded sim grasp frames (< 1 mm, rotation); IK round trip within limits |
| baseline A | right approach translated, the rest untransformed; left arm = source; A replays the hand-over (path ratio < 0.5, never novel) on all 24 poses |
| B / C | zero-θ B ≡ A; C accepted plans satisfy every C constraint and synchronisation; re-anchor tolerates attachment error and open loop does not (model); retiming keeps the arms in step |
| metrics | replica = recorded success 40/40 and stable 40/40; dropped can / shut-on-air / never-released → not stable; executed descriptor on a recorded episode |
| protocol | A/B/C paired by pose and source, θ shared; smoke poses disjoint; preflight refuses; Gate 1R files unchanged since `2779069` |

**To implement after review:** the Isaac runner (§5.2), with an offline unit test of its plan→action conversion; P0a
is its integration test.

## 10. Blockers

| id | blocker | resolution |
|---|---|---|
| B1 | per-step simulator truth for the 10 sim_10 sources is only on the server (`openarm_pringles_lerobot_sim_10_v00*.hdf5`) | copy the files (no regeneration). Until then, C's plan-time attachment is the 40-demo GT mean (std 1.35 cm), and P0a cannot check source attachments |
| B2 | no pilot runner | write after review (§5.2) |
| B3 | no contact or force recording; self-collision off | runner cfg copy adds pad contact sensors; collisions checked offline |
| B4 | the local GT sources are an older env (20 Hz, pre-pad) | used only for metric validation; P0a re-establishes baselines in the current env |

## 11. Recommendation

**GO for review of the pilot protocol; NO-GO for any generation at scale.**

* **What the dry run supports:**
  * the baseline cannot add hand-over diversity, by construction: 0/24 novel, receiver path ratio 0.13;
  * the sources themselves hand over at almost one place (pass std ≈ 1 cm against 6.4 cm of start spread);
  * a contact-constrained generator (C) can **plan** kinematically valid, synchronised, attachment-consistent
    hand-overs at new pass points (20/24 accepted, 11 of them novel at plan level).
* **What it does not support:**
  * that any of these succeed in simulation;
  * that the plans' contact predictions are trustworthy (6/10 on known successes);
  * that grasp-on-can diversity is achievable (2/20 planned).
* **Where it lands:** S2 is at the margin already at plan level (11/20 = 55 % against a 50 % threshold, before any
  execution failure). A plausible pilot outcome is **"executable but only marginally new"**. That would be a valid
  NO-GO for the novelty claim.
* **Before P0:** resolve B1 and B2, have the protocol reviewed, and record the approval. Nothing beyond the 89
  rollouts and the 3 GPU-hour cap is authorised by this plan.
