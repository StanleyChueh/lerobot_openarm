# Sensorless contact on OpenArm: feasibility of three approaches

Inputs: [HARDWARE_AUDIT.md](HARDWARE_AUDIT.md) (what the motors report) and
[TELEMETRY_SCHEMA.md](TELEMETRY_SCHEMA.md) (what we now log). Reference: Dou et al., *NeuralActuator: Neural
Actuation Modeling for Robot Dynamics and External Force Perception*, arXiv:2607.11734 (RSS 2026).

**No real-robot data has been collected for this study.** Every number about OpenArm below is either a
resolution derived from the wire format or explicitly marked as unknown.

## 1. Three different claims; keep them apart

| target | output | needs | what a paper may claim |
|---|---|---|---|
| **Contact classification** | contact yes/no (per arm, optionally per tick onset) | a residual or classifier whose no-contact distribution is separable from contact; contact/no-contact labels | "detects contact", with F1 / false-positive rate / onset latency |
| **Joint-space external torque** | `tau_ext ∈ R^7` per arm | dynamics model (gravity, friction, inertia) accurate below the contact signal; `tau` scale per joint | "estimates external joint torque" only if `tau` is scale-calibrated (known loads) |
| **Calibrated 3D force** | `f ∈ R^3` at a contact point, N | all of the above, plus a contact-location assumption (end effector), the Jacobian `J_v(q)` and **ground-truth force** (F/T sensor, gauge or known weights) to validate | "estimates force" with MAE in N against a reference |

The step from the second row to the third is not free. With 7 joints and 3 unknowns, `J_vᵀ f = tau_ext` is
overdetermined, so a least-squares `f` always exists, even when it is wrong. Only an external reference
shows that it is right. While `meta.provenance.torque_calibration` is `null`, OpenArm results can claim
**contact classification** and **uncalibrated effort residuals**, nothing in Newtons.

## 2. Signal budget on this arm

* Rate: 1 sample per control tick (30 Hz in lerobot-record). The MIT-reply and refresh-reply pair gives two
  frames per tick, but they are only ~ms apart. NeuralActuator ran at about 60 Hz with a 9-frame history.
* `tau` LSB: 0.026 N·m (J1–J2), 0.014 (J3–J4), 0.0049 (J5–J8). Not the bottleneck.
* Reported torque ≈ the controller's own effort, `kp·e + kd·ė + tau_ff` (audit §3). Contact appears as
  tracking error × kp. At the current gains:
  * proximal joints (kp 50–60): high effort per mrad but dominated by gravity-model error and planetary-gear
    friction;
  * wrist (kp 20): more error per N·m, lower friction, and closest to the end effector.
* Unknown and decisive: the free-motion residual noise floor per joint (friction + hysteresis + gravity-model
  error). Measure it before choosing an approach (§4, data item D2).
* Not available: supply voltage and phase current. The DM state frame carries only q, dq, tau and two
  temperatures; NeuralActuator also uses V and T.

## 3. The three approaches

### 3.1 Existing motor torque + residual-based contact detection

`r = tau − τ̂(q, dq, q̈)` with `τ̂ = G(q) + F(dq) [+ M(q)q̈ + C]`. The two options:

* quasi-static: `G` from the URDF at **measured** q, plus fitted Coulomb/viscous friction;
* generalized-momentum observer (GMO): `M`, `C` from the URDF.

Threshold `|r_j|` per joint, or a χ² test over joints. All inputs exist in the telemetry: `q, dq, tau`, plus
`cmd_tau` and `squeeze_tau` to remove feed-forward when using the `kp·e` form.

* **Pros:** no learning, CPU only, interpretable, the natural baseline any reviewer will ask for. The
  classification and onset-latency metrics come directly.
* **Cons:** URDF inertials are unidentified. Gearbox friction is history-dependent (hysteresis, backlash).
  NeuralActuator reports classical observers at **33–59 % false-positive contact rates** on low-cost servos,
  and their best classical baseline (GMO) at 0.66 N MAE vs 0.12 N for the learned model. Expect the
  same failure here unless friction is identified.
* **Feasible now?** For contact classification, **yes, once D1–D3 exist**. For calibrated force, no.

### 3.2 Lightweight learned contact estimator

A small temporal model (1-D CNN / GRU / MLP over a 0.3–0.5 s window) per arm. Inputs:
`q, dq, tau, cmd_q − q, cmd_dq, cmd_tau, squeeze_tau, t_rotor`, plus the residual from 3.1. Output:
`p(contact)`, optionally regressing the residual's magnitude.

* **Pros:**
  * learns friction and hysteresis the analytic model misses;
  * still tiny (≲100 k params, trains on CPU in minutes, so no GPU training is needed);
  * can run online inside the 33 ms tick.
* **Cons:**
  * needs **contact labels**;
  * high risk of learning task phase instead of contact. In the plate tasks, contact always happens at
    the same poses, so a classifier can score well by recognising the pose.
  * Mitigation: the training set must include the same poses *without* contact (D2/D4) and evaluation on
    held-out poses or tasks.
* **Feasible now?** **Yes after D2 + D5.** It is the most likely path to a usable contact signal for a
  force-aware policy.

### 3.3 NeuralActuator-based force estimation

From the paper:

* Transformer, 1.44 M params, 9-frame history.
* Inputs: commanded targets, q, q̇, actuator telemetry (current / load register / commanded torque),
  supply voltage, temperature, tracking error.
* Heads: torque surrogate (trained through MuJoCo MJX differentiable simulation from pose trajectories,
  no torque labels), 3-D EE force with a contact gate, and motor condition.
* Force labels: fixture-mounted 6-axis F/T sensor, known weights, gauge pushing.
* About 95 min of data on OpenManipulator-X, about 17 min on SO-101.
* Trained on an RTX 5080 (same class as ours).
* Reported: 0.08–0.57 N force MAE on small servo arms. Payloads ≤ 50 g sit below their noise floor. Franka
  results are offline and future-state-conditioned.

Fit to OpenArm:

| requirement | status here |
|---|---|
| per-tick telemetry with commanded targets and tracking error | **have it** (`cmd_*`, `q`, this branch) |
| current / torque channel | `tau` (firmware estimate), analogous to their Franka/SO-101 inputs |
| supply voltage | **missing**: not in the DM state frame. Would need a parameter query (sends frames) or an external meter |
| temperature | have `t_mos`, `t_rotor` |
| differentiable sim model (MJX) of the arm | partial: MuJoCo scenes exist (`openarm_mujoco`, the IK scene), but inertials are not identified, and dual-arm 7-DoF + grippers is larger than their 5–6-DoF arms |
| force ground truth | **missing**: no F/T sensor or gauge protocol in this repo |
| GPU training | **not allowed in this phase** |
| code | not inspected (no public code reviewed here) |

* **Feasible now?** **No.** It is the right *comparison target* for a 2027 paper, and the telemetry contract
  was shaped so it could be fed later, but it needs D4–D6 plus an MJX model first.

## 4. Minimum additional calibration data

All items need explicit approval and an operator. None of them have been run. Ordered by what they unlock:

| id | what | how (no new control code) | unlocks |
|---|---|---|---|
| D1 | firmware PMAX / VMAX / TMAX, KT, Gr per motor | read-only `query_param` with motors disabled (sends query frames, so it needs its own approval) | trust in `dq`/`tau` scaling (audit #5) |
| D2 | free-motion, **no contact**, ~10–15 min per arm, covering the task workspace and speeds, including the same poses as the tasks' contact phases | ordinary teleop (`lerobot-teleoperate` / `record`) with `--robot.telemetry_dir` | residual noise floor, friction fit, false-positive rate; the negative class for 3.2 |
| D3 | static holds, ~20–30 poses per arm, 2 s each, no contact | teleop to pose and hold | gravity-model error and static-friction band per joint |
| D4 | known loads: 3–4 masses (e.g. 0 / 0.2 / 0.5 / 1.0 kg) in the gripper, ~10 poses | teleop | `tau` scale per joint, vertical `J_vᵀ f` check: the minimum for any Newton claim |
| D5 | labelled contact: 20–30 contacts per task phase with a synchronized reference (kitchen scale or FSR under the plate/table, logged with `time.time()`), or frame-level video annotation | teleop | positives for 3.2, onset latency for 3.1 |
| D6 | 3-axis force reference: 6-axis F/T sensor at the fixture, or gauge pushes along known axes (NeuralActuator protocol), ~15 min | teleop | calibrated 3D force (3.3, or 3.1 with `J_vᵀ`) |

**Minimum for contact classification:** D2 + D3 + D5, with D1 strongly recommended.
**Minimum for a calibrated force claim:** D1–D4 + D6.

## 5. Recommendation

1. **Gate 1** (audit §8): a passive bench run to confirm wire visibility (`cmd_n`), value freshness
   (`state_fresh`), overhead, and no added CAN traffic.
2. Collect D2 + D3 with telemetry on. Fit the quasi-static residual (3.1) offline on CPU and report the
   per-joint noise floor. This single number decides how small a contact the arm can see.
3. Add D5 and train 3.2 on CPU. Compare against 3.1 on held-out poses: F1, false-positive rate on
   contact-free motion at task poses, onset latency.
4. Only with D4 + D6 (and approval for GPU work): calibrated force with `J_vᵀ`, and NeuralActuator as the
   learned comparison.

Feeding a contact signal into the VLA would change the policy's observation schema. It belongs in a later
phase, after offline results show the signal is real.
