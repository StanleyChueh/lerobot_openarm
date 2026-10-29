# Battery insertion: contact-data collection protocol (design only)

**Status:** design. Nothing here has been run. Prerequisites: a Gate-2 bench report of **GO**
([GATE2_HARDWARE_READINESS.md](GATE2_HARDWARE_READINESS.md)), the new instruments of §3, and a separate
explicit approval for each session. Primary task: `ethanCSL/openarm_insertbattery_lerobot_real_v00`. Pringles
handover is a secondary reference only.

## 1. The task, and why contact is hard to see in it

From the battery sim README (IsaacLab `scripts/tools/sysid/README.md`, built from the `_v01` recording):

* the **left arm lifts a charging station** (top face 11.5 × 6.5 cm, 3.75 cm thick);
* the **right arm inserts an AA cell** (5.05 × 1.45 cm) into its open bay, about 1.5 cm wide, 1.2 cm deep;
* the operator tilts the charger (about 34° in the sim's meeting pose);
* the fingertips are foam pads about 7 cm long.

**Confirm the roles on `_v00` video before collecting.**

Consequences for sensorless contact:

1. **The insertion force is internal to the two hands.** The cell pushes on the charger and the charger
   pushes back. Both are held by the robot, so the right arm feels `+f` and the left arm feels `−f`. That
   is a usable signature: a real contact must appear in both arms' residuals, consistent through each
   arm's Jacobian. It is also a confound: the left arm's residual also carries the charger's weight and
   inertia.
2. The clearance is about 0.5 mm, so contact is mostly sliding along the bay walls and lead-in: small
   normal forces, friction, and tilt. Expect forces of a few N at most, not a hard stop.
3. Motor `tau` is **not** a calibrated force (HARDWARE_AUDIT §3). It is the DM firmware's current-based
   effort. Nothing below converts it to Newtons without block B's loads and an external reference.

## 2. What the reported effort contains, and what separates each part

Per joint *j*, with the telemetry columns of [TELEMETRY_SCHEMA.md](TELEMETRY_SCHEMA.md):

```
tau_j  =  G_j(q)  +  P_j(q, payload)  +  F_j(dq, history)  +  [M(q) q̈ + C(q, dq) dq]_j  +  τ_ext,j  +  ε_j
          gravity    held object        friction/hysteresis   acceleration                 CONTACT     scale/offset error
```

The motor's own control law, which produced `tau`, is `≈ cmd_kp·(cmd_q − q) + cmd_kd·(cmd_dq − dq) + cmd_tau`,
where `cmd_tau` = gravity at the **commanded** q (+ `squeeze_tau` on J8).

| term | how it is separated | data block |
|---|---|---|
| gravity `G` | URDF gravity at **measured** q (not the commanded q the feed-forward uses), plus a per-joint offset/scale fitted on static holds without contact | B0 |
| held object `P` | the same holds carrying the cell (right) or the charger (left); weigh both objects | B1, B2 |
| friction `F` | Coulomb + viscous per joint and direction, fitted on free motion at three speeds; the static band from holds approached from both sides (hysteresis) | A1–A3, B |
| acceleration | `q̈` from `dq` (30 Hz finite difference, smoothed); `M`, `C` from the URDF; plus a **negatives matched in speed and acceleration** rule: every contact window has a contact-free window of similar `|dq|`, `|q̈|` and pose | A, C-twins, D |
| grip | `squeeze_tau` is known per row. The squeeze closes a loop through the object, so the arm joints see only the object's weight and inertia; J8 carries the squeeze. J8 is excluded as a contact feature unless its no-contact squeeze response is modeled from A2/B1 | A2, B1 |
| scale `ε` | known masses at known poses give each joint's `tau` scale; the load cell in C gives the end-effector reference | B3, C |
| **contact** | what remains, with the bimanual consistency check (§1, point 1) | C, D |

Residual families to compare offline (CPU only), simplest first:

* **R0:** `tau − cmd_tau`, i.e. the controller's PD effort. No model, so it is the baseline any reviewer
  asks for.
* **R1:** `tau − Ĝ(q_meas) − P̂ − F̂(dq)`, quasi-static, fitted on A + B.
* **R2:** R1 + `M q̈ + C dq` (URDF), or a generalized-momentum observer.
* **R3:** a small learned model (≤ 100 k parameters, 0.3–0.5 s window) fitted on A + B **negatives only**,
  or trained on C labels with session hold-out.

## 3. Independent ground truth for contact onset

Labels never come from motor effort or any residual: that would be circular. Every external signal is
stamped on the **same host** with `time.time_ns()` (CLOCK_REALTIME, the clock of telemetry `t_wall` and of the
kernel frame stamps), so no clock fitting is needed. The remaining error is the serial latency, measured once
(below).

| id | instrument | gives | used in |
|---|---|---|---|
| GT1 | **load cell** (5 kg bar or S-beam, 24-bit ADC ≥ 80 SPS, ideally ~1 kSPS) under a rigid charger fixture clamped to the table | contact onset (threshold 0.2 N above the unloaded noise) and force in N along its axis | C |
| GT2 | **thin FSR** (≤ 0.3 mm) on the bay floor, read by a microcontroller at 1 kHz | seating onset and release; uncalibrated magnitude | C3, D |
| GT3 | **video at ≥ 120 fps** (phone or wrist camera crop) of the bay, plus a microcontroller-driven LED in frame that blinks at known host times | first touch (frame-level), annotated **blind** to telemetry by two people; agreement reported | C, D |

Synchronization check, once per session:

* **Serial latency:** the microcontroller echoes 1000 host pings; host-stamp latency must have p99 ≤ 2 ms.
* **LED:** at least 3 blinks at the start and end. The fitted video-to-host offset must agree within one video
  frame.

Instrument safety:

* Use a **dummy AA** (3D-printed, or a fully discharged cell) of the same size and mass, and an
  **unpowered** charger, in blocks C and D. The FSR, tape and fixture then never touch a live cell.
* Weigh the cell, the charger and the fixture; log the masses.

## 4. The protocol

Each block is its own `lerobot-record` run, with:

* `--robot.telemetry_dir=$D/telemetry`;
* `--teleop.diag=true`;
* a fresh local root `$D/<block>`, `--dataset.push_to_hub=false`, repo_id `local/openarm_contact_<block>_v00`.

Alongside it go the sensor log `$D/<block>/sensors.parquet` (`t_ns`, instrument, value) and a `block.json`
(objects held and their masses, instrument calibration, operator, any deviation). Gate-2 §5 prerequisites
apply to every session. Speeds below are joint-speed caps the operator keeps to, checked afterwards from `dq`.

### A. Free motion, no contact (the negatives)

| id | what | duration |
|---|---|---|
| A1 | empty grippers **open**; sweep the task workspace, including the approach and meeting region of both hands, at slow (≤ 0.2 rad/s), task-like and fast (≤ `max_joint_speed`) | 4 min |
| A2 | empty grippers **closed** (squeeze active), same sweeps | 2 min |
| A3 | right hand holds the dummy cell, left hand holds the charger; task-like motions **without the objects touching** each other or anything else | 4 min |
| A4 | A3 along the real insertion approach, **stopping 2–3 cm short** of the bay, 15 approaches | 3 min |

A4 is the critical negative: same poses, speeds and grip as insertion, minus the contact. A detector that
scores well on D but also fires on A4 has learned the phase, not the contact.

### B. Static holds, no contact

* 20 poses per arm across the task region, including the meeting pose.
* Each pose is held 3 s and approached once from each side (above/below) for the friction band.

| id | load | poses × approaches |
|---|---|---|
| B0 | empty, grippers open | 20 × 2 |
| B1 | right holds the cell; left holds the charger | 20 × 2 |
| B2 | the same poses, grippers closed on nothing | 10 × 1 |
| B3 | known masses hung from the gripper (0 / 0.2 / 0.5 kg, weighed), at 6 poses per arm | 6 × 3 |

Holds count only when every joint has `|dq|` below 0.02 rad/s for at least 2 s.

### C. Controlled, labelled contact (fixture on the load cell)

The charger is clamped in its bay-up orientation on GT1, with GT2 in the bay and GT3 aimed at the bay. The
right arm holds the dummy cell.

| id | contact | reps | twin (no contact) |
|---|---|---|---|
| C1 | touch the cell's tip on the charger **top face** beside the bay, hold 1 s, retreat | 10 at two approach speeds | 10: the same path, stopping 1 cm above |
| C2 | push the cell **sideways** against the bay's lead-in wall, hold 1 s | 10 | 10 stopping 1 cm short |
| C3 | full **insertion** into the bay, release, withdraw the hand | 10 | 10 to 1 cm above the bay |
| C4 | left hand holding the charger presses it onto the load cell (the left arm's own contact) | 10 | 10 stopping 1 cm short |

Contacts and twins are interleaved in a pre-randomized order written into `block.json` before the session.

**Limits:**

* The operator approaches the last 3 cm at ≤ 0.1 rad/s.
* **Abort the trial if GT1 exceeds 15 N**, or on any hold or fault print.
* The supervisor watches the live GT1 readout.
* For scale: the wrapper's tracking guard holds a joint lagging 0.15 rad for 2 s. At kp 60 that is about
  9 N·m on J1/J4, and about 3 N·m on the kp-20 wrist joints. The fixture and load cell must take ≥ 50 N.

### D. Battery-insertion diagnostics (the real task, short)

* **D1:** 10 normal bimanual insertions, the left arm holding the charger as in `_v00`. GT2 is in the bay
  and GT3 is on the bay; there is no load cell, because the charger is in the hand.
* **D2:** 5 deliberately misaligned insertions: first touch off-center, then search and seat.
* **D3:** 5 approach-only episodes, stopping short. This is A4 within the same session as D1/D2.

These are teleoperated. **No policy runs here**: deploying the existing VLA is a separate gate.

## 5. Labels and splits

* **Contact onset:** GT1 (C), or GT3's blind annotation (D), on the host clock; GT2 marks seating.
* **Contact intervals:** C uses GT1 above 0.2 N. D uses GT3 annotation from first touch to release.
* **Phase labels in D**, in a frozen vocabulary: approach, first contact, search, seating, release,
  withdrawal. These follow the insertion phases in the Codex team's `EXPERIMENT_MATRIX.md`. Two annotators,
  with agreement reported.
* **Telemetry join:** `align_to_dataset` gives each frame its telemetry row. An external event at host time
  `T` belongs to the first row with `t_wall ≥ T`.
* **Splits** by session and block, never by window. Thresholds and model choices are selected on sessions
  that are excluded from the reported numbers.

## 6. Analysis and pre-registered success criteria (offline, CPU)

For R0–R3 per arm, report:

* separability on C, with GT1 intervals: AUROC, and F1 at the chosen threshold;
* **false alarms per minute** on A (all) and separately on A4, C-twins and D3;
* onset latency vs GT1 (C) and GT3 (D): median and p90;
* the scale of `tau` per joint from B3; the residual noise floor per joint from A + B.

Minimum to justify feeding a contact signal to a policy (Codex's option B/C):

* on held-out sessions, a detector with ≤ 1 false alarm per minute on A4 + C-twins detects ≥ 90 % of C
  contacts above 2 N within 100 ms of GT1 onset;
* the same detector improves contact discrimination over a state- and velocity-only detector at the same
  false-alarm rate. That is the falsification test in Codex's `EXPERIMENT_MATRIX.md`.

If the criteria fail, report the noise floor and stop: do not train a VLA on the signal. Newton estimates are
reported only along GT1's axis in C, and only with B3's scale fit. A 3-D force claim needs a 6-axis F/T
reference (D6 in CONTACT_FEASIBILITY), which is not in this protocol.

## 7. Budget

| block | robot time |
|---|---|
| A | 13 min |
| B | ~20 min including moves |
| C | ~40 min (80 trials + setup) |
| D | ~15 min |
| total | about 1.5 h of supervised robot time over two sessions, plus instrument setup and the sync checks |

Storage: about 1 MB/min of telemetry (rows + raw frames), plus the camera video.

## 8. Open items before approval

* the charger's and cell's masses;
* which bay the `_v00` demonstrations used;
* the roles in `_v00` (confirm against `_v01`);
* load cell and fixture hardware, and its rated load;
* FSR fit in a 1.5 cm bay without changing the clearance;
* whether a 6-axis F/T sensor is available (needed only for 3-D force);
* who annotates GT3.
