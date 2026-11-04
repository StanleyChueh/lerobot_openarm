# Contact observability: what the existing data and telemetry can and cannot show

Companion to [GATE0_AUDIT.md](GATE0_AUDIT.md). Numbers come from `outputs/summary.json`,
`tracking_residual_windows.csv`, `event_aligned_residual.json` and `figures/event_aligned_residual.png`
(VERIFIED), unless marked REUSED, SUPPORTED or UNKNOWN.

**Ground rule:** no estimated force exists for the historical datasets. Nothing below is converted to Newtons
or N·m, and nothing below may be called "force" in a paper.

## 1. Signal inventory

| signal | Pringles real_10_v00 | Battery v02 | new recordings with `--robot.telemetry_dir` |
|---|---|---|---|
| joint position, measured (`observation.state`) | yes, 30 Hz nominal | yes | yes, plus wire-traced age and freshness |
| joint position request (`action`) | yes; the **pre-clamp** request | yes | yes, plus the transmitted `cmd_q` (post-clamp) |
| joint velocity | no (finite differences only) | no | `dq` (motor) |
| motor torque estimate `tau` | **no** | **no** | yes: firmware current-based effort, **uncalibrated** (HARDWARE_AUDIT §3) |
| gains, gravity feed-forward, squeeze torque actually sent | no | no | `cmd_kp`, `cmd_kd`, `cmd_tau`, `squeeze_tau` |
| temperature | no | no | `t_mos`, `t_rotor` |
| external force reference (F/T, load cell, FSR) | no | no | no; needs new instruments (BATTERY_CONTACT_COLLECTION GT1–GT3) |
| RGB, 3 cameras | yes | yes | yes |
| camera frame time | nominal only | nominal only | nominal only (frame age unmeasured) |

## 2. Contact evidence obtainable from positions alone (historical data)

### 2.1 Jaw blocking: a geometric grasp indicator

When the gripper is commanded closed, the measured jaw position stalls at the object's width:

* **Pringles:** R openness 0.40–0.43 and L 0.34–0.38 → blocked in 10/10 for both arms.
* **Battery, left (charger):** 0.35–0.46 → blocked in 55/55.
* **Battery, right (AA cell):** −0.05 → blocked in **0/55**. The jaws close fully, so jaw position cannot tell
  whether the cell is held. This is a hard limit for battery grasp consistency.

What this shows is "the jaws are blocked by something of about this width". It says nothing about grip force:
the squeeze is a constant ±1.5 N·m feed-forward plus kp = 3 on J8 (HARDWARE_AUDIT §4), so the force is set by
the controller, not measured.

### 2.2 Load-induced tracking offset (battery left arm)

Residual `a[t] − s[t+2]` is the request minus the measured state 2 frames later. RMS over J1–J7, median over
episodes:

| window | right arm res / speed | left arm res / speed |
|---|---|---|
| Battery: free approach | 0.012 rad / 0.04 rad/s | 0.011 / 0.11 |
| Battery: bimanual carry (charger held, cell in hand) | 0.012 / 0.26 | **0.048 / 0.019** |
| Battery: last 1 s before right open 1 | 0.013 / 0.06 | **0.054 / 0.013** |
| Pringles: free approach | 0.012 / 0.36 | 0.011 / 0.04 |
| Pringles: right carry | 0.012 / 0.29 | 0.009 / 0.31 |
| Pringles: hand-over | 0.012 / 0.12 | 0.013 / 0.15 |

While the left arm holds the charger nearly still, its residual is 4–5× the free-motion level. This is a steady
offset, not a lag effect, because the speed is near zero.

It is **load-induced**, but it cannot be separated into:
* the charger's weight (not in the gravity feed-forward);
* URDF gravity error at the pose;
* gearbox static friction;
* any contact force from the insertion.

Order of magnitude, illustrative only: 0.05 rad × kp 20 (J5/J7) ≈ 1 N·m of PD effort, *if* the gains at
recording time were the current ones (UNKNOWN, GATE0_AUDIT U5).

### 2.3 Event-aligned kinematic residuals (candidate contact signatures)

Norm over J1–J7 of `a[t] − s[t+2]`, aligned on gripper events. The table gives the median across episodes and the
fraction of episodes in which the post-event mean exceeds the [−2.0, −0.33) s pre-event mean. Speeds are the
median joint-speed norm in the same window.

| event (n) | arm | pre: res / speed | [0, 1) s | [1, 2) s |
|---|---|---|---|---|
| Pringles left close (10) | right | 0.028 / 0.57 | 0.032 (80 %) / 0.45 | 0.041 (100 %) / **1.12** |
| | left | 0.022 / 0.74 | **0.029 (100 %) / 0.27** | 0.018 (10 %) / 0.77 |
| Battery right open 1 (54) | right | 0.032 / 0.25 | 0.029 (24 %) / 0.30 | 0.031 (35 %) / 0.19 |
| | left | 0.141 / 0.028 | 0.149 (94 %) / 0.020 | 0.155 (94 %) / 0.020 |
| Battery right close 2 (54) | right | 0.031 / 0.18 | 0.030 (39 %) / 0.22 | **0.039 (80 %) / 0.18** |
| | left | 0.146 / 0.026 | 0.155 (91 %) / 0.018 | 0.159 (89 %) / 0.019 |

The left-arm norms in this table (≈ 0.14–0.16) are much larger than the §2.2 RMS values (≈ 0.05). That is
expected: the norm sums over 7 joints and the windows differ. Compare within a table only.

Reading:

* **Pringles take-over:** in the first second after the left close, the left-arm residual rises in 10/10
  episodes while its speed falls. A residual increase with no speed increase is the pattern a pull or squeeze
  through the can would produce. Confounds: the left gripper's own closing transient and the operator's command.
  **Candidate, n = 10, unvalidated.**
* **Battery press after close 2:** over [1, 2) s the right-arm residual rises 25 % at unchanged speed in 80 % of
  episodes. This is consistent with pressing on the seated cell. Confound: a pose-dependent gravity/friction
  error as the arm moves to the press pose. **Candidate, unvalidated.**
* **Battery left arm:** its residual drifts upward through the whole hold, at open 1 as much as at close 2. It does
  not localize the insertion. It is not a contact signature at this resolution.

**Conclusion for positions-only data:**
* Contact *phase* is observable (gripper commands, jaw blocking, sustained load offsets).
* Candidate contact *events* leave kinematic traces at the 0.01-rad level.
* Contact *magnitude* and *onset timing* are not observable without an independent reference. These residuals
  must not be used as force labels or conditioning signals.

## 3. What a generated video can be checked against (no force needed)

These quantities are computable from FK + joint positions + a tracked 2D object. They are the basis of the Gate-1
metrics:

1. **Robot-pixel consistency:** FK-projected keypoints vs the robot in the frame. Today the body_cam model tracks
   the human can clicks with r = 0.97 / 0.98 and a constant offset of (−24, +87) px. The residual after removing
   that offset is 18.6 px median, 34 px p90 (50 pool episodes). The offset is expected, because the TCP at grasp
   height is not the clicked lid point and the can is pushed before the grasp. The residual is an upper bound, not
   a calibration error.
2. **Grasp rigidity:** while the jaws are blocked, the object must move rigidly with the grasping TCP.
3. **Bimanual relative pose at contact:**
   * Pringles: TCP distance 0.114–0.151 m at take-over.
   * Battery: right-in-left TCP std (0.8, 1.2, 0.6) cm at the release over the bay.
   A generated episode that moves the charger must move both hands coherently.
4. **Phase timing:** gripper events from the action stream must coincide with the visible jaw closure and object
   motion onset in the video.

## 4. Telemetry gate status

| claim | evidence | status |
|---|---|---|
| decode equals openarm_can 1.2.8; no transmit path; no change to commands or safety hold; R1–R4 fixed | offline suites, this commit | **PASS (offline)** |
| sidecar joins a real `lerobot-record` dataset bit-exactly | fake bus only | PASS (mock); real bus UNKNOWN |
| values are this tick's (S1), loopback complete (G5/G6), kernel stamps sub-ms (G4/G13), no timing impact (G10), clean bus (B1) | needs the Gate-2 bench | **UNKNOWN: never run on hardware** |
| `tau` scale, firmware PMAX/VMAX/TMAX (D1, D4) | needs a parameter query (sends frames) or known loads | UNKNOWN |
| `tau` = external force | — | **FAIL by construction** (current-based effort incl. gravity, friction, feed-forward) |

## 5. Minimum measurements before any force-conditioned claim

In order. Every item needs explicit approval and an operator. None has been run.

1. **Gate-2 passive bench** (GATE2_HARDWARE_READINESS §5–§7): A–B–A runs, no contact. Required: GO plus S1 (value
   freshness) PASS or an approved mitigation.
2. **D2 + D3:** free motion and static holds with telemetry, at the **same poses** as the contact phases, without
   contact. These give the residual noise floor; they are the negatives.
3. **D4:** known loads (0 / 0.2 / 0.5 / 1.0 kg, plus the weighed charger and cell). These give the `tau` scale per
   joint and the payload model that separates §2.2's offset from contact.
4. **D5 / GT1–GT3:** labelled contact on battery insertion (load cell under a charger fixture, or FSR in the bay,
   plus ≥ 120 fps video with an LED sync), stamped on the host clock.
5. **D6:** only if a Newton claim is wanted (F/T sensor or gauge pushes).

With 1–4, a paper may claim *contact classification and uncalibrated effort residuals*. A *force* claim also needs item 5 (D6). (Corrected 2026-11-04: an earlier version said "needs 6" for this five-item list.)
Force-conditioned *generation* additionally needs a generator that outputs a contact signal whose
distribution can be compared with the measured one. That is a Gate-3+ question.
