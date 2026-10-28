# OpenArm motor feedback audit: is the telemetry valid for sensorless contact research?

Scope: the dual-arm OpenArm follower as driven by `--robot.type=openarm_umeow` (lerobot-record /
lerobot-rollout / lerobot-teleoperate). Audited at `9abbe126128ce17f66c98c726688dac70b84a966` (branch
`official-lerobot-quest`). The work is on `force-aware/telemetry-gate1`.

**Method:** source reading plus offline tests only. No CAN socket was opened and no robot was connected,
moved, enabled, re-zeroed or re-gained. Every claim is tagged:

* **[SRC]**: read in source (file:line).
* **[TEST]**: verified by an offline test in this branch.
* **[UNKNOWN]**: needs the Gate-1 bench check in §8.

## 1. Signal chain

```
policy/teleop action (16 x motor rad)
  -> OpenArmUmeow.send_action        clamp_step (max_joint_speed*dt), safety guard, squeeze, compute_target_velocity
  -> OpenArmFollower.send_action     pinocchio gravity tau at the COMMANDED q  ->  MITParam(q, dq, tau, kp, kd) x16
  -> openarm_can 1.2.8 (C++)         clamp + quantise -> 8-byte DM MIT frame on 0x01..0x08 (per arm interface)
  -> DM motor                        tau_motor = kp(q*-q) + kd(dq*-dq) + tau_ff  ->  replies with a state frame on 0x11..0x18
  <- OpenArmFollower._read_motor_positions_once
                                     refresh_all() (0x7FF per motor) -> drain recv_all() until the tap says all answered
  <- openarm_can Motor               ONE retained (q, dq, tau, t_mos, t_rotor); every state frame overwrites it
  <- get_observation()               returns q only (16 x .pos); get_motor_health() exposes the rest
```

* Two CAN-FD interfaces on one USB adapter: right arm `can0`, left arm `can1`. The follower staggers the
  two arms' requests by 2 ms (`_REQUEST_STAGGER_S`, `openarm_follower.py:509`). [SRC]
* Motors: J1–J2 DM8009, J3–J4 DM4340, J5–J7 + gripper J8 DM4310, all in MIT mode
  (`config_openarm_follower.py`). [SRC]
* The installed `openarm_can` is 1.2.8. Its decoder and MIT encoder match this repo's re-implementation
  **bit-for-bit** on 1 600 random frames per codec (`test_telemetry.py` check 1). [TEST]
  * `/usr/local/lib/libopenarm_can.a` is 1.2.2. The Python binding's CMake asks for ≥ 1.2.8, so it was built
    against upstream 1.2.8. The decoder source below was read in the local 1.2.2 checkout, and the test
    above confirms the installed binary decodes the same way.

## 2. What each quantity is

| quantity | source | unit | decode (state frame byte layout) | resolution | measured / commanded |
|---|---|---|---|---|---|
| `q` | `Motor.get_position()` | rad, output side, motor's stored zero | `data[1:3]` 16 bit over ±pMax (12.5) | 3.8e-4 rad | measured |
| `dq` | `Motor.get_velocity()` | rad/s | 12 bit over ±vMax (45 / 8 / 30) | 0.022 / 0.0039 / 0.015 rad/s | measured |
| `tau` | `Motor.get_torque()` | N·m (nominal) | 12 bit over ±tMax (54 / 28 / 10) | 0.026 / 0.014 / 0.0049 N·m | motor **estimate** (§3) |
| `t_mos`, `t_rotor` | `get_state_tmos/trotor()` | °C, integer | `data[6]`, `data[7]` | 1 °C | measured |
| status nibble | **not** decoded by openarm_can. `data[0] >> 4`, read only by `can_monitor.py:165` | code | 0 dis, 1 en, 8–E faults | – | motor-reported |
| `is_enabled()` | `Motor` | bool | never set by the library | – | **dead, always False** (`openarm_follower.py:803`) |

Decode: `CanPacketDecoder::parse_motor_state_data` (`dm_motor_control.cpp`), using the limits table
`MOTOR_LIMIT_PARAMS`. The 1.2.8 binding reports identical limits through `Motor.get_limit_param`. [SRC][TEST]

The values are valid feedback in the sense that they are exactly what the motor reported, decoded
correctly. Two things about them are not established:

* The firmware's own PMAX/VMAX/TMAX registers must equal the library's table, or every `dq`/`tau` is
  scaled wrong. [UNKNOWN]
* What `tau` physically is. See §3.

## 3. `motor.get_torque()` is not an external joint torque

* `tau` is whatever the DM firmware puts in bits 36–47 of its state frame. Per Damiao's protocol this is
  derived from the measured q-axis current, scaled by the firmware's torque constant and gear ratio. There is
  no torque sensor in the joint. Nothing in this repository calibrates it: no load-cell fit, no known-weight
  test, no current-to-torque identification. **[UNKNOWN: scale and offset per joint.]**
* What it contains, even if the scale is perfect:
  `tau_meas ≈ tau_motor = kp·(q* − q) + kd·(dq* − dq) + tau_ff` (until saturation). This is the controller's
  **effort**. It sums gravity, gearbox friction (DM8009/4340 planetary, large Coulomb + hysteresis), inertia,
  cable drag, and only then contact. External force enters only as the tracking error it causes, times kp.
* Effective sensitivity therefore depends on the gains (`openarm_follower.py:65-66`):
  `kp = [60, 50, 50, 60, 20, 40, 20, 3]`, `kd = [2, 2, 2, 2.5, 1, 1.2, 1, 0.3]`. At kp = 20 (J5, J7), a 0.1 N·m
  contact torque is a 5 mrad error, i.e. 13 LSB of `q`. That is resolvable, but the same order as gearbox
  backlash.
* Sign convention: motor convention, `+tau` drives `+q`. The calibration signs are all `+1` for both arms
  (`calibration.json`), so motor q = URDF joint angle. This was verified by the earlier sign probes, not here.

## 4. Commanded quantities and confounders

| term | where | why it confounds contact estimation |
|---|---|---|
| step-limited target `q*` | `openarm_umeow.py:340` `clamp_step` | the motor tracks the clamped target, not the policy's request. Dataset `action` is the request |
| velocity feed-forward `dq*` | `openarm_umeow.py:343` `compute_target_velocity` (arms); **0 for grippers** (`openarm_follower.py:944, 979`) | `kd·dq*` is effort that is not contact |
| gravity feed-forward | `openarm_follower.py:919-920`: `pin.computeGeneralizedGravity` at the **commanded** q, clipped to URDF limits | `tau_ff` ≠ gravity at the measured q. During lag or contact the residual carries `G(q_meas) − G(q_cmd)`. URDF inertials are unidentified (and legacy `record.py` used `openarm_description_leader.urdf`) |
| gripper squeeze | `openarm_umeow.py:445` `_apply_squeeze`: ±1.5 N·m constant while the gripper is commanded closed | J8 `tau` reflects the squeeze, not an external force. Wrist joints see the reaction of whatever is grasped |
| safety hold | `openarm_umeow.py` `send_action` hold path re-sends `_last_sent` | contact against a held command looks like contact against a stationary target |
| quantisation | openarm_can encoder truncates: kp 0.122, kd 0.0012, tau_ff 0.026 N·m (DM8009) | small, documented |
| temperature | `t_mos`/`t_rotor` | torque constant drifts with rotor temperature (NeuralActuator models T explicitly) |

The `cmd_*` columns of the new telemetry capture the *transmitted* `q*, dq*, kp, kd, tau_ff`, and
`squeeze_tau` separates the squeeze. Every term in the table can therefore be subtracted per tick
without re-deriving it on the host.

## 5. Sampling, timestamps and freshness

**Rates.** One `get_observation()` per control tick (30 Hz in lerobot-record). Each tick produces, per motor:

* one refresh request and its reply;
* one MIT command and its reply;
* plus re-asks after `recv_retry_us` = 3 ms if a reply is missing (`config_openarm_follower.py:88`).

The read waits up to `recv_deadline_us` = 8 ms (`:79`). [SRC] Motor-side sampling of `tau` (current loop) is
much faster, but we see one value per frame.

**Timestamp sources, before this branch:**

* lerobot dataset `timestamp` = `frame_index / fps`: nominal, never measured.
* Wrapper `perf_counter()` in DIAG / eval logs.
* **None** on the motor value itself: `Motor` stores no time and no sequence number, and both reply types
  overwrite the same fields (`dm_motor_device.cpp`, STATE callback). [SRC]

**Freshness, before this branch.** `CanFeedbackMonitor` (`can_monitor.py`) counts frames per channel since
`mark_cycle()`, but it only drains its socket inside `poll()`, which runs only within a read. Between
`send_action()` and the next read nothing polls it. So the 16 MIT-command replies of the previous tick sit
unread in its queue, and the first `poll()` after `mark_cycle()` (`openarm_follower.py:763`) counts them as
answers to the new refresh.

Hypothesis: `_drain_until_answered` can therefore end early, leaving some `Motor` states at the previous
command's reply. That value is up to one tick (~33 ms) old, and `get_observation()` returns it as fresh.
**[UNKNOWN, plausible from source: whether it happens, and how often, depends on USB timing]**. The new
telemetry measures it directly: `state_fresh` = 0 and `state_age_s` ≈ one tick
(`test_telemetry.py` check 5 reproduces the case on the fake bus). [TEST]

**Synchronization errors to expect:**

1. The value-age issue above (0–33 ms, per joint).
2. Right vs left arm: 2 ms request stagger.
3. Cameras: `async_read()` returns the latest frame, of unmeasured age.
4. Command vs measurement: action k is sent after row k is read. Its effect is first visible in row k+1.
5. RTC / async rollout: actions are computed on older observations (measured in `eval_metrics`).

## 6. `get_motor_health()` (reused, unchanged)

`openarm_follower.py:777-815` reads retained `Motor` state with no CAN traffic of its own. It returns
`torque, velocity, position, t_mos, t_rotor` plus the tap's status nibble and cumulative frame totals. It is
correct for what it claims, with three caveats:

* `enabled` is dead.
* `frames` is cumulative since connect, so it says nothing about the current value.
* There is no timestamp.

The telemetry sidecar reads exactly the same getters through `_motor_channels()`. It uses
`get_feedback_status()` as the status fallback when its own tap is down.

## 7. What was built, and what it touches

`telemetry.py` is a receive-only CAN socket per interface (kernel timestamps, `SO_RXQ_OVFL` drop counter,
error frames). On each `get_observation()` it:

* reads the `Motor` getters;
* drains its socket;
* traces each `Motor` value to the wire frame it came from (`state_age_s`, `state_fresh`, `stale`);
* decodes the library's own looped-back MIT frames (`cmd_*`: the command actually applied, including the
  commanded velocity);
* counts enable/disable/set-zero frames;
* hands a row to a background Parquet writer.

Contract: [TELEMETRY_SCHEMA.md](TELEMETRY_SCHEMA.md).

Production files changed, all inert unless `--robot.telemetry_dir` is set:

| file | change |
|---|---|
| `config_openarm_umeow.py` | `telemetry_dir: str = ""` |
| `openarm_umeow.py` | `_telemetry = None` in `_init_umeow`; start after `OpenArmFollower.connect` in `_hw_connect`; one `snapshot()` after the DIAG row in `get_observation`; close after `OpenArmFollower.disconnect` in `_hw_disconnect` (inside `finally`, its own exceptions logged, never raised) |

Not changed:

* `openarm_follower.py`, `can_monitor.py`, `sim_bridge_common.py`;
* every gain, guard, ramp and shutdown path;
* the 16-D observation and action;
* `openarm_isaac` (it overrides `_hw_connect`, so telemetry never starts in simulation).

Safety properties, tested offline:

| property | evidence |
|---|---|
| no transmit path: no `send*` call in the module, no send method on `WireTap` | check 11 [TEST] |
| during `get_observation` + snapshot the arm objects see only getter calls (no `refresh_all`, `recv_all`, `mit_control_all`, `enable_all`, `disable_all`) | check 14 [TEST] |
| commands sent and the safety hold are identical with telemetry on and off | check 13 [TEST] |
| a failing `Motor` getter or writer never raises into the loop; the row is marked | check 8 [TEST] |
| snapshot cost with a full tick of traffic | check 12 [TEST]; see the test log for the number |
| the existing mock suites still pass | `test_robot_guard`, `test_return_lag`, `test_quest_safety`, `test_teleoperate_mock`, `test_record_mock`, `test_record_gate`, `test_rerun_status` [TEST] |

The extra socket's cost on the real bus is a kernel copy of frames that already arrive. This is the same
mechanism `can_monitor.py` has used on every connect since 2026-08-26.

## 8. Evidence summary and Gate-1 checks

| # | claim | status |
|---|---|---|
| 1 | Telemetry decode is bit-identical to installed openarm_can 1.2.8 | **PASS** [TEST] |
| 2 | `q, dq, tau, t_mos, t_rotor` come from the motor's state frame; units rad, rad/s, N·m (nominal), °C | **PASS** [SRC] |
| 3 | `is_enabled()` is meaningless; status must come from the raw nibble | **PASS** [SRC] |
| 4 | `tau` is a calibrated external joint torque | **FAIL**: it is a current-based effort estimate including gravity, friction, inertia and feed-forward |
| 5 | Firmware PMAX/VMAX/TMAX equal the library's decode table | **UNKNOWN**: read registers 21/22/23 (`oa.PMAX/VMAX/TMAX`) per motor |
| 6 | `tau` scale/offset per joint | **UNKNOWN**: needs a known-load test |
| 7 | Every observed value is from this tick's refresh | **UNKNOWN / suspected FAIL** (§5); measurable with `state_fresh` |
| 8 | A passive tap on the bus sees the library's MIT frames via loopback | **UNKNOWN** on this adapter. Kernel default is on and openarm_can does not disable it (`can_socket.cpp`). Expect `cmd_n` = 1 per tick |
| 9 | Kernel rx timestamps are monotonic and sub-ms on gs_usb | **UNKNOWN** |
| 10 | Telemetry adds no CAN traffic and changes no command | **PASS** [TEST, mocks]; confirm on bench with `tx_packets` unchanged vs telemetry off |
| 11 | Snapshot overhead fits the 33 ms tick | **PASS** [TEST, fakes]; confirm `snapshot_s` on hardware |

### Proposed Gate-1 bench run (requires explicit approval; nothing below has been run)

1. **Passive only, arm at rest, operator present.** Run a normal `lerobot-teleoperate` or
   `lerobot-record` session (the existing approach ramp) with `--robot.telemetry_dir=outputs/telemetry`,
   about 60 s, no contact. Report:
   * `L/R.tap_ok`, `rx_dropped` = 0;
   * `cmd_n` = 1 per tick (item 8);
   * the `state_fresh` = 0 rate per joint (item 7);
   * `snapshot_s` p99 (item 11);
   * `ctrl_frames` = 0 except the final row;
   * `tx_packets` delta per tick vs a run without telemetry (item 10).
2. **Read-only register query** of PMAX/VMAX/TMAX/KT_Value/Gr per motor (`query_param`). This sends
   frames, so it needs its own approval, and only with motors disabled (item 5).
3. Only after 1–2: a free-motion and known-weight session for the residual baseline (CONTACT_FEASIBILITY §4).
