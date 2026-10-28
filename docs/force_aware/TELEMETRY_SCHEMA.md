# OpenArm motor telemetry: integration contract (`openarm_telemetry/1`)

The producer is `plugins/lerobot_robot_openarm_umeow/lerobot_robot_openarm_umeow/telemetry.py`. It is
enabled with `--robot.telemetry_dir=<dir>` on `--robot.type=openarm_umeow` (real arm only) and is off by default.
This file is the full contract. A consumer needs `pandas` + `pyarrow` and nothing from this repository.
The evidence behind each unit and caveat is in [HARDWARE_AUDIT.md](HARDWARE_AUDIT.md).

The policy's 16-D `observation.state` / `action` and the LeRobot dataset are **not changed**. Telemetry is a
separate sidecar on disk.

## 1. Layout

```
<telemetry_dir>/<YYYYmmdd_HHMMSS>_<pid>[_n]/     one folder per robot connect; never reused or overwritten
    meta.json                                   written at start, rewritten at close ("closed": true)
    part-00001.parquet, part-00002.parquet ...  row groups flushed every 5 s; concatenate, sort by `tick`
```

A session with `"closed": false` in `meta.json` ended in a crash. Its parts up to the last flush are valid,
and up to 5 s of rows can be missing at the end.

## 2. Rows and time basis

* **One row per `robot.get_observation()` call** (`kind` = 0). In `lerobot-record` / `lerobot-rollout` that is
  one row per control tick. Connect-time reads, the start-pose ramp, reset phases and returns also produce
  rows, because they read the robot too.
* The last row has `kind` = 1. It is written after `disconnect()` sent the disable frames, so it holds the
  wire's final word. Its `q` comes from the library state, not from an observation.
* `tick` is a 0-based counter of rows in the session. It has no gaps: a failed snapshot is still a row (§5).
* `t` and `obs_t0` are `time.perf_counter()` seconds (CLOCK_MONOTONIC). This is the same clock as the
  wrapper's `outputs/teleop_diag` logs and `eval_metrics`. `t_wall` is `time.time()` seconds (UTC epoch).
  Both are sampled together at the start of the snapshot, i.e. right after the observation's hardware read
  returned. Use `t_wall` to compare with anything outside the process.
* Kernel receive timestamps of CAN frames (CLOCK_REALTIME) are compared with `t_wall` and stored only as ages
  in seconds (`*_age_s`). An age is measured back from the row's reference time (`t_wall`).
* Each frame on the wire is counted in exactly one row: the first row whose reference time is at or after
  the frame's kernel timestamp.

### Causality inside a tick

```
row k snapshot  ->  policy/teleop computes action k  ->  send_action(action k)  ->  row k+1 snapshot
```

* `q, dq, tau, ...` on row k are the measurement the policy saw for action k.
* `cmd_*`, `sent`, `desired` on row k describe the command **in effect before** row k's measurement, i.e.
  action k-1. Read **row k+1** for what action k actually commanded.
* For residual estimation, pair row k's `tau` with row k's `cmd_*`. That command produced the torque.

## 3. Columns

Every column is `float64` except `tick` (`int64`). Order is fixed and listed in `meta.json["columns"]`:

1. `tick`, then the row fields;
2. the side fields, `L.*` before `R.*`;
3. the joint fields, joint-major in the order **LJ1..LJ8, RJ1..RJ8**, i.e. `LJ1.q, LJ1.dq, ..., LJ1.desired,
   LJ2.q, ...`.

There are 340 columns in total (1 + 9 row + 2×5 side + 16×20 joint). J1–J7 are the arm joints, proximal to distal. J8 is the gripper motor.

> The LeRobot datasets recorded by `openarm_umeow` order their 16-D vectors **interleaved**
> (`RJ1.pos, LJ1.pos, RJ2.pos, ...`; see `meta/info.json` → `features.observation.state.names`). Isaac-converted
> datasets use L-then-R. Always join by name (`LJ3.pos` ↔ `LJ3.q`), never by index.

### 3.1 Row fields

| column | unit | meaning |
|---|---|---|
| `tick` | – | row counter, 0-based, consecutive |
| `kind` | – | 0 = observation tick, 1 = final row after disconnect |
| `t` | s | `perf_counter()` at snapshot start (≈ end of the observation's hardware read) |
| `t_wall` | s | `time.time()` at the same instant; reference for every `*_age_s` |
| `obs_t0` | s | `perf_counter()` when `get_observation()` started (NaN on `kind`=1) |
| `obs_read_s` | s | duration of the hardware read: CAN refresh + drain + camera `async_read` (NaN on `kind`=1) |
| `snapshot_ok` | 0/1 | 0 = the snapshot raised; joint and side columns are then NaN (§5) |
| `snapshot_s` | s | cost of this snapshot inside `get_observation()` (logging overhead) |
| `hold` | 0/1 | wrapper safety hold active (`_fault`), i.e. the last command was a hold re-send |
| `motor_fault` | 0/1 | wrapper's motor watchdog reports a DM protection fault |

### 3.2 Side fields (`L.<field>`, `R.<field>`; one CAN interface per arm)

| column | unit | meaning |
|---|---|---|
| `tap_ok` | 0/1 | the receive-only tap on this arm's interface is open. 0 makes every wire column of that arm NaN |
| `rx_dropped` | frames | cumulative frames the kernel dropped from the tap's queue (SO_RXQ_OVFL). Must stay 0 |
| `bus_errors` | frames | CAN error frames raised by the controller since the previous row |
| `ctrl_frames` | frames | enable / disable / **set-zero** frames seen on the command ids since the previous row. Expected 0 except the final row |
| `squeeze_tau` | N·m | gripper squeeze feed-forward included in J8's `cmd_tau`, signed in motor convention; 0 unless the gripper is commanded closed |

### 3.3 Joint fields (`<J>.<field>`, J ∈ LJ1..LJ8, RJ1..RJ8)

Measured, from openarm_can's retained `Motor` state (the same getters as `get_motor_health()`):

| column | unit | meaning |
|---|---|---|
| `q` | rad | position the policy saw: the observation's `<J>.pos`. Motor-side angle at the output, the motors' stored zero. **Not** URDF joint angle (map through `calibration.json`). Grippers: raw motor angle |
| `dq` | rad/s | motor-reported velocity |
| `tau` | N·m | motor-reported torque. **The DM firmware's own estimate, not a calibrated joint torque** (HARDWARE_AUDIT §3). It includes gravity, friction, inertia and the controller's effort |
| `t_mos` | °C | driver MOSFET temperature (integer). 0 = the channel has never answered |
| `t_rotor` | °C | rotor temperature (integer) |
| `status` | code | DM status nibble of the latest reply: 0 disabled, 1 enabled, 8 overvoltage, 9 undervoltage, 0xA overcurrent, 0xB MOS overtemp, 0xC rotor overtemp, 0xD lost comms, 0xE overload |

Freshness, from the wire (NaN when `tap_ok` = 0):

| column | unit | meaning |
|---|---|---|
| `rx_n` | frames | state frames this motor sent since the previous row (refresh replies + MIT-command replies). 0 = nothing arrived |
| `rx_age_s` | s | age of the newest state frame received (NaN if `rx_n` = 0) |
| `state_age_s` | s | age of the information in `q, dq, tau`: the newest received frame whose decode equals the library's state (to 1e-9). NaN = untraceable |
| `state_fresh` | 0/1 | 1 = that frame arrived after this motor's latest refresh request, i.e. it answers this tick's read. 0 = it is an older frame, e.g. the reply to the previous MIT command. NaN = unknown |
| `stale` | 0/1 | **1 = do not trust this row's `q/dq/tau` for this motor**: `state_age_s` > `meta.stale_s` (0.05 s), or untraceable. 0 = traced and recent. NaN = no tap, so unknown |

Commanded, decoded from the MIT frame actually transmitted. This is after openarm_can's clamping and
quantization, i.e. what the motor received. NaN until the first command:

| column | unit | meaning |
|---|---|---|
| `cmd_q` | rad | position target |
| `cmd_dq` | rad/s | velocity feed-forward. Arms: the wrapper's `compute_target_velocity`. Grippers: always 0 |
| `cmd_kp` | N·m/rad | stiffness gain |
| `cmd_kd` | N·m·s/rad | damping gain |
| `cmd_tau` | N·m | torque feed-forward = pinocchio gravity at the **commanded** q (+ `squeeze_tau` on J8) |
| `cmd_n` | frames | MIT frames sent to this motor since the previous row (1 per `send_action`) |
| `cmd_age_s` | s | age of the latest MIT frame |

Host side, from the `openarm_umeow` wrapper before encoding:

| column | unit | meaning |
|---|---|---|
| `sent` | rad | the last command the wrapper executed: after the step limit, or the held pose during a hold |
| `desired` | rad | the last request it received (teleop / policy action, before the step limit). Equals the dataset's `action` |

The motor's own control law, which the `cmd_*` columns parameterize, runs on the motor at its internal
rate:

```
tau_motor = cmd_kp * (cmd_q - q) + cmd_kd * (cmd_dq - dq) + cmd_tau      (saturated at ±tMax)
```

## 4. Resolution (decode limits in `meta.json["decode_limits"]`)

| motor (joints) | `q` LSB | `dq` LSB | `tau` LSB | tau range |
|---|---|---|---|---|
| DM8009 (J1, J2) | 3.8e-4 rad | 0.022 rad/s | 0.026 N·m | ±54 |
| DM4340 (J3, J4) | 3.8e-4 rad | 0.0039 rad/s | 0.014 N·m | ±28 |
| DM4310 (J5–J7, J8) | 3.8e-4 rad | 0.015 rad/s | 0.0049 N·m | ±10 |

Commands: `cmd_kp` LSB 0.122, `cmd_kd` LSB 0.00122. Values are truncated toward the range minimum, not
rounded.

## 5. Missing, stale and failed values

| situation | encoding |
|---|---|
| tap could not open (`tap_ok` = 0) | that arm's wire columns (`rx_*`, `state_*`, `stale`, `cmd_*`, `rx_dropped`, `bus_errors`, `ctrl_frames`) are NaN. Measured columns are still filled |
| motor never answered | `t_mos` = `t_rotor` = 0, `state_age_s` NaN, `stale` = 1 |
| motor stopped answering | `rx_n` = 0. `state_age_s` keeps growing from the last traced frame, and `stale` becomes 1 once it exceeds `stale_s`. `q/dq/tau` repeat the last value |
| snapshot raised | `snapshot_ok` = 0. Row fields kept, every side/joint column NaN. The count is in `meta.json["errors"]["snapshot"]` |
| write failed | rows of that flush are lost (a gap in `tick`). The count is in `meta.json["errors"]["writer"]` |
| no command yet / host value absent | `cmd_*` / `sent` / `desired` NaN |

NaN always means **unknown**, never zero. Do not `fillna(0)` freshness columns.

`state_age_s` traces the *information*: if an older frame and a newer frame decode identically (a joint at
rest), the value is attributed to the newer one. That is correct for the value but can hide a library that
lags by one frame. `state_fresh` = 0 shows when it actually happens.

## 6. `meta.json`

| key | content |
|---|---|
| `schema` | `"openarm_telemetry/1"`. A change of columns or meaning bumps it |
| `columns`, `joints`, `row_fields`, `side_fields`, `joint_fields` | the exact layout above |
| `stale_s` | the threshold behind `stale` |
| `clock` | `{perf_counter, time}` sampled together at start, mapping `t` → `t_wall` |
| `ports`, `can_ids`, `motor_types`, `decode_limits` | wire mapping and per-joint (pMax, vMax, tMax) used to decode |
| `gains` | the follower's kp / kd arrays at connect (J1..J7, J8 = gripper) |
| `gripper_squeeze_tau_config`, `max_joint_speed` | wrapper settings |
| `tap_available` | per arm, at start |
| `provenance` | `git_sha`, `git_dirty`, `openarm_can` version, URDF path + sha256 (gravity model), `calibration.json` path + sha256, `torque_calibration` (currently `null`: nothing calibrates `tau`) |
| `closed`, `rows`, `parts`, `errors`, `max_snapshot_s`, `ctrl_frames` | written at close. `ctrl_frames` lists every enable/disable/set-zero frame with time and joint |

**Calibration provenance rule:** a consumer that uses `tau` quantitatively must check
`provenance.torque_calibration`. While it is `null`, `tau` is uncalibrated and comparable only within a
session with the same `motor_types`, `decode_limits` and `gains`.

## 7. Aligning with a LeRobot episode

LeRobot stores no wall-clock time per frame. Alignment is by value: the episode's `observation.state` is a
run of consecutive `kind`=0 rows whose `q` equals it **bit-for-bit in float32**.

```python
from lerobot_robot_openarm_umeow.telemetry import load, align_to_dataset   # or copy align_to_dataset (15 lines)
df, meta = load(session_dir)
start = align_to_dataset(df, states, names)   # states: (T,16) observation.state of ONE episode; names from info.json
rows = df.iloc[start:start + T]               # rows[k] is frame k; df.iloc[start + k + 1] holds action k's command
```

`align_to_dataset` raises if no window matches or more than one matches. An episode where no joint
changes, e.g. a perfectly static hold, can be ambiguous and is then rejected rather than guessed. Checked
end to end by `plugins/tests/test_telemetry_record.py`. Each dataset frame matched exactly one window,
`action[k]` == `desired` of row k+1, and `cmd_q` == `sent` within one LSB.

## 8. Not in this schema

* Supply voltage and phase current: the DM state frame does not carry them.
* Joint acceleration: differentiate `dq` or `q` yourself, mind the 30 Hz rate.
* Camera frame timestamps.
* Simulation (`openarm_isaac`): telemetry is never started there.
