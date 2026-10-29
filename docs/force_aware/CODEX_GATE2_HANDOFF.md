# Handoff to Codex, Gate 2A: finalized telemetry contract and mock fixture

**From:** `StanleyChueh/lerobot_openarm`, branch `force-aware/gate2a-bench-prep`. The commit SHA is in the
delivery note. **For:** `StanleyChueh/lerobot`, branch `research/force-aware-offline-audit` @ `eb5990f`.

**Status:**

* No real-robot telemetry exists yet. Every file referenced here is offline or mock.
* The branches are **local** to the lab PC until the owner pushes them. Your `MODEL_INTEGRATION_DESIGN.md`
  says `TELEMETRY_SCHEMA.md` was absent because you inspected `9abbe12`; it lives on this branch.

## 1. The schema is final for Gate 2: `openarm_telemetry/1` + `openarm_frames/1`

* The **row table** is `part-*.parquet`: 340 columns, one row per `robot.get_observation()`. It is frozen;
  the full contract is [TELEMETRY_SCHEMA.md](TELEMETRY_SCHEMA.md). Gate 2A changed no column and no meaning.
  `rx_dropped` now also counts host-side overflow, and must still be 0.
* The **raw frame log** is `frames-*.parquet` (`openarm_frames/1`), additive: every CAN frame the receive-only
  tap saw (`tick, side, seq, can_id, data, t_ns`). You don't need it for features. It exists so derived
  columns can be audited.
* A change of columns or meaning bumps `/1` to `/2`. `plugins/tests/test_telemetry_fixture.py` fails if
  the producer drifts from the committed fixture.
* The **observation and action interface is unchanged**: 16-D float32, names `RJ1.pos, LJ1.pos, …, LJ8.pos`
  (interleaved). Telemetry is a sidecar on disk; nothing enters `observation.state`.

## 2. Answers to the open items in your `MODEL_INTEGRATION_DESIGN.md`

| your item | answer |
|---|---|
| hardware joint IDs | `LJ1..LJ8`, `RJ1..RJ8`; J1–J7 arm joints, proximal to distal; J8 = gripper motor. CAN ids in `meta.can_ids`; motor types per joint in `meta.motor_types` |
| gripper semantics | `J8.q` = **raw gripper motor angle in rad**, motor-side, the motors' stored zero; not metres, not normalized. Your fixture's "all-radian grippers" happens to be right, but declare it from `meta`, not by default |
| units | `q` rad, `dq` rad/s, `tau` **N·m nominal**, `t_*` °C (integer), ages s. Per-joint decode ranges in `meta.decode_limits` |
| raw vs calibrated effort | `tau` is the DM firmware's **current-based effort estimate**, uncalibrated, including gravity, friction, inertia and the controller's PD effort. `meta.provenance.torque_calibration` is `null`; refuse any Newton or "external torque" interpretation while it is |
| saturation flags | not a column. Derive from `meta.decode_limits[j]`: `tau` saturated if \|tau\| ≥ tMax − 1 LSB (LSB = 2·tMax/4095); command clamped if \|cmd_q\| ≥ pMax − 1 LSB or \|cmd_tau\| ≥ tMax − 1 LSB |
| timestamps, clocks, drift | `t`, `obs_t0` = `perf_counter()` (CLOCK_MONOTONIC; same clock as `eval_metrics` and teleop DIAG); `t_wall` = CLOCK_REALTIME; both sampled together per row. Kernel frame stamps are REALTIME, stored as ages back from `t_wall`. `meta.clock` maps the two clocks at start; the Gate-2 analyzer bounds the drift of `t_wall − t` at 2 ms per session |
| acquisition and arrival | per joint: **acquired** = `t − state_age_s` (perf clock); **available** to the policy = `t`. Arrival latency = `state_age_s`. `obs_read_s` = how long the hardware read took |
| sequence numbers | `tick` (rows, gap-free; a gap means a lost write, counted in `meta.errors.writer`); `frames.seq` per arm (wire delivery order) |
| per-channel freshness | per joint: `rx_n`, `rx_age_s`, `state_age_s`, `state_fresh`, `stale` (`stale` = 1 → do not use this joint's `q/dq/tau` on this row). `meta.stale_s` = 0.05 s is the threshold |
| calibration revisions | `meta.provenance`: `git_sha`, `git_dirty`, `openarm_can` version, URDF path + sha256, `calibration.json` path + sha256, `torque_calibration`; plus `meta.gains`, `meta.decode_limits` |
| episode identity | telemetry has none. A dataset episode is joined by value with `align_to_dataset` (§4), then labeled with your globally qualified id, e.g. `ethanCSL/<repo>@<revision>/<episode_index>` |
| label frames | not part of telemetry. Contact labels come from the external sensors in [BATTERY_CONTACT_COLLECTION.md](BATTERY_CONTACT_COLLECTION.md) §3, stamped on the host REALTIME clock. Join rule: an event at time T belongs to the first row with `t_wall ≥ T` |

## 3. Mapping onto your provisional `Packet`

For dataset frame *k*, aligned to telemetry row *r = start + k*:

| `Packet` field | from telemetry |
|---|---|
| `episode` | your qualified id of the aligned episode |
| `acquired_s` | `row.t − row.<J>.state_age_s`: **per joint**. A single packet time should be the oldest joint's, i.e. `t − max_j state_age_s` |
| `available_s` | `row.t` |
| `velocity`, `effort` | `<J>.dq`, `<J>.tau`, joined **by name** to your target order |
| units | as in §2 |
| validity | invalid if `snapshot_ok` ≠ 1, `tap_ok` ≠ 1 for that arm, `<J>.stale` ≠ 0, or the value is NaN. NaN always means unknown: never `fillna(0)` |
| observation time of frame k | `row r .t` |
| action k's target | row **r + 1** holds what action k commanded (`desired` = the dataset action, `sent` = after the step limit, `cmd_*` = what the motor received). Row r's `cmd_*` belong to action k − 1, the command that produced row r's `tau` |

Every field on row r was sampled no later than row r's snapshot, so nothing on row r is future information
for action k. Row r + 1 is future: use it only as a supervision target, never as an input.

## 4. Mock fixture for your integration tests

`plugins/tests/fixtures/openarm_telemetry_1/` (about 450 kB) was produced by the **production writer** on the
fake DM bus with `make_telemetry_fixture.py`. **It is mock data**: `meta.mock_fixture` says so.

```
session/meta.json, part-0000{1,2}.parquet, frames-0000{1,2}.parquet   one telemetry session (117 rows)
dataset/meta/info.json, dataset/data/chunk-000/file-000.parquet       LeRobot v3.0 tabular part, 2 x 45 frames
expected.json                                                          what a consumer must find
```

Scripted shape:

* connect read → start-pose ramp (commands, no reads) → reset loop with **one failed snapshot** (tick 6);
* **episode 0** (start tick 13) → 0.4 s save pause → reset;
* **episode 1** (start tick 70), with:
  * **RJ5 losing two replies** (stale at tick 81);
  * a **6-tick safety hold** (ticks 90–95: `sent` frozen, `desired` moving);
  * the right gripper squeezing;
* → return ramp → **disable frames** → final row (tick 116).

Contract tests worth porting (pandas + pyarrow only, no import from this repo):

1. load the parts, sort by `tick`, and check columns == `meta.columns`, `tick` int64, everything else float64;
2. align both episodes by float32 equality of `q` to `observation.state` (copy `align_to_dataset`, 15
   lines) → starts `[13, 70]`;
3. `action[k]` == `desired` on row start + k + 1;
4. RJ5 invalid at tick 81 only;
5. hold ticks 90–95 present but still valid measurements;
6. the failed-snapshot row (tick 6) all NaN and invalid;
7. no packet from the final row (`kind` = 1);
8. your joiner refuses the fixture's `tau` as calibrated force (`torque_calibration` null).

`expected.json` also holds `tools/force_aware/telemetry_check.py`'s verdict on the fixture. It is NO-GO by
design: G1 fails on the injected snapshot failure, and S1/S4 report RJ5. That verdict is a regression anchor,
not a quality claim.

## 5. What is still unknown on real hardware

All of these are pending the Gate-2 bench run ([GATE2_HARDWARE_READINESS.md](GATE2_HARDWARE_READINESS.md) §8):

* kernel timestamp quality and echo order on gs_usb;
* loopback completeness;
* whether observed values are this tick's (S1);
* the real `tau` scale.

Until a real session passes the Gate-2 analyzer, do **not** train on or tune against telemetry. Synthetic and
fixture tests only. GPU training stays out of scope.

## 6. Do not

* index telemetry by position or assume the dataset's order (it is interleaved R/L; telemetry is L then R);
* treat `stale` = NaN or `state_fresh` = NaN as fresh;
* use row r + 1 as an input for frame r;
* claim force in N while `torque_calibration` is null;
* copy the fixture's values as hardware defaults: only its **format** is real.
