# Gate 2A: telemetry review and hardware bench readiness

**Branch:** `force-aware/gate2a-bench-prep`, from `force-aware/telemetry-gate1` @ `8377bbf`. The commit SHA is
in the delivery note and `git log`. **Status:** offline work done; the bench session below has **not** been run
and needs its own explicit approval.

**Decision:** **GO** for the operator-supervised Gate-2 bench test, *if* every prerequisite in §5 holds.
**NO-GO** for contact collection ([BATTERY_CONTACT_COLLECTION.md](BATTERY_CONTACT_COLLECTION.md)) until the
bench report says GO.

Nothing in this phase connected to the robot, moved it, queried motor parameters or changed a gain, zero,
guard or dataset. The tests use fake taps and scratch folders. The only kernel socket opened is a UDP
loopback socket in test 16.

## 1. Gate-1 review: what could have invalidated real telemetry

| # | finding | evidence | severity | status |
|---|---|---|---|---|
| R1 | `WireTap.read()` polled with `select()`, which raises `ValueError` for a file descriptor ≥ 1024. `ulimit -n` on this PC is 1 048 576, so a recording process with cameras, encoders and threads can get such a descriptor. Every snapshot would then fail: all rows `snapshot_ok`=0, no wire data. | Reproduced on a real socket at fd 1500. Check 16. | **critical, if triggered** | fixed: non-blocking `recvmsg()` only |
| R2 | `sim_bridge_common.ramp_to` (connect-time start-pose approach, ~5 s; return to rest at disconnect) sends at 50 Hz **without** calling `get_observation()`, so nothing drains the tap. Its kernel queue (`rmem_max` 212 992, so `SO_RCVBUF` ≈ 426 kB, a few hundred frames) fills in under a second. The kernel then drops the **newest** frames. Effects: (a) the disable frames at shutdown are lost, so Gate 1's claim "the final row shows the disable frames" fails on hardware; (b) `rx_dropped` > 0 in every session, so "must stay 0" would false-alarm; (c) the ramp's TX accounting is impossible. | Reproduced with a capacity-bounded fake tap that drops like the kernel: 468 dropped, 0/8 disable frames. Check 18. | **high** (evidence loss, not control) | fixed: the background thread drains the taps whenever no snapshot ran for `POLL_S` = 0.1 s. It never touches the taps between 30 Hz ticks. |
| R3 | A frame without a kernel timestamp was stamped with the host clock **at read time**. That is after the row's reference time, so the frame moved to the next row: every value became `stale`=1, with `state_fresh` unknown. SocketCAN normally supplies the stamp, so this is latent. | Reproduced: a UDP socket without `SO_TIMESTAMPNS`, plus a fake that returns no stamps. Check 17. | medium (latent) | fixed: stamped at the row's reference time and **counted** (`meta.taps.*.untimed`); bench check G4 fails on any |
| R4 | The row table alone cannot prove the Gate-2 claims: delivery order of command echo vs reply, completeness of loopback, and "telemetry transmits nothing" on the real interface. | design gap | high for Gate 2 | added: every tapped frame verbatim in `frames-*.parquet` (`openarm_frames/1`), plus the interface's kernel counters (`/sys/class/net/<if>/statistics`, `ip -details -statistics link`) in `meta.json` at start and close. Host reads only, nothing on the bus. |
| R5 | `close()` and the writer thread could flush concurrently if a write took more than 6 s. | code reading; not reproduced | low | serialized (`_flush_lock`) |

Reviewed and found sound (no change):

* **Clock basis.** Kernel stamps and `t_wall` are both CLOCK_REALTIME, and an age never mixes in
  `perf_counter`. A wall-clock step would corrupt ages, so G4 bounds the spread of `t_wall − t` at 2 ms.
* **Freshness logic.** The trace matches a value to its frame, then checks `state_fresh` against this motor's
  last refresh echo. It is correct if the adapter delivers each echo before the reply it provokes. That
  order is **unknown** on gs_usb; G13 measures it from the frames log.
* **CAN loopback.** `ip link` shows `ECHO` (IFF_ECHO) on can0/can1: the driver loops each transmitted frame
  back on TX completion. So the command frames should be visible. G5 and G6 confirm or refute this.
* **Missing Parquet rows.** `lerobot-record` calls `robot.disconnect()` in `finally`, so Ctrl-C still closes
  the session. Only SIGKILL or a power cut loses up to 5 s. Ticks are contiguous, so any loss is visible.
* **Writer cost.** A 5 s flush of 150 rows takes 5 ms. The worst stall it caused in a thread sleeping 0.5 ms
  was 1.2 ms. Snapshot median is 0.094 ms with the frames log (fakes); the first snapshot after a long ramp
  ingests the ramp's backlog (a one-off spike outside episodes; G8 bounds the loop rows separately).
* **Episode alignment.** The value-based match is unique per moving episode. The analyzer now also compares
  the last action with the row after the episode, which may be the final row.

### Production code changed in Gate 2A

Only `telemetry.py`. Since Gate 1, `openarm_umeow.py` and `config_openarm_umeow.py` are unchanged; so are the
follower, `can_monitor.py`, every gain, guard, ramp, the shutdown path and the 16-D observation and action.
Check it with:

```bash
git diff 8377bbf -- plugins/lerobot_robot_openarm_umeow robots   # only telemetry.py
```

`openarm_telemetry/1` keeps its 340 columns and their meaning. `rx_dropped` now also counts host-side
overflow; it still must be 0. New: `frames-*.parquet` and the `meta` keys `frames_schema`, `poll_s`,
`iface`, `taps`, `errors.poll`, `frame_parts`. All are documented in
[TELEMETRY_SCHEMA.md](TELEMETRY_SCHEMA.md) §1, §6 and §9.

## 2. Offline evidence (reproducible)

```bash
R="env -u PYTHONPATH LD_LIBRARY_PATH=/usr/local/cuda/lib64 HF_HUB_OFFLINE=1 .venv/bin/python"
$R plugins/tests/test_telemetry.py <scratch>/u          # 1-15 Gate 1, 16-20 Gate 2A (R1-R4, frames log, analyzer)
$R plugins/tests/test_telemetry_record.py <scratch>/e2e  # official lerobot-record + analyzer on its session/dataset
$R plugins/tests/test_telemetry_fixture.py               # committed mock fixture still matches the producer
# unchanged suites: test_robot_guard, test_return_lag, test_quest_safety, test_teleoperate_mock (no args);
# test_record_mock, test_record_gate, test_rerun_status (scratch dir)
```

Results at the delivery commit are in the delivery note. A test never constructs a real `WireTap`: it patches
`telemetry.make_tap`, `telemetry.SYSFS_NET` and `telemetry._ip_link`, because can0/can1 are up on the lab PC.

Known on 2026-10-29: `test_quest_safety` check **P "out of reach: says so"** fails (43/44) in 3 of 3 runs. It
fails identically with Gate 1's `telemetry.py` restored, i.e. on Gate 1's exact production code, which passed
44/44 on 2026-10-28. The check is a timing race in the Quest IK path: the IK-jump pause fired before the
reach message. Telemetry is not involved. The PC was carrying a long-running `rerun` viewer at ~107 % CPU.
The cause is **UNKNOWN**; re-run it on an idle machine before the bench session.

## 3. Analysis tool

`tools/force_aware/telemetry_check.py` reads files only and needs only numpy, pandas and pyarrow; it imports
nothing from the plugin. Each check is PASS / FAIL / UNKNOWN (input missing). The exit status is 0 = GO,
1 = NO-GO, 3 = incomplete.

```bash
python tools/force_aware/telemetry_check.py <telemetry session> \
  --dataset <run B dataset root> --baseline-dataset <run A1 dataset root> \
  --diag <run B DIAG dir> --baseline-diag <run A1 DIAG dir> --baseline-diag <run A2 DIAG dir> \
  --json <out>/gate2_report.json
```

## 4. PASS/FAIL thresholds

"Loop rows" = the rows of the recorded dataset episodes, aligned by value.
**Gate** checks decide GO. **Stack** checks are findings about the existing robot stack: reported, reviewed by a
person, and not telemetry defects.

| id | group | PASS when | why this threshold |
|---|---|---|---|
| G1 | gate | `schema` = `openarm_telemetry/1`, `closed`, ticks 0..n−1, exactly one final row, `errors` all 0, `snapshot_ok`=1 on every row | any loss or exception invalidates rows |
| G2 | gate | `tap_available` L and R, `tap_ok`=1 on every row | both receive taps work |
| G3 | gate | `rx_dropped`=0 for both arms and `host_dropped`=0 | no frame lost before ingest |
| G4 | gate | `untimed`=0, no null `t_ns`, kernel stamps non-decreasing in delivery order, spread of `t_wall − t` ≤ 2 ms | ages are only valid on one undisturbed clock |
| G5 | gate | `cmd_n`=1 on ≥ 99 % of joint-rows; refresh echo seen (`state_fresh` known) on ≥ 99 %; \|`cmd_q` − `sent`\| ≤ 1 LSB (3.8e-4 rad) on every next row | the library's command and refresh frames are visible via loopback; one command per tick |
| G6 | gate | per interface, kernel `tx_packets` delta = library TX frames in the frames log (MIT + control frames on the command ids + 0x7FF), **exactly** | anything else transmitting, telemetry included, breaks the equality |
| G7 | gate | no enable, disable or set-zero frame on observation rows; no set-zero anywhere | nothing re-zeroed or re-enabled a motor during the session |
| G8 | gate | `snapshot_s` p99 ≤ 1 ms over all rows; max ≤ 5 ms on loop rows | ≤ 3 % / 15 % of a 33 ms tick |
| G10 | gate | telemetry ON (B) vs OFF (A1, A2), from DIAG `obs` periods: \|Δmedian\| ≤ 0.5 ms, Δp99 ≤ 2 ms, Δ(late > 1.5 periods) ≤ 0.5 pp, against each A. If A1 and A2 differ by more than this from each other: INCONCLUSIVE (UNKNOWN) and the runs are repeated longer. | "does not significantly affect 30 Hz recording" |
| G11 | gate | `observation.state`/`action` float32 [16], names in the interleaved order `RJ1,LJ1,…,LJ8`, no feature beyond images/index columns, `features` identical to run A's | the 16-D dataset format is preserved |
| G12 | gate | every episode aligns to exactly one run of consecutive rows; `action[k]` = `desired` of row k+1, bit-exact in float32 | telemetry joins the dataset without guessing |
| G13 | gate | per joint, refresh echo → its reply: 0 < median ≤ 2 ms, p99 ≤ 8 ms (`recv_deadline_us`); replies delivered before their request's echo ≤ 0.1 % | timestamp accuracy and delivery order; the follower's own note says replies take 0.26–1.02 ms |
| B1 | gate | 0 error frames; `rx_errors`, `tx_errors`, `rx_dropped`, `tx_dropped`, `rx_over_errors` unchanged; controller ERROR-ACTIVE at start and end | a sick bus invalidates every other number |
| S1 | stack | per joint on loop rows: `state_fresh` ≥ 99 %, `stale` ≤ 1 %, `state_age_s` p99 ≤ 5 ms | answers HARDWARE_AUDIT item 7 (one-tick-old values suspected). A FAIL means the follower serves old values, a research finding that must be fixed or modeled before contact work |
| S2 | stack | status nibble 1 for every motor on every loop row; `motor_fault`=0; `hold` rows counted (expected 0 in free motion) | motors healthy, guards quiet |
| S3 | stack | final row: 8 disable frames per arm | shutdown unchanged and visible |
| S4 | stack | lost replies ≤ 1 % per joint | intermittent feedback (follower notes J5–J8) |
| S5 | stack | loop period: median within 1 ms of 1/30 s, p99 ≤ 43 ms, late ≤ 1 % | absolute health of the existing loop; G10 is the telemetry criterion |

**Gate-2 report = GO** when every gate check is PASS. Stack FAILs go into the report with a decision; S1 in
particular decides whether contact features may use `q, dq, tau` as-is.

Not measurable on the bench, and covered offline instead:

* that the commands and the safety hold are identical with telemetry on and off: check 13;
* that the arm objects see only getter calls during a snapshot: check 14;
* that the module has no send path: check 11.

## 5. Safety prerequisites (all must hold before power-on)

Physical:

* [ ] Workspace clear of people and objects within both arms' reach plus 0.3 m, including the table area;
  nothing within reach of the start-pose approach. **No contact objects** in this gate: no battery, no charger.
* [ ] **Emergency stop:** a hardware stop that cuts motor power (not a software stop), within the operator's
  reach, tested by pressing it once with the motors powered **before** the session.
* [ ] Motor power supply at its normal voltage. Current limit set as for normal recording. Cabling and CAN
  connectors seated; no strain on the arm cables in the home pose.
* [ ] Arms hanging at rest at power-on (the connect-time pose is the return-to-rest target).

People:

* [ ] One operator in the headset, plus **one supervisor with a hand on the E-stop** for the whole session.
  The supervisor's sole job is the stop.
* [ ] Both have read the abort criteria below.

Software, checked before connecting:

* [ ] On the branch commit named in the delivery note, `git status` clean, offline suites green (§2).
* [ ] `ip -details -statistics link show can0` and `can1`: UP, `ERROR-ACTIVE`, berr counters 0. Read-only.
* [ ] No other process using the CAN interfaces (no stray teleop or record).
* [ ] Fresh, non-existent dataset roots, `--dataset.push_to_hub=false`, and a `local/...` repo_id: nothing
  can overwrite or upload a dataset.
* [ ] **Never** run `lerobot-calibrate`, `--robot.type=openarm_follower`, set-zero or parameter queries.

**Abort** (E-stop or Ctrl-C, then stop the session):

* any unexpected motion;
* any `MOTOR FAULT`, `SAFETY HOLD`, `intermittent CAN feedback` or `NOT at the start pose` print;
* any `[openarm_telemetry] NOT recording`, `telemetry tap unavailable` or `snapshot failed` warning (stop
  after the episode).

## 6. Bench procedure (about 20 minutes, no contact)

The normal README recording command, three times, changing only the flags marked. Use A–B–A order: two
telemetry-off runs bracket the telemetry-on run, so their difference measures run-to-run noise.

```bash
cd ~/Stanley_ws/lerobot_openarm && source .venv/bin/activate && unset PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
G=$HOME/force_aware_data/gate2_$(date +%Y%m%d); mkdir -p $G
CAMS="{body_cam: {type: opencv, index_or_path: /dev/rs_body, width: 640, height: 480, fps: 30}, wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_left, width: 640, height: 480, fps: 30}, right_wrist_cam: {type: opencv, index_or_path: /dev/rs_wrist_right, width: 640, height: 480, fps: 30}}"
run() {   # $1 = run name, $2 = extra robot flags
  test ! -e $G/$1 || { echo "$G/$1 exists, refusing"; return 1; }
  lerobot-record \
    --robot.type=openarm_umeow --robot.right_port=can0 --robot.left_port=can1 --robot.cameras="$CAMS" $2 \
    --teleop.type=openarm_quest --teleop.diag=true \
    --dataset.repo_id=local/openarm_gate2_$1 --dataset.root=$G/$1 --dataset.push_to_hub=false \
    --dataset.single_task="Gate 2 telemetry bench: free motion, no contact" \
    --dataset.num_episodes=3 --dataset.fps=30 --dataset.episode_time_s=40 --dataset.reset_time_s=10 \
    --dataset.streaming_encoding=true --dataset.encoder_threads=2 --display_data=true
}
run bench_A1 ""                                           # telemetry OFF
run bench_B  "--robot.telemetry_dir=$G/telemetry"         # telemetry ON
run bench_A2 ""                                           # telemetry OFF again
```

Each run: type `YES` (the arms go home), press X to start an episode. Move both arms slowly, in free space
within about 20 cm of home. Close and open each gripper twice in the air. Keep moving, because alignment needs
motion. Press X again to return home and save. Do this for 3 episodes, then Ctrl-C after the third save. Note
each `[openarm_diag] logging teleop diagnostics to ...` path.

## 7. Offline analysis after the session

```bash
python tools/force_aware/telemetry_check.py $G/telemetry/<session> \
  --dataset $G/bench_B --baseline-dataset $G/bench_A1 \
  --diag outputs/teleop_diag/<B> --baseline-diag outputs/teleop_diag/<A1> --baseline-diag outputs/teleop_diag/<A2> \
  --json $G/gate2_report.json
python -m lerobot_robot_openarm_umeow.diag_report outputs/teleop_diag/<A1>   # and <B>, <A2>: holds, faults, timing
```

Return `gate2_report.json`, the three `diag_report` outputs and the session's `meta.json`. Do **not** upload the
datasets.

## 8. Remaining unknowns (answered by the bench, not before)

| unknown | answered by |
|---|---|
| Does gs_usb deliver each TX echo before the reply it provokes? Are kernel stamps sub-ms? | G4, G13 |
| Does the tap see every library frame (loopback complete)? | G5, G6 |
| Are the observed values this tick's (HARDWARE_AUDIT item 7)? | S1 |
| Does telemetry change loop timing under real camera and encoder load? | G10, G8 |
| Is the real bus clean during recording? | B1, S4 |
| Firmware PMAX/VMAX/TMAX = the library's table; `tau` scale | **not** this gate: needs a motor parameter query (separate approval) or known loads ([BATTERY_CONTACT_COLLECTION.md](BATTERY_CONTACT_COLLECTION.md) block B) |
| Supply voltage | not in the DM state frame; an external meter if NeuralActuator-style inputs are wanted |

## 9. GO / NO-GO

* **Offline phase: GO.** Every suite passes. The three telemetry defects that would have corrupted real
  data (R1–R3) are reproduced and fixed. The bench claims each map to an automated check with a fixed
  threshold.
* **Operator-supervised bench test (§5–§7): GO, pending explicit approval**, and only with every
  prerequisite in §5 checked.
* **Contact data collection: NO-GO** until the bench report is GO and S1 is either PASS or has an approved
  mitigation.
