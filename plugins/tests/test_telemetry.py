"""The read-only telemetry sidecar (telemetry.py) against a fake DM bus. No hardware, no CAN socket.

    env -u PYTHONPATH LD_LIBRARY_PATH=/usr/local/cuda/lib64 python plugins/tests/test_telemetry.py <scratch dir>

Unit checks drive MotorTelemetry directly with telemetry_fakes.FakeWire; the wrapper checks run OpenArmUmeow
with the follower's hardware methods patched out (as test_robot_guard.py does) and telemetry.make_tap pointed
at the fake taps, so no socket is ever opened on the machine's CAN interfaces.
"""
import contextlib
import io
import json
import re
import sys
import tempfile
import time
import types
from pathlib import Path
from unittest import mock

import numpy as np

import telemetry_fakes as tf
from telemetry_fakes import LIMITS, FakeWire, lsb

import lerobot_robot_openarm_umeow  # noqa: F401  (adds the repo root to sys.path)
from lerobot_robot_openarm_umeow import telemetry as tel

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="openarm_telemetry_"))
REPO = Path(__file__).resolve().parents[2]
FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def make(wire: FakeWire, sub: str, fallback=None):
    arms = [tel._Arm(s, f"fake_{s}", wire.taps[s], tf.SEND, tf.RECV, [f"{s}J{i}" for i in range(1, 9)])
            for s in ("R", "L")]
    return tel.MotorTelemetry(ROOT / sub, wire.channels, LIMITS, arms, {"test": sub}, status_fallback=fallback)


def tick(t: tel.MotorTelemetry, wire: FakeWire, robot=None, period=0.0):
    t0 = time.perf_counter()
    obs = wire.read()
    t.snapshot(obs, t0, time.perf_counter() - t0, robot)
    if period:
        time.sleep(period)
    return obs


def finish(t: tel.MotorTelemetry):
    t.close()
    df, meta = tel.load(t.dir)
    return df[df["kind"] == tel.KIND_OBS].reset_index(drop=True), df, meta


ZERO = {f"{j}.pos": 0.0 for j in tel.JOINTS}

# ---------------------------------------------------------------------------------------------------- 1
rng = np.random.default_rng(0)
worst = {"q": 0.0, "dq": 0.0, "tau": 0.0, "kp": 0.0, "kd": 0.0}
for j, lim in LIMITS.items():
    p, v, tmax = lim
    for _ in range(200):
        q, dq, tau = rng.uniform(-3, 3), rng.uniform(-v, v), rng.uniform(-tmax, tmax)
        kp, kd = rng.uniform(0, 200), rng.uniform(0, 5)
        st = tel.decode_state(tf.enc_state(0xA, 3, q, dq, tau, 40, 38, lim), lim)
        mit = tel.decode_mit(tf.enc_mit(q, dq, kp, kd, tau, lim), lim)
        for k, a, b in (("q", st[1], q), ("dq", st[2], dq), ("tau", st[3], tau), ("q", mit[0], q),
                        ("dq", mit[1], dq), ("kp", mit[2], kp), ("kd", mit[3], kd), ("tau", mit[4], tau)):
            worst[k] = max(worst[k], abs(a - b) / lsb(lim, k))
        assert st[0] == 0xA and st[4:] == (40, 38)
check("1  wire decode round-trips the C++ encoder within one LSB (state and MIT frames, all motor types)",
      all(w <= 1.0 + 1e-9 for w in worst.values()), ", ".join(f"{k} {w:.2f} LSB" for k, w in worst.items()))
import openarm_can as oa  # noqa: E402

lib_state = lib_mit = 0
n_cases = 0
for j, lim in LIMITS.items():
    motor = oa.Motor(tf.TYPES[int(j[2:]) - 1], 0x01, 0x11)
    for _ in range(100):
        raw = bytes(rng.integers(0, 256, 8, dtype=np.uint8))
        ref = oa.CanPacketDecoder.parse_motor_state_data(motor, list(raw))  # the compiled library's decoder
        mine = tel.decode_state(raw, lim)
        lib_state += (mine[1], mine[2], mine[3], mine[4], mine[5]) != (ref.position, ref.velocity, ref.torque,
                                                                       ref.t_mos, ref.t_rotor)
        p = oa.MITParam()
        p.q, p.dq, p.kp, p.kd, p.tau = (rng.uniform(-3, 3), rng.uniform(-lim[1], lim[1]), rng.uniform(0, 200),
                                        rng.uniform(0, 5), rng.uniform(-lim[2], lim[2]))
        frame = bytes(oa.CanPacketEncoder.create_mit_control_command(motor, p).data)  # the library's encoder
        lib_mit += frame != tf.enc_mit(p.q, p.dq, p.kp, p.kd, p.tau, lim)
        n_cases += 1
check("1  against the INSTALLED openarm_can: state decode bit-equal, MIT frames byte-equal to its encoder",
      lib_state == 0 and lib_mit == 0, f"{n_cases} random cases per codec, {lib_state} + {lib_mit} mismatches")
check("1  enable / disable / set_zero frames are recognised and an MIT frame is not",
      [tel.ctrl_command(tf.enc_ctrl(c)) for c in (0xFC, 0xFD, 0xFE)] == ["enable", "disable", "set_zero"]
      and tel.ctrl_command(tf.enc_mit(0.1, 0, 50, 2, 1, LIMITS["LJ1"])) is None)

# ---------------------------------------------------------------------------------------------------- 2-3
wire = FakeWire()
for n, j in enumerate(tel.JOINTS, start=1):
    wire.true[j] = [0.01 * n, 0.0, 0.0]
t = make(wire, "order")
for _ in range(5):
    tick(t, wire, period=0.01)
obs_df, all_df, meta = finish(t)
order = [c.split(".")[0] for c in tel.COLUMNS if c.endswith(".q")]
q_last = [obs_df[f"{j}.q"].iloc[-1] for j in tel.JOINTS]
check("2  joint columns are LJ1..LJ8 then RJ1..RJ8, and each motor's value lands in its own column",
      order == tel.JOINTS and all(abs(q - 0.01 * n) <= lsb(LIMITS[j], "q")
                                   for n, (j, q) in enumerate(zip(tel.JOINTS, q_last), start=1)),
      f"q = {np.round(q_last, 3).tolist()}")
check("3  ticks are consecutive and both clocks increase; one final row (kind=1) closes the session",
      all_df["tick"].tolist() == list(range(6)) and np.all(np.diff(obs_df["t"]) > 0)
      and np.all(np.diff(obs_df["t_wall"]) > 0) and all_df["kind"].tolist() == [0] * 5 + [1])
ages = obs_df[[f"{j}.state_age_s" for j in tel.JOINTS]].to_numpy()
check("3  every value is traced to its wire frame: state_age_s >= 0 and under 5 ms, state_fresh = 1, stale = 0",
      np.all((ages >= 0) & (ages < 0.005)) and np.all(obs_df[[f"{j}.state_fresh" for j in tel.JOINTS]] == 1)
      and np.all(obs_df[[f"{j}.stale" for j in tel.JOINTS]] == 0), f"max age {ages.max() * 1e3:.2f} ms")

# ---------------------------------------------------------------------------------------------------- 4
wire = FakeWire()
t = make(wire, "dropped")
tick(t, wire, period=0.03)
wire.drop = {"LJ5"}
for _ in range(3):
    tick(t, wire, period=0.03)
wire.drop = set()
tick(t, wire)
d, _, _ = finish(t)
age = d["LJ5.state_age_s"].tolist()
check("4  a motor whose replies stop: rx_n = 0, the SAME frame's age grows, stale once older than stale_s",
      d["LJ5.rx_n"].tolist()[1:4] == [0, 0, 0] and age[1] < age[2] < age[3]
      and d["LJ5.stale"].tolist() == [0, 0, 1, 1, 0], f"age {np.round(age, 3).tolist()}, stale {d['LJ5.stale'].tolist()}")
check("4  ... while every other motor stays fresh on the same ticks",
      np.all(d[[f"{j}.stale" for j in tel.JOINTS if j != "LJ5"]] == 0))

# ---------------------------------------------------------------------------------------------------- 5
wire = FakeWire()
t = make(wire, "unread")
send_cmd = lambda q: wire.command({k: q for k in ZERO}, {}, {}, 50.0, 2.0)  # noqa: E731
tick(t, wire, period=0.02)
send_cmd(0.1)
wire.true["RJ2"][1] = 0.3  # the joint moves on, so the refresh reply differs from the command reply
wire.unread = {"RJ2"}
time.sleep(0.02)
tick(t, wire)
d, _, _ = finish(t)
check("5  a refresh reply that lands after the drain: the value is traced to the COMMAND reply -> state_fresh = 0",
      d["RJ2.state_fresh"].iloc[1] == 0 and d["RJ2.state_age_s"].iloc[1] > 0.015
      and d["RJ1.state_fresh"].iloc[1] == 1, f"RJ2 age {d['RJ2.state_age_s'].iloc[1] * 1e3:.1f} ms")

# ---------------------------------------------------------------------------------------------------- 6
wire = FakeWire()
wire.drop = {"RJ7"}
t = make(wire, "never")
tick(t, wire)
d, _, _ = finish(t)
check("6  a motor that never answered: state_age_s NaN, stale = 1 (not silently 0 / fresh)",
      np.isnan(d["RJ7.state_age_s"].iloc[0]) and d["RJ7.stale"].iloc[0] == 1 and d["RJ7.rx_n"].iloc[0] == 0)

# ---------------------------------------------------------------------------------------------------- 7
wire = FakeWire()
for tap in wire.taps.values():
    tap.available = False
t = make(wire, "notap", fallback=lambda: {j: 0x1 for j in tel.JOINTS})
wire.true["LJ2"] = [0.2, 0.0, 1.5]
tick(t, wire)
d, _, meta = finish(t)
wire_cols = [f"{j}.{f}" for j in tel.JOINTS for f in ("rx_n", "rx_age_s", "state_age_s", "state_fresh", "stale",
                                                         "cmd_q", "cmd_n")]
check("7  tap unavailable: wire columns NaN (unknown, not 0), tap_ok = 0, Motor values and fallback status kept",
      d[wire_cols].isna().all().all() and d["L.tap_ok"].iloc[0] == 0
      and abs(d["LJ2.tau"].iloc[0] - 1.5) <= lsb(LIMITS["LJ2"], "tau") and d["LJ2.status"].iloc[0] == 1)

# ---------------------------------------------------------------------------------------------------- 8
wire = FakeWire()
t = make(wire, "errors")
tick(t, wire)
wire.motors["RJ3"].fail = True
raised = None
try:
    tick(t, wire)
except Exception as e:  # noqa: BLE001
    raised = e
wire.motors["RJ3"].fail = False
tick(t, wire)
t._flush()
import pyarrow.parquet as pq  # noqa: E402

with mock.patch.object(pq, "write_table", side_effect=OSError("disk full")):
    t._flush()
    tick(t, wire)
    t._flush()
d, _, meta = finish(t)
check("8  a Motor read that raises: no exception reaches the caller, that row has snapshot_ok = 0 and NaN joints",
      raised is None and d["snapshot_ok"].tolist()[:3] == [1, 0, 1] and np.isnan(d["LJ1.q"].iloc[1])
      and d["tick"].tolist()[:3] == [0, 1, 2])
check("8  a writer failure is counted in meta.json, never raised", meta["errors"]["writer"] >= 1 and meta["closed"],
      json.dumps(meta["errors"]))

# ---------------------------------------------------------------------------------------------------- 9-10
wire = FakeWire()
t = make(wire, "command")
robot = types.SimpleNamespace(_last_sent={k: 0.25 for k in ZERO}, _last_desired={k: 0.3 for k in ZERO},
                              _fault="test hold", _motor_fault=None, gripper_squeeze_tau={"L": 0.0, "R": -1.5})
tick(t, wire, robot)
wire.command(robot._last_sent, {f"{j}.vel": 0.4 for j in tel.JOINTS}, {j: 0.7 for j in tel.JOINTS}, 60.0, 2.0)
wire.ctrl("L", 0xFD)
tick(t, wire, robot)
d, all_df, meta = finish(t)
r = d.iloc[1]
lim = LIMITS["LJ1"]
check("9  the applied MIT command is decoded from the wire: q, dq, kp, kd, tau within one LSB, cmd_n = 1",
      abs(r["LJ1.cmd_q"] - 0.25) <= lsb(lim, "q") and abs(r["LJ1.cmd_dq"] - 0.4) <= lsb(lim, "dq")
      and abs(r["LJ1.cmd_kp"] - 60) <= lsb(lim, "kp") and abs(r["LJ1.cmd_kd"] - 2) <= lsb(lim, "kd")
      and abs(r["LJ1.cmd_tau"] - 0.7) <= lsb(lim, "tau") and r["LJ1.cmd_n"] == 1 and d.iloc[0]["LJ1.cmd_n"] == 0)
check("9  host-side sent / desired / hold / squeeze columns come from the wrapper",
      r["RJ4.sent"] == 0.25 and r["RJ4.desired"] == 0.3 and r["hold"] == 1 and r["R.squeeze_tau"] == -1.5)
check("10 disable frames on the bus are counted per arm and listed in meta.json",
      r["L.ctrl_frames"] == 8 and r["R.ctrl_frames"] == 0
      and sum(e["command"] == "disable" for e in meta["ctrl_frames"]) == 8)

files = sorted(Path(t.dir).glob("part-*.parquet"))
schema = pq.read_schema(files[0])
doc = (REPO / "docs/force_aware/TELEMETRY_SCHEMA.md").read_text()
undocumented = [f for f in (*tel.ROW_FIELDS, *tel.SIDE_FIELDS, *tel.JOINT_FIELDS, "tick") if f"`{f}`" not in doc]
check("10 Parquet schema == telemetry.COLUMNS; tick int64, everything else float64",
      schema.names == tel.COLUMNS and str(schema.field("tick").type) == "int64"
      and all(str(schema.field(c).type) == "double" for c in tel.COLUMNS if c != "tick"),
      f"{len(tel.COLUMNS)} columns")
check("10 meta.json carries the contract and provenance fields; every column field is documented",
      {"schema", "columns", "joints", "stale_s", "clock", "closed", "rows"} <= set(meta) and not undocumented,
      f"undocumented: {undocumented}" if undocumented else "")

# ---------------------------------------------------------------------------------------------------- 11
src = (REPO / "plugins/lerobot_robot_openarm_umeow/lerobot_robot_openarm_umeow/telemetry.py").read_text()
check("11 telemetry.py never transmits: no send/sendto/sendmsg/sendall call, WireTap has no send method",
      not re.search(r"\.send(to|msg|all)?\(", src) and not any("send" in n for n in dir(tel.WireTap)))

# ---------------------------------------------------------------------------------------------------- 12
wire = FakeWire()
t = make(wire, "overhead")
cost = []
for i in range(300):
    wire.command({k: 0.001 * i for k in ZERO}, {}, {}, 50.0, 2.0)
    tick(t, wire, robot)
d, _, meta = finish(t)
med, p99 = np.median(d["snapshot_s"]) * 1e3, np.percentile(d["snapshot_s"], 99) * 1e3
check("12 snapshot cost with a full tick of traffic (16 MIT + 16 refresh + 32 replies): median < 1 ms",
      med < 1.0, f"median {med:.3f} ms, p99 {p99:.3f} ms, max {meta['max_snapshot_s'] * 1e3:.2f} ms; 33 ms tick at 30 Hz")

# ---------------------------------------------------------------------------------------------------- 13-15
import robots.umeow_openarm_follower.openarm_follower as fol  # noqa: E402
from lerobot_robot_openarm_umeow import OpenArmUmeow, OpenArmUmeowConfig  # noqa: E402
from lerobot_robot_openarm_umeow.common import MOTOR_KEYS  # noqa: E402

fol.oa.OpenArm = mock.MagicMock(side_effect=lambda *a, **k: mock.MagicMock())  # one distinct mock per arm
W = {"wire": None}
fol.OpenArmFollower.connect = lambda self, calibrate=False: setattr(self, "_is_connected", True)
fol.OpenArmFollower.disconnect = lambda self: (W["wire"].ctrl("R", 0xFD), W["wire"].ctrl("L", 0xFD),
                                              setattr(self, "_is_connected", False))
fol.OpenArmFollower.get_observation = lambda self: W["wire"].read()
fol.OpenArmFollower.send_action = lambda self, action, vel: (
    W["wire"].command({k: float(action[k]) for k in MOTOR_KEYS}, vel, {}, 50.0, 2.0), action)[1]
tel.make_tap = lambda port, ids, fd=True: W["wire"].taps["R" if port == "can0" else "L"]


def session(telemetry_dir: str):
    W["wire"] = wire = FakeWire()
    r = OpenArmUmeow(OpenArmUmeowConfig(start_pose="none", return_to_rest=False, telemetry_dir=telemetry_dir))
    for prefix, arm in (("R", r.right_arm), ("L", r.left_arm)):  # the REAL _motor_channels reads these
        arm.get_arm.return_value.get_motors.return_value = [wire.motors[f"{prefix}J{i}"] for i in range(1, 8)]
        arm.get_gripper.return_value.get_motor.return_value = wire.motors[f"{prefix}J8"]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        r.connect()
        t_loop = time.perf_counter()
        cmd, sent, calls = dict(ZERO), [], []
        for i in range(40):
            time.sleep(1 / 60)
            cmd["RJ4.pos"] = 0.01 * min(i, 20)
            if i == 25:
                cmd["RJ2.pos"] = 0.5  # one-tick jump: the guard must hold
            before = (len(r.right_arm.mock_calls), len(r.left_arm.mock_calls))
            r.get_observation()
            calls += r.right_arm.mock_calls[before[0]:] + r.left_arm.mock_calls[before[1]:]
            sent.append(r.send_action(dict(cmd)))
        tdir = r._telemetry.dir if r._telemetry is not None else None
        r.disconnect()
    return sent, calls, tdir, r, t_loop


sent_off, _, dir_off, r_off, _ = session("")
sent_on, calls, dir_on, r_on, t_loop = session(str(ROOT / "wrapper"))
check("13 telemetry off by default: no session is started", dir_off is None and r_off._telemetry is None)
check("13 telemetry on: the commands sent and the safety hold are identical to telemetry off",
      sent_on == sent_off and any(s["RJ2.pos"] == 0.0 for s in sent_on[25:]) and dir_on is not None,
      f"{len(sent_on)} commands")
names = sorted({c[0] for c in calls})
allowed = {"get_arm", "get_arm().get_motors", "get_gripper", "get_gripper().get_motor"}
check("14 during get_observation + snapshot the arms see only getter calls -- no refresh/recv/command/enable",
      set(names) <= allowed, f"calls: {names}")
d, meta = tel.load(dir_on)
obs_rows = d[(d["kind"] == 0) & (d["t"] >= t_loop)]
held = obs_rows["hold"].to_numpy()
check("15 one row per get_observation (plus connect-time reads), hold column follows the guard, final row shows the disable frames",
      len(obs_rows) == 40 and held[:26].sum() == 0 and held[26:].all() and meta["closed"]
      and d["kind"].iloc[-1] == 1 and d["R.ctrl_frames"].iloc[-1] == 8 and d["L.ctrl_frames"].iloc[-1] == 8,
      f"hold from row {int(np.argmax(held))}")
nxt = obs_rows.iloc[1:].reset_index(drop=True)
gap = max(abs(nxt[f"{k[:-4]}.cmd_q"] - nxt[f"{k[:-4]}.sent"]).max() for k in MOTOR_KEYS)
check("15 row k+1's wire command equals the host's sent command of tick k (within one LSB)",
      gap <= lsb(LIMITS["LJ1"], "q"), f"max {gap:.2e} rad")

print(f"\n{len(FAILS)} failed" + (f": {FAILS}" if FAILS else ""))
sys.exit(1 if FAILS else 0)
