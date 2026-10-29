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
# Hermetic: the interface counters come from a scratch folder, never the lab PC's real can0/can1.
tel.SYSFS_NET = ROOT / "sysfs"
tel._ip_link = lambda port: None


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def make(wire: FakeWire, sub: str, fallback=None):
    arms = [tel._Arm(s, f"fake_{s}", wire.taps[s], tf.SEND, tf.RECV, [f"{s}J{i}" for i in range(1, 9)])
            for s in ("R", "L")]
    meta = {"test": sub, "can_ids": {"send": tf.SEND, "recv": tf.RECV, "refresh": tel.REFRESH_ID},
            "tap_available": {a.side: a.tap.available for a in arms}}  # as for_robot() writes them
    return tel.MotorTelemetry(ROOT / sub, wire.channels, LIMITS, arms, meta, status_fallback=fallback)


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

# ---------------------------------------------------------------------------------------------------- 16-17
# WireTap.read() itself, on a real kernel socket: UDP over loopback carries the same SO_TIMESTAMPNS ancillary data
# and non-blocking semantics as a CAN_RAW socket, and needs no CAN interface.
import os  # noqa: E402
import resource  # noqa: E402
import socket  # noqa: E402
import struct  # noqa: E402


def udp_tap(timestamps=True, fd=None):
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    addr = rx.getsockname()
    if timestamps:
        rx.setsockopt(socket.SOL_SOCKET, tel.SO_TIMESTAMPNS, 1)
    if fd is not None:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft <= fd:
            resource.setrlimit(resource.RLIMIT_NOFILE, (fd + 16, hard))
        os.dup2(rx.fileno(), fd)
        rx.close()
        rx = socket.socket(fileno=fd)
    rx.setblocking(False)
    w = tel.WireTap.__new__(tel.WireTap)  # no CAN socket: hand it the UDP one
    w.port, w.available, w.dropped, w.errors, w.untimed, w._sock = "udp-loopback", True, 0, 0, 0, rx
    return w, addr


tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
payload = tf.enc_state(1, 1, 0.5, 0.0, 0.0, 30, 30, LIMITS["LJ1"])
w, addr = udp_tap(fd=1500)
t_send = time.time_ns()
tx.sendto(struct.pack("=IB3x", 0x11, 8) + payload, addr)
time.sleep(0.02)
try:
    got, err = w.read(), None
except Exception as e:  # noqa: BLE001
    got, err = [], e
t_read = time.time_ns()
ok = not err and len(got) == 1 and got[0][:2] == (0x11, payload) and got[0][2] is not None
check("16 WireTap.read on a socket with fd >= 1024 (select() raised ValueError there): reads the frame",
      ok, f"{type(err).__name__}: {err}" if err else f"fd {w._sock.fileno()}")
check("16 ... with the KERNEL's receive time, not the read time",
      ok and abs(got[0][2] - t_send) < 5e6 and t_read - got[0][2] > 15e6,
      f"stamp - send {(got[0][2] - t_send) / 1e6:.3f} ms, read - stamp {(t_read - got[0][2]) / 1e6:.1f} ms" if ok else "")
w._sock.close()

w, addr = udp_tap(timestamps=False)
tx.sendto(struct.pack("=IB3x", 0x11, 8) + payload, addr)
time.sleep(0.005)
got = w.read()
check("17 a frame without a kernel timestamp is returned with t = None and counted in `untimed`",
      len(got) == 1 and got[0][2] is None and w.untimed == 1)
w._sock.close()


class UntimedTap(tf.FakeTap):
    def read(self):
        out, self.queue = self.queue, []
        self.untimed += len(out)
        return [(c, d, None) for c, d, _ in out]


wire = FakeWire()
wire.taps = {"R": UntimedTap(), "L": UntimedTap()}
t = make(wire, "untimed")
for i in range(10):
    wire.command({k: 0.01 * i for k in ZERO}, {}, {}, 50.0, 2.0)
    tick(t, wire)
d, _, meta = finish(t)
d = d.iloc[1:]
check("17 untimed frames stay in the row that read them (stamped at its reference time), and meta counts them",
      (d["RJ3.cmd_n"] == 1).all() and (d["RJ3.rx_n"] == 2).all() and (d["RJ3.stale"] == 0).all()
      and meta["taps"]["R"]["untimed"] == meta["taps"]["R"]["frames"] > 0,
      f"untimed {meta['taps']['R']['untimed']} of {meta['taps']['R']['frames']} frames; before the fix every row was stale"
      " and every frame landed one row late")

# ---------------------------------------------------------------------------------------------------- 18
# A ramp (sim_bridge_common.ramp_to) commands at 50 Hz without reading the robot; the kernel queue holds a few
# hundred frames (rmem_max 212992 -> SO_RCVBUF 425984) and then drops the NEWEST -- the disable frames last.


def ramp_session(sub):
    wire = FakeWire()
    wire.taps = {"R": tf.FakeTap(capacity=500), "L": tf.FakeTap(capacity=500)}
    t = make(wire, sub)
    for i in range(3):
        wire.command({k: 0.0 for k in ZERO}, {}, {}, 50.0, 2.0)
        tick(t, wire, period=1 / 30)
    for i in range(60):  # 1.2 s at 50 Hz: 60 x 16 frames per arm
        wire.command({k: 0.002 * i for k in ZERO}, {}, {}, 50.0, 2.0)
        time.sleep(1 / 50)
    wire.ctrl("R", 0xFD)
    wire.ctrl("L", 0xFD)
    _, df, meta = finish(t)
    return df.iloc[-1], meta


poll = tel.POLL_S
tel.POLL_S = 3600.0  # the Gate-1 behaviour: nothing drains the taps between snapshots
last_old, meta_old = ramp_session("ramp_nodrain")
tel.POLL_S = poll
last, meta = ramp_session("ramp")
check("18 reproduced: with no drain between snapshots a ramp overflows the tap and the disable frames are lost",
      last_old["R.rx_dropped"] > 0 and last_old["R.ctrl_frames"] < 8,
      f"rx_dropped {last_old['R.rx_dropped']:.0f}, final-row disable frames {last_old['R.ctrl_frames']:.0f}/8")
check("18 fixed: the idle drain keeps the tap empty through the ramp; final row shows all 8 disable frames per arm",
      last["R.rx_dropped"] == 0 and last["L.rx_dropped"] == 0 and last["R.ctrl_frames"] == 8
      and last["L.ctrl_frames"] == 8 and meta["errors"]["poll"] == 0,
      f"rx_dropped {last['R.rx_dropped']:.0f}/{last['L.rx_dropped']:.0f}, disable {last['R.ctrl_frames']:.0f}/"
      f"{last['L.ctrl_frames']:.0f}, frames {meta['taps']['R']['frames']}")

# ---------------------------------------------------------------------------------------------------- 19-20
# The raw frames log and the interface counters, then the Gate-2 analyzer over a whole fake session.
sys.path.insert(0, str(REPO / "tools/force_aware"))
import telemetry_check as tc  # noqa: E402

sysfs = ROOT / "sysfs_fake"
STATS = ("tx_packets", "rx_packets", "rx_errors", "tx_errors", "rx_dropped", "tx_dropped", "rx_over_errors")


def write_sysfs(port, **vals):
    d = sysfs / port / "statistics"
    d.mkdir(parents=True, exist_ok=True)
    for k in STATS:
        (d / k).write_text(f"{vals.get(k, 0)}\n")


tel.SYSFS_NET = sysfs
tel._ip_link = lambda port: {"flags": ["NOARP", "UP", "LOWER_UP", "ECHO"], "linkinfo": {
    "info_kind": "can", "info_data": {"state": "ERROR-ACTIVE", "berr_counter": {"tx": 0, "rx": 0}},
    "info_xstats": {"restarts": 0, "bus_error": 0, "bus_off": 0}}}
for port in ("fake_R", "fake_L"):
    write_sysfs(port, tx_packets=1000)
wire = FakeWire()
t = make(wire, "frames")
wire.tx = {"R": 0, "L": 0}
host = types.SimpleNamespace(_fault=None, _motor_fault=None, gripper_squeeze_tau={"L": 0.0, "R": 0.0})
for i in range(60):
    host._last_sent = host._last_desired = {k: 0.002 * i for k in ZERO}
    wire.command(host._last_sent, {}, {}, 50.0, 2.0)
    tick(t, wire, host, period=1 / 30)
for side in ("R", "L"):
    wire.ctrl(side, 0xFD)
    write_sysfs(f"fake_{side}", tx_packets=1000 + wire.tx[side])
_, df, meta = finish(t)
fr = tel.load_frames(t.dir)
per_tick_mit = fr[fr.can_id.isin(tf.SEND) & fr.data.map(lambda b: b[:7] != b"\xff" * 7)].groupby(["tick", "side"]).size()
cmd_cols = {(int(k), s): int(df.loc[df.tick == k, [f"{s}J{i}.cmd_n" for i in range(1, 9)]].to_numpy().sum())
            for k in df.tick for s in "LR"}
seq_ok = all(np.array_equal(g["seq"].to_numpy(), np.arange(len(g))) for _, g in fr.groupby("side"))
check("19 frames log: every delivered frame once, in delivery order per arm, with the row that counted it",
      len(fr) == sum(meta["taps"][s]["frames"] for s in "LR") and seq_ok
      and all(per_tick_mit.get((k, s), 0) == n for (k, s), n in cmd_cols.items())
      and (fr.can_id == tel.REFRESH_ID).sum() == 16 * 60 and meta["frames_schema"] == tel.FRAMES_SCHEMA,
      f"{len(fr)} frames in {meta['frame_parts']} part(s)")
check("19 meta.json holds the interfaces' kernel counters at start and close",
      meta["iface"]["start"]["R"]["statistics"]["tx_packets"] == 1000
      and meta["iface"]["end"]["R"]["statistics"]["tx_packets"] == 1000 + wire.tx["R"]
      and meta["iface"]["end"]["L"]["can"]["state"] == "ERROR-ACTIVE")
R, _ = tc.run(t.dir)
st = {i["id"]: i["status"] for i in R.items}
must = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G13", "B1", "S1", "S2", "S3", "S5"]
check("20 telemetry_check on a healthy fake session: every gate check it can evaluate PASSes; dataset/A-B UNKNOWN",
      all(st[c] == "PASS" for c in must) and all(st[c] == "UNKNOWN" for c in ("G10", "G11", "G12")),
      ", ".join(f"{k} {v}" for k, v in st.items()))
for i in R.items:
    if i["status"] != "PASS":
        print(f"      {i['id']} {i['status']}: {i['detail'][:300]}")
meta_path = t.dir / "meta.json"
m = json.loads(meta_path.read_text())
m["iface"]["end"]["R"]["statistics"]["tx_packets"] += 1  # one frame the library did not send
meta_path.write_text(json.dumps(m))
R, _ = tc.run(t.dir)
st = {i["id"]: i for i in R.items}
check("20 telemetry_check: ONE transmitted frame that no library frame explains fails G6 (and the verdict)",
      st["G6"]["status"] == "FAIL" and R.verdict() == "NO-GO", st["G6"]["detail"])



def diag_dir(name, late_every=0):
    """A teleop DIAG folder (diag_log.py layout) whose 'obs' stream ticks at 30 Hz with 0.3 ms jitter."""
    d = ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    g = np.random.default_rng(1)
    dt = 1 / 30 + g.normal(0, 3e-4, 900)
    if late_every:
        dt[::late_every] += 0.03
    t = 1000 + np.cumsum(dt)
    np.savez(d / "chunk_00001.npz", obs=np.stack([t, np.full_like(t, 0.004)], axis=1))
    return d


R, _ = tc.run(t.dir, diag=diag_dir("diag_on"), baseline_diag=diag_dir("diag_off"))
same = next(i for i in R.items if i["id"] == "G10")
R, _ = tc.run(t.dir, diag=diag_dir("diag_on_late", late_every=50), baseline_diag=diag_dir("diag_off"))
late = next(i for i in R.items if i["id"] == "G10")
check("20 telemetry_check G10: equal on/off loop timing PASSes; 2 % extra late ticks with telemetry on FAILs",
      same["status"] == "PASS" and late["status"] == "FAIL", f"{same['detail']} || {late['detail']}")
R, _ = tc.run(t.dir, diag=diag_dir("diag_on2"), baseline_diag=[diag_dir("diag_off"), diag_dir("diag_off_late", late_every=50)])
aba = next(i for i in R.items if i["id"] == "G10")
check("20 telemetry_check G10 (A-B-A): two telemetry-off runs that disagree make the comparison INCONCLUSIVE (UNKNOWN)",
      aba["status"] == "UNKNOWN" and "INCONCLUSIVE" in aba["detail"], aba["detail"][-120:])

print(f"\n{len(FAILS)} failed" + (f": {FAILS}" if FAILS else ""))
sys.exit(1 if FAILS else 0)
