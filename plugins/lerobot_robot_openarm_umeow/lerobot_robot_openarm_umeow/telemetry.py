"""Read-only motor telemetry sidecar: one row per observation tick, for sensorless contact research.

Turned on by --robot.telemetry_dir=<dir> (openarm_umeow on the real arm only; off by default, and then the
whole cost is one attribute check per get_observation). The contract -- every column, unit, time basis and
missing-value rule -- is docs/force_aware/TELEMETRY_SCHEMA.md; the evidence behind it is
docs/force_aware/HARDWARE_AUDIT.md. Nothing here changes the 16-D observation or action.

Why a second, receive-only CAN socket
-------------------------------------
openarm_can's Motor keeps ONE retained (q, dq, tau, t_mos, t_rotor) per motor with no timestamp and no
sequence number, and every state frame overwrites it -- the reply to a refresh_all() AND the reply every MIT
command provokes. So "what did the policy actually see, and how old was it" cannot be answered from the
Motor object, and the follower's own CanFeedbackMonitor counts frames since its last poll (which only
happens inside a read), so a count of 1 can be a command reply from the previous tick. This tap answers it
from the wire instead:

  - every motor reply is decoded exactly as openarm_can decodes it and stamped with the kernel's receive
    time (SO_TIMESTAMPNS); the frame whose decode equals the Motor state is the one the policy saw, and its
    stamp is that value's age (`state_age_s`), and whether it answered this tick's refresh (`state_fresh`);
  - SocketCAN loops every frame a process transmits back to the OTHER sockets on the interface, so the
    library's own MIT frames arrive here too: decoding them gives the command the motor was actually sent
    (q, dq, kp, kd, tau feed-forward), after openarm_can's clamping and 12/16-bit quantisation;
  - enable / disable / set-zero frames on the command ids are counted (`ctrl_frames`), so a run can prove
    nothing re-zeroed or re-enabled a motor mid-episode.

The socket NEVER transmits (there is no send call in this module), observes the same frames openarm_can
consumes without consuming them, and is fail-soft: if it cannot be opened every wire column is NaN and the
rows are still written. Telemetry errors are counted and never raised into the control loop.
"""

import hashlib
import json
import logging
import os
import select
import socket
import struct
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from .common import REPO_ROOT

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "openarm_telemetry/1"
JOINTS = [f"LJ{i}" for i in range(1, 9)] + [f"RJ{i}" for i in range(1, 9)]
SIDES = ("L", "R")
JOINT_FIELDS = (
    "q", "dq", "tau", "t_mos", "t_rotor", "status",
    "rx_n", "rx_age_s", "state_age_s", "state_fresh", "stale",
    "cmd_q", "cmd_dq", "cmd_kp", "cmd_kd", "cmd_tau", "cmd_n", "cmd_age_s",
    "sent", "desired",
)
SIDE_FIELDS = ("tap_ok", "rx_dropped", "bus_errors", "ctrl_frames", "squeeze_tau")
ROW_FIELDS = ("kind", "t", "t_wall", "obs_t0", "obs_read_s", "snapshot_ok", "snapshot_s", "hold", "motor_fault")
COLUMNS = (
    ["tick", *ROW_FIELDS]
    + [f"{s}.{f}" for s in SIDES for f in SIDE_FIELDS]
    + [f"{j}.{f}" for j in JOINTS for f in JOINT_FIELDS]
)
_COL = {name: i for i, name in enumerate(COLUMNS)}

KIND_OBS, KIND_FINAL = 0, 1
STALE_S = 0.05      # a value older than this when the policy saw it is `stale` (1.5 ticks at 30 Hz)
FLUSH_S = 5.0       # rows reach disk this often, so a crash loses at most this much
MAX_FRAMES_PER_READ = 20_000  # a bound on one drain, so a flooded bus cannot hold the control loop here
# A decode here and openarm_can's are the same double arithmetic, but a compiler is free to fuse its
# multiply-add; 1e-9 absorbs that and is still ~1e-5 of the finest LSB (DM4310 torque, 4.9e-3 N-m).
MATCH_TOL = 1e-9

# SocketCAN / socket constants (Linux). Python's socket module does not export all of them.
CANFD_MTU, CAN_MTU = 72, 16
CAN_RAW_FD_FRAMES, CAN_RAW_ERR_FILTER = 5, 2
CAN_EFF_MASK, CAN_ERR_FLAG, CAN_ERR_MASK_ALL = 0x1FFFFFFF, 0x20000000, 0x1FFFFFFF
SO_TIMESTAMPNS = getattr(socket, "SO_TIMESTAMPNS", 35)
SO_RXQ_OVFL = getattr(socket, "SO_RXQ_OVFL", 40)
REFRESH_ID = 0x7FF  # openarm_can's refresh request: data = [send_id lo, send_id hi, 0xCC, 0...]
CTRL_BYTES = {0xFC: "enable", 0xFD: "disable", 0xFE: "set_zero"}  # data = FF x7 + one of these


# -- DM wire format, mirrored from openarm_can 1.2.8 (dm_motor_control.cpp) --------------------------------
def _u2f(x: int, lo: float, hi: float, bits: int) -> float:
    """CanPacketDecoder::uint_to_double, operation for operation, so a decode here is bit-equal to the library's."""
    return x / ((1 << bits) - 1) * (hi - lo) + lo


def decode_state(data: bytes, limits: tuple[float, float, float]):
    """A motor's state frame -> (status nibble, q rad, dq rad/s, tau N-m, t_mos C, t_rotor C), or None."""
    if len(data) < 8:
        return None
    p, v, t = limits
    q = _u2f((data[1] << 8) | data[2], -p, p, 16)
    dq = _u2f((data[3] << 4) | (data[4] >> 4), -v, v, 12)
    tau = _u2f(((data[4] & 0xF) << 8) | data[5], -t, t, 12)
    return data[0] >> 4, q, dq, tau, data[6], data[7]


def decode_mit(data: bytes, limits: tuple[float, float, float]):
    """An MIT command frame -> (q, dq, kp, kd, tau) as the motor received them, or None (wrong length)."""
    if len(data) < 8:
        return None
    p, v, t = limits
    q = _u2f((data[0] << 8) | data[1], -p, p, 16)
    dq = _u2f((data[2] << 4) | (data[3] >> 4), -v, v, 12)
    kp = _u2f(((data[3] & 0xF) << 8) | data[4], 0.0, 500.0, 12)
    kd = _u2f((data[5] << 4) | (data[6] >> 4), 0.0, 5.0, 12)
    tau = _u2f(((data[6] & 0xF) << 8) | data[7], -t, t, 12)
    return q, dq, kp, kd, tau


def ctrl_command(data: bytes) -> str | None:
    """'enable' / 'disable' / 'set_zero' for a DM control frame, None for anything else (an MIT command)."""
    if len(data) == 8 and data[:7] == b"\xff" * 7:
        return CTRL_BYTES.get(data[7])
    return None


def motor_limits(motor_type) -> tuple[float, float, float]:
    """(pMax, vMax, tMax) the installed openarm_can scales this motor type's frames with."""
    import openarm_can as oa

    p = oa.Motor.get_limit_param(motor_type)
    return float(p.pMax), float(p.vMax), float(p.tMax)


# -- the receive-only socket -------------------------------------------------------------------------------
class WireTap:
    """Kernel-timestamped frames from one CAN interface, restricted to `can_ids`. Receive only.

    Fail-soft like can_monitor.CanFeedbackMonitor: `available` is False if the socket cannot be opened, and
    read() then returns nothing.
    """

    def __init__(self, port: str, can_ids, fd: bool = True):
        self.port = port
        self.available = False
        self.dropped = 0   # frames the kernel dropped from THIS socket's queue (SO_RXQ_OVFL), cumulative
        self.errors = 0    # CAN error frames raised by the controller, cumulative
        self._sock = None
        try:
            sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
            if fd:
                sock.setsockopt(socket.SOL_CAN_RAW, CAN_RAW_FD_FRAMES, 1)
            filters = b"".join(struct.pack("=II", cid, CAN_EFF_MASK) for cid in sorted(set(can_ids)))
            sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, filters)
            sock.setsockopt(socket.SOL_CAN_RAW, CAN_RAW_ERR_FILTER, CAN_ERR_MASK_ALL)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
            sock.setsockopt(socket.SOL_SOCKET, SO_TIMESTAMPNS, 1)
            sock.setsockopt(socket.SOL_SOCKET, SO_RXQ_OVFL, 1)
            sock.bind((port,))
            sock.setblocking(False)
        except OSError as e:
            logger.warning(f"telemetry tap unavailable on {port} ({type(e).__name__}: {e}); wire columns will be NaN.")
            return
        self._sock = sock
        self.available = True

    def read(self) -> list[tuple[int, bytes, int]]:
        """Everything queued since the last read: [(can_id, payload, kernel receive time in ns since epoch)]."""
        if not self.available:
            return []
        frames = []
        for _ in range(MAX_FRAMES_PER_READ):
            if not select.select([self._sock], [], [], 0)[0]:
                break
            try:
                buf, anc, _, _ = self._sock.recvmsg(CANFD_MTU, 128)
            except BlockingIOError:
                break
            except OSError as e:
                logger.warning(f"telemetry tap on {self.port} stopped: {e}")
                self.available = False
                break
            t_ns = None
            for level, kind, data in anc:
                if level == socket.SOL_SOCKET and kind == SO_TIMESTAMPNS and len(data) >= 16:
                    sec, nsec = struct.unpack("=qq", data[:16])
                    t_ns = sec * 1_000_000_000 + nsec
                elif level == socket.SOL_SOCKET and kind == SO_RXQ_OVFL and len(data) >= 4:
                    self.dropped = struct.unpack("=I", data[:4])[0]
            if len(buf) < CAN_MTU:
                continue
            raw_id = struct.unpack("=I", buf[:4])[0]
            if raw_id & CAN_ERR_FLAG:
                self.errors += 1
                continue
            frames.append((raw_id & CAN_EFF_MASK, bytes(buf[8:8 + buf[4]]), t_ns if t_ns is not None else time.time_ns()))
        return frames

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None
        self.available = False


make_tap = WireTap  # tests replace this; nothing else should


class _Joint:
    """One motor's wire history between two rows, and what is carried across rows."""

    def __init__(self, limits):
        self.limits = limits
        self.replies: list[tuple] = []   # (t_ns, status, q, dq, tau, t_mos, t_rotor) received this row
        self.status = None               # last status nibble ever seen
        self.cmd = None                  # (t_ns, q, dq, kp, kd, tau) last MIT command ever seen
        self.cmd_n = 0
        self.refresh_ns = None           # last refresh request for this motor
        self.matched = None              # (t_ns, q, dq, tau) the frame the library state was last traced to


class _Arm:
    def __init__(self, side, port, tap, send_ids, recv_ids, joint_names):
        self.side, self.port, self.tap = side, port, tap
        self.reply = dict(zip(recv_ids, joint_names))
        self.cmd = dict(zip(send_ids, joint_names))
        self.pending: list[tuple[int, bytes, int]] = []
        self.errors_seen = tap.errors if tap is not None else 0
        self.ctrl_frames = 0
        self.ctrl_log: list[dict] = []


class MotorTelemetry:
    """Builds one row per snapshot() and writes them as Parquet parts in a background thread."""

    def __init__(self, out_dir: str | Path, channels, limits: dict, arms: list[_Arm], meta: dict,
                 status_fallback=None):
        """`channels` returns [(joint, Motor)] for the 16 motors (the follower's _motor_channels)."""
        self._channels = channels
        self._joints = {j: _Joint(limits[j]) for j in JOINTS}
        self._arms = arms
        self._status_fallback = status_fallback
        self._tick = 0
        self._lock = threading.Lock()
        self._rows: list[np.ndarray] = []
        self._part = 0
        self.errors = {"snapshot": 0, "writer": 0}
        self.max_snapshot_s = 0.0
        self.closed = False
        self.dir = _new_session_dir(Path(out_dir))
        self._meta = {**meta, "schema": SCHEMA_VERSION, "columns": COLUMNS, "joints": JOINTS,
                      "joint_fields": list(JOINT_FIELDS), "side_fields": list(SIDE_FIELDS),
                      "row_fields": list(ROW_FIELDS), "stale_s": STALE_S,
                      "clock": {"perf_counter": time.perf_counter(), "time": time.time()},
                      "closed": False}
        self._write_meta()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._flush_loop, daemon=True, name="openarm-telemetry")
        self._thread.start()

    # -- per tick ----------------------------------------------------------------------------------------
    def snapshot(self, obs: dict | None, obs_t0: float, obs_read_s: float, robot=None, kind: int = KIND_OBS) -> None:
        """Record the tick whose observation is `obs`. Never raises: a failure writes a row with snapshot_ok=0."""
        t = time.perf_counter()
        ref_ns = time.time_ns()
        row = np.full(len(COLUMNS), np.nan)
        row[_COL["tick"]], row[_COL["kind"]], row[_COL["t"]] = self._tick, kind, t
        row[_COL["t_wall"]], row[_COL["obs_t0"]], row[_COL["obs_read_s"]] = ref_ns / 1e9, obs_t0, obs_read_s
        try:
            self._fill(row, obs, ref_ns, robot)
            row[_COL["snapshot_ok"]] = 1.0
        except Exception as e:  # noqa: BLE001 -- a diagnostic must never take the control loop down
            self.errors["snapshot"] += 1
            if self.errors["snapshot"] <= 3:
                logger.warning(f"telemetry snapshot failed ({type(e).__name__}: {e}); row marked snapshot_ok=0")
            keep = [_COL[c] for c in ("tick", *ROW_FIELDS)]
            kept = row[keep]
            row[:] = np.nan
            row[keep] = kept
            row[_COL["snapshot_ok"]] = 0.0
        cost = time.perf_counter() - t
        row[_COL["snapshot_s"]] = cost
        self.max_snapshot_s = max(self.max_snapshot_s, cost)
        with self._lock:
            self._rows.append(row)
        self._tick += 1

    def _fill(self, row: np.ndarray, obs, ref_ns: int, robot) -> None:
        # 1. The library's retained state FIRST: it is what the observation was read from, and no recv runs
        #    between the read and here.
        lib = {}
        for joint, motor in self._channels():
            lib[joint] = (float(motor.get_position()), float(motor.get_velocity()), float(motor.get_torque()),
                          float(motor.get_state_tmos()), float(motor.get_state_trotor()))
        # 2. Then the wire, split at the reference time: later frames belong to the next row.
        for arm in self._arms:
            if arm.tap is not None and arm.tap.available:
                arm.pending.extend(arm.tap.read())
            arm.pending.sort(key=lambda f: f[2])
            split = 0
            while split < len(arm.pending) and arm.pending[split][2] <= ref_ns:
                split += 1
            now, arm.pending = arm.pending[:split], arm.pending[split:]
            for can_id, data, t_ns in now:
                self._ingest(arm, can_id, data, t_ns)
        fallback = {}
        if self._status_fallback is not None:
            try:
                fallback = self._status_fallback() or {}
            except Exception:  # noqa: BLE001
                fallback = {}

        sent = getattr(robot, "_last_sent", None) or {}
        desired = getattr(robot, "_last_desired", None) or {}
        squeeze = getattr(robot, "gripper_squeeze_tau", None) or {}
        if robot is not None:
            row[_COL["hold"]] = float(getattr(robot, "_fault", None) is not None)
            row[_COL["motor_fault"]] = float(getattr(robot, "_motor_fault", None) is not None)
        for arm in self._arms:
            s = arm.side
            ok = arm.tap is not None and arm.tap.available
            row[_COL[f"{s}.tap_ok"]] = float(ok)
            if ok:
                row[_COL[f"{s}.rx_dropped"]] = arm.tap.dropped
                row[_COL[f"{s}.bus_errors"]] = arm.tap.errors - arm.errors_seen
                row[_COL[f"{s}.ctrl_frames"]] = arm.ctrl_frames
            arm.errors_seen = arm.tap.errors if arm.tap is not None else 0
            arm.ctrl_frames = 0
            if s in squeeze:
                row[_COL[f"{s}.squeeze_tau"]] = squeeze[s]

        tap_ok = {j: False for j in JOINTS}
        for arm in self._arms:
            for j in arm.reply.values():
                tap_ok[j] = arm.tap is not None and arm.tap.available
        for j in JOINTS:
            c = lambda f: _COL[f"{j}.{f}"]  # noqa: E731
            jw = self._joints[j]
            q, dq, tau, t_mos, t_rotor = lib[j]
            key = f"{j}.pos"
            row[c("q")] = float(obs[key]) if obs is not None and key in obs else q
            row[c("dq")], row[c("tau")], row[c("t_mos")], row[c("t_rotor")] = dq, tau, t_mos, t_rotor
            if key in sent:
                row[c("sent")] = sent[key]
            if key in desired:
                row[c("desired")] = desired[key]
            status = jw.status if jw.status is not None else fallback.get(j)
            if status is not None:
                row[c("status")] = status
            if not tap_ok[j]:
                jw.replies, jw.cmd_n = [], 0
                continue
            # Trace the library's state to the frame it was decoded from: newest match this row, else the
            # frame already matched on an earlier row if the state has not changed since.
            match = None
            for r in reversed(jw.replies):
                if _same((r[2], r[3], r[4]), (q, dq, tau)):
                    match = (r[0], r[2], r[3], r[4])
                    break
            if match is None and jw.matched is not None and _same(jw.matched[1:], (q, dq, tau)):
                match = jw.matched
            jw.matched = match
            row[c("rx_n")] = len(jw.replies)
            if jw.replies:
                row[c("rx_age_s")] = (ref_ns - jw.replies[-1][0]) / 1e9
            if match is None:
                row[c("stale")] = 1.0   # the value cannot be traced to any frame: of unknown age
            else:
                age = (ref_ns - match[0]) / 1e9
                row[c("state_age_s")] = age
                row[c("stale")] = float(age > STALE_S)
                if jw.refresh_ns is not None:
                    row[c("state_fresh")] = float(match[0] >= jw.refresh_ns)
            row[c("cmd_n")] = jw.cmd_n
            if jw.cmd is not None:
                t_ns, cq, cdq, ckp, ckd, ctau = jw.cmd
                row[c("cmd_q")], row[c("cmd_dq")], row[c("cmd_kp")] = cq, cdq, ckp
                row[c("cmd_kd")], row[c("cmd_tau")] = ckd, ctau
                row[c("cmd_age_s")] = (ref_ns - t_ns) / 1e9
            jw.replies, jw.cmd_n = [], 0

    def _ingest(self, arm: _Arm, can_id: int, data: bytes, t_ns: int) -> None:
        joint = arm.reply.get(can_id)
        if joint is not None:
            jw = self._joints[joint]
            st = decode_state(data, jw.limits)
            if st is not None:
                jw.replies.append((t_ns, *st))
                jw.status = st[0]
            return
        joint = arm.cmd.get(can_id)
        if joint is not None:
            what = ctrl_command(data)
            if what is not None:
                arm.ctrl_frames += 1
                arm.ctrl_log.append({"t_wall": t_ns / 1e9, "joint": joint, "command": what})
                return
            jw = self._joints[joint]
            mit = decode_mit(data, jw.limits)
            if mit is not None:
                jw.cmd = (t_ns, *mit)
                jw.cmd_n += 1
            return
        if can_id == REFRESH_ID and len(data) >= 3 and data[2] == 0xCC:
            joint = arm.cmd.get(data[0] | (data[1] << 8))
            if joint is not None:
                self._joints[joint].refresh_ns = t_ns

    # -- output ------------------------------------------------------------------------------------------
    def _flush(self) -> None:
        with self._lock:
            rows, self._rows = self._rows, []
        if not rows:
            return
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            block = np.stack(rows)
            arrays = [pa.array(block[:, i].astype(np.int64) if name == "tick" else block[:, i])
                      for i, name in enumerate(COLUMNS)]
            self._part += 1
            pq.write_table(pa.Table.from_arrays(arrays, names=COLUMNS), self.dir / f"part-{self._part:05d}.parquet")
        except Exception as e:  # noqa: BLE001 -- never take the recording down
            self.errors["writer"] += 1
            logger.warning(f"telemetry write failed ({type(e).__name__}: {e}); {len(rows)} rows lost")

    def _flush_loop(self) -> None:
        while not self._stop.wait(FLUSH_S):
            self._flush()

    def _write_meta(self) -> None:
        try:
            (self.dir / "meta.json").write_text(json.dumps(self._meta, indent=1, default=str))
        except Exception as e:  # noqa: BLE001
            self.errors["writer"] += 1
            logger.warning(f"telemetry meta write failed: {e}")

    def close(self, robot=None) -> None:
        """One last row (kind=1: what the wire said after the last observation, e.g. the disable frames), then
        flush and close the taps. Idempotent."""
        if self.closed:
            return
        self.closed = True
        self.snapshot(None, float("nan"), float("nan"), robot, kind=KIND_FINAL)
        self._stop.set()
        self._thread.join(timeout=FLUSH_S + 1)
        self._flush()
        for arm in self._arms:
            if arm.tap is not None:
                arm.tap.close()
        self._meta.update(closed=True, rows=self._tick, parts=self._part, errors=dict(self.errors),
                          max_snapshot_s=self.max_snapshot_s,
                          ctrl_frames=[e | {"side": a.side} for a in self._arms for e in a.ctrl_log])
        self._write_meta()
        print(f"[openarm_telemetry] {self._tick} rows in {self.dir}", flush=True)

    # -- construction --------------------------------------------------------------------------------------
    @classmethod
    def for_robot(cls, robot, out_dir: str | Path) -> "MotorTelemetry | None":
        """Telemetry for a connected openarm_umeow on CAN. Returns None (and says why) instead of raising."""
        try:
            cfg = robot.config
            send = list(cfg.send_ids) + [cfg.gripper_motor_send_id]
            recv = list(cfg.recv_ids) + [cfg.gripper_motor_recv_id]
            types = list(cfg.motor_types) + [cfg.gripper_motor_type]
            limits, arms = {}, []
            for side, port in (("R", cfg.right_port), ("L", cfg.left_port)):
                names = [f"{side}J{i}" for i in range(1, 9)]
                limits.update({n: motor_limits(t) for n, t in zip(names, types)})
                tap = make_tap(port, [*send, *recv, REFRESH_ID], fd=cfg.enable_fd)
                arms.append(_Arm(side, port, tap, send, recv, names))
            umeow = getattr(robot, "umeow_config", None)
            meta = {
                "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "robot": getattr(robot, "name", None),
                "ports": {"R": cfg.right_port, "L": cfg.left_port},
                "can_ids": {"send": send, "recv": recv, "refresh": REFRESH_ID},
                "motor_types": {n: getattr(t, "name", str(t)) for side in SIDES
                                for n, t in zip([f"{side}J{i}" for i in range(1, 9)], types)},
                "decode_limits": {j: dict(zip(("pMax", "vMax", "tMax"), limits[j])) for j in JOINTS},
                "gains": {"kp": list(robot.KPs), "kd": list(robot.KDs), "index": "J1..J7, J8 = gripper, both arms"},
                "gripper_squeeze_tau_config": getattr(umeow, "gripper_squeeze_tau", None),
                "max_joint_speed": getattr(umeow, "max_joint_speed", None),
                "tap_available": {a.side: bool(a.tap is not None and a.tap.available) for a in arms},
                "provenance": _provenance(cfg.model_path, getattr(umeow, "calibration", None)),
            }
            return cls(out_dir, robot._motor_channels, limits, arms, meta, status_fallback=robot.get_feedback_status)
        except Exception as e:  # noqa: BLE001 -- telemetry must never block a connect
            logger.warning(f"telemetry not started ({type(e).__name__}: {e})")
            print(f"[openarm_telemetry] NOT recording telemetry: {type(e).__name__}: {e}", flush=True)
            return None


def _same(a, b) -> bool:
    return all(abs(x - y) <= MATCH_TOL for x, y in zip(a, b))


def _new_session_dir(root: Path) -> Path:
    """A fresh folder per session; never reuses (and so never overwrites) an existing one."""
    root.mkdir(parents=True, exist_ok=True)
    base = root / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    path, n = base, 0
    while True:
        try:
            path.mkdir()
            return path
        except FileExistsError:
            n += 1
            path = Path(f"{base}_{n}")


def _sha256(path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except Exception:  # noqa: BLE001
        return None


def _provenance(urdf, calibration) -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=2).stdout.strip()
        except Exception:  # noqa: BLE001
            return None

    try:
        from importlib.metadata import version
        oa_version = version("openarm_can")
    except Exception:  # noqa: BLE001
        oa_version = None
    return {
        "git_sha": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "openarm_can": oa_version,
        "urdf": str(urdf), "urdf_sha256": _sha256(urdf),
        "calibration": str(calibration) if calibration else None,
        "calibration_sha256": _sha256(calibration) if calibration else None,
        "torque_calibration": None,  # tau is the motor's own estimate; nothing here calibrates it
    }


# -- consumer side (no robot needed) -------------------------------------------------------------------------
def load(session_dir: str | Path):
    """(DataFrame of every row in tick order, meta dict) of one telemetry session folder."""
    import pandas as pd

    session_dir = Path(session_dir)
    meta = json.loads((session_dir / "meta.json").read_text())
    parts = sorted(session_dir.glob("part-*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True) if parts else pd.DataFrame(columns=COLUMNS)
    return df.sort_values("tick", kind="stable").reset_index(drop=True), meta


def align_to_dataset(df, states: np.ndarray, names: list[str]) -> int:
    """Row index of `df` where a recorded episode starts: its observation.state (T x 16, columns `names`,
    e.g. "RJ1.pos") must equal the telemetry `q` columns bit-for-bit in float32 over T consecutive
    observation rows. Raises ValueError if there is no match or more than one."""
    obs = df[df["kind"] == KIND_OBS]
    tele = np.stack([obs[f"{n.removesuffix('.pos')}.q"].to_numpy() for n in names], axis=1).astype(np.float32)
    states = np.asarray(states, dtype=np.float32)
    T = len(states)
    hits = [i for i in np.flatnonzero((tele[: len(tele) - T + 1] == states[0]).all(axis=1))
            if np.array_equal(tele[i:i + T], states)]
    if len(hits) != 1:
        raise ValueError(f"episode of {T} frames matched {len(hits)} telemetry windows (need exactly 1)")
    return int(obs.index[hits[0]])
