"""Quest UDP input: receiver, left-to-right-handed pose mapping and smoothing.

Copied from dora-openarm-data-collection/nodes/dora-openarm-vr (udp_receiver.py, smoothing.py and
the pose-mapping half of quest_receiver.py) with the dora/pyarrow plumbing and the matplotlib
viewers removed. The maths and constants are unchanged, so the same controller motion produces the
same end-effector targets as the dora pipeline. Keep them in step if one side is retuned.

Incoming JSON (one datagram per headset frame): t; lc / rc / rf pose objects {x, y, z, qx, qy, qz,
qw} in Unity left-handed world coordinates; lt / rt triggers and lg / rg grips 0..1; lsx / lsy /
rsx / rsy sticks; a / b / x / y buttons; v / vl / vr validity (0 OK, 1 STALE, 2 INVALID).
"""

import collections
import json
import select
import socket
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

# ── Frame alignment (quest_receiver.py) ─────────────────────────────────────────
_FRAME_ROT = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
FRAME_OFFSET_NECK = np.array([-0.2, 0, -0.3], dtype=np.float64)
_R_FRAME = Rotation.from_matrix(_FRAME_ROT)
_R_FIX = Rotation.from_euler("z", 90, degrees=True)
_IDENTITY_REF = {"x": 0.0, "y": 0.0, "z": 0.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}

VALID_OK, VALID_STALE, VALID_INVALID = 0, 1, 2


class JsonUdpReceiver:
    """Background thread that binds a UDP socket and keeps the latest parsed JSON packet."""

    def __init__(self, host: str, port: int, buf_size: int = 4096) -> None:
        self._host, self._port, self._buf_size = host, port, buf_size
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._latest_t: float | None = None  # perf_counter() of the newest packet
        self._count = 0
        self.sender: str | None = None  # "ip:port" of the Quest, once a packet has arrived
        # Network delay, from the headset's own clock: arrival time minus the packet's "t", relative to the
        # smallest such offset seen (the best-case path) = the EXTRA delay of this packet (the two clocks'
        # constant offset cancels). Plus the arrival gaps longer than 100 ms.
        self._offset_best: float | None = None
        self._extra_s: float | None = None
        self._gaps: collections.deque = collections.deque()  # arrival times of >100 ms gaps
        self._prev_arrival: float | None = None
        self.has_headset_clock: bool | None = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="quest-udp")
        self._thread.start()

    def latest(self) -> tuple[dict | None, float | None, int]:
        """(newest packet, its arrival perf_counter(), packets received so far)."""
        with self._lock:
            return self._latest, self._latest_t, self._count

    def net_stats(self) -> dict:
        """{"extra_ms": extra one-way delay vs the best seen (None without "t"), "gaps_10s": arrival gaps
        over 100 ms in the last 10 s, "has_t": whether packets carry the headset clock}."""
        now = time.perf_counter()
        with self._lock:
            while self._gaps and now - self._gaps[0] > 10.0:
                self._gaps.popleft()
            return {"extra_ms": None if self._extra_s is None else self._extra_s * 1000.0,
                    "gaps_10s": len(self._gaps), "has_t": self.has_headset_clock}

    def _measure(self, msg: dict, arrival: float) -> None:
        t = msg.get("t")
        with self._lock:
            if self._prev_arrival is not None and arrival - self._prev_arrival > 0.1:
                self._gaps.append(arrival)
            self._prev_arrival = arrival
            if not isinstance(t, (int, float)):
                self.has_headset_clock = False
                return
            self.has_headset_clock = True
            offset = arrival - float(t)
            if self._offset_best is None or offset < self._offset_best or offset - self._offset_best > 5.0:
                self._offset_best = offset  # first packet, a faster path, or the app restarted (clock jump)
            self._extra_s = offset - self._offset_best

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)

    @staticmethod
    def _parse(data: bytes) -> dict | None:
        try:
            line = data.decode("utf-8", errors="replace").strip()
            return json.loads(line) if line else None
        except json.JSONDecodeError:
            return None

    def _loop(self) -> None:
        while self._running:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as srv:
                    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    srv.bind((self._host, self._port))
                    srv.settimeout(0.5)
                    print(f"[openarm_quest] listening for the Quest on UDP {self._host}:{self._port}", flush=True)
                    while self._running:
                        try:
                            data, addr = srv.recvfrom(self._buf_size)
                        except TimeoutError:
                            continue
                        if self.sender is None:
                            self.sender = f"{addr[0]}:{addr[1]}"
                            print(f"[openarm_quest] first Quest packet received from {self.sender}", flush=True)
                        last, n = self._parse(data), 1
                        # Drain anything queued and keep only the freshest packet.
                        while select.select([srv], [], [], 0.0)[0]:
                            parsed = self._parse(srv.recvfrom(self._buf_size)[0])
                            n += 1
                            if parsed is not None:
                                last = parsed
                        if last is not None:
                            arrival = time.perf_counter()
                            self._measure(last, arrival)
                            with self._lock:
                                self._latest, self._latest_t = last, arrival
                                self._count += n
            except OSError as e:
                if self._running:
                    print(f"[openarm_quest] UDP socket error ({e}); retrying in 1 s", flush=True)
                    time.sleep(1.0)


def parse_lh_to_rh(c: dict) -> tuple[np.ndarray, Rotation]:
    """Unity left-handed pose dict -> right-handed (position, Rotation): z -> -z, qx -> -qx, qy -> -qy."""
    pos = np.array([c["x"], c["y"], -c["z"]], dtype=np.float64)
    rot = Rotation.from_quat([-c["qx"], -c["qy"], c["qz"], c["qw"]])
    return pos, rot


def pose_to_array(pos: np.ndarray, rot: Rotation) -> np.ndarray:
    q = rot.as_quat()
    return np.array([pos[0], pos[1], pos[2], q[3], q[0], q[1], q[2]], dtype=np.float32)


def reference_pose(msg: dict) -> tuple[np.ndarray, Rotation]:
    """The packet's rf reference pose (the headset, worn at the neck), right-handed."""
    return parse_lh_to_rh(msg.get("rf") or _IDENTITY_REF)


def mapped_controller_poses(
    msg: dict, reference: tuple[np.ndarray, Rotation] | None = None
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """(right, left) controller poses mapped into the robot arm_origin frame, [x y z qw qx qy qz].

    Relative to a reference pose, then rotated by _R_FRAME, offset by FRAME_OFFSET_NECK and
    post-rotated by R_FIX -- quest_receiver.py's QuestPoseProcessor. The reference defaults to this
    packet's live rf (what the dora pipeline does); pass a latched one to stop the headset's own
    motion -- it hangs at the operator's neck -- from moving the targets.
    """
    p_ref, r_ref = reference if reference is not None else reference_pose(msg)
    r_ref_inv = r_ref.inv()

    def convert(packet: dict | None) -> np.ndarray | None:
        if packet is None:
            return None
        p_world, r_world = parse_lh_to_rh(packet)
        p_rel = r_ref_inv.apply(p_world - p_ref)
        r_rel = r_ref_inv * r_world
        return pose_to_array(_R_FRAME.apply(p_rel) + FRAME_OFFSET_NECK, _R_FRAME * r_rel * _R_FIX)

    return convert(msg.get("rc")), convert(msg.get("lc"))


def _slerp_quat(q1: np.ndarray, q2: np.ndarray, alpha: float) -> np.ndarray:
    dot = np.dot(q1, q2)
    if dot < 0.0:
        q2, dot = -q2, -dot
    if dot > 0.9995:
        res = q1 + alpha * (q2 - q1)
        return res / np.linalg.norm(res)
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * alpha
    s0 = np.cos(theta) - dot * np.sin(theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return s0 * q1 + s1 * q2


class OneEuroPoseSmoother:
    """1 Euro Filter on position (adaptive cutoff) and rotation (SLERP)."""

    def __init__(self, min_cutoff: float = 10.0, beta: float = 0.8, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self) -> None:
        self.p_prev = None
        self.q_prev = None
        self.dp_prev = np.zeros(3)
        self.t_prev = None

    def smooth(self, t: float, target_pose: np.ndarray | None) -> np.ndarray | None:
        if target_pose is None:
            return None
        t_p, t_q = target_pose[0:3], target_pose[3:7]
        if self.t_prev is None or self.p_prev is None:
            self.p_prev, self.q_prev, self.t_prev = t_p.copy(), t_q.copy(), t
            return target_pose.copy()
        dt = t - self.t_prev
        if dt <= 0.0:
            return target_pose.copy()

        def alpha(dt: float, cutoff: float) -> float:
            return 1.0 / (1.0 + (1.0 / (2 * np.pi * cutoff)) / dt)

        dp_filtered = alpha(dt, self.d_cutoff) * (t_p - self.p_prev) / dt + (1.0 - alpha(dt, self.d_cutoff)) * self.dp_prev
        alpha_p = alpha(dt, self.min_cutoff + self.beta * np.linalg.norm(dp_filtered))
        p_hat = self.p_prev + alpha_p * (t_p - self.p_prev)
        q_hat = _slerp_quat(self.q_prev, t_q, alpha_p)
        self.p_prev, self.q_prev, self.dp_prev, self.t_prev = p_hat, q_hat, dp_filtered, t
        return np.array([*p_hat, *q_hat], dtype=np.float32)
