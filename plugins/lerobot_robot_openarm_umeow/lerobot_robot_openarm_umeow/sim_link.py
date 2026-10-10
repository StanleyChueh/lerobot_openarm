"""The link between the openarm_isaac robot (this venv) and IsaacLab's lerobot_sim_server.py (conda env).

The two run in different Python environments (lerobot's venv, Isaac Lab's conda env), so the robot talks
to the simulator over a localhost TCP socket. BOTH sides import this one file -- the server loads it by
path -- so the wire format cannot drift between them. Standard library + numpy only, for that reason.

Frames are an 8-byte length followed by a pickle of builtins (see _to_wire). Pickle executes code on load, so the server binds to
127.0.0.1 by default: only trusted local peers.

Requests (dict with "cmd") and what the server answers:

  hello                     {"protocol", "joint_names", "cameras": {name: [h, w]}, "control_hz", "task", ...}
  reset   joints=dict|None  re-randomize the scene (env.reset()), put the robot at `joints` (sim joint
                            names -> rad / m), let it settle, and answer an observation
  step    joints=dict       ONE control step (1 / control_hz of sim time) toward the joint targets, then
                            an observation. The simulator only advances on `step`: lockstep with the
                            caller, so every recorded frame is exactly one control period of sim time,
                            however fast the machine renders.
  observe                   the latest observation, without stepping

An observation is {"joints": {sim joint name: value}, "images": {camera: HxWx3 uint8}, "success": bool,
"sim_time": float, "step": int}. Any request can instead be answered {"error": "..."}.
"""

import pickle
import socket
import struct
import threading

import numpy as np

PROTOCOL = 1
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5710

_HEADER = struct.Struct("!Q")
_ND = "__ndarray__"


def _to_wire(obj):
    """Builtins only on the wire: the two sides run different numpy majors (Isaac Lab 1.x, lerobot 2.x),
    and a pickled numpy object from one does not load in the other. Arrays go as (dtype, shape, bytes)."""
    if isinstance(obj, dict):
        return {k: _to_wire(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_wire(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return {_ND: (obj.dtype.str, obj.shape, np.ascontiguousarray(obj).tobytes())}
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _from_wire(obj):
    if isinstance(obj, dict):
        if _ND in obj:
            dtype, shape, data = obj[_ND]
            return np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape).copy()  # writable, like a camera frame
        return {k: _from_wire(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_wire(v) for v in obj]
    return obj


def send_msg(sock: socket.socket, obj) -> None:
    data = pickle.dumps(_to_wire(obj), protocol=5)
    sock.sendall(_HEADER.pack(len(data)) + data)  # one call: a frame never arrives as a lone header


def _recv_exact(sock: socket.socket, n: int) -> bytearray:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        r = sock.recv_into(view[got:])
        if r == 0:
            raise ConnectionError("the other side closed the connection")
        got += r
    return buf


def recv_msg(sock: socket.socket):
    """One whole frame. Call it on a BLOCKING socket: a timeout part-way through a frame would drop the bytes
    already read and desynchronize the stream (wait for readability with select() instead)."""
    (n,) = _HEADER.unpack(_recv_exact(sock, _HEADER.size))
    return _from_wire(pickle.loads(_recv_exact(sock, n)))


class SimClient:
    """One blocking request/answer connection to lerobot_sim_server.py."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout_s: float = 60.0):
        self.host, self.port, self.timeout_s = host, port, timeout_s
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()  # lerobot-rollout's RTC thread and its main loop share the robot

    def connect(self) -> dict:
        try:
            self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        except OSError as e:
            raise ConnectionError(
                f"No Isaac Sim server at {self.host}:{self.port} ({e}). Start it first, in the IsaacLab conda env:\n"
                "  ./isaaclab.sh -p scripts/tools/lerobot_sim_server.py --enable_cameras"
            ) from e
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        hello = self.request("hello")
        if hello.get("protocol") != PROTOCOL:
            self.close()
            raise ConnectionError(f"Server speaks protocol {hello.get('protocol')}, this robot {PROTOCOL}: update both.")
        return hello

    def request(self, cmd: str, **kwargs) -> dict:
        with self._lock:
            if self._sock is None:
                raise ConnectionError("not connected to the Isaac Sim server")
            send_msg(self._sock, {"cmd": cmd, **kwargs})
            reply = recv_msg(self._sock)
        if "error" in reply:
            raise RuntimeError(f"Isaac Sim server: {reply['error']}")
        return reply

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None
