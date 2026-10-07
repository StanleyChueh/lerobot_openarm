"""Quest packets -> mink IK -> float[16] joint command, on a background thread.

This is the dora pipeline's udp-receiver + ik pair (dora_openarm_vr/quest_receiver.py and
dora_openarm_ik/ik.py) folded into one loop, with the same behaviour:

  - It ticks at a fixed rate (ik_hz) on the NEWEST packet, like the dora graph ticking every 2 ms:
    the smoother keeps converging between packets and the solver gets one solve per tick. That
    rate matters -- openarm_control's per-solve limits (frame_position_error_limit and friends)
    were tuned at it, and solving only once per 30 Hz lerobot frame would cap the end effector at
    a fraction of the speed the operator gets in the dora pipeline.
  - Hold / anchor / reset on X and Y exactly as ik.py's --hold-until-anchor: it starts holding the
    keyframe pose; X (1st) anchors the operator's current controller pose onto the keyframe EE pose
    and goes live; X (2nd) or Y resets to the keyframe and holds it.
  - Triggers map to the finger joint (0.0 released -> open, 1.0 -> closed), read off the model.

The output is openarm_control's driver vector: right[7 joints, finger] + left[7 joints, finger], in
the IK model's (sim) joint convention. Mapping it to motor radians is the teleoperator's job.
"""

import argparse
import shlex
import threading
import time

import mujoco
import numpy as np
from openarm_control import (
    Kinematics,
    ik_params_from_args,
    register_common_args,
    register_ik_args,
    setup_from_args,
)

from .quest_input import VALID_INVALID, VALID_OK, JsonUdpReceiver, OneEuroPoseSmoother, mapped_controller_poses


def _gripper_endpoints(model: mujoco.MjModel, side: str) -> tuple[float, float]:
    """(open, closed) finger command for one arm, read off the model (ik.py)."""
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_finger1_ctrl")
    if aid < 0:
        raise RuntimeError(f"Actuator '{side}_finger1_ctrl' not found in the IK model")
    lo, hi = (float(v) for v in model.actuator_ctrlrange[aid])
    return (lo, hi) if abs(lo) > abs(hi) else (hi, lo)


def _anchored_pose(home: np.ndarray, anchor: np.ndarray, current: np.ndarray) -> np.ndarray:
    """p = p_home + (p_ctrl - p_anchor), R = R_home * (R_anchor^-1 * R_ctrl) (ik.py)."""
    delta_quat, inv_anchor, quat = np.empty(4), np.empty(4), np.empty(4)
    mujoco.mju_negQuat(inv_anchor, np.asarray(anchor[3:7], dtype=np.float64))
    mujoco.mju_mulQuat(delta_quat, inv_anchor, np.asarray(current[3:7], dtype=np.float64))
    mujoco.mju_mulQuat(quat, np.asarray(home[3:7], dtype=np.float64), delta_quat)
    mujoco.mju_normalize4(quat)
    pose = np.empty(7, dtype=np.float32)
    pose[:3] = np.asarray(home[:3]) + (np.asarray(current[:3]) - np.asarray(anchor[:3]))
    pose[3:7] = quat
    return pose


def build_kinematics(xml: str, keyframe: str, ik_args: str) -> Kinematics:
    """Kinematics configured from the dora ik node's own argument string, through openarm_control's
    own parsers -- so every IK parameter means exactly what it means in dataflow-vr-mujoco-ros2.yaml."""
    parser = argparse.ArgumentParser(add_help=False)
    register_common_args(parser)
    register_ik_args(parser)
    args = parser.parse_args(shlex.split(ik_args) + ["--xml", xml, "--keyframe", keyframe])
    return Kinematics(setup_from_args(args), ik_params_from_args(args))


class QuestIKDriver:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        xml: str,
        keyframe: str,
        ik_args: str,
        ik_hz: float,
        smoothing: tuple[float, float, float],
        hold_until_anchor: bool,
        on_button=None,
    ):
        self.kin = build_kinematics(xml, keyframe, ik_args)
        self.keyframe = keyframe
        self.ik_hz = ik_hz
        self._on_button = on_button  # called as on_button(name) on a rising edge of a / b

        model = self.kin.setup.model
        self.grip = {s: _gripper_endpoints(model, s) for s in ("right", "left")}
        # ArmSetup leaves data at the keyframe: these are the poses X/Y reset to and anchor onto.
        home_qpos = self.kin.setup.data.qpos.copy()
        self.home_pose = {
            s: np.asarray(self.kin.setup.read_ee_pose(s), dtype=np.float64) for s in self.kin.setup.sides
        }
        resolver = self.kin.setup.joint_resolver
        jr, fr = resolver.get_driver(home_qpos, "right")
        jl, fl = resolver.get_driver(home_qpos, "left")
        self.home_driver = np.concatenate([np.append(jr, float(fr)), np.append(jl, float(fl))]).astype(np.float32)
        # Hold with the grippers OPEN, not at the keyframe's closed fingers: the real gripper gets a
        # squeeze torque whenever it is commanded closed (openarm_umeow), and holding that against
        # its own stop until X is pressed is a sustained stall. Open is also how the Isaac pipeline
        # started every episode (sim_bridge_common.sim_init_pose_action).
        self.home_driver[7], self.home_driver[15] = self.grip["right"][0], self.grip["left"][0]

        self._smoothers = {s: OneEuroPoseSmoother(*smoothing) for s in ("right", "left")}
        self._prev_valid = {"right": VALID_OK, "left": VALID_OK}
        self._anchor: dict[str, np.ndarray] = {}
        self._latest_target: dict[str, np.ndarray] = {}
        self._buttons_prev = {"x": False, "y": False, "a": False, "b": False}
        self._x_presses = 0
        self._gripper = {s: self.grip[s][0] for s in ("right", "left")}  # open until a trigger reading
        self._hold: np.ndarray | None = self.home_driver.copy() if hold_until_anchor else None

        self._lock = threading.Lock()
        self._command = self.home_driver.copy()
        self._solves = 0
        self._failed_solves = 0
        self._last_solve_t: float | None = None

        self.receiver = JsonUdpReceiver(host, port)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="quest-ik")
        self._thread.start()
        if hold_until_anchor:
            print(f"[openarm_quest] holding keyframe '{keyframe}' -- press X to anchor and start driving.", flush=True)

    # ── public ────────────────────────────────────────────────────────────────

    def command(self) -> np.ndarray:
        with self._lock:
            return self._command.copy()

    def status(self) -> dict:
        _, packet_t, packets = self.receiver.latest()
        with self._lock:
            return {
                "live": self._hold is None,
                "anchored": bool(self._anchor),
                "packets": packets,
                "packet_age_s": None if packet_t is None else time.perf_counter() - packet_t,
                "solves": self._solves,
                "failed_solves": self._failed_solves,
            }

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
        self.receiver.close()

    # ── loop ──────────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        period = 1.0 / self.ik_hz
        next_t = time.perf_counter()
        while self._running:
            try:
                self._step()
            except Exception as e:  # never let one bad packet kill teleop silently
                print(f"[openarm_quest] IK step failed: {e!r}", flush=True)
            next_t += period
            remaining = next_t - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            elif remaining < -period:
                next_t = time.perf_counter()

    def _step(self) -> None:
        msg, _, _ = self.receiver.latest()
        if msg is None:
            return
        now = time.perf_counter()

        for name in ("x", "y", "a", "b"):
            if name in msg:
                pressed = bool(msg[name])
                if pressed and not self._buttons_prev[name]:
                    self._on_press(name)
                self._buttons_prev[name] = pressed

        for side, key in (("right", "rt"), ("left", "lt")):
            if key in msg:
                open_v, closed_v = self.grip[side]
                t = min(max(float(msg[key]), 0.0), 1.0)
                self._gripper[side] = open_v + t * (closed_v - open_v)
                self.kin.set_gripper(side, self._gripper[side])

        mapped = dict(zip(("right", "left"), mapped_controller_poses(msg)))
        for side, vkey in (("right", "vr"), ("left", "vl")):
            valid = int(msg.get(vkey, VALID_OK))
            if valid == VALID_INVALID:
                if self._prev_valid[side] != VALID_INVALID:
                    self._smoothers[side].reset()
                pose = None
            else:
                pose = self._smoothers[side].smooth(now, mapped[side])
            self._prev_valid[side] = valid
            if pose is not None and side in self.kin.setup.sides:
                self._latest_target[side] = np.asarray(pose, dtype=np.float64)
                self.kin.set_target(side, self._target_pose(side))

        if not self.kin.ready():
            return
        result = self.kin.solve()
        with self._lock:
            if result is None:
                self._failed_solves += 1
                return
            self._solves += 1
            # While held, the solve only pins the configuration to the keyframe; the frozen command
            # is what ships, so the arms cannot creep and the recorded action stays constant.
            self._command = (self._hold if self._hold is not None else result).copy()

    # ── buttons (ik.py) ───────────────────────────────────────────────────────

    def _on_press(self, name: str) -> None:
        if name in ("a", "b"):
            if self._on_button is not None:
                self._on_button(name)
            return
        if name == "y":
            self._x_presses = 0
            self._reset_to_home("Y")
        elif self._x_presses == 0:
            if self._anchor_now():
                self._x_presses = 1
                with self._lock:
                    self._hold = None
                print("[openarm_quest] X: anchored -- the operator is driving the arms.", flush=True)
        else:
            self._x_presses = 0
            self._reset_to_home("X")

    def _target_pose(self, side: str) -> np.ndarray:
        if self._hold is not None:
            return self.home_pose[side].astype(np.float32)
        current = self._latest_target[side]
        if side not in self._anchor:
            return current.astype(np.float32)
        return _anchored_pose(self.home_pose[side], self._anchor[side], current)

    def _anchor_now(self) -> bool:
        missing = [s for s in self.kin.setup.sides if s not in self._latest_target]
        if missing:
            print(f"[openarm_quest] X: no controller pose yet for {', '.join(missing)} -- not anchored.", flush=True)
            return False
        for side in self.kin.setup.sides:
            self._anchor[side] = self._latest_target[side].copy()
        return True

    def _reset_to_home(self, source: str) -> None:
        self.kin.sync(self.home_driver)
        command = self.home_driver.copy()
        command[7], command[15] = self._gripper["right"], self._gripper["left"]
        with self._lock:
            self._hold = command
            self._command = command.copy()
        print(f"[openarm_quest] {source}: reset to keyframe '{self.keyframe}' and holding -- X to anchor again.", flush=True)
