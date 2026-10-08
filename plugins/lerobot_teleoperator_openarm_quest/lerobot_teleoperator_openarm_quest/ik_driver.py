"""Quest packets -> mink IK -> float[16] joint command, on a background thread, with safety states.

The IK is the dora pipeline's (openarm_control, configured from the dora ik node's own arguments),
ticked at a fixed rate (ik_hz) on the newest packet like the dora graph. What surrounds it is NOT a
copy of the dora ik node: the dora output was low-passed by Isaac Sim's physics and a 0.3 rad/s cap
before it reached the real arm, and without that filter its weak points move the arm directly. This
driver therefore runs an explicit state machine:

  HELD       at the home pose, grippers open, waiting for X. The command is constant.
  LIVE       the arms follow the controllers.
  RETURNING  a slow, linear ramp to the home pose (every joint <= return_speed rad/s), starting from
             the command the robot last actually executed; the grippers open on arrival -> HELD.
  PAUSED     frozen where the robot last was, after a safety trip. Y discards and returns home.

Buttons:
  X  HELD -> LIVE (anchor), when x_allowed() (lerobot-record: only while it waits for X, not while it
     resets the scene). LIVE -> RETURNING with "save on arrival": the return is part of the episode, the
     "save" decision is sent when the arms reach home, and the grippers open only after that (at once
     without a recorder; when the recorder says the episode has ended with one -- release_grippers()).
     Ignored while RETURNING or PAUSED.
  Y  LIVE / PAUSED: "discard the episode" at once, then RETURNING; grippers open on arrival. Ignored
     otherwise.
  Triggers drive the grippers, LIVE only.

Safety, and why each is needed:

  - The reference pose is latched on X. The packets' rf is the headset's LIVE pose, and the headset
    hangs at the operator's neck: in the dora mapping every swing, tilt or re-localisation of it
    moves every target even when the hands are still (5 degrees at 0.5 m is 4 cm). From X to the
    next reset, targets are relative to the headset pose at that X only.
  - X anchors only on FRESH, VALID poses of both controllers, from the current packet -- never from
    a pose remembered from before (a controller that had gone to sleep or lost tracking).
  - A controller that loses tracking and gets it back is re-anchored (clutched) where it reappears,
    so the hand's untracked motion is not replayed as a jump.
  - A tracked pose moving faster than glitch_speed_mps / glitch_rot_speed_dps PAUSES both arms: a
    tracking glitch (a pose snapping several cm within one headset frame), or a motion too fast to follow
    safely. The speed is the change since the previous packet over the time between them, by the
    headset's own clock (packet field "t"; arrival time if absent) -- NOT a distance per processed packet:
    if this thread is delayed for a moment (e.g. the video encoders starting at the first recorded frame),
    an ordinary hand movement over that gap must not look like a jump. A gap longer than pose_fresh_s
    re-anchors the hand where it is instead (as after lost tracking): no catch-up rush either.
  - The IK follows a target step completely in a single solve, so any discontinuity becomes a joint
    jump. A raw solution that moves more than ik_jump_rad in one solve PAUSES; the shipped command is
    additionally rate-limited to max_joint_speed, and if the solution runs more than max_lead_rad
    ahead of it, that PAUSES too.
  - If the robot's own guard refuses a command (shared.ROBOT_STATE), the teleop PAUSES at the pose
    the robot holds, so the two agree and X can resume without a jump.

Output: openarm_control's driver vector, right[7 joints, finger] + left[7 joints, finger], in the IK
model's (sim) joint convention. Mapping to motor radians is the teleoperator's job.
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
from lerobot_robot_openarm_umeow.shared import ROBOT_STATE, TELEOP_STATE
from scipy.spatial.transform import Rotation

from .quest_input import (
    VALID_INVALID,
    VALID_OK,
    JsonUdpReceiver,
    OneEuroPoseSmoother,
    mapped_controller_poses,
    reference_pose,
)

HELD, LIVE, RETURNING, PAUSED = "HELD", "LIVE", "RETURNING", "PAUSED"
SIDES = ("right", "left")
ARM = np.r_[0:7, 8:15]  # driver-vector indices of the 14 arm joints (7 and 15 are the fingers)
GRIP = {"right": 7, "left": 15}
VALID_KEY = {"right": "vr", "left": "vl"}


def _gripper_endpoints(model: mujoco.MjModel, side: str) -> tuple[float, float]:
    """(open, closed) finger command for one arm, read off the model (ik.py)."""
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_finger1_ctrl")
    if aid < 0:
        raise RuntimeError(f"Actuator '{side}_finger1_ctrl' not found in the IK model")
    lo, hi = (float(v) for v in model.actuator_ctrlrange[aid])
    return (lo, hi) if abs(lo) > abs(hi) else (hi, lo)


def _anchored_pose(base: np.ndarray, anchor: np.ndarray, current: np.ndarray) -> np.ndarray:
    """p = p_base + (p_ctrl - p_anchor), R = R_base * (R_anchor^-1 * R_ctrl) (ik.py)."""
    delta_quat, inv_anchor, quat = np.empty(4), np.empty(4), np.empty(4)
    mujoco.mju_negQuat(inv_anchor, np.asarray(anchor[3:7], dtype=np.float64))
    mujoco.mju_mulQuat(delta_quat, inv_anchor, np.asarray(current[3:7], dtype=np.float64))
    mujoco.mju_mulQuat(quat, np.asarray(base[3:7], dtype=np.float64), delta_quat)
    mujoco.mju_normalize4(quat)
    pose = np.empty(7, dtype=np.float32)
    pose[:3] = np.asarray(base[:3]) + (np.asarray(current[:3]) - np.asarray(anchor[:3]))
    pose[3:7] = quat
    return pose


def _pose_jump(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(position distance m, rotation angle deg) between two [x y z qw qx qy qz] poses."""
    ra = Rotation.from_quat([a[4], a[5], a[6], a[3]])
    rb = Rotation.from_quat([b[4], b[5], b[6], b[3]])
    return float(np.linalg.norm(a[:3] - b[:3])), float(np.degrees((ra.inv() * rb).magnitude()))


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
        max_joint_speed: float,
        return_speed: float,
        glitch_speed_mps: float,
        glitch_rot_speed_dps: float,
        ik_jump_rad: float,
        max_lead_rad: float,
        pose_fresh_s: float,
        held_command=None,
        robot_fault=None,
        on_episode_key=None,
    ):
        self.kin = build_kinematics(xml, keyframe, ik_args)
        self.port = port
        self.keyframe = keyframe
        self.ik_hz = ik_hz
        self.max_joint_speed = max_joint_speed
        self.return_speed = return_speed
        self.glitch_speed_mps = glitch_speed_mps
        self.glitch_rot_speed_dps = glitch_rot_speed_dps
        self.ik_jump_rad = ik_jump_rad
        self.max_lead_rad = max_lead_rad
        self.pose_fresh_s = pose_fresh_s
        # held_command() -> driver16 the robot last executed (or None); robot_fault() -> str | None;
        # on_episode_key("right" | "left"). All optional, so the driver also runs without a robot.
        self._held_command = held_command or (lambda: None)
        self._robot_fault = robot_fault or (lambda: None)
        episode_key = on_episode_key or (lambda key: None)

        def on_key(key: str) -> None:
            TELEOP_STATE.episode_key(key)
            episode_key(key)

        self._on_episode_key = on_key

        setup = self.kin.setup
        self.grip = {s: _gripper_endpoints(setup.model, s) for s in SIDES}
        home_qpos = setup.data.qpos.copy()  # ArmSetup leaves data at the keyframe
        jr, fr = setup.joint_resolver.get_driver(home_qpos, "right")
        jl, fl = setup.joint_resolver.get_driver(home_qpos, "left")
        self.home_driver = np.concatenate([np.append(jr, float(fr)), np.append(jl, float(fl))]).astype(np.float32)
        # Home always has the grippers OPEN: the robot squeezes a gripper commanded closed, and holding
        # that against its own stop while waiting for X is a sustained stall.
        for s in SIDES:
            self.home_driver[GRIP[s]] = self.grip[s][0]

        self._smoothers = {s: OneEuroPoseSmoother(*smoothing) for s in SIDES}
        self._state = HELD
        self._reason = ""
        TELEOP_STATE.publish(HELD, "")
        self._ref = None  # reference pose latched on X
        self._base: dict[str, np.ndarray] = {}  # EE pose per arm at anchoring
        self._anchor: dict[str, np.ndarray] = {}  # controller pose per arm at anchoring
        self._last_raw: dict[str, np.ndarray | None] = {s: None for s in SIDES}
        self._last_raw_t: dict[str, float | None] = {s: None for s in SIDES}  # packet time of _last_raw
        self._reach = {s: {"since": None, "told": False} for s in SIDES}  # target-out-of-reach notices
        self._reach_check_t = 0.0
        self._last_target: dict[str, np.ndarray] = {}
        self._lost = {s: False for s in SIDES}
        self._gripper = {s: self.grip[s][0] for s in SIDES}
        self._ik_prev: np.ndarray | None = None
        self._return: tuple[np.ndarray, float, float] | None = None  # (start, t0, duration)
        self._buttons_prev = {"x": False, "y": False, "a": False, "b": False}
        self._last_count = 0
        self._last_t = time.perf_counter()
        self._stale_warned = False
        self._started_t = time.perf_counter()
        self._packet_warn_t = 0.0
        self._return_request: tuple[str, bool] | None = None  # (source, save), set from other threads
        self._release_request = False  # open the grippers held closed after a save (release_grippers)
        self._save_on_arrival = False
        # Hooks for a recorder (record_gate.py): whether X may start driving now, and whether the grippers
        # must stay as they are until the recorder confirms the saved episode has ended.
        self.x_allowed = lambda: True
        self.defer_gripper_open = False

        self._lock = threading.Lock()
        self._command = self.home_driver.copy()
        self._solves = 0
        self._failed_solves = 0

        self.receiver = JsonUdpReceiver(host, port)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="quest-ik")
        self._thread.start()
        self._log(f"HELD at '{keyframe}' with grippers open -- press X to anchor and start driving.")

    # ── public ────────────────────────────────────────────────────────────────

    def command(self) -> np.ndarray:
        with self._lock:
            return self._command.copy()

    def request_return(self, source: str, save: bool = False) -> None:
        """Ask for the slow return home; with save=True the "save" decision is sent on arrival (as for the
        2nd X). Served on the IK thread; ignored unless LIVE or PAUSED."""
        self._return_request = (source, save)

    def release_grippers(self) -> None:
        """Open grippers held closed after a save, once the recorder has ended the episode."""
        self._release_request = True

    def status(self) -> dict:
        _, packet_t, packets = self.receiver.latest()
        with self._lock:
            return {
                "state": self._state,
                "reason": self._reason,
                "live": self._state == LIVE,
                "packets": packets,
                "packet_age_s": None if packet_t is None else time.perf_counter() - packet_t,
                "solves": self._solves,
                "failed_solves": self._failed_solves,
            }

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
        self.receiver.close()
        TELEOP_STATE.publish(None, "")

    # ── loop ──────────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        period = 1.0 / self.ik_hz
        next_t = time.perf_counter()
        while self._running:
            try:
                self._step()
            except Exception as e:
                self._pause(f"internal error {e!r}")
            next_t += period
            remaining = next_t - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            elif remaining < -period:
                next_t = time.perf_counter()

    def _step(self) -> None:
        msg, msg_t, count = self.receiver.latest()
        now = time.perf_counter()
        dt = min(max(now - self._last_t, 0.0), 0.05)
        self._last_t = now
        new_packet = count != self._last_count
        self._last_count = count
        fresh = msg is not None and msg_t is not None and now - msg_t < self.pose_fresh_s
        if new_packet:
            self._publish_quest(msg, msg_t, count)
        self._warn_if_no_packets(now, msg_t)

        request, self._return_request = self._return_request, None
        if request and self._state in (LIVE, PAUSED):
            self._start_return(request[0], save=request[1] and self._state == LIVE)
        if self._release_request:
            self._release_request = False
            if self._state == HELD:
                with self._lock:
                    for s in SIDES:
                        self._command[GRIP[s]] = self.home_driver[GRIP[s]]
                self._log("grippers opened -- press X for the next episode.")

        fault = self._robot_fault()
        if fault and self._state != PAUSED:
            self._pause(f"the robot refused a command ({fault})")

        if msg is not None and new_packet:
            for name in ("x", "y", "a", "b"):
                pressed = bool(msg.get(name, False))
                if pressed and not self._buttons_prev[name]:
                    self._on_press(name, msg, fresh, now)
                self._buttons_prev[name] = pressed

        if self._state == RETURNING:
            self._advance_return(now)
        if self._state != LIVE:
            return

        if not fresh:
            if not self._stale_warned:
                self._log("no fresh Quest packets -- holding the arms where they are.")
                self._stale_warned = True
            return
        self._stale_warned = False

        for side, key in (("right", "rt"), ("left", "lt")):
            if key in msg:
                open_v, closed_v = self.grip[side]
                t = min(max(float(msg[key]), 0.0), 1.0)
                self._gripper[side] = open_v + t * (closed_v - open_v)

        raw = dict(zip(SIDES, mapped_controller_poses(msg, self._ref)))
        packet_t = float(msg["t"]) if isinstance(msg.get("t"), (int, float)) else msg_t  # headset clock if sent
        headset_lost = int(msg.get("v", VALID_OK)) == VALID_INVALID  # controllers are tracked by it
        for side in SIDES:
            pose = raw[side]
            if headset_lost or int(msg.get(VALID_KEY[side], VALID_OK)) == VALID_INVALID or pose is None:
                if not self._lost[side]:
                    self._log(f"{side} controller lost tracking -- that arm holds its last target.")
                self._lost[side] = True
                self.kin.set_target(side, self._last_target[side])  # the solver needs both arms' targets
                continue
            if self._lost[side]:
                # Onto the target the IK is already tracking for this arm, so the solver sees no step.
                self._clutch(side, pose, now, base=self._last_target[side])
                self._log(f"{side} controller tracked again -- re-anchored where it reappeared, no jump.")
            elif new_packet and self._last_raw_t[side] is not None and packet_t - self._last_raw_t[side] > self.pose_fresh_s:
                # A gap in packets (or in this thread): the hand's motion meanwhile is not replayed as a
                # catch-up rush -- re-anchor where it is now, as after lost tracking.
                gap_ms = (packet_t - self._last_raw_t[side]) * 1000
                self._clutch(side, pose, now, base=self._last_target[side])
                self._log(f"{side}: {gap_ms:.0f} ms without Quest packets -- re-anchored where the hand is now, no jump.")
            elif new_packet and self._last_raw[side] is not None and self._last_raw_t[side] is not None:
                dp, dr = _pose_jump(self._last_raw[side], pose)
                dt = max(packet_t - self._last_raw_t[side], 1.0 / 90.0)  # at least one headset frame
                if dp / dt > self.glitch_speed_mps or dr / dt > self.glitch_rot_speed_dps:
                    self._pause(f"{side} controller moved {dp * 100:.0f} cm / {dr:.0f} deg in {dt * 1000:.0f} ms"
                                f" ({dp / dt:.1f} m/s, {dr / dt:.0f} deg/s): tracking glitch or too fast")
                    return
            if new_packet:
                self._last_raw[side] = pose
                self._last_raw_t[side] = packet_t
            smoothed = self._smoothers[side].smooth(now, pose)
            self._last_target[side] = _anchored_pose(self._base[side], self._anchor[side], smoothed)
            self.kin.set_target(side, self._last_target[side])

        if not self.kin.ready():
            return
        result = self.kin.solve()
        if result is None:
            self._failed_solves += 1
            return
        if self._ik_prev is not None:
            jump = float(np.abs(result[ARM] - self._ik_prev[ARM]).max())
            if jump > self.ik_jump_rad:
                self._pause(f"IK solution jumped {jump:.2f} rad in one solve")
                return
        self._ik_prev = result.copy()
        self._check_reach(result, now)

        with self._lock:
            cmd = self._command.copy()
        step = self.max_joint_speed * dt
        cmd[ARM] += np.clip(result[ARM] - cmd[ARM], -step, step)
        for s in SIDES:
            cmd[GRIP[s]] = self._gripper[s]
        lead = float(np.abs(result[ARM] - cmd[ARM]).max())
        if lead > self.max_lead_rad:
            self._pause(f"the IK target ran {lead:.2f} rad ahead of what the arm may follow"
                        f" at {self.max_joint_speed:g} rad/s")
            return
        with self._lock:
            self._command = cmd
            self._solves += 1

    def _check_reach(self, result: np.ndarray, now: float) -> None:
        """Say so (terminal + rerun panel) when an arm cannot follow its target: the IK stops short at a
        joint limit, or slows at a straight-arm / aligned-wrist (singular) pose. Display only."""
        if now - self._reach_check_t < 0.2:
            return
        self._reach_check_t = now
        right, left = self.kin.fk_bimanual(result[0:8], result[8:16])
        for side, ee in (("right", right), ("left", left)):
            gap = float(np.linalg.norm(np.asarray(ee[:3]) - np.asarray(self._last_target[side][:3])))
            r = self._reach[side]
            if gap > 0.03:
                r["since"] = r["since"] or now
                if not r["told"] and now - r["since"] > 0.5:
                    r["told"] = True
                    self._log(f"{side} arm cannot reach the target ({gap * 100:.0f} cm short): a joint limit or a"
                              " straight-arm / aligned-wrist pose -- move that hand back toward the robot.")
            else:
                if r["told"]:
                    self._log(f"{side} arm is following again.")
                r["since"], r["told"] = None, False

    # ── state changes ─────────────────────────────────────────────────────────

    def _on_press(self, name: str, msg: dict, fresh: bool, now: float) -> None:
        if name == "y":
            if self._state in (LIVE, PAUSED):
                self._log("Y: DISCARD the episode.")
                self._on_episode_key("left")
                self._start_return("Y", save=False)
            else:
                self._log(f"Y ignored: nothing to discard while {self._state}.")
        elif name != "x":
            return  # A / B: no function
        elif self._state == HELD:
            if self.x_allowed():
                self._go_live(msg, fresh, now)
            else:
                self._log("X ignored: lerobot is resetting the scene -- wait for WAITING for X.")
        elif self._state == LIVE:
            self._start_return("X", save=True)
        elif self._state == PAUSED:
            self._log("X ignored while PAUSED -- press Y to discard the episode and return home.")
        else:  # RETURNING
            self._log("X ignored: still returning home -- wait for HELD.")

    def _go_live(self, msg: dict, fresh: bool, now: float) -> None:
        if ROBOT_STATE.connecting:
            self._log("X ignored: the robot is still starting up (moving to home) -- press X when it is done.")
            return
        if not fresh:
            self._log("X ignored: no fresh Quest packet to anchor on.")
            return
        if self._robot_fault():
            self._log("X ignored: the robot is still holding after a refused command -- press X again.")
            return
        if int(msg.get("v", VALID_OK)) == VALID_INVALID:
            self._log("X ignored: the headset reports lost tracking.")
            return
        for side in SIDES:
            if int(msg.get(VALID_KEY[side], VALID_OK)) == VALID_INVALID:
                self._log(f"X ignored: the {side} controller is not tracked -- wake it / bring it into view.")
                return
        # Latch the reference from THIS packet; every pose until the next reset is relative to it.
        self._ref = reference_pose(msg)
        raw = dict(zip(SIDES, mapped_controller_poses(msg, self._ref)))
        if any(raw[s] is None for s in SIDES):
            self._log("X ignored: a controller pose is missing from the packet.")
            return
        with self._lock:
            cmd = self._command.copy()
        self.kin.sync(cmd)  # the IK starts exactly at the joints being shipped
        for side in SIDES:
            self._clutch(side, raw[side], now, cmd)
            self._gripper[side] = float(cmd[GRIP[side]])
        self._ik_prev = None
        prev = self._state
        self._set_state(LIVE, "")
        self._log(f"X: anchored ({'resumed from the paused pose' if prev == PAUSED else 'from home'})"
                  " -- the arms follow the controllers.")

    def _clutch(self, side: str, pose: np.ndarray, now: float, cmd: np.ndarray | None = None,
                base: np.ndarray | None = None) -> None:
        """Anchor one arm: from now on its EE target is `base` (default: the EE pose of `cmd`, or of the
        current command) moved by this controller's motion away from `pose`."""
        if base is None:
            if cmd is None:
                with self._lock:
                    cmd = self._command.copy()
            right, left = self.kin.fk_bimanual(cmd[0:8], cmd[8:16])
            base = right if side == "right" else left
        self._base[side] = np.asarray(base, dtype=np.float64)
        self._anchor[side] = np.asarray(pose, dtype=np.float64)
        self._smoothers[side].reset()
        self._smoothers[side].smooth(now, pose)
        self._last_raw[side] = pose
        self._last_raw_t[side] = None  # no speed check against the packet before the anchor
        self._last_target[side] = self._base[side].astype(np.float32)
        self._lost[side] = False

    def _pause(self, reason: str) -> None:
        if self._state == PAUSED and self._reason == reason:
            return  # already paused for this; do not re-log it every tick
        held = self._held_command()
        with self._lock:
            if held is not None:
                self._command[ARM] = held[ARM]  # freeze exactly where the robot is commanded to be
            cmd = self._command.copy()
        self.kin.sync(cmd)
        self._ik_prev = None
        self._return = None
        self._set_state(PAUSED, reason)
        self._log(f"PAUSED: {reason}. Y = discard the episode and return home.")

    def _start_return(self, source: str, save: bool = False) -> None:
        held = self._held_command()
        with self._lock:
            start = self._command.copy()
        if held is not None:
            start[ARM] = held[ARM]  # start from what the robot actually executed
        dist = float(np.abs(self.home_driver[ARM] - start[ARM]).max())
        duration = max(dist / self.return_speed, 0.5)
        with self._lock:
            self._command = start.copy()
        self._return = (start, time.perf_counter(), duration)
        self._save_on_arrival = save
        self._set_state(RETURNING, source)
        self._log(f"{source}: returning home slowly ({dist:.2f} rad over {duration:.1f} s,"
                  f" <= {self.return_speed:g} rad/s)"
                  + ("; the episode is SAVED on arrival, return included." if save else "; grippers open on arrival."))

    def _advance_return(self, now: float) -> None:
        start, t0, duration = self._return
        a = min((now - t0) / duration, 1.0)
        cmd = start.copy()
        cmd[ARM] = start[ARM] + a * (self.home_driver[ARM] - start[ARM])
        if a >= 1.0:
            cmd = self.home_driver.copy()
            keep_grippers = self._save_on_arrival and self.defer_gripper_open
            if keep_grippers:  # the saved episode ends with the arms home and the grippers as they were
                for s in SIDES:
                    cmd[GRIP[s]] = start[GRIP[s]]
            self.kin.sync(cmd)
            self._return = None
            self._set_state(HELD, "")
            if self._save_on_arrival:
                self._save_on_arrival = False
                self._log("home: SAVE the episode.")
                self._on_episode_key("right")
            self._log("HELD at home" + (" -- grippers open once the episode is saved." if keep_grippers
                                        else ", grippers open -- press X for the next episode."))
        with self._lock:
            self._command = cmd

    def _publish_quest(self, msg: dict, msg_t: float, count: int) -> None:
        """Summarise the newest packet for the rerun status panel."""

        def pos(key: str):
            c = msg.get(key)
            return None if c is None else tuple(round(float(c[a]), 3) for a in ("x", "y", "z"))

        def f(key: str) -> float:
            return float(msg.get(key, 0.0) or 0.0)

        quest = {
            "v": int(msg.get("v", VALID_OK)), "vr": int(msg.get("vr", VALID_OK)), "vl": int(msg.get("vl", VALID_OK)),
            "buttons": [b.upper() for b in ("x", "y", "a", "b") if msg.get(b)],
            "rt": f("rt"), "lt": f("lt"), "rg": f("rg"), "lg": f("lg"),
            "rstick": (f("rsx"), f("rsy")), "lstick": (f("lsx"), f("lsy")),
            "rc": pos("rc"), "lc": pos("lc"), "rf": pos("rf"),
        }
        TELEOP_STATE.publish_quest(quest, count, msg_t, self.receiver.sender)

    def _warn_if_no_packets(self, now: float, msg_t: float | None) -> None:
        """Say so in the terminal, every 5 s, while the Quest is silent."""
        if now - self._packet_warn_t < 5.0:
            return
        if msg_t is None and now - self._started_t > 3.0:
            self._log(f"NO packets from the Quest on UDP port {self.port} yet: is the Quest app running and"
                      " sending to this PC's IP and port, and is the dora dataflow stopped?")
            self._packet_warn_t = now
        elif msg_t is not None and now - msg_t > 2.0:
            self._log(f"Quest packets STOPPED {now - msg_t:.0f} s ago -- the arms hold. Headset asleep or app closed?")
            self._packet_warn_t = now

    def _set_state(self, state: str, reason: str) -> None:
        with self._lock:
            self._state, self._reason = state, reason
        TELEOP_STATE.publish(state, reason)

    @staticmethod
    def _log(text: str) -> None:
        print(f"[openarm_quest] {text}", flush=True)
        TELEOP_STATE.log(text)
