"""lerobot Robot wrapper around robots/umeow_openarm_follower for lerobot-record / lerobot-rollout.

What it adds to the follower, and why:

  - send_action(action) with ONE argument, as lerobot's record and rollout loops call it. The
    follower's own send_action(action, target_vel) requires the velocity feed-forward argument;
    here it is derived from consecutive commands.
  - A per-tick step limit (max_joint_speed), the role max_relative_target plays in the official
    robots.
  - A safety guard that REFUSES a request whose arm joints jump in one tick or sit far from the
    measured joints, and holds the arm until requests come back near the held pose -- see
    OpenArmUmeowConfig.max_command_jump / max_tracking_error. The state is published to
    shared.ROBOT_STATE so the openarm_quest teleop pauses with it instead of fighting it.
  - A speed-limited approach to the start pose on connect, with mirror_bridge.py's typed-YES and
    max-delta gates, and a ramp back to the connect-time pose before de-energising on disconnect.
  - The gripper squeeze torque mirror_bridge.py applies while a gripper is commanded closed.
  - A motor watchdog. A Damiao motor that trips its own protection (over-current from twisting a joint
    against its stop or cable, over-temperature, lost comms) switches itself OFF: the joint goes limp
    and stays wherever it is, while the follower keeps reporting its last position -- the arm "will
    not return home" with nothing in the logs. Every 0.5 s the motors' status nibbles are checked; a
    fault holds the arm (the teleop pauses with it), says which motor, why, and its temperatures, and
    the hold is not released while the motor stays faulted. And a joint that stays more than 0.15
    rad from a steady command for 2 s is reported as not following.
"""

import logging
import time

from .common import (
    MOTOR_KEYS,
    keyframe_sim_joints,
    load_calibration,
    sim_joints_to_motor_action,
)
from .config_openarm_umeow import OpenArmUmeowConfig
from .rerun_status import BOARD
from .shared import ROBOT_STATE

from robots.umeow_openarm_follower import OpenArmFollower, OpenArmFollowerConfig  # noqa: E402  (after common's sys.path)
from robots.umeow_openarm_follower.can_monitor import is_fault, status_name  # noqa: E402
from sim_bridge_common import (  # noqa: E402
    GRIPPER_SIM_OPEN,
    approach_pose,
    check_arms_not_crossed,
    clamp_step,
    compute_target_velocity,
    get_current_pos_action,
)

logger = logging.getLogger(__name__)

STATS_PERIOD_S = 5.0
ARM_KEYS = [k for k in MOTOR_KEYS if not k.endswith("8.pos")]
HEALTH_PERIOD_S = 0.5
LAG_RAD, LAG_S = 0.15, 2.0
HOME_TOL_RAD = 0.1  # a reset counts as done only with every arm joint this close to the start pose  # a joint this far from a steady command for this long is not following


class _RawSender:
    """Hands sim_bridge_common's ramp helpers the follower's unclamped two-argument send_action."""

    def __init__(self, robot: "OpenArmUmeow"):
        self._robot = robot

    def send_action(self, action, target_vel):
        return self._robot._hw_send_action(action, target_vel)

    def get_observation(self):
        return self._robot.get_observation()


def _slow_down_rollout_returns(speed: float) -> None:
    """Stretch lerobot-rollout's return_to_initial_position so no arm joint moves faster than `speed`.

    lerobot-rollout calls it with a fixed duration (1 s between episodic episodes, 3 s at shutdown),
    whatever the distance: a 1.5 rad return in 1 s is the fast, dangerous reset the Quest teleop's slow
    return replaced. A no-op outside lerobot-rollout."""
    import sys

    core = sys.modules.get("lerobot.rollout.strategies.core")
    if core is None or getattr(core, "_openarm_slow_return", False):
        return
    original = core.RolloutStrategy.return_to_initial_position

    def slow_return(hw, duration_s: float = 3.0, fps: int = 50) -> bool:
        try:
            obs = hw.robot_wrapper.get_observation()
            dist = max((abs(hw.initial_position[k] - obs[k]) for k in hw.initial_position
                        if k in obs and k in ARM_KEYS), default=0.0)
        except Exception:
            dist = 0.0
        duration = max(duration_s, dist / speed)
        robot = getattr(hw.robot_wrapper, "_robot", hw.robot_wrapper)  # under lerobot-rollout's ThreadSafeRobot
        ran = getattr(robot, "steps_since_reset", 0) > 0  # simulation: an episode ran since the last scene reset
        print(f"[openarm_umeow] returning to the start pose slowly: {dist:.2f} rad over {duration:.1f} s"
              f" (<= {speed:g} rad/s)", flush=True)
        BOARD.note(f"returning to the start pose ({duration:.0f} s)")
        result = original(hw, duration_s=duration, fps=fps)
        # Every reset must END at the start pose, whatever the episode did: check it, do not assume it.
        try:
            time.sleep(1.0)  # settle
            obs = hw.robot_wrapper.get_observation()
            key = max((k for k in hw.initial_position if k in obs and k in ARM_KEYS),
                      key=lambda k: abs(hw.initial_position[k] - obs[k]))
            off = obs[key] - hw.initial_position[key]
            if abs(off) > HOME_TOL_RAD:
                print(f"[openarm_umeow] NOT at the start pose after the reset: {key} is {off:+.2f} rad off"
                      " (motor off, blocked, or twisted against its cable/stop -- see any MOTOR FAULT above).",
                      flush=True)
                BOARD.note(f"NOT at the start pose: {key} {off:+.2f} rad off")
        except Exception:
            logger.debug("start-pose check failed", exc_info=True)
        # In simulation (openarm_isaac) the table is reset here, where a person would reset it.
        if hasattr(robot, "reset_scene"):
            robot.reset_scene(report=ran)
        return result

    core.RolloutStrategy.return_to_initial_position = staticmethod(slow_return)
    core._openarm_slow_return = True


class OpenArmUmeow(OpenArmFollower):
    config_class = OpenArmUmeowConfig
    name = "openarm_umeow"

    def __init__(self, config: OpenArmUmeowConfig):
        follower_cfg = OpenArmFollowerConfig(
            id=config.id,
            calibration_dir=config.calibration_dir,
            right_port=config.right_port,
            left_port=config.left_port,
            enable_fd=True,  # CAN-FD everywhere in this codebase
            model_path=config.model_path,
            cameras=config.cameras,
        )
        super().__init__(follower_cfg)
        self._init_umeow(config)

    def _init_umeow(self, config: OpenArmUmeowConfig) -> None:
        """The wrapper's own state, separate from the follower's hardware setup (openarm_isaac reuses it)."""
        self.umeow_config = config
        self.calib = load_calibration(config.calibration)
        self._last_sent: dict | None = None
        self._last_t: float | None = None
        self._rest_action: dict | None = None
        self._last_desired: dict | None = None
        self._measured: dict | None = None
        self._measured_t = 0.0
        self._fault: str | None = None
        self._motor_fault: str | None = None  # a motor reports a protection fault (see _check_motors)
        self._health_t = 0.0
        self._seen_enabled: set[str] = set()
        self._lag: dict[str, tuple[float, float, bool]] = {}  # key -> (since, command then, told)
        self._stats_reset(time.perf_counter())

    # The hardware layer: the follower's CAN bus and cameras. openarm_isaac (openarm_isaac.py) overrides
    # these five and nothing else, so everything above them -- start pose, step limit, safety guard,
    # squeeze, slow returns -- is the same code on the real arm and in simulation.
    def _hw_connect(self) -> None:
        OpenArmFollower.connect(self, calibrate=False)

    def _hw_get_observation(self):
        return OpenArmFollower.get_observation(self)

    def _hw_send_action(self, action, target_vel):
        return OpenArmFollower.send_action(self, action, target_vel)

    def _hw_disconnect(self) -> None:
        OpenArmFollower.disconnect(self)

    def _hw_feedback_status(self) -> dict:
        return self.get_feedback_status()

    # lerobot asks for these; the follower's own raise NotImplementedError. This robot's
    # calibration is calibration.json plus the motors' stored zero, neither of which is changed here.
    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        BOARD.start()  # rerun status panel; no-op unless --display_data=true
        _slow_down_rollout_returns(self.umeow_config.return_speed)
        ROBOT_STATE.connecting = True
        try:
            self._connect()
        finally:
            ROBOT_STATE.connecting = False

    def _connect(self) -> None:
        self._hw_connect()
        cfg = self.umeow_config
        try:
            check_arms_not_crossed(self, self.calib)
            self._rest_action = get_current_pos_action(self)
            current = dict(self._rest_action)
            if cfg.start_pose == "keyframe":
                start = keyframe_sim_joints(cfg.ik_xml, cfg.start_keyframe)
                # Grippers open, matching the teleop's hold pose (see its ik_driver.py) rather than the
                # keyframe's closed fingers.
                for side in ("left", "right"):
                    start[f"openarm_{side}_finger_joint1"] = GRIPPER_SIM_OPEN
                target = sim_joints_to_motor_action(start, self.calib)
                approached = approach_pose(
                    _RawSender(self),
                    target,
                    label=f"the '{cfg.start_keyframe}' keyframe pose",
                    arm_speed=cfg.approach_speed,
                    gripper_speed=cfg.gripper_max_speed,
                    max_delta=cfg.max_approach_delta,
                    assume_yes=cfg.assume_yes,
                )
                if approached is None:
                    raise RuntimeError("Start-pose approach refused or not confirmed; not starting.")
                current = approached
            elif cfg.start_pose != "none":
                raise ValueError(f"start_pose must be 'keyframe' or 'none', got {cfg.start_pose!r}")
        except BaseException:
            self._hw_disconnect()
            raise
        self._last_sent = {k: current[k] for k in MOTOR_KEYS}
        self._last_desired = dict(self._last_sent)
        self._last_t = time.perf_counter()
        self._fault = self._motor_fault = None
        self._lag = {}
        ROBOT_STATE.publish(self._last_sent, None)
        self._stats_reset(self._last_t)  # the startup ramp is not part of the command-rate stats

    def get_observation(self):
        t0 = time.perf_counter()
        obs = self._hw_get_observation()
        read = time.perf_counter() - t0
        self._stats_read_sum += read
        self._stats_read_n += 1
        self._stats_read_max = max(self._stats_read_max, read)
        # Kept for the tracking guard: lerobot's loops read the observation right before sending.
        self._measured = {k: float(obs[k]) for k in MOTOR_KEYS}
        self._measured_t = time.perf_counter()
        return obs

    def _check_request(self, desired: dict, now: float) -> str | None:
        """Why this request must not be executed, or None if it may be."""
        cfg = self.umeow_config
        if self._last_desired is not None:
            key = max(ARM_KEYS, key=lambda k: abs(desired[k] - self._last_desired[k]))
            jump = abs(desired[key] - self._last_desired[key])
            if jump > cfg.max_command_jump:
                return (f"{key} request jumped {jump:.2f} rad in one tick"
                        f" (> max_command_jump {cfg.max_command_jump:g})")
        if self._measured is not None and now - self._measured_t < 0.25:
            key = max(ARM_KEYS, key=lambda k: abs(desired[k] - self._measured[k]))
            err = abs(desired[key] - self._measured[key])
            if err > cfg.max_tracking_error:
                return (f"{key} request is {err:.2f} rad from the measured joint"
                        f" (> max_tracking_error {cfg.max_tracking_error:g})")
        return None

    def send_action(self, action, target_vel: dict | None = None):
        cfg = self.umeow_config
        desired = {k: float(action[k]) for k in MOTOR_KEYS}
        now = time.perf_counter()
        if self._last_sent is None:
            self._last_sent = get_current_pos_action(self)
            self._last_desired = dict(self._last_sent)
            self._last_t = now

        self._check_motors(now)
        if self._fault is None:
            reason = self._check_request(desired, now)
            if reason is not None:
                self._fault = reason
                print(f"\n[openarm_umeow] SAFETY HOLD: {reason}. The arm holds where it is; it resumes"
                      f" once requests come back within {cfg.resume_tolerance:g} rad of the held pose"
                      " (Quest: X = return home and keep recording, Y = discard).", flush=True)
        elif self._motor_fault is None and max(abs(desired[k] - self._last_sent[k]) for k in ARM_KEYS) < cfg.resume_tolerance:
            print("[openarm_umeow] safety hold released: requests are back at the held pose.", flush=True)
            self._fault = None
        self._last_desired = desired

        if self._fault is not None:
            # Hold: re-send the last executed command with zero velocity. Grippers keep their command
            # and squeeze, so a held object is not dropped.
            hold = dict(self._last_sent)
            self._hw_send_action(hold, {k.replace(".pos", ".vel"): 0.0 for k in MOTOR_KEYS})
            self._check_following(now, hold)
            self._last_t = now
            ROBOT_STATE.publish(hold, self._fault)
            return hold

        # The elapsed time sets this tick's allowance, floored so a burst of calls cannot crawl and
        # capped so a long pause (a blocking reset, a slow first frame) cannot release a jump.
        dt = min(max(now - self._last_t, 1.0 / 200.0), 0.1)
        sent = clamp_step(
            self._last_sent, desired, cfg.max_joint_speed * dt, cfg.gripper_max_speed * dt
        )
        vel = target_vel if target_vel is not None else compute_target_velocity(
            self._last_sent, sent, dt, cfg.max_joint_speed
        )
        self._apply_squeeze(desired)
        self._hw_send_action(sent, vel)
        self._stats_record(now, desired, sent)
        self._check_following(now, sent)
        self._last_sent, self._last_t = sent, now
        ROBOT_STATE.publish(sent, None)
        return sent

    def _check_motors(self, now: float) -> None:
        """Every HEALTH_PERIOD_S: a motor reporting a protection fault (or disabled) -> hold, and say so."""
        if now - self._health_t < HEALTH_PERIOD_S:
            return
        self._health_t = now
        try:
            status = self._hw_feedback_status()
        except Exception:
            return
        self._seen_enabled.update(k for k, c in status.items() if c == 0x1)
        # "disabled" counts only for a motor seen enabled in this session (a switch-off, not a start-up state)
        bad = {k: c for k, c in sorted(status.items()) if is_fault(c) or (c == 0x0 and k in self._seen_enabled)}
        if bad and self._motor_fault is None:
            try:
                health = self.get_motor_health()
            except Exception:
                health = {}

            def temps(k):
                h = health.get(k) or {}
                return f"MOS {h.get('t_mos', '?')} C, rotor {h.get('t_rotor', '?')} C"

            what = ", ".join(f"{k} {status_name(c)}" for k, c in bad.items())
            self._motor_fault = f"motor fault: {what}"
            self._fault = self._motor_fault
            detail = "\n".join(f"           {k}: {status_name(c)} ({temps(k)})" for k, c in bad.items())
            print(f"\n[openarm_umeow] MOTOR FAULT -- the motor switched itself OFF:\n{detail}\n"
                  "           That joint is limp: it stays wherever it is and will NOT return home. The other\n"
                  "           joints hold. OVERCURRENT / OVERLOAD usually means the joint was twisted against\n"
                  "           its stop or its cable (e.g. a wrist rotated too far); OVERTEMP: let it cool.\n"
                  "           Ctrl+C (saved episodes are safe), support the arm, then restart lerobot-record.",
                  flush=True)
            BOARD.note(f"MOTOR FAULT: {what} -- that joint is limp; Ctrl+C and restart")
        elif not bad and self._motor_fault is not None:
            print(f"[openarm_umeow] motors report enabled again (was: {self._motor_fault}).", flush=True)
            self._motor_fault = None

    def _check_following(self, now: float, sent: dict) -> None:
        """Say so when a joint stays LAG_RAD from a steady command for LAG_S: it is not following."""
        if self._measured is None or now - self._measured_t > 0.25:
            return
        worst = max(ARM_KEYS, key=lambda k: abs(self._measured[k] - sent[k]))
        ROBOT_STATE.publish_tracking(worst, sent[worst], self._measured[worst])
        for k in ARM_KEYS:
            err = self._measured[k] - sent[k]
            entry = self._lag.get(k)
            if abs(err) <= LAG_RAD:
                if entry and entry[2]:
                    print(f"[openarm_umeow] {k} is following its command again.", flush=True)
                self._lag.pop(k, None)
            elif entry is None or abs(sent[k] - entry[1]) > 0.05:
                self._lag[k] = (now, sent[k], bool(entry and entry[2]))  # (re)start: the command moved
            elif not entry[2] and now - entry[0] > LAG_S:
                self._lag[k] = (entry[0], entry[1], True)
                print(f"[openarm_umeow] {k} is NOT following: commanded {sent[k]:+.2f} rad, measured"
                      f" {self._measured[k]:+.2f} rad ({err:+.2f}) for {now - entry[0]:.0f} s -- the motor is off,"
                      " blocked, or twisted against its cable/stop.", flush=True)
                BOARD.note(f"{k} not following its command ({err:+.2f} rad off)")

    def _apply_squeeze(self, desired: dict) -> None:
        """Same rule as mirror_bridge.py: extra closing torque while a gripper is commanded closed."""
        tau = self.umeow_config.gripper_squeeze_tau
        for side, prefix in (("left", "L"), ("right", "R")):
            grip = self.calib[side]["gripper"]
            span = grip["open_raw"] - grip["closed_raw"]
            closed_cmd = abs(desired[f"{prefix}J8.pos"] - grip["closed_raw"]) < 0.05 * abs(span)
            closing_sign = 1.0 if span < 0 else -1.0  # toward closed_raw, in motor angle
            self.gripper_squeeze_tau[prefix] = closing_sign * tau if (tau and closed_cmd) else 0.0

    def _stats_reset(self, now: float) -> None:
        self._stats_t0, self._stats_n, self._stats_clamped, self._stats_behind = now, 0, 0, 0.0
        self._stats_read_sum, self._stats_read_n, self._stats_read_max = 0.0, 0, 0.0

    def _stats_record(self, now: float, desired: dict, sent: dict) -> None:
        self._stats_n += 1
        behind = max(abs(desired[k] - sent[k]) for k in ARM_KEYS)
        self._stats_clamped += behind > 1e-6
        self._stats_behind = max(self._stats_behind, behind)
        span = now - self._stats_t0
        if span >= STATS_PERIOD_S:
            pct = 100.0 * self._stats_clamped / max(1, self._stats_n)
            line = (
                f"[openarm_umeow] {self._stats_n / span:.0f} Hz commands | step limit"
                f" {self.umeow_config.max_joint_speed:g} rad/s bound {pct:.0f}% of them,"
                f" worst {self._stats_behind:.3f} rad short of the request"
            )
            if self._stats_read_n:
                line += (f" | observation read {1000 * self._stats_read_sum / self._stats_read_n:.0f} ms avg,"
                         f" {1000 * self._stats_read_max:.0f} max")
            if pct > 5:
                line += "  <-- recorded actions now lead the arm; check the teleop/policy for jumps"
            print(line, flush=True)
            self._stats_reset(now)

    def disconnect(self) -> None:
        BOARD.stop("SHUTDOWN")
        cfg = self.umeow_config
        if self.is_connected and cfg.return_to_rest and self._rest_action is not None:
            try:
                self.gripper_squeeze_tau = {"L": 0.0, "R": 0.0}  # release any grasp first
                print("[openarm_umeow] Returning to the connect-time rest pose before disabling...")
                approach_pose(
                    _RawSender(self),
                    self._rest_action,
                    label="the connect-time rest pose",
                    arm_speed=cfg.approach_speed,
                    gripper_speed=cfg.gripper_max_speed,
                    max_delta=10.0,  # it is where the arm started; never refuse the way home
                    assume_yes=True,
                )
            except Exception:
                logger.exception("Return to rest failed; disabling where the arm is.")
        ROBOT_STATE.clear()
        self._hw_disconnect()
