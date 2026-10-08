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


class _RawSender:
    """Hands sim_bridge_common's ramp helpers the follower's unclamped two-argument send_action."""

    def __init__(self, robot: "OpenArmUmeow"):
        self._robot = robot

    def send_action(self, action, target_vel):
        return OpenArmFollower.send_action(self._robot, action, target_vel)

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
        print(f"[openarm_umeow] returning to the start pose slowly: {dist:.2f} rad over {duration:.1f} s"
              f" (<= {speed:g} rad/s)", flush=True)
        BOARD.note(f"returning to the start pose ({duration:.0f} s)")
        return original(hw, duration_s=duration, fps=fps)

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
        self.umeow_config = config
        self.calib = load_calibration(config.calibration)
        self._last_sent: dict | None = None
        self._last_t: float | None = None
        self._rest_action: dict | None = None
        self._last_desired: dict | None = None
        self._measured: dict | None = None
        self._measured_t = 0.0
        self._fault: str | None = None
        self._stats_reset(time.perf_counter())

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
        super().connect(calibrate=False)
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
            super().disconnect()
            raise
        self._last_sent = {k: current[k] for k in MOTOR_KEYS}
        self._last_desired = dict(self._last_sent)
        self._last_t = time.perf_counter()
        self._fault = None
        ROBOT_STATE.publish(self._last_sent, None)
        self._stats_reset(self._last_t)  # the startup ramp is not part of the command-rate stats

    def get_observation(self):
        obs = super().get_observation()
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

        if self._fault is None:
            reason = self._check_request(desired, now)
            if reason is not None:
                self._fault = reason
                print(f"\n[openarm_umeow] SAFETY HOLD: {reason}. The arm holds where it is; it resumes"
                      f" once requests come back within {cfg.resume_tolerance:g} rad of the held pose"
                      " (Quest: X = return home and keep recording, Y = discard).", flush=True)
        elif max(abs(desired[k] - self._last_sent[k]) for k in ARM_KEYS) < cfg.resume_tolerance:
            print("[openarm_umeow] safety hold released: requests are back at the held pose.", flush=True)
            self._fault = None
        self._last_desired = desired

        if self._fault is not None:
            # Hold: re-send the last executed command with zero velocity. Grippers keep their command
            # and squeeze, so a held object is not dropped.
            hold = dict(self._last_sent)
            super().send_action(hold, {k.replace(".pos", ".vel"): 0.0 for k in MOTOR_KEYS})
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
        super().send_action(sent, vel)
        self._stats_record(now, desired, sent)
        self._last_sent, self._last_t = sent, now
        ROBOT_STATE.publish(sent, None)
        return sent

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
        super().disconnect()
