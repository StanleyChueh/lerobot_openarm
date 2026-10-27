"""--robot.type=openarm_isaac: the openarm_umeow robot with Isaac Sim in place of the CAN bus and cameras.

Subclasses OpenArmUmeow and replaces only its hardware layer (the _hw_* methods), so the start-pose
approach, the step limit, the safety guard, the slow returns and the status output are the very code the
real arm runs. Together with the unchanged openarm_quest teleop and lerobot-record, a sim dataset gets
the real datasets' keys and units (motor radians, grippers in raw motor angle via calibration.json), the
same action meaning (the teleop's command, not the next measured pose), 30 Hz, and the same episodes
(start on X at home, end after the recorded return home).

The simulator is IsaacLab's scripts/tools/lerobot_sim_server.py; the wire format is sim_link.py. It is
LOCKSTEP: every send_action advances the simulation by exactly one control step (1/30 s of sim time),
and get_observation returns the state and camera frames that step produced. A frame is therefore always
one control period of sim time, even when rendering three cameras is slower than real time.

With the server's --mimic_hdf5, every episode lerobot-record saves is also written there as an Isaac Lab
Mimic source demo (begin_episode / end_episode, called by the Quest record gate at X and at lerobot's own
save_episode / clear_episode_buffer), so Mimic can multiply the very demos the lerobot dataset holds.

The scene is re-randomized (the can moved, randomization re-drawn) by reset_scene(), which leaves the
robot where it is -- like a person resetting the table. The Quest record gate calls it each time an
episode starts waiting for X; lerobot-rollout and the async robot client call it after each return to
the start pose, and report whether the episode reached the task's success condition.
"""

import time

from lerobot.robots.robot import Robot

from .common import MOTOR_KEYS, keyframe_sim_joints, motor_action_to_sim_joints, sim_joints_to_motor_action
from .config_openarm_umeow import OpenArmIsaacConfig
from .openarm_umeow import ARM_KEYS, HOME_TOL_RAD, OpenArmUmeow
from .rerun_status import BOARD
from .sim_link import SimClient

from sim_bridge_common import GRIPPER_SIM_OPEN  # noqa: E402  (on sys.path via common)


class OpenArmIsaac(OpenArmUmeow):
    config_class = OpenArmIsaacConfig
    # lerobot-record writes robot.name as the dataset's robot_type, and lerobot's dataset merge refuses
    # datasets whose robot_type differs: a sim dataset must say openarm_umeow to be merged with the real
    # ones it is meant to be trained with. It is the same arm, keys and units; name the repo "..._sim_...".
    name = "openarm_umeow"

    def __init__(self, config: OpenArmIsaacConfig):
        Robot.__init__(self, config)  # not OpenArmFollower's: no CAN bus, no pinocchio model
        self.config = config
        self.cameras = {}  # the simulator's cameras arrive with each observation
        self.gripper_squeeze_tau = {"L": 0.0, "R": 0.0}  # set by the shared code; the sim grips by its own PD
        self._is_connected = False
        self._sim = SimClient(config.host, config.port)
        self._latest: dict | None = None
        self.episode_success = False  # the task's success condition was met since the last scene reset
        self._results: list[bool] = []
        self.steps_since_reset = 0
        self.sim_steps = 0  # every simulation step this robot has asked for
        self.scene_resets = 0
        self.last_reset: dict | None = None  # matched_eval.check_reset() of the last reset that placed the can
        self.step_log: list[dict] | None = None  # set to a list to keep each step's ground truth (matched_eval)
        # lerobot-rollout's and robot_client's returns to the start pose reset the scene (see openarm_umeow's
        # slow_return); a matched evaluation places the can itself and turns this off.
        self.auto_scene_reset = True
        self._start_sim_joints: dict | None = None  # the start pose in sim joints (set on connect)
        self._init_umeow(config)

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {cam: (self.config.camera_height, self.config.camera_width, 3) for cam in self.config.sim_cameras}

    # -- the hardware layer -----------------------------------------------------------------------------
    def _hw_connect(self) -> None:
        cfg = self.config
        hello = self._sim.connect()
        served = hello["cameras"]
        wrong = {c: served.get(c) for c in cfg.sim_cameras if served.get(c) != [cfg.camera_height, cfg.camera_width]}
        if wrong:
            self._sim.close()
            raise ValueError(
                f"Camera mismatch: this robot expects {cfg.sim_cameras} at {cfg.camera_height}x{cfg.camera_width},"
                f" the simulator serves {served} (got {wrong}). Fix --robot.sim_cameras / camera_width / camera_height."
            )
        if hello["control_hz"] != 30:
            print(f"[openarm_isaac] WARNING: the simulator steps at {hello['control_hz']} Hz; the real pipeline"
                  " records at 30 (unset OPENARM_CONTROL_HZ on the server).", flush=True)
        # Start where the real arm's connect-time approach ends: the IK model's keyframe, grippers open.
        start = None
        if cfg.start_pose == "keyframe":
            start = keyframe_sim_joints(cfg.ik_xml, cfg.start_keyframe)
            for side in ("left", "right"):
                start[f"openarm_{side}_finger_joint1"] = GRIPPER_SIM_OPEN
            start = {k: v for k, v in start.items() if k in hello["joint_names"]}
        self._start_sim_joints = start
        self._latest = self._sim.request("reset", joints=start)
        self.episode_success = False
        self._is_connected = True
        print(f"[openarm_isaac] connected to Isaac Sim at {cfg.host}:{cfg.port}: {hello['task']}"
              f" ({hello['task_mode']}), randomization {hello['domain_randomization']}, cameras {list(served)}.",
              flush=True)

    def _hw_get_observation(self):
        if self._latest is None:
            self._latest = self._sim.request("observe")
        obs = sim_joints_to_motor_action(self._latest["joints"], self.calib)
        for cam in self.config.sim_cameras:
            obs[cam] = self._latest["images"][cam]
        return obs

    def _hw_send_action(self, action, target_vel):
        joints = motor_action_to_sim_joints(action, self.calib)
        self._latest = self._sim.request("step", joints=joints)
        self.episode_success = self.episode_success or bool(self._latest["success"])
        self.steps_since_reset += 1
        self.sim_steps += 1
        if self.step_log is not None:
            from .matched_eval import step_record

            state = sim_joints_to_motor_action(self._latest["joints"], self.calib)
            self.step_log.append(step_record(self._latest, [float(action[k]) for k in MOTOR_KEYS],
                                             [float(state[k]) for k in MOTOR_KEYS], time.perf_counter()))
        return action

    def _hw_disconnect(self) -> None:
        self._sim.close()
        self._is_connected = False

    def _hw_feedback_status(self) -> dict:
        return {}  # simulated motors never trip

    # -- Mimic source demos (the server's --mimic_hdf5) ------------------------------------------------------
    def begin_episode(self) -> None:
        """A lerobot episode starts recording: the server starts the matching Mimic source demo."""
        self._sim.request("episode_begin")

    def close_episode(self) -> None:
        """The episode's last frame is recorded (lerobot's reset phase follows before it saves or discards)."""
        self._sim.request("episode_close")

    def end_episode(self, saved: bool) -> None:
        """lerobot saved (or discarded) the episode: the server writes (or drops) the Mimic source demo."""
        reply = self._sim.request("episode_end", save=saved)
        if reply.get("saved"):
            print(f"[openarm_isaac] also saved as Mimic source demo {reply['saved']} ({reply['steps']} steps).", flush=True)

    def _home_error(self) -> float | None:
        """Largest arm-joint distance (rad) from the start pose, measured on the latest simulator reply."""
        if not self.start_action or not self._latest:
            return None
        now = sim_joints_to_motor_action(self._latest["joints"], self.calib)
        return max(abs(now[k] - self.start_action[k]) for k in ARM_KEYS if k in now and k in self.start_action)

    # -- the table --------------------------------------------------------------------------------------
    def reset_scene(self, report: bool = False, object_xy: tuple[float, float] | None = None) -> None:
        """Re-randomize the scene, leaving the robot where it is. With `report`, first print whether the episode
        that just ended met the task's success condition, and the running tally. With `object_xy`, the can is
        placed there (env-local m, upright, at the task's resting height) instead of at a random spot, and
        checked: matched_eval.PoseNotApplied if the simulator did not put it there."""
        if report:
            self._results.append(self.episode_success)
            n, ok = len(self._results), sum(self._results)
            verdict = "SUCCESS" if self.episode_success else "no success"
            print(f"[openarm_isaac] episode {n}: {verdict} (task success condition) -- {ok}/{n} so far.", flush=True)
            BOARD.note(f"sim episode {n}: {verdict} ({ok}/{n})")
        t0 = time.perf_counter()
        rehome, home_error = False, None
        if object_xy is None:
            self._latest = self._sim.request("reset", joints=None)
        else:
            # A trial starts from the start pose. The return before it normally gets there (within HOME_TOL_RAD);
            # when it did not (an arm blocked by a dropped can), the reset also puts the arm back at the start
            # joints -- otherwise it would sit where the can is placed. Recorded in last_reset.
            home_error = self._home_error()
            rehome = home_error is not None and home_error > HOME_TOL_RAD and self._start_sim_joints is not None
            self._latest = self._sim.request("reset", joints=self._start_sim_joints if rehome else None,
                                             object_xy=[float(v) for v in object_xy], object_at_rest=True)
            if rehome:
                self.prepare_return(self.get_observation())  # the guard's last command = the new measured pose
                print(f"[openarm_isaac] the arm was {home_error:.2f} rad from the start pose: put back there with"
                      " the scene reset.", flush=True)
        self.episode_success = False
        self.steps_since_reset = 0
        self.scene_resets += 1
        self.last_reset = None
        if object_xy is not None:
            from .matched_eval import check_reset

            self.last_reset = {**check_reset(object_xy, self._latest.get("handover")), "reset": self.scene_resets,
                               "home_error_rad_before": home_error, "rehomed": rehome}
        where = "" if object_xy is None else f", can at ({object_xy[0]:.4f}, {object_xy[1]:.4f})"
        print(f"[openarm_isaac] scene reset ({time.perf_counter() - t0:.1f} s{where}).", flush=True)
