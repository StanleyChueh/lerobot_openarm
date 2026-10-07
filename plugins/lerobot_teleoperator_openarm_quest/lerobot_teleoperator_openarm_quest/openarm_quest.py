"""lerobot Teleoperator: Meta Quest controllers driving the dual-arm OpenArm through mink IK.

get_action() returns the latest IK solution mapped to motor radians through calibration.json, keyed
like the openarm_umeow robot's action features, so lerobot-record stores it as `action` next to the
robot's measured `observation.state` -- the same joint-target-vs-measured-joint pairing a leader arm
gives on Koch / SO-100.
"""

import time

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot_robot_openarm_umeow.common import (
    MOTOR_KEYS,
    driver16_to_sim_joints,
    load_calibration,
    motor_action_to_sim_joints,
    sim_joints_to_driver16,
    sim_joints_to_motor_action,
)
from lerobot_robot_openarm_umeow.shared import ROBOT_STATE

from .config_openarm_quest import OpenArmQuestConfig
from .ik_driver import QuestIKDriver


class OpenArmQuest(Teleoperator):
    config_class = OpenArmQuestConfig
    name = "openarm_quest"

    def __init__(self, config: OpenArmQuestConfig):
        super().__init__(config)
        self.config = config
        self.calib = load_calibration(config.calibration)
        self.driver: QuestIKDriver | None = None
        self._keyboard = None

    @property
    def action_features(self) -> dict[str, type]:
        return {k: float for k in MOTOR_KEYS}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.driver is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        cfg = self.config
        from .record_gate import install

        install(type(self))  # lerobot-record only: episodes start on the first X
        if cfg.episode_buttons:
            try:
                from pynput.keyboard import Controller

                self._keyboard = Controller()
            except Exception as e:
                print(f"[openarm_quest] episode_buttons disabled ({e}); use the keyboard arrows.", flush=True)
        self.driver = QuestIKDriver(
            host=cfg.host,
            port=cfg.port,
            xml=cfg.ik_xml,
            keyframe=cfg.keyframe,
            ik_args=cfg.ik_args,
            ik_hz=cfg.ik_hz,
            smoothing=(cfg.smoothing_min_cutoff, cfg.smoothing_beta, cfg.smoothing_d_cutoff),
            max_joint_speed=cfg.max_joint_speed,
            return_speed=cfg.return_speed,
            glitch_position_m=cfg.glitch_position_m,
            glitch_rotation_deg=cfg.glitch_rotation_deg,
            ik_jump_rad=cfg.ik_jump_rad,
            max_lead_rad=cfg.max_lead_rad,
            pose_fresh_s=cfg.pose_fresh_s,
            held_command=self._robot_held_command,
            robot_fault=ROBOT_STATE.fault,
            on_episode_key=self._press_episode_key,
        )
        print(
            "[openarm_quest] connected. X = start driving; 2nd X = save + return home; Y = discard +"
            " return home; triggers = grippers"
            + ("" if self._keyboard else " (episode keys off: use the keyboard arrows)"),
            flush=True,
        )

    def get_action(self) -> dict[str, float]:
        if self.driver is None:
            raise RuntimeError("openarm_quest is not connected")
        motor = sim_joints_to_motor_action(driver16_to_sim_joints(self.driver.command()), self.calib)
        return {k: float(motor[k]) for k in MOTOR_KEYS}

    def send_feedback(self, feedback: dict) -> None:
        pass

    def _robot_held_command(self):
        """The robot's last executed command as an IK driver vector, or None without a robot."""
        motor = ROBOT_STATE.last_sent()
        if motor is None:
            return None
        return sim_joints_to_driver16(motor_action_to_sim_joints(motor, self.calib))

    def _press_episode_key(self, key: str) -> None:
        if self._keyboard is None:
            return
        from pynput.keyboard import Key

        k = Key.right if key == "right" else Key.left
        self._keyboard.press(k)
        time.sleep(0.02)
        self._keyboard.release(k)

    def disconnect(self) -> None:
        if self.driver is not None:
            self.driver.close()
            self.driver = None
