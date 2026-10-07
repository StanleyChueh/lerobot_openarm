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
    sim_joints_to_motor_action,
)

from .config_openarm_quest import OpenArmQuestConfig
from .ik_driver import QuestIKDriver

STALE_WARN_S = 0.5


class OpenArmQuest(Teleoperator):
    config_class = OpenArmQuestConfig
    name = "openarm_quest"

    def __init__(self, config: OpenArmQuestConfig):
        super().__init__(config)
        self.config = config
        self.calib = load_calibration(config.calibration)
        self.driver: QuestIKDriver | None = None
        self._keyboard = None
        self._stale_warned = False

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
            hold_until_anchor=cfg.hold_until_anchor,
            on_button=self._on_episode_button,
        )
        print(
            "[openarm_quest] connected. X = anchor / reset, Y = reset and hold, triggers = grippers"
            + (", A = save episode (Right arrow), B = re-record (Left arrow)" if self._keyboard else ""),
            flush=True,
        )

    def get_action(self) -> dict[str, float]:
        if self.driver is None:
            raise RuntimeError("openarm_quest is not connected")
        st = self.driver.status()
        age = st["packet_age_s"]
        if st["live"] and (age is None or age > STALE_WARN_S):
            if not self._stale_warned:
                print("[openarm_quest] no fresh Quest packets -- holding the last command.", flush=True)
                self._stale_warned = True
        else:
            self._stale_warned = False
        motor = sim_joints_to_motor_action(driver16_to_sim_joints(self.driver.command()), self.calib)
        return {k: float(motor[k]) for k in MOTOR_KEYS}

    def send_feedback(self, feedback: dict) -> None:
        pass

    def _on_episode_button(self, name: str) -> None:
        if self._keyboard is None:
            return
        from pynput.keyboard import Key

        key = Key.right if name == "a" else Key.left
        self._keyboard.press(key)
        time.sleep(0.02)
        self._keyboard.release(key)
        print(f"[openarm_quest] {name.upper()}: sent {'Right' if name == 'a' else 'Left'} arrow", flush=True)

    def disconnect(self) -> None:
        if self.driver is not None:
            self.driver.close()
            self.driver = None
