from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot_robot_openarm_umeow.common import DEFAULT_CALIBRATION, DEFAULT_IK_XML


@TeleoperatorConfig.register_subclass("openarm_quest")
@dataclass(kw_only=True)
class OpenArmQuestConfig(TeleoperatorConfig):
    """Meta Quest controllers -> mink IK -> joint targets for the openarm_umeow robot.

    The headset app is the one the dora pipeline uses (UDP JSON to port 5006); only one listener can
    own that port, so stop the dora dataflow first.
    """

    host: str = "0.0.0.0"
    port: int = 5006
    calibration: str = DEFAULT_CALIBRATION

    # The IK, configured exactly like the dora ik node in dataflow-vr-mujoco-ros2.yaml.
    ik_xml: str = DEFAULT_IK_XML
    keyframe: str = "home"
    ik_args: str = "--mode bimanual --max-iters 10 --dt 0.1 --damping 0.1 --posture-cost 0.01 --lm-damping 0.01"
    # Solve rate. The dora graph ticks every 2 ms; see ik_driver.py for why the rate matters.
    ik_hz: float = 500.0
    # 1 Euro filter (min_cutoff, beta, d_cutoff) on each controller pose -- quest_receiver.py's values.
    smoothing_min_cutoff: float = 2.0
    smoothing_beta: float = 0.04
    smoothing_d_cutoff: float = 1.5
    # Start holding the keyframe pose until X anchors the operator (ik.py --hold-until-anchor).
    hold_until_anchor: bool = True

    # Quest A / B press the Right / Left arrow keys, which lerobot-record's keyboard listener reads
    # as "end this episode (save)" / "re-record this episode". Needs an X display; Esc still stops.
    episode_buttons: bool = True
