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

    # The IK, configured like the dora ik node in dataflow-vr-mujoco-ros2.yaml, minus its pose shaping:
    # no posture pull (--posture-cost 0), no null-space return toward home (--nullspace-cost 0, default
    # 8.5) and no singularity braking (--singularity-max-approach-rate 0, default 0.25). The arm goes
    # where the hand goes, bounded only by the joint limits (hardware stops) and the speed limits below.
    # Replaying recorded insertion poses (wrist at its limit, with hand tremor) gave the same smoothness
    # either way. dora's own: "--posture-cost 0.01", the other two at their defaults.
    ik_xml: str = DEFAULT_IK_XML
    keyframe: str = "home"
    ik_args: str = (
        "--mode bimanual --max-iters 10 --dt 0.1 --damping 0.1 --posture-cost 0 --lm-damping 0.01"
        " --nullspace-cost 0 --singularity-max-approach-rate 0"
    )
    # Solve rate. The dora graph ticks every 2 ms; see ik_driver.py for why the rate matters.
    ik_hz: float = 500.0
    # 1 Euro filter (min_cutoff, beta, d_cutoff) on each controller pose. quest_receiver.py's beta 0.04 left it
    # a fixed 2 Hz low-pass, 64-82 ms behind the hand; 5.0 opens it up while the hand moves (12-14 ms measured
    # on the real arm, diag_report) and keeps a still hand filtered at 2 Hz.
    smoothing_min_cutoff: float = 2.0
    smoothing_beta: float = 5.0
    smoothing_d_cutoff: float = 1.5

    # Safety (see ik_driver.py's module docstring for why each exists).
    # The shipped joint command moves at most this fast (rad/s); keep it below the robot's max_joint_speed.
    # 2.0: measured with brisk 30 cm reaches -- at 1.0 the elbow / wrist (J4 / J7) fell behind and paused
    # already at 0.5 m/s of hand speed; at 2.0 reaches up to 1.6 m/s followed without a pause.
    max_joint_speed: float = 2.0
    # Speed of the 2nd-X / Y return to home (rad/s, every joint).
    return_speed: float = 0.3
    # A controller moving faster than this (by the headset's clock) is a tracking glitch or a motion too
    # fast to follow safely -> PAUSE. Teleoperation hand motion stays well below; a glitch snaps several cm
    # within one headset frame (~9 m/s and up).
    glitch_speed_mps: float = 4.0
    glitch_rot_speed_dps: float = 900.0
    # How many such jumps within 2 s PAUSE. Fewer are a tracking SNAP (the Quest re-locating a controller
    # it had lost sight of, e.g. held low with the headset at the neck): the jump is ignored -- the arm
    # holds still -- and the hand is re-anchored where the controller now is, so driving continues.
    # 1 = pause on every jump.
    glitch_pause_count: int = 3
    # The raw IK solution moving more than this in one solve -> PAUSE (rad).
    ik_jump_rad: float = 0.2
    # The raw IK solution running more than this ahead of the rate-limited command -> PAUSE (rad):
    # at max_joint_speed 2.0 that is ~0.5 s behind the hand.
    max_lead_rad: float = 1.0
    # X anchors only on a packet younger than this (s); LIVE holds the arms when packets get older.
    # 0.3: the Quest Pro on this lab's Wi-Fi holds its packets back 120-250 ms a few times a second (diag_report);
    # at 0.15 every such hole re-anchored the hand and the motion made during it was lost (50x in 27 s).
    pose_fresh_s: float = 0.3

    # lerobot-record: the Quest's save (2nd X, applied when the arms are home) and discard (Y) end the
    # episode, straight through lerobot-record's own event flags (record_gate.py) -- no key presses. False:
    # only the keyboard (Right / Left / Esc) ends episodes. No effect in lerobot-teleoperate.
    episode_buttons: bool = True

    # Log the Quest packets, every IK tick and every robot read / command to outputs/teleop_diag/<time>/
    # (diag_log.py) -- to find where jerky motion comes from. Report:
    # python -m lerobot_robot_openarm_umeow.diag_report
    diag: bool = False
