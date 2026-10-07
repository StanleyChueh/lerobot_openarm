from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig

from .common import DEFAULT_CALIBRATION, DEFAULT_IK_XML, DEFAULT_URDF


@RobotConfig.register_subclass("openarm_umeow")
@dataclass(kw_only=True)
class OpenArmUmeowConfig(RobotConfig):
    """The lab's dual-arm OpenArm follower, driven by the official lerobot CLIs.

    Actions and observations are raw motor radians, keyed LJ1.pos..LJ8.pos / RJ1.pos..RJ8.pos
    exactly as robots/umeow_openarm_follower reports them -- the same space for both, the way
    lerobot-record expects. calibration.json is used only for the start pose and the gripper
    squeeze; the motors' zero is never touched (unlike the official openarm_follower's calibrate()).
    """

    right_port: str = "can2"
    left_port: str = "can3"
    # URDF for the follower's pinocchio gravity feed-forward.
    model_path: str = DEFAULT_URDF
    calibration: str = DEFAULT_CALIBRATION
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Per-tick limit on every command, from the last command sent: max_joint_speed * dt rad per
    # arm joint. Only a safety net against a jump (a reset snap, a bad IK solve, a policy glitch):
    # at the default the clamp should almost never bind during normal teleop, so the recorded
    # action is what the arm was actually sent. The Isaac pipeline's 0.3 rad/s default bound
    # constantly, which is what left the recorded actions far ahead of the arm.
    max_joint_speed: float = 2.0
    gripper_max_speed: float = 8.0
    # Extra closing torque (N-m) while a gripper is commanded fully closed -- see mirror_bridge.py's
    # --gripper-squeeze-tau. 1.5 is record_demos_openarm.py's --real_arm_gripper_squeeze_tau default.
    gripper_squeeze_tau: float = 1.5

    # Pose ramped to on connect, so every episode starts where the teleop's hold pose (and the
    # dataset's first frames) are: "keyframe" = start_keyframe of the IK model, "none" = stay put.
    start_pose: str = "keyframe"
    ik_xml: str = DEFAULT_IK_XML
    start_keyframe: str = "home"
    approach_speed: float = 0.3
    # Refuse the connect-time approach if any joint would travel further than this (rad): a gap
    # that large means the calibration or zeroing is wrong. Same gate as mirror_bridge.py.
    max_approach_delta: float = 1.8
    # Skip the typed YES before the connect-time approach.
    assume_yes: bool = False
    # On disconnect, ramp back to the pose read at connect (arms hanging) before de-energising,
    # so the arms do not drop.
    return_to_rest: bool = True
