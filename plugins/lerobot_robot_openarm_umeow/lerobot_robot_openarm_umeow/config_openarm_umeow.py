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

    right_port: str = "can0"
    left_port: str = "can1"
    # URDF for the follower's pinocchio gravity feed-forward.
    model_path: str = DEFAULT_URDF
    calibration: str = DEFAULT_CALIBRATION
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Per-tick limit on every command, from the last command sent: max_joint_speed * dt rad per arm
    # joint -- a backstop. Keep it ABOVE the openarm_quest teleop's own max_joint_speed (2.0): with the
    # same value, the jitter between the teleop's and this loop's clocks made it clip commands that were
    # within the teleop's limit (real session: "bound 8-20%", recorded actions leading the arm).
    max_joint_speed: float = 2.5
    gripper_max_speed: float = 8.0

    # Safety guard -- refuse a command instead of executing it. If any ARM joint's request jumps by
    # more than max_command_jump (rad) from the previous request in one tick, or sits more than
    # max_tracking_error (rad) from the measured joint, the arm HOLDS where it was last commanded and
    # the reason is printed. It stays held until a request comes back within resume_tolerance (rad)
    # of the held command on every arm joint (the openarm_quest teleop does that itself: it pauses
    # on the hold and resumes from the held pose on X, or returns home on Y).
    max_command_jump: float = 0.25
    max_tracking_error: float = 0.5
    resume_tolerance: float = 0.1
    # Extra closing torque (N-m) while a gripper is commanded fully closed -- see mirror_bridge.py's
    # --gripper-squeeze-tau. 1.5 is record_demos_openarm.py's --real_arm_gripper_squeeze_tau default.
    gripper_squeeze_tau: float = 1.5

    # Pose ramped to on connect, so every episode starts where the teleop's hold pose (and the
    # dataset's first frames) are: "keyframe" = start_keyframe of the IK model, "none" = stay put.
    start_pose: str = "keyframe"
    ik_xml: str = DEFAULT_IK_XML
    start_keyframe: str = "home"
    approach_speed: float = 0.3
    # lerobot-rollout returns the arm to its start pose between episodes in a FIXED 1 s (episodic) or
    # 3 s (shutdown), however far it is; those returns are stretched so no joint exceeds this (rad/s).
    return_speed: float = 0.3
    # Refuse the connect-time approach if any joint would travel further than this (rad): a gap
    # that large means the calibration or zeroing is wrong. Same gate as mirror_bridge.py.
    max_approach_delta: float = 1.8
    # lerobot-rollout only: precision the policy is loaded in. "auto" = bf16 for GR00T (its fp32 weights
    # do not fit a 16 GB GPU), unchanged otherwise; "bf16" / "fp32" force it. See policy_loading.py.
    policy_dtype: str = "auto"
    # Skip the typed YES before the connect-time approach.
    assume_yes: bool = False
    # On disconnect, ramp back to the pose read at connect (arms hanging) before de-energising,
    # so the arms do not drop.
    return_to_rest: bool = True
