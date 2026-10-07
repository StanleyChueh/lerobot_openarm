"""Pieces shared by the openarm_umeow robot and the openarm_quest teleoperator.

Both plugins live inside the lerobot_openarm checkout and reuse its top-level modules
(robots/umeow_openarm_follower, sim_bridge_common.py), which are not an installed package. The
editable install leaves this file in the checkout, so the checkout root is found from here and
appended to sys.path -- appended, not prepended, so nothing in it can shadow an installed module.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from sim_bridge_common import (  # noqa: E402
    load_calibration,
    motor_action_to_sim_joints,
    sim_joints_to_motor_action,
)

DEFAULT_CALIBRATION = str(REPO_ROOT / "calibration.json")
DEFAULT_URDF = str(REPO_ROOT / "model" / "openarm_description.urdf")
# The MuJoCo model the dora ik node solves against (dataflow-vr-mujoco-ros2.yaml's --xml). Its
# joint convention is the one calibration.json maps to the motors. Referenced in place, not
# copied: the scene and its meshes are ~54 MB, and two copies could silently drift apart.
DEFAULT_IK_XML = str(
    REPO_ROOT.parent
    / "dora-openarm-data-collection/nodes/dora-openarm-mujoco/src/dora_openarm_mujoco/scenes/v1_camera/scene.xml"
)

# The follower's own feature order (robots/umeow_openarm_follower _motors_ft): R and L interleaved.
MOTOR_KEYS = [f"{p}J{i}.pos" for i in range(1, 9) for p in ("R", "L")]


def driver16_to_sim_joints(command: np.ndarray) -> dict:
    """openarm_control's float[16] driver vector (right[7 joints + finger], left[...]) -> sim joint dict."""
    sim = {}
    for side, base in (("right", 0), ("left", 8)):
        for n in range(1, 8):
            sim[f"openarm_{side}_joint{n}"] = float(command[base + n - 1])
        sim[f"openarm_{side}_finger_joint1"] = float(command[base + 7])
    return sim


def keyframe_sim_joints(xml_path: str, keyframe: str = "home") -> dict:
    """The arm and finger joint values of one MJCF keyframe, by sim joint name."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(xml_path)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    if key_id < 0:
        raise ValueError(f"keyframe '{keyframe}' not found in {xml_path}")
    qpos = model.key_qpos[key_id]
    names = [f"openarm_{s}_joint{n}" for s in ("left", "right") for n in range(1, 8)]
    names += ["openarm_left_finger_joint1", "openarm_right_finger_joint1"]
    sim = {}
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"joint '{name}' not found in {xml_path}")
        sim[name] = float(qpos[model.jnt_qposadr[jid]])
    return sim


__all__ = [
    "DEFAULT_CALIBRATION",
    "DEFAULT_IK_XML",
    "DEFAULT_URDF",
    "MOTOR_KEYS",
    "REPO_ROOT",
    "driver16_to_sim_joints",
    "keyframe_sim_joints",
    "load_calibration",
    "motor_action_to_sim_joints",
    "sim_joints_to_motor_action",
]
