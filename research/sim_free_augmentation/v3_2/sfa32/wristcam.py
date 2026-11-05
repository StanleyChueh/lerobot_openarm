"""Wrist-camera models for the two OpenArm wrist cameras (no simulator).

* Intrinsics: MEASURED with `rs-enumerate-devices -c` (research/task_aware_real2sim/m2a/WRIST_CALIBRATION_M2A.md):
    wrist_cam (LEFT arm)        fx 605.44 fy 605.13 cx 322.85 cy 246.44, distortion 0
    right_wrist_cam (RIGHT arm) fx 607.79 fy 607.61 cx 322.58 cy 248.52, distortion 0
* Nominal mount (HAND-tuned in the Isaac GUI, never validated): camera in openarm_{side}_ee_tcp == link7,
  pos (0.05342, 0.001, 0.12199) m, quaternion (w,x,y,z) = (0.68057, -0.19188, -0.19188, 0.68057), ROS optical convention
  (z forward, x right, y down) - the same convention as sfa.robot.Camera.
* v3.2 replaces the mount ROTATION by a hand-eye fit (scripts/calib_wrist.py) and keeps the nominal translation unless
  the fit is well conditioned; both are recorded in outputs/calibration/wrist_calibration.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sfa3.recon3 import rb  # noqa: F401  (also puts v1/v2 on sys.path)

HERE = Path(__file__).resolve().parents[1]
CAM_OF = {"L": "observation.images.wrist_cam", "R": "observation.images.right_wrist_cam"}
SIDE_OF = {v: k for k, v in CAM_OF.items()}
INTR = {"L": {"fx": 605.44, "fy": 605.13, "cx": 322.85, "cy": 246.44, "W": 640, "H": 480},
        "R": {"fx": 607.79, "fy": 607.61, "cx": 322.58, "cy": 248.52, "W": 640, "H": 480}}
NOMINAL = {"pos": [0.05342, 0.001, 0.12199], "quat_wxyz": [0.68057, -0.19188, -0.19188, 0.68057],
           "parent": "openarm_{side}_link7", "source": "IsaacLab stack_ik_abs_visuomotor_env_cfg.py (HAND, GUI-tuned)"}
CALIB = HERE / "outputs/calibration/wrist_calibration.json"


def mount(side: str, which="calibrated") -> np.ndarray:
    """4x4 camera pose in the link7 frame."""
    X = np.eye(4)
    if which == "calibrated" and CALIB.exists():
        c = json.loads(CALIB.read_text())[side]["mount"]
        X[:3, :3] = np.array(c["R"])
        X[:3, 3] = c["p"]
        return X
    X[:3, :3] = rb.quat_R(NOMINAL["quat_wxyz"])
    X[:3, 3] = NOMINAL["pos"]
    return X


def lag(side: str) -> float:
    if CALIB.exists():
        return float(json.loads(CALIB.read_text())[side]["lag_frames"])
    return 0.0


def K(side):
    c = INTR[side]
    return np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1.0]])


def link7(robot, side):
    return robot.m.getFrameId(f"openarm_{'right' if side == 'R' else 'left'}_link7")


def cam_pose(robot, qR, qL, side, X) -> np.ndarray:
    robot.fk(qR, qL)
    return robot.frame(link7(robot, side)) @ X


def camera(side, T_world_cam) -> "rb.Camera":
    c = dict(INTR[side])
    c.update(pos=T_world_cam[:3, 3].tolist(), quat_wxyz=[1, 0, 0, 0], source=f"wrist {side}")
    return rb.Camera(c, R=T_world_cam[:3, :3], t=T_world_cam[:3, 3])
