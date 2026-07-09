"""Differential IK for OpenArm's left arm -- converts the SmolVLA pickup policy's predicted
end-effector pose delta into a target left-arm joint configuration for the real robot.

Why this exists: the policy was trained on actions recorded as EE-space pose deltas (see
scripts/tools/convert_hdf5_to_lerobot.py: `actions = ep["actions"][:, :7]` = EEF delta pose
(0:6) + gripper (6)), matching Isaac Lab's DifferentialIKControllerCfg(command_type="pose",
use_relative_mode=True, ik_method="dls") used during recording/generation -- NOT joint targets.
OpenArmFollower (robots/umeow_openarm_follower/openarm_follower.py) only accepts joint-space
targets and has no IK of its own, so something has to bridge the two. This module does that
bridge, matching the sim controller's math exactly (see below), using the same
model/openarm_description.urdf OpenArmFollower already loads internally for gravity comp --
confirmed by reading that class's send_action(): it feeds `action['LJ1.pos']` etc. straight into
`pin.computeGeneralizedGravity(self.model, self.data, q)`, which only makes sense if those
values are already in the URDF's native radian convention -- so no separate calibration.json
sign/offset transform is needed here (unlike mirror_bridge.py / sim_bridge_common.py, which talk
to the CAN bus directly, bypassing OpenArmFollower's internal handling).

Math, verified directly against isaaclab.controllers.differential_ik.DifferentialIKController
and isaaclab.utils.math.{apply_delta_pose,compute_pose_error}:

  apply_delta_pose(pos, quat, cmd):
      target_pos  = pos + cmd[0:3]                       # world-aligned axes, no rotation
      target_quat = quat_mul(quat_from_angle_axis(cmd[3:6]), quat)   # LEFT-multiplied

  compute_pose_error(pos, quat, target_pos, target_quat):
      pos_error  = target_pos - pos                        = cmd[0:3]           (exactly)
      quat_error = target_quat * quat^-1
                 = (delta_quat * quat) * quat^-1 = delta_quat                    (exactly)

  i.e. composing the raw command onto the current pose and then computing the pose error back
  is an EXACT identity for delta-mode commands -- the sim controller's two-step "compose then
  decompose" is mathematically equivalent to just running its DLS solve directly on the raw 6D
  command. This module does that directly instead of replicating the redundant round trip:

      delta_q = J^T (J J^T + lambda^2 I)^-1 @ delta_pose_6d
      q_target = q_current + delta_q

  where J is the (6, 7) geometric Jacobian of `openarm_left_hand_tcp` w.r.t. the 7 left-arm
  joints, in pinocchio's LOCAL_WORLD_ALIGNED frame (world-aligned axes, origin at the frame's
  current position) -- matching the world-aligned composition above, and matching the
  world-frame convention PhysX/Isaac Sim's own Articulation Jacobian uses. lambda=0.01 matches
  DifferentialIKControllerCfg's default for ik_method="dls" (see
  source/isaaclab/isaaclab/controllers/differential_ik_cfg.py), which the OpenArm task configs
  never override.

Verified against the real model/openarm_description.urdf (2026-07-09): `openarm_left_joint1..7`
exist with exactly those names (matching Isaac Sim's convention), `openarm_left_hand_tcp` exists
as a frame, nq==nv==18 (no floating base/multi-DOF joints anywhere), and the 7 left-arm joints'
idx_q/idx_v both equal 0..6 -- see the smoke test this module's logic was validated against
(computeJointJacobians + getFrameJacobian at q=neutral() produced a numerically sane, non-NaN
(6,7) Jacobian with plausible magnitudes matching this arm's link lengths).
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin

LEFT_ARM_JOINT_NAMES = [f"openarm_left_joint{i}" for i in range(1, 8)]
LEFT_EE_FRAME_NAME = "openarm_left_hand_tcp"
DLS_LAMBDA = 0.01  # matches DifferentialIKControllerCfg's default for ik_method="dls"


class LeftArmDifferentialIK:
    """Single-step damped-least-squares differential IK for OpenArm's left arm.

    Stateless across calls except for the pinocchio model/data buffers (which are reused for
    performance but fully overwritten each call) -- safe to call repeatedly at control-loop rate.
    """

    def __init__(self, urdf_path: str, lambda_val: float = DLS_LAMBDA):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        if not self.model.existFrame(LEFT_EE_FRAME_NAME):
            raise ValueError(f"URDF '{urdf_path}' has no frame named '{LEFT_EE_FRAME_NAME}'")
        self._frame_id = self.model.getFrameId(LEFT_EE_FRAME_NAME)

        missing = [n for n in LEFT_ARM_JOINT_NAMES if not self.model.existJointName(n)]
        if missing:
            raise ValueError(f"URDF '{urdf_path}' is missing expected left-arm joints: {missing}")
        joint_ids = [self.model.getJointId(n) for n in LEFT_ARM_JOINT_NAMES]
        self._left_q_idx = [self.model.joints[jid].idx_q for jid in joint_ids]
        self._left_v_idx = [self.model.joints[jid].idx_v for jid in joint_ids]
        self._lambda_sq = float(lambda_val) ** 2

        # Full-model q buffer, reused across calls. Only the 7 left-arm entries are ever written
        # before each solve -- the rest (right arm, both grippers) don't affect the left EE
        # frame's Jacobian (disjoint kinematic branch), but forwardKinematics/Jacobian
        # computation still needs *some* value for every DOF in the model.
        self._q_full = pin.neutral(self.model)

    def compute_target_joint_angles(
        self, current_left_arm_q: np.ndarray, delta_pose_6d: np.ndarray
    ) -> np.ndarray:
        """
        Args:
            current_left_arm_q: shape (7,), radians, order = openarm_left_joint1..7 -- read
                directly from OpenArmFollower.get_observation()'s LJ1.pos..LJ7.pos.
            delta_pose_6d: shape (6,), (dx, dy, dz, d_axis_angle_x, d_axis_angle_y,
                d_axis_angle_z) -- the policy's raw action[0:6] output, world-aligned axes (see
                module docstring for why no extra frame conversion is needed here).

        Returns:
            Target left-arm joint angles, shape (7,), radians, same order as the input --
            ready to write directly into LJ1.pos..LJ7.pos for OpenArmFollower.send_action().
        """
        current_left_arm_q = np.asarray(current_left_arm_q, dtype=np.float64).reshape(7)
        delta_pose_6d = np.asarray(delta_pose_6d, dtype=np.float64).reshape(6)

        self._q_full[self._left_q_idx] = current_left_arm_q
        pin.computeJointJacobians(self.model, self.data, self._q_full)
        J_full = pin.getFrameJacobian(
            self.model, self.data, self._frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        J = J_full[:, self._left_v_idx]  # (6, 7)

        # Damped least squares: delta_q = J^T (J J^T + lambda^2 I)^-1 @ delta_pose
        damped = J @ J.T + self._lambda_sq * np.eye(6)
        delta_q = J.T @ np.linalg.solve(damped, delta_pose_6d)

        return current_left_arm_q + delta_q
