"""CPU kinematics on model/openarm_description.urdf (pinocchio).

The sim grasp frame (FrameTransformer on *_ee_tcp + 0.1025 m) coincides with URDF `openarm_*_hand` + 2.4 mm along z:
fitted on 3 recorded GT episodes, position residual median 0.8 mm / max 1.8 mm, rotation identical
(BASELINE_GENERATOR_AUDIT.md §3.4). Everything here is offline geometry: no physics, no contact.
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin

from . import config as C

URDF = C.REPO / "model/openarm_description.urdf"
SIDE = {"L": "left", "R": "right"}
# capsule chain per arm: consecutive frame origins, each segment a capsule
ARM_CHAIN = ["link2", "link3", "link4", "link5", "link6", "link7"]
FINGERS = ["left_finger", "right_finger"]


class Kin:
    def __init__(self):
        self.m = pin.buildModelFromUrdf(str(URDF))
        self.d = self.m.createData()
        self.qidx = {s: [self.m.idx_qs[self.m.getJointId(f"openarm_{SIDE[s]}_joint{i}")] for i in range(1, 8)]
                     for s in "RL"}
        self.vidx = {s: [self.m.idx_vs[self.m.getJointId(f"openarm_{SIDE[s]}_joint{i}")] for i in range(1, 8)]
                     for s in "RL"}
        self.fidx = {s: [self.m.idx_qs[self.m.getJointId(f"openarm_{SIDE[s]}_finger_joint{i}")] for i in (1, 2)]
                     for s in "RL"}
        self.hand = {s: self.m.getFrameId(f"openarm_{SIDE[s]}_hand") for s in "RL"}
        self.chain = {s: [self.m.getFrameId(f"openarm_{SIDE[s]}_{n}") for n in ARM_CHAIN] for s in "RL"}
        self.fing = {s: [self.m.getFrameId(f"openarm_{SIDE[s]}_{n}") for n in FINGERS] for s in "RL"}
        self.lo = {s: self.m.lowerPositionLimit[self.qidx[s]].copy() for s in "RL"}
        self.hi = {s: self.m.upperPositionLimit[self.qidx[s]].copy() for s in "RL"}
        self.grasp_off = pin.SE3(np.eye(3), np.array(C.GRASP_POINT_IN_HAND))

    # ---------------------------------------------------------------- forward
    def _full(self, qR=None, qL=None, fR=C.FINGER_OPEN, fL=C.FINGER_OPEN):
        x = pin.neutral(self.m)
        for s, q, f in (("R", qR, fR), ("L", qL, fL)):
            if q is not None:
                x[self.qidx[s]] = q
            x[self.fidx[s]] = f
        return x

    def grasp_pose(self, side: str, q7: np.ndarray) -> np.ndarray:
        """(4,4) sim grasp-frame pose for one configuration."""
        x = self._full(**{f"q{side}": q7})
        pin.framesForwardKinematics(self.m, self.d, x)
        return (self.d.oMf[self.hand[side]] * self.grasp_off).homogeneous.copy()

    def grasp_traj(self, side: str, Q: np.ndarray) -> np.ndarray:
        return np.stack([self.grasp_pose(side, q) for q in Q])

    def capsules(self, qR, qL, fR=C.FINGER_OPEN, fL=C.FINGER_OPEN):
        """{side: (arm_points link2..link7 (6,3), finger_tips (2,3), hand origin (3,))} for one bimanual configuration."""
        pin.framesForwardKinematics(self.m, self.d, self._full(qR, qL, fR, fL))
        out = {}
        for s in "RL":
            arm = np.stack([self.d.oMf[f].translation.copy() for f in self.chain[s]])   # link2 .. link7 (wrist)
            # finger tips: finger link origin + 5 cm along the hand approach axis
            Rh = self.d.oMf[self.hand[s]].rotation
            tips = np.stack([self.d.oMf[f].translation + Rh[:, 2] * 0.05 for f in self.fing[s]])
            out[s] = (arm, tips, self.d.oMf[self.hand[s]].translation.copy())
        return out

    # ---------------------------------------------------------------- inverse
    def ik(self, side: str, T_target: np.ndarray, q0: np.ndarray, iters: int = 60, damp: float = 1e-3,
           limit_margin: float = C.JOINT_LIMIT_MARGIN, elbow_min: float = C.ELBOW_MIN):
        """Damped least squares to a grasp-frame pose, clamped inside the joint limits (and the elbow floor).

        Returns (q, pos_err_m, rot_err_deg). Mirrors the intent of OpenArmLimitedIKAction (DLS with limits), not its
        exact null-space behaviour.
        """
        lo = self.lo[side] + limit_margin
        hi = self.hi[side] - limit_margin
        lo[3] = max(lo[3], elbow_min)
        q = np.clip(np.array(q0, float), lo, hi)
        target = pin.SE3(T_target[:3, :3], T_target[:3, 3])
        fid = self.hand[side]
        vi = self.vidx[side]
        for _ in range(iters):
            x = self._full(**{f"q{side}": q})
            pin.framesForwardKinematics(self.m, self.d, x)
            cur = self.d.oMf[fid] * self.grasp_off
            err = pin.log6(cur.inverse() * target).vector
            if np.linalg.norm(err[:3]) < 2e-4 and np.linalg.norm(err[3:]) < 2e-3:
                break
            J = pin.computeFrameJacobian(self.m, self.d, x, fid, pin.LOCAL)[:, vi]
            # grasp offset is a pure translation along z of the hand: shift the Jacobian to the grasp point
            Jg = pin.SE3(np.eye(3), np.array(C.GRASP_POINT_IN_HAND)).inverse().action @ J
            dq = Jg.T @ np.linalg.solve(Jg @ Jg.T + damp * np.eye(6), err)
            q = np.clip(q + dq, lo, hi)
        x = self._full(**{f"q{side}": q})
        pin.framesForwardKinematics(self.m, self.d, x)
        cur = self.d.oMf[fid] * self.grasp_off
        e = pin.log6(cur.inverse() * target).vector
        return q, float(np.linalg.norm(cur.translation - target.translation)), float(np.degrees(np.linalg.norm(e[3:])))

    def ik_traj(self, side: str, Ts: np.ndarray, q0: np.ndarray):
        Q, pe, re = [], [], []
        q = q0
        for T in Ts:
            q, p, r = self.ik(side, T, q)
            Q.append(q); pe.append(p); re.append(r)
        return np.array(Q), np.array(pe), np.array(re)


def seg_seg_dist(p1, q1, p2, q2) -> float:
    """Minimum distance between segments [p1,q1] and [p2,q2] (Ericson, Real-Time Collision Detection 5.1.9)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    eps = 1e-12
    if a <= eps and e <= eps:
        return float(np.linalg.norm(r))
    if a <= eps:
        s, t = 0.0, np.clip(f / e, 0, 1)
    else:
        c = d1 @ r
        if e <= eps:
            t, s = 0.0, np.clip(-c / a, 0, 1)
        else:
            b = d1 @ d2
            den = a * e - b * b
            s = np.clip((b * f - c * e) / den, 0, 1) if den > eps else 0.0
            t = (b * s + f) / e
            if t < 0:
                t, s = 0.0, np.clip(-c / a, 0, 1)
            elif t > 1:
                t, s = 1.0, np.clip((b - c) / a, 0, 1)
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def arm_arm_clearance(caps) -> tuple[float, float]:
    """(min surface distance between the two arms' link capsules, min distance between the two hands' finger tips/hand)."""
    aR, tR, hR = caps["R"]
    aL, tL, hL = caps["L"]
    dmin = np.inf
    for i in range(len(aR) - 1):
        for j in range(len(aL) - 1):
            dmin = min(dmin, seg_seg_dist(aR[i], aR[i + 1], aL[j], aL[j + 1]))
    hand_pts_R = np.vstack([tR, hR[None]])
    hand_pts_L = np.vstack([tL, hL[None]])
    dh = np.min(np.linalg.norm(hand_pts_R[:, None] - hand_pts_L[None], axis=2))
    return dmin - 2 * C.LINK_RADIUS, float(dh - 2 * C.FINGER_RADIUS)
