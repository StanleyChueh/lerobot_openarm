"""Object-centric segment transform for ONE arm, any side (generalises battery/bsfa/bplan.py, which is right-arm only).

An object that rests on the table and is picked by arm `side` is translated by d (table plane). The arm's TCP
trajectory is translated by w(t) d:
  w: smoothstep 0 -> 1 from the arm's motion onset to 15 frames before its grasp (the final approach and the grasp are
     the recorded motion RELATIVE TO THE OBJECT), w = 1 through grasp + 10 frames, then 1 -> 0 (smoothstep) until the
     rejoin frame, after which the trajectory is the recording.
Orientation is unchanged, so the hand (and the object it holds) only translates relative to the recording: any held
object keeps its recorded orientation and its recorded pose in the hand (MimicGen, Mandlekar et al. CoRL 2023;
DemoGen, Xue et al. RSS 2025). The other arm and both grippers are untouched.
Joints: damped-least-squares IK bounded by [URDF limits U recorded pose], warm-started along the source motion, joint
offset Gaussian-smoothed (sigma = 3 frames) and re-verified by FK. Actions keep the recorded command-to-state lead k:
a'[t] = a[t] + (q' - q)[t + k]."""
from __future__ import annotations
import numpy as np
import pinocchio as pin
from scipy.ndimage import gaussian_filter1d
from sfa.robot import Robot, NAMES
from sfa.data import action_lead
from sfa.plan import smoothstep

SIGMA = 3.0
J = {s: [NAMES.index(f"{s}J{i}.pos") for i in range(1, 8)] for s in "RL"}


def ik(robot, side, T_target, q0, q_other, lo, hi, iters=100, damp=1e-3):
    q = np.clip(np.asarray(q0, float), lo, hi); fid = robot.tcp[side]; target = pin.SE3(T_target[:3, :3], T_target[:3, 3])
    full = (lambda q: robot.q_full(q, q_other)) if side == "R" else (lambda q: robot.q_full(q_other, q))
    for _ in range(iters):
        x = full(q); pin.framesForwardKinematics(robot.m, robot.d, x)
        e = pin.log6(robot.d.oMf[fid].inverse() * target).vector
        if np.linalg.norm(e[:3]) < 1e-4 and np.linalg.norm(e[3:]) < 1e-3:
            break
        Jm = pin.computeFrameJacobian(robot.m, robot.d, x, fid, pin.LOCAL)[:, robot.vidx[side]]
        q = np.clip(q + Jm.T @ np.linalg.solve(Jm @ Jm.T + damp * np.eye(6), e), lo, hi)
    return q


def tcp_err(robot, side, q, q_other, T):
    robot.fk(*((q, q_other) if side == "R" else (q_other, q))); cur = robot.frame(robot.tcp[side])
    return (float(np.linalg.norm(cur[:3, 3] - T[:3, 3])),
            float(np.degrees(np.arccos(np.clip((np.trace(cur[:3, :3].T @ T[:3, :3]) - 1) / 2, -1, 1)))))


def weights(T, t_motion, t_grasp, t_rejoin, pre=15, post=10):
    t_in, t_out = t_grasp - pre, t_grasp + post
    assert t_motion < t_in and t_out < t_rejoin, (t_motion, t_in, t_out, t_rejoin)
    t = np.arange(T, dtype=float)
    w = np.where(t < t_in, smoothstep((t - t_motion) / (t_in - t_motion)), 1.0)
    return np.where(t > t_out, 1 - smoothstep((t - t_out) / (t_rejoin - t_out)), w)


def plan(robot, state, action, side, d, t_motion, t_grasp, t_rejoin):
    d = np.asarray(d, float); T = len(state)
    w = weights(T, t_motion, t_grasp, t_rejoin)
    other = "L" if side == "R" else "R"
    sp = Robot.split(state); q = sp[side].copy(); qo = sp[other]
    TT = robot.tcp_traj({"R": sp["R"], "L": sp["L"]}, side)
    lo_t = np.minimum(robot.lo[side][None], q); hi_t = np.maximum(robot.hi[side][None], q)
    qn = q.copy(); act = w > 1e-9
    for t in np.flatnonzero(act):
        Tt = TT[t].copy(); Tt[:3, 3] += w[t] * d
        q0 = qn[t - 1] + (q[t] - q[t - 1]) if t > 0 and act[t - 1] else q[t]
        qn[t] = ik(robot, side, Tt, q0, qo[t], lo_t[t], hi_t[t])
    dq = gaussian_filter1d(qn - q, SIGMA, axis=0, mode="nearest") * act[:, None]
    qn = np.clip(q + dq, lo_t, hi_t)
    err = np.zeros((T, 2))
    for t in np.flatnonzero(act):
        Tt = TT[t].copy(); Tt[:3, 3] += w[t] * d
        err[t] = tcp_err(robot, side, qn[t], qo[t], Tt)
    S = state.copy(); S[:, J[side]] = qn
    k = action_lead(state, action)["k_best"]
    idx = np.minimum(np.arange(T) + k, T - 1)
    A = action.copy(); A[:, J[side]] = action[:, J[side]] + (qn - q)[idx]
    alo = np.minimum(robot.lo[side], action[:, J[side]].min(0)); ahi = np.maximum(robot.hi[side], action[:, J[side]].max(0))
    A[:, J[side]] = np.clip(A[:, J[side]], alo, ahi)
    return {"state": S, "action": A, "q": {side: qn, other: qo.copy()}, "w": w, "ik_err": err, "lead": int(k),
            "tcp_new": robot.tcp_traj({"R": qn if side == "R" else qo, "L": qn if side == "L" else qo}, side)}
