"""Battery-position variants (object-centric trajectory adaptation, cf. MimicGen / DemoGen):
the battery's resting pose is translated by d on the table; the right-arm TCP trajectory is translated by w(t) d with
  w = smoothstep ramp 0 -> 1 from the right arm's motion onset to 15 frames before the grasp (approach + grasp are the
      recorded motion relative to the battery), w = 1 through the grasp and lift,
  w = 1 -> 0 smoothstep during the lift, finishing at the lift apex (first height maximum within 4 s of the grasp,
      and >= 40 frames before the release), after which the hand moves to the station and inserts,
so the insertion (bimanual contact) is the recorded, physically executed motion. Orientation is unchanged. Joint
trajectory by damped-least-squares IK (URDF limits - 0.02 rad), warm-started along the source motion; the left arm and
both grippers are untouched. Actions keep the recorded command-to-state lead k:  a'[t] = a[t] + (q' - q)[t + k]."""
from __future__ import annotations
import numpy as np
from sfa.robot import Robot, NAMES
from sfa.data import action_lead
from sfa.plan import smoothstep

D_CONTACT = 0.12
SIGMA = 3.0
RJ = [NAMES.index(f"RJ{i}.pos") for i in range(1, 8)]


def schedule(rec):
    sp = Robot.split(rec.state)
    TR = rec.robot.tcp_traj({"R": sp["R"], "L": sp["L"]}, "R"); TL = rec.robot.tcp_traj({"R": sp["R"], "L": sp["L"]}, "L")
    v = np.r_[0, np.linalg.norm(np.diff(TR[:, :3, 3], axis=0), axis=1)] * 30
    moving = np.flatnonzero(v[: rec.tg] > 0.02)
    t0 = int(moving[0]) if len(moving) else 0
    # insertion onset = the lift apex: first maximum of right-hand height within 120 frames after the grasp
    win = np.arange(rec.tg, min(rec.tg + 120, rec.tr))
    t_apex = int(win[np.argmax(TR[win, 2, 3])])
    t_contact = t_apex
    t_rejoin = min(t_apex, rec.tr - 40)
    t_in_end = rec.tg - 15; t_out_start = rec.tg + 10
    assert t0 < t_in_end and t_out_start < t_rejoin, (t0, t_in_end, t_out_start, t_rejoin)
    t = np.arange(rec.T, dtype=float)
    w = np.where(t < t_in_end, smoothstep((t - t0) / (t_in_end - t0)), 1.0)
    w = np.where(t > t_out_start, 1 - smoothstep((t - t_out_start) / (t_rejoin - t_out_start)), w)
    return w, {"t_motion": t0, "t_grasp": rec.tg, "t_ramp_in_end": t_in_end, "t_ramp_out_start": t_out_start,
               "t_contact": t_contact, "t_rejoin": t_rejoin, "t_release": rec.tr}, TR


def ik(robot, T_target, q0, lo, hi, iters=100, damp=1e-3):
    """DLS IK of the right TCP, bounded by [lo, hi] (no extra margin, so a recorded pose is always feasible)."""
    import pinocchio as pin
    q = np.clip(np.asarray(q0, float), lo, hi); fid = robot.tcp["R"]; target = pin.SE3(T_target[:3, :3], T_target[:3, 3])
    for _ in range(iters):
        x = robot.q_full(q, np.zeros(7)); pin.framesForwardKinematics(robot.m, robot.d, x)
        e = pin.log6(robot.d.oMf[fid].inverse() * target).vector
        if np.linalg.norm(e[:3]) < 1e-4 and np.linalg.norm(e[3:]) < 1e-3:
            break
        J = pin.computeFrameJacobian(robot.m, robot.d, x, fid, pin.LOCAL)[:, robot.vidx["R"]]
        q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), e), lo, hi)
    x = robot.q_full(q, np.zeros(7)); pin.framesForwardKinematics(robot.m, robot.d, x); cur = robot.d.oMf[fid]
    e = pin.log6(cur.inverse() * target).vector
    return q, float(np.linalg.norm(cur.translation - target.translation)), float(np.degrees(np.linalg.norm(e[3:])))


def plan(rec, dxy):
    d = np.array([dxy[0], dxy[1], 0.0])
    w, marks, TR = schedule(rec)
    sp = Robot.split(rec.state); qR = sp["R"].copy(); qn = qR.copy()
    err = np.zeros((rec.T, 2))
    for t in range(rec.T):
        if w[t] < 1e-9:
            qn[t] = qR[t]; continue
        T = TR[t].copy(); T[:3, 3] += w[t] * d
        q0 = qn[t - 1] + (qR[t] - qR[t - 1]) if t > 0 else qR[t]
        lo = np.minimum(rec.robot.lo["R"], qR[t]); hi = np.maximum(rec.robot.hi["R"], qR[t])   # URDF limits U recorded pose
        qn[t], pe, re = ik(rec.robot, T, q0, lo, hi)
        err[t] = pe, re
    # temporally smooth the joint offset (removes frame-to-frame IK jitter), then re-check the TCP error by FK
    from scipy.ndimage import gaussian_filter1d
    dq0 = qn - qR
    act = w > 1e-9
    dqs = gaussian_filter1d(dq0, SIGMA, axis=0, mode="nearest") * act[:, None]
    qn = qR + dqs
    lo = np.minimum(rec.robot.lo["R"][None], qR); hi = np.maximum(rec.robot.hi["R"][None], qR)
    qn = np.clip(qn, lo, hi)
    for t in np.flatnonzero(act):
        T = TR[t].copy(); T[:3, 3] += w[t] * d
        rec.robot.fk(qn[t], sp["L"][t]); cur = rec.robot.frame(rec.robot.tcp["R"])
        err[t, 0] = np.linalg.norm(cur[:3, 3] - T[:3, 3])
        err[t, 1] = np.degrees(np.arccos(np.clip((np.trace(cur[:3, :3].T @ T[:3, :3]) - 1) / 2, -1, 1)))
    S = rec.state.copy(); S[:, RJ] = qn
    k = action_lead(rec.state, rec.action)["k_best"]
    dq = qn - qR; idx = np.minimum(np.arange(rec.T) + k, rec.T - 1)
    A = rec.action.copy(); A[:, RJ] = rec.action[:, RJ] + dq[idx]
    alo = np.minimum(rec.robot.lo["R"], rec.action[:, RJ].min(0)); ahi = np.maximum(rec.robot.hi["R"], rec.action[:, RJ].max(0))
    A[:, RJ] = np.clip(A[:, RJ], alo, ahi)
    # battery track: table pose shifted by d until the grasp, then attached to the (new) right TCP, then the source
    can = np.zeros((rec.T, 4, 4))
    for t in range(rec.T):
        h = rec.holder(t)
        if h == "table":
            can[t] = rec.table.copy(); can[t][:3, 3] += d
        elif h == "R":
            rec.robot.fk(qn[t], sp["L"][t]); can[t] = rec.robot.frame(rec.robot.tcp["R"]) @ rec.XR
        else:
            can[t] = rec.source_can(rec.robot, t)
    return {"state": S.astype(np.float32), "action": A.astype(np.float32), "qR": qn, "qL": sp["L"].copy(), "can": can,
            "tau": np.arange(rec.T, dtype=float), "w": w, "ik_err": err, "lead": k, "marks": marks, "shift_m": d}
