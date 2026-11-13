"""Trajectory synthesis (kinematics only). Two variation families, defined once for every task:

F1 object position (MimicGen / DemoGen-style object-centric segment transform)
   An object resting on the table is translated by d. The arm that grasps it gets the TCP offset w(t) d:
   w = smoothstep 0->1 from the arm's motion onset to grasp-15, 1 through grasp+10, smoothstep 1->0 until the REJOIN
   frame = the lift apex (first maximum of that TCP's height within 4 s after the grasp, >= 30 frames after it and
   >= 10 frames before the next gripper event of any arm). After the rejoin the trajectory is the recording.

F2 interaction location (coordinated multi-arm transform, cf. DexMimicGen)
   The multi-arm interaction (hand-over, insertion) is moved by d: every arm's TCP gets w(t) d, so the hands and every
   object they hold move rigidly together and the recorded hand-hand / hand-object relations are kept at every frame.
   Interaction window = first entry to last exit of "TCPs closer than d_interact", clipped so that it starts after
   every grasp of an object resting on the table (+10 frames) and ends before any object is put down (-10 frames);
   w ramps 0->1 over `ramp` frames before the window and 1->0 over `ramp` frames after it. Objects on the table
   therefore never move.

Both: orientation unchanged; joints by damped-least-squares IK bounded by [URDF limits U recorded pose], warm-started
along the source motion; joint offsets Gaussian-smoothed (sigma 3 frames), then refined by IK on the full-offset (contact)
frames so the hand is exact where it touches, and re-verified by FK; actions
a'(t) = a(t) + dq(t + k) with the recorded command-to-state lead k; grippers bit-identical to the recording."""
from __future__ import annotations
import numpy as np
import pinocchio as pin
from scipy.ndimage import gaussian_filter1d


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0); return x * x * (3 - 2 * x)


def action_lead(S, A, kmax=6):
    errs = [np.abs(A[:len(A) - k] - S[k:]).mean() for k in range(kmax + 1)]
    return int(np.argmin(errs))


def ik(robot, side, T_target, q0, q_all, lo, hi, iters=100, damp=1e-3):
    q = np.clip(np.asarray(q0, float), lo, hi); fid = robot.tcp[side]; target = pin.SE3(T_target[:3, :3], T_target[:3, 3])
    qa = dict(q_all)
    for _ in range(iters):
        qa[side] = q; x = robot.q_full(qa); pin.framesForwardKinematics(robot.m, robot.d, x)
        e = pin.log6(robot.d.oMf[fid].inverse() * target).vector
        if np.linalg.norm(e[:3]) < 1e-4 and np.linalg.norm(e[3:]) < 1e-3:
            break
        J = np.asarray(pin.computeFrameJacobian(robot.m, robot.d, x, fid, pin.LOCAL)).reshape(6,robot.m.nv)[:, robot.vidx[side]]
        q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), e), lo, hi)
    return q


def state_bounds(rec, s, qs, action_bounds=False, margin=0.0):
    """Per-frame IK bounds for arm s: URDF ∪ recorded state; with action_bounds also keep each shifted label
    A(t) + dq(t + k) inside the checker's URDF ∪ source-action envelope, `margin` rad inside it (retiming
    interpolation can add ~0.1 deg); never tighter than the recorded pose."""
    robot, S, A = rec.robot, rec.state, rec.action
    lo = np.minimum(robot.lo[s][None], qs); hi = np.maximum(robot.hi[s][None], qs)
    if action_bounds:
        a = A[:, robot.sidx[s]]; k = action_lead(S, A); tgt = np.minimum(np.arange(len(S)) + k, len(S) - 1)
        amax = np.full_like(qs, -np.inf); amin = np.full_like(qs, np.inf)
        np.maximum.at(amax, tgt, a); np.minimum.at(amin, tgt, a)  # labels that read dq at frame t'
        with np.errstate(invalid="ignore"):
            hi = np.minimum(hi, np.where(np.isfinite(amax), qs + np.maximum(robot.hi[s], a.max(0)) - amax - margin, np.inf))
            lo = np.maximum(lo, np.where(np.isfinite(amin), qs + np.minimum(robot.lo[s], a.min(0)) - amin + margin, -np.inf))
        if margin:
            hi = np.maximum(hi, qs); lo = np.minimum(lo, qs)
    return lo, hi


def _retarget(rec, W, d, sides, action_bounds=False):
    """W: (T,) weight per frame; d: (3,) world offset; sides: arms that get T(t) + W(t) d.
    action_bounds: also bound each IK state so its shifted label A(t) + dq(t + k) stays inside the checker's
    URDF ∪ source-action envelope (a feasibility constraint on the solve; labels are never clipped afterwards)."""
    robot, S, A = rec.robot, rec.state, rec.action
    sp = robot.split(S); q = {s: sp[s].copy() for s in robot.arms}; qn = {s: sp[s].copy() for s in robot.arms}
    TT = {s: robot.tcp_traj({a: sp[a] for a in robot.arms}, s) for s in sides}
    act = W > 1e-9; err = np.zeros((len(S), 2))
    for s in sides:
        lo, hi = state_bounds(rec, s, q[s], action_bounds)
        for t in np.flatnonzero(act):
            Tt = TT[s][t].copy(); Tt[:3, 3] += W[t] * d
            q0 = qn[s][t - 1] + (q[s][t] - q[s][t - 1]) if t > 0 and act[t - 1] else q[s][t]
            qn[s][t] = ik(robot, s, Tt, q0, {a: q[a][t] for a in robot.arms}, lo[t], hi[t])
        dq = gaussian_filter1d(qn[s] - q[s], 3.0, axis=0, mode="nearest") * act[:, None]
        qn[s] = np.clip(q[s] + dq, lo, hi)
        # exact at contact: where the full offset applies (W = 1), refine the smoothed solution by IK (warm start = smoothed
        # joints, so the correction is the small smoothing residual); blending frames keep the smoothed solution
        for t in np.flatnonzero(W >= 0.999):
            Tt = TT[s][t].copy(); Tt[:3, 3] += W[t] * d
            qn[s][t] = ik(robot, s, Tt, qn[s][t], {a: (qn[a][t] if a in sides else q[a][t]) for a in robot.arms}, lo[t], hi[t], iters=30)
    for t in np.flatnonzero(act):
        robot.fk({a: qn[a][t] for a in robot.arms})
        for s in sides:
            Tt = TT[s][t].copy(); Tt[:3, 3] += W[t] * d; cur = robot.frame(robot.tcp[s])
            err[t, 0] = max(err[t, 0], np.linalg.norm(cur[:3, 3] - Tt[:3, 3]))
            err[t, 1] = max(err[t, 1], np.degrees(np.arccos(np.clip((np.trace(cur[:3, :3].T @ Tt[:3, :3]) - 1) / 2, -1, 1))))
    Sn = S.copy(); An = A.copy(); k = action_lead(S, A); idx = np.minimum(np.arange(len(S)) + k, len(S) - 1)
    for s in sides:
        Sn[:, robot.sidx[s]] = qn[s]
        An[:, robot.sidx[s]] = A[:, robot.sidx[s]] + (qn[s] - q[s])[idx]
        # Keep the exact nominal label. Out-of-limit actions must be rejected, never clipped away
        # from the planned future state while retaining its RGB/state supervision.
    return Sn, An, qn, err, k


def object_tracks(rec, qn, moved_obj=None, d=None):
    """per object (T,4,4) world pose for the candidate joints: resting objects stay (or are shifted by d if it is the
    moved object), held objects follow their holder's NEW TCP with the recorded attachment."""
    robot = rec.robot; tracks = []
    for k, o in enumerate(rec.objects):
        P = np.zeros((rec.T, 4, 4)); xi = -1
        for t in range(rec.T):
            h = rec.holder_at(k, t)
            if h == "table":
                P[t] = o["table"].copy()
                if k == moved_obj:
                    P[t][:3, 3] += d
            elif h == "placed":
                P[t] = rec.obj_pose(k, t)
            else:
                robot.fk({a: qn[a][t] for a in robot.arms}); P[t] = robot.frame(robot.tcp[h]) @ _attach(o, rec, h, t)
        tracks.append(P)
    return tracks


def _attach(o, rec, h, t):
    xi = 0
    for (hh, a, b) in o["holder"]:
        if hh in rec.arms:
            if a <= t < b:
                return o["X"][xi]
            xi += 1
    raise ValueError("no holder")


def f1_schedule(rec, k):
    o = rec.objects[k]; arm = o["arm"]; tg = o["grasp_frame"]
    p = rec.robot.tcp_from_state(rec.state, arm)[:, :3, 3]
    v = np.r_[0, np.linalg.norm(np.diff(p, axis=0), axis=1)] * rec.ctx.src.fps
    mov = np.flatnonzero(v[: tg - 15] > 0.02); t0 = int(mov[0]) if len(mov) else 0
    nxt = min([e for s in rec.ev for kind in ("closes", "opens") for e in rec.ev[s][kind] if e > tg] + [rec.T])
    win = np.arange(tg + 30, min(tg + 4 * int(rec.ctx.src.fps), nxt - 10))
    if len(win) == 0:
        return None
    tr = int(win[np.argmax(p[win, 2])])
    if not (t0 < tg - 15 and tg + 10 < tr):
        return None
    return {"t_motion": t0, "t_grasp": tg, "t_rejoin": tr, "next_event": int(nxt)}


def f1_plan(rec, k, d, action_bounds=False):
    m = f1_schedule(rec, k)
    if m is None:
        return None
    t = np.arange(rec.T, dtype=float); tin, tout = m["t_grasp"] - 15, m["t_grasp"] + 10
    W = np.where(t < tin, smoothstep((t - m["t_motion"]) / (tin - m["t_motion"])), 1.0)
    W = np.where(t > tout, 1 - smoothstep((t - tout) / (m["t_rejoin"] - tout)), W)
    d = np.asarray(d, float); arm = rec.objects[k]["arm"]
    Sn, An, qn, err, lead = _retarget(rec, W, d, [arm], action_bounds)
    return {"family": "object_position", "object": rec.objects[k]["name"], "shift_m": d.tolist(), "marks": m, "w": W,
            "state": Sn, "action": An, "q": qn, "ik_err": err, "lead": lead, "moved_arms": [arm],
            "objects": object_tracks(rec, qn, k, d), "edited": (W > 1e-9), "t_rejoin": m["t_rejoin"]}


def f2_schedule(rec, d_interact=0.20, ramp=45):
    arms = rec.arms
    if len(arms) < 2:
        return None
    P = {s: rec.robot.tcp_from_state(rec.state, s)[:, :3, 3] for s in arms}
    dist = np.min([np.linalg.norm(P[a] - P[b], axis=1) for i, a in enumerate(arms) for b in arms[i + 1:]], axis=0)
    close = np.flatnonzero(dist < d_interact)
    if not len(close):
        return None
    t_a, t_b = int(close[0]), int(close[-1])
    grasps_table = [o["grasp_frame"] for o in rec.objects]
    puts = [a for o in rec.objects for (h, a, b) in o["holder"] if h == "placed"]
    t_a = max([t_a] + [g + 10 + ramp for g in grasps_table])
    t_b = min([t_b] + [p_ - 10 - ramp for p_ in puts] + [rec.T - 1 - ramp])
    if t_b - t_a < 10:
        return None
    return {"t_ramp_in": t_a - ramp, "t_window_start": t_a, "t_window_end": t_b, "t_ramp_out_end": t_b + ramp,
            "min_tcp_distance_m": float(dist.min())}


def f2_plan(rec, d, d_interact=0.20, ramp=45):
    m = f2_schedule(rec, d_interact, ramp)
    if m is None:
        return None
    t = np.arange(rec.T, dtype=float)
    W = np.minimum(smoothstep((t - m["t_ramp_in"]) / ramp), 1 - smoothstep((t - m["t_window_end"]) / ramp))
    d = np.asarray(d, float)
    Sn, An, qn, err, lead = _retarget(rec, W, d, list(rec.arms))
    return {"family": "interaction_location", "shift_m": d.tolist(), "marks": m, "w": W, "state": Sn, "action": An, "q": qn,
            "ik_err": err, "lead": lead, "moved_arms": list(rec.arms), "objects": object_tracks(rec, qn),
            "edited": (W > 1e-9), "t_rejoin": m["t_ramp_out_end"]}
