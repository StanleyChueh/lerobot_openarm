"""v3 hand-over generators: G+ (strong geometric baseline) and C (grasp-relative SE(3)), kinematics only.

Every variation becomes ONE `Spec` shared by both conditions: time warp, a new standing-can pose (initial-position
shift Delta0 in the table plane), the hand-over transform D, an optional dual-hold tilt D_dual(t), a receiver regrasp
Gamma, an approach via-point, and a list of CONTACT WAYPOINTS (hand, frame, target TCP pose) at the shared command
schedule: reach end / grasp / lift (giver, if the can moved), pass and release (giver), capture, tilt end and carry
hold (receiver). Both conditions must reach every contact waypoint within the common tolerance.

  G+  ordinary waypoint interpolation of each hand's world delta T* T^w(t_k)^-1 through ITS waypoints (screw-linear
      between consecutive waypoints with smoothstep timing, Gaussian tails sigma=20 frames outside them); the object
      follows its holder; the post-release receiver attachment is the regrasp-aware one (same formula as C) - the
      corrected baseline requested by the v2 review.
  C   phase profiles; the can is carried by the giver (can = T_R X_R) from the shifted table pose; the receiver is
      planned continuously in the carried can's frame (source receiver-in-can relation + regrasp), so the hand-hand
      relation is preserved at every frame of the transfer, not only at waypoints.

Commands: arm a(t) = q(t+lead) + r_src(tau(t)) (residual linearly interpolated in source time); gripper commands are a
TRUE zero-order hold of the source command stream: a_g(t) = a_src[floor(tau(t))] (the last command issued at or before
the source time). Planned gripper STATE is the source state linearly interpolated at tau(t).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .recon3 import recon2, rb
from sfa2 import se3  # noqa: E402
from sfa2.recon import rc  # noqa: E402

CLOSED = recon2.CLOSED
CONDITIONS = ("G+", "C")
SCALES = (1.0, 0.7, 0.49)
IK_TOL = (0.004, 2.0)
CARRY_DELAY, CARRY_RAMP = 10, 40
SIGMA = 20.0
REACH_STANDOFF = 0.18        # m, source TCP-to-lid distance at which the reach retarget must be complete
REACH_RAMP_MIN = 30          # frames: the reach retarget ramp never runs faster than this (may start before the source reach)


def ss(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


@dataclass
class Spec:
    var: dict
    scale: float
    tau: np.ndarray
    marks: dict
    D: np.ndarray
    Gamma: np.ndarray
    a: np.ndarray
    init: np.ndarray            # 4x4 world shift of the standing can (identity if none)
    table_new: np.ndarray
    dual: callable              # t -> 4x4 world transform applied during the dual hold (identity if none)
    waypoints: dict             # side -> list of (t, T*, kind) ; kind "contact" | "via" | "hold"
    warped: dict = field(default_factory=dict)


def zoh_index(tau, T):
    return np.clip(np.floor(tau + 1e-9).astype(int), 0, T - 1)


def make_spec(rec, robot, var: dict, scale: float) -> Spec:
    T, ev = rec.T, rec.ev
    lift = ev["right_close"] + 12
    g_src = rb.Robot.split(rec.state)
    lc0, ro0 = ev["left_close"], ev["right_open"]
    lcs = next((t for t in range(lc0, T) if g_src["gL"][t] < CLOSED), lc0)        # receiver state closed
    rel = next((t for t in range(ro0, T) if g_src["gR"][t] >= CLOSED), ro0)       # giver state open
    dual = var.get("dual_tilt_deg")
    # time warps keep slope 1 around every gripper event, so command->state latencies are the source's
    if dual:
        kc, Ld = int(var.get("close_offset", 0)), int(var["dual_frames"])
        Pe = lc0 + Ld - (ro0 - lcs)                                               # end of the inserted hold
        kn_new = [0, lift, lc0 + kc, lcs + kc, Pe, Pe + (rel - lcs), T - 1]
        kn_src = [0, lift, lc0, lcs, lcs, rel, T - 1]                              # source paused at lcs
        k = 0
        lc, ro, released = lc0 + kc, Pe + (ro0 - lcs), Pe + (rel - lcs)
    else:
        k = int(round(var.get("dt_frames", 0) * scale))
        kn_new = [0, lift, lc0 + k, rel + k, T - 1]
        kn_src = [0, lift, lc0, rel, T - 1]
        lc, ro, released = lc0 + k, ro0 + k, rel + k
    assert all(np.diff(kn_new) > 0), (var["id"], kn_new)
    tau = np.interp(np.arange(T, dtype=float), kn_new, kn_src)
    q_w = {s: np.stack([rc.interp_q(rec.g[s], tau[t]) for t in range(T)]) for s in "RL"}
    T_w = {s: [] for s in "RL"}
    for t in range(T):
        robot.fk(q_w["R"][t], q_w["L"][t])
        for s in "RL":
            T_w[s].append(robot.frame(robot.tcp[s]))
    can_w = [rec.source_can(robot, tau[t]) for t in range(T)]
    iR, iL = rb.NAMES.index("RJ8.pos"), rb.NAMES.index("LJ8.pos")
    gs = {s: np.interp(tau, np.arange(T), rec.state[:, i]) for s, i in (("R", iR), ("L", iL))}
    zi = zoh_index(tau, T)
    ga = {s: rec.action[zi, i] for s, i in (("R", iR), ("L", iL))}
    # marks (new timeline)
    pR = np.array([Tm[:3, 3] for Tm in T_w["R"]])
    pL = np.array([Tm[:3, 3] for Tm in T_w["L"]])
    reach_start = next((t for t in range(T) if np.linalg.norm(pR[t] - pR[0]) > 0.02), 0)
    # object-relative reach: the retarget is complete once the (source) hand is REACH_STANDOFF from the can, so the
    # final approach is the demonstrated hand-can relative path (v3a ramped to rc-10 and swept the hand through the can)
    lid0 = rec.table[:3, 3]
    reach_near = next((t for t in range(reach_start, ev["right_close"]) if np.linalg.norm(pR[t] - lid0) < REACH_STANDOFF),
                      ev["right_close"] - 10)
    reach_near = int(min(max(reach_near, reach_start + 5), ev["right_close"] - 10))
    reach_ramp0 = int(max(0, min(reach_start, reach_near - REACH_RAMP_MIN)))   # >= REACH_RAMP_MIN frames of ramp
    la = next((t for t in range(ev["right_close"], T) if np.linalg.norm(pL[t] - pL[0]) > 0.02), ev["right_close"])
    m = {"right_close": ev["right_close"], "grasp": ev["right_close"] + 8, "lift": lift, "reach_start": reach_start, "reach_ramp0": reach_ramp0,
         "reach_end": reach_near, "left_approach": la, "left_close": lc, "right_open": ro,
         "giver_released": released, "pass": lc - 6, "capture": lc, "approach_mid": int(round(0.5 * (la + lc - 3))),
         "T": T, "dt_frames": k, "knots_new": kn_new, "knots_src": kn_src}
    if dual:
        m.update(tilt_start=lc0 + 2, tilt_end=lc0 + int(var["dual_frames"]) - 2, pass_=lc0 - 6)
        m["pass"] = lc0 - 6                     # pass and tilt window fixed; only the receiver closure moves
    # transforms
    tp, tc, ta = m["pass"], m["capture"], m["approach_mid"]
    sh = np.zeros(3)
    sh[:2] = scale * np.asarray(var.get("init_shift_m", [0, 0]), float)
    init = se3.trans(sh)
    table_new = init @ rec.table
    pivot = can_w[tp][:3, 3]
    D = se3.trans(scale * np.asarray(var.get("delta_m", [0, 0, 0]), float))
    if var.get("rot_deg", 0):
        D = D @ se3.about(se3.rot(var["rot_axis"], scale * var["rot_deg"]), pivot)
    Gamma = se3.rot([0, 0, 1], scale * var.get("regrasp_spin_deg", 0.0)) @ se3.trans([0, 0, scale * var.get("regrasp_shift_m", 0.0)])
    a = scale * np.asarray(var.get("approach_m", [0, 0, 0]), float)
    if dual:
        piv = (D @ can_w[tp])[:3, 3]
        ax, deg = var["dual_axis"], scale * var["dual_tilt_deg"]
        dual_fn = lambda t: se3.about(se3.rot(ax, deg * float(ss((t - m["tilt_start"]) / (m["tilt_end"] - m["tilt_start"])))), piv)  # noqa: E731
    else:
        dual_fn = lambda t: np.eye(4)  # noqa: E731
    S = lambda t: se3.inv(can_w[t]) @ T_w["L"][t]  # noqa: E731
    wp = {"R": [], "L": []}
    if np.linalg.norm(sh) > 0:
        for t in (m["reach_end"], m["grasp"], m["lift"]):
            wp["R"].append((t, init @ T_w["R"][t], "contact" if t != m["reach_end"] else "via"))
    wp["R"].append((tp, D @ T_w["R"][tp], "contact"))
    if dual:
        te = m["tilt_end"]
        wp["R"].append((te, dual_fn(te) @ D @ T_w["R"][te], "contact"))
    last = wp["R"][-1]
    wp["R"].append((released, last[1] @ se3.inv(T_w["R"][last[0]]) @ T_w["R"][released], "hold"))
    if np.linalg.norm(a) > 0:
        wp["L"].append((ta, se3.trans(a) @ T_w["L"][ta], "via"))
    wp["L"].append((tc, dual_fn(tc) @ D @ can_w[tc] @ Gamma @ S(tc), "contact"))
    if dual:
        te = m["tilt_end"]
        wp["L"].append((te, dual_fn(te) @ D @ can_w[te] @ Gamma @ S(te), "contact"))
    lastL = wp["L"][-1]
    th = released + CARRY_DELAY
    wp["L"].append((th, lastL[1] @ se3.inv(T_w["L"][lastL[0]]) @ T_w["L"][th], "hold"))
    return Spec(var, scale, tau, m, D, Gamma, a, init, table_new, dual_fn, wp,
                {"q": q_w, "T": T_w, "can": can_w, "grip_state_raw": gs, "grip_action_raw": ga})


# ----------------------------------------------------------------------------- trajectories
def waypoint_delta(wps, Tw, t):
    """G+: world delta at frame t by interpolation through waypoint deltas (Gaussian tails outside)."""
    ds = [(tk, Tk @ se3.inv(Tw[tk])) for tk, Tk, _ in wps]
    if t <= ds[0][0]:
        return se3.power(ds[0][1], float(np.exp(-0.5 * ((t - ds[0][0]) / SIGMA) ** 2)))
    if t >= ds[-1][0]:
        return se3.power(ds[-1][1], float(np.exp(-0.5 * ((t - ds[-1][0]) / SIGMA) ** 2)))
    for (t0, D0), (t1, D1) in zip(ds[:-1], ds[1:]):
        if t0 <= t <= t1:
            u = float(ss((t - t0) / max(1e-9, t1 - t0)))
            return se3.power(D1 @ se3.inv(D0), u) @ D0


def ik_follow(robot, side, targets, q_ref):
    qs, es = np.zeros_like(q_ref), np.zeros((len(q_ref), 2))
    q = q_ref[0]
    for t, Tt in enumerate(targets):
        if Tt is None:
            q = q_ref[t]
            qs[t] = q
            continue
        q, pe, re = robot.ik(side, Tt, 0.5 * (q + q_ref[t]))
        qs[t], es[t] = q, (pe, re)
    return qs, es


def close(A, B):
    return np.abs(A - B).max() < 1e-12


def holder_of(rec, spec, t):
    return rec.source_holder(int(round(spec.tau[t])))


def regrasp_attachment(rec, spec):
    ro = spec.marks["right_open"]
    S_r = se3.inv(spec.warped["can"][ro - 1]) @ spec.warped["T"]["L"][ro - 1]
    return S_r, se3.inv(S_r) @ se3.inv(spec.Gamma) @ S_r @ rec.X["L"]


def plan(rec, robot, spec: Spec, cond: str) -> dict:
    T, m = rec.T, spec.marks
    Tw, qw, canw = spec.warped["T"], spec.warped["q"], spec.warped["can"]
    t_ = np.arange(T, dtype=float)
    S_r, XLn = regrasp_attachment(rec, spec)
    ro = m["right_open"]
    if cond == "G+":
        tgtR = [waypoint_delta(spec.waypoints["R"], Tw["R"], t) @ Tw["R"][t] for t in range(T)]
        qR, eR = ik_follow(robot, "R", [None if close(x, Tw["R"][t]) else x for t, x in enumerate(tgtR)], qw["R"])
        tgtL = [waypoint_delta(spec.waypoints["L"], Tw["L"], t) @ Tw["L"][t] for t in range(T)]
        qL, eL = ik_follow(robot, "L", [None if close(x, Tw["L"][t]) else x for t, x in enumerate(tgtL)], qw["L"])
        can = []
        for t in range(T):
            h = holder_of(rec, spec, t)
            if h == "table":
                can.append(spec.table_new.copy())
            else:
                robot.fk(qR[t], qL[t])
                can.append(robot.frame(robot.tcp[h]) @ (rec.X["R"] if h == "R" else XLn))
    else:
        # giver: reach to the moved can, carry to the pass, optional dual tilt, ramp out after release
        w_reach = ss((t_ - m["reach_ramp0"]) / max(1, m["reach_end"] - m["reach_ramp0"]))
        w_pass = ss((t_ - m["lift"]) / max(1, m["pass"] - m["lift"]))
        out_R = 1 - ss((t_ - m["giver_released"]) / 30.0)
        tgtR = []
        for t in range(T):
            if t < m["lift"]:
                Dt = se3.power(spec.init, float(w_reach[t]))
            else:
                Dt = spec.dual(t) @ se3.power(spec.D @ se3.inv(spec.init), float(w_pass[t])) @ spec.init
                Dt = se3.power(Dt, float(out_R[t]))
            x = Dt @ Tw["R"][t]
            tgtR.append(None if close(x, Tw["R"][t]) else x)
        qR, eR = ik_follow(robot, "R", tgtR, qw["R"])
        canC = [None] * T
        for t in range(T):
            h = holder_of(rec, spec, t)
            if h == "table":
                canC[t] = spec.table_new.copy()
            elif h == "R":
                robot.fk(qR[t], qw["L"][t])
                canC[t] = robot.frame(robot.tcp["R"]) @ rec.X["R"]
        tc = m["capture"]
        span = max(1, tc - 3 - m["left_approach"])
        wL = ss((t_ - m["left_approach"]) / span)
        wa = np.sin(np.pi * np.clip((t_ - m["left_approach"]) / span, 0, 1)) ** 2
        out_L = 1 - ss((t_ - m["giver_released"] - CARRY_DELAY) / CARRY_RAMP)
        G_r = se3.inv(S_r) @ spec.Gamma @ S_r
        qL, eL = np.zeros_like(qw["L"]), np.zeros((T, 2))
        q = qw["L"][0]
        Hc = None
        for t in range(T):
            if t < ro:
                T_obj = canC[t] @ spec.Gamma @ se3.inv(canw[t]) @ Tw["L"][t]
                x = T_obj if t >= tc else se3.blend(Tw["L"][t], T_obj, float(wL[t]))
                x = se3.trans(spec.a * wa[t]) @ x
            else:
                if Hc is None:
                    robot.fk(qR[ro - 1], qL[ro - 1])
                    Hc = robot.frame(robot.tcp["L"]) @ se3.inv(Tw["L"][ro - 1] @ G_r)
                x = se3.power(Hc, float(out_L[t])) @ Tw["L"][t] @ G_r
            if close(x, Tw["L"][t]):
                q = qw["L"][t]
                qL[t] = q
                continue
            q, pe, re = robot.ik("L", x, 0.5 * (q + qw["L"][t]))
            qL[t], eL[t] = q, (pe, re)
        for t in range(ro, T):
            robot.fk(qR[t], qL[t])
            canC[t] = robot.frame(robot.tcp["L"]) @ XLn
        can = canC
    Q = {"R": qR, "L": qL}
    # common-tolerance target check on contact waypoints
    tg = []
    for s in "RL":
        for tk, Tk, kind in spec.waypoints[s]:
            robot.fk(Q["R"][tk], Q["L"][tk])
            Ta = robot.frame(robot.tcp[s])
            tg.append({"side": s, "frame": int(tk), "kind": kind, "err_mm": float(1000 * np.linalg.norm(Ta[:3, 3] - Tk[:3, 3])),
                       "err_deg": se3.angle_deg(Ta[:3, :3], Tk[:3, :3]),
                       "requested_mm": float(1000 * np.linalg.norm(Tk[:3, 3] - Tw[s][tk][:3, 3]))})
    pe = max(float(eR[:, 0].max()), float(eL[:, 0].max()))
    re = max(float(eR[:, 1].max()), float(eL[:, 1].max()))
    gs, ga = spec.warped["grip_state_raw"], spec.warped["grip_action_raw"]
    state = rb.Robot.merge(qR, qL, gs["R"], gs["L"])
    action, valid = actions(rec, Q, spec)
    can = np.stack(can)
    return {"cond": cond, "var": spec.var["id"], "scale": spec.scale, "state": state, "action": action, "valid": valid,
            "q": Q, "can": can, "tau": spec.tau, "marks": m, "targets": tg,
            "ik": {"max_pos_err_mm": 1000 * pe, "max_rot_err_deg": re, "ok": bool(pe <= IK_TOL[0] and re <= IK_TOL[1])},
            "init_shift_m": (spec.init[:3, 3]).tolist(), "table_new": spec.table_new,
            "pass_can_displacement_mm": float(1000 * np.linalg.norm(can[m["pass"]][:3, 3] - canw[m["pass"]][:3, 3]))}


def actions(rec, Q, spec):
    T, L = rec.T, rec.lead
    valid = np.arange(T) <= T - 1 - L
    out = {}
    for s in "RL":
        r = rec.ga[s][: T - L] - rec.g[s][L:]
        rw = np.stack([np.interp(spec.tau, np.arange(T - L), r[:, j]) for j in range(7)], 1)
        a = np.full((T, 7), np.nan)
        a[valid] = Q[s][np.arange(T)[valid] + L] + rw[valid]
        out[s] = a
    ga = spec.warped["grip_action_raw"]
    act = rb.Robot.merge(out["R"], out["L"], ga["R"], ga["L"])
    act[~valid] = np.nan
    return act, valid


def targets_ok(p, tol) -> bool:
    return all(t["err_mm"] <= tol["mm"] and t["err_deg"] <= tol["deg"] for t in p["targets"] if t["kind"] == "contact")
