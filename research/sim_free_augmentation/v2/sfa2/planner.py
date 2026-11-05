"""Matched G / P / C hand-over generators (kinematics only; no physics, no simulator).

Every variation is first turned into ONE set of desired targets (`Spec`), shared by all three conditions:
    * time warp tau (new frame -> source frame) for timing variations (identity otherwise),
    * giver TCP target at the pass frame          T_R* = D . T_R^w(t_p)
    * receiver TCP target at the capture frame    T_L* = D . can^w(t_c) . Gamma . S_c,  S_c = can^w(t_c)^-1 T_L^w(t_c)
    * receiver approach waypoint offset `a` at the mid-approach frame t_a (zero at capture),
where D is a world SE(3) hand-over transform (pivot at the source lid at t_p) and Gamma a regrasp in the can frame.
The conditions differ ONLY in how they turn those targets into trajectories and in their object model:

  G  geometry-only: per-hand world deltas Delta_h = T_h* T_h^w(t_key)^-1 applied with a phase-blind Gaussian time
     profile centred on each target's own frame (sigma frozen on episode 0 by maximising G's own acceptance, so G is
     tuned, not handicapped). Object follows its source holder with the source attachment.
  P  + contact-phase coordination: the same deltas with phase-aware profiles - giver delta 0 until the lift, full by
     the pass, held through dual contact, released after the giver opens; receiver ramps in over its approach, full
     before it closes, held through the carry; approach detour zero before closing. Object as in G.
  C  + grasp-relative SE(3): giver as in P; the can is carried by the giver (can = T_R X_R); the receiver is planned
     IN THE CAN FRAME: its source pose relative to the source can, transplanted to the new can (plus the regrasp),
     blended in over the approach, exact from capture to release, then the world delta frozen at the release and
     ramped out over the carry (the can returns to its source carry path, the regrasp relation is kept); the
     can then follows the receiver with the regrasped attachment S_r^-1 Gamma^-1 S_r X_L (S_r: source
     receiver-in-can relation at the release), which keeps the transfer drift equal to the source's.

Identical inputs, solver, warm start, joint margins and repair budget (scale 1.0 -> 0.7 -> 0.49) for all conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import se3
from .recon import CLOSED as CLOSED_OPENNESS, Recon, rb, rc

CONDITIONS = ("G", "P", "C")
SCALES = (1.0, 0.7, 0.49)
IK_TOL = (0.004, 2.0)        # m, deg - targets the solver must reach for the plan to count as realised
CARRY_DELAY, CARRY_RAMP = 10, 40   # frames after the giver's release before / over which P and C return to the source carry


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


@dataclass
class Spec:
    var: dict
    scale: float
    tau: np.ndarray                    # (T,) new frame -> source frame (state time)
    marks: dict                        # event frames on the NEW timeline
    D: np.ndarray
    Gamma: np.ndarray
    G_L: np.ndarray
    a: np.ndarray
    targets: dict                      # "R": (t_p, T_R*), "L": (t_c, T_L*), "La": (t_a, a)
    warped: dict = field(default_factory=dict)   # q_w, T_w, can_w, grip_state_raw, grip_action_raw


def time_knots(rec: Recon, k: int):
    ev, T = rec.ev, rec.T
    lift = ev["right_close"] + 12
    new = [0, lift, ev["left_close"] + k, ev["right_open"] + k, T - 1]
    src = [0, lift, ev["left_close"], ev["right_open"], T - 1]
    return np.array(new, float), np.array(src, float)


def inverse_time(kn_new, kn_src, t_src):
    return float(np.interp(t_src, kn_src, kn_new))


def make_spec(rec: Recon, robot, var: dict, scale: float) -> Spec:
    T, ev = rec.T, rec.ev
    k = int(round(var.get("dt_frames", 0) * scale))
    kn_new, kn_src = time_knots(rec, k)
    tau = np.interp(np.arange(T, dtype=float), kn_new, kn_src)
    # warped source
    q_w = {s: np.stack([rc.interp_q(rec.g[s], tau[t]) for t in range(T)]) for s in "RL"}
    T_w = {s: [] for s in "RL"}
    for t in range(T):
        robot.fk(q_w["R"][t], q_w["L"][t])
        for s in "RL":
            T_w[s].append(robot.frame(robot.tcp[s]))
    can_w = [rec.source_can(robot, tau[t]) for t in range(T)]
    iR, iL = rb.NAMES.index("RJ8.pos"), rb.NAMES.index("LJ8.pos")
    gs = {s: np.interp(tau, np.arange(T), rec.state[:, i]) for s, i in (("R", iR), ("L", iL))}
    ga = {s: rec.action[np.clip(np.round(tau).astype(int), 0, T - 1), i] for s, i in (("R", iR), ("L", iL))}
    # phase marks on the new timeline
    m_src = {"right_close": ev["right_close"], "left_close": ev["left_close"], "right_open": ev["right_open"]}
    pL = np.array([Tm[:3, 3] for Tm in T_w["L"]])
    la = next((t for t in range(ev["right_close"], T) if np.linalg.norm(pL[t] - pL[0]) > 0.02), ev["right_close"])
    lc, ro = ev["left_close"] + k, ev["right_open"] + k
    gR_src = rb.Robot.split(rec.state)["gR"]
    rel_src = next((t for t in range(ev["right_open"], T) if gR_src[t] >= CLOSED_OPENNESS), ev["right_open"])
    released = int(round(inverse_time(kn_new, kn_src, rel_src)))     # giver gripper state open (release complete)
    marks = {"right_close": ev["right_close"], "grasp": ev["right_close"] + 8, "lift": ev["right_close"] + 12,
             "left_approach": la, "left_close": lc, "right_open": ro, "giver_released": released, "pass": lc - 6, "capture": lc,
             "approach_mid": int(round(0.5 * (la + lc - 3))), "T": T, "dt_frames": k, "source": m_src}
    # desired transforms
    tp, tc, ta = marks["pass"], marks["capture"], marks["approach_mid"]
    pivot = can_w[tp][:3, 3]
    D = se3.trans(scale * np.asarray(var.get("delta_m", [0, 0, 0]), float))
    if var.get("rot_deg", 0):
        D = D @ se3.about(se3.rot(var["rot_axis"], scale * var["rot_deg"]), pivot)
    Gamma = se3.rot([0, 0, 1], scale * var.get("regrasp_spin_deg", 0.0)) @ se3.trans([0, 0, scale * var.get("regrasp_shift_m", 0.0)])
    # regrasp Gamma acts in the frame of the can the receiver actually takes (giver-carried at capture):
    # T_L* = D can_w(t_c) Gamma S_c  with S_c = can_w(t_c)^-1 T_L^w(t_c), i.e. right-multiplication by G_L
    S_c = se3.inv(can_w[lc]) @ T_w["L"][lc]
    G_L = se3.inv(S_c) @ Gamma @ S_c
    a = scale * np.asarray(var.get("approach_m", [0, 0, 0]), float)
    targets = {"R": (tp, D @ T_w["R"][tp]), "L": (tc, D @ T_w["L"][tc] @ G_L), "La": (ta, a)}
    return Spec(var, scale, tau, marks, D, Gamma, G_L, a, targets,
                {"q": q_w, "T": T_w, "can": can_w, "grip_state_raw": gs, "grip_action_raw": ga})


# ----------------------------------------------------------------------------- time profiles
def profiles(cond: str, m: dict, sigma: float) -> dict:
    t = np.arange(m["T"], dtype=float)
    if cond == "G":
        g = lambda c: np.exp(-0.5 * ((t - c) / sigma) ** 2)   # noqa: E731
        return {"R": g(m["pass"]), "L": g(m["capture"]), "a": g(m["approach_mid"])}
    wR = np.minimum(smoothstep((t - m["lift"]) / max(1, m["pass"] - m["lift"])),
                    1 - smoothstep((t - m["giver_released"]) / 30.0))
    span = max(1, m["capture"] - 3 - m["left_approach"])
    out = 1 - smoothstep((t - m["giver_released"] - CARRY_DELAY) / CARRY_RAMP)     # back to the source carry
    wL = np.minimum(smoothstep((t - m["left_approach"]) / span), out)
    x = np.clip((t - m["left_approach"]) / span, 0, 1)
    wa = np.sin(np.pi * x) ** 2
    return {"R": wR, "L": wL, "a": wa, "out": out}


def ik_track(robot, side, targets: list, q_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """targets: per frame 4x4 or None (= keep the warped source configuration)."""
    qs, es = [], []
    q = q_ref[0]
    for t, Tt in enumerate(targets):
        if Tt is None:
            q = q_ref[t]
            qs.append(q)
            es.append((0.0, 0.0))
            continue
        q, pe, re = robot.ik(side, Tt, 0.5 * (q + q_ref[t]))
        qs.append(q)
        es.append((pe, re))
    return np.array(qs), np.array(es)


def same(A, B, tol=1e-12):
    return np.abs(A - B).max() < tol


def plan(rec: Recon, robot, spec: Spec, cond: str, sigma_G: float) -> dict:
    T, m = rec.T, spec.marks
    w = profiles(cond, m, sigma_G)
    Tw, qw, canw = spec.warped["T"], spec.warped["q"], spec.warped["can"]
    DR = spec.targets["R"][1] @ se3.inv(Tw["R"][spec.targets["R"][0]])
    DL = spec.targets["L"][1] @ se3.inv(Tw["L"][spec.targets["L"][0]])
    a = spec.a
    # giver (identical construction in P and C)
    tgtR = []
    for t in range(T):
        Tt = se3.power(DR, float(w["R"][t])) @ Tw["R"][t] if w["R"][t] > 1e-6 else None
        tgtR.append(None if Tt is not None and same(Tt, Tw["R"][t]) else Tt)
    qR, eR = ik_track(robot, "R", tgtR, qw["R"])
    # object (can) trajectory, and the receiver
    holder = []
    for t in range(T):
        ts = int(round(spec.tau[t]))
        holder.append(rec.source_holder(ts))
    tgtL = []
    if cond in ("G", "P"):
        for t in range(T):
            Tt = se3.trans(a * w["a"][t]) @ se3.power(DL, float(w["L"][t])) @ Tw["L"][t]
            tgtL.append(None if same(Tt, Tw["L"][t]) else Tt)
        qL, eL = ik_track(robot, "L", tgtL, qw["L"])
        can = []
        for t in range(T):
            h = holder[t]
            if h == "table":
                can.append(rec.table.copy())
            else:
                robot.fk(qR[t] if h == "R" else qw["R"][t], qL[t])
                can.append(robot.frame(robot.tcp[h]) @ rec.X[h])
    else:  # C
        canC = [None] * T
        for t in range(T):
            if holder[t] == "table":
                canC[t] = rec.table.copy()
            elif holder[t] == "R":
                robot.fk(qR[t], qw["L"][t])
                canC[t] = robot.frame(robot.tcp["R"]) @ rec.X["R"]
        ro, tc = m["right_open"], m["capture"]
        Hc = None
        qL = np.zeros_like(qw["L"])
        eL = np.zeros((T, 2))
        q = qw["L"][0]
        S_r = se3.inv(canw[ro - 1]) @ Tw["L"][ro - 1]          # source receiver-in-can relation at release
        G_r = se3.inv(S_r) @ spec.Gamma @ S_r
        for t in range(T):
            if t < ro:
                T_obj = canC[t] @ spec.Gamma @ se3.inv(canw[t]) @ Tw["L"][t]
                Tt = T_obj if t >= tc else se3.blend(Tw["L"][t], T_obj, float(w["L"][t]))
                Tt = se3.trans(a * w["a"][t]) @ Tt
            else:
                if Hc is None:
                    robot.fk(qR[ro - 1], qL[ro - 1])
                    Hc = robot.frame(robot.tcp["L"]) @ se3.inv(Tw["L"][ro - 1] @ G_r)
                Tt = se3.power(Hc, float(w["out"][t])) @ Tw["L"][t] @ G_r
            if same(Tt, Tw["L"][t]):
                q = qw["L"][t]
                qL[t], eL[t] = q, (0.0, 0.0)
                continue
            q, pe, re = robot.ik("L", Tt, 0.5 * (q + qw["L"][t]))
            qL[t], eL[t] = q, (pe, re)
        XLn = se3.inv(S_r) @ se3.inv(spec.Gamma) @ S_r @ rec.X["L"]   # regrasped receiver attachment
        for t in range(ro, T):
            robot.fk(qR[t], qL[t])
            canC[t] = robot.frame(robot.tcp["L"]) @ XLn
        can = canC
    Q = {"R": qR, "L": qL}
    # realised vs requested targets
    ach = {}
    for s in "RL":
        tk, Tst = spec.targets[s]
        robot.fk(Q["R"][tk], Q["L"][tk])
        Ta = robot.frame(robot.tcp[s])
        req = Tst @ se3.inv(Tw[s][tk])
        got = Ta @ se3.inv(Tw[s][tk])
        ach[s] = {"frame": int(tk), "requested_mm": float(1000 * np.linalg.norm(Tst[:3, 3] - Tw[s][tk][:3, 3])),
                  "achieved_mm": float(1000 * np.linalg.norm(Ta[:3, 3] - Tw[s][tk][:3, 3])),
                  "requested_rot_deg": se3.angle_deg(req[:3, :3]), "achieved_rot_deg": se3.angle_deg(got[:3, :3]),
                  "target_err_mm": float(1000 * np.linalg.norm(Ta[:3, 3] - Tst[:3, 3])),
                  "target_err_deg": se3.angle_deg(Ta[:3, :3], Tst[:3, :3])}
    pe = max(float(eR[:, 0].max()), float(eL[:, 0].max()))
    re = max(float(eR[:, 1].max()), float(eL[:, 1].max()))
    # dataset vectors
    gs, ga = spec.warped["grip_state_raw"], spec.warped["grip_action_raw"]
    state = rb.Robot.merge(qR, qL, gs["R"], gs["L"])
    action, valid = actions(rec, Q, spec)
    return {"cond": cond, "var": spec.var["id"], "scale": spec.scale, "state": state, "action": action,
            "valid": valid, "q": Q, "can": np.stack(can), "tau": spec.tau, "marks": m, "holder_assumed": holder,
            "ik": {"max_pos_err_mm": 1000 * pe, "max_rot_err_deg": re,
                   "ok": bool(pe <= IK_TOL[0] and re <= IK_TOL[1])}, "targets": ach,
            "pass_can_displacement_mm": float(1000 * np.linalg.norm(can[m["pass"]][:3, 3] - canw[m["pass"]][:3, 3]))}


def actions(rec: Recon, Q: dict, spec: Spec):
    """a_new(t) = q_new(t + lead) + r_src(tau(t)),  r_src(t) = a_src(t) - q_src(t + lead)  (arm joints).

    The recorded command-to-state residual is transported along the time warp; gripper commands are the source
    commands under zero-order hold of the warp. Rows t > T-1-lead have no supported q_new(t+lead): NaN, invalid."""
    T, L = rec.T, rec.lead
    valid = np.arange(T) <= T - 1 - L
    out = {}
    for s in "RL":
        r = rec.ga[s][: T - L] - rec.g[s][L:]
        rw = np.stack([np.array([np.interp(spec.tau[t], np.arange(T - L), r[:, j]) for j in range(7)]) for t in range(T)])
        a = np.full((T, 7), np.nan)
        a[valid] = Q[s][np.arange(T)[valid] + L] + rw[valid]
        out[s] = a
    ga = spec.warped["grip_action_raw"]
    act = rb.Robot.merge(out["R"], out["L"], ga["R"], ga["L"])
    act[~valid] = np.nan
    return act, valid


def plan_with_repair(rec, robot, var, cond, sigma_G, verify_fn) -> dict:
    """identical repair budget for every condition: shrink the whole variation 1.0 -> 0.7 -> 0.49."""
    tries = []
    for s in SCALES:
        spec = make_spec(rec, robot, var, s)
        p = plan(rec, robot, spec, cond, sigma_G)
        v = verify_fn(p, spec)
        p["verify"] = v
        p["accepted"] = bool(p["ik"]["ok"] and v["ok"])
        tries.append({"scale": s, "ik_ok": p["ik"]["ok"], "verify_ok": v["ok"], "failed": v["failed"]})
        if p["accepted"]:
            break
    p["tries"] = tries
    p["spec"] = spec
    return p
