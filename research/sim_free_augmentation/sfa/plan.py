"""New hand-over trajectories by IK, with and without a lightweight kinematic contact model. No physics anywhere.

Both generators move the hand-over to a new place by Cartesian TCP offsets solved with IK on the URDF, warm-started
from the recorded configuration (so the arm posture stays close to the demonstrator's).

* geometry-only (G): each arm gets its own smooth offset (same magnitude, independent direction), applied with a
  time bump centred on the hand-over and blind to contact phases. The can is rendered with whichever hand held it
  in the ORIGINAL episode (the most favourable thing a geometry-only pipeline can do).
* contact-constrained (K): ONE offset shared by both hands, switched on and off by the contact phases:
    - the can stays put on the table until the giver has grasped it (giver offset = 0 before the lift);
    - the giver carries the can (object attachment: can = T_giver . X_R) to the new pass point;
    - the receiver's take pose is the recorded relative grasp applied to the can's NEW pose;
    - during dual contact both hands keep their recorded relation to the can (transfer coordination);
    - the receiver closes before the giver opens, with the recorded overlap (phase transitions);
    - IK within joint limits, joint speed / command jump contracts, arm-arm capsule clearance;
    - if a check fails the offset is shrunk (x0.7, <= 5 tries) instead of accepting a violating plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .recover import interp_q
from .robot import Robot, arm_clearance

MAX_SPEED = 2.5          # rad/s, OpenArmUmeow.send_action step cap (real-robot wrapper contract)
MAX_JUMP = 0.25          # rad, safety-hold command jump
MAX_TRACK = 0.5          # rad, safety-hold request-state gap
LIMIT_MARGIN = 0.02      # rad
MIN_CLEARANCE = 0.02     # m, arm-arm capsule surface distance (capsule radius 4.5 cm is approximate)
CAPTURE_TOL = 0.015      # m, receiver-to-can relative position deviation still counted as a capture
IK_TOL = (0.004, 2.0)    # m, deg


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


@dataclass
class Plan:
    name: str
    offsets: dict                 # side -> (T,3) world TCP offset applied
    q: dict                       # side -> (T,7) new joint trajectory (state timeline)
    ik_err: dict                  # side -> (T,2) pos m / rot deg
    can_holder: list              # per frame: "table" | "R" | "L"
    can: list                     # per frame: (4,4) can frame (state timeline) or None
    params: dict = field(default_factory=dict)
    checks: dict = field(default_factory=dict)
    action: np.ndarray | None = None


def phase_marks(robot: Robot, g: dict, ev: dict) -> dict:
    T = len(g["R"])
    pL = robot.tcp_traj(g, "L")[:, :3, 3]
    la = next((t for t in range(ev["right_close"], T) if np.linalg.norm(pL[t] - pL[0]) > 0.02), ev["right_close"])
    return {"right_close": ev["right_close"], "lift": ev["right_close"] + 12, "left_approach": la,
            "left_close": ev["left_close"], "right_open": ev["right_open"], "T": T}


def weights_contact(m: dict) -> dict:
    t = np.arange(m["T"])
    pass_t = m["left_close"] - 6
    wR = np.minimum(smoothstep((t - m["lift"]) / max(1, pass_t - m["lift"])),
                    1 - smoothstep((t - m["right_open"]) / 30.0))
    wL = smoothstep((t - m["left_approach"]) / max(1, m["left_close"] - 3 - m["left_approach"]))
    return {"R": wR, "L": wL}     # the receiver keeps the offset through the carry (it holds the moved can)


def weights_bump(m: dict, sigma=45.0) -> dict:
    t = np.arange(m["T"])
    c = 0.5 * (m["left_close"] + m["right_open"])
    w = np.exp(-0.5 * ((t - c) / sigma) ** 2)
    return {"R": w, "L": w}


def solve(robot: Robot, g: dict, offsets: dict) -> tuple[dict, dict]:
    Q, E = {}, {}
    for s in "RL":
        Tt = robot.tcp_traj(g, s)
        qs, es = [], []
        q = g[s][0]
        for t in range(len(Tt)):
            if np.linalg.norm(offsets[s][t]) < 1e-6:
                qs.append(g[s][t]); es.append((0.0, 0.0)); q = g[s][t]
                continue
            T = Tt[t].copy()
            T[:3, 3] += offsets[s][t]
            q0 = 0.5 * (q + g[s][t])   # warm start between the previous solution and the recorded posture
            q, pe, re = robot.ik(s, T, q0)
            qs.append(q); es.append((pe, re))
        Q[s], E[s] = np.array(qs), np.array(es)
    return Q, E


def can_track(robot: Robot, Q: dict, holder: list, X: dict, table_pose: np.ndarray):
    out = []
    for t, h in enumerate(holder):
        if h == "table":
            out.append(table_pose)
        else:
            robot.fk(Q["R"][t], Q["L"][t])
            out.append(robot.frame(robot.tcp[h]) @ X[h])
    return out


def checks(robot: Robot, g: dict, plan: Plan, m: dict, X: dict, table_pose, fps: float) -> dict:
    """The kinematic contact model, evaluated on any plan (G or K)."""
    T = m["T"]
    out = {}
    # joint limits / IK
    lim, pe, re = 0, 0.0, 0.0
    for s in "RL":
        lo, hi = robot.lo[s] + LIMIT_MARGIN - 1e-9, robot.hi[s] - LIMIT_MARGIN + 1e-9
        lim += int(((plan.q[s] < lo) | (plan.q[s] > hi)).any(1).sum())
        pe = max(pe, plan.ik_err[s][:, 0].max())
        re = max(re, plan.ik_err[s][:, 1].max())
    out["ik_max_pos_err_mm"], out["ik_max_rot_err_deg"] = 1000 * pe, re
    out["ik_ok"] = bool(pe <= IK_TOL[0] and re <= IK_TOL[1])
    out["joint_limit_violation_frames"] = lim
    sp = max(np.abs(np.diff(plan.q[s], axis=0)).max() for s in "RL") * fps
    out["max_joint_speed_rad_s"] = float(sp)
    sp0 = max(np.abs(np.diff(g[s], axis=0)).max() for s in "RL") * fps
    out["source_max_joint_speed_rad_s"] = float(sp0)
    out["speed_ok"] = bool(sp <= max(MAX_SPEED, sp0 + 1e-9))      # no NEW excess over the recorded demo
    # clearance
    cl = [arm_clearance(robot, plan.q["R"][t], plan.q["L"][t]) for t in range(0, T, 2)]
    cl0 = [arm_clearance(robot, g["R"][t], g["L"][t]) for t in range(0, T, 2)]
    out["min_arm_clearance_m"], out["source_min_arm_clearance_m"] = float(min(cl)), float(min(cl0))
    out["clearance_ok"] = bool(min(cl) >= min(MIN_CLEARANCE, min(cl0)))
    # object attachment / transfer: relative pose of the can to each hand that touches it, vs the recording
    def rel(Qd, side, t, can):
        robot.fk(Qd["R"][t], Qd["L"][t])
        return np.linalg.inv(robot.frame(robot.tcp[side])) @ can

    orig_holder = ["table" if t < m["right_close"] + 8 else ("R" if t < m["right_open"] else "L") for t in range(T)]
    can0 = can_track(robot, {"R": g["R"], "L": g["L"]}, orig_holder, X, table_pose)
    dev = {"R": [], "L": []}
    for side, (a, b) in (("R", (m["right_close"] + 8, m["right_open"])), ("L", (m["left_close"], T))):
        for t in range(a, b):
            if plan.can[t] is None:
                continue
            d = rel(plan.q, side, t, plan.can[t])[:3, 3] - rel(g, side, t, can0[t])[:3, 3]
            dev[side].append(np.linalg.norm(d))
    out["attach_dev_giver_max_m"] = float(max(dev["R"])) if dev["R"] else None
    out["attach_dev_receiver_max_m"] = float(max(dev["L"])) if dev["L"] else None
    out["receiver_capture_dev_m"] = float(dev["L"][0]) if dev["L"] else None
    out["attachment_ok"] = bool(dev["R"] and dev["L"] and max(dev["R"]) <= CAPTURE_TOL and max(dev["L"]) <= CAPTURE_TOL)
    # the can must not move on the table before the grasp, nor jump between frames
    P = np.array([c[:3, 3] for c in plan.can])
    jump = np.linalg.norm(np.diff(P, axis=0), axis=1)
    out["can_max_step_m"] = float(jump.max())
    P0 = np.array([c[:3, 3] for c in can0])
    out["source_can_max_step_m"] = float(np.linalg.norm(np.diff(P0, axis=0), axis=1).max())
    out["can_continuous"] = bool(jump.max() <= max(0.02, 1.5 * out["source_can_max_step_m"]))
    pre = slice(0, m["right_close"])
    out["can_moved_before_grasp_m"] = float(np.linalg.norm(P[pre] - P0[pre], axis=1).max())
    # gripper reach at the grasp: did the giver's TCP stay at the can when it closed?
    robot.fk(plan.q["R"][m["right_close"]], plan.q["L"][m["right_close"]])
    tcpR = robot.frame(robot.tcp["R"])[:3, 3]
    robot.fk(g["R"][m["right_close"]], g["L"][m["right_close"]])
    tcpR0 = robot.frame(robot.tcp["R"])[:3, 3]
    out["giver_grasp_offset_m"] = float(np.linalg.norm((tcpR - P[m["right_close"]]) - (tcpR0 - P0[m["right_close"]])))
    # phase order (commands unchanged in time): receiver closes before the giver opens
    out["phase_order_ok"] = bool(m["left_close"] < m["right_open"])
    out["dual_contact_frames"] = int(m["right_open"] - m["left_close"])
    out["contact_ok"] = bool(out["attachment_ok"] and out["can_continuous"] and out["can_moved_before_grasp_m"] < 0.005
                             and out["giver_grasp_offset_m"] <= CAPTURE_TOL and out["phase_order_ok"])
    out["all_ok"] = bool(out["contact_ok"] and out["ik_ok"] and out["joint_limit_violation_frames"] == 0
                         and out["speed_ok"] and out["clearance_ok"])
    return out


def actions(g_state: dict, g_action: dict, raw_action: np.ndarray, plan: Plan, lead: int) -> np.ndarray:
    """a_new[t] = a_rec[t] + (q_new - q_rec)[t + lead] on the arm joints; gripper commands unchanged.

    Keeps the recorded command-to-state relation (the demo's own tracking offset and lead)."""
    T = len(raw_action)
    idx = np.minimum(np.arange(T) + lead, T - 1)
    R = g_action["R"] + (plan.q["R"] - g_state["R"])[idx]
    L = g_action["L"] + (plan.q["L"] - g_state["L"])[idx]
    from .robot import NAMES
    return Robot.merge(R, L, raw_action[:, NAMES.index("RJ8.pos")], raw_action[:, NAMES.index("LJ8.pos")])


def action_checks(robot: Robot, act: np.ndarray, q: dict, lead: int) -> dict:
    a = Robot.split(act)
    T = len(act)
    gap = max(np.abs(a[s][: T - lead] - q[s][lead:]).max() for s in "RL")
    gap0 = max(np.abs(a[s] - q[s]).max() for s in "RL")
    jump = max(np.abs(np.diff(a[s], axis=0)).max() for s in "RL")
    lim = sum(int(((a[s] < robot.lo[s]) | (a[s] > robot.hi[s])).any(1).sum()) for s in "RL")
    return {"max_cmd_state_gap_rad": float(gap0), "max_cmd_vs_state_at_lead_rad": float(gap),
            "max_cmd_jump_rad": float(jump), "cmd_limit_violation_frames": lim,
            "contracts_ok": bool(gap0 <= MAX_TRACK and jump <= MAX_JUMP and lim == 0)}


def make_plan(robot, g, ev, X, table_pose, fps, kind: str, delta, seed=0, max_tries=5):
    m = phase_marks(robot, g, ev)
    T = m["T"]
    rng = np.random.default_rng(seed)
    delta = np.asarray(delta, float)
    tries = []
    scale = 1.0
    for k in range(max_tries if kind == "contact" else 1):
        if kind == "contact":
            w = weights_contact(m)
            offs = {s: w[s][:, None] * (scale * delta)[None] for s in "RL"}
            holder = ["table" if t < m["right_close"] + 8 else ("R" if t < m["right_open"] else "L") for t in range(T)]
        else:
            w = weights_bump(m)
            d2 = rng.normal(size=3)
            d2 = d2 / np.linalg.norm(d2) * np.linalg.norm(delta)
            offs = {"R": w["R"][:, None] * delta[None], "L": w["L"][:, None] * d2[None]}
            holder = ["table" if t < m["right_close"] + 8 else ("R" if t < m["right_open"] else "L") for t in range(T)]
        Q, E = solve(robot, g, offs)
        can = can_track(robot, Q, holder, X, table_pose)
        p = Plan(kind, offs, Q, E, holder, can, {"delta_R": offs["R"].max(0).tolist(), "scale": scale,
                                                  "delta": (scale * delta).tolist() if kind == "contact" else None})
        p.checks = checks(robot, g, p, m, X, table_pose, fps)
        tries.append({"scale": scale, "all_ok": p.checks["all_ok"]})
        if kind != "contact" or p.checks["all_ok"]:
            break
        scale *= 0.7
    p.params["tries"] = tries
    p.params["marks"] = m
    return p
