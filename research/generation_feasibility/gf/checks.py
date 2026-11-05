"""Offline dry-run checks of a plan (kinematics and declared constraints only; NO physics, NO contact).

Passing these checks means "the plan is kinematically executable and respects the declared constraints under the
plan's own can prediction". It does NOT mean the hand-over will succeed in simulation; that is what the pilot measures.
"""

from __future__ import annotations

import numpy as np

from . import config as C
from .descriptors import handover_descriptor
from .kin import Kin, arm_arm_clearance
from .plans import Plan, SourceInfo, plan_A, plan_phase


def solve_joints(plan: Plan, src: SourceInfo, kin: Kin, stride: int = 1):
    J = {}
    for s in "RL":
        Q, pe, re = kin.ik_traj(s, plan.target[s][::stride], src.ep.q[s][0])
        J[s] = {"q": Q, "pos_err": pe, "rot_err": re}
    plan.joint = J
    return J


def jaw_capture(X_can_in_grasp: np.ndarray) -> bool:
    x, y, z = X_can_in_grasp[:3, 3]
    jc = C.JAW_CAPTURE
    return bool(abs(y) <= jc["y"] and jc["z"][0] <= z <= jc["z"][1] and abs(x) <= jc["x"])


def grasp_in_can(plan: Plan, side: str, t: int):
    X = np.linalg.inv(plan.can_pred[t]) @ plan.target[side][t]
    p = X[:3, 3]
    return float(np.hypot(p[0], p[1])), float(p[2])


def check_plan(plan: Plan, src: SourceInfo, kin: Kin, stride: int = 1) -> dict:
    J = plan.joint
    if J is None or len(J["R"]["q"]) != len(plan.target["R"][::stride]):
        J = solve_joints(plan, src, kin, stride)
    m = plan.marks
    out = {}
    # -------- kinematic feasibility
    pe = max(J[s]["pos_err"].max() for s in "RL")
    re = max(J[s]["rot_err"].max() for s in "RL")
    out["ik_max_pos_err_mm"] = 1000 * pe
    out["ik_max_rot_err_deg"] = re
    out["ik_ok"] = bool(pe <= C.IK_POS_TOL and re <= C.IK_ROT_TOL_DEG)
    dq = {s: np.abs(np.diff(J[s]["q"], axis=0)).max(1) / stride for s in "RL"}
    out["max_joint_step_rad"] = float(max(dq[s].max() for s in "RL"))
    out["speed_violations"] = int(sum((dq[s] > C.STEP_CAP).sum() for s in "RL"))
    out["jump_violations"] = int(sum((dq[s] > C.MAX_TARGET_JUMP).sum() for s in "RL"))
    lim = 0
    for s in "RL":
        lo, hi = kin.lo[s] + C.JOINT_LIMIT_MARGIN, kin.hi[s] - C.JOINT_LIMIT_MARGIN
        lim += int(((J[s]["q"] <= lo + 1e-6) | (J[s]["q"] >= hi - 1e-6)).any(1).sum())
    out["steps_at_joint_limit"] = lim
    # -------- clearances on the IK joint trajectory
    dual = (m["L_close"] - 10, m["R_release_eff"] + 10)
    pick = (m["R_close"] - 30, m["R_sig"] + 15)
    arm_min, hand_min, table_min, can_bottom_min = np.inf, np.inf, np.inf, np.inf
    zc = plan.can_pred[:, 2, 3]
    up = np.flatnonzero((np.arange(len(zc)) >= m["R_close"]) & (zc >= C.CAN_REST_Z + C.LIFT_MIN))
    lifted = int(up[0]) if len(up) else len(zc)
    out["planned_lift_step"] = lifted
    n = len(J["R"]["q"])
    for i in range(n):
        t = i * stride
        caps = kin.capsules(J["R"]["q"][i], J["L"]["q"][i])
        a, h = arm_arm_clearance(caps)
        arm_min = min(arm_min, a)
        if not (dual[0] <= t <= dual[1]):
            hand_min = min(hand_min, h)
        if not (pick[0] <= t <= pick[1]):
            z = min(caps["R"][1][:, 2].min(), caps["L"][1][:, 2].min())
            table_min = min(table_min, z - C.PAD_TOP_Z)
        if t >= lifted:
            Tc = plan.can_pred[t]
            bottom = Tc[:3, 3][2] - abs(Tc[2, 2]) * C.CAN_HALF_HEIGHT
            can_bottom_min = min(can_bottom_min, bottom - C.PAD_TOP_Z)
    out["arm_clearance_min_m"] = float(arm_min)
    out["hand_clearance_min_m"] = float(hand_min)
    out["finger_table_clearance_min_m"] = float(table_min)
    out["can_table_clearance_min_m"] = float(can_bottom_min)
    out["clearance_ok"] = bool(arm_min >= C.ARM_CLEARANCE and hand_min >= C.HAND_CLEARANCE
                               and table_min >= C.TABLE_CLEARANCE and can_bottom_min >= C.TABLE_CLEARANCE)
    # -------- phase order and synchronisation (plan time)
    out["phase_order_ok"] = bool(m["R_close"] <= m["R_sig"] < m["L_close"] < m["R_release_eff"] <= m["end"])
    out["receiver_after_giver_arrival"] = bool(m["L_close"] >= m["giver_arrive"])
    # -------- grasp geometry at the planned take, under the plan's own can prediction
    rad, ax = grasp_in_can(plan, "L", m["L_close"])
    radR, axR = grasp_in_can(plan, "R", m["L_close"])
    out["receiver_radial_m"], out["receiver_axial_m"] = rad, ax
    out["giver_axial_m"] = axR
    out["receiver_in_grasp_cylinder"] = bool(rad < C.EEF_TO_CAN_RADIAL and abs(ax) < C.GRASP_AXIAL_TOL)
    out["receiver_on_can_body"] = bool(abs(ax) <= C.CAN_HALF_HEIGHT - 0.02)
    out["receiver_jaw_capture"] = jaw_capture(np.linalg.inv(plan.target["L"][m["L_close"]]) @ plan.can_pred[m["L_close"]])
    out["grasp_separation_m"] = float(abs(ax - axR))
    out["grasp_separation_ok"] = bool(abs(ax - axR) >= C.MIN_GRASP_SEPARATION)
    out["all_ok"] = bool(out["ik_ok"] and out["jump_violations"] == 0 and out["clearance_ok"] and out["phase_order_ok"]
                         and out["receiver_in_grasp_cylinder"] and out["receiver_jaw_capture"])
    out["all_ok_C"] = bool(out["all_ok"] and out["speed_violations"] == 0 and out["receiver_on_can_body"]
                           and out["grasp_separation_ok"] and out["receiver_after_giver_arrival"])
    return out


def plan_descriptor(plan: Plan) -> dict:
    m = plan.marks
    # the pass happens when both the receiver has closed and the giver has arrived
    t_pass = max(m["L_close"], m["giver_arrive"])
    return handover_descriptor(plan.target["R"], plan.target["L"], plan.can_pred, t_pass, m["giver_arrive"],
                               m["L_approach"], m["R_release_eff"], t_close=m["L_close"])


def source_descriptor(src: SourceInfo) -> dict:
    can = np.stack([src.can_at(t) for t in range(src.ep.T)])
    t_rel = max(src.t_rr, src.t_lc)
    return handover_descriptor(src.TR, src.TL, can, src.t_lc, src.t_ga, src.t_la, t_rel)


def retime(plan: Plan, src: SourceInfo, kin: Kin, passes: int = 2) -> int:
    """Insert interpolated targets (both arms at once, so their relative timing is kept) wherever a planned joint step
    exceeds the robot's step cap. Returns the number of inserted steps."""
    from .plans import interp_pose

    added = 0
    for _ in range(passes):
        J = solve_joints(plan, src, kin)
        step = np.maximum(*(np.abs(np.diff(J[s]["q"], axis=0)).max(1) for s in "RL"))
        bad = np.flatnonzero(step > C.STEP_CAP * 0.98)
        if not len(bad):
            break
        newT = {s: [plan.target[s][0]] for s in "RL"}
        newG = {s: [plan.grip[s][0]] for s in "RL"}
        newC = [plan.can_pred[0]]
        ins_at = []
        for t in range(1, plan.T):
            k = int(np.ceil(step[t - 1] / (C.STEP_CAP * 0.9))) - 1 if step[t - 1] > C.STEP_CAP * 0.98 else 0
            for j in range(k):
                f = (j + 1) / (k + 1)
                for s in "RL":
                    newT[s].append(interp_pose(plan.target[s][t - 1], plan.target[s][t], f))
                    newG[s].append(plan.grip[s][t - 1])
                newC.append(interp_pose(plan.can_pred[t - 1], plan.can_pred[t], f))
            if k:
                ins_at.append((t, k))
            for s in "RL":
                newT[s].append(plan.target[s][t])
                newG[s].append(plan.grip[s][t])
            newC.append(plan.can_pred[t])
        for key, v in plan.marks.items():
            plan.marks[key] = v + sum(k for t, k in ins_at if t <= v)
        plan.target = {s: np.stack(newT[s]) for s in "RL"}
        plan.grip = {s: np.array(newG[s]) for s in "RL"}
        plan.can_pred = np.stack(newC)
        if plan.runtime.get("receiver_reanchor"):
            plan.runtime["receiver_reanchor"].update(anchor_step=plan.marks["anchor"], pass_step=plan.marks["L_close"])
        added += sum(k for _, k in ins_at)
        plan.joint = None
    return added


def make_C(src: SourceInfo, can0, params, kin: Kin, idx=0, stride=1):
    """Condition C: try the sampled parameters, then shrink spatial parameters; reject if all tries fail.
    Joint-speed excess is repaired by retiming (both arms), not by shrinking."""
    tried = []
    for sc in C.C_SHRINK:
        p = plan_phase(src, can0, params, "C", idx, scale=sc)
        n_ins = retime(p, src, kin)
        if n_ins:
            p.params["retimed_steps"] = n_ins
        r = check_plan(p, src, kin, stride)
        tried.append({"scale": sc, "all_ok_C": r["all_ok_C"],
                      "failed": [k for k in ("ik_ok", "clearance_ok", "phase_order_ok", "receiver_in_grasp_cylinder", "receiver_jaw_capture",
                                             "receiver_on_can_body", "grasp_separation_ok",
                                             "receiver_after_giver_arrival") if not r[k]]
                      + (["speed"] if r["speed_violations"] else []) + (["jump"] if r["jump_violations"] else [])})
        if r["all_ok_C"]:
            p.notes.append(f"C accepted at scale {sc}; tries {tried}")
            return p, r, tried
    p.status = "rejected"
    p.notes.append(f"C rejected after {len(tried)} tries: {tried}")
    return p, r, tried


def attachment_sensitivity(plan: Plan, src: SourceInfo, rng, n=200, settle_drift=C.SETTLE_DRIFT, attach_spread=C.ATTACH_STD):
    """Model prediction, NOT physics: fraction of sampled giver attachments for which the actually-attached can lies
    in the receiver's jaw-capture region at the planned take (config.JAW_CAPTURE).

    Spreads from the local GT sources (BASELINE_GENERATOR_AUDIT.md §3.5): source-to-source attachment std up to
    1.5 / 0.3 / 0.4 cm (grasp-frame x/y/z), settle->take drift median 5-7 mm. Without re-anchoring (A, B) the whole
    attachment error reaches the take; with C's runtime re-anchor only the post-anchor drift does.
    """
    m = plan.marks
    t = m["L_close"]
    TRt = plan.target["R"][t]
    TLt = plan.target["L"][t]
    reanchor = plan.runtime.get("receiver_reanchor") is not None
    ok = 0
    for _ in range(n):
        drift = rng.normal(0, settle_drift / np.sqrt(3), 3)
        if reanchor:
            err = drift
        else:
            err = rng.normal(0, attach_spread) + drift
        X = src.X_attach.copy()
        X[:3, 3] += err
        can = TRt @ X
        if reanchor:   # the runtime moves the receiver by the predicted-can correction measured at the anchor
            Xa = src.X_attach.copy()
            Xa[:3, 3] += err - drift
            corr = (TRt @ Xa) @ np.linalg.inv(TRt @ src.X_attach)
            TL_use = corr @ TLt
        else:
            TL_use = TLt
        ok += jaw_capture(np.linalg.inv(TL_use) @ can)
    return ok / n
