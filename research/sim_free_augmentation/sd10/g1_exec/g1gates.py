"""SD10-G1 G1-A gates (SD10_G1_METRICS.md items 1-9), absolute thresholds identical for every source and condition.

All geometry is recomputed from the EXPORTED arrays (float32 states/actions, can poses) plus the full planned q needed
for action reconstruction. Several checks are construction-aligned (attachment/holds); they are engineering
consistency screens, not force, contact, collision-free execution or dynamics validation.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

import g1lib as G

A = G.ACC
CAN_R, CAN_H = 0.0375, 0.23
Z_TABLE = G.TGT["candidate_table_height_m"]
DISTAL = ("link5", "link6", "link7", "hand", "left_finger", "right_finger")
FINGERS = ("left_finger", "right_finger")


def rot_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def axis_deg(a, b):
    a, b = np.asarray(a) / np.linalg.norm(a), np.asarray(b) / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))


def cylinder_min_z(C):
    a, z0 = C[:3, 2], C[2, 3]
    return float(min(z0, z0 + CAN_H * a[2]) - CAN_R * np.sqrt(max(0.0, 1 - a[2] ** 2)))


class Geometry:
    """Full-mesh convex hulls (all hull vertices) + the robot's seed-0 surface samples (<=400/link)."""

    def __init__(self, robot):
        self.robot = robot
        self.eq, self.sph, self.pts, self.part = {}, {}, {}, {}
        for i, (name, side, H, S) in enumerate(robot.links):
            if side not in "RL":
                continue
            part = name.split("_", 1)[1]
            self.part[i] = (side, part)
            self.pts[i] = np.concatenate([H, S])
            if part in DISTAL:
                self.eq[i] = ConvexHull(H).equations
                c = H.mean(0)
                self.sph[i] = (c, float(np.linalg.norm(H - c, axis=1).max()))

    def world(self, i):
        M = self.robot.vd.oMg[i]
        return self.pts[i] @ M.rotation.T + M.translation

    def frame(self, qR, qL, gR, gL, can):
        self.robot.fk(qR, qL, gR, gL)
        W = {i: self.world(i) for i in self.pts}
        out = {"distal_min_z": min(W[i][:, 2].min() for i in W if self.part[i][1] in DISTAL)}
        best = 0.0
        for a, b in (("R", "L"), ("L", "R")):
            P = np.concatenate([W[i] for i in W if self.part[i] == (a, self.part[i][1]) and self.part[i][1] in DISTAL])
            for j in self.eq:
                if self.part[j][0] != b:
                    continue
                M = self.robot.vd.oMg[j]
                loc = (P - M.translation) @ M.rotation
                c, rad = self.sph[j]
                loc = loc[np.einsum("ij,ij->i", loc - c, loc - c) < rad * rad]
                if not len(loc):
                    continue
                d = loc @ self.eq[j][:, :3].T + self.eq[j][:, 3]
                ins = (d <= 0).all(1)
                if ins.any():
                    best = max(best, float((-d[ins].max(1)).max()))
        out["arm_arm_pen"] = best
        # can cylinder (lid origin, +z lid->bottom) vs links
        def cyl_depth(P):
            loc = (P - can[:3, 3]) @ can[:3, :3]
            r = np.hypot(loc[:, 0], loc[:, 1])
            ins = (loc[:, 2] >= 0) & (loc[:, 2] <= CAN_H) & (r <= CAN_R)
            if not ins.any():
                return 0.0
            return float(np.minimum.reduce([CAN_R - r[ins], loc[ins, 2], CAN_H - loc[ins, 2]]).max())
        out["can_nonfinger_pen"] = max(cyl_depth(W[i]) for i in W if self.part[i][1] not in FINGERS)
        out["can_finger_overlap"] = max(cyl_depth(W[i]) for i in W if self.part[i][1] in FINGERS)
        return out


def first(mask, start):
    idx = np.flatnonzero(mask[start:])
    return int(idx[0] + start) if len(idx) else None


def evaluate(rec, robot, geom, spec, p, ex, target):
    """Returns {"Y_A": bool, "gates": {...}, "failures": [...] }."""
    from sfa import robot as rb
    T, rows = rec.T, ex["native_rows"]
    n = len(rows)
    a0, a1 = int(rows[0]), int(rows[-1]) + 1                      # [3, T-6)
    g = {}
    S, Ac, can = ex["state"].astype(float), ex["action"].astype(float), ex["can"]
    ev = rec.ev
    rc_, lc_, ro_ = ev["right_close"], ev["left_close"], ev["right_open"]
    sp = rb.Robot.split(S)
    # ---------------- 1. arrays / support / provenance
    fin = all(np.isfinite(x).all() for x in (S, Ac, can, ex["timestamp_s"]))
    shapes = S.shape == (T - 9, 16) and Ac.shape == (T - 9, 16) and can.shape == (T - 9, 4, 4) and n == T - 9
    ortho = max(float(np.abs(C[:3, :3].T @ C[:3, :3] - np.eye(3)).max()) for C in can)
    det = max(abs(float(np.linalg.det(C[:3, :3])) - 1) for C in can)
    rowmap = np.array_equal(rows, np.arange(3, T - 6)) and np.allclose(ex["timestamp_s"], np.arange(n) / 30.0)
    trans_in = all(a0 <= t < a1 for t in (rc_, lc_, ro_))
    released = first(sp["gR"] >= 0.7, max(0, ro_ - a0))                 # planned giver state reopening (output index)
    released_native = None if released is None else released + a0
    post = None if released is None else n - released
    g["1_arrays_support"] = {"ok": bool(fin and shapes and ortho <= 1e-6 and det <= 1e-6 and rowmap and trans_in
                                        and post is not None and post >= A["post_release_min_frames"]),
                             "finite": fin, "shapes": shapes, "max_orthogonality_err": ortho, "max_det_err": det,
                             "row_map_ok": rowmap, "transitions_rc_lc_ro": [rc_, lc_, ro_], "transitions_inside": trans_in,
                             "donor_state_release_native": released_native, "post_release_frames": post}
    # ---------------- 2. requested initial candidate
    tgt = np.asarray(target["candidate_lid_xyz_robot_base_m"])
    pos_err = float(np.linalg.norm(can[0][:3, 3] - tgt))
    ax_err = axis_deg(can[0][:3, 2], [0, 0, -1.0])
    from sfa.robot import Camera
    uv, _ = Camera().project(can[0][:3, 3][None])
    px = float(np.linalg.norm(uv[0] - np.asarray(target["requested_lid_uv_native_px"])))
    g["2_initial_candidate"] = {"ok": bool(pos_err <= A["target_lid_position_error_m"] and ax_err <= A["initial_axis_error_deg"]),
                                "lid_position_err_m": pos_err, "axis_err_deg": ax_err,
                                "projected_lid_err_px_selfconsistency": px}
    # ---------------- 3. IK residual vs requested pre-IK targets (all planned frames) and contact waypoints
    req = G.requested_targets(spec)
    pe, re = np.zeros((T, 2)), np.zeros((T, 2))
    for t in range(T):
        if a0 <= t < a1:                                  # exported rows: FK from the EXPORTED float32 states
            robot.fk(sp["R"][t - a0], sp["L"][t - a0])
        else:                                             # rows outside export (needed for action lead): planned q
            robot.fk(p["q"]["R"][t], p["q"]["L"][t])
        for k, s in enumerate("RL"):
            Ta = robot.frame(robot.tcp[s])
            pe[t, k] = np.linalg.norm(Ta[:3, 3] - req[s][t][:3, 3])
            re[t, k] = rot_deg(Ta[:3, :3], req[s][t][:3, :3])
    exp_pe, exp_re = pe[a0:a1], re[a0:a1]
    cw = [x for x in p["targets"] if x["kind"] == "contact"]
    cw_mm = max([x["err_mm"] for x in cw] or [0.0])
    cw_deg = max([x["err_deg"] for x in cw] or [0.0])
    g["3_ik_residual"] = {"ok": bool(pe.max() <= A["ik_max_position_error_m"] and re.max() <= A["ik_max_rotation_error_deg"]
                                     and cw_mm <= 1000 * A["contact_waypoint_position_m"] and cw_deg <= A["contact_waypoint_rotation_deg"]),
                          "max_pos_m_all_frames": float(pe.max()), "max_rot_deg_all_frames": float(re.max()),
                          "p95_pos_m": float(np.percentile(pe, 95)), "median_pos_m": float(np.median(pe)),
                          "max_pos_m_export_rows": float(exp_pe.max()), "max_rot_deg_export_rows": float(exp_re.max()),
                          "frames_over_pos": int((pe.max(1) > A["ik_max_position_error_m"]).sum()),
                          "frames_over_rot": int((re.max(1) > A["ik_max_rotation_error_deg"]).sum()),
                          "contact_waypoint_max_mm": cw_mm, "contact_waypoint_max_deg": cw_deg, "waypoints": p["targets"]}
    # ---------------- 4. limits / speed / accel / command jumps (exported rows)
    lim, lim_cmd, vel, acc, jump, at_clamp = {}, {}, {}, {}, {}, 0
    ga = rb.Robot.split(Ac)
    for s in "RL":
        lo, hi = robot.lo[s], robot.hi[s]
        lim[s] = float(max((lo - sp[s]).max(), (sp[s] - hi).max()))
        lim_cmd[s] = float(max((lo - ga[s]).max(), (ga[s] - hi).max()))
        vel[s] = float((np.abs(np.diff(sp[s], axis=0)) * 30 / robot.vel[s]).max())
        acc[s] = float((np.abs(np.diff(sp[s], 2, axis=0)) * 900).max())
        jump[s] = float(np.abs(np.diff(ga[s], axis=0)).max())
        q = p["q"][s]
        at_clamp += int(((np.abs(q - (lo + 0.02)) < 1e-9) | (np.abs(q - (hi - 0.02)) < 1e-9)).any(1).sum())
    tol = A["joint_position_slack_rad"]
    g["4_kinematic_limits"] = {"ok": bool(max(lim.values()) <= tol and max(lim_cmd.values()) <= tol and max(vel.values()) <= 1.0
                                          and max(acc.values()) <= A["joint_accel_max_rad_s2"] and max(jump.values()) <= A["command_step_max_rad"]),
                               "state_limit_excess_rad": lim, "command_limit_excess_rad": lim_cmd,
                               "max_speed_over_urdf_ratio": vel, "max_accel_rad_s2": acc, "max_command_step_rad": jump,
                               "planned_frames_at_ik_clamp": at_clamp}
    # ---------------- 5. table clearance and penetration (every exported row)
    per = [geom.frame(sp["R"][t], sp["L"][t], sp["gR"][t], sp["gL"][t], can[t]) for t in range(n)]
    cmin = min(cylinder_min_z(C) for C in can)
    dmin = min(x["distal_min_z"] for x in per)
    aa = max(x["arm_arm_pen"] for x in per)
    cn = max(x["can_nonfinger_pen"] for x in per)
    zfloor = Z_TABLE - A["can_table_penetration_max_m"]
    g["5_clearance_penetration"] = {"ok": bool(cmin >= zfloor and dmin >= zfloor and aa <= A["distal_arm_arm_penetration_max_m"]
                                               and cn <= A["can_nonfinger_link_penetration_max_m"]),
                                    "can_lowest_rim_z": cmin, "distal_min_z": dmin, "z_floor": zfloor,
                                    "distal_arm_arm_pen_m": aa, "can_nonfinger_pen_m": cn,
                                    "finger_can_overlap_m_grasp_proxy": max(x["can_finger_overlap"] for x in per),
                                    "rows_checked": n, "note": "sampled hulls can miss collisions; no collision-free claim"}
    # ---------------- 6. steady holds vs fitted attachments
    def hold(side, lo_, hi_):
        tr, ro = [], []
        for t in range(max(lo_, a0), min(hi_, a1)):
            i = t - a0
            robot.fk(sp["R"][i], sp["L"][i])
            Rel = np.linalg.inv(robot.frame(robot.tcp[side])) @ can[i]
            tr.append(np.linalg.norm(Rel[:3, 3] - rec.X[side][:3, 3]))
            ro.append(rot_deg(Rel[:3, :3], rec.X[side][:3, :3]))
        return (max(tr) if tr else np.nan), (max(ro) if ro else np.nan), len(tr)
    gt, gr, gn = hold("R", rc_ + 12, lc_ - 3)
    rt, rr, rn = hold("L", ro_ + 12, T - 6)
    g["6_steady_holds"] = {"ok": bool(gn > 0 and rn > 0 and gt <= A["steady_attachment_error_m"] and gr <= A["steady_attachment_rotation_deg"]
                                      and rt <= A["steady_attachment_error_m"] and rr <= A["steady_attachment_rotation_deg"]),
                           "giver_max_m": gt, "giver_max_deg": gr, "giver_rows": gn, "receiver_max_m": rt,
                           "receiver_max_deg": rr, "receiver_rows": rn, "note": "construction-aligned; not contact evidence"}
    # ---------------- 7. receiver capture
    cap = first(sp["gL"] < 0.7, max(0, lc_ - a0))
    if cap is None:
        g["7_receiver_capture"] = {"ok": False, "reason": "no measured-state receiver closure before retained end"}
    else:
        robot.fk(sp["R"][cap], sp["L"][cap])
        C_L = robot.frame(robot.tcp["L"]) @ rec.X["L"]
        et, er = float(np.linalg.norm(C_L[:3, 3] - can[cap][:3, 3])), rot_deg(C_L[:3, :3], can[cap][:3, :3])
        both = (sp["gR"] < 0.7) & (sp["gL"] < 0.7)
        overlap = int(both[cap:].sum()) if released is None else (released - cap)
        g["7_receiver_capture"] = {"ok": bool(et <= A["receiver_capture_error_m"] and er <= A["receiver_capture_rotation_deg"]),
                                   "closure_native_row": cap + a0, "discrepancy_m": et, "discrepancy_deg": er,
                                   "dual_hold_overlap_frames_state": overlap,
                                   "note": "state-threshold proxy; not measured simultaneous contact"}
    # ---------------- 8. object continuity and holder switches
    st = [float(np.linalg.norm(can[i + 1][:3, 3] - can[i][:3, 3])) for i in range(n - 1)]
    sr = [rot_deg(can[i][:3, :3], can[i + 1][:3, :3]) for i in range(n - 1)]
    sw = {}
    for name, t in (("table_to_giver", rc_ + 8), ("giver_to_receiver", ro_)):
        i = t - a0
        sw[name] = {"native_row": t, "step_m": st[i - 1], "step_deg": sr[i - 1]}
    g["8_object_continuity"] = {"ok": bool(max(st) <= A["can_step_max_m"] and max(sr) <= A["can_step_max_deg"]
                                           and all(v["step_m"] <= A["holder_switch_step_m"] and v["step_deg"] <= A["holder_switch_step_deg"]
                                                   for v in sw.values())),
                                "max_step_m": max(st), "max_step_deg": max(sr), "switches": sw}
    # ---------------- 9. label reconstruction (independent of planner bookkeeping)
    L = rec.lead
    dev = 0.0
    for s in "RL":
        cols = [rb.NAMES.index(f"{s}J{i}.pos") for i in range(1, 8)]
        exp = p["q"][s][rows + L] + (rec.action[rows][:, cols] - rec.state[rows + L][:, cols])
        dev = max(dev, float(np.abs(Ac[:, cols] - exp).max()))
    sdev = max(float(np.abs(sp[s] - p["q"][s][rows]).max()) for s in "RL")      # exported state == planned q
    gcols = [rb.NAMES.index("RJ8.pos"), rb.NAMES.index("LJ8.pos")]
    gdev = float(np.abs(Ac[:, gcols] - rec.action[rows][:, gcols].astype(np.float32)).max())
    def trans(x, thr):
        b = x < thr
        return [int(t) for t in range(1, len(b)) if b[t] != b[t - 1]]
    gen_t = {s: [t + a0 for t in trans(ga["g" + s], 0.5)] for s in "RL"}
    src_g = rb.Robot.split(rec.action)
    src_t = {s: [t for t in trans(src_g["g" + s], 0.5) if a0 <= t < a1] for s in "RL"}
    st_ok = (hashlib_ok := (G.arr_sha(rec.state.astype(np.float32)) == G.CFG["source_metadata"][str(rec.e)]["state_sha256"]
                            and G.arr_sha(rec.action.astype(np.float32)) == G.CFG["source_metadata"][str(rec.e)]["action_sha256"]))
    g["9_labels"] = {"ok": bool(dev <= A["label_construction_abs_tolerance_rad"] and sdev <= A["label_construction_abs_tolerance_rad"]
                                and gdev == 0.0 and gen_t == src_t
                                and gen_t["R"] == [rc_, ro_] and gen_t["L"] == [lc_] and st_ok),
                     "max_arm_label_dev_rad": dev, "exported_state_vs_plan_dev_rad": sdev, "gripper_cmd_dev": gdev, "gripper_transitions": gen_t,
                     "source_transitions": src_t, "source_lead": L, "input_hash_ok": hashlib_ok}
    fails = [k for k, v in g.items() if not v["ok"]]
    return {"Y_A": not fails, "failed_gates": fails, "gates": g}
