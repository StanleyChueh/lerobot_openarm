"""Independent kinematic/geometric verifier for candidate hand-overs (no physics; offline geometry only).

Inputs are ONLY what a candidate exports - 16-D state, 16-D action, the object (can) trajectory and the time map
tau - plus the SOURCE reconstruction as reference. It does not read the planner's deltas, regrasp, holder labels or
attachments, and it uses quantities the planner never optimised:

* contact intervals are inferred from the exported gripper STATE (openness < CLOSED), not from the plan's holders;
* 6D attachment: for every inferred closed interval of a hand, the drift of the object's pose relative to that hand,
  inv(rel(t0)) rel(t), must match the source's drift at the corresponding source time tau(t) (excess translation,
  axis tilt and spin about the can axis). This covers grasp-onset slip, carry attachment, transfer closure across
  the release and object-rotation corruptions in one test, and cancels the source reconstruction's own artefacts;
* unsupported object motion: the object may only move while some gripper is closed;
* release order from the exported commands and states; dual-closed overlap >= 1 frame;
* grasp feasibility at each grasp onset from regrasp-invariant geometry (TCP-to-can-axis distance, closing direction
  vs can axis, axial position within the can body), relative to the source grasp;
* joint limits, joint speed / acceleration (no new excess over the source), command jump;
* mesh-sample arm-arm clearance (all links incl. hands and fingers), link penetration into the can cylinder and
  robot/can height above the table plane, relative to the source.
Spin about the can's symmetry axis is not observable in the images; its check is reported as model-internal.
Passing says the candidate is kinematically consistent with the source's contact structure; it says nothing about
forces, friction, grasp stability or real-robot executability.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from . import se3
from .recon import CLOSED, Recon, rb, rc

TOL = {"attach_mm": 10.0, "attach_tilt_deg": 5.0, "attach_spin_deg": 5.0, "object_step_mm": 2.0,
       "object_step_deg": 1.0, "grasp_axis_dist_mm": 10.0, "grasp_angle_deg": 10.0, "grasp_axial_margin_m": 0.01,
       "speed_ratio": 1.05, "speed_abs": 2.5, "accel_ratio": 1.5, "cmd_jump_rad": 0.25, "clearance_m": 0.02,
       "clearance_slack_m": 0.005, "penetration_excess_mm": 10.0, "table_slack_m": 0.01, "collision_stride": 3}


def openness(state16):
    g = rb.Robot.split(state16)
    return {"R": g["gR"], "L": g["gL"]}


def intervals(mask) -> list:
    out, start = [], None
    for t, v in enumerate(mask):
        if v and start is None:
            start = t
        if not v and start is not None:
            out.append((start, t))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


class Kin:
    """FK cache of one trajectory: TCP poses per side and link sample points on demand."""

    def __init__(self, robot, state16):
        self.robot, self.x = robot, state16
        self.g = rb.Robot.split(state16)
        self.T = len(state16)
        self.tcp = {s: [] for s in "RL"}
        for t in range(self.T):
            robot.fk(self.g["R"][t], self.g["L"][t])
            for s in "RL":
                self.tcp[s].append(robot.frame(robot.tcp[s]))

    def points(self, t):
        self.robot.fk(self.g["R"][t], self.g["L"][t], self.g["gR"][t], self.g["gL"][t])
        return self.robot.link_points("sample")


def source_at(rec: Recon, robot, t_src: float):
    qR, qL = rc.interp_q(rec.g["R"], t_src), rc.interp_q(rec.g["L"], t_src)
    robot.fk(qR, qL)
    return {s: robot.frame(robot.tcp[s]) for s in "RL"}, rec.source_can(robot, t_src)


def excess(drift_src, drift_gen):
    E = se3.inv(drift_src) @ drift_gen
    return 1000 * np.linalg.norm(E[:3, 3]), se3.axis_tilt_deg(drift_src[:3, 2], drift_gen[:3, 2]), \
        abs(se3.spin_deg(drift_src[:3, :3], drift_gen[:3, :3]))


def grasp_geometry(Th, can):
    a = can[:3, 2]
    d = Th[:3, 3] - can[:3, 3]
    axial = float(d @ a)
    radial = float(np.linalg.norm(d - axial * a))
    close_dir = Th[:3, 1]                      # finger closing direction (hand y)
    ang = float(np.degrees(np.arccos(np.clip(abs(close_dir @ a), 0, 1))))
    return radial, ang, axial


def penetration(points, can):
    """max depth (m) of any point inside the can cylinder (lid origin, z = lid -> bottom)."""
    Q = (points - can[:3, 3]) @ can[:3, :3]
    r = np.hypot(Q[:, 0], Q[:, 1])
    inside = (Q[:, 2] > 0) & (Q[:, 2] < rc.CAN_HEIGHT) & (r < rc.CAN_RADIUS)
    return float((rc.CAN_RADIUS - r[inside]).max()) if inside.any() else 0.0


def collision_stats(kin: Kin, can, frames):
    out = {"arm_arm_min_m": [], "pen_R_m": [], "pen_L_m": [], "min_z_robot_m": [], "can_bottom_z_m": []}
    for t in frames:
        pts = kin.points(t)
        P = {s: np.concatenate([p for n, sd, p in pts if sd == s]) for s in "RL"}
        d, _ = cKDTree(P["R"]).query(P["L"], k=1)
        out["arm_arm_min_m"].append(float(d.min()))
        for s in "RL":
            out[f"pen_{s}_m"].append(penetration(P[s], can[t]))
        out["min_z_robot_m"].append(float(min(P["R"][:, 2].min(), P["L"][:, 2].min())))
        out["can_bottom_z_m"].append(float(can[t][2, 3] + rc.CAN_HEIGHT * can[t][2, 2]))
    return {k: np.array(v) for k, v in out.items()}


class Verifier:
    def __init__(self, rec: Recon, robot, tol=TOL):
        self.rec, self.robot, self.tol = rec, robot, dict(tol)
        T = rec.T
        self.src_can = np.stack([rec.source_can(robot, t) for t in range(T)])
        self.src = Kin(robot, rec.state)
        g = self.src.g
        fps = rec.fps
        self.src_speed = max(np.abs(np.diff(g[s], axis=0)).max() for s in "RL") * fps
        self.src_accel = max(np.abs(np.diff(g[s], n=2, axis=0)).max() for s in "RL") * fps ** 2
        fr = list(range(0, T, self.tol["collision_stride"]))
        self.src_col = collision_stats(self.src, self.src_can, fr)
        self.src_col_frames = np.array(fr)

    # ------------------------------------------------------------------------------------------------------------
    def __call__(self, cand: dict) -> dict:
        rec, robot, tol = self.rec, self.robot, self.tol
        x, act, can, tau = cand["state"], cand["action"], cand["can"], cand["tau"]
        valid = cand.get("valid", np.isfinite(act).all(1))
        T = len(x)
        fps = rec.fps
        kin = Kin(robot, x)
        g = kin.g
        res = {}

        def put(name, ok, value, thr, note=None):
            res[name] = {"ok": bool(ok), "value": value, "threshold": thr}
            if note:
                res[name]["note"] = note

        # -- joint limits, speed, acceleration, command jump
        lim = sum(int(((g[s] < robot.lo[s] - 1e-6) | (g[s] > robot.hi[s] + 1e-6)).any(1).sum()) for s in "RL")
        a = rb.Robot.split(np.where(np.isfinite(act), act, 0.0))
        alim = sum(int((((a[s] < robot.lo[s] - 1e-6) | (a[s] > robot.hi[s] + 1e-6)).any(1) & valid).sum()) for s in "RL")
        put("joint_limits", lim == 0 and alim == 0, {"state_frames": lim, "action_frames": alim}, 0)
        sp = max(np.abs(np.diff(g[s], axis=0)).max() for s in "RL") * fps
        put("joint_speed", sp <= max(tol["speed_abs"], tol["speed_ratio"] * self.src_speed), float(sp),
            float(max(tol["speed_abs"], tol["speed_ratio"] * self.src_speed)))
        ac = max(np.abs(np.diff(g[s], n=2, axis=0)).max() for s in "RL") * fps ** 2
        put("joint_accel", ac <= tol["accel_ratio"] * self.src_accel, float(ac), float(tol["accel_ratio"] * self.src_accel))
        av = act[valid]
        jump = max(np.abs(np.diff(rb.Robot.split(av)[s], axis=0)).max() for s in "RL")
        put("action_jump", jump <= tol["cmd_jump_rad"], float(jump), tol["cmd_jump_rad"])

        # -- inferred contact (from the exported gripper state)
        op = {"R": g["gR"], "L": g["gL"]}
        closed = {s: op[s] < CLOSED for s in "RL"}
        ivs = {s: intervals(closed[s]) for s in "RL"}

        # -- 6D attachment / transfer closure, relative to the source drift at tau(t)
        worst = {"mm": 0.0, "tilt": 0.0, "spin": 0.0}
        where = {}
        for s in "RL":
            for (t0, t1) in ivs[s]:
                if t1 - t0 < 2:
                    continue
                Ts0, can0 = source_at(rec, robot, tau[t0])
                rel_s0 = se3.inv(Ts0[s]) @ can0
                rel_g0 = se3.inv(kin.tcp[s][t0]) @ can[t0]
                for t in range(t0 + 1, t1):
                    Ts, cs = source_at(rec, robot, tau[t])
                    d_src = se3.inv(rel_s0) @ (se3.inv(Ts[s]) @ cs)
                    d_gen = se3.inv(rel_g0) @ (se3.inv(kin.tcp[s][t]) @ can[t])
                    mm, tl, spn = excess(d_src, d_gen)
                    for k, v in (("mm", mm), ("tilt", tl), ("spin", spn)):
                        if v > worst[k]:
                            worst[k], where[k] = v, (s, int(t))
        put("attach6d_translation", worst["mm"] <= tol["attach_mm"], worst["mm"], tol["attach_mm"])
        put("attach6d_tilt", worst["tilt"] <= tol["attach_tilt_deg"], worst["tilt"], tol["attach_tilt_deg"])
        put("attach6d_spin", worst["spin"] <= tol["attach_spin_deg"], worst["spin"], tol["attach_spin_deg"],
            "rotation about the can symmetry axis: model-internal, not observable in the images")
        res["attach6d_worst_at"] = where

        # -- object may only move while some gripper is closed (and as much as the source moves then)
        any_closed = closed["R"] | closed["L"]
        bad = []
        for t in range(1, T):
            step = 1000 * np.linalg.norm(can[t][:3, 3] - can[t - 1][:3, 3])
            rstep = se3.angle_deg(can[t - 1][:3, :3], can[t][:3, :3])
            if (step > tol["object_step_mm"] or rstep > tol["object_step_deg"]) and not (any_closed[t] or any_closed[t - 1]):
                bad.append(int(t))
        put("unsupported_object_motion", len(bad) == 0, {"frames": len(bad), "first": bad[:5]}, 0)

        # -- release order / dual contact (commands and states)
        ca = rb.Robot.split(np.where(np.isfinite(act), act, rec.action))
        cmd_closed = {"R": ca["gR"] < 0.5, "L": ca["gL"] < 0.5}
        rc_ = int(np.argmax(cmd_closed["R"])) if cmd_closed["R"].any() else None
        lc_ = next((t for t in range(rc_ or 0, T) if cmd_closed["L"][t]), None) if rc_ is not None else None
        ro_ = next((t for t in range(rc_, T) if not cmd_closed["R"][t]), None) if rc_ is not None else None
        dual = int((closed["R"] & closed["L"]).sum())
        st_L = next((t for t in range(T) if closed["L"][t]), None)
        st_R_open = next((t for t in range(st_L or 0, T) if not closed["R"][t]), None) if st_L is not None else None
        order_ok = (rc_ is not None and lc_ is not None and ro_ is not None and lc_ < ro_ and dual >= 1
                    and st_L is not None and st_R_open is not None and st_L < st_R_open)
        put("release_order", order_ok, {"cmd_right_close": rc_, "cmd_left_close": lc_, "cmd_right_open": ro_,
                                        "state_left_closed": st_L, "state_right_opened": st_R_open,
                                        "dual_closed_frames": dual}, "receiver closes before giver opens")

        # -- grasp feasibility at each grasp onset (regrasp-invariant geometry, vs the source grasp)
        for s in "RL":
            first = [iv for iv in ivs[s] if iv[1] - iv[0] >= 2]
            if not first:
                put(f"grasp_feasibility_{s}", False, "no closed interval", None)
                continue
            t0 = first[0][0]
            Ts, cs = source_at(rec, robot, tau[t0])
            r_s, a_s, x_s = grasp_geometry(Ts[s], cs)
            r_g, a_g, x_g = grasp_geometry(kin.tcp[s][t0], can[t0])
            m_ = tol["grasp_axial_margin_m"]
            ok = (abs(r_g - r_s) * 1000 <= tol["grasp_axis_dist_mm"] and abs(a_g - a_s) <= tol["grasp_angle_deg"]
                  and -m_ <= x_g <= rc.CAN_HEIGHT + m_)
            put(f"grasp_feasibility_{s}", ok, {"frame": int(t0), "axis_dist_mm": 1000 * r_g, "source_axis_dist_mm": 1000 * r_s,
                                               "closing_vs_axis_deg": a_g, "source_closing_vs_axis_deg": a_s,
                                               "axial_m": x_g, "source_axial_m": x_s},
                {"axis_dist_mm": tol["grasp_axis_dist_mm"], "angle_deg": tol["grasp_angle_deg"]})

        # -- collisions / penetration / table, relative to the source at the nearest source frame
        fr = list(range(0, T, tol["collision_stride"]))
        col = collision_stats(kin, can, fr)
        thr_cl = min(tol["clearance_m"], float(self.src_col["arm_arm_min_m"].min()) - tol["clearance_slack_m"])
        put("arm_arm_clearance", col["arm_arm_min_m"].min() >= thr_cl, float(col["arm_arm_min_m"].min()), thr_cl)
        pen_ex = 0.0
        for i, t in enumerate(fr):
            j = int(np.argmin(np.abs(self.src_col_frames - tau[t])))
            for s in "RL":
                pen_ex = max(pen_ex, 1000 * (col[f"pen_{s}_m"][i] - self.src_col[f"pen_{s}_m"][j]))
        put("can_penetration", pen_ex <= tol["penetration_excess_mm"], pen_ex, tol["penetration_excess_mm"],
            "excess over the source's own (calibration-limited) finger/can overlap")
        zt = rec.z_table - tol["table_slack_m"]
        zr = min(zt, float(self.src_col["min_z_robot_m"].min()) - tol["table_slack_m"])
        zc = min(zt, float(self.src_col["can_bottom_z_m"].min()) - tol["table_slack_m"])
        put("table_clearance", col["min_z_robot_m"].min() >= zr and col["can_bottom_z_m"].min() >= zc,
            {"robot_min_z": float(col["min_z_robot_m"].min()), "can_bottom_min_z": float(col["can_bottom_z_m"].min())},
            {"robot": zr, "can": zc})

        failed = [k for k, v in res.items() if isinstance(v, dict) and "ok" in v and not v["ok"]]
        return {"ok": not failed, "failed": failed, "checks": res,
                "inferred_contact": {s: [list(map(int, iv)) for iv in ivs[s]] for s in "RL"}}


def drift_series(V: "Verifier", cand: dict) -> dict:
    """per-frame excess 6D drift (vs source) for each hand's inferred closed intervals (NaN elsewhere) - for plots."""
    rec, robot = V.rec, V.robot
    kin = Kin(robot, cand["state"])
    T = len(cand["state"])
    out = {s: {k: np.full(T, np.nan) for k in ("mm", "tilt", "spin")} for s in "RL"}
    op = {"R": kin.g["gR"], "L": kin.g["gL"]}
    for s in "RL":
        for (t0, t1) in intervals(op[s] < CLOSED):
            if t1 - t0 < 2:
                continue
            Ts0, can0 = source_at(rec, robot, cand["tau"][t0])
            rel_s0 = se3.inv(Ts0[s]) @ can0
            rel_g0 = se3.inv(kin.tcp[s][t0]) @ cand["can"][t0]
            for t in range(t0, t1):
                Ts, cs = source_at(rec, robot, cand["tau"][t])
                mm, tl, sp = excess(se3.inv(rel_s0) @ (se3.inv(Ts[s]) @ cs), se3.inv(rel_g0) @ (se3.inv(kin.tcp[s][t]) @ cand["can"][t]))
                out[s]["mm"][t], out[s]["tilt"][t], out[s]["spin"][t] = mm, tl, sp
    return out
