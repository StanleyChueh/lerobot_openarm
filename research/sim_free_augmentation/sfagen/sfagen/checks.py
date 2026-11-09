"""Replay checks: identical rules and thresholds for every task (thresholds from config `checks:`; defaults below).
All are computed on the commanded action stream and the planned state; a candidate is published only if all pass.

limits       URDF joint limits, extended to the source demo's own recorded range (the demo was executed)
speed/accel  per-joint |da|/dt <= speed_ratio x and |d2a|/dt2 <= accel_ratio x the max over all source demos
clearance    coal minimum distance between arms and arm-vs-base (URDF collision meshes) >= min(source over the same
             frames, clearance_floor) - 1 mm
hands        distance between TCPs of different arms >= source over the same frames - hand_tol (multi-arm only)
table        lowest arm point >= lowest point over the source demos - table_tol
objects      moved / held object proxies: lowest point >= source - 5 mm; resting position in view, on the table
             (plate appearance like the original resting place), >= 1 cm from other objects, and the robot never closer
             to it before the final approach than the recorded robot to the recorded object (hull points within 1 cm)
untouched    start state == recorded start; arms not moved and grippers bit-identical; identical after the rejoin
IK           FK error of the realised TCP <= 1 mm, 0.5 deg on contact-relevant frames (full offset applied: at the object /
             inside the interaction window), <= 5 mm, 2 deg on the blending frames"""
from __future__ import annotations
import cv2, numpy as np
import pinocchio as pin

DEFAULTS = {"speed_ratio": 1.25, "accel_ratio": 1.5, "clearance_floor_m": 0.02, "hand_tol_m": 0.005, "table_tol_m": 0.01,
            "ik_pos_mm": 1.0, "ik_rot_deg": 0.5, "ik_blend_pos_mm": 5.0, "ik_blend_rot_deg": 2.0, "start_tol_rad": 0.01, "plate_appearance_tol": 30, "object_gap_m": 0.01}


class Checker:
    def __init__(self, ctx):
        self.ctx, self.robot = ctx, ctx.robot; self.cfg = {**DEFAULTS, **ctx.cfg.get("checks", {})}
        r = self.robot; cm = r.col; self.cm = cm
        side = r.col_arm; skip = set(ctx.cfg.get("checks", {}).get("skip_collision_links", []))
        for i in range(len(side)):
            for j in range(i + 1, len(side)):
                a, b = side[i], side[j]
                if a == b or {a, b} == {"B"}:
                    continue
                n = cm.geometryObjects[i].name + cm.geometryObjects[j].name
                if any(s in n for s in skip):
                    continue
                cm.addCollisionPair(pin.CollisionPair(i, j))
        self.cd = pin.GeometryData(cm)
        self.arm_cols = [ (i, j) for i in range(len(r.arms)) for j in range(i + 1, len(r.arms))]
        self.env = None

    def kin(self, A):
        fps = self.ctx.src.fps; idx = [i for s in self.robot.arms for i in self.robot.sidx[s]]
        return np.abs(np.diff(A[:, idx], axis=0)).max(0) * fps, np.abs(np.diff(A[:, idx], 2, axis=0)).max(0) * fps ** 2

    def min_dist(self, S, step=3):
        sp = self.robot.split(S); out = []
        for t in range(0, len(S), step):
            q = self.robot.q_full({s: sp[s][t] for s in self.robot.arms}, {s: sp["g" + s][t] for s in self.robot.arms if "g" + s in sp})
            pin.computeDistances(self.robot.m, self.robot.d, self.cm, self.cd, q)
            out.append(min(r.min_distance for r in self.cd.distanceResults) if len(self.cd.distanceResults) else np.inf)
        return np.array(out)

    def lowest(self, S, step=3):
        sp = self.robot.split(S); out = []; skip = tuple(self.ctx.cfg.get("checks", {}).get("table_skip_links", []))
        for t in range(0, len(S), step):
            self.robot.fk({s: sp[s][t] for s in self.robot.arms}, {s: sp["g" + s][t] for s in self.robot.arms if "g" + s in sp})
            out.append(min(P[:, 2].min() for n, s, P in self.robot.link_points() if s != "B" and not any(k in n for k in skip)))
        return np.array(out)

    def envelope(self, episodes):
        V, Acc, Z = [], [], []
        for e in episodes:
            S, A = self.ctx.src.arrays(e); v, a = self.kin(A); V.append(v); Acc.append(a); Z.append(float(self.lowest(S).min()))
        self.env = {"speed": np.max(V, 0).tolist(), "accel": np.max(Acc, 0).tolist(), "min_z_m": float(min(Z)), "episodes": list(episodes)}
        return self.env

    def check(self, rec, c):
        r, cfg, env = self.robot, self.cfg, self.env
        S, A, S0, A0 = c["state"], c["action"], rec.state, rec.action
        idx = [i for s in r.arms for i in r.sidx[s]]
        lo = np.concatenate([r.lo[s] for s in r.arms]); hi = np.concatenate([r.hi[s] for s in r.arms])
        lim = int(sum(((X[:, idx] < np.minimum(lo, Y[:, idx].min(0)) - 1e-6) | (X[:, idx] > np.maximum(hi, Y[:, idx].max(0)) + 1e-6)).sum()
                      for X, Y in ((S, S0), (A, A0))))
        v, a = self.kin(A); ed = c["edited"]
        dmin = self.min_dist(S[ed]) if ed.any() else np.array([np.inf]); dsrc = self.min_dist(S0[ed]) if ed.any() else np.array([np.inf])
        hold = c["w"] >= 0.999                                    # contact-relevant frames: full offset (at the object / in the interaction)
        ck = {"ik_pos_err_mm_max": float(1000 * c["ik_err"][hold, 0].max()) if hold.any() else 0.0, "ik_rot_err_deg_max": float(c["ik_err"][hold, 1].max()) if hold.any() else 0.0,
              "ik_pos_err_mm_max_blend": float(1000 * c["ik_err"][:, 0].max()), "ik_rot_err_deg_max_blend": float(c["ik_err"][:, 1].max()),
              "limit_violations": lim, "speed_ratio_to_real_max": float((v / np.maximum(env["speed"], 1e-9)).max()),
              "accel_ratio_to_real_max": float((a / np.maximum(env["accel"], 1e-9)).max()),
              "min_mesh_distance_mm": float(1000 * dmin.min()), "source_min_mesh_distance_same_frames_mm": float(1000 * dsrc.min()),
              "lowest_arm_z_m": float(self.lowest(S[ed]).min()) if ed.any() else np.inf, "real_lowest_arm_z_m": env["min_z_m"],
              "start_state_err_rad": float(np.abs(S[0] - S0[0]).max())}
        if len(r.arms) > 1:
            P = {s: r.tcp_from_state(S, s)[:, :3, 3] for s in r.arms}; P0 = {s: r.tcp_from_state(S0, s)[:, :3, 3] for s in r.arms}
            pairs = [(a_, b_) for i, a_ in enumerate(r.arms) for b_ in r.arms[i + 1:]]
            hn = np.min([np.linalg.norm(P[a_] - P[b_], axis=1) for a_, b_ in pairs], 0); hs = np.min([np.linalg.norm(P0[a_] - P0[b_], axis=1) for a_, b_ in pairs], 0)
            ck["min_hand_distance_mm"] = float(1000 * hn[ed].min()) if ed.any() else np.inf
            ck["source_min_hand_distance_same_frames_mm"] = float(1000 * hs[ed].min()) if ed.any() else np.inf
        untouched = [s for s in r.arms if s not in c["moved_arms"]]
        keep = [i for s in untouched for i in r.sidx[s]] + [r.grip[s]["state_index"] for s in r.arms if r.grip[s]]
        ck["untouched_bit_identical"] = bool(np.array_equal(S[:, keep], S0[:, keep]) and np.array_equal(A[:, keep], A0[:, keep]))
        tr = c["t_rejoin"]; ck["identical_after_rejoin"] = bool(np.array_equal(S[tr + 1:], S0[tr + 1:]))
        # objects
        src_tracks = [np.stack([rec.obj_pose(k, t) for t in range(rec.T)]) for k in range(len(rec.objects))]
        lowz = []
        for k, o in enumerate(rec.objects):
            Pn = c["objects"][k]; zc = min((Pn[t][:3, :3] @ o["points"][::20].T).T[:, 2].min() + Pn[t][2, 3] for t in np.flatnonzero(ed)) if ed.any() else np.inf
            zs = min((src_tracks[k][t][:3, :3] @ o["points"][::20].T).T[:, 2].min() + src_tracks[k][t][2, 3] for t in np.flatnonzero(ed)) if ed.any() else np.inf
            lowz.append(zc - zs)
        ck["object_lowest_point_vs_source_mm"] = float(1000 * min(lowz)) if lowz else 0.0
        ok_obj = ck["object_lowest_point_vs_source_mm"] >= -5.0
        if c["family"] == "object_position":
            ck["placement"] = self.placement(rec, c); ok_obj &= ck["placement"]["ok"]
        ck["pass"] = bool(ck["ik_pos_err_mm_max"] <= cfg["ik_pos_mm"] and ck["ik_rot_err_deg_max"] <= cfg["ik_rot_deg"]
                          and ck["ik_pos_err_mm_max_blend"] <= cfg["ik_blend_pos_mm"] and ck["ik_rot_err_deg_max_blend"] <= cfg["ik_blend_rot_deg"] and lim == 0
                          and ck["speed_ratio_to_real_max"] <= cfg["speed_ratio"] and ck["accel_ratio_to_real_max"] <= cfg["accel_ratio"]
                          and ck["min_mesh_distance_mm"] >= min(ck["source_min_mesh_distance_same_frames_mm"], 1000 * cfg["clearance_floor_m"]) - 1.0
                          and ck.get("min_hand_distance_mm", np.inf) >= ck.get("source_min_hand_distance_same_frames_mm", 0) - 1000 * cfg["hand_tol_m"]
                          and ck["lowest_arm_z_m"] >= env["min_z_m"] - cfg["table_tol_m"] and ck["start_state_err_rad"] <= cfg["start_tol_rad"]
                          and ck["untouched_bit_identical"] and ck["identical_after_rejoin"] and ok_obj)
        return ck

    def placement(self, rec, c):
        k = [o["name"] for o in rec.objects].index(c["object"]); o = rec.objects[k]; T0 = c["objects"][k][0]
        P = (T0[:3, :3] @ o["points"].T).T + T0[:3, 3]; uv, _ = rec.cam.project(P)
        inside = bool((uv[:, 0] > 10).all() and (uv[:, 0] < rec.W - 10).all() and (uv[:, 1] > 10).all() and (uv[:, 1] < rec.H - 10).all())
        hsv = cv2.cvtColor(rec.plate, cv2.COLOR_RGB2HSV).astype(int)
        new = rec.obj_sil(k, T0); orig = rec.obj_sil(k, o["table"])
        ring = (cv2.dilate(orig.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0) & ~(cv2.dilate(orig.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
        ring &= ~np.isin(rec.robot_sil(0, sides=True), rec.arms)
        dv = np.abs(np.median(hsv[new], 0) - np.median(hsv[ring], 0)) if ring.any() and new.any() else np.array([999, 999, 999])
        on_table = bool(dv[1] <= self.cfg["plate_appearance_tol"] and dv[2] <= self.cfg["plate_appearance_tol"])
        gap = np.inf
        for j, oo in enumerate(rec.objects):
            if j == k:
                continue
            Q = (oo["table"][:3, :3] @ oo["points"][::5].T).T + oo["table"][:3, 3]
            gap = min(gap, float(np.min(np.linalg.norm(P[::5, None, :2] - Q[None, :, :2], axis=2))))
        def near(S, Tb, t_end):
            sp = rec.robot.split(S); out = []
            Tinv = np.linalg.inv(Tb); lo, hi = o["points"].min(0) - 0.01, o["points"].max(0) + 0.01
            for t in range(0, t_end, 3):
                rec.robot.fk({s: sp[s][t] for s in rec.arms}, {s: sp["g" + s][t] for s in rec.arms if "g" + s in sp}); h = 0
                for n, s, Q in rec.robot.link_points():
                    if s == "B":
                        continue
                    ql = Q @ Tinv[:3, :3].T + Tinv[:3, 3]; h += int(np.all((ql > lo) & (ql < hi), axis=1).sum())
                out.append(h)
            return np.array(out)
        te = c["marks"]["t_grasp"] - 15
        extra = int(np.maximum(near(c["state"], T0, te) - near(rec.state, o["table"], te), 0).sum())
        return {"in_view": inside, "on_table": on_table, "plate_hsv_diff": dv.tolist(), "gap_to_other_objects_m": gap,
                "robot_points_near_object_extra_vs_source": extra,
                "ok": inside and on_table and gap >= self.cfg["object_gap_m"] and extra == 0}
