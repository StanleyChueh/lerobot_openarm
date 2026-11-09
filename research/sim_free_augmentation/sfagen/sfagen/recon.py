"""Per-episode reconstruction, identical for every task (no manual labels, no SAM2, no CAD models of objects):

1. video/state lag   photometric: the lag that best re-renders real frames from nearby real frames (after masks exist)
2. robot             z-buffered URDF silhouettes per arm (masks), mesh depth (renderer)
3. plate (pass 1)    temporal median over pixels free of the dilated robot silhouette
4. objects           each configured object is segmented in frame 0 as what DISAPPEARS when it is picked up (frame 0 vs the
                     median of robot-free pixels after the grasp), not robot, connected to the grasping hand's projected
                     TCP at the grasp. Known-size cylinders: pose + table height by render-and-compare constrained by the
                     grasp kinematics (centre along the approach axis, lying axis perpendicular to the finger-closing
                     axis). Unknown-size objects: oriented box from the segmentation (PCA extents).
5. holders           object on the table until its grasp (+3 frames), attached to the grasping hand
                     (X = T_hand^-1 T_object), handed over to another closed hand when the holder opens while that hand
                     is closed, otherwise left where it was released
6. plate (pass 2)    temporal median over pixels free of robot AND objects
7. robot masks      SAM2.1 video segmentation prompted automatically from the projected URDF (segment.py)
8. object masks      plate-difference foreground inside the dilated proxy silhouette, minus robot"""
from __future__ import annotations
import json
from pathlib import Path
import cv2, numpy as np, torch
from . import mesh


def interp_rows(X, t):
    T = len(X); t = np.clip(t, 0, T - 1); i = np.floor(t).astype(int); j = np.minimum(i + 1, T - 1); f = (t - i)[..., None]
    return X[i] * (1 - f) + X[j] * f


def cylinder_points(r, L, spacing=0.0015, rng=None):
    rng = rng or np.random.default_rng(1)
    n_cap = max(30, int(np.pi * r * r / spacing ** 2)); n_side = max(60, int(2 * np.pi * r * L / spacing ** 2))
    out = []
    for z0 in (0.0, L):
        rr = r * np.sqrt(rng.random(n_cap)); th = 2 * np.pi * rng.random(n_cap)
        out.append(np.stack([rr * np.cos(th), rr * np.sin(th), np.full(n_cap, z0)], 1))
    th = 2 * np.pi * rng.random(n_side); out.append(np.stack([r * np.cos(th), r * np.sin(th), L * rng.random(n_side)], 1))
    return np.concatenate(out)


def box_points(ext, spacing=0.0015, rng=None):
    rng = rng or np.random.default_rng(3); ex, ey, h = ext; pts = []
    for (a, b, fa, fv, ua, ub) in ((ex, ey, 2, h, 0, 1), (ex, ey, 2, 0.0, 0, 1), (ex, h, 1, ey / 2, 0, 2),
                                   (ex, h, 1, -ey / 2, 0, 2), (ey, h, 0, ex / 2, 1, 2), (ey, h, 0, -ex / 2, 1, 2)):
        n = max(50, int(a * b / spacing ** 2)); P = np.zeros((n, 3))
        P[:, ua] = (rng.random(n) - (0.5 if ua < 2 else 0.0)) * a; P[:, ub] = (rng.random(n) - (0.5 if ub < 2 else 0.0)) * b
        P[:, fa] = fv; pts.append(P)
    return np.concatenate(pts)


class Recon:
    def __init__(self, ctx, e, device="cuda:0", verbose=False):
        self.ctx, self.e, self.dev = ctx, e, device
        self.robot, self.cam, cfg = ctx.robot, ctx.cam, ctx.cfg
        ep = ctx.src.load(e, cams=[ctx.src.body]); self.ep = ep
        self.F = ep.frames[ctx.src.body]; self.T = len(self.F); self.H, self.W = self.F.shape[1:3]
        self.state, self.action = ep.state, ep.action
        self.sp = self.robot.split(self.state); self.ev = self.robot.gripper_events(self.action)
        self.arms = self.robot.arms
        self.G = mesh.Geometry(self.robot, device=device, spacing=0.003)
        rc = cfg.get("reconstruction", {})
        self.fg_thr = rc.get("foreground_threshold", 30)
        cache = ctx.cache_dir / f"recon_ep{e}.npz"
        self.objects = []
        if cache.exists():
            z = np.load(cache, allow_pickle=True); self.lag = float(z["lag"]); self.plate = z["plate"]
            self.objects = json.loads(str(z["objects"]))
            for o in self.objects:
                o["table"] = np.array(o["table"]); o["X"] = [np.array(x) if x is not None else None for x in o["X"]]
            self.info = json.loads(str(z["info"])); self.z_table = self.info["table"]["z"]; self._video_states(); self._finish_objects(); self._masks()
            return
        self.info = {}
        self.lag = float(rc.get("lag_default", 0.0)); self._video_states()
        self.plate = self._plate(objects=False)
        if rc.get("lag_scan", True):
            base = self.lag
            sc = self._lag_scan(np.arange(*rc.get("lag_scan_range", [-4.0, 1.51, 0.25])))
            best = max(sc, key=sc.get)
            self.lag = best if sc[best] > sc.get(round(base, 3), -1) + 0.01 else base   # default kept unless clearly better
            self._video_states()
            if self.lag != base:
                self.plate = self._plate(objects=False)
            self.info["lag_scan"] = sc
        self.info["lag"] = self.lag
        self.z_table = self._fit_table(cfg["objects"])
        for name, oc in cfg["objects"].items():
            self.objects.append(self._segment_object(name, oc))
        self._finish_objects()
        self.plate = self._plate(objects=True)
        self._masks()
        if rc.get("lag_method", "photometric") == "photometric":
            L = self._photometric_lag(rc.get("lag_candidates", [-3.0, 0.51, 0.25]))
            if L != self.lag:
                self.lag = L; self._video_states(); self.plate = self._plate(objects=True); self._masks()
        np.savez_compressed(cache, lag=self.lag, plate=self.plate, info=json.dumps(self.info),
                            objects=json.dumps([{k: (v.tolist() if isinstance(v, np.ndarray) else ([x.tolist() if x is not None else None for x in v] if k == "X" else v))
                                                 for k, v in o.items() if k not in ("points", "holder")} for o in self.objects]))

    # ------------------------------------------------------------------ robot at video time
    def _video_states(self):
        ts = np.arange(self.T) + self.lag
        self.gv = {s: interp_rows(self.sp[s], ts) for s in self.arms}
        self.grip_v = {s: interp_rows(self.sp["g" + s][:, None], ts)[:, 0] for s in self.arms if "g" + s in self.sp}

    def q_at(self, t_state):
        return {s: interp_rows(self.sp[s], np.array([t_state]))[0] for s in self.arms}

    def tcp(self, side, t_state):
        self.robot.fk(self.q_at(t_state)); return self.robot.frame(self.robot.tcp[side])

    def robot_sil(self, j, sides=False):
        q = {s: self.gv[s][j] for s in self.arms}; g = {s: float(self.grip_v[s][j]) for s in self.grip_v}
        _, lab, _ = mesh.splat(self.G.world(self.G.placements(q, g)), self.G.gid, self.cam, dev=self.dev)
        side = np.array(self.G.side + ["X"])[lab.cpu().numpy()]
        return side if sides else np.isin(side, self.arms)

    def _fg(self, j, plate, thr=None):
        fg = np.abs(self.F[j].astype(np.int16) - plate.astype(np.int16)).max(2) > (thr or self.fg_thr)
        return cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0

    # ------------------------------------------------------------------ plate
    def _plate(self, objects, dil=15):
        occ = np.zeros((self.T, self.H, self.W), bool)
        for j in range(self.T):
            m = cv2.dilate(self.robot_sil(j).astype(np.uint8), np.ones((dil, dil), np.uint8)) > 0
            if objects:
                for k in range(len(self.objects)):
                    m |= self.obj_sil(k, self.obj_pose(k, j + self.lag), dil=31)
            occ[j] = m
        plate = np.zeros((self.H, self.W, 3), np.uint8)
        F = torch.tensor(self.F, device=self.dev); O = torch.tensor(occ, device=self.dev)
        for c in range(3):
            x = F[..., c].half().masked_fill(O, float("nan"))
            med = torch.nanmedian(x.view(self.T, -1), dim=0).values.view(self.H, self.W)
            plate[..., c] = torch.nan_to_num(med, nan=0).clamp(0, 255).byte().cpu().numpy()
        never = occ.all(0); self.info["plate_never_free_frac" + ("_2" if objects else "_1")] = float(never.mean())
        del F, O; torch.cuda.empty_cache()
        return cv2.inpaint(plate, never.astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA)

    def _lag_scan(self, lags):
        js = np.linspace(40, self.T - 40, 30).astype(int); fg = [self._fg(j, self.plate, 25) for j in js]; out = {}
        for L in lags:
            self.lag = float(L); self._video_states(); s = []
            for j, f in zip(js, fg):
                r = self.robot_sil(j); roi = cv2.dilate(r.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
                ff = f & roi; s.append((ff & r).sum() / max(1, (ff | r).sum()))
            out[round(float(L), 3)] = float(np.mean(s))
        return out

    def _photometric_lag(self, rng, k=6, n_pairs=16):
        """video/state lag = the value that best re-renders real frames from nearby real frames (mean PSNR inside the
        camera-independent foreground of both frames) over `n_pairs` motion pairs (j, j+k), j on a grid offset from the
        benchmark grid so that estimation and evaluation pairs never coincide."""
        from .render import Renderer
        med = torch.tensor(self.F[::3], device=self.dev).float().median(0).values.cpu().numpy()
        fgm = lambda j: cv2.morphologyEx((np.abs(self.F[j].astype(np.int16) - med.astype(np.int16)).max(2) > 30).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0  # noqa: E731
        qd = np.abs(np.gradient(np.concatenate([self.sp[s] for s in self.arms], 1), axis=0)).max(1)
        cand = np.arange(17, self.T - k - 10, 7); cand = cand[np.argsort(-qd[cand])][:n_pairs]
        R = Renderer(self, self.dev, {**self.ctx.cfg.get("render", {}), "part_fill": False}); base = self.lag; scores = {}
        regions = {j: cv2.dilate((fgm(j) | fgm(j + k)).astype(np.uint8), np.ones((11, 11), np.uint8)) > 0 for j in cand}
        for L in np.arange(*rng):
            self.lag = float(L); self._video_states(); R._cache.clear(); ps = []
            for j in cand:
                t = j + k; q = {s: self.gv[s][t] for s in self.arms}; g = {s: float(self.grip_v[s][t]) for s in self.grip_v}
                im, _ = R.render(int(j), q, g, [self.obj_pose(kk, t + self.lag) for kk in range(len(self.objects))], exclude=(t - 5, t + 5))
                ps.append(10 * np.log10(255 ** 2 / max(float(((im.astype(float) - self.F[t]) ** 2)[regions[j]].mean()), 1e-6)))
            scores[round(float(L), 3)] = float(np.mean(ps))
        self.lag = base; self._video_states(); del R; torch.cuda.empty_cache()
        best = max(scores, key=scores.get); self.info["lag_photometric"] = scores; self.info["lag"] = best
        return float(best)

    # ------------------------------------------------------------------ objects
    def _fit_table(self, objs):
        """table plane height: 1-D search that maximises the IoU of a known-size cylinder's projected silhouette (placed at
        the grasp point) with the frame-0 foreground; candidates from the lowest arm point down 15 cm. Without such an
        object: lowest arm point over the source demos - 5 mm (config-independent fallback)."""
        z_hi = float(self.ctx.table_height)
        cyl = [(n, o) for n, o in objs.items() if o["proxy"]["type"] == "cylinder"]
        if not cyl:
            self.info["table"] = {"z": z_hi, "rule": "lowest arm point - 5 mm"}; return z_hi
        n, oc = cyl[0]; p = oc["proxy"]; arm = oc["grasped_by"]; tg = self.ev[arm]["closes"][oc.get("grasp_index", 0)]
        TG = self.tcp(arm, tg); r = cv2.dilate(self.robot_sil(0).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        fg = self._fg(0, self.plate) & ~r; pts = cylinder_points(p["radius"], p["length"], 0.003)
        uv0, _ = self.cam.project(TG[:3, 3][None]); roi = np.zeros(fg.shape, np.uint8); cv2.circle(roi, (int(uv0[0, 0]), int(uv0[0, 1])), 160, 1, -1)
        best, scores = None, {}
        for z in np.arange(z_hi - 0.15, z_hi + 1e-9, 0.0025):
            T = np.eye(4)
            if p.get("axis", "vertical") == "horizontal":
                ax = TG[:3, 0].copy(); ax[2] = 0; ax /= np.linalg.norm(ax) + 1e-9
                T[:3, 0], T[:3, 2] = [0, 0, 1.0], ax; T[:3, 1] = np.cross(T[:3, 2], T[:3, 0]); T[:3, 3] = [TG[0, 3], TG[1, 3], z + p["radius"]] - ax * p["length"] / 2
            else:
                T[:3, 3] = [TG[0, 3], TG[1, 3], z]
            P = (T[:3, :3] @ pts.T).T + T[:3, 3]; uv, zc = self.cam.project(P)
            ok = (zc > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < self.W) & (uv[:, 1] >= 0) & (uv[:, 1] < self.H)
            m = np.zeros(fg.shape, np.uint8); m[uv[ok, 1].astype(int), uv[ok, 0].astype(int)] = 1
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0
            f = fg & (roi > 0); scores[round(float(z), 4)] = float((f & m).sum() / max(1, (f | m).sum()))
        zb = max(scores, key=scores.get)
        self.info["table"] = {"z": zb, "rule": f"silhouette fit of '{n}' in frame 0", "iou": scores[zb], "upper_bound_lowest_arm": z_hi}
        return zb

    def _segment_object(self, name, oc):
        arm = oc["grasped_by"]; tg = self.ev[arm]["closes"][oc.get("grasp_index", 0)]
        TG = self.tcp(arm, tg); uv, _ = self.cam.project(TG[:3, 3][None]); u0, v0 = uv[0]
        z_tab = float(self.z_table)
        r = cv2.dilate(self.robot_sil(0).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        # the object = what disappears when it is picked up: frame 0 vs the median of robot-free pixels AFTER the grasp
        ref, valid = self._after_ref(tg + 20)
        fg = (np.abs(self.F[0].astype(np.int16) - ref.astype(np.int16)).max(2) > self.fg_thr) & valid
        fg = (cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0) & ~r
        for o in self.objects:
            fg &= ~self.obj_sil(self.objects.index(o), o["table"], dil=9)
        n, lab, st, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8))
        rn, rf = oc.get("seed_radius_px", 45), oc.get("max_radius_px", 130)
        near = np.zeros(fg.shape, np.uint8); cv2.circle(near, (int(u0), int(v0)), rn, 1, -1)
        far = np.zeros(fg.shape, np.uint8); cv2.circle(far, (int(u0), int(v0)), rf, 1, -1)
        keep = [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= 40 and (near[lab == k] > 0).any()]
        m = np.isin(lab, keep) & (far > 0)
        p = oc["proxy"]
        if p["type"] == "cylinder":
            h_obj = p["length"] if p.get("axis", "vertical") == "vertical" else 2 * p["radius"]
        else:
            h_obj = float(np.clip(2 * (TG[2, 3] - z_tab), p.get("min_height", 0.02), p.get("max_height", 0.08))) if p.get("height") is None else p["height"]
        info = {"name": name, "seg_px": int(m.sum()), "tcp_at_grasp": TG[:3, 3].tolist(), "grasp_frame": int(tg)}
        if p["type"] == "cylinder":
            # known-size object: render-and-compare pose fit constrained by the grasp kinematics (see _fit_cylinder)
            T, iou, z_fit, s_fit = self._fit_cylinder(TG, arm, tg, p, fg, int(u0), int(v0))
            z_tab = z_fit; self.z_table = z_fit; c = T[:2, 3] if p.get("axis", "vertical") == "vertical" else (T[:3, 3] + T[:3, 2] * p["length"] / 2)[:2]
            info.update({"source": "render-and-compare (grasp-constrained)", "silhouette_iou": iou, "table_z": z_fit, "offset_along_approach_m": s_fit})
            self.info["table"] = {"z": z_fit, "rule": f"joint pose + table fit of '{name}'", "iou": iou}
            self.info.setdefault("objects", {})[name] = {**info, "proxy": "cylinder", "dims": [p["radius"], p["length"]], "centre_xy": [float(c[0]), float(c[1])],
                                                         "centre_vs_tcp_xy_mm": float(1000 * np.linalg.norm(c - TG[:2, 3]))}
            return {"name": name, "arm": arm, "grasp_frame": int(tg), "table": T, "dims": [p["radius"], p["length"]], "proxy": "cylinder", "X": [],
                    "points": cylinder_points(p["radius"], p["length"])}
        elif m.sum() >= oc.get("min_pixels", 150):
            ys, xs = np.nonzero(m); P = self.cam.backproject_plane(np.stack([xs, ys], 1).astype(float), z_tab + 0.75 * h_obj)
            c = P[:, :2].mean(0); U = np.linalg.svd(P[:, :2] - c)[2]
            proj = (P[:, :2] - c) @ U.T; lo, hi = np.percentile(proj, 3, 0), np.percentile(proj, 97, 0)
            c = c + ((lo + hi) / 2) @ U; ext = hi - lo; info["source"] = "segmentation"
        else:                                    # not visible in frame 0: TCP position at the grasp, TCP x-axis as long axis
            c = TG[:2, 3].copy(); U = np.eye(2); ax = TG[:2, 0]; U[0] = ax / (np.linalg.norm(ax) + 1e-9); U[1] = [-U[0, 1], U[0, 0]]
            ext = np.array([p.get("length", 0.05), 2 * p.get("radius", 0.02)]); info["source"] = "tcp_fallback"
        T = np.eye(4)
        if p["type"] == "cylinder" and p.get("axis", "vertical") == "horizontal":
            zax = np.array([U[0, 0], U[0, 1], 0.0]); xax = np.array([0, 0, 1.0]); yax = np.cross(zax, xax)
            T[:3, 0], T[:3, 1], T[:3, 2] = xax, yax, zax
            T[:3, 3] = [c[0], c[1], z_tab + p["radius"]] - zax * p["length"] / 2
            pts = cylinder_points(p["radius"], p["length"]); dims = [p["radius"], p["length"]]
        elif p["type"] == "cylinder":
            T[:3, 3] = [c[0], c[1], z_tab]; pts = cylinder_points(p["radius"], p["length"]); dims = [p["radius"], p["length"]]
        else:
            T[:3, 0] = [U[0, 0], U[0, 1], 0]; T[:3, 1] = [U[1, 0], U[1, 1], 0]; T[:3, 2] = np.cross(T[:3, 0], T[:3, 1])
            if T[2, 2] < 0:
                T[:3, 1] *= -1; T[:3, 2] = np.cross(T[:3, 0], T[:3, 1])
            T[:3, 3] = [c[0], c[1], z_tab]; dims = [float(ext[0]), float(ext[1]), h_obj]; pts = box_points(dims)
        info.update({"proxy": p["type"], "dims": dims, "centre_xy": [float(c[0]), float(c[1])],
                     "centre_vs_tcp_xy_mm": float(1000 * np.linalg.norm(c - TG[:2, 3]))})
        self.info.setdefault("objects", {})[name] = info
        return {"name": name, "arm": arm, "grasp_frame": int(tg), "table": T, "dims": dims, "proxy": p["type"], "X": [], "points": pts}

    def _after_ref(self, t_from, dil=15, step=2):
        """per-pixel median of frames t >= t_from where the pixel is free of the (dilated) robot; valid = observed."""
        js = np.arange(min(t_from, self.T - 1), self.T, step)
        occ = np.stack([cv2.dilate(self.robot_sil(j).astype(np.uint8), np.ones((dil, dil), np.uint8)) > 0 for j in js])
        F = torch.tensor(self.F[js], device=self.dev); O = torch.tensor(occ, device=self.dev); out = np.zeros((self.H, self.W, 3), np.uint8)
        for c in range(3):
            x = F[..., c].half().masked_fill(O, float("nan"))
            out[..., c] = torch.nan_to_num(torch.nanmedian(x.view(len(js), -1), dim=0).values.view(self.H, self.W), nan=0).clamp(0, 255).byte().cpu().numpy()
        valid = ~occ.all(0); del F, O; torch.cuda.empty_cache()
        return out, valid

    def _fit_cylinder(self, TG, arm, tg, p, fg, u0, v0):
        """Pose of a known-size cylinder resting on the table, from the grasp + frame 0:
        centre = TCP + s * a + l * b (a: horizontal component of the gripper approach axis, b: horizontal closing axis,
        s in [-8, 8] cm, l in [-4, 4] cm), axis vertical, or for a lying cylinder horizontal and perpendicular to the finger
        closing axis (fingers close across the diameter); table height z fixed for the rig (`sfagen table`) or searched; maximise the IoU of the projected proxy
        silhouette with the frame-0 foreground (not robot) within 160 px of the grasp point."""
        self.robot.fk(self.q_at(tg)); fa = self.robot.finger_axis(arm)
        b_ = (fa if fa is not None else TG[:3, 1]).copy(); b_[2] = 0; b_ /= np.linalg.norm(b_) + 1e-9
        a_ = TG[:3, 2].copy(); a_[2] = 0; a_ = a_ - b_ * (a_ @ b_); a_ /= np.linalg.norm(a_) + 1e-9
        horiz = p.get("axis", "vertical") == "horizontal"
        pts = cylinder_points(p["radius"], p["length"], spacing=0.004)
        roi = np.zeros(fg.shape, np.uint8); cv2.circle(roi, (u0, v0), 160, 1, -1); f = fg & (roi > 0)
        z_hi = float(self.ctx.lowest_arm_z) - 0.005
        zs_coarse = [float(self.ctx.table_height)] if self.ctx.table_fixed else np.arange(z_hi - 0.15, z_hi + 1e-9, 0.01)
        best = (-1.0, None, None, None)
        def pose(cxy, z):
            T = np.eye(4)
            if horiz:
                zax = np.cross([0, 0, 1.0], b_); zax /= np.linalg.norm(zax); xax = np.array([0, 0, 1.0]); yax = np.cross(zax, xax)
                T[:3, 0], T[:3, 1], T[:3, 2] = xax, yax, zax; T[:3, 3] = np.array([cxy[0], cxy[1], z + p["radius"]]) - zax * p["length"] / 2
            else:
                T[:3, 3] = [cxy[0], cxy[1], z]
            return T
        def score(T):
            uv, zc = self.cam.project((T[:3, :3] @ pts.T).T + T[:3, 3]); ok = (zc > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < self.W) & (uv[:, 1] >= 0) & (uv[:, 1] < self.H)
            sil = np.zeros((self.H, self.W), np.uint8); sil[uv[ok, 1].astype(int), uv[ok, 0].astype(int)] = 1
            sil = cv2.morphologyEx(sil, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0
            return float((sil & f).sum() / max(1, (sil | f).sum()))
        for z in zs_coarse:                                                     # coarse
            for s_ in np.arange(-0.08, 0.081, 0.01):
                for l_ in np.arange(-0.04, 0.041, 0.01):
                    cxy = TG[:2, 3] + s_ * a_[:2] + l_ * b_[:2]; sc = score(pose(cxy, z))
                    if sc > best[0]:
                        best = (sc, z, s_, l_)
        _, z0, s0, l0 = best
        for z in ([z0] if self.ctx.table_fixed else np.arange(z0 - 0.01, z0 + 0.0101, 0.0025)):    # fine
            for s_ in np.arange(s0 - 0.01, s0 + 0.0101, 0.0025):
                for l_ in np.arange(l0 - 0.01, l0 + 0.0101, 0.005):
                    if z > z_hi + 1e-9:
                        continue
                    cxy = TG[:2, 3] + s_ * a_[:2] + l_ * b_[:2]; sc = score(pose(cxy, z))
                    if sc > best[0]:
                        best = (sc, z, s_, l_)
        sc, z, s_, l_ = best
        return pose(TG[:2, 3] + s_ * a_[:2] + l_ * b_[:2], z), sc, float(z), float(s_)

    def _finish_objects(self):
        """holders: table -> grasping arm -> (hand-over to a closed arm | released on the spot)."""
        for o in self.objects:
            o["points"] = cylinder_points(o["dims"][0], o["dims"][1]) if o["proxy"] == "cylinder" else box_points(o["dims"])
            seg, arm, t = [], o["arm"], o["grasp_frame"] + 3
            seg.append(("table", 0, t)); X = []
            pose_at = lambda arm_, t_, Xm: self.tcp(arm_, t_) @ Xm  # noqa: E731
            cur = np.linalg.inv(self.tcp(arm, t)) @ o["table"]
            while True:
                opens = [x for x in self.ev[arm]["opens"] if x > t]
                t_end = opens[0] if opens else self.T
                seg.append((arm, t, t_end)); X.append(cur)
                if t_end >= self.T:
                    break
                Tw = pose_at(arm, t_end, cur)
                others = [s for s in self.ev if s != arm and any(c <= t_end for c in self.ev[s]["closes"]) and
                          (lambda s_: max([c for c in self.ev[s_]["closes"] if c <= t_end] + [-1]) > max([q for q in self.ev[s_]["opens"] if q <= t_end] + [-1]))(s)]
                if others and np.linalg.norm(self.tcp(others[0], t_end)[:3, 3] - Tw[:3, 3]) < 0.25:
                    arm, t = others[0], t_end; cur = np.linalg.inv(self.tcp(arm, t)) @ Tw
                    continue
                seg.append(("placed", t_end, self.T)); X.append(Tw); break
            o["holder"] = seg; o["X"] = [np.asarray(x) for x in X]

    def obj_pose(self, k, t_state):
        o = self.objects[k]; t = float(t_state); xi = 0
        for (h, a, b) in o["holder"]:
            if h == "table" and t < b:
                return o["table"]
            if h in self.arms:
                if t < b:
                    return self.tcp(h, t) @ o["X"][xi]
                xi += 1
            if h == "placed":
                return o["X"][xi]
        return o["table"]

    def holder_at(self, k, t):
        for (h, a, b) in self.objects[k]["holder"]:
            if a <= t < b:
                return h
        return "table"

    def obj_sil(self, k, Tw, dil=1):
        o = self.objects[k]; P = (Tw[:3, :3] @ o["points"][::4].T).T + Tw[:3, 3]
        uv, zc = self.cam.project(P); ok = (zc > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < self.W) & (uv[:, 1] >= 0) & (uv[:, 1] < self.H)
        m = np.zeros((self.H, self.W), np.uint8); m[uv[ok, 1].astype(int), uv[ok, 0].astype(int)] = 1
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        return cv2.dilate(m, np.ones((dil, dil), np.uint8)) > 0 if dil > 1 else m > 0

    # ------------------------------------------------------------------ masks
    def _masks(self):
        """arm masks: SAM2 video segmentation prompted from the URDF (reconstruction.robot_masks = "sam2", default) or the
        z-buffered mesh silhouettes ("mesh"); object masks: plate-difference foreground in the dilated proxy silhouette."""
        rc = self.ctx.cfg.get("reconstruction", {}); mode = rc.get("robot_masks", "sam2")
        self.masks = {s: np.zeros((self.T, self.H, self.W), np.uint8) for s in self.arms}
        for k in range(len(self.objects)):
            self.masks[f"obj{k}"] = np.zeros((self.T, self.H, self.W), np.uint8)
        sam = None
        if mode == "sam2":
            cp = self.ctx.cache_dir / f"sam2_ep{self.e}.npz"
            if cp.exists():
                z = np.load(cp); sam = {s: z[s] for s in self.arms}
            else:
                from .segment import segment_arms
                sam, meta = segment_arms(self, self.dev, rc.get("sam2_keyframe_every", 15), tuple(rc.get("sam2_exclude_links", [])))
                np.savez_compressed(cp, **sam); self.info["sam2"] = meta
        for j in range(self.T):
            sd = self.robot_sil(j, sides=True); robot = np.isin(sd, self.arms)
            for s in self.arms:
                self.masks[s][j] = sam[s][j] if sam is not None else (sd == s)
            if sam is not None:
                robot = robot | np.any([sam[s][j] > 0 for s in self.arms], axis=0)
            fg = self._fg(j, self.plate, 25)
            for k in range(len(self.objects)):
                self.masks[f"obj{k}"][j] = self.obj_sil(k, self.obj_pose(k, j + self.lag), dil=21) & fg & ~robot
                for s in self.arms:
                    self.masks[s][j][self.masks[f"obj{k}"][j] > 0] = 0
