"""Wrist-camera view synthesis: the MGW body renderer generalised to a camera that moves with the arm. Task-agnostic: it
uses only the URDF, joint states, the body-camera object reconstruction and the source wrist video (no colours, labels,
simulator or learned generator).

Camera model (config `wrist_cameras_model`, a dict or a YAML path): C(t) = FK_parent(q(t)) . X at state time
t = video frame + lag.

Per source wrist frame j (cached):
  labels  z-buffered URDF point clouds of both arms + object proxies at their reconstructed poses, seen from C_j
  depth   geometry depth on labelled pixels; elsewhere background depth, by `depth`:
            'da2'   Depth Anything V2 metric depth, scale-aligned per frame (median ratio) to visible geometry and to the
                    table plane, then snapped to the plane where they agree within 10 %
            'raft'  two-view RAFT flow to frame j +- 3, triangulated along the known camera motion (as v3_2)
            'plane' the table plane where the ray hits it within `table_max_m`, else `far_m`
Per output frame (target joints q', grippers g', object poses):
  every source pixel moves rigidly with its element, P' = inv(C') M'_e inv(M_e) C_j P (static world: M = I), so parts
  rigid w.r.t. the camera (own hand, fingers at the same opening, held object) are unchanged automatically;
  forward splat with z-test and pinhole closing -> static-world holes filled from the nearest-pose source frames
  (multiview) -> Telea inpainting. The source frame is returned unchanged when no element changes its camera-relative pose.
Provenance per pixel: 0 unchanged source, 2 warped, 3 multiview, 5 inpainted.
Design follows the v3_2 study (frozen, research/sim_free_augmentation/v3_2/sfa32/render_wrist.py), with geometry labels in
place of its red-HSV can mask, and depth alignment after Real2Edit2Real / Augmentron (see results/LITERATURE.md)."""
from __future__ import annotations
from pathlib import Path
import cv2, numpy as np, torch, yaml
import torch.nn.functional as Fnn
from . import mesh
from .camera import Camera, quat_R
from .recon import interp_rows

WORLD = -1


def load_spec(ctx):
    m = ctx.cfg.get("wrist_cameras_model")
    if isinstance(m, str):
        m = yaml.safe_load((ctx.base / m).read_text())
    return m or {}


class WristCam:
    def __init__(self, name, spec, robot):
        self.name, self.arm, self.lag = name, spec["arm"], float(spec.get("lag", 0.0))
        self.fid = robot.m.getFrameId(spec["parent_frame"]); assert self.fid < robot.m.nframes, spec["parent_frame"]
        X = np.eye(4); X[:3, :3] = np.array(spec["R"]) if "R" in spec else quat_R(spec["quat_wxyz"]); X[:3, 3] = spec["pos"]; self.X = X
        i = spec["intrinsics"]; self.W, self.H = int(i["width"]), int(i["height"])
        self.K = np.array([[i["fx"], 0, i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1.0]])

    def camera(self, C):
        c = Camera.__new__(Camera); c.W, c.H, c.K, c.R, c.t, c.cfg = self.W, self.H, self.K, C[:3, :3].copy(), C[:3, 3].copy(), {}
        return c


_DA = {}


def da2_depth(frames, device, model="depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf", bs=16):
    """Metric depth (m) per frame at half resolution, float16."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    if model not in _DA:
        _DA[model] = (AutoImageProcessor.from_pretrained(model), AutoModelForDepthEstimation.from_pretrained(model).to(device).eval())
    proc, net = _DA[model]; H, W = frames.shape[1:3]; out = []
    with torch.no_grad():
        for a in range(0, len(frames), bs):
            x = proc(images=list(frames[a:a + bs]), return_tensors="pt").to(device)
            d = net(**x).predicted_depth[:, None]
            out.append(Fnn.interpolate(d, (H // 2, W // 2), mode="bilinear", align_corners=False)[:, 0].half().cpu().numpy())
    return np.concatenate(out)


_RAFT = {}


def raft_flow(a, b, device):
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    if "m" not in _RAFT:
        _RAFT["m"] = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()
    t = lambda x: (torch.tensor(x, device=device).permute(2, 0, 1)[None].float() / 127.5 - 1)  # noqa: E731
    with torch.no_grad():
        return _RAFT["m"](t(a), t(b))[-1][0].permute(1, 2, 0).cpu().numpy()


def triangulate(flow, A, K, far):
    """Depth in camera j from flow j->k and A = inv(C_j) C_k (midpoint method; parallax < 0.4 deg -> far). From v3_2."""
    H, W = flow.shape[:2]; gy, gx = np.mgrid[0:H, 0:W]
    x = np.stack([gx, gy, np.ones_like(gx)], -1).reshape(-1, 3).astype(np.float64); f = flow.reshape(-1, 2)
    Ki = np.linalg.inv(K); rj = x @ Ki.T; xk = x.copy(); xk[:, :2] += f; rk = xk @ Ki.T
    R, c = A[:3, :3], A[:3, 3]; dk = rk @ R.T
    a, b, cc, d, e = (rj * rj).sum(1), (rj * dk).sum(1), (dk * dk).sum(1), rj @ c, dk @ c
    den = a * cc - b * b; den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    lam, mu = (d * cc - b * e) / den, (b * d - a * e) / den
    par = np.degrees(np.arccos(np.clip(b / np.sqrt(a * cc), -1, 1)))
    ok = (par > 0.4) & (lam > 0.05) & (mu > 0) & (lam < 1.6 * far)
    return cv2.medianBlur(np.where(ok, lam, far).reshape(H, W).astype(np.float32), 5)


class WristRenderer:
    def __init__(self, rec, frames, wc, cfg=None, device="cuda:0"):
        self.cfg = {"depth": "da2", "label_grow": 3, "far_m": 2.5, "table_max_m": 1.5, "multiview": 2, "spacing": 0.002, "cache": 300, "attached_px": 200, **(cfg or {})}
        self.rec, self.F, self.wc, self.dev = rec, frames, wc, device
        self.T, self.H, self.W = len(frames), frames.shape[1], frames.shape[2]; self.robot = rec.robot
        self.G = mesh.Geometry(self.robot, device=device, spacing=self.cfg["spacing"])
        loc, gid = [self.G.local], [self.G.gid]; self.OBJ0 = self.G.ng; self.K = len(rec.objects)
        for k, o in enumerate(rec.objects):
            P = torch.tensor(o["points"], dtype=torch.float32, device=device); loc.append(P)
            gid.append(torch.full((len(P),), self.OBJ0 + k, dtype=torch.long, device=device))
        self.local, self.gid = torch.cat(loc), torch.cat(gid); self.NE = self.OBJ0 + self.K
        v, u = np.mgrid[0:self.H, 0:self.W].astype(np.float64); Kc = wc.K
        self.ray = torch.tensor(np.stack([(u - Kc[0, 2]) / Kc[0, 0], (v - Kc[1, 2]) / Kc[1, 1], np.ones_like(u)], -1), dtype=torch.float32, device=device)
        self._src, self._da, self.last_prov = {}, None, None
        self.pos = np.array([self.cam_pose(*self.scene(j + wc.lag)[:2])[:3, 3] for j in range(self.T)])

    # ---------------------------------------------------------------- scene
    def scene(self, t):
        rec = self.rec; t = np.array([float(t)])
        q = {s: interp_rows(rec.sp[s], t)[0] for s in rec.arms}
        g = {s: float(interp_rows(rec.sp["g" + s][:, None], t)[0, 0]) for s in rec.arms if "g" + s in rec.sp}
        return q, g, [rec.obj_pose(k, float(t[0])) for k in range(self.K)]

    def cam_pose(self, q, g):
        self.robot.fk(q, g); return self.robot.frame(self.wc.fid) @ self.wc.X

    def placements(self, q, g, objs):
        Ms = self.G.placements(q, g); C = self.robot.frame(self.wc.fid) @ self.wc.X
        far = np.eye(4); far[2, 3] = -10.0
        return (np.concatenate([Ms, np.stack([o if o is not None else far for o in objs])]) if self.K else Ms), C

    def world(self, Ms):
        M = torch.tensor(Ms, dtype=torch.float32, device=self.dev)[self.gid]
        return torch.einsum("nij,nj->ni", M[:, :3, :3], self.local) + M[:, :3, 3]

    # ---------------------------------------------------------------- background depth
    def plane_depth(self, C):
        """Camera-z depth of the table plane per pixel (inf where the ray misses it or beyond table_max_m)."""
        Rt = torch.tensor(C[:3, :3], dtype=torch.float32, device=self.dev); dz = (self.ray @ Rt.T)[..., 2]
        lam = (float(self.rec.z_table) - float(C[2, 3])) / dz
        return torch.where((lam > 0.03) & (lam < self.cfg["table_max_m"]), lam, torch.full_like(lam, float("inf")))

    def bg_depth(self, j, C, gdepth, glab):
        mode = self.cfg["depth"]; far = self.cfg["far_m"]; pl = self.plane_depth(C)
        if mode == "plane":
            return torch.where(torch.isfinite(pl), pl, torch.full_like(pl, far))
        if mode == "raft":
            k = j + 3 if j + 3 < self.T else j - 3
            Ck = self.cam_pose(*self.scene(k + self.wc.lag)[:2])
            d = triangulate(raft_flow(self.F[j], self.F[k], self.dev), np.linalg.inv(C) @ Ck, self.wc.K, far)
            return torch.tensor(d, device=self.dev)
        if self._da is None:
            self._da = da2_depth(self.F, self.dev)
        da = Fnn.interpolate(torch.tensor(self._da[j], device=self.dev).float()[None, None], (self.H, self.W), mode="bilinear", align_corners=False)[0, 0]
        geo = (glab >= 0) & torch.isfinite(gdepth)
        r_geo = (gdepth[geo] / da[geo]).log(); r_pl = (pl / da)[torch.isfinite(pl)].log()
        s0 = r_geo.median() if len(r_geo) >= 500 else (r_pl.median() if len(r_pl) else torch.tensor(0.0, device=self.dev))
        r = torch.cat([r_geo, r_pl[(r_pl - s0).abs() < 0.2]]); s = (r.median() if len(r) else s0).exp()
        d = (s * da).clamp(0.03, 2 * far)
        snap = torch.isfinite(pl) & ((d - pl).abs() < 0.1 * pl)
        return torch.where(snap, pl, d)

    # ---------------------------------------------------------------- per source frame
    def source(self, j):
        if j in self._src:
            return self._src[j]
        q, g, objs = self.scene(j + self.wc.lag); Ms, C = self.placements(q, g, objs)
        d, lab, _ = mesh.splat(self.world(Ms), self.gid, self.wc.camera(C), dev=self.dev)
        G = self.cfg["label_grow"]
        if G:
            labn = lab.cpu().numpy(); ok = labn >= 0
            if ok.any():
                dist, nl = cv2.distanceTransformWithLabels((~ok).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
                ys, xs = np.nonzero(ok); idx = np.zeros(nl.max() + 1, np.int64); idx[nl[ys, xs]] = ys * self.W + xs
                near = idx[nl].reshape(-1); grow = torch.tensor((~ok & (dist <= G)).reshape(-1), device=self.dev)
                nt = torch.tensor(near, device=self.dev)
                lab = torch.where(grow, lab.view(-1)[nt], lab.view(-1)).view(self.H, self.W); d = torch.where(grow, d.view(-1)[nt], d.view(-1)).view(self.H, self.W)
        bg = self.bg_depth(j, C, d, lab)
        depth = torch.where((lab >= 0) & torch.isfinite(d), d, bg)
        if self.cfg["attached_px"]:
            lab, depth = self._attach_static(j, C, lab, depth)
        S = {"Ms": Ms, "C": C, "lab": lab, "P": self.ray * depth[..., None]}
        if len(self._src) > self.cfg["cache"]:
            self._src.pop(next(iter(self._src)))
        self._src[j] = S
        return S

    def _attach_static(self, j, C, lab, depth, min_flow=3.0, margin=4.0):
        """Static-world pixels whose colour in frames j -+ 4 is explained better by staying at the same image position
        than by the static-world motion (> min_flow px; 7x7 box-averaged colour error, margin in grey levels) belong to
        something rigid with the camera or moving with the hand (fingers, a held or just-grasped object): projections miss
        them at close range under calibration / pose error. They take the label and depth of the nearest own-arm or object
        geometry within attached_px."""
        ks = [k for k in (j - 4, j + 4) if 0 <= k < self.T]
        world = (lab < 0).cpu().numpy()
        if not ks or not world.any():
            return lab, depth
        img = lambda x: torch.tensor(x, device=self.dev).float().permute(2, 0, 1)[None]  # noqa: E731
        box = lambda e: Fnn.avg_pool2d(e[None, None], 7, 1, 3)[0, 0]  # noqa: E731
        Fj = img(self.F[j]); P = (self.ray * depth[..., None]).view(-1, 3); Kc = self.wc.K
        v0, u0 = torch.meshgrid(torch.arange(self.H, device=self.dev).float(), torch.arange(self.W, device=self.dev).float(), indexing="ij")
        e_att, e_w, moved = 0, 0, torch.zeros((self.H, self.W), dtype=torch.bool, device=self.dev)
        for k in ks:
            Fk = img(self.F[k]); Dw = torch.tensor(np.linalg.inv(self.cam_pose(*self.scene(k + self.wc.lag)[:2])) @ C, dtype=torch.float32, device=self.dev)
            Q = (P @ Dw[:3, :3].T + Dw[:3, 3]).view(self.H, self.W, 3); z = Q[..., 2].clamp(min=1e-3)
            u = Kc[0, 0] * Q[..., 0] / z + Kc[0, 2]; v = Kc[1, 1] * Q[..., 1] / z + Kc[1, 2]
            moved |= torch.hypot(u - u0, v - v0) > min_flow
            g = torch.stack([u / (self.W - 1) * 2 - 1, v / (self.H - 1) * 2 - 1], -1)[None]
            Wk = Fnn.grid_sample(Fk, g, mode="bilinear", padding_mode="border", align_corners=True)
            e_w = e_w + box((Wk - Fj).abs().mean(1)[0]); e_att = e_att + box((Fk - Fj).abs().mean(1)[0])
        att = (moved & (e_att + margin * len(ks) < e_w)).cpu().numpy() & world
        att = cv2.morphologyEx(att.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
        labn = lab.cpu().numpy(); side = np.array(self.G.side)
        okc = np.isin(labn, [i for i in range(self.G.ng) if side[i] == self.wc.arm] + [self.OBJ0 + k for k in range(self.K)])
        if not att.any() or not okc.any():
            return lab, depth
        dist, nl = cv2.distanceTransformWithLabels((~okc).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(okc); idx = np.zeros(nl.max() + 1, np.int64); idx[nl[ys, xs]] = ys * self.W + xs
        sel = torch.tensor((att & (dist <= self.cfg["attached_px"])).reshape(-1), device=self.dev); nt = torch.tensor(idx[nl].reshape(-1), device=self.dev)
        lab = torch.where(sel, lab.view(-1)[nt], lab.view(-1)).view(self.H, self.W); depth = torch.where(sel, depth.view(-1)[nt], depth.view(-1)).view(self.H, self.W)
        return lab, depth

    # ---------------------------------------------------------------- one output frame
    def _deltas(self, S, Mt, Ct):
        """(NE + 1, 4, 4) camera-frame motion per element; last row = static world."""
        iCt = np.linalg.inv(Ct); D = np.einsum("ij,njk,nkl,lm->nim", iCt, Mt, np.linalg.inv(S["Ms"]), S["C"])
        return np.concatenate([D, (iCt @ S["C"])[None]])

    def _splat(self, P, col, out_shape):
        H, W = out_shape; Kc = self.wc.K; z = P[:, 2]
        u = Kc[0, 0] * P[:, 0] / z + Kc[0, 2]; v = Kc[1, 1] * P[:, 1] / z + Kc[1, 2]; cp, cz, cc = [], [], []
        for du in (0, 1):
            for dv in (0, 1):
                uu = torch.floor(u).long() + du; vv = torch.floor(v).long() + dv
                ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H) & (z > 0.02)
                cp.append((vv * W + uu)[ok]); cz.append(z[ok]); cc.append(col[ok])
        pix, zz, cc = torch.cat(cp), torch.cat(cz), torch.cat(cc)
        front = torch.full((H * W,), float("inf"), device=self.dev); front.scatter_reduce_(0, pix, zz, reduce="amin")
        vis = zz <= front[pix] * 1.01 + 1e-4
        acc = torch.zeros((H * W, 3), device=self.dev); cnt = torch.zeros(H * W, device=self.dev)
        acc.index_add_(0, pix[vis], cc[vis]); cnt.index_add_(0, pix[vis], torch.ones_like(zz[vis]))
        hit0 = (cnt.view(1, 1, H, W) > 0).float(); k5 = torch.ones(1, 1, 5, 5, device=self.dev)
        region = -Fnn.max_pool2d(-Fnn.max_pool2d(hit0, 5, 1, 2), 5, 1, 2)
        avg = Fnn.conv2d(acc.T.reshape(3, 1, H, W), k5, padding=2).view(3, -1).T; ws = Fnn.conv2d(cnt.view(1, 1, H, W), k5, padding=2).view(-1)
        fm = ((region > 0) & (hit0 == 0)).view(-1) & (ws > 0); h = cnt > 0
        img = torch.zeros((H * W, 3), device=self.dev); img[h] = acc[h] / cnt[h, None]; img[fm] = avg[fm] / ws[fm, None]
        zbuf = torch.where(h | fm, front, torch.full_like(front, float("inf")))
        return img.view(H, W, 3), (h | fm).view(H, W), zbuf.view(H, W)

    @torch.no_grad()
    def render(self, j, q, g, objs, exclude=None):
        S = self.source(j); Mt, Ct = self.placements(q, g, objs); D = self._deltas(S, Mt, Ct)
        lab = S["lab"]; idx = torch.where(lab >= 0, lab, torch.full_like(lab, self.NE)).view(-1)
        if np.abs(D - np.eye(4)).max() < 1e-6:                       # nothing changed its pose relative to this camera
            self.last_prov = np.zeros((self.H, self.W), np.uint8); return self.F[j].copy(), {"copied": True}
        Dt = torch.tensor(D, dtype=torch.float32, device=self.dev)[idx]
        P = torch.einsum("nij,nj->ni", Dt[:, :3, :3], S["P"].view(-1, 3)) + Dt[:, :3, 3]
        col = torch.tensor(self.F[j], device=self.dev).float().view(-1, 3)
        img, hit, zb = self._splat(P, col, (self.H, self.W))
        prov = np.where(hit.cpu().numpy(), 2, 0).astype(np.uint8); info = {"warped_px": int(hit.sum())}
        # target geometry: where moved robot/object surfaces are expected, static-world fill must not be used
        dt, lt, _ = mesh.splat(self.world(Mt), self.gid, self.wc.camera(Ct), dev=self.dev)
        geo_t = (lt >= 0).cpu().numpy(); hole = ~hit.cpu().numpy(); mv = 0
        if self.cfg["multiview"] and (hole & ~geo_t).any():
            cand = [k for k in range(self.T) if k != j and not (exclude is not None and exclude[0] <= k <= exclude[1])]
            for k in sorted(cand, key=lambda k: np.linalg.norm(self.pos[k] - Ct[:3, 3]))[: self.cfg["multiview"]]:
                Sk = self.source(k); wk = (Sk["lab"] < 0).view(-1); Dk = torch.tensor(np.linalg.inv(Ct) @ Sk["C"], dtype=torch.float32, device=self.dev)
                Pk = Sk["P"].view(-1, 3)[wk] @ Dk[:3, :3].T + Dk[:3, 3]
                ik, hk, zk = self._splat(Pk, torch.tensor(self.F[k], device=self.dev).float().view(-1, 3)[wk], (self.H, self.W))
                fill = torch.tensor(hole & ~geo_t, device=self.dev) & hk
                img[fill] = ik[fill]; f_ = fill.cpu().numpy(); prov[f_] = 3; hole &= ~f_; mv += int(f_.sum())
        info["multiview_px"] = mv
        out = img.clamp(0, 255).byte().cpu().numpy()
        if hole.any():
            out = cv2.inpaint(out, hole.astype(np.uint8), 3, cv2.INPAINT_TELEA); prov[hole] = 5
        info["inpainted_px"] = int(hole.sum()); info["inpainted_frac"] = float(hole.mean()); self.last_prov = prov
        return out, info
