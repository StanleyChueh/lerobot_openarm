"""Mesh-guided forward warping (MGW) of real pixels, robot- and task-agnostic. Port of render_v4/mgw.py (MGWForward)
to any URDF robot and any number of objects. Per output frame (source frame j, target joints / object poses):
 1 moved parts       robot geometries / objects whose projection moves >= 0.5 px
 2 background        source pixels of moved parts (robust masks, dilated 7 px, never over static robot pixels) are
                     replaced by the clean plate, optionally relit to frame j (normalised-convolution log-ratio)
 3 forward 3-D flow  every real pixel of a moved part gets a 3-D point from the z-buffered mesh / proxy depth and is
                     moved by its part's rigid motion; z-tested, splatted to 4 neighbours; pinholes closed (5x5)
 4 new surfaces      target silhouette pixels (eroded tm_erode) not reached are looked up in source frames j +- 3k
 5 holes             part-aware fill from the same part's warped pixels, else Telea inpainting
 6 clean-up          3x3 median despeckle, 1-px anti-aliasing
Robust masks: per arm/object, the mesh/proxy silhouette U plate-difference foreground within 12 px of it.
No generative model, no simulator."""
from __future__ import annotations
import cv2, numpy as np, torch
import torch.nn.functional as Fnn
from . import mesh
from .recon import interp_rows

BORROW = [d for k in range(1, 16) for d in (-3 * k, 3 * k)]


def _nearest(okn, W):
    _, lbl = cv2.distanceTransformWithLabels((~okn).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(okn); idx_of = np.zeros(lbl.max() + 1, np.int64); idx_of[lbl[ys, xs]] = ys * W + xs
    return idx_of[lbl]


class Renderer:
    def __init__(self, rec, device="cuda:0", cfg=None):
        cfg = {"tm_erode": 21, "despeckle": True, "part_fill": True, "plate_adapt": 10, **(cfg or {})}
        self.cfg, self.rec, self.dev = cfg, rec, device
        self.robot, self.cam = rec.robot, rec.cam; self.H, self.W = rec.H, rec.W
        self.G = mesh.Geometry(self.robot, device=device)
        loc, gid = [self.G.local], [self.G.gid]
        self.K = len(rec.objects); self.OBJ0 = self.G.ng
        for k, o in enumerate(rec.objects):
            P = torch.tensor(o["points"], dtype=torch.float32, device=device); loc.append(P)
            gid.append(torch.full((len(P),), self.OBJ0 + k, dtype=torch.long, device=device))
        self.local = torch.cat(loc); self.gid = torch.cat(gid)
        self.cats = list(rec.arms) + [f"obj{k}" for k in range(self.K)]                 # mask categories
        cat_of = [self.cats.index(s) if s in self.cats else -1 for s in self.G.side] + [len(rec.arms) + k for k in range(self.K)]
        self.cat_of = np.array(cat_of); self.cat_of_t = torch.tensor(self.cat_of, device=device)
        self.movable = torch.tensor(self.cat_of >= 0, device=device)
        self.F = torch.tensor(rec.F, device="cpu"); self.plate = torch.tensor(rec.plate, dtype=torch.float32, device=device)
        self._cache = {}

    # ------------------------------------------------------------------ geometry
    def placements(self, q, g, objs):
        Ms = self.G.placements(q, g)
        far = np.eye(4); far[2, 3] = -10.0
        return np.concatenate([Ms, np.stack([o if o is not None else far for o in objs])], 0) if self.K else Ms

    def world(self, Ms, sub=None):
        gid = self.gid if sub is None else self.gid[sub]; loc = self.local if sub is None else self.local[sub]
        M = torch.tensor(Ms, dtype=torch.float32, device=self.dev)[gid]
        return torch.einsum("nij,nj->ni", M[:, :3, :3], loc) + M[:, :3, 3]

    def project(self, P):
        R = torch.tensor(self.cam.R, dtype=torch.float32, device=self.dev); t = torch.tensor(self.cam.t, dtype=torch.float32, device=self.dev)
        c = (P - t) @ R; K = self.cam.K
        return torch.stack([K[0, 0] * c[:, 0] / c[:, 2] + K[0, 2], K[1, 1] * c[:, 1] / c[:, 2] + K[1, 2]], 1), c[:, 2]

    def zbuf(self, Ms):
        return mesh.splat(self.world(Ms), self.gid, self.cam, dev=self.dev)

    def source_state(self, j):
        if j in self._cache:
            return self._cache[j]
        rec = self.rec
        q = {s: rec.gv[s][j] for s in rec.arms}; g = {s: float(rec.grip_v[s][j]) for s in rec.grip_v}
        Ms = self.placements(q, g, [rec.obj_pose(k, j + rec.lag) for k in range(self.K)])
        d, lab, _ = self.zbuf(Ms); cat = self.robust_masks(j, lab)
        labn, dep = lab.cpu().numpy(), d.cpu().numpy()
        part = np.full((self.H, self.W), -1, np.int64); depth = np.full((self.H, self.W), np.inf)
        for c in range(len(self.cats)):
            m = cat == c
            if not m.any():
                continue
            ok = (labn >= 0) & (self.cat_of[np.clip(labn, 0, None)] == c)
            if not ok.any():
                continue
            near = _nearest(ok, self.W)[m]; part[m] = labn.reshape(-1)[near]; depth[m] = dep.reshape(-1)[near]
        v, u = np.mgrid[0:self.H, 0:self.W].astype(np.float64); K = self.cam.K; z = np.where(np.isfinite(depth), depth, 1.0)
        P = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1) @ self.cam.R.T + self.cam.t
        S = {"Ms": Ms, "depth": d, "lab": lab, "cat": torch.tensor(cat, device=self.dev), "part": torch.tensor(part, device=self.dev),
             "sdepth": torch.tensor(depth, dtype=torch.float32, device=self.dev), "P": torch.tensor(P, dtype=torch.float32, device=self.dev)}
        if len(self._cache) > 120:
            self._cache.pop(next(iter(self._cache)))
        self._cache[j] = S
        return S

    def robust_masks(self, j, lab):
        rec = self.rec; img = rec.F[j].astype(np.int16)
        diff = np.abs(img - rec.plate.astype(np.int16)).max(2) > 28
        diff = cv2.morphologyEx(diff.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        labn = lab.cpu().numpy(); rs = np.where(labn >= 0, self.cat_of[np.clip(labn, 0, None)], -1)
        nc = len(self.cats); cat = np.full((self.H, self.W), -1, np.int64)
        dist = np.stack([cv2.distanceTransform((rs != c).astype(np.uint8), cv2.DIST_L2, 3) if (rs == c).any() else np.full((self.H, self.W), 1e9, np.float32) for c in range(nc)])
        best = dist.argmin(0); near = dist.min(0) <= 12
        fg = diff & near; cat[fg] = best[fg]
        inside = (rs >= 0) & (np.abs(img - rec.plate.astype(np.int16)).max(2) > 12); cat[inside] = rs[inside]
        for c, name in enumerate(self.cats):
            cat[rec.masks[name][j] > 0] = c
        return cat

    def lookup(self, j, fn, labels, tol=0.015):
        S = self.source_state(j); Pw = fn(S["Ms"]); uv, z = self.project(Pw)
        ui, vi = uv[:, 0].round().long(), uv[:, 1].round().long()
        inb = (ui >= 1) & (ui < self.W - 1) & (vi >= 1) & (vi < self.H - 1) & (z > 0.05); ui, vi = ui.clamp(1, self.W - 2), vi.clamp(1, self.H - 2)
        cat = self.cat_of_t[labels]
        valid = inb & ((z - S["depth"][vi, ui]).abs() < tol) & (S["lab"][vi, ui] == labels)
        for du, dv in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            valid &= S["cat"][vi + dv, ui + du] == cat
        img = self.F[j].to(self.dev).permute(2, 0, 1)[None].float()
        g = torch.stack([uv[:, 0] / (self.W - 1) * 2 - 1, uv[:, 1] / (self.H - 1) * 2 - 1], 1)[None, None]
        return Fnn.grid_sample(img, g, mode="bilinear", align_corners=True)[0, :, 0].T, valid

    def adapted_plate(self, img, fg):
        bgm = ~torch.tensor(cv2.dilate(fg.cpu().numpy().astype(np.uint8), np.ones((9, 9), np.uint8)) > 0, device=self.dev)
        lr = (torch.log(img + 8.0) - torch.log(self.plate + 8.0)).clamp(-1.0, 1.0)
        sg = float(self.cfg["plate_adapt"]); r = int(3 * sg); x = torch.arange(-r, r + 1, device=self.dev, dtype=torch.float32)
        gk = torch.exp(-0.5 * (x / sg) ** 2); gk = gk / gk.sum()
        blur = lambda t: Fnn.conv2d(Fnn.conv2d(t[:, None], gk.view(1, 1, 1, -1), padding=(0, r)), gk.view(1, 1, -1, 1), padding=(r, 0))[:, 0]  # noqa: E731
        w = bgm.float()[None]; num = blur(lr.permute(2, 0, 1) * w); den = blur(w)
        corr = torch.where(den > 0.05, num / den.clamp(min=1e-6), torch.zeros_like(num)).permute(1, 2, 0)
        return (self.plate + 8.0) * torch.exp(corr) - 8.0

    # ------------------------------------------------------------------ one output frame
    @torch.no_grad()
    def render(self, j, q, g, objs, exclude=None):
        H, W, dev = self.H, self.W, self.dev
        S = self.source_state(j); Mt = self.placements(q, g, objs)
        Mtt = torch.tensor(Mt, dtype=torch.float32, device=dev); Mst = torch.tensor(S["Ms"], dtype=torch.float32, device=dev)
        rel = Mtt @ torch.linalg.inv(Mst)
        sub = torch.arange(0, len(self.local), 50, device=dev)
        uo, _ = self.project(self.world(S["Ms"], sub)); un, _ = self.project(self.world(Mt, sub))
        disp = torch.zeros(len(Mt), device=dev).scatter_reduce(0, self.gid[sub], (un - uo).norm(dim=1), reduce="amax")
        moved = (disp >= 0.5) & self.movable
        part = S["part"]; img_s = self.F[j].to(dev).float(); out = img_s.clone()
        mv_all = (part >= 0) & moved[part.clamp(min=0)]; static = (part >= 0) & ~moved[part.clamp(min=0)]
        rm = torch.tensor(cv2.dilate(mv_all.cpu().numpy().astype(np.uint8), np.ones((7, 7), np.uint8)) > 0, device=dev) & ~static
        out[rm] = (self.adapted_plate(img_s, (part >= 0) | rm) if self.cfg["plate_adapt"] else self.plate)[rm]
        zb = torch.where(static, S["sdepth"], torch.full_like(S["sdepth"], float("inf")))
        ys, xs = mv_all.nonzero(as_tuple=True); info = {"moved_parts": int(moved.sum()), "warped_px": int(len(ys))}
        hit = torch.zeros((H, W), dtype=torch.bool, device=dev)
        if len(ys):
            Rm = rel[part[ys, xs]]; Pn = torch.einsum("nij,nj->ni", Rm[:, :3, :3], S["P"][ys, xs]) + Rm[:, :3, 3]
            uv, z = self.project(Pn); col = img_s[ys, xs]; cp, cz, cc = [], [], []
            for du in (0, 1):
                for dv in (0, 1):
                    uu = torch.floor(uv[:, 0]).long() + du; vv = torch.floor(uv[:, 1]).long() + dv
                    ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H) & (z > 0.05)
                    cp.append((vv * W + uu)[ok]); cz.append(z[ok]); cc.append(col[ok])
            pix, zz, cc = torch.cat(cp), torch.cat(cz), torch.cat(cc)
            front = torch.full((H * W,), float("inf"), device=dev); front.scatter_reduce_(0, pix, zz, reduce="amin")
            vis = (zz <= front[pix] + 1e-6) & (zz < zb.view(-1)[pix] - 0.005)
            acc = torch.zeros((H * W, 3), device=dev); cnt = torch.zeros(H * W, device=dev)
            acc.index_add_(0, pix[vis], cc[vis]); cnt.index_add_(0, pix[vis], torch.ones_like(zz[vis]))
            hit0 = (cnt.view(1, 1, H, W) > 0).float(); k5 = torch.ones(1, 1, 5, 5, device=dev)
            region = -Fnn.max_pool2d(-Fnn.max_pool2d(hit0, 5, 1, 2), 5, 1, 2)
            avg = Fnn.conv2d(acc.T.reshape(3, 1, H, W), k5, padding=2).view(3, -1).T; wsum = Fnn.conv2d(cnt.view(1, 1, H, W), k5, padding=2).view(-1)
            fm = ((region > 0) & (hit0 == 0)).view(-1) & (wsum > 0); h = cnt > 0
            out.view(-1, 3)[h] = acc[h] / cnt[h, None]; out.view(-1, 3)[fm] = avg[fm] / wsum[fm, None]
            hit = (h | fm).view(H, W)
        dt, lt, pt = self.zbuf(Mt)
        tmask = (lt >= 0) & moved[lt.clamp(min=0)] & (dt < zb - 0.005)
        tm = cv2.erode(tmask.cpu().numpy().astype(np.uint8), np.ones((self.cfg["tm_erode"],) * 2, np.uint8)) > 0
        hn = tm & ~hit.cpu().numpy(); hitn = hit.cpu().numpy()
        small = hn & (cv2.dilate(hitn.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & (cv2.blur(hitn.astype(np.float32), (5, 5)) > 0.6)
        big = torch.tensor(hn & ~small, device=dev); pixb = big.nonzero(); filled_n = 0
        if len(pixb):
            pid = pt[pixb[:, 0], pixb[:, 1]]; labs = lt[pixb[:, 0], pixb[:, 1]]; loc = self.local[pid]
            col = torch.zeros((len(pixb), 3), device=dev); filled = torch.zeros(len(pixb), dtype=torch.bool, device=dev)
            for d in [0] + BORROW:
                if filled.all():
                    break
                jj = j + d
                if jj < 0 or jj >= self.rec.T or (exclude is not None and exclude[0] <= jj <= exclude[1]):
                    continue
                ri = (~filled).nonzero()[:, 0]
                fn = lambda Ms, ri=ri: torch.einsum("nij,nj->ni", torch.tensor(Ms, dtype=torch.float32, device=dev)[labs[ri]][:, :3, :3], loc[ri]) + torch.tensor(Ms, dtype=torch.float32, device=dev)[labs[ri]][:, :3, 3]  # noqa: E731
                rgb, v = self.lookup(jj, fn, labs[ri]); col[ri[v]] = rgb[v]; filled[ri[v]] = True
            out[pixb[filled][:, 0], pixb[filled][:, 1]] = col[filled]; filled_n = int(filled.sum()); big[pixb[filled][:, 0], pixb[filled][:, 1]] = False
        info["borrowed_px"] = filled_n
        img = out.clamp(0, 255).byte().cpu().numpy(); rest = small | big.cpu().numpy()
        info['part_filled_px']=0
        if self.cfg["part_fill"]:
            full = tmask.cpu().numpy() & ~hitn & ~(~rest & tm)
            if full.any() and hitn.any():
                labn = lt.cpu().numpy(); done = np.zeros_like(full)
                for p_ in np.unique(labn[full]):
                    src = hitn & (labn == p_)
                    if src.sum() < 20:
                        continue
                    sel = full & (labn == p_); img.reshape(-1, 3)[np.flatnonzero(sel)] = img.reshape(-1, 3)[_nearest(src, W)[sel]]; done |= sel
                rest &= ~done
                info['part_filled_px']=int(done.sum())
                if done.any():
                    bl = cv2.GaussianBlur(img, (5, 5), 1.2); img[done] = bl[done]
        if rest.any():
            img = cv2.inpaint(img, rest.astype(np.uint8), 3, cv2.INPAINT_TELEA)
        info["inpainted_px"] = int(rest.sum()); wm = (hitn | rest).astype(np.uint8)
        if self.cfg["despeckle"]:
            reg = cv2.dilate(wm, np.ones((5, 5), np.uint8)) > 0; med = cv2.medianBlur(img, 3)
            sel = reg & (np.abs(img.astype(np.int16) - med.astype(np.int16)).max(2) > 40); img[sel] = med[sel]
        edge = (cv2.dilate(wm, np.ones((3, 3), np.uint8)) - cv2.erode(wm, np.ones((3, 3), np.uint8))) > 0
        bl = cv2.GaussianBlur(img, (3, 3), 0.7); img[edge] = bl[edge]
        return img, info


def candidate_frames_tw(rec,c):
    """Output index, source RGB index and continuous generated scene at the exposure time."""
    from .timing import source_frame_map, pose_at, source_time_at
    from .plan import _attach
    jm=source_frame_map(rec,c);sp=rec.robot.split(c['state'])
    for i,j in enumerate(jm):
        ts=i+rec.lag
        q={a:interp_rows(c['q'][a],np.array([ts]))[0] for a in rec.arms}
        g={a:float(interp_rows(sp['g'+a][:,None],np.array([ts]))[0,0]) for a in rec.arms if 'g'+a in sp}
        # FK of interpolated joints, rather than interpolating object translations independently,
        # keeps a held object exactly attached at fractional camera-exposure times.
        source_time=float(np.clip(source_time_at(rec,c,ts),0,rec.T-1))
        rec.robot.fk(q,g);tcp={a:rec.robot.frame(rec.robot.tcp[a]) for a in rec.arms}
        objects=[]
        for k,P in enumerate(c['objects']):
            holder=rec.holder_at(k,source_time)
            if holder in rec.arms:objects.append(tcp[holder]@_attach(rec.objects[k],rec,holder,source_time))
            elif holder=='table':objects.append(P[0].copy())
            else:objects.append(pose_at(P,ts))
        yield i,int(j),q,g,objects


def candidate_frames(rec,c):
    return [(i,q,g,objs) for i,j,q,g,objs in candidate_frames_tw(rec,c)]
