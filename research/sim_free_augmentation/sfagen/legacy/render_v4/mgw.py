"""MGW: mesh-guided warping renderer (v4) for simulation-free demonstration augmentation, body_cam.

Replaces v3's per-link 2-D homography layers with dense 3-D correspondences through the robot's URDF meshes and a
cylinder can model, under a camera + joint-offset calibration fitted to the SAM2 arm masks (calib.json):

  1. target z-buffer: the robot (calibrated FK of the candidate state) and the can (candidate can pose) are splatted
     into the target view -> per pixel: which surface point of which part is visible (true 3-D occlusion between arms,
     can and gripper; no per-link depth sorting).
  2. texture: each visible target surface point is mapped back to where that SAME point lies in the source frame
     (source pose + calibrated camera) and is sampled there only if it is visible in the source z-buffer and inside
     the SAM2 mask of its part. Points hidden in the source frame are borrowed from nearby source frames (|dt| <= 45)
     where they are visible (multi-frame texture). Remaining holes inside a part are inpainted from that part's pixels.
  3. background: pixels that showed a MOVED part in the source frame (SAM2 masks, dilated) are replaced by the
     episode's clean plate; unmoved parts keep their original pixels exactly.
No generative model, no simulator, no physics.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as Fnn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mesh  # noqa: E402

SFA = mesh.SFA
sys.path[:0] = [str(SFA / "v3"), str(SFA / "v2")]
import pinocchio as pin  # noqa: E402
from sfa.robot import Camera, Robot  # noqa: E402
from sfa2 import render2  # noqa: E402
from sfa2.recon import rc  # noqa: E402
from sfa3 import render3  # noqa: E402

H, W = 480, 640
BORROW = [d for k in range(1, 16) for d in (-3 * k, 3 * k)]


def load_calib(path=HERE / "calib.json"):
    p = np.array(json.loads(Path(path).read_text())["params"]) if Path(path).exists() else np.zeros(21)
    cam0 = Camera(); c = Camera(); s = np.exp(p[6])
    c.R = cam0.R @ pin.exp3(p[:3]); c.t = cam0.t + p[3:6]; c.K = cam0.K.copy(); c.K[0, 0] *= s; c.K[1, 1] *= s
    return c, {"R": p[7:14], "L": p[14:21]}


def can_points(spacing=0.0015, rng=np.random.default_rng(1)):
    R, Hc = rc.CAN_RADIUS, rc.CAN_HEIGHT
    n_lid = int(np.pi * R * R / spacing ** 2); n_side = int(2 * np.pi * R * Hc / spacing ** 2)
    r = R * np.sqrt(rng.random(n_lid)); th = 2 * np.pi * rng.random(n_lid)
    lid = np.stack([r * np.cos(th), r * np.sin(th), np.zeros(n_lid)], 1)
    th = 2 * np.pi * rng.random(n_side); z = Hc * rng.random(n_side)
    side = np.stack([R * np.cos(th), R * np.sin(th), z], 1)
    return np.concatenate([lid, side])


class MGW:
    CACHE_N = 120   # source-frame cache size (speed only)
    def __init__(self, rec, device="cuda:0", calib=HERE / "calib.json"):
        self.rec, self.dev = rec, device
        self.robot = Robot()
        self.cam, self.off = load_calib(calib)
        self.G = mesh.Geometry(self.robot, device=device)
        cp = can_points()
        self.CAN = self.G.ng                                   # label of the can
        self.local = torch.cat([self.G.local, torch.tensor(cp, dtype=torch.float32, device=device)])
        self.gid = torch.cat([self.G.gid, torch.full((len(cp),), self.CAN, dtype=torch.long, device=device)])
        self.part_side = np.array(self.G.side + ["C"])       # R, L, B (body), C (can)
        self.F = torch.tensor(rec.F, device="cpu")            # uint8 (T,H,W,3) on cpu; frames moved to gpu on demand
        self.plate = torch.tensor(rec.plate, dtype=torch.float32, device=device)
        self.masks = rec.masks
        self._src_cache = {}

    # ---------------------------------------------------------------- geometry
    def placements(self, q, g, can):
        Ms = self.G.placements(q["R"] + self.off["R"], q["L"] + self.off["L"], g["R"], g["L"])
        C = can if can is not None else np.eye(4) * 0 + np.diag([1, 1, 1, 1]) + np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -10], [0, 0, 0, 0]])
        return np.concatenate([Ms, C[None]], 0)

    def world(self, Ms):
        M = torch.tensor(Ms, dtype=torch.float32, device=self.dev)[self.gid]
        return torch.einsum("nij,nj->ni", M[:, :3, :3], self.local) + M[:, :3, 3]

    def project(self, P):
        R = torch.tensor(self.cam.R, dtype=torch.float32, device=self.dev); t = torch.tensor(self.cam.t, dtype=torch.float32, device=self.dev)
        c = (P - t) @ R; K = self.cam.K
        return torch.stack([K[0, 0] * c[:, 0] / c[:, 2] + K[0, 2], K[1, 1] * c[:, 1] / c[:, 2] + K[1, 2]], 1), c[:, 2]

    def zbuf(self, Ms):
        return mesh.splat(self.world(Ms), self.gid, self.cam, dev=self.dev)

    def source_state(self, j):
        if j not in self._src_cache:
            rec = self.rec
            q = {s: rec.gv[s][j] for s in "RL"}; g = {s: float(rec.grip_v[s][j]) for s in "RL"}
            Ms = self.placements(q, g, rec.source_can(self.robot, j + rec.lag))
            d, lab, _ = self.zbuf(Ms)
            sam = self.robust_masks(j, lab)
            if len(self._src_cache) > self.CACHE_N:
                self._src_cache.pop(next(iter(self._src_cache)))
            self._src_cache[j] = {"Ms": Ms, "depth": d, "lab": lab, "sam": torch.tensor(sam, device=self.dev),
                                  "img": self.F[j].to(self.dev).permute(2, 0, 1)[None].float()}
        return self._src_cache[j]

    def robust_masks(self, j, lab):
        """per-pixel part side (0 R, 1 L, 2 can, -1 none): SAM2 masks UNION render-guided foreground, i.e. pixels within
        12 px of a rendered part of that side that differ from the clean plate (SAM2 misses whole gripper bodies)."""
        img = self.rec.F[j].astype(np.int16); diff = np.abs(img - self.rec.plate.astype(np.int16)).max(2) > 28
        diff = cv2.morphologyEx(diff.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        labn = lab.cpu().numpy()
        side_of_lab = np.select([self.part_side == "R", self.part_side == "L", self.part_side == "C"], [0, 1, 2], -1)
        rs = np.where(labn >= 0, side_of_lab[np.clip(labn, 0, None)], -1)
        sam = np.full((H, W), -1, np.int64)
        near = {}
        for k, s in enumerate(("R", "L", "can")):
            ren = (rs == k).astype(np.uint8)
            near[k] = cv2.dilate(ren, np.ones((25, 25), np.uint8)) > 0
        dist = {}
        for k in range(3):
            ren = (rs == k)
            dist[k] = cv2.distanceTransform((~ren).astype(np.uint8), cv2.DIST_L2, 3) if ren.any() else np.full((H, W), 1e9, np.float32)
        best = np.argmin(np.stack([dist[0], dist[1], dist[2]]), 0)
        fg = diff & (near[0] | near[1] | near[2])
        sam[fg] = best[fg]
        sam[rs >= 0] = np.where(rs[rs >= 0] >= 0, rs[rs >= 0], -1) * 0 + sam[rs >= 0]   # keep fg decision inside render
        inside = (rs >= 0) & (np.abs(img - self.rec.plate.astype(np.int16)).max(2) > 12)
        sam[inside] = rs[inside]
        for k, s in enumerate(("R", "L", "can")):
            sam[self.masks[s][j] > 0] = k
        return sam

    # ---------------------------------------------------------------- texture lookup
    def lookup(self, j, pts_local_world_fn, labels, tol=0.015):
        """sample source frame j at the source-pose positions of the given target surface points.
        returns rgb (N,3), valid (N,) bool."""
        S = self.source_state(j)
        Pw = pts_local_world_fn(S["Ms"])
        uv, z = self.project(Pw)
        ui, vi = uv[:, 0].round().long(), uv[:, 1].round().long()
        inb = (ui >= 1) & (ui < W - 1) & (vi >= 1) & (vi < H - 1) & (z > 0.05)
        ui, vi = ui.clamp(1, W - 2), vi.clamp(1, H - 2)
        dz = z - S["depth"][vi, ui]
        side = torch.tensor(np.select([self.part_side == "R", self.part_side == "L", self.part_side == "C"], [0, 1, 2], -1),
                            device=self.dev)[labels]
        vis = inb & (dz.abs() < tol) & (S["lab"][vi, ui] == labels)
        insam = (S["sam"][vi, ui] == side) | (side < 0)
        # also require the 4-neighbourhood to be inside the part (keeps rim/background pixels out)
        for du, dv in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            insam &= (S["sam"][(vi + dv), (ui + du)] == side) | (side < 0)
        valid = vis & insam
        g = torch.stack([uv[:, 0] / (W - 1) * 2 - 1, uv[:, 1] / (H - 1) * 2 - 1], 1)[None, None]
        rgb = Fnn.grid_sample(S["img"], g, mode="bilinear", align_corners=True)[0, :, 0].T
        return rgb, valid

    # ---------------------------------------------------------------- one output frame
    @torch.no_grad()
    def render(self, j, q_new, g_new, can_new, min_disp=0.5, exclude=None):
        rec = self.rec
        S = self.source_state(j)
        Mt = self.placements(q_new, g_new, can_new)
        depth_t, lab_t, pt_t = self.zbuf(Mt)
        # moved parts: max projected displacement of each part's points between source and target pose
        sub = torch.arange(0, len(self.local), 50, device=self.dev)
        uo, _ = self.project(self._world_sub(S["Ms"], sub)); un, _ = self.project(self._world_sub(Mt, sub))
        disp = torch.zeros(self.CAN + 1, device=self.dev).scatter_reduce(0, self.gid[sub], (un - uo).norm(dim=1), reduce="amax")
        moved = (disp >= min_disp) & torch.tensor(self.part_side != "B", device=self.dev)
        if can_new is None:
            moved[self.CAN] = False
        out = self.F[j].to(self.dev).float()                  # (H,W,3)
        # 3. background: remove source pixels of moved parts -> plate
        lab_s = S["lab"].clone()
        sam = S["sam"]
        sam_part = self._sam_part(lab_s, sam)                  # part id for every SAM pixel (nearest rendered part of its side)
        remove = (sam >= 0) & moved[sam_part.clamp(min=0)] & (sam_part >= 0)
        remove = torch.tensor(cv2.dilate(remove.cpu().numpy().astype(np.uint8), np.ones((7, 7), np.uint8)) > 0, device=self.dev)
        keep_static = (sam >= 0) & ~moved[sam_part.clamp(min=0)] & (sam_part >= 0)
        remove &= ~keep_static
        out[remove] = self.plate[remove]
        # 1-2. paint target pixels of moved parts (and unmoved parts disoccluded by a moved part)
        tgt = lab_t >= 0
        paint = tgt & (moved[lab_t.clamp(min=0)] | (remove & (lab_t >= 0) & torch.tensor(self.part_side != "B", device=self.dev)[lab_t.clamp(min=0)]))
        # keep static source pixels where they are in front in the target too
        pix = paint.nonzero()
        info = {"moved_parts": int(moved.sum()), "painted_px": int(len(pix))}
        if len(pix):
            pid = pt_t[pix[:, 0], pix[:, 1]]; labs = lab_t[pix[:, 0], pix[:, 1]]
            loc = self.local[pid]
            fn = lambda Ms: torch.einsum("nij,nj->ni", torch.tensor(Ms, dtype=torch.float32, device=self.dev)[labs][:, :3, :3], loc) \
                + torch.tensor(Ms, dtype=torch.float32, device=self.dev)[labs][:, :3, 3]  # noqa: E731
            rgb, valid = self.lookup(j, fn, labs)
            col = torch.where(valid[:, None], rgb, torch.zeros_like(rgb))
            filled = valid.clone()
            used = 0
            for d in BORROW:
                if filled.all():
                    break
                jj = j + d
                if jj < 0 or jj >= rec.T or (exclude is not None and exclude[0] <= jj <= exclude[1]):
                    continue
                rem = ~filled
                fn2 = lambda Ms, rem=rem: torch.einsum("nij,nj->ni", torch.tensor(Ms, dtype=torch.float32, device=self.dev)[labs[rem]][:, :3, :3], loc[rem]) \
                    + torch.tensor(Ms, dtype=torch.float32, device=self.dev)[labs[rem]][:, :3, 3]  # noqa: E731
                rgb2, v2 = self.lookup(jj, fn2, labs[rem])
                idx = rem.nonzero()[:, 0]
                col[idx[v2]] = rgb2[v2]; filled[idx[v2]] = True; used += 1
            info["src_visible_frac"] = float(valid.float().mean()); info["filled_frac"] = float(filled.float().mean())
            out[pix[:, 0], pix[:, 1]] = col
            hole = torch.zeros((H, W), dtype=torch.bool, device=self.dev); hole[pix[~filled][:, 0], pix[~filled][:, 1]] = True
        else:
            hole = torch.zeros((H, W), dtype=torch.bool, device=self.dev)
        img = out.clamp(0, 255).byte().cpu().numpy()
        if hole.any():
            img = cv2.inpaint(img, hole.cpu().numpy().astype(np.uint8), 3, cv2.INPAINT_TELEA)
        # 1-px anti-aliasing at the boundary of painted regions
        pm = paint.cpu().numpy().astype(np.uint8)
        edge = (cv2.dilate(pm, np.ones((3, 3), np.uint8)) - cv2.erode(pm, np.ones((3, 3), np.uint8))) > 0
        blur = cv2.GaussianBlur(img, (3, 3), 0.7)
        img[edge] = blur[edge]
        info["hole_px"] = int(hole.sum())
        return img, info

    def _world_sub(self, Ms, sub):
        M = torch.tensor(Ms, dtype=torch.float32, device=self.dev)[self.gid[sub]]
        return torch.einsum("nij,nj->ni", M[:, :3, :3], self.local[sub]) + M[:, :3, 3]

    def _sam_part(self, lab_s, sam):
        """part id for every SAM pixel: the rendered part there if it is of the SAM side, else nearest such part."""
        side_of_lab = torch.tensor(np.select([self.part_side == "R", self.part_side == "L", self.part_side == "C"], [0, 1, 2], -1),
                                   device=self.dev)
        out = torch.full_like(lab_s, -1)
        for k in range(3):
            m = (sam == k).cpu().numpy()
            if not m.any():
                continue
            ok = (lab_s >= 0) & (side_of_lab[lab_s.clamp(min=0)] == k)
            okn = ok.cpu().numpy()
            if not okn.any():
                continue
            _, lbl = cv2.distanceTransformWithLabels((~okn).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
            ys, xs = np.nonzero(okn)
            idx_of = np.zeros(lbl.max() + 1, np.int64); idx_of[lbl[ys, xs]] = ys * W + xs
            src = idx_of[lbl[m]]
            flat = lab_s.view(-1)
            vals = flat[torch.tensor(src, device=self.dev)]
            mm = torch.tensor(m, device=self.dev)
            out[mm] = vals
        return out


def candidate_video(rec, z):
    """per output frame i: (source frame j, target q, target grip openness, target can) as in v3 render3."""
    cand = {"q": {"R": z["qR"], "L": z["qL"]}, "state": z["state"], "can": z["can"], "tau": z["tau"]}
    j_of = render2.source_frame_map(rec, cand["tau"])
    out = []
    for i in range(rec.T):
        ts = i + rec.lag
        q = {s: rc.interp_q(cand["q"][s], ts) for s in "RL"}
        g = render3.openness_at(cand["state"], ts)
        out.append((int(j_of[i]), q, g, render2.can_at(cand["can"], ts)))
    return out


# ============================================================================ forward mode (default)
def _nearest(okn):
    """for every pixel: flat index of the nearest True pixel of okn (cv2 distance-transform labels)."""
    _, lbl = cv2.distanceTransformWithLabels((~okn).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(okn)
    idx_of = np.zeros(lbl.max() + 1, np.int64); idx_of[lbl[ys, xs]] = ys * W + xs
    return idx_of[lbl]


class MGWForward(MGW):
    FAR, ERODE, TM_ERODE, DESPECKLE, CLEAR_CAD, PART_FILL = False, False, 21, True, 0, False   # Pringles v3-dataset config
    PLATE_ADAPT = 0   # >0: Gaussian sigma (px) of the per-frame illumination correction of the clean plate (0 = off, v3 config)   # chosen on episode 8 (ablate.py); validated on held-out episodes
    """Forward 3-D-flow warp of the real segmented pixels (SAM2 silhouettes) using mesh depth; CAD backward lookup
    only for surfaces that become visible."""

    def src_points(self, j):
        S = self.source_state(j)
        if "P" in S:
            return S
        lab = S["lab"].cpu().numpy(); dep = S["depth"].cpu().numpy(); sam = S["sam"].cpu().numpy()
        side_of_lab = np.select([self.part_side == "R", self.part_side == "L", self.part_side == "C"], [0, 1, 2], -1)
        part = np.full((H, W), -1, np.int64); depth = np.full((H, W), np.inf)
        for k in range(3):
            m = sam == k
            if not m.any():
                continue
            ok = (lab >= 0) & (side_of_lab[np.clip(lab, 0, None)] == k)
            if not ok.any():
                continue
            near = _nearest(ok)[m]
            yy, xx = np.nonzero(m)
            far = np.hypot(yy - near // W, xx - near % W) > 12        # no rendered surface nearby: depth unknown
            pl = lab.reshape(-1)[near]; dl = dep.reshape(-1)[near]
            if self.FAR:
                pl[far] = -2; dl[far] = np.inf                           # -2: robot pixel of unknown geometry (removed, not warped)
            part[m] = pl; depth[m] = dl
        v, u = np.mgrid[0:H, 0:W].astype(np.float64)
        K = self.cam.K
        z = np.where(np.isfinite(depth), depth, 1.0)
        c = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1)
        P = c @ self.cam.R.T + self.cam.t
        S["part"] = torch.tensor(part, device=self.dev); S["sdepth"] = torch.tensor(depth, dtype=torch.float32, device=self.dev)
        S["P"] = torch.tensor(P, dtype=torch.float32, device=self.dev)
        return S

    def adapted_plate(self, img, fg):
        """clean plate relit to the current frame: per-channel log-ratio frame/plate on visible background pixels (not
        robot/object, not removed), smoothed by normalised convolution (Gaussian, sigma = PLATE_ADAPT px), applied to the
        plate. Removes the dark/bright halo where the static plate's lighting/shadows differ from this frame."""
        bgm = ~torch.tensor(cv2.dilate(fg.cpu().numpy().astype(np.uint8), np.ones((9, 9), np.uint8)) > 0, device=self.dev)
        lr = (torch.log(img + 8.0) - torch.log(self.plate + 8.0)).clamp(-1.0, 1.0)
        sg = float(self.PLATE_ADAPT); r = int(3 * sg); x = torch.arange(-r, r + 1, device=self.dev, dtype=torch.float32)
        g = torch.exp(-0.5 * (x / sg) ** 2); g = g / g.sum()
        def blur(t):   # separable Gaussian on (C,H,W)
            t = Fnn.conv2d(t[:, None], g.view(1, 1, 1, -1), padding=(0, r))
            return Fnn.conv2d(t, g.view(1, 1, -1, 1), padding=(r, 0))[:, 0]
        w = bgm.float()[None]
        num = blur((lr.permute(2, 0, 1) * w)); den = blur(w)
        corr = torch.where(den > 0.05, num / den.clamp(min=1e-6), torch.zeros_like(num)).permute(1, 2, 0)
        return (self.plate + 8.0) * torch.exp(corr) - 8.0

    @torch.no_grad()
    def render(self, j, q_new, g_new, can_new, min_disp=0.5, exclude=None):
        rec = self.rec
        S = self.src_points(j)
        Mt = self.placements(q_new, g_new, can_new)
        Ms_t = torch.tensor(Mt, dtype=torch.float32, device=self.dev); Ms_s = torch.tensor(S["Ms"], dtype=torch.float32, device=self.dev)
        rel = Ms_t @ torch.linalg.inv(Ms_s)                    # per part: source -> target rigid motion
        sub = torch.arange(0, len(self.local), 50, device=self.dev)
        uo, _ = self.project(self._world_sub(S["Ms"], sub)); un, _ = self.project(self._world_sub(Mt, sub))
        disp = torch.zeros(self.CAN + 1, device=self.dev).scatter_reduce(0, self.gid[sub], (un - uo).norm(dim=1), reduce="amax")
        moved = (disp >= min_disp) & torch.tensor(self.part_side != "B", device=self.dev)
        if can_new is None:
            moved[self.CAN] = False
        part = S["part"]
        img_s = self.F[j].to(self.dev).float()
        out = img_s.clone()
        is_moved_all = ((part >= 0) & moved[part.clamp(min=0)]) | (part == -2)
        static = (part >= 0) & ~moved[part.clamp(min=0)]
        # warp only reliable interior pixels: drop unknown-geometry pixels and the 1-px mask rim (background bleed)
        core = torch.tensor(cv2.erode(((part >= 0) | (part == -2)).cpu().numpy().astype(np.uint8), np.ones((3, 3), np.uint8)) > 0, device=self.dev)
        is_moved = (part >= 0) & moved[part.clamp(min=0)] & (core if self.ERODE else True)
        # background: remove moved pixels (dilated, but never over static robot pixels)
        rm = torch.tensor(cv2.dilate(is_moved_all.cpu().numpy().astype(np.uint8), np.ones((7, 7), np.uint8)) > 0, device=self.dev) & ~static
        if self.CLEAR_CAD:
            # also clear the moved parts' SOURCE mesh silhouette (dilated): real pixels the masks missed (ghost remnants)
            ls = S["lab"]; cad = (ls >= 0) & moved[ls.clamp(min=0)]
            cad = torch.tensor(cv2.dilate(cad.cpu().numpy().astype(np.uint8), np.ones((self.CLEAR_CAD, self.CLEAR_CAD), np.uint8)) > 0, device=self.dev)
            rm |= cad & ~static
        out[rm] = self.adapted_plate(img_s, (part >= 0) | (part == -2) | rm)[rm] if self.PLATE_ADAPT else self.plate[rm]
        zbuf = torch.where(static, S["sdepth"], torch.full_like(S["sdepth"], float("inf")))
        # forward warp of moved pixels
        ys, xs = is_moved.nonzero(as_tuple=True)
        info = {"moved_parts": int(moved.sum()), "warped_px": int(len(ys))}
        hit = torch.zeros((H, W), dtype=torch.bool, device=self.dev)
        if len(ys):
            P = S["P"][ys, xs]; pr = part[ys, xs]
            Rm = rel[pr]
            Pn = torch.einsum("nij,nj->ni", Rm[:, :3, :3], P) + Rm[:, :3, 3]
            uv, z = self.project(Pn)
            col = img_s[ys, xs]
            cand_pix, cand_z, cand_c = [], [], []
            for du in (0, 1):
                for dv in (0, 1):
                    uu = torch.floor(uv[:, 0]).long() + du; vv = torch.floor(uv[:, 1]).long() + dv
                    ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H) & (z > 0.05)
                    cand_pix.append((vv * W + uu)[ok]); cand_z.append(z[ok]); cand_c.append(col[ok])
            pix = torch.cat(cand_pix); zz = torch.cat(cand_z); cc = torch.cat(cand_c)
            front = torch.full((H * W,), float("inf"), device=self.dev)
            front.scatter_reduce_(0, pix, zz, reduce="amin")
            win = zz <= front[pix] + 1e-6
            vis = win & (zz < zbuf.view(-1)[pix] - 0.005)
            acc = torch.zeros((H * W, 3), device=self.dev); cnt = torch.zeros(H * W, device=self.dev)
            acc.index_add_(0, pix[vis], cc[vis]); cnt.index_add_(0, pix[vis], torch.ones_like(zz[vis]))
            # close magnification pinholes: normalised convolution over warped samples inside the closed region
            acc2 = acc.T.reshape(1, 3, H, W); cnt2 = cnt.view(1, 1, H, W)
            hit0 = (cnt2 > 0).float()
            k5 = torch.ones(1, 1, 5, 5, device=self.dev)
            region = Fnn.max_pool2d(hit0, 5, 1, 2)
            region = -Fnn.max_pool2d(-region, 5, 1, 2)          # closing
            avg = Fnn.conv2d(acc2.view(3, 1, H, W), k5, padding=2).view(1, 3, H, W)
            wsum = Fnn.conv2d(cnt2, k5, padding=2)
            fillm = (region > 0) & (hit0 == 0) & (wsum > 0)
            h = cnt > 0
            out.view(-1, 3)[h] = acc[h] / cnt[h, None]
            fm = fillm.view(-1)
            out.view(-1, 3)[fm] = (avg.view(3, -1).T[fm] / wsum.view(-1)[fm, None])
            hit = (h | fm).view(H, W)
        # holes: target CAD silhouette of moved parts (eroded 2 px) that no warped pixel reached and no static part covers
        depth_t, lab_t, pt_t = self.zbuf(Mt)
        tmask = (lab_t >= 0) & moved[lab_t.clamp(min=0)] & (depth_t < zbuf - 0.005)
        tm = cv2.erode(tmask.cpu().numpy().astype(np.uint8), np.ones((self.TM_ERODE, self.TM_ERODE), np.uint8)) > 0
        hole = torch.tensor(tm, device=self.dev) & ~hit
        # pinholes: fill from warped neighbours first
        hn = hole.cpu().numpy(); hitn = hit.cpu().numpy()
        small = hn & (cv2.dilate(hitn.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & \
            (cv2.blur(hitn.astype(np.float32), (5, 5)) > 0.6)
        info["hole_px"] = int(hn.sum()); info["pinhole_px"] = int(small.sum())
        big = torch.tensor(hn & ~small, device=self.dev)
        # newly visible surfaces: CAD backward lookup in nearby source frames
        pixb = big.nonzero()
        filled_n = 0
        if len(pixb):
            pid = pt_t[pixb[:, 0], pixb[:, 1]]; labs = lab_t[pixb[:, 0], pixb[:, 1]]; loc = self.local[pid]
            col = torch.zeros((len(pixb), 3), device=self.dev); filled = torch.zeros(len(pixb), dtype=torch.bool, device=self.dev)
            for d in [0] + BORROW:
                if filled.all():
                    break
                jj = j + d
                if jj < 0 or jj >= rec.T or (exclude is not None and exclude[0] <= jj <= exclude[1]):
                    continue
                remi = (~filled).nonzero()[:, 0]
                fn = lambda Ms, remi=remi: torch.einsum("nij,nj->ni", torch.tensor(Ms, dtype=torch.float32, device=self.dev)[labs[remi]][:, :3, :3], loc[remi]) \
                    + torch.tensor(Ms, dtype=torch.float32, device=self.dev)[labs[remi]][:, :3, 3]  # noqa: E731
                rgb, v = self.lookup(jj, fn, labs[remi])
                col[remi[v]] = rgb[v]; filled[remi[v]] = True
            out[pixb[filled][:, 0], pixb[filled][:, 1]] = col[filled]
            filled_n = int(filled.sum())
            big[pixb[filled][:, 0], pixb[filled][:, 1]] = False
        info["borrowed_px"] = filled_n
        img = out.clamp(0, 255).byte().cpu().numpy()
        rest = small | big.cpu().numpy()
        if self.PART_FILL:
            # part-aware fill: every pixel of a moved part's target silhouette that no warped / borrowed pixel reached
            # (incl. the rim excluded by TM_ERODE) takes the colour of the nearest warped pixel of the SAME part
            full = (tmask.cpu().numpy() & ~hit.cpu().numpy()) & ~(~rest & tm)
            full &= ~hit.cpu().numpy()
            if full.any() and hit.any():
                labn = lab_t.cpu().numpy(); hitn = hit.cpu().numpy(); filled_any = np.zeros_like(full)
                for pid in np.unique(labn[full]):
                    src_ok = hitn & (labn == pid)
                    if src_ok.sum() < 20:
                        continue
                    sel = full & (labn == pid)
                    near = _nearest(src_ok)[sel]
                    img.reshape(-1, 3)[np.flatnonzero(sel)] = img.reshape(-1, 3)[near]
                    filled_any |= sel
                info["part_filled_px"] = int(filled_any.sum())
                rest = rest & ~filled_any
                if filled_any.any():
                    blur = cv2.GaussianBlur(img, (5, 5), 1.2); img[filled_any] = blur[filled_any]
        if rest.any():
            img = cv2.inpaint(img, rest.astype(np.uint8), 3, cv2.INPAINT_TELEA)
        info["inpainted_px"] = int(rest.sum())
        wm = (hit.cpu().numpy() | rest).astype(np.uint8)
        if self.DESPECKLE:
            # isolated outliers inside / along the warped region (background pixels dragged along): 3x3 median
            reg = cv2.dilate(wm, np.ones((5, 5), np.uint8)) > 0
            med = cv2.medianBlur(img, 3)
            dev_ = np.abs(img.astype(np.int16) - med.astype(np.int16)).max(2) > 40
            sel = reg & dev_
            img[sel] = med[sel]
            info["despeckled_px"] = int(sel.sum())
        # 1-px anti-aliasing at warped-region borders
        edge = (cv2.dilate(wm, np.ones((3, 3), np.uint8)) - cv2.erode(wm, np.ones((3, 3), np.uint8))) > 0
        blur = cv2.GaussianBlur(img, (3, 3), 0.7)
        img[edge] = blur[edge]
        return img, info
