"""Dense point sampling of the URDF visual meshes + a GPU two-pass point-splat z-buffer (no simulator, no physics)."""
from __future__ import annotations
import numpy as np, torch
from .robot import read_mesh


def sample_mesh(V, spacing, rng):
    tri = V.reshape(-1, 3, 3)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    n = int(max(200, area.sum() / spacing ** 2))
    k = rng.choice(len(tri), size=n, p=area / area.sum()); r1, r2 = rng.random(n), rng.random(n); s = np.sqrt(r1)
    a, b, c = 1 - s, s * (1 - r2), s * r2
    return a[:, None] * tri[k, 0] + b[:, None] * tri[k, 1] + c[:, None] * tri[k, 2]


class Geometry:
    """robot visual geometries (one part per geometry) sampled every `spacing` metres."""
    def __init__(self, robot, device="cuda:0", spacing=0.0015):
        rng = np.random.default_rng(0); self.robot, self.dev = robot, device
        pts, gid = [], []
        for i, g in enumerate(robot.vis.geometryObjects):
            P = sample_mesh(read_mesh(g.meshPath) * g.meshScale, spacing, rng); pts.append(P); gid.append(np.full(len(P), i))
        self.side = list(robot.geom_arm); self.ng = len(pts)
        self.local = torch.tensor(np.concatenate(pts), dtype=torch.float32, device=device)
        self.gid = torch.tensor(np.concatenate(gid), dtype=torch.long, device=device)

    def placements(self, q, g):
        self.robot.fk(q, g)
        return np.stack([M.homogeneous for M in self.robot.vd.oMg])

    def world(self, Ms):
        M = torch.tensor(Ms, dtype=torch.float32, device=self.dev)[self.gid]
        return torch.einsum("nij,nj->ni", M[:, :3, :3], self.local) + M[:, :3, 3]


def splat(Pw, ids, cam, foot=1, dev="cuda:0", hole_fill=True):
    if hole_fill:
        dc, _, _ = _splat(Pw, ids, cam, 2, dev)
        d, lab, pt = _splat(Pw, ids, cam, foot, dev, prior=dc)
        return fill_holes(d, lab, pt, ids)
    return _splat(Pw, ids, cam, foot, dev)


def _splat(Pw, ids, cam, foot=1, dev="cuda:0", prior=None):
    H, W = cam.H, cam.W
    R = torch.tensor(cam.R, dtype=torch.float32, device=dev); t = torch.tensor(cam.t, dtype=torch.float32, device=dev)
    c = (Pw - t) @ R; z = c[:, 2]; K = cam.K
    u = K[0, 0] * c[:, 0] / z + K[0, 2]; v = K[1, 1] * c[:, 1] / z + K[1, 2]
    ok = z > 0.05; ui, vi = torch.round(u).long(), torch.round(v).long(); idx = torch.arange(len(Pw), device=dev)
    us, vs, zs, js = [], [], [], []
    for du in range(-foot, foot + 1):
        for dv in range(-foot, foot + 1):
            uu, vv = ui + du, vi + dv; m = ok & (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            us.append(uu[m]); vs.append(vv[m]); zs.append(z[m]); js.append(idx[m])
    uu, vv, zz, jj = torch.cat(us), torch.cat(vs), torch.cat(zs), torch.cat(js); pix = vv * W + uu
    if prior is not None:
        keep = zz <= prior.view(-1)[pix] + 0.01; pix, zz, jj = pix[keep], zz[keep], jj[keep]
    depth = torch.full((H * W,), float("inf"), device=dev); depth.scatter_reduce_(0, pix, zz, reduce="amin")
    win = zz <= depth[pix] + 1e-7
    point = torch.full((H * W,), -1, dtype=torch.long, device=dev); point[pix[win]] = jj[win]
    lab = torch.full((H * W,), -1, dtype=torch.long, device=dev); has = point >= 0; lab[has] = ids[point[has]]
    return depth.view(H, W), lab.view(H, W), point.view(H, W)


def fill_holes(d, lab, pt, ids, it=2):
    H, W = d.shape
    for _ in range(it):
        empty = pt < 0
        if not empty.any():
            break
        dp = torch.nn.functional.pad(d[None, None], (1, 1, 1, 1), value=float("inf"))
        pp = torch.nn.functional.pad(pt[None, None].float(), (1, 1, 1, 1), value=-1)
        D = torch.nn.functional.unfold(dp, 3).view(9, H, W); P = torch.nn.functional.unfold(pp, 3).view(9, H, W)
        n = (P >= 0).sum(0); best = D.argmin(0); cand = torch.gather(P, 0, best[None])[0].long()
        m = empty & (n >= 5) & (cand >= 0)
        pt = torch.where(m, cand, pt); d = torch.where(m, torch.gather(D, 0, best[None])[0], d)
    lab = torch.full_like(pt, -1); has = pt >= 0; lab[has] = ids[pt[has]]
    return d, lab, pt
