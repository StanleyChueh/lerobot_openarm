"""Simulation-free wrist-camera view synthesis by depth-based reprojection of the REAL wrist video (no simulator,
no learned generator).

Per source episode and wrist camera (cached):
  * camera poses T_j = FK_link7(q(j + lag_w)) . X (X = mount, lag_w = measured per-camera lag);
  * per source frame j a depth map from RAFT flow to a neighbour frame k (|k - j| = 3) triangulated with the KNOWN
    relative pose inv(T_j) T_k (midpoint method; parallax < 0.4 deg -> 'far', 2.5 m, low confidence);
  * 'attached' pixels: measured flow < 1 px while the pure-rotation flow predicted for that pixel is > 3 px
    (own finger pads, and the can while THIS hand holds it) - they move with the camera;
  * can pixels: convex hulls of the large HSV-red components (moving object, excluded from the static-scene splat).
Per generated frame i (depicting candidate state time i + lag_w; source frame j = round(tau(i + lag_w) - lag_w)):
  1. static scene: forward splat (z-buffer) of source frame j into the new pose T'_i = FK(q_new) . X;
  2. holes: splat the two source frames whose camera poses are closest to T'_i (multiview fill), then OpenCV
     Telea inpainting for what remains;
  3. can not attached to this camera: rigid homography of its pixels from (T_j, can_src) to (T'_i, can_new);
  4. attached pixels: copied unchanged (the hand-camera-object relation is preserved by both generators).
A label map records, per pixel: 0 reprojected (same frame), 1 attached copy, 2 rigid can warp, 3 multiview fill,
4 inpainted, 5 unknown (inpainted region farther than 25 px from any known pixel).
"""

from __future__ import annotations

import cv2
import numpy as np

from sfa3.recon3 import rb
from sfa.recover import interp_q, cylinder_points

from . import wristcam as wc

FAR = 2.5
CLOSED = 0.7          # gripper openness (state) below which the gripper is closed - same rule as the v3 verifier
LAB = {"reprojected": 0, "attached": 1, "can_warp": 2, "multiview": 3, "inpainted": 4, "unknown": 5}


def can_mask(rgb, min_area=800):
    """image-only can mask: convex hull of each large HSV-red component (fills the white label text and printing
    inside the can body). The yellow-lid range was dropped: it also matches the beige wall in the wrist views."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    red = (((hsv[..., 0] < 8) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 120) & (hsv[..., 2] > 60)).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(red)
    keep = np.zeros(red.shape, np.uint8)
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] >= min_area:
            ys, xs = np.nonzero(lab == k)
            hull = cv2.convexHull(np.stack([xs, ys], 1).astype(np.int32))
            cv2.fillConvexPoly(keep, hull, 1)
    return cv2.dilate(keep, np.ones((7, 7), np.uint8)) > 0


def poses(robot, g, side, X, lag, T):
    return [wc.cam_pose(robot, interp_q(g["R"], t + lag), interp_q(g["L"], t + lag), side, X) for t in range(T)]


def triangulate(flow, A, Kc, stride=2):
    """depth (H, W) for camera j from flow j->k and A = inv(T_j) T_k; returns depth, parallax_ok, attached."""
    H, W = flow.shape[:2]
    gy, gx = np.mgrid[0:H:stride, 0:W:stride]
    x = np.stack([gx, gy, np.ones_like(gx)], -1).reshape(-1, 3).astype(np.float64)
    f = flow[gy, gx].reshape(-1, 2)
    Ki = np.linalg.inv(Kc)
    rj = x @ Ki.T
    xk = x.copy()
    xk[:, :2] += f
    rk = xk @ Ki.T
    R, c = A[:3, :3], A[:3, 3]
    dk = rk @ R.T
    a = (rj * rj).sum(1)
    b = (rj * dk).sum(1)
    cc = (dk * dk).sum(1)
    d = rj @ c
    e = dk @ c
    den = a * cc - b * b
    lam = (d * cc - b * e) / np.where(np.abs(den) < 1e-12, 1e-12, den)
    mu = (b * d - a * e) / np.where(np.abs(den) < 1e-12, 1e-12, den)
    cosang = b / np.sqrt(a * cc)
    par = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    rot = (rj @ R) @ Kc.T                               # pure-rotation prediction of the pixel in camera k
    rot = rot[:, :2] / rot[:, 2:3]
    rot_flow = np.linalg.norm(rot - x[:, :2], axis=1)
    attached = (np.linalg.norm(f, axis=1) < 1.0) & (rot_flow > 3.0)
    ok = (par > 0.4) & (lam > 0.08) & (mu > 0) & (lam < FAR * 1.6) & ~attached
    depth = np.where(ok, lam, FAR)
    up = lambda v: cv2.resize(v.reshape(gy.shape).astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)  # noqa: E731
    dep = cv2.medianBlur(up(depth), 5)
    return dep, up(ok.astype(np.float32)) > 0.5, up(attached.astype(np.float32)) > 0.5


def splat(img, depth, M, Kc, use, H, W):
    """forward-project pixels `use` of img (depth in its camera) by M = inv(T_new) T_src; nearest-wins z-buffer."""
    ys, xs = np.nonzero(use)
    if not len(xs):
        return np.zeros_like(img), np.zeros((H, W), bool)
    Ki = np.linalg.inv(Kc)
    r = np.stack([xs, ys, np.ones_like(xs)], 1).astype(np.float64) @ Ki.T
    P = r * depth[ys, xs][:, None]
    Q = P @ M[:3, :3].T + M[:3, 3]
    z = Q[:, 2]
    uv = (Q @ Kc.T)
    u, v = uv[:, 0] / np.maximum(z, 1e-6), uv[:, 1] / np.maximum(z, 1e-6)
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    ok = (z > 0.05) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui, vi, z, src = ui[ok], vi[ok], z[ok], (ys[ok], xs[ok])
    order = np.argsort(-z)                               # far first, near overwrites
    out = np.zeros_like(img)
    valid = np.zeros((H, W), bool)
    out[vi[order], ui[order]] = img[src[0][order], src[1][order]]
    valid[vi[order], ui[order]] = True
    return out, valid


def close_small_holes(img, valid):
    hole = ~valid
    closed = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    small = hole & closed
    if small.any():
        img = cv2.inpaint(img, small.astype(np.uint8), 2, cv2.INPAINT_TELEA)
    return img, valid | small


class WristSource:
    """precomputed per-episode, per-camera geometry of the SOURCE wrist video."""

    def __init__(self, ep, side, robot, flows, X=None, lag=None):
        self.side = side
        self.F = ep.frames[wc.CAM_OF[side]]
        self.T = len(self.F)
        self.X = wc.mount(side) if X is None else X
        self.lag = wc.lag(side) if lag is None else lag
        self.K = wc.K(side)
        g = rb.Robot.split(ep.state)
        self.g = g
        self.Tc = poses(robot, g, side, self.X, self.lag, self.T)
        self.depth, self.ok, self.attached, self.can = [], [], [], []
        for j in range(self.T):
            k = j + 3 if j + 3 < self.T else j - 3
            A = np.linalg.inv(self.Tc[j]) @ self.Tc[k]
            d, o, a = triangulate(flows[(j, k)], A, self.K)
            self.depth.append(d)
            self.ok.append(o)
            self.attached.append(a)
            self.can.append(can_mask(self.F[j]))
        self.pos = np.array([T[:3, 3] for T in self.Tc])

    def nearest_frames(self, T_new, exclude, n=2, window=40):
        cand = [k for k in range(max(0, exclude - window), min(self.T, exclude + window)) if k != exclude]
        rot = np.array([np.linalg.norm(cv2.Rodrigues(self.Tc[k][:3, :3].T @ T_new[:3, :3])[0]) for k in cand])
        dist = np.linalg.norm(self.pos[cand] - T_new[:3, 3], axis=1) + 0.1 * rot
        return [cand[i] for i in np.argsort(dist)[:n]]


def render_frame(src: WristSource, j, T_new, can_src, can_new, can_attached=False):
    """can_attached: this hand's gripper holds the can in the source frame AND in the candidate (rigidly attached to
    the camera) -> its pixels are copied like the finger pads."""
    H, W = src.F.shape[1:3]
    img = src.F[j]
    M = np.linalg.inv(T_new) @ src.Tc[j]
    lab = np.full((H, W), LAB["unknown"], np.uint8)
    same_can = can_src is None or can_new is None or np.abs(np.linalg.inv(src.Tc[j]) @ can_src - np.linalg.inv(T_new) @ can_new).max() < 1e-3
    if np.abs(M - np.eye(4)).max() < 1e-9 and same_can:
        lab[:] = LAB["reprojected"]
        return img.copy(), lab
    att = src.attached[j] | (src.can[j] if can_attached else False)
    canm = src.can[j] & ~att
    static = ~att & ~canm
    out, valid = splat(img, src.depth[j], M, src.K, static, H, W)
    out, valid = close_small_holes(out, valid)
    lab[valid] = LAB["reprojected"]
    # multiview fill from the nearest source camera poses
    for k in src.nearest_frames(T_new, j):
        if (~valid).sum() == 0:
            break
        Mk = np.linalg.inv(T_new) @ src.Tc[k]
        o2, v2 = splat(src.F[k], src.depth[k], Mk, src.K, ~src.attached[k] & ~src.can[k], H, W)
        o2, v2 = close_small_holes(o2, v2)
        fill = v2 & ~valid
        out[fill] = o2[fill]
        lab[fill] = LAB["multiview"]
        valid |= fill
    # inpaint the rest (labelled; far from any known pixel -> unknown)
    hole = ~valid
    if hole.any():
        # inpaint at half resolution (cost), write back only into the hole pixels
        small = cv2.resize(out, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        hs = cv2.resize(hole.astype(np.uint8), (W // 2, H // 2), interpolation=cv2.INTER_NEAREST)
        fill = cv2.resize(cv2.inpaint(small, hs, 3, cv2.INPAINT_TELEA), (W, H), interpolation=cv2.INTER_LINEAR)
        out[hole] = fill[hole]
        dist = cv2.distanceTransform(hole.astype(np.uint8), cv2.DIST_L2, 3)
        lab[hole & (dist <= 25)] = LAB["inpainted"]
        lab[hole & (dist > 25)] = LAB["unknown"]
    # rigid can layer (not attached to this camera)
    if canm.any() and can_src is not None and can_new is not None:
        P = cylinder_points([0, 0, 1.0], 24, 6)
        cs, cn = wc.camera(src.side, src.Tc[j]), wc.camera(src.side, T_new)
        uo, zo = cs.project((can_src[:3, :3] @ P.T).T + can_src[:3, 3])
        un, zn = cn.project((can_new[:3, :3] @ P.T).T + can_new[:3, 3])
        front = (zo > 0.05) & (zn > 0.05)
        if front.sum() >= 8:
            Hm, _ = cv2.findHomography(uo[front].astype(np.float32), un[front].astype(np.float32), 0)
            if Hm is not None:
                wimg = cv2.warpPerspective(img, Hm, (W, H), flags=cv2.INTER_LINEAR)
                wm = cv2.warpPerspective(canm.astype(np.uint8), Hm, (W, H), flags=cv2.INTER_NEAREST) > 0
                out[wm] = wimg[wm]
                lab[wm] = LAB["can_warp"]
    # attached pixels stay where they are
    out[att] = img[att]
    lab[att] = LAB["attached"]
    return out, lab
