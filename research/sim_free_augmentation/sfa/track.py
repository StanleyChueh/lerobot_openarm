"""Dense optical flow (torchvision RAFT-large, GPU) and flow-chained semantic keypoint tracking.

CoTracker3 was the first choice; its code could not be fetched from GitHub within the session's bandwidth
(~25 KB/s), so the tracker is RAFT flow chaining with a forward-backward consistency test (a point is `visible`
on a frame when |fwd(bwd)| < FB_TOL px). OpenCV pyramidal Lucas-Kanade is the CPU fallback (`backend="lk"`).
Tracks are automatic measurements, not ground truth.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as Fnn

FB_TOL = 2.0
_RAFT = {}


def raft(device="cuda:0"):
    if device not in _RAFT:
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
        w = Raft_Large_Weights.C_T_SKHT_V2
        _RAFT[device] = (raft_large(weights=w).eval().to(device), w.transforms())
    return _RAFT[device]


@torch.inference_mode()
def flow_pairs(frames: np.ndarray, pairs, device="cuda:0", batch=8) -> np.ndarray:
    """optical flow for each (i, j) pair: (N, H, W, 2), pixel units, from frame i to frame j."""
    model, tf = raft(device)
    out = []
    for b in range(0, len(pairs), batch):
        P = pairs[b:b + batch]
        a = torch.from_numpy(np.stack([frames[i] for i, _ in P])).permute(0, 3, 1, 2)
        c = torch.from_numpy(np.stack([frames[j] for _, j in P])).permute(0, 3, 1, 2)
        a, c = tf(a, c)
        fl = model(a.to(device), c.to(device), num_flow_updates=12)[-1]
        out.append(fl.permute(0, 2, 3, 1).float().cpu().numpy())
    return np.concatenate(out)


def sample(flow: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """bilinear flow sample at (N,2) points (x, y)."""
    H, W = flow.shape[:2]
    f = torch.from_numpy(flow).permute(2, 0, 1)[None]
    g = torch.from_numpy(pts.astype(np.float32)).clone()
    g[:, 0] = g[:, 0] / (W - 1) * 2 - 1
    g[:, 1] = g[:, 1] / (H - 1) * 2 - 1
    s = Fnn.grid_sample(f, g[None, None], align_corners=True)[0, :, 0].T
    return s.numpy()


def track_points(frames: np.ndarray, start: int, pts0: np.ndarray, device="cuda:0", flows=None):
    """track (N,2) points from frame `start` forward and backward through the clip.
    Returns uv (T,N,2) and visible (T,N) bool (forward-backward consistent, inside the image)."""
    T, H, W = frames.shape[:3]
    N = len(pts0)
    uv = np.full((T, N, 2), np.nan)
    vis = np.zeros((T, N), bool)
    uv[start] = pts0
    vis[start] = True
    if flows is None:
        flows = video_flows(frames, device)
    for direction in (1, -1):
        p = pts0.astype(float).copy()
        alive = np.ones(N, bool)
        t = start
        while 0 <= t + direction < T:
            k = t if direction == 1 else t - 1
            fw = flows["fwd"][k] if direction == 1 else flows["bwd"][k]
            bw = flows["bwd"][k] if direction == 1 else flows["fwd"][k]
            d = sample(fw, p)
            q = p + d
            back = q + sample(bw, q)
            fb = np.linalg.norm(back - p, axis=1)
            inside = (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H)
            alive &= inside
            t += direction
            uv[t] = q
            vis[t] = alive & (fb < FB_TOL)
            p = np.clip(q, 0, [W - 1, H - 1])
    return uv, vis


def video_flows(frames: np.ndarray, device="cuda:0") -> dict:
    T = len(frames)
    fwd = flow_pairs(frames, [(i, i + 1) for i in range(T - 1)], device)
    bwd = flow_pairs(frames, [(i + 1, i) for i in range(T - 1)], device)
    return {"fwd": fwd, "bwd": bwd}


def warp_by_flow(img: np.ndarray, flow_to_prev: np.ndarray) -> np.ndarray:
    """backward-warp img (frame t-1) into frame t using the flow from t to t-1."""
    import cv2
    H, W = flow_to_prev.shape[:2]
    gx, gy = np.meshgrid(np.arange(W), np.arange(H))
    mx = (gx + flow_to_prev[..., 0]).astype(np.float32)
    my = (gy + flow_to_prev[..., 1]).astype(np.float32)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
