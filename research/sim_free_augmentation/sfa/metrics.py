"""Stage 3 evidence metrics for generated clips. All image-based numbers are automatic measurements on the clips;
none is a ground-truth label, and none says anything about real-world executability or force."""

from __future__ import annotations

import cv2
import numpy as np

from . import track as trk


def temporal_warp_error(frames: np.ndarray, flows: dict, region: np.ndarray | None = None) -> dict:
    """mean |G_t - warp(G_{t-1}, flow t->t-1)| over forward-backward-consistent pixels (8-bit units), and the
    plain frame-difference flicker inside `region` (T,H,W) bool (e.g. static background)."""
    T, H, W = frames.shape[:3]
    errs, flick = [], []
    gx, gy = np.meshgrid(np.arange(W), np.arange(H))
    for t in range(1, T):
        b = flows["bwd"][t - 1]            # t -> t-1
        f = flows["fwd"][t - 1]            # t-1 -> t
        mx, my = (gx + b[..., 0]).astype(np.float32), (gy + b[..., 1]).astype(np.float32)
        fb = b + cv2.remap(f, mx, my, cv2.INTER_LINEAR)
        ok = np.linalg.norm(fb, axis=2) < 1.5
        w = trk.warp_by_flow(frames[t - 1], b)
        d = np.abs(frames[t].astype(np.float32) - w.astype(np.float32)).mean(2)
        errs.append(float(d[ok].mean()) if ok.any() else np.nan)
        if region is not None:
            r = region[t] & region[t - 1]
            if r.any():
                flick.append(float(np.abs(frames[t].astype(np.float32) - frames[t - 1].astype(np.float32)).mean(2)[r].mean()))
    out = {"warp_err_mean": float(np.nanmean(errs)), "warp_err_p95": float(np.nanpercentile(errs, 95)),
           "warp_err_per_frame": errs}
    if flick:
        out["static_flicker_mean"] = float(np.mean(flick))
    return out


def follow_error(uv_gen, vis_gen, uv_target, valid) -> dict:
    """keypoint trajectory following: tracked position in the generated clip vs its commanded (target) position."""
    ok = vis_gen & valid & np.isfinite(uv_gen).all(-1) & np.isfinite(uv_target).all(-1)
    e = np.linalg.norm(uv_gen - uv_target, axis=-1)[ok]
    if not len(e):
        return {"n": 0}
    return {"n": int(len(e)), "median_px": float(np.median(e)), "p90_px": float(np.percentile(e, 90)),
            "mean_px": float(e.mean()), "frac_within_5px": float((e < 5).mean())}


def mask_iou(a, b):
    a, b = a > 0, b > 0
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else np.nan


def silhouette_alignment(sam_masks: dict, proj_masks: dict, frames_idx) -> dict:
    """IoU between SAM2 arm masks of a clip and the projected URDF silhouettes of the state it should show."""
    out = {}
    for s in "RL":
        v = [mask_iou(sam_masks[s][t], proj_masks[s][t]) for t in frames_idx]
        out[s] = {"median_iou": float(np.nanmedian(v)), "p10_iou": float(np.nanpercentile(v, 10))}
    return out


def lid_jumps(uv: np.ndarray, ok: np.ndarray) -> dict:
    """frame-to-frame lid displacement: on `ok` frames, and on every frame with a finite position (incl. uncertain)."""
    d = np.linalg.norm(np.diff(uv, axis=0), axis=1)
    m = ok[1:] & ok[:-1]
    f = np.isfinite(d)
    return {"max_step_px_ok": float(d[m].max()) if m.any() else None,
            "max_step_px_any": float(d[f].max()) if f.any() else None,
            "lost_frames": int((~np.isfinite(uv).all(1)).sum())}


def foreground(frame, plate, thr=30, k=5):
    d = np.abs(frame.astype(np.int16) - plate.astype(np.int16)).max(2)
    m = (d > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)) > 0


def pose_discrimination(frames, plate, proj_cmd, proj_alt, idx) -> dict:
    """prompt-free robot/state alignment: IoU of the plate-difference foreground with the projected silhouettes of the
    state the clip SHOULD show (cmd) vs the state it should NOT show (alt). A clip whose robot pixels moved to the
    commanded pose has IoU(cmd) > IoU(alt) on the edited frames."""
    a = [mask_iou(foreground(frames[t], plate), proj_cmd[t]) for t in idx]
    b = [mask_iou(foreground(frames[t], plate), proj_alt[t]) for t in idx]
    a, b = np.array(a), np.array(b)
    return {"iou_commanded_median": float(np.nanmedian(a)), "iou_alternative_median": float(np.nanmedian(b)),
            "frac_frames_commanded_better": float(np.mean(a > b)), "n_frames": int(len(idx))}


def window_follow(uv_gen, vis_gen, uv_target, valid, t0, t1):
    sl = slice(max(0, t0), t1)
    return follow_error(uv_gen[sl], vis_gen[sl], uv_target[sl], valid[sl])
