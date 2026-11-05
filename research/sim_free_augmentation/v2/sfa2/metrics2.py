"""Image-level evidence for generated clips (automatic measurements, never ground truth).

Fixes over v1 (Codex review): the flow-warp error is restricted to the requested region AND to forward-backward-
consistent, in-bounds pixels; edited and background regions are reported separately; keypoint/lid metrics report
coverage, and missing observations never count as successes. Independent image cues used here: the Gate 1 HSV lid
tracker (no robot state), an HSV red-body mask for the can axis (not SAM2, not the renderer's masks) and
plate-difference foreground (prompt-free).
"""

from __future__ import annotations

import sys

import cv2
import numpy as np

from .recon import REPO, V1, rb, rc

sys.path.insert(0, str(V1))
sys.path.insert(0, str(V1 / "scripts"))
sys.path.insert(0, str(REPO / "research/contact_aware_augmentation/gate1"))
from sfa import metrics as M, track as trk  # noqa: E402

PROMPT_LINKS = ("link4", "link5", "link6", "link7", "hand", "left_finger", "right_finger")


def warp_error_regions(frames, flows, regions: dict, fb_tol=1.5) -> dict:
    T, H, W = frames.shape[:3]
    gx, gy = np.meshgrid(np.arange(W), np.arange(H))
    acc = {k: [0.0, 0] for k in regions}
    per = {k: [] for k in regions}
    for t in range(1, T):
        b, f = flows["bwd"][t - 1], flows["fwd"][t - 1]
        mx, my = (gx + b[..., 0]).astype(np.float32), (gy + b[..., 1]).astype(np.float32)
        inb = (mx >= 0) & (mx <= W - 1) & (my >= 0) & (my <= H - 1)
        fb = b + cv2.remap(f, mx, my, cv2.INTER_LINEAR)
        ok = inb & (np.linalg.norm(fb, axis=2) < fb_tol)
        w = trk.warp_by_flow(frames[t - 1], b)
        d = np.abs(frames[t].astype(np.float32) - w.astype(np.float32)).mean(2)
        for k, R in regions.items():
            m = ok & R[t]
            n = int(m.sum())
            if n:
                acc[k][0] += float(d[m].sum())
                acc[k][1] += n
                per[k].append(float(d[m].mean()))
    return {k: {"mean": acc[k][0] / acc[k][1] if acc[k][1] else None, "p95_frame": float(np.percentile(per[k], 95)) if per[k] else None,
                "valid_px": acc[k][1]} for k in regions}


def projected(robot, cam, q, grip, cans, T, H=480, W=640, links=PROMPT_LINKS, with_can=True):
    out = np.zeros((T, H, W), np.uint8)
    P = rc.cylinder_points([0, 0, 1.0], 24, 6)
    for t in range(T):
        for name, side, m, z, hull in rb.link_silhouettes(robot, cam, q["R"][t], q["L"][t], grip["R"][t], grip["L"][t]):
            if name.split("_", 1)[1] in links:
                out[t] |= m
        if with_can and cans[t] is not None:
            uv, z = cam.project((cans[t][:3, :3] @ P.T).T + cans[t][:3, 3])
            if (z > 0.05).all():
                cv2.fillConvexPoly(out[t], cv2.convexHull(uv.astype(np.float32)).astype(np.int32), 1)
    return out


def lid_track(frames, cam, cans):
    import lid_tracker as lt
    uv0, _ = cam.project(cans[0][:3, 3][None])
    tr = lt.track(frames, uv0[0])
    return tr.uv, tr.ok()


def lid_error(uv, ok, cam, cans, idx) -> dict:
    idx = list(idx)
    pred = cam.project(np.stack([cans[t][:3, 3] for t in idx]))[0]
    okk = ok[idx] & np.isfinite(uv[idx]).all(1)
    e = np.linalg.norm(uv[idx] - pred, axis=1)[okk]
    return {"coverage": float(okk.mean()) if idx else 0.0, "n": int(okk.sum()),
            "median_px": float(np.median(e)) if len(e) else None, "p90_px": float(np.percentile(e, 90)) if len(e) else None}


def red_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return ((hsv[..., 0] < 8) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 120) & (hsv[..., 2] > 60)


def can_axis_error(frames, cam, cans, idx, min_px=300) -> dict:
    """angle between the principal axis of the red can body (HSV, inside the dilated projected can) and the projected
    commanded can axis. Measures image/object-orientation agreement in 2D; mod 180 deg."""
    P = rc.cylinder_points([0, 0, 1.0], 24, 6)
    errs, n_try = [], 0
    for t in idx:
        c = cans[t]
        uv, z = cam.project((c[:3, :3] @ P.T).T + c[:3, 3])
        if (z < 0.05).any():
            continue
        n_try += 1
        reg = np.zeros(frames.shape[1:3], np.uint8)
        cv2.fillConvexPoly(reg, cv2.convexHull(uv.astype(np.float32)).astype(np.int32), 1)
        reg = cv2.dilate(reg, np.ones((31, 31), np.uint8)) > 0
        m = red_mask(frames[t]) & reg
        yy, xx = np.nonzero(m)
        if len(xx) < min_px:
            continue
        X = np.stack([xx, yy], 1).astype(float)
        X -= X.mean(0)
        w, V = np.linalg.eigh(X.T @ X / len(X))
        if w[1] < 1.5 * w[0]:          # blob not elongated enough to define an axis
            continue
        d_img = V[:, 1]
        e2, _ = cam.project(np.stack([c[:3, 3], c[:3, 3] + rc.CAN_HEIGHT * c[:3, 2]]))
        d_cmd = e2[1] - e2[0]
        d_cmd /= np.linalg.norm(d_cmd) + 1e-9
        ang = np.degrees(np.arccos(np.clip(abs(d_img @ d_cmd), 0, 1)))
        errs.append(float(ang))
    return {"coverage": len(errs) / max(1, n_try), "n": len(errs), "median_deg": float(np.median(errs)) if errs else None,
            "p90_deg": float(np.percentile(errs, 90)) if errs else None}


def ghost(frames, plate, proj_old, proj_new, idx) -> dict:
    """fraction of pixels that the robot/can LEFT (old silhouette minus dilated new silhouette) still differing from
    the background plate (should be ~noise floor); noise floor = same statistic on far background pixels."""
    k = np.ones((9, 9), np.uint8)
    g, nf = [], []
    for t in idx:
        fg = M.foreground(frames[t], plate)
        O = (proj_old[t] > 0) & ~(cv2.dilate(proj_new[t], k) > 0)
        if O.sum() > 200:
            g.append(float(fg[O].mean()))
        far = ~(cv2.dilate(((proj_old[t] > 0) | (proj_new[t] > 0)).astype(np.uint8), np.ones((41, 41), np.uint8)) > 0)
        nf.append(float(fg[far].mean()))
    return {"ghost_frac_median": float(np.median(g)) if g else None, "ghost_frac_p90": float(np.percentile(g, 90)) if g else None,
            "n": len(g), "background_noise_frac": float(np.median(nf)) if nf else None}


def lag_scan(robot, cam, state16, X_R, lid_uv, lid_ok, idx, lags=np.arange(-4.0, 1.01, 0.25)) -> dict:
    """image-to-state lag from exported state + image only: lid predicted by FK(state(i + l)) . X_R in the giver-hold
    window vs the HSV lid track of the image. Returns the error curve and its argmin."""
    g = rb.Robot.split(state16)
    idx = [i for i in idx if lid_ok[i]]
    curve = {}
    for l in lags:
        e = []
        for i in idx:
            robot.fk(rc.interp_q(g["R"], i + l), rc.interp_q(g["L"], i + l))
            p = (robot.frame(robot.tcp["R"]) @ X_R)[:3, 3]
            e.append(np.linalg.norm(cam.project(p[None])[0][0] - lid_uv[i]))
        curve[float(l)] = float(np.median(e)) if e else np.inf
    best = min(curve, key=curve.get)
    return {"lag": best, "curve": curve, "n": len(idx)}


def label_consistency(frames, plate, proj_own, proj_other, idx) -> dict:
    """prompt-free: does the image agree better with its own exported state than with an alternative label set?"""
    r = M.pose_discrimination(frames, plate, proj_own, proj_other, idx)
    return r


def keypoint_follow(uv_o, vis_o, uv_g, vis_g, tgt, layer_names, windows: dict) -> dict:
    out = {}
    for wname, (a, b) in windows.items():
        sl = slice(max(0, a), b)
        res = {}
        for grp in ("R", "L", "can"):
            idx = [i for i, l in enumerate(layer_names) if l == grp or l.startswith(grp + "_")]
            if not idx:
                continue
            ok_o = vis_o[sl][:, idx] & np.isfinite(tgt[sl][:, idx]).all(-1)
            ok = ok_o & vis_g[sl][:, idx] & np.isfinite(uv_g[sl][:, idx]).all(-1)
            e = np.linalg.norm(uv_g[sl][:, idx] - tgt[sl][:, idx], axis=-1)[ok]
            res[grp] = {"coverage": float(ok.sum() / max(1, ok_o.sum())), "n": int(ok.sum()),
                        "median_px": float(np.median(e)) if len(e) else None,
                        "p90_px": float(np.percentile(e, 90)) if len(e) else None}
        out[wname] = res
    return out
