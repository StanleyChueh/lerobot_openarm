"""v3 rendering: v2's layered warp (split lid/body can), with the sampling contract made explicit and consistent:

* generated frame i depicts the candidate at state time i + lag;
* its arm geometry AND gripper aperture are the candidate's planned state interpolated at i + lag (v2 used the
  source frame's aperture for both poses, up to 0.026 openness off on timing warps);
* the source pose in source frame j = round(tau(i + lag) - lag) is the source state at j + lag (source aperture);
* for new initial positions the standing can is moved by the can homography from its source table pose to the
  shifted table pose (pixels of the source frame; the vacated table comes from the background plate).
"""

from __future__ import annotations

import cv2
import numpy as np

from .recon3 import rb
from sfa2 import render2  # noqa: E402
from sfa2.recon import rc, rd  # noqa: E402


def render_frame(frame, plate, robot, cam, q_old, q_new, g_old, g_new, masks, can_old, can_new, min_disp=0.5):
    H, W = frame.shape[:2]
    labels = rd.link_label_map(robot, cam, q_old["R"], q_old["L"], g_old["R"], g_old["L"], masks)
    robot.fk(q_old["R"], q_old["L"], g_old["R"], g_old["L"])
    pts_old = robot.link_points("sample")
    robot.fk(q_new["R"], q_new["L"], g_new["R"], g_new["L"])
    pts_new = robot.link_points("sample")
    layers, removed = [], np.zeros((H, W), bool)
    info = {"homographies": {}}
    for i, ((name, side, Po), (_, _, Pn)) in enumerate(zip(pts_old, pts_new)):
        if side == "B":
            continue
        m = labels[side] == i
        if not m.any():
            continue
        uo, _ = cam.project(Po)
        un, zn = cam.project(Pn)
        if np.linalg.norm(un - uo, axis=1).max() < min_disp:
            continue
        Hm = rd.homography(uo, un)
        layers.append((float(np.median(zn)), m, Hm, f"{side}_{name}"))
        removed |= m
        info["homographies"][f"{side}_{name}"] = Hm.tolist()
    if can_old is not None and can_new is not None and masks["can"].any():
        proj = lambda C, P: cam.project((C[:3, :3] @ P.T).T + C[:3, 3])  # noqa: E731
        uo_l, _ = proj(can_old, render2.lid_points())
        un_l, zn_l = proj(can_new, render2.lid_points())
        uo_b, _ = proj(can_old, render2.body_points())
        un_b, zn_b = proj(can_new, render2.body_points())
        if max(np.linalg.norm(un_l - uo_l, axis=1).max(), np.linalg.norm(un_b - uo_b, axis=1).max()) >= min_disp:
            cm = masks["can"] > 0
            lid = np.zeros((H, W), np.uint8)
            cv2.fillConvexPoly(lid, cv2.convexHull(uo_l.astype(np.float32)).astype(np.int32), 1)
            lid = (cv2.dilate(lid, np.ones((5, 5), np.uint8)) > 0) & cm
            body = cm & ~lid
            Hb, Hl = rd.homography(uo_b, un_b), rd.homography(uo_l, un_l)
            zb = float(np.median(zn_b))
            layers.append((zb, body, Hb, "can"))
            layers.append((min(float(np.median(zn_l)), zb - 1e-3), lid, Hl, "can_lid"))
            removed |= cm
            info["homographies"]["can"] = Hb.tolist()
    rem = cv2.dilate(removed.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    a = rd.feather(rem)[..., None]
    out = frame.astype(np.float32) * (1 - a) + plate.astype(np.float32) * a
    static = ((masks["R"] > 0) | (masks["L"] > 0) | (masks["can"] > 0)) & ~removed
    out[static] = frame[static]
    new_mask = np.zeros((H, W), bool)
    for z, m, Hm, name in sorted(layers, key=lambda l: -l[0]):
        wf = cv2.warpPerspective(frame, Hm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        wm = cv2.warpPerspective(m.astype(np.float32), Hm, (W, H), flags=cv2.INTER_LINEAR)
        al = np.clip(cv2.GaussianBlur(wm, (0, 0), 1.0), 0, 1)[..., None]
        out = out * (1 - al) + wf.astype(np.float32) * al
        new_mask |= wm > 0.5
    info["n_layers"] = len(layers)
    return np.clip(out, 0, 255).astype(np.uint8), new_mask, info


def openness_at(state16, t):
    g = rb.Robot.split(state16)
    T = len(state16)
    return {s: float(np.interp(np.clip(t, 0, T - 1), np.arange(T), g["g" + s])) for s in "RL"}


def render(rec, robot, cam, cand, min_disp=0.5):
    T = rec.T
    j_of = render2.source_frame_map(rec, cand["tau"])
    frames, newm, infos = [], [], []
    for i in range(T):
        j = int(j_of[i])
        ts = i + rec.lag
        q_old = {s: rec.gv[s][j] for s in "RL"}
        q_new = {s: rc.interp_q(cand["q"][s], ts) for s in "RL"}
        g_old = {s: rec.grip_v[s][j] for s in "RL"}
        g_new = openness_at(cand["state"], ts)
        can_old = rec.source_can(robot, j + rec.lag)
        can_new = render2.can_at(cand["can"], ts)
        fr, nm, info = render_frame(rec.F[j], rec.plate, robot, cam, q_old, q_new, g_old, g_new,
                                    {k: rec.masks[k][j] for k in rec.masks}, can_old, can_new, min_disp)
        info["source_frame"] = j
        frames.append(fr)
        newm.append(nm)
        infos.append(info)
    return np.stack(frames), np.stack(newm), infos, j_of


def video_states(rec, cand):
    T = rec.T
    ts = np.arange(T) + rec.lag
    q = {s: np.stack([rc.interp_q(cand["q"][s], t) for t in ts]) for s in "RL"}
    gr = [openness_at(cand["state"], t) for t in ts]
    grip = {s: np.array([g[s] for g in gr]) for s in "RL"}
    can = [render2.can_at(cand["can"], t) for t in ts]
    return q, grip, can
