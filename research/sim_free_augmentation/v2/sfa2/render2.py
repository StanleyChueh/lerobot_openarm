"""Rendering of a candidate into body_cam RGB (v1 layered-warp renderer + split lid/body can), generalised to time warps.

Generated video frame i depicts the candidate state at state time i + lag (the same video-to-state relation measured
in the source episode). Its pixels come from source video frame j = round(tau(i + lag) - lag): the source frame that
shows the source configuration the candidate was warped from. Each link / the can is moved by the homography from
its source pose in frame j to its candidate pose at i + lag.
"""

from __future__ import annotations

import numpy as np

import cv2

from . import se3
from .recon import Recon, rc, rd


def lid_points(n=24):
    """planar lid (origin = lid centre, z = axis): centre + two rings - an exact homography under any rigid motion."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = lambda r: np.stack([r * np.cos(th), r * np.sin(th), np.zeros(n)], 1)  # noqa: E731
    return np.concatenate([np.zeros((1, 3)), ring(0.5 * rc.CAN_RADIUS), ring(rc.CAN_RADIUS)])


def body_points(n=24, m=6):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.concatenate([np.stack([rc.CAN_RADIUS * np.cos(th), rc.CAN_RADIUS * np.sin(th), np.full(n, s)], 1)
                           for s in np.linspace(0.015, rc.CAN_HEIGHT, m)])


def render_frame(frame, plate, robot, cam, q_old, q_new, grip, masks, can_old, can_new, min_disp=0.5):
    """v1 layered warp (sfa/render.py) with ONE change: the can is two layers - the planar lid, warped by its exact
    homography, and the body, warped by a homography fitted to body points only. v1 used one homography for the
    whole cylinder, which misplaced the lid by ~10-16 px under out-of-plane rotations (episode-0 tilt/yaw runs)."""
    H, W = frame.shape[:2]
    labels = rd.link_label_map(robot, cam, q_old["R"], q_old["L"], grip["R"], grip["L"], masks)
    robot.fk(q_old["R"], q_old["L"], grip["R"], grip["L"])
    pts_old = robot.link_points("sample")
    robot.fk(q_new["R"], q_new["L"], grip["R"], grip["L"])
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
        uo_l, zo_l = proj(can_old, lid_points())
        un_l, zn_l = proj(can_new, lid_points())
        uo_b, _ = proj(can_old, body_points())
        un_b, zn_b = proj(can_new, body_points())
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
            info["homographies"]["can_lid"] = Hl.tolist()
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
    info["moved_px"] = int(removed.sum())
    return np.clip(out, 0, 255).astype(np.uint8), new_mask, info


def can_at(can: np.ndarray, t: float) -> np.ndarray:
    T = len(can)
    t = float(np.clip(t, 0, T - 1))
    i = int(np.floor(t))
    j = min(i + 1, T - 1)
    f = t - i
    return can[i] if f < 1e-9 or i == j else se3.blend(can[i], can[j], f)


def tau_at(tau: np.ndarray, t):
    """time warp at fractional times; outside [0, T-1] it is extended with slope 1 (the warp is the identity there)."""
    T = len(tau)
    t = np.asarray(t, float)
    out = np.interp(t, np.arange(T), tau)
    out = np.where(t < 0, tau[0] + t, out)
    return np.where(t > T - 1, tau[-1] + (t - (T - 1)), out)


def source_frame_map(rec: Recon, tau: np.ndarray) -> np.ndarray:
    T = rec.T
    ts_src = tau_at(tau, np.arange(T) + rec.lag)
    return np.clip(np.round(ts_src - rec.lag).astype(int), 0, T - 1)


def render(rec: Recon, robot, cam, cand: dict, min_disp=0.5):
    T = rec.T
    j_of = source_frame_map(rec, cand["tau"])
    frames, newm, infos = [], [], []
    for i in range(T):
        j = int(j_of[i])
        ts = i + rec.lag
        q_old = {s: rec.gv[s][j] for s in "RL"}
        q_new = {s: rc.interp_q(cand["q"][s], ts) for s in "RL"}
        grip = {s: rec.grip_v[s][j] for s in "RL"}
        can_old = rec.source_can(robot, j + rec.lag)
        can_new = can_at(cand["can"], ts)
        fr, nm, info = render_frame(rec.F[j], rec.plate, robot, cam, q_old, q_new, grip,
                                    {k: rec.masks[k][j] for k in rec.masks}, can_old, can_new, min_disp)
        info["source_frame"] = j
        frames.append(fr)
        newm.append(nm)
        infos.append(info)
    return np.stack(frames), np.stack(newm), infos, j_of


def video_states(rec: Recon, cand: dict):
    """candidate joints / grip / can at the state time depicted by each video frame (i + lag)."""
    T = rec.T
    ts = np.arange(T) + rec.lag
    q = {s: np.stack([rc.interp_q(cand["q"][s], t) for t in ts]) for s in "RL"}
    j_of = source_frame_map(rec, cand["tau"])
    grip = {s: rec.grip_v[s][j_of] for s in "RL"}
    can = [can_at(cand["can"], t) for t in ts]
    return q, grip, can
