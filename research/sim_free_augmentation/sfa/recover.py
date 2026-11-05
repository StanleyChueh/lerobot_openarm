"""Recover robot and object motion from one real episode, using only FK + the camera model + 2D image evidence.

* Robot: FK of the recorded joint state (metric, robot frame). The video is offset from the state by a fractional
  lag (Gate 1: -1.25 ... -1.75 frames, model-dependent); it is re-estimated here on this episode.
* Can: no 3D sensor and a single static camera, so a static can cannot be triangulated. While a hand holds the can,
  its lid centre is a FIXED point in that hand's TCP frame; observing it under the hand's known motion gives a
  metric 3D point (structure from known motion). That fit is the only source of can 3D here:
    giver hold   -> p_R (lid in the right TCP frame)
    receiver hold-> p_L (lid in the left TCP frame)
    table phase  -> world point T_R(grasp) . p_R (the can stands still until the grasp); checked against the image
  The can axis (needed only to warp the can region) is chosen by silhouette IoU against the SAM2 can mask.
  Nothing is reported in 3D where neither fit is valid.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .robot import Camera, Robot

CAN_RADIUS = 0.0375      # declared prior: Pringles can, 7.5 cm diameter
CAN_HEIGHT = 0.23        # declared prior: ~23 cm tall (only used for the warp region / silhouette check)


def interp_q(Q: np.ndarray, t: float) -> np.ndarray:
    t = float(np.clip(t, 0, len(Q) - 1))
    i = int(np.floor(t))
    j = min(i + 1, len(Q) - 1)
    f = t - i
    return (1 - f) * Q[i] + f * Q[j]


def tcp_pose(robot: Robot, g: dict, side: str, t: float) -> np.ndarray:
    robot.fk(interp_q(g["R"], t), interp_q(g["L"], t))
    return robot.frame(robot.tcp[side])


def fit_point(robot, cam: Camera, g, side, frames, uv, lag) -> dict:
    """lid point in the `side` TCP frame from 2D observations uv[frames] (video frame t <-> state t + lag)."""
    Ts = [tcp_pose(robot, g, side, t + lag) for t in frames]
    obs = uv[frames]

    def res(p):
        P = np.stack([T[:3, :3] @ p + T[:3, 3] for T in Ts])
        u, z = cam.project(P)
        return (u - obs).ravel()

    r = least_squares(res, x0=np.array([0.0, 0.0, 0.05]), loss="soft_l1", f_scale=5.0)
    e = np.linalg.norm(r.fun.reshape(-1, 2), axis=1)
    J = r.jac
    cov = np.linalg.pinv(J.T @ J) * np.median(e) ** 2
    return {"p": r.x, "rms_px": float(np.sqrt(np.mean(e ** 2))), "median_px": float(np.median(e)),
            "p90_px": float(np.percentile(e, 90)), "n": int(len(frames)), "std_m": np.sqrt(np.diag(cov)).tolist()}


def estimate_lag(robot, cam, g, side, frames, uv, lags=np.arange(-3.0, 1.01, 0.25)) -> dict:
    out = {float(l): fit_point(robot, cam, g, side, frames, uv, l)["rms_px"] for l in lags}
    best = min(out, key=out.get)
    return {"lag": best, "rms_by_lag": out}


def cylinder_points(axis, n_ring=24, n_len=6):
    """points on a can surface in a frame whose origin is the lid centre; axis points from lid to bottom."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    u = np.cross(a, [1, 0, 0])
    if np.linalg.norm(u) < 1e-3:
        u = np.cross(a, [0, 1, 0])
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    th = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    pts = [np.zeros(3)]
    for s in np.linspace(0, CAN_HEIGHT, n_len):
        for c, d in zip(np.cos(th), np.sin(th)):
            pts.append(s * a + CAN_RADIUS * (c * u + d * v))
    return np.array(pts)


def fibonacci_dirs(n=300):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def fit_axis(robot, cam, g, side, p_lid, frames, masks, lag, H=480, W=640) -> dict:
    """can axis in the TCP frame maximising mean IoU of the projected cylinder with the can mask on `frames`."""
    import cv2
    Ts = [tcp_pose(robot, g, side, t + lag) for t in frames]
    best = (-1, None)
    for a in fibonacci_dirs(300):
        ious = []
        P0 = cylinder_points(a, 16, 4) + p_lid
        for T, t in zip(Ts, frames):
            uv, z = cam.project((T[:3, :3] @ P0.T).T + T[:3, 3])
            if (z < 0.05).any():
                ious.append(0)
                continue
            m = np.zeros((H, W), np.uint8)
            cv2.fillConvexPoly(m, cv2.convexHull(uv.astype(np.float32)).astype(np.int32), 1)
            gt = masks[t] > 0
            ious.append((m.astype(bool) & gt).sum() / max(1, (m.astype(bool) | gt).sum()))
        s = float(np.mean(ious))
        if s > best[0]:
            best = (s, a)
    return {"axis": best[1], "mean_iou": best[0]}


def can_world_track(robot, g, ev, fits, T_len, lag):
    """(T,4,4) can frame (origin lid centre, z axis = can axis lid->bottom) for every frame, or None if unknown.

    table: fixed at the right-grasp pose; giver hold .. right open: right TCP . X_R; after: left TCP . X_L."""
    def frame_from(side, t, fit):
        T = tcp_pose(robot, g, side, t + lag)
        X = np.eye(4)
        a = fit["axis"] / np.linalg.norm(fit["axis"])
        u = np.cross(a, [1, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1, 0])
        u /= np.linalg.norm(u)
        X[:3, :3] = np.stack([u, np.cross(a, u), a], 1)
        X[:3, 3] = fit["p"]
        return T @ X, X

    out = [None] * T_len
    holder = [None] * T_len
    t_grasp = ev["right_close"] + 8
    T_table, _ = frame_from("R", t_grasp, fits["R"])
    for t in range(T_len):
        if t < t_grasp:
            out[t], holder[t] = T_table, "table"
        elif t < ev["right_open"]:
            out[t], holder[t] = frame_from("R", t, fits["R"])[0], "R"
        elif fits.get("L") is not None:
            out[t], holder[t] = frame_from("L", t, fits["L"])[0], "L"
    return out, holder
