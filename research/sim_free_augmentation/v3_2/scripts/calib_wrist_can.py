"""Wrist mount fit from CAN silhouettes (no board, no simulator), with held-out validation.

The can's world pose comes from the body-cam reconstruction (standing on the table before the giver grasps; carried
by the giver before the receiver grasps). In those phases the can is NOT attached to the observing wrist camera and
is large in view, so the HSV red-body mask constrains the camera pose. For each wrist camera we fit the 6-DoF mount
X (link7 -> camera) maximising the mean IoU between the projected can body (cylinder, 2-23 cm below the lid) and the
red mask, from the nominal mount (Powell), on episodes 0-4; IoU on episodes 5-9 decides adoption (>= +0.05).
The video-to-state lag is the per-camera value measured by calib_wrist.py. The fitted mount absorbs the body-cam
calibration and can-attachment errors, so it is a CONSISTENCY calibration for rendering, not a verified true mount.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / "v3"))
sys.path.insert(0, str(HERE))
from sfa3 import recon3  # noqa: E402
from sfa3.recon3 import data, rb  # noqa: E402
from sfa32 import wristcam as wc  # noqa: E402
from sfa.recover import interp_q, CAN_RADIUS, CAN_HEIGHT  # noqa: E402

FIT, VAL = [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]
H, W = 480, 640
TH = np.linspace(0, 2 * np.pi, 32, endpoint=False)
BODY = np.concatenate([np.stack([CAN_RADIUS * np.cos(TH), CAN_RADIUS * np.sin(TH), np.full(32, s)], 1) for s in np.linspace(0.02, CAN_HEIGHT, 6)])


def red(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = (((hsv[..., 0] < 8) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 120) & (hsv[..., 2] > 60)).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0


def samples(E, side, robot):
    ep = data.load(E, cams=(wc.CAM_OF[side],))
    rec = recon3.build(E, with_plate=False, log=lambda *a: None)
    F = ep.frames[wc.CAM_OF[side]]
    g = rb.Robot.split(ep.state)
    ev = ep.events
    lag = wc.lag(side)
    rng = range(5, ev["right_close"] - 2) if side == "R" else range(ev["right_close"] + 12, ev["left_close"] - 1)
    out = []
    for t in rng:
        m = red(F[t])
        f = m.mean()
        if 0.03 <= f <= 0.6:
            robot.fk(interp_q(g["R"], t + lag), interp_q(g["L"], t + lag))
            L7 = robot.frame(wc.link7(robot, side))
            can = rec.source_can(robot, t + lag)
            P = (can[:3, :3] @ BODY.T).T + can[:3, 3]
            out.append({"t": t, "mask": cv2.resize(m.astype(np.uint8), (W // 4, H // 4), interpolation=cv2.INTER_AREA) > 0,
                        "L7": L7, "P": P})
    return out[::2]


def iou_for(X, S, side):
    Kc = wc.K(side) / 4.0
    Kc[2, 2] = 1
    vals = []
    for s in S:
        Tc = s["L7"] @ X
        c = (s["P"] - Tc[:3, 3]) @ Tc[:3, :3]
        z = c[:, 2]
        if (z > 0.03).sum() < 20:
            vals.append(0.0)
            continue
        c = c[z > 0.03]
        uv = np.stack([Kc[0, 0] * c[:, 0] / c[:, 2] + Kc[0, 2], Kc[1, 1] * c[:, 1] / c[:, 2] + Kc[1, 2]], 1)
        uv = np.clip(uv, -2000, 2000)
        pm = np.zeros((H // 4, W // 4), np.uint8)
        cv2.fillConvexPoly(pm, cv2.convexHull(uv.astype(np.float32)).astype(np.int32), 1)
        pm = pm > 0
        u = (pm | s["mask"]).sum()
        vals.append((pm & s["mask"]).sum() / u if u else 0.0)
    return float(np.mean(vals))


def X_of(v, X0):
    X = np.eye(4)
    X[:3, :3] = X0[:3, :3] @ Rot.from_rotvec(v[:3]).as_matrix()
    X[:3, 3] = X0[:3, 3] + v[3:]
    return X


def main():
    t0 = time.time()
    robot = rb.Robot()
    calib = json.loads(wc.CALIB.read_text())
    for side in "RL":
        S_fit = [s for e in FIT for s in samples(e, side, robot)]
        S_val = [s for e in VAL for s in samples(e, side, robot)]
        X0 = wc.mount(side, "nominal")
        f0, v0 = iou_for(X0, S_fit, side), iou_for(X0, S_val, side)
        best = None
        for start in ([0, 0, 0, 0, 0, 0], [0.15, 0, 0, 0, 0, 0], [-0.15, 0, 0, 0, 0, 0], [0, 0.15, 0, 0, 0, 0], [0, -0.15, 0, 0, 0, 0]):
            r = minimize(lambda v: -iou_for(X_of(v, X0), S_fit, side), np.array(start, float), method="Powell",
                         options={"maxiter": 3000, "xtol": 1e-3, "ftol": 1e-4})
            if best is None or r.fun < best.fun:
                best = r
        X1 = X_of(best.x, X0)
        f1, v1 = iou_for(X1, S_fit, side), iou_for(X1, S_val, side)
        adopt = bool(v1 >= v0 + 0.05)
        c = calib[side]
        c["can_silhouette_fit"] = {"n_fit": len(S_fit), "n_val": len(S_val), "iou_fit_nominal": f0, "iou_fit_fitted": f1,
                                   "iou_val_nominal": v0, "iou_val_fitted": v1, "adopted": adopt,
                                   "rotation_change_deg": float(np.degrees(np.linalg.norm(best.x[:3]))),
                                   "translation_change_m": float(np.linalg.norm(best.x[3:])), "X": X1.tolist()}
        if adopt:
            c["mount"] = {"R": X1[:3, :3].tolist(), "p": X1[:3, 3].tolist(), "R_source": "can-silhouette fit (eps 0-4, validated 5-9)",
                          "p_source": "can-silhouette fit (eps 0-4, validated 5-9)"}
        print(side, json.dumps(c["can_silhouette_fit"] | {"X": None}), flush=True)
    calib["can_fit_runtime_s"] = time.time() - t0
    wc.CALIB.write_text(json.dumps(calib, indent=1))


if __name__ == "__main__":
    main()
