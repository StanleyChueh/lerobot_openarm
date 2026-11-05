"""Wrist-camera calibration diagnostics and rotation hand-eye fit from the real videos + FK (no simulator, no board).

    cd research/sim_free_augmentation/v3_2
    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python scripts/calib_wrist.py

For frame pairs (j, j+3) in which the wrist camera sees mostly static background (red-can fraction < 0.15) and the hand
rotates >= 0.6 deg, RAFT flow (forward-backward consistent, own finger-pad pixels excluded) gives an essential matrix
with the MEASURED intrinsics -> camera rotation A. FK of link7 at state time (frame + lag) gives hand rotation B.
  1. lag: the rotation ANGLE is invariant to the unknown mount (A = X^-1 B X), so |angle(A)| vs |angle(B)| over lags
     estimates the per-camera video-to-state lag independently of the mount;
  2. mount rotation R_X: argmin sum angle(R_X R_A R_X^T, R_B) (Kabsch on rotation axes, then nonlinear refinement),
     fitted on episodes 0-4 and validated on 5-9 against the nominal (GUI-tuned) mount; adopted only if it lowers
     the held-out median residual by >= 0.1 deg;
  3. mount translation: linear least squares (R_B - I) p = R_X (s t_A) - t_B with per-pair scales s; reported with
     its condition number, adopted only if well conditioned and within 3 cm of nominal.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / "v3"))
sys.path.insert(0, str(HERE))
from sfa3.recon3 import data, rb, recon2  # noqa: E402
from sfa32 import wristcam as wc  # noqa: E402
from sfa.recover import interp_q  # noqa: E402
from sfa import track as trk  # noqa: E402

FIT, VAL = [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]
GAP = 3


def red_frac(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return float((((hsv[..., 0] < 8) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 120) & (hsv[..., 2] > 60)).mean())


def hand_rel(robot, g, side, t0, t1):
    robot.fk(interp_q(g["R"], t0), interp_q(g["L"], t0))
    A = robot.frame(wc.link7(robot, side))
    robot.fk(interp_q(g["R"], t1), interp_q(g["L"], t1))
    B = robot.frame(wc.link7(robot, side))
    return np.linalg.inv(A) @ B


def pairs_for(ep, side, robot, device):
    cam = wc.CAM_OF[side]
    F = ep.frames[cam]
    g = rb.Robot.split(ep.state)
    T = len(F)
    cand = []
    for j in range(0, T - GAP, 2):
        B = hand_rel(robot, g, side, j, j + GAP)
        ang = np.degrees(np.linalg.norm(Rot.from_matrix(B[:3, :3]).as_rotvec()))
        if ang >= 0.6 and red_frac(F[j]) < 0.15 and red_frac(F[j + GAP]) < 0.15:
            cand.append(j)
    if not cand:
        return []
    fw = trk.flow_pairs(F, [(j, j + GAP) for j in cand], device)
    bw = trk.flow_pairs(F, [(j + GAP, j) for j in cand], device)
    Kc = wc.K(side)
    H, W = F.shape[1:3]
    gy, gx = np.mgrid[8:H - 8:8, 8:W - 8:8]
    P = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)
    # own finger pads / attached pixels: near-zero flow while the camera clearly rotates (median over pairs)
    mags = np.stack([np.linalg.norm(f, axis=2) for f in fw])
    med_mag = np.median(mags, axis=0)
    attached = med_mag < 0.15 * np.median(med_mag)
    out = []
    for i, j in enumerate(cand):
        f = fw[i]
        q = P + trk.sample(f, P)
        back = q + trk.sample(bw[i], q)
        ok = (np.linalg.norm(back - P, axis=1) < 1.0) & ~attached[P[:, 1].astype(int), P[:, 0].astype(int)] & \
             (q[:, 0] > 0) & (q[:, 0] < W - 1) & (q[:, 1] > 0) & (q[:, 1] < H - 1)
        if ok.sum() < 200:
            continue
        E, inl = cv2.findEssentialMat(P[ok], q[ok], Kc, cv2.RANSAC, 0.999, 1.0)
        if E is None or E.shape != (3, 3):
            continue
        n, R, t, _ = cv2.recoverPose(E, P[ok], q[ok], Kc, mask=inl.copy())
        if n < 150:
            continue
        A = np.eye(4)
        A[:3, :3] = R.T                       # camera j+GAP expressed in camera j (x_k = R x_j + t)
        A[:3, 3] = (-R.T @ t).ravel()         # unit-norm direction only
        out.append({"episode": ep.index, "j": j, "A": A, "inliers": int(n), "n": int(ok.sum())})
    return out


def angle(R):
    return float(np.degrees(np.linalg.norm(Rot.from_matrix(R).as_rotvec())))


def fit_rotation(As, Bs, R0):
    def res(v):
        RX = Rot.from_rotvec(v).as_matrix()
        return np.array([np.radians(angle(RX @ A[:3, :3] @ RX.T @ B[:3, :3].T)) for A, B in zip(As, Bs)])
    # initial: Kabsch on axes weighted by angle
    a = np.stack([Rot.from_matrix(A[:3, :3]).as_rotvec() for A in As])
    b = np.stack([Rot.from_matrix(B[:3, :3]).as_rotvec() for B in Bs])
    Hm = a.T @ b
    U, _, Vt = np.linalg.svd(Hm)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    Rk = Vt.T @ D @ U.T
    best = None
    for init in (Rk, R0):
        r = least_squares(res, Rot.from_matrix(init).as_rotvec(), loss="soft_l1", f_scale=np.radians(0.5))
        if best is None or r.cost < best.cost:
            best = r
    return Rot.from_rotvec(best.x).as_matrix()


def fit_translation(As, Bs, RX):
    n = len(As)
    M = np.zeros((3 * n, 3 + n))
    y = np.zeros(3 * n)
    for i, (A, B) in enumerate(zip(As, Bs)):
        M[3 * i:3 * i + 3, :3] = B[:3, :3] - np.eye(3)
        M[3 * i:3 * i + 3, 3 + i] = -(RX @ A[:3, 3])
        y[3 * i:3 * i + 3] = -B[:3, 3]
    sol, *_ = np.linalg.lstsq(M, y, rcond=None)
    s = np.linalg.svd(M[:, :3], compute_uv=False)
    return sol[:3], float(s[0] / max(s[-1], 1e-12)), sol[3:]


def main():
    t0 = time.time()
    robot = rb.Robot()
    out = {}
    lags = np.arange(-4.0, 2.01, 0.5)
    allpairs = {"L": [], "R": []}
    for e in FIT + VAL:
        ep = data.load(e)
        for side in "RL":
            allpairs[side] += [dict(p, state=ep.state) for p in pairs_for(ep, side, robot, "cuda:0")]
        print(f"ep{e}: pairs R {sum(p['episode'] == e for p in allpairs['R'])} L {sum(p['episode'] == e for p in allpairs['L'])} ({time.time() - t0:.0f} s)", flush=True)
    for side in "RL":
        P = allpairs[side]
        # 1. lag from rotation magnitudes (mount-independent)
        curve = {}
        for l in lags:
            d = []
            for p in P:
                g = rb.Robot.split(p["state"])
                d.append(abs(angle(p["A"][:3, :3]) - angle(hand_rel(robot, g, side, p["j"] + l, p["j"] + GAP + l)[:3, :3])))
            curve[float(l)] = float(np.median(d))
        lag = min(curve, key=curve.get)
        for p in P:
            g = rb.Robot.split(p["state"])
            p["B"] = hand_rel(robot, g, side, p["j"] + lag, p["j"] + GAP + lag)
        fitp = [p for p in P if p["episode"] in FIT]
        valp = [p for p in P if p["episode"] in VAL]
        R0 = wc.mount(side, "nominal")[:3, :3]
        RX = fit_rotation([p["A"] for p in fitp], [p["B"] for p in fitp], R0)

        def resid(RXc, ps):
            return [angle(RXc @ p["A"][:3, :3] @ RXc.T @ p["B"][:3, :3].T) for p in ps]
        r_nom_val, r_fit_val = resid(R0, valp), resid(RX, valp)
        p_fit, cond, scales = fit_translation([p["A"] for p in fitp], [p["B"] for p in fitp], RX)
        p_nom = np.array(wc.NOMINAL["pos"])
        adopt_t = bool(cond < 50 and np.linalg.norm(p_fit - p_nom) < 0.03)
        adopt_R = bool(np.median(r_fit_val) <= np.median(r_nom_val) - 0.1)     # must improve HELD-OUT residual by >= 0.1 deg
        out[side] = {"camera": wc.CAM_OF[side], "rotation_adopted": adopt_R, "n_pairs_fit": len(fitp), "n_pairs_val": len(valp),
                     "lag_frames": lag, "lag_curve_median_abs_angle_diff_deg": curve,
                     "rot_angle_magnitude_corr": float(np.corrcoef([angle(p["A"][:3, :3]) for p in P], [angle(p["B"][:3, :3]) for p in P])[0, 1]),
                     "val_residual_deg_nominal": {"median": float(np.median(r_nom_val)), "p90": float(np.percentile(r_nom_val, 90))},
                     "val_residual_deg_fitted": {"median": float(np.median(r_fit_val)), "p90": float(np.percentile(r_fit_val, 90))},
                     "fit_residual_deg_fitted": float(np.median(resid(RX, fitp))),
                     "rotation_change_from_nominal_deg": angle(RX @ R0.T),
                     "view_axis_in_link7": {"nominal": R0[:, 2].tolist(), "fitted": RX[:, 2].tolist()},
                     "translation": {"fit": p_fit.tolist(), "nominal": p_nom.tolist(), "condition_number": cond,
                                     "distance_to_nominal_m": float(np.linalg.norm(p_fit - p_nom)), "adopted": adopt_t},
                     "fitted_rotation": RX.tolist(),
                     "mount": {"R": (RX if adopt_R else R0).tolist(), "p": (p_fit if adopt_t else p_nom).tolist(),
                               "R_source": "rotation hand-eye fit (episodes 0-4)" if adopt_R else "nominal (fit did not improve held-out residual)",
                               "p_source": "fit" if adopt_t else "nominal (fit not trusted)"}}
        print(side, json.dumps({k: v for k, v in out[side].items() if k not in ("lag_curve_median_abs_angle_diff_deg", "mount")}, default=float), flush=True)
    out["runtime_s"] = time.time() - t0
    wc.CALIB.parent.mkdir(parents=True, exist_ok=True)
    wc.CALIB.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
