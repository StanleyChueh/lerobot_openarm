"""Calibration by direct silhouette-IoU maximisation (z-buffered mesh render vs SAM2 arm masks, both arms), Powell.
Stage 1: camera (rotation 3, translation 3, log focal 1). Stage 2: + per-joint offsets (|off| <= 0.05 rad), kept only if
held-out IoU improves. Fit/held-out frames disjoint (alternating). Writes calib.json."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from scipy.optimize import minimize
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import mesh
SFA = mesh.SFA; sys.path[:0] = [str(SFA / "v3"), str(SFA / "v2")]
from sfa3 import recon3  # noqa
from sfa.robot import Robot, Camera
import pinocchio as pin
DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
robot, cam0 = Robot(), Camera(); G = mesh.Geometry(robot, device=DEV, spacing=0.003)
SIDE = torch.tensor([0 if s == "R" else 1 if s == "L" else 2 for s in G.side] + [3], device=DEV)


def cam_from(p):
    c = Camera(); s = np.exp(p[6])
    c.R = cam0.R @ pin.exp3(p[:3]); c.t = cam0.t + p[3:6]; c.K = cam0.K.copy(); c.K[0, 0] *= s; c.K[1, 1] *= s
    return c


def load():
    fr = []
    for E in range(10):
        rec = recon3.build(E, with_plate=False, log=lambda *a: None)
        for n, j in enumerate(range(3, rec.T, 8)):
            mR, mL = rec.masks["R"][j] > 0, rec.masks["L"][j] > 0
            if mR.sum() < 800 or mL.sum() < 800:
                continue
            fr.append({"E": E, "j": j, "fit": n % 2 == 0, "q": {s: rec.gv[s][j].copy() for s in "RL"},
                       "g": {s: float(rec.grip_v[s][j]) for s in "RL"},
                       "m": torch.tensor(np.stack([mR, mL]), device=DEV)})
    return fr


def ious(p, frames, with_off):
    c = cam_from(p); off = {"R": p[7:14], "L": p[14:21]} if with_off else {"R": 0, "L": 0}; out = []
    for f in frames:
        Ms = G.placements(f["q"]["R"] + off["R"], f["q"]["L"] + off["L"], f["g"]["R"], f["g"]["L"])
        _, lab, _ = mesh.splat(G.world(Ms), G.gid, c, foot=1, dev=DEV)
        sd = SIDE[lab.clamp(min=0)]; sd[lab < 0] = 3
        for k in range(2):
            a, b = f["m"][k], sd == k
            out.append(float((a & b).sum() / (a | b).sum().clamp(min=1)))
    return np.array(out)


def main():
    t0 = time.time(); frames = load(); fit = [f for f in frames if f["fit"]]; val = [f for f in frames if not f["fit"]]
    print("frames fit", len(fit), "val", len(val), round(time.time() - t0), flush=True)
    p = np.zeros(21)
    base_fit, base_val = ious(p, fit, False).mean(), ious(p, val, False)
    print("inherited: fit IoU", base_fit, "val IoU", base_val.mean(), flush=True)
    sc1 = np.r_[[0.02] * 3, [0.02] * 3, 0.02]
    f1 = lambda x: -ious(np.r_[x * sc1, np.zeros(14)], fit, False).mean()  # noqa: E731
    r1 = minimize(f1, np.zeros(7), method="Powell", options={"maxfev": 600, "xtol": 1e-2, "ftol": 1e-4})
    p1 = np.r_[r1.x * sc1, np.zeros(14)]
    v1 = ious(p1, val, False)
    print("stage1 cam: fit IoU", -r1.fun, "val IoU", v1.mean(), "evals", r1.nfev, round(time.time() - t0), flush=True)
    sc2 = np.r_[sc1 * 0.5, [0.02] * 14]
    f2 = lambda x: -ious(p1 + x * sc2, fit, True).mean() + 0.0 * np.sum(x ** 2)  # noqa: E731
    r2 = minimize(f2, np.zeros(21), method="Powell", bounds=[(-2.5, 2.5)] * 21, options={"maxfev": 2000, "xtol": 1e-2, "ftol": 1e-4})
    p2 = p1 + r2.x * sc2
    v2 = ious(p2, val, True)
    print("stage2 cam+joints: fit IoU", -r2.fun, "val IoU", v2.mean(), "evals", r2.nfev, round(time.time() - t0), flush=True)
    use2 = v2.mean() > v1.mean() + 0.005
    p = p2 if use2 else p1
    vb = base_val.reshape(-1, 2); vf = (v2 if use2 else v1).reshape(-1, 2)
    rep = {"params": p.tolist(), "stage_used": 2 if use2 else 1, "focal_scale": float(np.exp(p[6])), "cam_rvec": p[:3].tolist(),
           "cam_dt_m": p[3:6].tolist(), "joint_offset_rad": {"R": p[7:14].tolist(), "L": p[14:21].tolist()},
           "heldout_iou": {"inherited": {"R": float(vb[:, 0].mean()), "L": float(vb[:, 1].mean())},
                           "stage1": {"R": float(v1.reshape(-1, 2)[:, 0].mean()), "L": float(v1.reshape(-1, 2)[:, 1].mean())},
                           "stage2": {"R": float(v2.reshape(-1, 2)[:, 0].mean()), "L": float(v2.reshape(-1, 2)[:, 1].mean())}},
           "heldout_iou_p10": {"inherited": np.percentile(vb, 10, axis=0).tolist(), "final": np.percentile(vf, 10, axis=0).tolist()},
           "n_fit_frames": len(fit), "n_heldout_frames": len(val), "runtime_s": time.time() - t0}
    (HERE / "calib.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))

main()
