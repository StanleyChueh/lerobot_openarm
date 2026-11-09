"""Joint fit of body_cam extrinsics + focal scale + per-joint encoder offsets (both arms) to the SAM2 arm masks of all
10 source episodes (silhouette chamfer, robust). Fit frames and held-out frames are disjoint. Output: calib.json."""
import json, sys, time
from pathlib import Path
import numpy as np, torch, cv2
from scipy.optimize import least_squares
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import mesh
SFA = mesh.SFA; sys.path[:0] = [str(SFA / "v3"), str(SFA / "v2")]
from sfa3 import recon3  # noqa
from sfa.robot import Robot, Camera
import pinocchio as pin
DEV = "cuda:0"; H, W = 480, 640; CAP = 40.0
robot, cam0 = Robot(), Camera(); G = mesh.Geometry(robot, spacing=0.004)
side_of = np.array(G.side)
arm_pts = torch.isin(G.gid, torch.tensor([i for i, s in enumerate(G.side) if s in "RL"], device=DEV))


def cam_from(p):
    c = Camera(); s = np.exp(p[6])
    c.R = cam0.R @ pin.exp3(p[:3]); c.t = cam0.t + p[3:6]
    c.K = cam0.K.copy(); c.K[0, 0] *= s; c.K[1, 1] *= s
    return c


def project(c, P):
    R = torch.tensor(c.R, dtype=torch.float32, device=DEV); t = torch.tensor(c.t, dtype=torch.float32, device=DEV)
    x = (P - t) @ R; K = c.K
    return torch.stack([K[0, 0] * x[:, 0] / x[:, 2] + K[0, 2], K[1, 1] * x[:, 1] / x[:, 2] + K[1, 2]], 1)


def load(step=24):
    fr = []
    for E in range(10):
        rec = recon3.build(E, with_plate=False, log=lambda *a: None)
        for j in range(5, rec.T, step // 2):
            m = {s: rec.masks[s][j] > 0 for s in ("R", "L", "can")}
            if m["R"].sum() < 500 or m["L"].sum() < 500:
                continue
            union = (m["R"] | m["L"] | m["can"]).astype(np.uint8)
            dt_out = cv2.distanceTransform(1 - union, cv2.DIST_L2, 3)
            rec_pts = {s: np.argwhere(m[s])[::7][:, ::-1].astype(np.float32) for s in "RL"}
            fr.append({"E": E, "j": j, "fit": ((j - 5) // (step // 2)) % 2 == 0,
                       "q": {s: rec.gv[s][j].copy() for s in "RL"}, "g": {s: float(rec.grip_v[s][j]) for s in "RL"},
                       "dt_out": torch.tensor(dt_out, device=DEV), "rec_pts": rec_pts, "mask": m})
        print("loaded", E, len(fr), flush=True)
    return fr


def frame_geom(f, p):
    off = {"R": p[7:14], "L": p[14:21]}
    Ms = G.placements(f["q"]["R"] + off["R"], f["q"]["L"] + off["L"], f["g"]["R"], f["g"]["L"])
    return G.world(Ms)


def residuals(p, frames):
    c = cam_from(p); res = []
    for f in frames:
        P = frame_geom(f, p)
        uv = project(c, P)
        for s in "RL":
            sel = G.gid.new_tensor(side_of == s, dtype=torch.bool)[G.gid]
            u = uv[sel]
            ui = u[:, 0].round().long().clamp(0, W - 1); vi = u[:, 1].round().long().clamp(0, H - 1)
            inside = (u[:, 0] >= 0) & (u[:, 0] < W) & (u[:, 1] >= 0) & (u[:, 1] < H)
            prec = torch.where(inside, f["dt_out"][vi, ui].clamp(max=CAP), torch.full_like(u[:, 0], CAP)).mean()
            m = torch.zeros(H * W, device=DEV); m[(vi * W + ui)[inside]] = 1
            m = torch.nn.functional.max_pool2d(m.view(1, 1, H, W), 7, 1, 3)[0, 0].cpu().numpy().astype(np.uint8)
            dt_r = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3)
            rp = f["rec_pts"][s].astype(int)
            rec = np.minimum(dt_r[rp[:, 1], rp[:, 0]], CAP).mean()
            res += [float(prec), float(rec)]
    return np.array(res)


def iou_eval(p, frames):
    c = cam_from(p); G2 = mesh.Geometry(robot, spacing=0.0015) if not hasattr(iou_eval, "G2") else iou_eval.G2
    iou_eval.G2 = G2; out = []
    for f in frames:
        off = {"R": p[7:14], "L": p[14:21]}
        Ms = G2.placements(f["q"]["R"] + off["R"], f["q"]["L"] + off["L"], f["g"]["R"], f["g"]["L"])
        _, lab, _ = mesh.splat(G2.world(Ms), G2.gid, c)
        sd = np.array(G2.side + ["X"])[lab.cpu().numpy()]
        for s in "RL":
            a, b = f["mask"][s], sd == s
            out.append({"E": f["E"], "j": f["j"], "side": s, "iou": float((a & b).sum() / max(1, (a | b).sum()))})
    return out


def main():
    t0 = time.time()
    frames = load()
    fit = [f for f in frames if f["fit"]]; val = [f for f in frames if not f["fit"]]
    print("fit", len(fit), "val", len(val), flush=True)
    p0 = np.zeros(21)
    lb = np.r_[[-0.08] * 3, [-0.06] * 3, -0.1, [-0.12] * 14]; ub = -lb
    r0 = residuals(p0, fit); print("initial mean px", r0.mean(), flush=True)
    sol = least_squares(residuals, p0, args=(fit,), bounds=(lb, ub), loss="soft_l1", f_scale=5.0, diff_step=1e-3,
                        x_scale=np.r_[[0.01] * 3, [0.01] * 3, 0.01, [0.02] * 14], max_nfev=400, verbose=1)
    print("final mean px", sol.fun.mean(), "time", round(time.time() - t0), flush=True)
    before, after = iou_eval(p0, val), iou_eval(sol.x, val)
    agg = lambda L, s: float(np.mean([x["iou"] for x in L if x["side"] == s]))
    rep = {"params": sol.x.tolist(), "cam_rvec": sol.x[:3].tolist(), "cam_dt_m": sol.x[3:6].tolist(),
           "focal_scale": float(np.exp(sol.x[6])), "joint_offset_rad": {"R": sol.x[7:14].tolist(), "L": sol.x[14:21].tolist()},
           "fit_residual_px": {"before": float(r0.mean()), "after": float(sol.fun.mean())},
           "heldout_iou": {s: {"before": agg(before, s), "after": agg(after, s)} for s in "RL"},
           "heldout_iou_p10": {s: {"before": float(np.percentile([x["iou"] for x in before if x["side"] == s], 10)),
                                   "after": float(np.percentile([x["iou"] for x in after if x["side"] == s], 10))} for s in "RL"},
           "n_fit_frames": len(fit), "n_heldout_frames": len(val), "per_frame_heldout": {"before": before, "after": after}}
    (HERE / "calib.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "per_frame_heldout"}, indent=1))

main()
