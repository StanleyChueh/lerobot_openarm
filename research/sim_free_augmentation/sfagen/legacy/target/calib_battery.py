"""Body-camera calibration for the battery rig by render-and-compare (cf. CtRNet, Lu et al. CVPR 2023; RoboPose,
Labbe et al. CVPR 2021): maximise IoU between the z-buffered URDF mesh silhouette and the image foreground.
Foreground = |frame - per-episode temporal median of ALL frames| (camera-independent: no render enters the plate).
Only near-static frames (max joint speed < 0.05 rad/s) are used, so video/state lag cannot bias the fit.
Fit on even episodes, held-out evaluation on odd episodes. Parameters: camera rotation (3), translation (3), log focal (1),
relative to the inherited Pringles-rig camera. Writes calib_battery.json."""
import json, sys, time
from pathlib import Path
import numpy as np, torch, cv2
from scipy.optimize import minimize
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "battery/bsfa"), str(HERE.parent / "render_v4"), "/home/stanley/Stanley_ws/lerobot_openarm/research/sim_free_augmentation"]
import bdata, mesh
import pinocchio as pin
from sfa.robot import Robot, Camera
DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
robot, cam0 = Robot(), Camera(); G = mesh.Geometry(robot, device=DEV, spacing=0.003)
SIDE = torch.tensor([0 if s == "R" else 1 if s == "L" else 2 for s in G.side] + [3], device=DEV)


def cam_from(p):
    c = Camera(); s = np.exp(p[6])
    c.R = cam0.R @ pin.exp3(p[:3]); c.t = cam0.t + p[3:6]; c.K = cam0.K.copy(); c.K[0, 0] *= s; c.K[1, 1] *= s
    return c


def load(per_ep=14):
    fr = []
    for E in range(10):
        ep = bdata.load(E, cams=(bdata.CAMS[0],)); F = ep.frames[bdata.CAMS[0]]; S = ep.state
        Ft = torch.tensor(F[::3], device=DEV)
        med = Ft.float().median(0).values.cpu().numpy(); del Ft; torch.cuda.empty_cache()
        sp = Robot.split(S); qd = np.abs(np.gradient(np.concatenate([sp["R"], sp["L"]], 1), axis=0)).max(1) * 30
        cand = np.flatnonzero(qd < 0.05); cand = cand[(cand > 5) & (cand < len(S) - 5)]
        js = cand[np.linspace(0, len(cand) - 1, min(per_ep, len(cand))).astype(int)] if len(cand) else []
        for j in js:
            fg = np.abs(F[j].astype(np.int16) - med.astype(np.int16)).max(2) > 30
            fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
            fr.append({"E": E, "j": int(j), "fit": E % 2 == 0, "q": {s: sp[s][j].copy() for s in "RL"},
                       "g": {s: float(sp["g" + s][j]) for s in "RL"}, "fg": torch.tensor(fg, device=DEV)})
        print("loaded", E, len(js), flush=True)
    return fr


def ious(p, frames):
    c = cam_from(p); out = []
    k = torch.ones(1, 1, 41, 41, device=DEV)
    for f in frames:
        Ms = G.placements(f["q"]["R"], f["q"]["L"], f["g"]["R"], f["g"]["L"])
        _, lab, _ = mesh.splat(G.world(Ms), G.gid, c, foot=1, dev=DEV)
        sd = SIDE[lab.clamp(min=0)]; sd[lab < 0] = 3
        r = (sd == 0) | (sd == 1)
        roi = torch.nn.functional.conv2d(r.float()[None, None], k, padding=20)[0, 0] > 0
        a = f["fg"] & roi
        out.append(float((a & r).sum() / (a | r).sum().clamp(min=1)))
    return np.array(out)


def main():
    t0 = time.time(); frames = load(); fit = [f for f in frames if f["fit"]]; val = [f for f in frames if not f["fit"]]
    print("frames fit", len(fit), "val", len(val), flush=True)
    p0 = np.zeros(7); b_fit, b_val = ious(p0, fit), ious(p0, val)
    print("inherited: fit IoU %.4f val IoU %.4f" % (b_fit.mean(), b_val.mean()), flush=True)
    best = None
    sc = np.r_[[0.02] * 3, [0.02] * 3, 0.02]
    for start in (np.zeros(7),):
        f1 = lambda x: -ious(x * sc, fit).mean()  # noqa: E731
        r = minimize(f1, start, method="Powell", options={"maxfev": 900, "xtol": 1e-2, "ftol": 1e-5})
        if best is None or r.fun < best.fun:
            best = r
        print("powell", -r.fun, r.nfev, round(time.time() - t0), flush=True)
    p = best.x * sc; v = ious(p, val)
    rep = {"params7": p.tolist(), "cam_rvec": p[:3].tolist(), "cam_dt_m": p[3:6].tolist(), "focal_scale": float(np.exp(p[6])),
           "fit_iou": {"inherited": float(b_fit.mean()), "fitted": float(-best.fun)},
           "heldout_iou": {"inherited": float(b_val.mean()), "fitted": float(v.mean())},
           "heldout_iou_p10": {"inherited": float(np.percentile(b_val, 10)), "fitted": float(np.percentile(v, 10))},
           "heldout_frames_improved": float((v > b_val).mean()), "n_fit": len(fit), "n_heldout": len(val),
           "per_episode_heldout": {int(E): [float(b_val[[i for i, f in enumerate(val) if f["E"] == E]].mean()), float(v[[i for i, f in enumerate(val) if f["E"] == E]].mean())] for E in sorted({f["E"] for f in val})},
           "method": __doc__, "runtime_s": time.time() - t0}
    (HERE / "calib_battery.json").write_text(json.dumps(rep, indent=1)); print(json.dumps({k: rep[k] for k in rep if k != "method"}, indent=1))


main()
