"""Body-camera calibration by render-and-compare (cf. CtRNet, RoboPose): maximise IoU between the z-buffered URDF
silhouette and a camera-independent foreground (|frame - temporal median of all frames| > 30) on near-static frames.
Stage 0 (optional, for a new rig): coarse random search around the configured initial guess (+-search_deg, +-search_m).
Stage 1: Powell on (rotation 3, translation 3, log focal 1). Fit on even source episodes, evaluated on odd ones."""
from __future__ import annotations
import json, time
import cv2, numpy as np, torch
from scipy.optimize import minimize
from . import mesh


def load_frames(ctx, max_eps=10, per_ep=14, device="cuda:0"):
    fr = []
    for i, e in enumerate(ctx.episodes()[:max_eps]):
        S, _ = ctx.src.arrays(e); F = ctx.src.decode(ctx.src.body, e)
        med = torch.tensor(F[::3], device=device).float().median(0).values.cpu().numpy()
        sp = ctx.robot.split(S); qd = np.abs(np.gradient(np.concatenate([sp[s] for s in ctx.robot.arms], 1), axis=0)).max(1) * ctx.src.fps
        cand = np.flatnonzero(qd < 0.05); cand = cand[(cand > 5) & (cand < len(S) - 5)]
        js = cand[np.linspace(0, len(cand) - 1, min(per_ep, len(cand))).astype(int)] if len(cand) else []
        for j in js:
            fg = cv2.morphologyEx((np.abs(F[j].astype(np.int16) - med.astype(np.int16)).max(2) > 30).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
            fr.append({"e": e, "fit": i % 2 == 0, "q": {s: sp[s][j] for s in ctx.robot.arms}, "g": {s: float(sp["g" + s][j]) for s in ctx.robot.arms if "g" + s in sp},
                       "fg": torch.tensor(fg, device=device)})
    return fr


def ious(ctx, G, cam, frames, device):
    k = torch.ones(1, 1, 41, 41, device=device); out = []; arm = torch.tensor([s != "B" for s in G.side] + [False], device=device)
    for f in frames:
        _, lab, _ = mesh.splat(G.world(G.placements(f["q"], f["g"])), G.gid, cam, foot=1, dev=device)
        r = arm[lab.clamp(min=0)] & (lab >= 0)
        roi = torch.nn.functional.conv2d(r.float()[None, None], k, padding=20)[0, 0] > 0; a = f["fg"] & roi
        out.append(float((a & r).sum() / (a | r).sum().clamp(min=1)))
    return np.array(out)


def run(ctx, device="cuda:0", search=False, log=print):
    t0 = time.time(); G = mesh.Geometry(ctx.robot, device=device, spacing=0.003)
    frames = load_frames(ctx, device=device); fit = [f for f in frames if f["fit"]]; val = [f for f in frames if not f["fit"]]
    sc = np.r_[[0.02] * 3, [0.02] * 3, 0.02]; start = np.zeros(7)
    base_fit, base_val = ious(ctx, G, ctx.cam0, fit, device).mean(), ious(ctx, G, ctx.cam0, val, device)
    log(f"initial camera: fit IoU {base_fit:.4f}, held-out IoU {base_val.mean():.4f}")
    if search:
        cs = ctx.cfg.get("calibration", {}); deg, m = cs.get("search_deg", 15), cs.get("search_m", 0.15); rng = np.random.default_rng(0)
        best = (base_fit, np.zeros(7))
        sub = fit[::3]
        for _ in range(cs.get("search_samples", 400)):
            p = np.r_[rng.uniform(-1, 1, 3) * np.radians(deg), rng.uniform(-1, 1, 3) * m, rng.uniform(-0.1, 0.1)]
            s_ = ious(ctx, G, ctx.cam0.perturbed(p), sub, device).mean()
            if s_ > best[0]:
                best = (s_, p)
        start = best[1] / sc; log(f"coarse search best IoU {best[0]:.4f}")
    r = minimize(lambda x: -ious(ctx, G, ctx.cam0.perturbed(x * sc), fit, device).mean(), start, method="Powell",
                 options={"maxfev": ctx.cfg.get("calibration", {}).get("maxfev", 900), "xtol": 1e-2, "ftol": 1e-5})
    p = r.x * sc; v = ious(ctx, G, ctx.cam0.perturbed(p), val, device)
    rep = {"params7": p.tolist(), "fit_iou": {"initial": float(base_fit), "fitted": float(-r.fun)},
           "heldout_iou": {"initial": float(base_val.mean()), "fitted": float(v.mean())}, "heldout_frames_improved": float((v > base_val).mean()),
           "n_fit": len(fit), "n_heldout": len(val), "runtime_s": time.time() - t0,
           "adopted": bool(v.mean() > base_val.mean() + 0.005),
           "rule": "adopted if held-out IoU improves by > 0.005; confirm with `sfagen bench --compare-camera`"}
    (ctx.work / "calibration.json").write_text(json.dumps(rep, indent=1)); log(json.dumps({k: rep[k] for k in ("fit_iou", "heldout_iou", "adopted")}))
    return rep
