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
            img = F[j]
            if ctx.cfg.get("reconstruction", {}).get("photometric_align"):   # R07: camera auto exposure
                from .recon import match_exposure
                img = match_exposure(img, med.astype(np.uint8))
            fg = cv2.morphologyEx((np.abs(img.astype(np.int16) - med.astype(np.int16)).max(2) > 30).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
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
    nf = 7 if ctx.cfg.get("calibration", {}).get("fit_focal", True) else 6   # R05: trusted intrinsics keep the focal length
    base_fit, base_val = ious(ctx, G, ctx.cam0, fit, device).mean(), ious(ctx, G, ctx.cam0, val, device)
    log(f"initial camera: fit IoU {base_fit:.4f}, held-out IoU {base_val.mean():.4f}")
    if search:
        cs = ctx.cfg.get("calibration", {}); deg, m = cs.get("search_deg", 15), cs.get("search_m", 0.15); rng = np.random.default_rng(0)
        best = (base_fit, np.zeros(7))
        sub = fit[::3]
        for _ in range(cs.get("search_samples", 400)):
            p = np.r_[rng.uniform(-1, 1, 3) * np.radians(deg), rng.uniform(-1, 1, 3) * m, rng.uniform(-0.1, 0.1) * (nf == 7)]
            s_ = ious(ctx, G, ctx.cam0.perturbed(p), sub, device).mean()
            if s_ > best[0]:
                best = (s_, p)
        start = best[1] / sc; log(f"coarse search best IoU {best[0]:.4f}")
    full = lambda x: np.r_[x, np.zeros(7 - nf)] * sc  # noqa: E731
    r = minimize(lambda x: -ious(ctx, G, ctx.cam0.perturbed(full(x)), fit, device).mean(), start[:nf], method="Powell",
                 options={"maxfev": ctx.cfg.get("calibration", {}).get("maxfev", 900), "xtol": 1e-2, "ftol": 1e-5})
    p = full(r.x); v = ious(ctx, G, ctx.cam0.perturbed(p), val, device)
    rep = {"params7": p.tolist(), "fit_iou": {"initial": float(base_fit), "fitted": float(-r.fun)},
           "heldout_iou": {"initial": float(base_val.mean()), "fitted": float(v.mean())}, "heldout_frames_improved": float((v > base_val).mean()),
           "n_fit": len(fit), "n_heldout": len(val), "runtime_s": time.time() - t0,
           "adopted": bool(v.mean() > base_val.mean() + 0.005),
           "rule": "adopted if held-out IoU improves by > 0.005; confirm with `sfagen bench --compare-camera`"}
    (ctx.work / "calibration.json").write_text(json.dumps(rep, indent=1)); log(json.dumps({k: rep[k] for k in ("fit_iou", "heldout_iou", "adopted")}))
    return rep


# ---------------------------------------------------------------- R06: refinement on the reconstruction's arm masks
def contour_frames(rec, every=4):
    """per frame: URDF points (world), and for the union of the final arm masks the nearest-edge lookup; pixels next to an
    object mask are excluded (occlusion edges are not robot edges)."""
    from scipy.ndimage import distance_transform_edt
    out, k3 = [], np.ones((3, 3), np.uint8); sc = rec.W / 640; ko = np.ones((2 * round(5 * sc) + 1,) * 2, np.uint8)
    for j in range(3, rec.T - 3, every):
        m = np.any([rec.masks[s][j] > 0 for s in rec.arms], 0)
        if m.sum() < 400 * sc * sc:
            continue
        edge = m & ~(cv2.erode(m.astype(np.uint8), k3) > 0)
        ob = np.any([rec.masks[f"obj{k}"][j] > 0 for k in range(len(rec.objects))], 0) if rec.objects else np.zeros_like(m)
        dist, (iy, ix) = distance_transform_edt(~edge, return_indices=True)
        Wp = rec.G.world(rec.G.placements({s: rec.gv[s][j] for s in rec.arms}, {s: float(rec.grip_v[s][j]) for s in rec.grip_v}))
        out.append((Wp, iy, ix, dist, cv2.dilate(ob.astype(np.uint8), ko) > 0))
    return out


def contour_pairs(G, cam, frames, tau, device, per_frame=400):
    """3-D point behind each outer-contour pixel of the projected robot, paired with the nearest mask edge (<= tau px)."""
    arm = torch.tensor([s != "B" for s in G.side] + [False], device=device); sc = cam.W / 640; k3 = np.ones((3, 3), np.uint8); kc = np.ones((5, 5), np.uint8)
    border = np.zeros((cam.H, cam.W), bool); b = max(2, round(4 * sc)); border[:b] = border[-b:] = True; border[:, :b] = border[:, -b:] = True
    X, U, D = [], [], []
    for Wp, iy, ix, dist, near_obj in frames:
        _, lab, pt = mesh.splat(Wp, G.gid, cam, foot=1, dev=device); r = (arm[lab.clamp(min=0)] & (lab >= 0)).cpu().numpy(); pt = pt.cpu().numpy()
        rf = cv2.morphologyEx(r.astype(np.uint8), cv2.MORPH_CLOSE, kc)                          # splat holes are not contours
        n, cc, st, _ = cv2.connectedComponentsWithStats(1 - rf); rf = (rf > 0) | np.isin(cc, [k for k in range(1, n) if st[k, 4] < 400 * sc * sc])
        ys, xs = np.nonzero(rf & ~(cv2.erode(rf.astype(np.uint8), k3) > 0) & ~border & ~near_obj & (pt >= 0))
        d = dist[ys, xs]; D.append(np.minimum(d, 40 * sc)); ok = d <= tau; ys, xs = ys[ok], xs[ok]
        if len(ys) > per_frame:
            s_ = np.linspace(0, len(ys) - 1, per_frame).astype(int); ys, xs = ys[s_], xs[s_]
        X.append(Wp[torch.tensor(pt[ys, xs], device=device)].cpu().numpy()); U.append(np.c_[ix[ys, xs], iy[ys, xs]])
    return np.concatenate(X).astype(np.float64), np.concatenate(U).astype(np.float64), float(np.concatenate(D).mean()) if D else np.inf


def contour_icp(G, cam, frames, device, log=print):
    """camera pose by contour ICP: pair, solve PnP (LM from the current pose, intrinsics fixed), shrink tau 40 -> 3 px."""
    cam = cam.copy(); sc = cam.W / 640
    for tau in (40, 30, 20, 15, 10, 7, 5, 3):
        for _ in range(3):
            X, U, _ = contour_pairs(G, cam, frames, tau * sc, device)
            if len(X) < 50:
                return cam
            rv, _ = cv2.Rodrigues(cam.R.T); tv = (-cam.R.T @ cam.t).reshape(3, 1)
            _, rv, tv = cv2.solvePnP(X, U, cam.K, None, rv.copy(), tv.copy(), useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
            Rc, _ = cv2.Rodrigues(rv); cam.R = Rc.T; cam.t = (-Rc.T @ tv).ravel()
    return cam


def refine(ctx, device="cuda:0", log=print):
    """R06 (`sfagen calibrate CONFIG --refine`, after `calibrate`): rounds of [reconstruct `refine_episodes` (a count of the first
    source episodes, or a list of ids) with the current camera -> contour ICP on the even ones]; adopted if the mean contour distance on the odd
    (held-out) ones drops. Needs the R06 mask flags. Clears cache/ and candidates/ (they depend on the camera)."""
    import pinocchio as pin, shutil
    from .recon import Recon
    cs = ctx.cfg.get("calibration", {}); n = cs.get("refine_episodes", 4); eps = list(n) if isinstance(n, (list, tuple)) else ctx.episodes()[:n]; fit_e, val_e = eps[0::2], eps[1::2] or eps[0::2]
    keep = ctx.cache_dir; hist = []; t0 = time.time(); G = None
    save = lambda: (ctx.work / "calibration.json").write_text(json.dumps({**(ctx.calib or {}), "refine": {  # noqa: E731
        "method": "contour ICP on arm masks (R06)", "episodes_fit": fit_e, "episodes_heldout": val_e, "rounds": hist, "runtime_s": time.time() - t0}}, indent=1))
    for rnd in range(cs.get("refine_rounds", 2)):
        ctx.cache_dir = ctx.work / f"cache_refine{rnd}"; ctx.cache_dir.mkdir(exist_ok=True); fr = {}
        for e in eps:
            rec = Recon(ctx, e, device); G = rec.G; fr[e] = contour_frames(rec); del rec
        f_fit = [x for e in fit_e for x in fr[e]]; f_val = [x for e in val_e for x in fr[e]]
        before = (contour_pairs(G, ctx.cam, f_fit, 0, device)[2], contour_pairs(G, ctx.cam, f_val, 0, device)[2])
        cam = contour_icp(G, ctx.cam, f_fit, device, log)
        after = (contour_pairs(G, cam, f_fit, 0, device)[2], contour_pairs(G, cam, f_val, 0, device)[2])
        p = np.array(ctx.calib["params7"] if ctx.calib else np.zeros(7), float); p[:3] = pin.log3(ctx.cam0.R.T @ cam.R); p[3:6] = cam.t - ctx.cam0.t
        h = {"round": rnd, "contour_px_fit": [before[0], after[0]], "contour_px_heldout": [before[1], after[1]], "adopted": bool(after[1] < before[1]),
             "params7": p.tolist(), "frames_fit": len(f_fit), "frames_heldout": len(f_val)}
        hist.append(h); log(json.dumps(h))
        shutil.rmtree(ctx.cache_dir, ignore_errors=True)
        if not h["adopted"]:
            break
        ctx.calib = {**(ctx.calib or {"adopted": True}), "params7": p.tolist(), "adopted": True}; ctx.cam = ctx.cam0.perturbed(p)
        save()                                           # after every adopted round: a stopped run keeps its progress
    ctx.cache_dir = keep; save()
    for d in ("cache", "candidates"):
        shutil.rmtree(ctx.work / d, ignore_errors=True)
    ctx.cache_dir.mkdir(exist_ok=True)
    return json.loads((ctx.work / "calibration.json").read_text())
