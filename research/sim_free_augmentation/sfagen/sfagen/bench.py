"""Real-ground-truth pose-transfer benchmark: re-render real frame j+k from real frame j only (texture borrowing never
within +-5 of the target); PSNR/SSIM inside the camera-independent foreground of both frames. Compares renderer
configurations and/or the initial vs fitted camera; episode-bootstrap 95% CI of the paired difference."""
from __future__ import annotations
import json
import cv2, numpy as np, torch
from .recon import Recon
from .render import Renderer


def ssim_masked(a, b, m):
    a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64); b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    mu_a, mu_b = cv2.GaussianBlur(a, (7, 7), 1.5), cv2.GaussianBlur(b, (7, 7), 1.5)
    s_a = cv2.GaussianBlur(a * a, (7, 7), 1.5) - mu_a ** 2; s_b = cv2.GaussianBlur(b * b, (7, 7), 1.5) - mu_b ** 2; s_ab = cv2.GaussianBlur(a * b, (7, 7), 1.5) - mu_a * mu_b
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float((((2 * mu_a * mu_b + C1) * (2 * s_ab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (s_a + s_b + C2)))[m].mean())


def run(ctx, episodes, variants, device="cuda:0", ks=(4, 8, 15), step=30, log=print):
    """variants: {name: {"render": {...overrides}, "camera": "fitted"|"initial"}}; the first one is the baseline."""
    rows = []
    for e in episodes:
        recs = {}
        for name, v in variants.items():
            cam = v.get("camera", "fitted")
            if cam not in recs:
                keep = ctx.cam; ctx.cam = ctx.cam if cam == "fitted" else ctx.cam0
                old = ctx.cache_dir
                if cam != "fitted":                           # the fitted camera is the pipeline's own: reuse its cache
                    ctx.cache_dir = ctx.work / f"cache_{cam}"; ctx.cache_dir.mkdir(exist_ok=True)
                recs[cam] = Recon(ctx, e, device); ctx.cam, ctx.cache_dir = keep, old
        rends = {name: Renderer(recs[v.get("camera", "fitted")], device, {**ctx.cfg.get("render", {}), **v.get("render", {})}) for name, v in variants.items()}
        r0 = next(iter(recs.values())); F = r0.F
        med = torch.tensor(F[::3], device=device).float().median(0).values.cpu().numpy()
        fgm = lambda j: cv2.morphologyEx((np.abs(F[j].astype(np.int16) - med.astype(np.int16)).max(2) > 30).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0  # noqa: E731
        for k in ks:
            for j in range(10, r0.T - k - 10, step):
                t = j + k; m = cv2.dilate((fgm(j) | fgm(t)).astype(np.uint8), np.ones((11, 11), np.uint8)) > 0; r = {"e": e, "j": j, "k": k}
                ref = F[j]; r["copy_psnr"] = 10 * np.log10(255 ** 2 / max(float(((ref.astype(float) - F[t]) ** 2)[m].mean()), 1e-6)); r["copy_ssim"] = ssim_masked(ref, F[t], m)
                for name, v in variants.items():
                    rec = recs[v.get("camera", "fitted")]
                    q = {s: rec.gv[s][t] for s in rec.arms}; g = {s: float(rec.grip_v[s][t]) for s in rec.grip_v}
                    img, _ = rends[name].render(j, q, g, [rec.obj_pose(kk, t + rec.lag) for kk in range(len(rec.objects))], exclude=(t - 5, t + 5))
                    r[f"{name}_psnr"] = 10 * np.log10(255 ** 2 / max(float(((img.astype(float) - F[t]) ** 2)[m].mean()), 1e-6)); r[f"{name}_ssim"] = ssim_masked(img, F[t], m)
                rows.append(r)
        log("bench episode", e, "pairs", len(rows)); del rends, recs; torch.cuda.empty_cache()
    names = list(variants); summ = {"pairs": len(rows), "episodes": list(episodes), "mean": {}}
    for n in ["copy"] + names:
        summ["mean"][n] = {"psnr": float(np.mean([r[f"{n}_psnr"] for r in rows])), "ssim": float(np.mean([r[f"{n}_ssim"] for r in rows]))}
    rng = np.random.default_rng(0); E = np.array([r["e"] for r in rows])
    for n in names[1:] + ["copy"]:
        d = np.array([r[f"{n}_psnr"] - r[f"{names[0]}_psnr"] for r in rows])
        bs = [np.mean(np.concatenate([d[E == e] for e in rng.choice(np.unique(E), len(np.unique(E)))])) for _ in range(2000)]
        summ[f"{n}_minus_{names[0]}_psnr"] = {"mean": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "better_frac": float((d > 0).mean())}
    return rows, summ
