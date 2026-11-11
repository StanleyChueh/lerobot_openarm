"""Cross-pipeline real-ground-truth benchmark: the renderer of the PUBLISHED dataset vs the unified sfagen renderer, on
identical frame pairs (re-render real frame j+k from real frame j only; no texture borrowed within +-5 of the target) and
an identical scoring region that depends on neither pipeline (foreground of both frames vs the temporal median of all
frames, dilated 11 px). Held-out episodes only. Episode-bootstrap 95% CI of the paired PSNR difference.
usage: xbench.py pringles|battery EPS DEVICE OUT.json
       xbench.py pringles|battery --imports-only     (CPU: load both pipelines and print which files they come from)
All code comes from this repository: sfagen from ../sfagen, the published baselines from ../legacy and the frozen v2/v3."""
import json, sys
from pathlib import Path
import numpy as np, cv2, torch
from _provenance import HERE, SFA, LEGACY, baseline_modules
task = sys.argv[1]
sys.path[:0] = [str(HERE)]
from sfagen.context import Context
from sfagen.recon import Recon
from sfagen.render import Renderer
from sfagen.bench import ssim_masked
cfg = {"pringles": HERE / "configs/openarm_pringles_handover.yaml", "battery": HERE / "configs/openarm_battery_insertion.yaml"}[task]
if task == "pringles":
    sys.path[:0] = [str(LEGACY / "render_v4"), str(SFA / "v3"), str(SFA / "v2"), str(SFA)]
    import mgw
    from sfa3 import recon3
    def legacy(e):
        rec = recon3.build(e, with_plate=True, log=lambda *a: None); R = mgw.MGWForward(rec, device=dev)
        def render(j, t):
            q = {s: rec.gv[s][t] for s in "RL"}; g = {s: float(rec.grip_v[s][t]) for s in "RL"}
            return R.render(j, q, g, rec.source_can(R.robot, t + rec.lag), exclude=(t - 5, t + 5))[0]
        return render
else:
    sys.path[:0] = [str(LEGACY / "target"), str(LEGACY / "battery_bsfa"), str(LEGACY / "render_v4")]
    import srecon, srender
    srender.mgw.MGWForward.PLATE_ADAPT = 10
    def legacy(e):
        rec = srecon.BRecon2(e, device=dev); R = srender.renderer(rec, dev, "battery")
        def render(j, t):
            q = {s: rec.gv[s][t] for s in "RL"}; g = {s: float(rec.grip_v[s][t]) for s in "RL"}
            return R.render(j, q, g, rec.source_can(rec.robot, t + rec.lag), exclude=(t - 5, t + 5))[0]
        return render
code = baseline_modules()  # raises if any imported research module is untracked
if sys.argv[2] == "--imports-only":
    print(json.dumps(code, indent=1)); sys.exit()
eps, dev, out = [int(x) for x in sys.argv[2].split(",")], sys.argv[3], sys.argv[4]
ctx = Context(cfg)
rows = []
for e in eps:
    rec = Recon(ctx, e, dev); U = Renderer(rec, dev, ctx.cfg.get("render", {})); L = legacy(e); F = rec.F
    med = torch.tensor(F[::3], device=dev).float().median(0).values.cpu().numpy()
    fgm = lambda j: cv2.morphologyEx((np.abs(F[j].astype(np.int16) - med.astype(np.int16)).max(2) > 30).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0  # noqa: E731
    for k in (4, 8, 15):
        for j in range(10, rec.T - k - 10, 24 if task == "pringles" else 30):
            t = j + k; m = cv2.dilate((fgm(j) | fgm(t)).astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
            q = {s: rec.gv[s][t] for s in rec.arms}; g = {s: float(rec.grip_v[s][t]) for s in rec.grip_v}
            imgs = {"copy": F[j], "published": L(j, t), "unified": U.render(j, q, g, [rec.obj_pose(kk, t + rec.lag) for kk in range(len(rec.objects))], exclude=(t - 5, t + 5))[0]}
            r = {"e": e, "j": j, "k": k}
            for n, im in imgs.items():
                r[f"{n}_psnr"] = 10 * np.log10(255 ** 2 / max(float(((im.astype(float) - F[t]) ** 2)[m].mean()), 1e-6)); r[f"{n}_ssim"] = ssim_masked(im, F[t], m)
            rows.append(r)
    print(task, "episode", e, "pairs", len(rows), flush=True)
    del U, L, rec; torch.cuda.empty_cache()
E = np.array([r["e"] for r in rows]); rng = np.random.default_rng(0)
def ci(d):
    bs = [np.mean(np.concatenate([d[E == e] for e in rng.choice(np.unique(E), len(np.unique(E)))])) for _ in range(2000)]
    return [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float((d > 0).mean())]
summ = {"task": task, "episodes": eps, "pairs": len(rows), **{f"{n}_{m}": float(np.mean([r[f"{n}_{m}"] for r in rows])) for n in ("copy", "published", "unified") for m in ("psnr", "ssim")},
        "unified_minus_published_psnr": ci(np.array([r["unified_psnr"] - r["published_psnr"] for r in rows])),
        "unified_minus_copy_psnr": ci(np.array([r["unified_psnr"] - r["copy_psnr"] for r in rows]))}
Path(out).write_text(json.dumps({"summary": summ, "code": code, "rows": rows}, indent=1)); print(json.dumps(summ, indent=1))
