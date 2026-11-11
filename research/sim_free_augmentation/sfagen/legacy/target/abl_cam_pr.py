"""Pringles: inherited vs fitted body camera (calib_pringles.json), MGW renderer, real-ground-truth pose transfer on the
7 held-out episodes; region = union of both frames' SAM2-based robot/can masks (camera-independent)."""
import json, sys
from pathlib import Path
import numpy as np, cv2, torch
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "render_v4")]
from bench import ssim_masked, phase  # noqa: E402
import mgw  # noqa: E402
from sfa3 import recon3  # noqa: E402
eps = [int(x) for x in sys.argv[1].split(",")]; dev = sys.argv[2]; tag = sys.argv[3]
cf = HERE / "cache_pr"; cf.mkdir(exist_ok=True)
(cf / "mgw_calib.json").write_text(json.dumps({"params": json.loads((HERE / "calib_pringles.json").read_text())["params7"] + [0.0] * 14}))
rows = []
for E in eps:
    rec = recon3.build(E, with_plate=True, log=lambda *a: None)
    Rs = {"inherited": mgw.MGWForward(rec, device=dev, calib=ROOT / "render_v4/calib.json"), "fitted": mgw.MGWForward(rec, device=dev, calib=cf / "mgw_calib.json")}
    for k in (4, 8, 15):
        for j in range(4, rec.T - k, 24):
            t = j + k; q = {s: rec.gv[s][t] for s in "RL"}; g = {s: float(rec.grip_v[s][t]) for s in "RL"}; can = rec.source_can(Rs["fitted"].robot, t + rec.lag)
            m = np.zeros((480, 640), np.uint8)
            for kk in ("R", "L", "can"):
                m |= rec.masks[kk][t] > 0; m |= rec.masks[kk][j] > 0
            m = cv2.dilate(m, np.ones((11, 11), np.uint8)) > 0; gt = rec.F[t]
            r = {"E": E, "j": j, "k": k, "phase": phase(t, rec.ev)}
            for name, R in Rs.items():
                img, _ = R.render(j, q, g, can, exclude=(t - 5, t + 5))
                mse = float(((img.astype(np.float64) - gt) ** 2)[m].mean())
                r[f"{name}_psnr"] = 10 * np.log10(255 ** 2 / max(mse, 1e-6)); r[f"{name}_ssim"] = ssim_masked(img, gt, m)
            rows.append(r)
    print(E, "done", len(rows), flush=True)
    del Rs, rec; torch.cuda.empty_cache()
(HERE / f"{tag}.json").write_text(json.dumps(rows, indent=1, default=float))
import pandas as pd
df = pd.DataFrame(rows); print(df[[c for c in df.columns if c.endswith(("psnr", "ssim"))]].mean().round(3).to_dict())
print(df.groupby("phase")[["inherited_psnr", "fitted_psnr"]].mean().round(3))
d = df.fitted_psnr - df.inherited_psnr; rng = np.random.default_rng(0); E_ = df.E.unique()
bs = [np.mean(np.concatenate([d[df.E == e].to_numpy() for e in rng.choice(E_, len(E_))])) for _ in range(2000)]
print(f"fitted - inherited: {d.mean():.3f} dB CI [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}] better {(d > 0).mean():.2f} dSSIM {(df.fitted_ssim - df.inherited_ssim).mean():.4f}")
