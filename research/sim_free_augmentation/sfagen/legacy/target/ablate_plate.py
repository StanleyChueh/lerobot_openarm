"""Ablation of MGW PLATE_ADAPT (per-frame relit clean plate) on real-ground-truth pose transfer.
task=pringles: region = union of both frames' robot/can masks (render_v4/bench.py); task=battery: camera-independent
foreground region (bbench2.py), calibrated battery camera. Usage: ablate_plate.py task eps values tag dev"""
import json, sys
from pathlib import Path
import numpy as np, cv2, torch
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
task, eps, vals, tag, dev = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], [int(x) for x in sys.argv[3].split(",")], sys.argv[4], sys.argv[5]
sys.path[:0] = [str(HERE), str(ROOT / "render_v4")]
from bench import ssim_masked, phase  # noqa: E402
import mgw  # noqa: E402
rows = []
for E in eps:
    if task == "pringles":
        from sfa3 import recon3
        rec = recon3.build(E, with_plate=True, log=lambda *a: None); R = mgw.MGWForward(rec, device=dev)
        ks, step, j0, j1 = (4, 8, 15), 24, 4, rec.T
        def region(j, t):
            m = np.zeros((480, 640), np.uint8)
            for kk in ("R", "L", "can"):
                m |= rec.masks[kk][t] > 0; m |= rec.masks[kk][j] > 0
            return cv2.dilate(m, np.ones((11, 11), np.uint8)) > 0
        robot = R.robot
        state = lambda t: ({s: rec.gv[s][t] for s in "RL"}, {s: float(rec.grip_v[s][t]) for s in "RL"}, rec.source_can(robot, t + rec.lag))  # noqa: E731
    else:
        import srender, srecon
        rec = srecon.BRecon2(E, device=dev); R = srender.renderer(rec, dev, "battery")
        ks, step, j0, j1 = (4, 8, 15), 30, 10, rec.tr + 40
        med = torch.tensor(rec.F[::3], device=dev).float().median(0).values.cpu().numpy()
        fgm = lambda j: cv2.morphologyEx((np.abs(rec.F[j].astype(np.int16) - med.astype(np.int16)).max(2) > 30).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0  # noqa: E731
        region = lambda j, t: cv2.dilate((fgm(j) | fgm(t)).astype(np.uint8), np.ones((11, 11), np.uint8)) > 0  # noqa: E731
        state = lambda t: ({s: rec.gv[s][t] for s in "RL"}, {s: float(rec.grip_v[s][t]) for s in "RL"}, rec.source_can(rec.robot, t + rec.lag))  # noqa: E731
    for k in ks:
        for j in range(j0, j1 - k, step):
            t = j + k; q, g, can = state(t); m = region(j, t); gt = rec.F[t]
            r = {"E": E, "j": j, "k": k}
            if task == "pringles":
                r["phase"] = phase(t, rec.ev)
            for v in vals:
                mgw.MGWForward.PLATE_ADAPT = v
                img, _ = R.render(j, q, g, can, exclude=(t - 5, t + 5))
                mse = float(((img.astype(np.float64) - gt) ** 2)[m].mean())
                r[f"pa{v}_psnr"] = 10 * np.log10(255 ** 2 / max(mse, 1e-6)); r[f"pa{v}_ssim"] = ssim_masked(img, gt, m)
            rows.append(r)
    print(E, "done", len(rows), flush=True)
    del R, rec; torch.cuda.empty_cache()
(HERE / f"{tag}.json").write_text(json.dumps(rows, indent=1, default=float))
import pandas as pd
df = pd.DataFrame(rows); cols = [c for c in df.columns if c.startswith("pa")]
print(df[cols].mean().round(3).to_dict())
base = vals[0]
for v in vals[1:]:
    d = df[f"pa{v}_psnr"] - df[f"pa{base}_psnr"]; rng = np.random.default_rng(0); E_ = df.E.unique()
    bs = [np.mean(np.concatenate([d[df.E == e].to_numpy() for e in rng.choice(E_, len(E_))])) for _ in range(2000)]
    print(f"pa{v} - pa{base}: {d.mean():.3f} dB CI [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}] better {(d > 0).mean():.2f}  dSSIM {(df[f'pa{v}_ssim'] - df[f'pa{base}_ssim']).mean():.4f}")
