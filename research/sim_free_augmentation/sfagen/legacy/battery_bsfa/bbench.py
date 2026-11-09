"""Pose-transfer benchmark on the battery recordings (real ground truth): re-render real frame j+k from frame j only
(texture borrowing never within +-5 of the target); metrics inside the robot+battery region of both frames."""
import json, sys, time
from pathlib import Path
import numpy as np, cv2, torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import brender, brecon
sys.path.insert(0, "/home/stanley/Stanley_ws/sfa_hf_export/render_v4")
from bench import ssim_masked
CONFIGS = {"pringles_cfg": dict(TM_ERODE=21, PART_FILL=False), "partfill_tm21": dict(TM_ERODE=21, PART_FILL=True),
           "partfill_tm5": dict(TM_ERODE=5, PART_FILL=True)}


def main(eps, dev, ks=(4, 8, 15), step=30, configs=CONFIGS):
    rows = []
    for E in eps:
        rec = brecon.BRecon(E, device=dev)
        R = brender.renderer(rec, dev)
        for k in ks:
            for j in range(10, rec.tr + 40 - k, step):
                t = j + k
                q = {s: rec.gv[s][t] for s in "RL"}; g = {s: float(rec.grip_v[s][t]) for s in "RL"}; can = rec.source_can(rec.robot, t + rec.lag)
                m = np.zeros((480, 640), bool)
                for kk in ("R", "L", "can"):
                    m |= rec.masks[kk][t] > 0; m |= rec.masks[kk][j] > 0
                m = cv2.dilate(m.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
                gt = rec.F[t]; r = {"E": E, "j": j, "k": k}
                imgs = {"copy": rec.F[j]}
                for name, cfg in configs.items():
                    for a, b in cfg.items():
                        setattr(brender.mgw.MGWForward, a, b)
                    imgs[name], _ = R.render(j, q, g, can, exclude=(t - 5, t + 5))
                for name, img in imgs.items():
                    mse = float(((img.astype(np.float64) - gt) ** 2)[m].mean())
                    r[f"{name}_psnr"] = 10 * np.log10(255 ** 2 / max(mse, 1e-6)); r[f"{name}_ssim"] = ssim_masked(img, gt, m)
                rows.append(r)
        print(E, "done", len(rows), flush=True)
        del R, rec; torch.cuda.empty_cache()
    return rows


if __name__ == "__main__":
    import pandas as pd
    eps = [int(x) for x in sys.argv[1].split(",")]; tag = sys.argv[2]; dev = sys.argv[3]
    rows = main(eps, dev)
    (HERE.parent / f"{tag}.json").write_text(json.dumps(rows, indent=1, default=float))
    df = pd.DataFrame(rows); cols = [c for c in df.columns if c.endswith("_psnr") or c.endswith("_ssim")]
    print(df[cols].mean().round(3)); print(df.groupby("k")[cols].mean().round(3))
