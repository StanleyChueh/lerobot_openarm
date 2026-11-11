"""Pose-transfer benchmark (real ground truth) on held-out battery episodes: inherited camera vs fitted camera.
Re-render real frame j+k from frame j only (texture borrowing never within +-5 of the target), MGW battery config.
Region: CAMERA-INDEPENDENT foreground of frames j and j+k (|frame - temporal median of all frames| > 30, dilated 11 px),
so neither camera defines where it is scored."""
import json, sys
from pathlib import Path
import numpy as np, cv2, torch
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "battery/bsfa"), str(ROOT / "render_v4")]
import srecon, brecon, brender  # noqa: E402
from bench import ssim_masked  # noqa: E402
from sfa.robot import Camera  # noqa: E402
brender.mgw.MGWForward.PART_FILL = True
OLD = {"Camera": Camera, "CACHE": ROOT / "battery/cache", "calib": ROOT / "render_v4/calib.json"}
NEW = {"Camera": lambda: srecon._cam(), "CACHE": srecon.CACHE, "calib": None}


def build(E, which, dev):
    cfg = OLD if which == "inherited" else NEW
    brecon.Camera, brecon.CACHE = cfg["Camera"], cfg["CACHE"]
    rec = brecon.BRecon(E, device=dev)
    calib = cfg["calib"] if cfg["calib"] is not None else srecon.mgw_calib_file()
    return rec, brender.mgw.MGWForward(rec, device=dev, calib=calib)


def main(eps, dev, ks=(4, 8, 15), step=30):
    rows = []
    for E in eps:
        recs = {w: build(E, w, dev) for w in ("inherited", "fitted")}
        F = recs["fitted"][0].F
        med = torch.tensor(F[::3], device=dev).float().median(0).values.cpu().numpy()
        fgm = lambda j: cv2.morphologyEx((np.abs(F[j].astype(np.int16) - med.astype(np.int16)).max(2) > 30).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0  # noqa: E731
        rec0 = recs["fitted"][0]
        for k in ks:
            for j in range(10, rec0.tr + 40 - k, step):
                t = j + k
                m = cv2.dilate((fgm(j) | fgm(t)).astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
                gt = F[t]; r = {"E": E, "j": j, "k": k}
                imgs = {"copy": F[j]}
                for w, (rec, R) in recs.items():
                    q = {s: rec.gv[s][t] for s in "RL"}; g = {s: float(rec.grip_v[s][t]) for s in "RL"}; can = rec.source_can(rec.robot, t + rec.lag)
                    imgs[w], _ = R.render(j, q, g, can, exclude=(t - 5, t + 5))
                for name, img in imgs.items():
                    mse = float(((img.astype(np.float64) - gt) ** 2)[m].mean())
                    r[f"{name}_psnr"] = 10 * np.log10(255 ** 2 / max(mse, 1e-6)); r[f"{name}_ssim"] = ssim_masked(img, gt, m)
                rows.append(r)
        print(E, "done", len(rows), flush=True)
        del recs; torch.cuda.empty_cache()
    return rows


if __name__ == "__main__":
    import pandas as pd
    eps = [int(x) for x in sys.argv[1].split(",")]; tag = sys.argv[2]; dev = sys.argv[3]
    rows = main(eps, dev)
    (HERE / f"{tag}.json").write_text(json.dumps(rows, indent=1, default=float))
    df = pd.DataFrame(rows); cols = [c for c in df.columns if c.endswith("_psnr") or c.endswith("_ssim")]
    print(df[cols].mean().round(3)); print(df.groupby("k")[cols].mean().round(3))
    d = df["fitted_psnr"] - df["inherited_psnr"]; rng = np.random.default_rng(0)
    eps_ = df.E.unique(); bs = [np.mean(np.concatenate([d[df.E == e].to_numpy() for e in rng.choice(eps_, len(eps_))])) for _ in range(2000)]
    print("fitted - inherited PSNR: %.3f dB, 95%% CI [%.3f, %.3f], pairs better %.2f" % (d.mean(), *np.percentile(bs, [2.5, 97.5]), (d > 0).mean()))
