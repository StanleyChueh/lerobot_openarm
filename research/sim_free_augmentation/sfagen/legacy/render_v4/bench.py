"""Pose-transfer benchmark with real ground truth: re-render real frame j+k's robot/can pose from real frame j only
(MGW may borrow texture from other frames but never within +-5 of the target), compare to the real frame j+k inside
the union of both frames' SAM2 robot+can masks (dilated 5 px). Renderers: copy (no motion), v3 (homography layers,
inherited camera; = the published dataset's renderer), MGW (mesh-guided, calibrated)."""
import json, sys, time
from pathlib import Path
import numpy as np, cv2, torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import mgw, proxy
from sfa3 import recon3, render3
from sfa.robot import Robot, Camera


def ssim_masked(a, b, m):
    a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64); b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    mu_a, mu_b = cv2.GaussianBlur(a, (7, 7), 1.5), cv2.GaussianBlur(b, (7, 7), 1.5)
    s_a = cv2.GaussianBlur(a * a, (7, 7), 1.5) - mu_a ** 2; s_b = cv2.GaussianBlur(b * b, (7, 7), 1.5) - mu_b ** 2
    s_ab = cv2.GaussianBlur(a * b, (7, 7), 1.5) - mu_a * mu_b
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    S = ((2 * mu_a * mu_b + C1) * (2 * s_ab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (s_a + s_b + C2))
    return float(S[m].mean())


def phase(t, ev):
    if t < ev["right_close"] - 10: return "approach"
    if t < ev["right_close"] + 10: return "grasp"
    if t < ev["left_close"] - 15: return "carry"
    if t < ev["right_open"] + 10: return "handover"
    return "release"


MODE = "fwd"


def main(eps, ks=(6, 15), step=16, dev="cuda:0", save_examples=True):
    rows = []
    robot, cam0 = Robot(), Camera()
    for E in eps:
        rec = recon3.build(E, with_plate=True, log=lambda *a: None)
        R = (mgw.MGWForward if MODE == "fwd" else mgw.MGW)(rec, device=dev)
        ex = []
        for k in ks:
            for j in range(4, rec.T - k, step):
                t = j + k
                q_new = {s: rec.gv[s][t] for s in "RL"}; g_new = {s: float(rec.grip_v[s][t]) for s in "RL"}
                can_new = rec.source_can(robot, t + rec.lag); can_old = rec.source_can(robot, j + rec.lag)
                q_old = {s: rec.gv[s][j] for s in "RL"}; g_old = {s: float(rec.grip_v[s][j]) for s in "RL"}
                v3img, _, _ = render3.render_frame(rec.F[j], rec.plate, robot, cam0, q_old, q_new, g_old, g_new,
                                                   {kk: rec.masks[kk][j] for kk in rec.masks}, can_old, can_new)
                t0 = time.time()
                mimg, info = R.render(j, q_new, g_new, can_new, exclude=(t - 5, t + 5))
                dt = time.time() - t0
                gt = rec.F[t]
                d_t, lab_t, _ = R.zbuf(R.placements(q_new, g_new, can_new)); lab_t = lab_t.cpu().numpy()
                tmask = (lab_t >= 0) & (R.part_side[np.clip(lab_t, 0, None)] != "B")
                movedm = np.zeros_like(tmask)
                for kk in ("R", "L", "can"):
                    movedm |= rec.masks[kk][j] > 0
                m = np.zeros(gt.shape[:2], np.uint8)
                for kk in ("R", "L", "can"):
                    m |= rec.masks[kk][t] > 0; m |= rec.masks[kk][j] > 0
                m = cv2.dilate(m, np.ones((11, 11), np.uint8)) > 0
                r = {"E": E, "j": j, "k": k, "phase": phase(t, rec.ev), "mask_px": int(m.sum()), "mgw_s": dt, **{f"mgw_{a}": b for a, b in info.items()}}
                for name, img in (("copy", rec.F[j]), ("v3", v3img), ("mgw", mimg)):
                    mse = float(((img.astype(np.float64) - gt) ** 2)[m].mean())
                    r[f"{name}_psnr"] = 10 * np.log10(255 ** 2 / max(mse, 1e-6)); r[f"{name}_ssim"] = ssim_masked(img, gt, m)
                    r[f"{name}_proxy"] = proxy.score(img, rec.plate, tmask, movedm)
                rows.append(r)
                if save_examples and k == max(ks) and r["phase"] in ("handover", "grasp", "carry") and len(ex) < 6:
                    ex.append(np.concatenate([rec.F[j], v3img, mimg, gt], 1))
        print(E, "done", len(rows), flush=True)
        if ex:
            cv2.imwrite(str(HERE / f"bench_examples_ep{E}.jpg"), cv2.cvtColor(np.concatenate(ex, 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        del R, rec; torch.cuda.empty_cache()
    return rows


if __name__ == "__main__":
    eps = [int(x) for x in sys.argv[1].split(",")]; tag = sys.argv[2] if len(sys.argv) > 2 else "bench"
    rows = main(eps, ks=(4, 8, 15), step=24, dev=sys.argv[3] if len(sys.argv) > 3 else "cuda:0")
    (HERE / f"{tag}.json").write_text(json.dumps(rows, indent=1, default=float))
    import pandas as pd
    df = pd.DataFrame(rows)
    print(df.groupby("phase")[["copy_psnr", "v3_psnr", "mgw_psnr", "copy_ssim", "v3_ssim", "mgw_ssim"]].mean().round(3))
    print(df[["copy_psnr", "v3_psnr", "mgw_psnr", "copy_ssim", "v3_ssim", "mgw_ssim", "mgw_s"]].mean().round(3))
