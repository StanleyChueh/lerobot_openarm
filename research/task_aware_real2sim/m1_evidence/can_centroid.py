"""M1 audit: same-timestep real vs sim-replay red-can centroid/area per camera (classical HSV threshold, CPU)."""
import glob, json, sys
import av, cv2, numpy as np, pandas as pd

ROOT = "/home/csl/.cache/huggingface/lerobot/ethanCSL/"
REAL, SIM = ROOT + "openarm_pringles_lerobot_real_v00", ROOT + "openarm_pringles_lerobot_sim_v01_reallook_physfit"
SIM2REAL = {0: 0, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 9, 7: 10}
CAMS = ["body_cam", "wrist_cam", "right_wrist_cam"]
STRIDE = 3


def eps(d):
    return pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(d + "/meta/episodes/*/*.parquet"))).set_index("episode_index")


def episode_frames(d, E, cam, e):
    key = f"videos/observation.images.{cam}/"
    path = f"{d}/videos/observation.images.{cam}/chunk-{E.loc[e, key + 'chunk_index']:03d}/file-{E.loc[e, key + 'file_index']:03d}.mp4"
    t0, t1 = E.loc[e, key + "from_timestamp"], E.loc[e, key + "to_timestamp"]
    out = []
    with av.open(path) as c:
        s = c.streams.video[0]
        c.seek(int(max(t0 - 1.0, 0) / s.time_base), stream=s)
        for f in c.decode(s):
            if f.time < t0 - 1e-3:
                continue
            if f.time >= t1 - 1e-3:
                break
            k = len(out)
            out.append(cv2.resize(f.to_ndarray(format="rgb24"), (160, 120)) if True else None)
    return out


def red_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = ((hsv[..., 0] < 10) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 120) & (hsv[..., 2] > 60)
    return m


RE, SE = eps(REAL), eps(SIM)
res = {}
for cam in CAMS:
    dx, dy, ar, iou, n_both, n_real_only, n_sim_only = [], [], [], [], 0, 0, 0
    for se, re_ in SIM2REAL.items():
        R, S = episode_frames(REAL, RE, cam, re_), episode_frames(SIM, SE, cam, se)
        for t in range(0, min(len(R), len(S)), STRIDE):
            mr, ms = red_mask(R[t]), red_mask(S[t])
            fr, fs = mr.mean(), ms.mean()
            vr, vs = fr > 0.01, fs > 0.01
            if vr and vs:
                n_both += 1
                yr, xr = np.nonzero(mr); ys, xs = np.nonzero(ms)
                dx.append((xs.mean() - xr.mean()) * 4); dy.append((ys.mean() - yr.mean()) * 4)  # full-res px
                ar.append(fs / fr); iou.append((mr & ms).sum() / (mr | ms).sum())
            elif vr:
                n_real_only += 1
            elif vs:
                n_sim_only += 1
    q = lambda a: [round(float(np.percentile(a, p)), 2) for p in (25, 50, 75)] if a else None
    res[cam] = {"frames_both_visible": n_both, "real_only": n_real_only, "sim_only": n_sim_only,
                "dx_px_q25_50_75": q(dx), "dy_px_q25_50_75": q(dy), "area_sim_over_real_q": q(ar), "mask_iou_q": q(iou)}
    print(cam, json.dumps(res[cam]), flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
