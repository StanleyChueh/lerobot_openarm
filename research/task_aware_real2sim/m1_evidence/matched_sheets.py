"""M1 audit: real vs sim frames at nearest arm joint state, one contact sheet per camera (CPU decode)."""
import glob, json, sys
import av, cv2, numpy as np, pandas as pd

ROOT = "/home/csl/.cache/huggingface/lerobot/ethanCSL/"
REAL, SIM = ROOT + "openarm_pringles_lerobot_real_v00", ROOT + "openarm_pringles_lerobot_sim_v01_reallook_physfit"
CAMS = ["body_cam", "wrist_cam", "right_wrist_cam"]
OUT = sys.argv[1]


def load(d):
    df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(d + "/data/*/*.parquet"))).reset_index(drop=True)
    eps = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(d + "/meta/episodes/*/*.parquet"))).set_index("episode_index")
    return df, eps


def frame(d, eps, cam, e, t):
    key = f"videos/observation.images.{cam}/"
    path = f"{d}/videos/observation.images.{cam}/chunk-{eps.loc[e, key + 'chunk_index']:03d}/file-{eps.loc[e, key + 'file_index']:03d}.mp4"
    ts = eps.loc[e, key + "from_timestamp"] + t
    with av.open(path) as c:
        s = c.streams.video[0]
        c.seek(int(max(ts - 1.0, 0) / s.time_base), stream=s)
        best = None
        for f in c.decode(s):
            if f.time is not None and f.time >= ts - 1e-3:
                best = f
                break
            best = f
        return best.to_ndarray(format="rgb24")


rdf, reps = load(REAL)
sdf, seps = load(SIM)
RS, SS = np.stack(rdf["observation.state"].values), np.stack(sdf["observation.state"].values)
arm = [i for i in range(16) if i not in (14, 15)]
# 6 real query frames: episodes spread over the set, phases spread over each episode (approach, grasp, handover, ...)
picks = []
for k, (e, ph) in enumerate(zip([3, 12, 21, 30, 39, 47], [0.15, 0.3, 0.45, 0.55, 0.65, 0.8])):
    idx = np.where(rdf["episode_index"].values == e)[0]
    picks.append(int(idx[int(ph * len(idx))]))
pairs = []
for i in picks:
    d = np.linalg.norm(SS[:, arm] - RS[i, arm], axis=1)
    j = int(np.argmin(d))
    pairs.append((i, j, float(d[j]), float(np.abs(SS[j, arm] - RS[i, arm]).max())))
meta = []
for cam in CAMS:
    rows = []
    for i, j, dist, dmax in pairs:
        r = frame(REAL, reps, cam, int(rdf.loc[i, "episode_index"]), float(rdf.loc[i, "timestamp"]))
        s = frame(SIM, seps, cam, int(sdf.loc[j, "episode_index"]), float(sdf.loc[j, "timestamp"]))
        blend = cv2.addWeighted(r, 0.5, s, 0.5, 0)
        row = np.concatenate([r, s, blend], axis=1)
        txt = f"real ep{rdf.loc[i,'episode_index']} f{rdf.loc[i,'frame_index']} | sim ep{sdf.loc[j,'episode_index']} f{sdf.loc[j,'frame_index']} | joint L2 {dist:.2f} max {dmax:.2f} rad"
        cv2.putText(row, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        rows.append(cv2.resize(row, (row.shape[1] // 2, row.shape[0] // 2)))
    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(f"{OUT}/sheet_{cam}.jpg", cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
for i, j, dist, dmax in pairs:
    meta.append({"real": [int(rdf.loc[i, "episode_index"]), int(rdf.loc[i, "frame_index"])],
                 "sim": [int(sdf.loc[j, "episode_index"]), int(sdf.loc[j, "frame_index"])], "l2": round(dist, 3), "max": round(dmax, 3)})
json.dump(meta, open(f"{OUT}/pairs.json", "w"), indent=1)
print(json.dumps(meta))
