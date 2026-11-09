"""Battery-insertion source data (ethanCSL/openarm_insertbattery_lerobot_real_10_v02 @ a1c29d77), local snapshot."""
from dataclasses import dataclass
from pathlib import Path
import av, numpy as np, pandas as pd

ROOT = Path("/home/stanley/Stanley_ws/sfa_hf_export/battery/src")
REV = "a1c29d77dabcaf5a0a9ff658ba742a2d941e1570"
CAMS = ("observation.images.body_cam", "observation.images.wrist_cam", "observation.images.right_wrist_cam")
NAMES = [f"{s}J{j}.pos" for j in range(1, 9) for s in "RL"]
_EP = None; _DF = None


def episodes():
    global _EP
    if _EP is None:
        _EP = pd.read_parquet(ROOT / "meta/episodes/chunk-000/file-000.parquet")
    return _EP


def table():
    global _DF
    if _DF is None:
        _DF = pd.read_parquet(sorted((ROOT / "data").rglob("*.parquet")))
    return _DF


@dataclass
class Episode:
    e: int
    state: np.ndarray
    action: np.ndarray
    frames: dict


def decode(cam, e):
    r = episodes(); r = r[r.episode_index == e].iloc[0]
    ci, fi = int(r[f"videos/{cam}/chunk_index"]), int(r[f"videos/{cam}/file_index"])
    t0 = float(r[f"videos/{cam}/from_timestamp"]); n = int(r["length"])
    out = np.zeros((n, 480, 640, 3), np.uint8); got = np.zeros(n, bool)
    with av.open(str(ROOT / f"videos/{cam}/chunk-{ci:03d}/file-{fi:03d}.mp4")) as f:
        s = f.streams.video[0]
        f.seek(int(max(0.0, t0 - 1.0) / s.time_base), stream=s)
        for fr in f.decode(s):
            k = int(round((float(fr.pts * s.time_base) - t0) * 30))
            if 0 <= k < n:
                out[k] = fr.to_ndarray(format="rgb24"); got[k] = True
            if k >= n - 1:
                break
    assert got.all(), (cam, e, int(got.sum()), n)
    return out


def load(e, cams=CAMS):
    df = table(); d = df[df.episode_index == e].sort_values("frame_index")
    S = np.stack(d["observation.state"].to_numpy()).astype(np.float64); A = np.stack(d["action"].to_numpy()).astype(np.float64)
    return Episode(e, S, A, {c: decode(c, e) for c in cams})


def grip_events(A, thr=0.5):
    """gripper command events: openness = raw/open_raw (R open -1.298, L open 1.2121); close when < thr."""
    out = {}
    for s, k, o in (("R", NAMES.index("RJ8.pos"), -1.298), ("L", NAMES.index("LJ8.pos"), 1.2121)):
        g = np.clip(A[:, k] / o, 0, 1.2); closed = g < thr
        ch = np.flatnonzero(np.diff(closed.astype(int)))
        out[s] = {"closes": [int(c + 1) for c in ch if closed[c + 1]], "opens": [int(c + 1) for c in ch if not closed[c + 1]],
                  "min_open": float(g.min()), "max_open": float(g.max())}
    return out
