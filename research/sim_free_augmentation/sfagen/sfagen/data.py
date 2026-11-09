"""Read-only access to a LeRobot v3 source dataset (local root or Hub repo@revision)."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import av, numpy as np, pandas as pd


@dataclass
class Episode:
    e: int
    state: np.ndarray
    action: np.ndarray
    frames: dict


class Source:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        root = cfg.get("root")
        if not root:
            from huggingface_hub import snapshot_download
            root = snapshot_download(cfg["repo_id"], repo_type="dataset", revision=cfg.get("revision"))
        self.root = Path(root)
        self.eps = pd.read_parquet(sorted((self.root / "meta/episodes").rglob("*.parquet")))
        self.df = pd.read_parquet(sorted((self.root / "data").rglob("*.parquet")))
        self.info = __import__("json").loads((self.root / "meta/info.json").read_text())
        self.fps = self.info["fps"]
        self.body = cfg["body_camera"]; self.wrists = list(cfg.get("wrist_cameras", []))
        self.task = pd.read_parquet(self.root / "meta/tasks.parquet").index[0]

    @property
    def episode_ids(self):
        return sorted(int(e) for e in self.eps.episode_index)

    def arrays(self, e):
        d = self.df[self.df.episode_index == e].sort_values("frame_index")
        return np.stack(d["observation.state"].to_numpy()).astype(np.float64), np.stack(d["action"].to_numpy()).astype(np.float64)

    def decode(self, cam, e):
        r = self.eps[self.eps.episode_index == e].iloc[0]
        ci, fi = int(r[f"videos/{cam}/chunk_index"]), int(r[f"videos/{cam}/file_index"])
        t0 = float(r[f"videos/{cam}/from_timestamp"]); n = int(r["length"])
        H, W = self.info["features"][cam]["shape"][:2]
        out = np.zeros((n, H, W, 3), np.uint8); got = np.zeros(n, bool)
        with av.open(str(self.root / f"videos/{cam}/chunk-{ci:03d}/file-{fi:03d}.mp4")) as f:
            s = f.streams.video[0]; f.seek(int(max(0.0, t0 - 1.0) / s.time_base), stream=s)
            for fr in f.decode(s):
                k = int(round((float(fr.pts * s.time_base) - t0) * self.fps))
                if 0 <= k < n:
                    out[k] = fr.to_ndarray(format="rgb24"); got[k] = True
                if k >= n - 1:
                    break
        assert got.all(), (cam, e, int(got.sum()), n)
        return out

    def load(self, e, cams=None):
        S, A = self.arrays(e)
        cams = [self.body] + self.wrists if cams is None else cams
        return Episode(e, S, A, {c: self.decode(c, e) for c in cams})
