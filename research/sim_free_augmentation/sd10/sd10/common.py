"""Shared, read-only data access for the SD10 source-diversity curation study (no training, no generation).

Datasets are read from pinned Hugging Face snapshots only. Every file is checked against the LFS SHA-256 that the
Hub records for the pinned revision (scripts/verify_data.py) before features are trusted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]           # research/sim_free_augmentation/sd10
REPO = HERE.parents[2]
OUT = HERE / "outputs"
FIG = HERE / "figures"

POOL_REPO = "ethanCSL/openarm_pringles_lerobot_real_v00"
POOL_REV = "2c1bc860347c44fc0fbae7567469d36ee49baa8d"
TEN_REPO = "ethanCSL/openarm_pringles_lerobot_real_10_v00"
TEN_REV = "cebdee98115ab6827a4ec2655335c536b5dd085b"

POOL_SNAP = Path.home() / f".cache/huggingface/hub/datasets--ethanCSL--openarm_pringles_lerobot_real_v00/snapshots/{POOL_REV}"
TEN_SNAP = Path.home() / f".cache/huggingface/lerobot/hub/datasets--ethanCSL--openarm_pringles_lerobot_real_10_v00/snapshots/{TEN_REV}"
# The pool snapshot in the HF cache holds only the body camera. The two pool wrist MP4s were fetched read-only at the
# same pinned revision for v4A; they are located here and accepted only if their SHA-256 equals the Hub LFS hash.
POOL_EXTRA_DIRS = [REPO / "artifacts/codex_sim_free_v4a_freeze_20261107/inputs/pool"]

CAMS = ("observation.images.body_cam", "observation.images.wrist_cam", "observation.images.right_wrist_cam")
BODY, LEFT_WRIST, RIGHT_WRIST = CAMS
FPS = 30

# v4A locked evaluator records (V4A_LOCKED_REAL_VIEW_PROTOCOL.md). Both wrist streams of these records are globally
# withheld: this study never decodes their wrist pixels (packet/PTS integrity only) and never selects them.
V4A_LOCKED = (10, 17, 24, 31, 38, 45)


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def array_sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def resolve(snapshot: Path, rel: str, extra=()) -> Path | None:
    for base in (snapshot, *extra):
        p = base / rel
        if p.exists():
            return p.resolve()
    return None


class Dataset:
    """Minimal LeRobot v3.0 reader for one pinned snapshot."""

    def __init__(self, snapshot: Path, extra=()):
        self.snap, self.extra = snapshot, tuple(extra)
        self.info = json.loads((snapshot / "meta/info.json").read_text())
        self.ep = pd.read_parquet(snapshot / "meta/episodes/chunk-000/file-000.parquet").set_index("episode_index", drop=False)
        files = sorted((snapshot / "data").rglob("*.parquet"))
        self.df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    @property
    def episodes(self) -> list[int]:
        return [int(e) for e in self.ep.index]

    def rows(self, e: int) -> pd.DataFrame:
        r = self.ep.loc[e]
        return self.df.iloc[int(r["dataset_from_index"]):int(r["dataset_to_index"])]

    def vec(self, e: int, key: str) -> np.ndarray:
        return np.stack(self.rows(e)[key].to_numpy()).astype(np.float32)

    def video(self, cam: str, e: int) -> tuple[Path | None, float, float]:
        r = self.ep.loc[e]
        rel = self.info["video_path"].format(video_key=cam, chunk_index=int(r[f"videos/{cam}/chunk_index"]),
                                             file_index=int(r[f"videos/{cam}/file_index"]))
        return resolve(self.snap, rel, self.extra), float(r[f"videos/{cam}/from_timestamp"]), float(r[f"videos/{cam}/to_timestamp"])

    def decode(self, cam: str, e: int, frames=None) -> np.ndarray:
        """Decode RGB frames of one episode (all, or the requested in-episode indices). Refuses the v4A-withheld
        wrist streams of the locked records."""
        import av
        if cam != BODY and e in V4A_LOCKED and self.snap == POOL_SNAP:
            raise PermissionError(f"v4A whole-stream exclusion: {cam} of pool record {e} is evaluator-only")
        path, t0, _ = self.video(cam, e)
        n = int(self.ep.loc[e]["length"])
        want = set(range(n)) if frames is None else {int(f) for f in frames}
        last = max(want)
        out = {}
        with av.open(str(path)) as c:
            st = c.streams.video[0]
            c.seek(int(max(t0 - 0.5, 0) / st.time_base), stream=st)
            for fr in c.decode(st):
                if fr.time is None or fr.time < t0 - 0.5 / FPS:
                    continue
                k = int(round((fr.time - t0) * FPS))
                if k > last or k >= n:
                    break
                if k in want and k not in out:
                    out[k] = fr.to_ndarray(format="rgb24")
        missing = want - set(out)
        assert not missing, (cam, e, sorted(missing)[:5])
        keys = sorted(want)
        return np.stack([out[k] for k in keys])


def pool() -> Dataset:
    return Dataset(POOL_SNAP, POOL_EXTRA_DIRS)


def ten() -> Dataset:
    return Dataset(TEN_SNAP)


def dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
