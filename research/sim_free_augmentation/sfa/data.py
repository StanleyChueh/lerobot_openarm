"""Read-only access to one real episode of ethanCSL/openarm_pringles_lerobot_real_10_v00 (local HF cache)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .robot import REPO, Robot

sys.path.insert(0, str(REPO))

DATASET = "ethanCSL/openarm_pringles_lerobot_real_10_v00"
REV = "cebdee98115ab6827a4ec2655335c536b5dd085b"
ROOT = Path.home() / f".cache/huggingface/lerobot/hub/datasets--ethanCSL--openarm_pringles_lerobot_real_10_v00/snapshots/{REV}"
CAMS = ("observation.images.body_cam", "observation.images.wrist_cam", "observation.images.right_wrist_cam")


@dataclass
class Episode:
    index: int
    fps: float
    state: np.ndarray        # (T,16)
    action: np.ndarray       # (T,16)
    frames: dict             # camera key -> (T,H,W,3) uint8 RGB
    events: dict             # command-defined: right_close, left_close, right_open (frames)

    @property
    def T(self):
        return len(self.state)


def decode(view, cam: str, e: int) -> np.ndarray:
    import av
    rel, t0 = view.video_path(cam, e)
    n = int(view.episodes.loc[e]["length"])
    out = []
    with av.open(str(view.root / rel)) as c:
        st = c.streams.video[0]
        c.seek(int(max(t0 - 0.5, 0) / st.time_base), stream=st)
        for fr in c.decode(st):
            if fr.time is None or fr.time < t0 - 0.5 / 30:
                continue
            k = int(round((fr.time - t0) * 30))
            if k >= n:
                break
            if k == len(out):
                out.append(fr.to_ndarray(format="rgb24"))
    assert len(out) == n, (cam, e, len(out), n)
    return np.stack(out)


def first(mask, start=0):
    idx = np.flatnonzero(mask[start:])
    return int(idx[0] + start) if len(idx) else None


def command_events(action: np.ndarray) -> dict:
    g = Robot.split(action)
    cR, cL = g["gR"] < 0.5, g["gL"] < 0.5
    rc = first(cR)
    return {"right_close": rc, "left_close": first(cL, rc or 0), "right_open": first(~cR, rc) if rc is not None else None}


def load(e: int, cams=CAMS) -> Episode:
    from tools.dataset_diagnostics.loader import load_dataset

    view = load_dataset(root=ROOT, label="pringles_real_10_v00", episodes=[e])
    assert view.names("action") == view.names("observation.state")
    S = view.vectors("observation.state")[e].astype(float)
    A = view.vectors("action")[e].astype(float)
    frames = {c: decode(view, c, e) for c in cams}
    return Episode(e, float(view.fps), S, A, frames, command_events(A))


def action_lead(state, action, kmax=6) -> dict:
    """mean |a[t] - s[t+k]| over arm joints for k = 0..kmax; the argmin is the command's lead over the state."""
    g_s, g_a = Robot.split(state), Robot.split(action)
    err = {}
    for k in range(kmax + 1):
        e = [np.abs(g_a[s][: len(state) - k] - g_s[s][k:]).mean() for s in "RL"]
        err[k] = float(np.mean(e))
    return {"k_best": min(err, key=err.get), "mean_abs_err_rad": err}
