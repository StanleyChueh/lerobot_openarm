"""Shared loaders for the generated-data diagnosis (read-only; local caches verified identical to the Hub SHAs)."""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))
from sim_bridge_common import load_calibration, sim_joints_to_motor_action  # noqa: E402

FIG = ROOT / "figures"
DATA = ROOT / "data"
CACHE = Path.home() / ".cache/huggingface/lerobot"
DATASETS = {
    "manual": CACHE / "hub/datasets--ethanCSL--openarm_pringles_lerobot_sim_10_v00/snapshots/b6d39dae630b8ce0b5822bee7a40bffc68a28819",
    "generated": CACHE / "ethanCSL/openarm_pringles_lerobot_sim_10_v00_generated_500",
    "eval_manual": CACHE / "ethanCSL/eval_openarm_pringles_lerobot_sim_10_smolvla_v00",
    "eval_generated": CACHE / "ethanCSL/eval_openarm_pringles_lerobot_sim_10_generated_500_smolvla_v00",
}
SHAS = {"manual": "b6d39dae", "generated": "860741a2", "eval_manual": "a759d6ad", "eval_generated": "1e482964"}
FPS = 30
# dataset column order (interleaved RJ/LJ, verified from info.json names)
NAMES = [f"{s}J{i}.pos" for i in range(1, 9) for s in "RL"]
ARM = [i for i, n in enumerate(NAMES) if not n.endswith("J8.pos")]
GRIP = {"R": NAMES.index("RJ8.pos"), "L": NAMES.index("LJ8.pos")}
CALIB = load_calibration(str(REPO / "calibration.json"))
OPEN_RAW = {"R": CALIB["right"]["gripper"]["open_raw"], "L": CALIB["left"]["gripper"]["open_raw"]}


def openness(raw, side):
    """Gripper raw motor value -> 0 (closed) .. 1 (open)."""
    return np.asarray(raw) / OPEN_RAW[side]


def load(name: str) -> dict:
    """{'df': frame table, 'state': (N,16), 'action': (N,16), 'episodes': [(start, stop), ...], 'info': info.json}"""
    root = DATASETS[name]
    info = json.loads((root / "meta/info.json").read_text())
    assert info["features"]["observation.state"]["names"] == NAMES, "unexpected state order"
    assert info["features"]["action"]["names"] == NAMES, "unexpected action order"
    df = pd.concat([pd.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet"))], ignore_index=True)
    df = df.sort_values("index").reset_index(drop=True)
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    action = np.stack(df["action"].to_numpy()).astype(np.float64)
    ep = df["episode_index"].to_numpy()
    bounds = np.flatnonzero(np.diff(ep)) + 1
    starts = np.r_[0, bounds]
    stops = np.r_[bounds, len(ep)]
    return {"df": df, "state": state, "action": action, "episodes": list(zip(starts, stops)), "info": info,
            "episode_index": ep[starts]}


def episodes(d: dict, key: str):
    arr = d[key]
    return [arr[a:b] for a, b in d["episodes"]]


def load_hdf5() -> dict:
    with open(DATA / "hdf5_arrays.pkl", "rb") as fh:
        return pickle.load(fh)


def to_motor(named: dict) -> np.ndarray:
    """HDF5 {joint_name: (T,)} -> (T,16) in the dataset's motor units and order (calibration.json)."""
    steps = len(next(iter(named.values())))
    out = np.zeros((steps, 16))
    for t in range(steps):
        m = sim_joints_to_motor_action({k: float(v[t]) for k, v in named.items()}, CALIB)
        out[t] = [m[n] for n in NAMES]
    return out


ISAAC_ORDER = None  # filled by isaac_names()


def isaac_names(h: dict) -> list[str]:
    """Articulation joint order of states/articulation/robot/joint_position (18 cols), from the generated file."""
    e = h["generated"]["episodes"][0]
    pos = e["lerobot/joint_pos"]
    st = e["states/articulation/robot/joint_position"]
    names = []
    for c in range(st.shape[1]):
        # post-step state of step t == pre-step joint_pos of step t+1
        best = min(pos, key=lambda k: np.abs(pos[k][1:] - st[:-1, c]).max())
        names.append(best)
    return names


def savefig(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=130, bbox_inches="tight")
