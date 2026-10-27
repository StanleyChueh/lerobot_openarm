"""Extract the small per-step arrays of the Mimic source / generated HDF5 files (no images) into one pickle.

Run with Isaac Lab's python (h5py), read-only:
    ~/miniforge3/envs/env_isaaclab/bin/python extract_hdf5.py <out.pkl>

Per episode: success attr, initial can pose, per-step can pose (post-step), both EEF positions (pre-step obs),
articulation joint positions (post-step states), lerobot/joint_pos (pre-step) and lerobot/joint_pos_target
(post-step) by joint name, the 14D/16D `actions`, and the annotated subtask signals where present.
"""

import pickle
import sys
from pathlib import Path

import h5py
import numpy as np

DEMOS = Path.home() / "Stanley_ws/IsaacLab/logs/demos"
FILES = {
    "source": DEMOS / "openarm_pringles_lerobot_sim_10_v00.hdf5",
    "annotated": DEMOS / "openarm_pringles_lerobot_sim_10_v00_annotated.hdf5",
    "generated": DEMOS / "openarm_pringles_lerobot_sim_10_v00_generated.hdf5",
}


def _get(e, key):
    return e[key][()] if key in e else None


def extract(path: Path) -> dict:
    out = {"path": str(path), "size": path.stat().st_size, "mtime": path.stat().st_mtime, "episodes": []}
    with h5py.File(path, "r") as f:
        d = f["data"]
        out["env_args"] = d.attrs["env_args"]
        for name in sorted(d, key=lambda n: int(n.split("_")[1])):
            e = d[name]
            ep = {"name": name, "attrs": {k: (v.item() if hasattr(v, "item") else v) for k, v in e.attrs.items()}}
            for key in ["actions", "processed_actions", "obs/eef_pos", "obs/eef_quat", "obs/right_eef_pos",
                        "obs/right_eef_quat", "states/articulation/robot/joint_position",
                        "states/articulation/robot/joint_velocity", "states/rigid_object/can/root_pose",
                        "states/rigid_object/can/root_velocity", "initial_state/rigid_object/can/root_pose",
                        "initial_state/articulation/robot/joint_position"]:
                ep[key] = _get(e, key)
            for grp in ["lerobot/joint_pos", "lerobot/joint_pos_target"]:
                if grp in e:
                    ep[grp] = {k: e[grp][k][()] for k in e[grp]}
            sig = "obs/datagen_info/subtask_term_signals"
            if sig in e:
                ep["signals"] = {k: e[sig][k][()].reshape(-1) for k in e[sig]}
            out["episodes"].append(ep)
    return out


if __name__ == "__main__":
    res = {k: extract(p) for k, p in FILES.items()}
    with open(sys.argv[1], "wb") as fh:
        pickle.dump(res, fh)
    for k, v in res.items():
        print(k, len(v["episodes"]), "episodes from", v["path"])
