"""Read-only loaders for the source demonstrations.

* `load_sim10()`: the 10 manual sim demos (`ethanCSL/openarm_pringles_lerobot_sim_10_v00` @ b6d39dae), the SAME source
  demos the failed generated_500 set was made from. Joint state and joint-target action at 30 Hz, current env (-v1).
  The LeRobot copy carries no object pose, so the initial can xy (simulator truth from the server HDF5, saved by the
  earlier diagnosis) and the right-hand grasp offset at the right close are joined from `research/generated_data_diagnosis`.
* `load_gt_hdf5(name)`: local Mimic-annotated recordings with per-step simulator ground truth (can pose, both EEF
  poses, joints, subtask signals). Older env (-v0, 20 Hz, before the foam pads); used to validate metric code and to
  measure attachment geometry, never as pilot sources.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field

import numpy as np

from . import config as C


@dataclass
class Episode:
    name: str
    fps: int
    q: dict            # {"R": (T,7), "L": (T,7)} measured arm joints, rad (sim joint = motor rad, calibration sign +1 offset 0)
    a: dict            # {"R": (T,7), "L": (T,7)} joint targets
    grip_state: dict   # {"R": (T,), "L": (T,)} openness 0 (closed) .. 1 (open)
    grip_cmd: dict     # same for the command
    can0: np.ndarray   # (3,) initial can position (world = env-local; robot root at identity)
    can: np.ndarray | None = None        # (T,7) per-step can pose x,y,z,qw,qx,qy,qz (GT sources only)
    eef: dict | None = None              # {"R": (T,4,4), "L": (T,4,4)} sim grasp-frame poses (GT sources only)
    signals: dict | None = None          # Mimic subtask signals (GT sources only)
    meta: dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        return len(self.q["R"])


def _split16(x: np.ndarray):
    idx = {s: [C.NAMES.index(f"{s}J{i}.pos") for i in range(1, 8)] for s in ("R", "L")}
    g = {s: C.NAMES.index(f"{s}J8.pos") for s in ("R", "L")}
    return {s: x[:, idx[s]] for s in idx}, {s: x[:, g[s]] / C.GRIP_OPEN_RAW[s] for s in g}


def load_sim10(root=C.SIM10_ROOT) -> list[Episode]:
    import pandas as pd

    info = json.loads((root / "meta/info.json").read_text())
    assert info["features"]["observation.state"]["names"] == C.NAMES
    assert info["features"]["action"]["names"] == C.NAMES
    assert info["fps"] == C.FPS
    df = pd.concat([pd.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet"))], ignore_index=True)
    df = df.sort_values("index").reset_index(drop=True)
    starts = json.loads(C.SIM20_START_POSES.read_text())
    events = {int(r["episode"]): r for r in csv.DictReader(open(C.EPISODE_EVENTS)) if r["set"] == "manual"}
    out = []
    for e, g in df.groupby("episode_index"):
        S = np.stack(g["observation.state"].to_numpy()).astype(float)
        A = np.stack(g["action"].to_numpy()).astype(float)
        q, gs = _split16(S)
        a, gc = _split16(A)
        ev = events[int(e)]
        x0, y0 = starts["xy"][int(e)]
        assert abs(float(ev["x0"]) - x0) < 1e-6, "episode_events and sim20_start_poses disagree"
        assert int(ev["n_frames"]) == len(S), "episode length mismatch with the diagnosis events"
        meta = {"source_dataset": f"{C.SIM10_REPO}@{C.SIM10_REV}", "episode": int(e),
                "grasp_offset_world": [float(ev["grasp_dx"]), float(ev["grasp_dy"]), float(ev["grasp_dz"])],
                "events_csv": {k: ev[k] for k in ("t_R_close", "t_lift", "t_L_close", "t_R_release",
                                                  "sig_grasp_right", "sig_presented", "sig_handover",
                                                  "tilt_max_deg", "can_speed_max", "overlap_frames")}}
        out.append(Episode(f"sim10_ep{int(e)}", C.FPS, q, a, gs, gc, np.array([x0, y0, C.CAN_REST_Z]), meta=meta))
    return out


# 18-D articulation order in the IsaacLab recordings: L1,R1,L2,R2,...,L7,R7, Lf1,Lf2,Rf1,Rf2 (verified: elbows at
# indices 6/7 read 1.57 at rest; fingers 14,15 stay open while the right hand holds the can)
_ART = {"L": [0, 2, 4, 6, 8, 10, 12], "R": [1, 3, 5, 7, 9, 11, 13]}
_ART_F = {"L": [14, 15], "R": [16, 17]}
# 16-D joint action in the recordings: [0:7] left arm, [7] left finger (m), [8:15] right arm, [15] right finger
_ACT = {"L": slice(0, 7), "R": slice(8, 15)}
_ACT_F = {"L": 7, "R": 15}


def load_gt_hdf5(name: str, max_episodes: int | None = None) -> list[Episode]:
    import h5py

    path = C.LOCAL_GT_HDF5[name]
    out = []
    with h5py.File(path, "r") as f:
        d = f["data"]
        sim = json.loads(d.attrs["env_args"])["sim_args"]
        hz = int(round(1.0 / (sim["dt"] * sim["decimation"])))
        names = sorted(d, key=lambda n: int(n.split("_")[1]))[:max_episodes]
        for n in names:
            e = d[n]
            q18 = e["states/articulation/robot/joint_position"][()].astype(float)
            act = e["actions"][()].astype(float)
            q = {s: q18[:, _ART[s]] for s in "RL"}
            a = {s: act[:, _ACT[s]] for s in "RL"}
            gs = {s: q18[:, _ART_F[s]].mean(1) / C.FINGER_OPEN for s in "RL"}
            gc = {s: act[:, _ACT_F[s]] / C.FINGER_OPEN for s in "RL"}
            can = e["states/rigid_object/can/root_pose"][()].astype(float)
            can0 = e["initial_state/rigid_object/can/root_pose"][0, :3].astype(float)
            eef = {"L": e["obs/datagen_info/eef_pose/left_eef"][()].astype(float),
                   "R": e["obs/datagen_info/eef_pose/right_eef"][()].astype(float)}
            sig = {k: e[f"obs/datagen_info/subtask_term_signals/{k}"][()].ravel() > 0.5
                   for k in e["obs/datagen_info/subtask_term_signals"]}
            out.append(Episode(f"{name}_{n}", hz, q, a, gs, gc, can0, can=can, eef=eef, signals=sig,
                               meta={"path": str(path), "success_attr": bool(e.attrs.get("success", False)),
                                     "can_post_step": True, "eef_pre_step": True}))
    return out
