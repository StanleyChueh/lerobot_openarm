"""Optional OpenArm adapter: everything here knows about OpenArm joints, grippers, URDF or Isaac files.

1. Demonstration quality: runs the existing analyze_dataset_quality.analyze() (gripper / handover / timing /
   action-spike / tracking / near-duplicate trajectory diagnostics) unchanged. That function was written for
   the old Isaac gripper scale (open ~0.044, closing = decreasing). openarm_umeow datasets store raw motor
   values with binary gripper commands (right: open -1.30 -> closed 0, left: open 1.21 -> closed 0), so the
   gripper channels are first mapped to OPENNESS in [0, 1] (1 = commanded-open value, 0 = commanded-closed
   value) and the thresholds are given in that unit. Default `gripper_signal="hybrid"`: open/close instants come
   from the binary command, hold levels from the measured state (it stalls on the can). "state" segments the
   state alone (its slow re-opening ramp inflates hold_slip); "action" has no hold level at all.
2. Recorded object poses (source A) from an IsaacLab Mimic source HDF5 written by lerobot_sim_server.py,
   matched to dataset episodes by length AND joint trajectory, never by index alone.
3. Grasp-position proxy: forward kinematics (pinocchio, the follower's URDF) of the right TCP at the first
   frame the right gripper is commanded closed. It is the gripper position at grasp time, NOT the object's
   initial position; where object poses exist (sim) the proxy is validated against them.
"""

from __future__ import annotations

import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from .checks import FAIL, PASS, UNKNOWN, WARN, Check
from .loader import DatasetView
from .positions import RECORDED, Position

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = REPO_ROOT / "model" / "openarm_description.urdf"
WORLD_FRAME = "openarm_world"  # URDF `world` == IsaacLab env frame (robot root at the origin), checked in sim

# analyze_dataset_quality.py defaults, re-expressed in openness units for openarm_umeow grippers.
QUALITY_DEFAULTS = dict(
    right_gripper="RJ8.pos", left_gripper="LJ8.pos", right_prefix="RJ", left_prefix="LJ",
    open_tol=0.06,  # state sits ~4% below the commanded-open value at rest
    close_delta=0.12,
    empty_thresh=0.12,  # real holds stall at openness ~0.34-0.40 (sim ~0.45-0.75); below this = closed on nothing
    min_hold=10,
    # Overlap = right release command - left jaws starting to close (state). The sim-tuned 3 frames flagged 6/20
    # episodes of real_20, the set whose policy succeeds ~100% on the robot; 1 flags only a release at or before
    # the left jaws move. Measured 2026-10-24: real_20 0-8 frames (median 3), sim_10 1-9 (median 1).
    min_overlap=1,
    settle=15,
    slip_thresh=0.06,
    static_eps=1e-4, pause_frames=30, spike_floor=0.05, track_mult=3.0, dup_thresh=0.01, z_thresh=3.0,
)


def _import_quality_module():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import analyze_dataset_quality as adq  # the existing OpenArm audit, reused as is

    return adq


def gripper_endpoints(view: DatasetView, name: str) -> tuple[float, float]:
    """(open, closed) commanded values: open = the first-frame command, closed = the farthest command from it."""
    j = view.names("action").index(name)
    acts = view.vectors("action")
    first = np.median([a[0, j] for a in acts.values() if len(a)])
    allv = np.concatenate([a[:, j] for a in acts.values() if len(a)])
    closed = allv[np.argmax(np.abs(allv - first))]
    return float(first), float(closed)


def to_openness(x: np.ndarray, open_v: float, closed_v: float) -> np.ndarray:
    return (np.asarray(x, float) - closed_v) / (open_v - closed_v)


def run_quality(view: DatasetView, *, gripper_signal: str = "hybrid", overrides: dict | None = None
                ) -> tuple[pd.DataFrame, dict, list[Check], dict]:
    adq = _import_quality_module()
    opts = {**QUALITY_DEFAULTS, **(overrides or {})}
    names = view.names("action")
    if view.names("observation.state") != names:
        return pd.DataFrame(), {}, [Check("openarm_quality", UNKNOWN,
                                          "state and action names differ; adapter needs a shared joint order")], {}
    ends = {g: gripper_endpoints(view, opts[g]) for g in ("right_gripper", "left_gripper")}
    jr, jl = names.index(opts["right_gripper"]), names.index(opts["left_gripper"])
    df = view.data[["episode_index", "frame_index", "action", "observation.state"]].copy()
    act = np.stack(df["action"].values).astype(np.float64)
    st = np.stack(df["observation.state"].values).astype(np.float64)
    act_m = act.copy()
    st_m = st.copy()
    for g, j in (("right_gripper", jr), ("left_gripper", jl)):
        o, c = ends[g]
        cmd, meas = to_openness(act[:, j], o, c), to_openness(st[:, j], o, c)
        if gripper_signal == "hybrid":
            # Segment on the binary command (exact open/close instants) but read hold levels from the measured state.
            # Segmenting the state itself counts the slow re-opening ramp after a release as "still holding" and
            # flags every real episode as hold_slip.
            sig = np.where(cmd < 0.5, meas, 1.0)
        else:
            sig = meas if gripper_signal == "state" else cmd
        act_m[:, j] = sig  # analyze() segments the action column
        st_m[:, j] = meas
    df["action"] = list(act_m)
    df["observation.state"] = list(st_m)
    args = Namespace(**opts)
    try:
        ep, _traces, ctx = adq.analyze(df, view.info, args)
        summ = adq.summarize(ep, ctx, view.info)
    except ValueError as exc:  # e.g. no episode has a real hold under this signal
        return pd.DataFrame(), {}, [Check("openarm_quality", UNKNOWN, f"analyze_dataset_quality failed with "
                                          f"gripper_signal={gripper_signal!r}: {exc}")], {}
    excluded = sorted(int(e) for e in ep.index[ep["exclude"]])
    n = len(ep)
    checks = [Check("openarm_gripper_mapping", PASS,
                    f"gripper openness from {gripper_signal}: right open/closed {ends['right_gripper']}, "
                    f"left {ends['left_gripper']}", {"endpoints": ends, "signal": gripper_signal,
                                                     "thresholds_openness": opts})]
    frac = len(excluded) / max(n, 1)
    checks.append(Check("openarm_exclude_flags", PASS if not excluded else (WARN if frac <= 0.2 else FAIL),
                        f"{len(excluded)}/{n} episodes carry a severe flag {sorted(adq.EXCLUDE_FLAGS)}"
                        + (f": {excluded}" if excluded else ""),
                        {"excluded": excluded, "reasons": {int(e): ep.loc[e, "reasons"] for e in excluded}}))
    minor = {f: int(ep[f].sum()) for f in adq.FLAGS if f not in adq.EXCLUDE_FLAGS and ep[f].sum()}
    checks.append(Check("openarm_minor_flags", WARN if minor else PASS,
                        f"non-severe flags: {minor}" if minor else "no non-severe flags", minor))
    dup = int(ep["near_duplicate"].sum())
    checks.append(Check("openarm_trajectory_duplicates", WARN if dup else PASS,
                        f"{dup} episodes have a near-duplicate arm trajectory (nn_dist < {opts['dup_thresh']})",
                        {"nn_dist_median": float(ep["nn_dist"].median())}))
    hold = {"right_hold_openness_median": float(ep["right_hold_value"].median()),
            "left_hold_openness_median": float(ep["left_hold_value"].median()),
            "handover_overlap_frames_median": float(ep["handover_overlap"].median()),
            "handover_overlap_frames_min": float(ep["handover_overlap"].min())}
    checks.append(Check("openarm_handover_overlap", PASS,
                        f"both-hold overlap {hold['handover_overlap_frames_min']:.0f}-"
                        f"{ep['handover_overlap'].max():.0f} frames (median {hold['handover_overlap_frames_median']:.0f}); "
                        f"drop_risk below {opts['min_overlap']}", {"per_episode": ep["handover_overlap"].to_dict()}))
    return ep, summ, checks, {"endpoints": ends, **hold}


# ----------------------------------------------------------------------------- recorded poses (Isaac Mimic HDF5)


def _sim_to_interleaved(jp: np.ndarray) -> np.ndarray:
    """Isaac articulation order L1,R1,L2,R2,... (first 14) -> lerobot RJ1,LJ1,RJ2,LJ2,... (measured: pairwise swap)."""
    out = jp[:, :14].copy()
    out[:, 0::2], out[:, 1::2] = jp[:, 1:14:2], jp[:, 0:14:2]
    return out


def isaac_hdf5_poses(view: DatasetView, h5_path: str | Path, *, object_name: str = "can",
                     tol: float = 1e-4, max_shift: int = 3) -> tuple[dict[int, Position], list[Check], dict]:
    """Initial object pose per dataset episode from an IsaacLab Mimic source HDF5, after proving which demo is
    which episode (equal length and arm-joint trajectories equal within `tol` up to a few frames of shift)."""
    import h5py

    states = view.vectors("observation.state")
    out: dict[int, Position] = {}
    match: dict[int, dict] = {}
    with h5py.File(h5_path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1]))
        for name in demos:
            g = f["data"][name]
            jp = _sim_to_interleaved(g["states/articulation/robot/joint_position"][:])
            best = None
            for e, s in states.items():
                if len(s) != len(jp):
                    continue
                for sh in range(-max_shift, max_shift + 1):
                    a = jp[max(0, -sh): len(jp) - max(0, sh)]
                    b = s[max(0, sh): len(s) - max(0, -sh), :14]
                    err = float(np.abs(a - b).max())
                    if best is None or err < best[0]:
                        best = (err, e, sh)
            if best is None or best[0] > tol:
                match[name] = {"matched": None, "best_err": None if best is None else best[0]}
                continue
            err, e, sh = best
            pose = g[f"initial_state/rigid_object/{object_name}/root_pose"][0]
            root = g["initial_state/articulation/robot/root_pose"][0]
            traj = g[f"states/rigid_object/{object_name}/root_pose"][:, :3]
            out[e] = Position(e, RECORDED, x=float(pose[0] - root[0]), y=float(pose[1] - root[1]),
                              z=float(pose[2] - root[2]), frame_id=WORLD_FRAME, verified=True,
                              notes=f"{Path(h5_path).name}:{name}",
                              extra={"demo": name, "match_err": err, "shift": sh, "object_traj": traj,
                                     "robot_root_xyz": root[:3].tolist()})
            match[name] = {"matched": e, "max_abs_joint_err": err, "frame_shift": sh}
    unmatched = [n for n, m in match.items() if m["matched"] is None]
    eps_without = sorted(set(states) - set(out))
    dup_targets = len(out) != len([m for m in match.values() if m["matched"] is not None])
    status = FAIL if dup_targets else (PASS if not unmatched and not eps_without else WARN)
    checks = [Check("isaac_pose_episode_matching", status,
                    f"{len(out)}/{len(states)} episodes matched to {Path(h5_path).name} by trajectory "
                    f"(max |Δq| <= {tol}); unmatched demos {unmatched}; episodes without pose {eps_without}",
                    {"matches": match})]
    return out, checks, match


# ----------------------------------------------------------------------------- grasp-position proxy (FK)


def _pinocchio():
    # ROS humble on LD_LIBRARY_PATH/PYTHONPATH breaks the pip pinocchio wheel (memory: update-dev-v2-eval-compat).
    import importlib

    return importlib.import_module("pinocchio")


def grasp_proxy(view: DatasetView, *, urdf: str | Path = DEFAULT_URDF, gripper: str = "RJ8.pos",
                tcp_frame: str = "openarm_right_hand_tcp", side: str = "right") -> tuple[dict[int, dict], list[Check]]:
    try:
        pin = _pinocchio()
        model = pin.buildModelFromUrdf(str(urdf))
    except Exception as exc:
        hint = " (run with `env -u LD_LIBRARY_PATH -u PYTHONPATH` if ROS is sourced)" if os.environ.get(
            "LD_LIBRARY_PATH", "").find("/opt/ros") >= 0 else ""
        return {}, [Check("grasp_proxy", UNKNOWN, f"pinocchio/URDF unavailable: {type(exc).__name__}: {exc}{hint}")]
    data = model.createData()
    fid = model.getFrameId(tcp_frame)
    names = view.names("observation.state")
    jg = view.names("action").index(gripper)
    open_v, closed_v = gripper_endpoints(view, gripper)
    q_idx = {}
    for n in range(1, 8):
        for s, prefix in (("right", "R"), ("left", "L")):
            q_idx[names.index(f"{prefix}J{n}.pos")] = model.idx_qs[model.getJointId(f"openarm_{s}_joint{n}")]
    out = {}
    acts, states = view.vectors("action"), view.vectors("observation.state")
    for e, a in acts.items():
        closing = np.where(to_openness(a[:, jg], open_v, closed_v) < 0.5)[0]
        if not len(closing):
            continue
        k = int(closing[0])
        q = pin.neutral(model)
        for col, qi in q_idx.items():
            q[qi] = states[e][k, col]
        pin.framesForwardKinematics(model, data, q)
        p = data.oMf[fid].translation
        out[e] = {"grasp_frame": k, "x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "frame_id": WORLD_FRAME,
                  "side": side}
    checks = [Check("grasp_proxy", PASS if len(out) == len(acts) else WARN,
                    f"{side} TCP at first commanded close for {len(out)}/{len(acts)} episodes (FK, {Path(urdf).name})",
                    {"episodes_without_close": sorted(set(acts) - set(out))})]
    return out, checks


def validate_grasp_proxy(proxy: dict[int, dict], recorded: dict[int, Position], max_err_m: float = 0.03) -> Check:
    """Proxy vs the recorded object position at the grasp frame and at the start (sim only)."""
    at_grasp, at_start = [], []
    for e, g in proxy.items():
        p = recorded.get(e)
        if p is None or "object_traj" not in p.extra:
            continue
        traj = p.extra["object_traj"]
        k = min(g["grasp_frame"], len(traj) - 1)
        root = np.asarray(p.extra.get("robot_root_xyz", [0, 0, 0]))[:2]
        at_grasp.append(np.linalg.norm(np.array([g["x"], g["y"]]) - (traj[k, :2] - root)))
        at_start.append(np.linalg.norm(np.array([g["x"], g["y"]]) - np.array([p.x, p.y])))
    if not at_grasp:
        return Check("grasp_proxy_validation", UNKNOWN, "no recorded object poses to validate the FK proxy against")
    ag, as_ = np.asarray(at_grasp), np.asarray(at_start)
    ev = {"n": len(ag), "xy_err_vs_object_at_grasp_median_m": round(float(np.median(ag)), 4),
          "xy_err_vs_object_at_grasp_max_m": round(float(ag.max()), 4),
          "xy_err_vs_object_start_median_m": round(float(np.median(as_)), 4),
          "xy_err_vs_object_start_max_m": round(float(as_.max()), 4)}
    status = PASS if ag.max() <= max_err_m else WARN
    return Check("grasp_proxy_validation", status,
                 f"FK grasp TCP vs recorded object xy at the grasp frame: median {ev['xy_err_vs_object_at_grasp_median_m']}"
                 f" m, max {ev['xy_err_vs_object_at_grasp_max_m']} m; vs object START: median "
                 f"{ev['xy_err_vs_object_start_median_m']} m (object is pushed before grasp, so the proxy describes "
                 "the grasp position, not the start)", ev)
